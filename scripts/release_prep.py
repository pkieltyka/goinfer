#!/usr/bin/env python3
"""Single-commit release prep and verification for goinfer root and backend modules.

Stages the canonical root module zip from a git revision in a local file:// Go
module proxy so selected nested backend modules (backend/cuda, backend/vulkan,
and optionally backend/metal) can pin and verify the candidate root version
outside go.work BEFORE any release tag is created or pushed.

Because Go's module zip specification excludes nested directories that contain
their own go.mod file, updating only nested backend/*/go.{mod,sum} files leaves
the root module zip bit-for-bit identical. Root and selected submodule tags can
therefore all point to one single release commit while matching proxy.golang.org
and sum.golang.org hashes exactly.

This tool NEVER creates or pushes git tags.

Subcommands:
  stage-proxy    Build a file:// Go module proxy zip + metadata from a git rev.
  prep           Update & verify selected submodule go.mod/go.sum outside go.work.
  verify-commit  Re-verify canonical root hashes, submodule pins, and same-commit tags.
  verify-public  Verify published tags, public proxy checksums, and consumer builds.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_MODULE = "github.com/pkieltyka/goinfer"
AIKIT_MODULE = "github.com/pkieltyka/aikit"

MAX_ZIP_FILE = 500 << 20
MAX_GO_MOD = 16 << 20
MAX_LICENSE = 16 << 20

SEMVER_RE = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?$"
)
PSEUDO_RE = re.compile(r"-(?:0\.)?\d{14}-[0-9a-f]{12}$")
CYCLE_RE = re.compile(
    r"github\.com/pkieltyka/goinfer/(?:backend/(?:cuda|metal|vulkan|webgpu)|cuda|gpu|metal)\b"
)


@dataclass(frozen=True)
class SubmoduleSpec:
    rel_dir: str
    module_path: str
    goos: str
    goarch: str
    cgo_enabled: str
    build_tags: str
    consumer_import: str
    runtime_note: str


SUBMODULES: Dict[str, SubmoduleSpec] = {
    "backend/cuda": SubmoduleSpec(
        rel_dir="backend/cuda",
        module_path=f"{ROOT_MODULE}/backend/cuda",
        goos="linux",
        goarch="",
        cgo_enabled="0",
        build_tags="cuda",
        consumer_import=f"{ROOT_MODULE}/backend/cuda",
        runtime_note="Linux NVIDIA CUDA device gate (§C1-C) runs separately on real hardware",
    ),
    "backend/vulkan": SubmoduleSpec(
        rel_dir="backend/vulkan",
        module_path=f"{ROOT_MODULE}/backend/vulkan",
        goos="linux",
        goarch="",
        cgo_enabled="0",
        build_tags="",
        consumer_import=f"{ROOT_MODULE}/backend/vulkan",
        runtime_note="Discrete AMD Radeon Vulkan device check runs separately on real hardware",
    ),
    "backend/metal": SubmoduleSpec(
        rel_dir="backend/metal",
        module_path=f"{ROOT_MODULE}/backend/metal",
        goos="darwin",
        goarch="arm64",
        cgo_enabled="0",
        build_tags="metal",
        consumer_import=f"{ROOT_MODULE}/backend/metal",
        runtime_note="Cross-compiles with GOOS=darwin GOARCH=arm64 on Linux; macOS device gate (§C1-M) runs on Apple Silicon",
    ),
}

DEFAULT_MODULES: Tuple[str, ...] = ("backend/cuda", "backend/vulkan")


class ReleasePrepError(RuntimeError):
    """Raised when a release prep or verification invariant fails."""


def _make_writable_and_rmtree(path: Path) -> None:
    """Remove a directory tree even when Go marked modcache entries 0444/0555."""
    if not path.exists():
        return

    def _onerror(func, p, _exc_info):
        try:
            os.chmod(p, 0o700)
        except OSError:
            pass
        try:
            parent = os.path.dirname(p)
            if parent:
                os.chmod(parent, 0o700)
        except OSError:
            pass
        func(p)

    for root, dirs, files in os.walk(path):
        try:
            os.chmod(root, 0o700)
        except OSError:
            pass
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o700)
            except OSError:
                pass
        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o600)
            except OSError:
                pass
    shutil.rmtree(path, onerror=_onerror)


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=True,
        text=text,
    )
    if check and proc.returncode != 0:
        out = proc.stdout if text else "<binary>"
        err = proc.stderr if text else proc.stderr.decode("utf-8", errors="replace")
        raise ReleasePrepError(
            f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"cwd: {cwd}\nstdout:\n{out}\nstderr:\n{err}"
        )
    return proc


def git_root(repo: Optional[str] = None) -> Path:
    base = Path(repo).resolve() if repo else Path.cwd()
    out = run_cmd(["git", "rev-parse", "--show-toplevel"], cwd=base).stdout.strip()
    return Path(out)


def git_rev_parse(repo: Path, rev: str) -> str:
    return run_cmd(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"], cwd=repo).stdout.strip()


def git_commit_utc(repo: Path, rev: str) -> str:
    raw = run_cmd(["git", "show", "-s", "--format=%cI", rev], cwd=repo).stdout.strip()
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def escape_proxy_segment(s: str) -> str:
    """Go module proxy case-encoding: uppercase ASCII becomes '!' + lowercase."""
    out = []
    for ch in s:
        if "A" <= ch <= "Z":
            out.append("!" + ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def validate_semver(version: str, module_path: str = ROOT_MODULE) -> None:
    m = SEMVER_RE.match(version)
    if not m:
        raise ReleasePrepError(
            f"invalid canonical Go module version {version!r} (expected e.g. v0.6.2 or v0.6.2-rc.1, no +build metadata)"
        )
    major = int(m.group(1))
    # Root and backend modules have no /vN suffix, so major version must be 0 or 1.
    has_major_suffix = bool(re.search(r"/v[2-9][0-9]*$", module_path))
    if major >= 2 and not has_major_suffix:
        raise ReleasePrepError(
            f"version {version!r} has major {major} >= 2 but module path {module_path!r} has no /v{major} suffix"
        )
    if major < 2 and has_major_suffix:
        raise ReleasePrepError(
            f"version {version!r} has major {major} < 2 incompatible with module path {module_path!r}"
        )


def parse_modules_arg(raw: str) -> List[str]:
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() == "none":
        return []
    seen = []
    for part in cleaned.split(","):
        m = part.strip().strip("/")
        if not m:
            continue
        if m not in SUBMODULES:
            known = ", ".join(sorted(SUBMODULES))
            raise ReleasePrepError(f"unknown backend module {m!r}; known modules: {known} (or 'none')")
        if m not in seen:
            seen.append(m)
    return seen


def parse_gomod_go_version(text: str) -> Tuple[int, int]:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//"):
            continue
        m = re.match(r"^go\s+(\d+)\.(\d+)(?:\.\d+)?(?:\s|$)", s)
        if m:
            return int(m.group(1)), int(m.group(2))
    return (1, 16)


def parse_gomod_info(text: str) -> dict:
    """Parse module path, requires, and replace directives from go.mod text."""
    module_path = ""
    requires: Dict[str, str] = {}
    replaces: List[str] = []
    in_require = False
    in_replace = False

    for raw_line in text.splitlines():
        # Strip inline // comments (go.mod uses // only).
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if line == ")":
            in_require = False
            in_replace = False
            continue
        if in_replace:
            replaces.append(line)
            continue
        if in_require:
            parts = line.split()
            if len(parts) >= 2:
                requires[parts[0]] = parts[1]
            continue

        if line.startswith("module "):
            module_path = line.split()[1].strip('"')
        elif line.startswith("replace "):
            if line.endswith("("):
                in_replace = True
            else:
                replaces.append(line[len("replace "):].strip())
        elif line == "replace(":
            in_replace = True
        elif line.startswith("require "):
            if line.endswith("("):
                in_require = True
            else:
                parts = line[len("require "):].strip().split()
                if len(parts) >= 2:
                    requires[parts[0]] = parts[1]
        elif line == "require(":
            in_require = True

    return {
        "module": module_path,
        "requires": requires,
        "replaces": replaces,
    }


def is_pseudo_version(ver: str) -> bool:
    return bool(PSEUDO_RE.search(ver))


def verify_gomod_text(
    text: str,
    label: str,
    *,
    expected_module: str,
    expected_root_version: Optional[str] = None,
    expected_aikit: Optional[str] = None,
    is_root: bool = False,
) -> List[str]:
    errors: List[str] = []
    info = parse_gomod_info(text)
    if info["module"] != expected_module:
        errors.append(f"{label}: module path is {info['module']!r}, expected {expected_module!r}")
    if info["replaces"]:
        errors.append(f"{label}: forbidden replace directive(s) present: {info['replaces']}")
    if is_root:
        for req_mod in info["requires"]:
            if CYCLE_RE.search(req_mod):
                errors.append(f"{label}: root module must not require backend submodule {req_mod!r}")
    else:
        got_root = info["requires"].get(ROOT_MODULE)
        if expected_root_version is not None and got_root != expected_root_version:
            errors.append(
                f"{label}: requires {ROOT_MODULE} {got_root!r}, expected {expected_root_version!r}"
            )
    aikit_ver = info["requires"].get(AIKIT_MODULE)
    if is_root:
        if not aikit_ver:
            errors.append(f"{label}: root go.mod does not require {AIKIT_MODULE}")
        elif is_pseudo_version(aikit_ver):
            errors.append(f"{label}: {AIKIT_MODULE} pin {aikit_ver!r} is a pseudo-version (releases require a tag)")
    elif aikit_ver:
        if is_pseudo_version(aikit_ver):
            errors.append(f"{label}: {AIKIT_MODULE} pin {aikit_ver!r} is a pseudo-version")
        if expected_aikit and aikit_ver != expected_aikit:
            errors.append(
                f"{label}: {AIKIT_MODULE} pin {aikit_ver!r} disagrees with root {expected_aikit!r}"
            )
    return errors


def _is_vendored_package(name: str, go_vers: Tuple[int, int]) -> bool:
    """Mirror golang.org/x/mod/zip.isVendoredPackage."""
    if go_vers >= (1, 24) and name == "vendor/modules.txt":
        return True
    if name.startswith("vendor/"):
        rest = name[len("vendor/"):]
        return "/" in rest
    j = name.find("/vendor/")
    if j >= 0:
        if go_vers >= (1, 24):
            rest = name[j + len("/vendor/"):]
        else:
            rest = name[len("/vendor/"):]
        return "/" in rest
    return False


def _dirhash_h1(entries: Dict[str, bytes]) -> str:
    """Mirror golang.org/x/mod/sumdb/dirhash.Hash1."""
    outer = hashlib.sha256()
    for name in sorted(entries.keys()):
        if "\n" in name:
            raise ReleasePrepError(f"filename contains newline: {name!r}")
        fh = hashlib.sha256(entries[name]).hexdigest()
        outer.update(f"{fh}  {name}\n".encode("utf-8"))
    return "h1:" + base64.b64encode(outer.digest()).decode("ascii")


def archive_module_files(
    repo: Path,
    rev: str,
    mod_path: str,
    version: str,
    subdir: str = "",
) -> Tuple[Dict[str, bytes], bytes]:
    """Build the exact file map and go.mod bytes for a Go module zip from git rev.

    Matches golang.org/x/mod/zip.CreateFromVCS (filesInGitRepo + checkFiles).
    """
    validate_semver(version, mod_path)
    subdir = subdir.strip("/")
    cmd = ["git", "-c", "core.autocrlf=input", "-c", "core.eol=lf", "archive", "--format=zip", rev]
    if subdir:
        cmd.append(subdir)
    proc = run_cmd(cmd, cwd=repo, text=False)
    with zipfile.ZipFile(io.BytesIO(proc.stdout)) as zf:
        return _extract_module_files_from_zip(zf, repo, rev, mod_path, version, subdir)


def _extract_module_files_from_zip(
    zf: zipfile.ZipFile,
    repo: Path,
    rev: str,
    mod_path: str,
    version: str,
    subdir: str,
) -> Tuple[Dict[str, bytes], bytes]:

    raw_entries: List[Tuple[str, bool, bool, zipfile.ZipInfo]] = []
    have_license = False
    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            continue
        if subdir:
            if not (name.startswith(subdir + "/") or name == subdir):
                continue
            name = name[len(subdir):].lstrip("/")
            if not name:
                continue
        mode = info.external_attr >> 16
        is_symlink = stat.S_ISLNK(mode) if mode else False
        is_reg = (stat.S_ISREG(mode) or mode == 0) and not is_symlink
        raw_entries.append((name, is_reg, is_symlink, info))
        if name == "LICENSE":
            have_license = True

    have_gomod_dirs = set()
    go_vers = (1, 16)
    root_gomod_bytes: Optional[bytes] = None
    for name, is_reg, _, info in raw_entries:
        d, base = posixpath.split(name)
        if base.lower() == "go.mod" and is_reg:
            if d:
                have_gomod_dirs.add(d)
            elif base == "go.mod":
                root_gomod_bytes = zf.read(info)
                go_vers = parse_gomod_go_version(root_gomod_bytes.decode("utf-8"))

    if root_gomod_bytes is None:
        loc = f"{subdir}/go.mod" if subdir else "go.mod"
        raise ReleasePrepError(f"missing {loc} at revision {rev}")

    def in_submodule(p: str) -> bool:
        d = posixpath.dirname(p)
        while d:
            if d in have_gomod_dirs:
                return True
            d = posixpath.dirname(d)
        return False

    # Case-insensitive path & file/dir collision checker (mirrors zip.collisionChecker).
    seen_fold: Dict[str, Tuple[str, bool]] = {}

    def check_collision(p: str, is_dir: bool) -> None:
        fold = p.casefold()
        if fold in seen_fold:
            prev_path, prev_dir = seen_fold[fold]
            if p != prev_path:
                raise ReleasePrepError(f"case-insensitive collision: {prev_path!r} vs {p!r}")
            if is_dir != prev_dir:
                raise ReleasePrepError(f"entry {p!r} is both a file and a directory")
            if not is_dir:
                raise ReleasePrepError(f"duplicate file entry {p!r}")
        else:
            seen_fold[fold] = (p, is_dir)
        parent = posixpath.dirname(p)
        if parent and parent != ".":
            check_collision(parent, True)

    prefix = f"{mod_path}@{version}/"
    files: Dict[str, bytes] = {}
    total_size = 0

    for name, is_reg, is_symlink, info in raw_entries:
        if posixpath.normpath(name) != name or name.startswith("/") or name.startswith("../"):
            raise ReleasePrepError(f"invalid archive path {name!r}")
        if _is_vendored_package(name, go_vers):
            continue
        if in_submodule(name):
            continue
        if name == ".hg_archival.txt":
            continue
        if name.lower() == "go.mod" and name != "go.mod":
            raise ReleasePrepError(f"go.mod must be lowercase, found {name!r}")
        check_collision(name, False)
        if is_symlink or not is_reg:
            continue
        data = zf.read(info)
        size = len(data)
        total_size += size
        if total_size > MAX_ZIP_FILE:
            raise ReleasePrepError(f"module uncompressed size exceeds {MAX_ZIP_FILE} bytes")
        if name == "go.mod" and size > MAX_GO_MOD:
            raise ReleasePrepError(f"go.mod size {size} exceeds {MAX_GO_MOD} bytes")
        if name == "LICENSE" and size > MAX_LICENSE:
            raise ReleasePrepError(f"LICENSE size {size} exceeds {MAX_LICENSE} bytes")
        files[prefix + name] = data

    if subdir and not have_license:
        lic = run_cmd(["git", "cat-file", "blob", f"{rev}:LICENSE"], cwd=repo, check=False, text=False)
        if lic.returncode == 0:
            check_collision("LICENSE", False)
            if len(lic.stdout) > MAX_LICENSE:
                raise ReleasePrepError(f"root LICENSE exceeds {MAX_LICENSE} bytes")
            total_size += len(lic.stdout)
            if total_size > MAX_ZIP_FILE:
                raise ReleasePrepError(f"module uncompressed size exceeds {MAX_ZIP_FILE} bytes")
            files[prefix + "LICENSE"] = lic.stdout

    return files, root_gomod_bytes


def compute_module_hashes(
    repo: Path,
    rev: str,
    mod_path: str,
    version: str,
    subdir: str = "",
) -> dict:
    files, gomod_bytes = archive_module_files(repo, rev, mod_path, version, subdir=subdir)
    return {
        "module": mod_path,
        "version": version,
        "rev": git_rev_parse(repo, rev),
        "subdir": subdir,
        "file_count": len(files),
        "zip_h1": _dirhash_h1(files),
        "gomod_h1": _dirhash_h1({"go.mod": gomod_bytes}),
    }


def stage_proxy_module(
    proxy_dir: Path,
    repo: Path,
    rev: str,
    mod_path: str,
    version: str,
    subdir: str = "",
) -> dict:
    """Write a canonical Go module entry into a file:// GOPROXY directory."""
    files, gomod_bytes = archive_module_files(repo, rev, mod_path, version, subdir=subdir)
    commit_sha = git_rev_parse(repo, rev)
    commit_time = git_commit_utc(repo, commit_sha)

    mod_dir = proxy_dir / escape_proxy_segment(mod_path) / "@v"
    mod_dir.mkdir(parents=True, exist_ok=True)
    esc_ver = escape_proxy_segment(version)

    # Update @v/list preserving any other staged versions.
    list_path = mod_dir / "list"
    existing = set()
    if list_path.exists():
        existing = {ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
    existing.add(version)
    list_path.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")

    # Write <ver>.info and <ver>.mod.
    info_path = mod_dir / f"{esc_ver}.info"
    info_path.write_text(
        json.dumps({"Version": version, "Time": commit_time}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    mod_file_path = mod_dir / f"{esc_ver}.mod"
    mod_file_path.write_bytes(gomod_bytes)

    # Write deterministic <ver>.zip.
    zip_path = mod_dir / f"{esc_ver}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for fname in sorted(files.keys()):
            zi = zipfile.ZipInfo(filename=fname, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zout.writestr(zi, files[fname])

    zip_size = zip_path.stat().st_size
    if zip_size > MAX_ZIP_FILE:
        raise ReleasePrepError(f"zip file size {zip_size} exceeds {MAX_ZIP_FILE} bytes")

    return {
        "module": mod_path,
        "version": version,
        "rev": commit_sha,
        "commit_time": commit_time,
        "subdir": subdir,
        "file_count": len(files),
        "zip_bytes": zip_size,
        "zip_h1": _dirhash_h1(files),
        "gomod_h1": _dirhash_h1({"go.mod": gomod_bytes}),
        "proxy_dir": str(proxy_dir),
        "proxy_url": proxy_dir.resolve().as_uri(),
        "zip_path": str(zip_path),
    }


def verify_gosum_entries(
    gosum_text: str,
    mod_path: str,
    version: str,
    zip_h1: str,
    gomod_h1: str,
    label: str,
) -> List[str]:
    errors: List[str] = []
    zip_hashes = set()
    mod_hashes = set()
    for line in gosum_text.splitlines():
        parts = line.strip().split()
        if len(parts) != 3 or parts[0] != mod_path:
            continue
        if parts[1] == version:
            zip_hashes.add(parts[2])
        elif parts[1] == f"{version}/go.mod":
            mod_hashes.add(parts[2])
    if zip_hashes != {zip_h1}:
        errors.append(
            f"{label}: {mod_path} {version} hash mismatch: got {sorted(zip_hashes)}, expected [{zip_h1}]"
        )
    if mod_hashes != {gomod_h1}:
        errors.append(
            f"{label}: {mod_path} {version}/go.mod hash mismatch: got {sorted(mod_hashes)}, expected [{gomod_h1}]"
        )
    return errors


def _build_go_env(
    *,
    gomodcache: Path,
    goproxy: str,
    gosumdb: Optional[str] = None,
    gonosumdb: Optional[str] = None,
    goos: Optional[str] = None,
    goarch: Optional[str] = None,
    cgo_enabled: Optional[str] = None,
    gomaxprocs: Optional[int] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["GOWORK"] = "off"
    env["GOMODCACHE"] = str(gomodcache)
    env["GOPROXY"] = goproxy
    if gosumdb is not None:
        env["GOSUMDB"] = gosumdb
    if gonosumdb is not None:
        env["GONOSUMDB"] = gonosumdb
        env["GOPRIVATE"] = ""
    if goos:
        env["GOOS"] = goos
    if goarch:
        env["GOARCH"] = goarch
    elif "GOARCH" in env and goos and goos != "darwin":
        # Preserve host GOARCH unless explicitly overridden.
        pass
    if cgo_enabled is not None:
        env["CGO_ENABLED"] = cgo_enabled
    if gomaxprocs is not None:
        env["GOMAXPROCS"] = str(gomaxprocs)
    return env


def _local_download_proxy_url() -> Optional[str]:
    proc = run_cmd(["go", "env", "GOMODCACHE"], check=False)
    if proc.returncode != 0:
        return None
    cache_dl = Path(proc.stdout.strip()) / "cache" / "download"
    if cache_dl.is_dir():
        return cache_dl.resolve().as_uri()
    return None


def _compose_staged_proxy(staged_url: str, *, offline_deps: bool, use_local_cache: bool = True) -> str:
    parts = [staged_url]
    if use_local_cache:
        local_dl = _local_download_proxy_url()
        if local_dl:
            parts.append(local_dl)
    if not offline_deps:
        parts.extend(["https://proxy.golang.org", "direct"])
    return ",".join(parts)


def _git_status_paths(repo: Path) -> List[str]:
    proc = run_cmd(["git", "status", "--porcelain", "-z"], cwd=repo, text=False)
    raw = proc.stdout.split(b"\0")
    paths: List[str] = []
    i = 0
    while i < len(raw):
        entry = raw[i]
        i += 1
        if not entry:
            continue
        status = entry[:2].decode("utf-8", errors="replace")
        path = entry[3:].decode("utf-8", errors="replace")
        paths.append(path)
        if status[0] in ("R", "C") and i < len(raw):
            # Skip original path for renames/copies.
            i += 1
    return paths


def _check_existing_local_tags(repo: Path, version: str, selected: Sequence[str]) -> List[str]:
    tags = [version] + [f"{m}/{version}" for m in selected]
    existing = []
    for t in tags:
        res = run_cmd(["git", "rev-parse", "-q", "--verify", f"refs/tags/{t}"], cwd=repo, check=False)
        if res.returncode == 0:
            existing.append(t)
    return existing


def _detect_root_consumer_pkg(repo: Path, rev: str) -> str:
    """Pick a lightweight real root package to import in consumer smoke checks."""
    for pkg in ("embeddinggemma2", "decoder/config"):
        res = run_cmd(["git", "ls-tree", "-r", "--name-only", rev, pkg], cwd=repo, check=False)
        if res.returncode == 0 and any(ln.endswith(".go") and not ln.endswith("_test.go") for ln in res.stdout.splitlines()):
            return f"{ROOT_MODULE}/{pkg}"
    return ROOT_MODULE


def _run_consumer_suite(
    *,
    version: str,
    selected: Sequence[str],
    root_import: str,
    root_zip_h1: str,
    sub_zip_h1s: Dict[str, str],
    goproxy: str,
    gosumdb: Optional[str],
    gonosumdb: Optional[str],
    gomaxprocs: Optional[int],
) -> List[str]:
    """Build & vet tiny out-of-tree consumer modules against the given Go proxy."""
    errors: List[str] = []
    cases: List[Tuple[str, Optional[SubmoduleSpec]]] = [("root-only", None)]
    for m in selected:
        cases.append((m, SUBMODULES[m]))

    tmp_root = Path(tempfile.mkdtemp(prefix="goinfer-consumer-"))
    try:
        modcache = tmp_root / "modcache"
        modcache.mkdir()
        for idx, (label, spec) in enumerate(cases):
            cdir = tmp_root / f"case-{idx}-{label.replace('/', '_')}"
            cdir.mkdir()
            imports = [f'\t_ "{root_import}"']
            reqs = [f"\t{ROOT_MODULE} {version}"]
            if spec is not None:
                imports.append(f'\t_ "{spec.consumer_import}"')
                reqs.append(f"\t{spec.module_path} {version}")
            (cdir / "go.mod").write_text(
                "module example.com/goinfer-release-consumer\n\n"
                "go 1.27.0\n\n"
                "require (\n" + "\n".join(reqs) + "\n)\n",
                encoding="utf-8",
            )
            (cdir / "main.go").write_text(
                "package main\n\nimport (\n" + "\n".join(imports) + "\n)\n\nfunc main() {}\n",
                encoding="utf-8",
            )
            env = _build_go_env(
                gomodcache=modcache,
                goproxy=goproxy,
                gosumdb=gosumdb,
                gonosumdb=gonosumdb,
                goos=spec.goos if spec else None,
                goarch=spec.goarch if spec else None,
                cgo_enabled=spec.cgo_enabled if spec else "0",
                gomaxprocs=gomaxprocs,
            )
            try:
                run_cmd(["go", "mod", "tidy"], cwd=cdir, env=env)
                run_cmd(["go", "mod", "verify"], cwd=cdir, env=env)
                # Confirm consumer go.sum records exact canonical zip hashes.
                csum = (cdir / "go.sum").read_text(encoding="utf-8")
                if f"{ROOT_MODULE} {version} {root_zip_h1}" not in csum:
                    errors.append(f"consumer[{label}]: go.sum missing expected root zip hash {root_zip_h1}")
                if spec is not None and spec.rel_dir in sub_zip_h1s:
                    want_sub = f"{spec.module_path} {version} {sub_zip_h1s[spec.rel_dir]}"
                    if want_sub not in csum:
                        errors.append(f"consumer[{label}]: go.sum missing expected submodule hash {want_sub}")
                tag_args = ["-tags", spec.build_tags] if (spec and spec.build_tags) else []
                run_cmd(["go", "vet", *tag_args, "./..."], cwd=cdir, env=env)
                run_cmd(["go", "build", *tag_args, "-o", str(cdir / "consumer.bin"), "./..."], cwd=cdir, env=env)
                print(f"  [PASS] consumer {label}: go mod tidy + verify + vet + build")
            except ReleasePrepError as exc:
                errors.append(f"consumer[{label}]: {exc}")
                print(f"  [FAIL] consumer {label}: {exc}", file=sys.stderr)
    finally:
        _make_writable_and_rmtree(tmp_root)
    return errors


def _parse_ls_remote_tags(output: str) -> Dict[str, str]:
    """Return map tag_name -> commit_sha, peeling annotated tags (^{})."""
    raw: Dict[str, str] = {}
    peeled: Dict[str, str] = {}
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        sha, ref = parts
        if not ref.startswith("refs/tags/"):
            continue
        name = ref[len("refs/tags/"):]
        if name.endswith("^{}"):
            peeled[name[:-3]] = sha
        else:
            raw[name] = sha
    merged = dict(raw)
    merged.update(peeled)
    return merged


# ---- Subcommand implementations ----


def cmd_stage_proxy(args: argparse.Namespace) -> int:
    repo = git_root(args.repo)
    rev = git_rev_parse(repo, args.rev)
    validate_semver(args.version, ROOT_MODULE)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root_meta = stage_proxy_module(out_dir, repo, rev, ROOT_MODULE, args.version, subdir="")
    result: dict = {"root": root_meta, "submodules": {}}
    if args.include_submodules:
        for mod in parse_modules_arg(args.include_submodules):
            spec = SUBMODULES[mod]
            result["submodules"][mod] = stage_proxy_module(
                out_dir, repo, rev, spec.module_path, args.version, subdir=spec.rel_dir
            )
    print(json.dumps(result, indent=2))
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    repo = git_root(args.repo)
    rev = git_rev_parse(repo, args.rev)
    validate_semver(args.version, ROOT_MODULE)
    selected = parse_modules_arg(args.modules)
    omitted = [m for m in SUBMODULES if m not in selected]

    print(f"release_prep prep: version={args.version} rev={rev[:12]} selected={selected or ['(root-only)']} omitted={omitted}")

    # 1. Refuse if any target tag already exists locally.
    existing_tags = _check_existing_local_tags(repo, args.version, selected)
    if existing_tags:
        print(f"ERROR: target tag(s) already exist locally: {', '.join(existing_tags)}", file=sys.stderr)
        print("VERDICT: FAIL (existing local tags)")
        return 1

    # 2. Refuse if any file outside the selected submodules' go.mod/go.sum is dirty.
    allowed_dirty = set()
    for m in selected:
        allowed_dirty.add(f"{m}/go.mod")
        allowed_dirty.add(f"{m}/go.sum")
    dirty = _git_status_paths(repo)
    disallowed = [p for p in dirty if p not in allowed_dirty]
    if disallowed:
        print(
            "ERROR: working tree has uncommitted changes outside selected submodule go.mod/go.sum "
            "(commit root release files first so the staged root zip matches HEAD):\n  "
            + "\n  ".join(disallowed),
            file=sys.stderr,
        )
        print("VERDICT: FAIL (dirty root/unrelated files)")
        return 1

    # 3. Validate root go.mod at rev.
    root_gomod_text = run_cmd(["git", "show", f"{rev}:go.mod"], cwd=repo).stdout
    root_info = parse_gomod_info(root_gomod_text)
    root_aikit = root_info["requires"].get(AIKIT_MODULE)
    root_errs = verify_gomod_text(
        root_gomod_text, "go.mod", expected_module=ROOT_MODULE, is_root=True
    )
    if root_errs:
        for e in root_errs:
            print(f"ERROR: {e}", file=sys.stderr)
        print("VERDICT: FAIL (invalid root go.mod)")
        return 1

    # 4. Stage canonical root zip in isolated file:// proxy + isolated GOMODCACHE.
    keep_dir = Path(args.keep_proxy_dir).resolve() if args.keep_proxy_dir else None
    tmp_root = Path(tempfile.mkdtemp(prefix="goinfer-release-prep-"))
    try:
        proxy_dir = keep_dir if keep_dir else (tmp_root / "proxy")
        proxy_dir.mkdir(parents=True, exist_ok=True)
        modcache = tmp_root / "modcache"
        modcache.mkdir()

        root_meta = stage_proxy_module(proxy_dir, repo, rev, ROOT_MODULE, args.version, subdir="")
        print(
            f"staged root proxy: files={root_meta['file_count']} zip_bytes={root_meta['zip_bytes']}\n"
            f"  zip_h1   = {root_meta['zip_h1']}\n"
            f"  gomod_h1 = {root_meta['gomod_h1']}\n"
            f"  proxy    = {root_meta['proxy_url']}"
        )

        goproxy = _compose_staged_proxy(
            root_meta["proxy_url"],
            offline_deps=args.offline_deps,
            use_local_cache=not args.no_local_cache,
        )
        gosumdb = "off" if args.offline_deps else None
        gonosumdb = None if args.offline_deps else ROOT_MODULE

        errors: List[str] = []
        for m in selected:
            spec = SUBMODULES[m]
            mod_dir = repo / spec.rel_dir
            env = _build_go_env(
                gomodcache=modcache,
                goproxy=goproxy,
                gosumdb=gosumdb,
                gonosumdb=gonosumdb,
                goos=spec.goos,
                goarch=spec.goarch or None,
                cgo_enabled=spec.cgo_enabled,
                gomaxprocs=args.gomaxprocs,
            )
            print(f"updating {m} (GOWORK=off, GOOS={spec.goos}{',' + spec.goarch if spec.goarch else ''}, CGO_ENABLED={spec.cgo_enabled}, tags={spec.build_tags or 'none'})...")
            run_cmd(["go", "mod", "edit", f"-require={ROOT_MODULE}@{args.version}"], cwd=mod_dir, env=env)
            run_cmd(["go", "mod", "tidy"], cwd=mod_dir, env=env)

            gomod_txt = (mod_dir / "go.mod").read_text(encoding="utf-8")
            gosum_txt = (mod_dir / "go.sum").read_text(encoding="utf-8")
            errors.extend(
                verify_gomod_text(
                    gomod_txt,
                    f"{m}/go.mod",
                    expected_module=spec.module_path,
                    expected_root_version=args.version,
                    expected_aikit=root_aikit,
                    is_root=False,
                )
            )
            errors.extend(
                verify_gosum_entries(
                    gosum_txt,
                    ROOT_MODULE,
                    args.version,
                    root_meta["zip_h1"],
                    root_meta["gomod_h1"],
                    f"{m}/go.sum",
                )
            )
            run_cmd(["go", "mod", "verify"], cwd=mod_dir, env=env)

            if not args.skip_build:
                tag_args = ["-tags", spec.build_tags] if spec.build_tags else []
                run_cmd(["go", "vet", *tag_args, "./..."], cwd=mod_dir, env=env)
                run_cmd(["go", "build", *tag_args, "./..."], cwd=mod_dir, env=env)
                print(f"  [PASS] {m}: go.mod/go.sum pinned + go mod verify + standalone vet/build")
            else:
                print(f"  [PASS] {m}: go.mod/go.sum pinned + go mod verify (--skip-build set)")
            print(f"  note: {spec.runtime_note}")

        if omitted:
            print("omitted backend module(s) (untouched; document omission in CHANGELOG & GitHub Release):")
            for om in omitted:
                om_gomod = repo / om / "go.mod"
                cur = "(missing)"
                if om_gomod.exists():
                    cur = parse_gomod_info(om_gomod.read_text(encoding="utf-8"))["requires"].get(ROOT_MODULE, "(none)")
                print(f"  - {om} (current root pin: {cur})")

        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            print(f"VERDICT: FAIL ({len(errors)} validation error(s))")
            return 1

        # Print exact manual next steps (never auto-commit or auto-tag).
        touched_files = " ".join(f"{m}/go.mod {m}/go.sum" for m in selected)
        tags_list = " ".join([args.version] + [f"{m}/{args.version}" for m in selected])
        print("\nNext steps (no commit or tags were created):")
        if selected:
            print(f"  1. Fold into your unpushed release commit (or create one release commit):")
            print(f"       git add -- {touched_files}")
            print(f"       git commit --amend --no-edit   # or: git commit -m \"release: {args.version} (<summary>)\"")
        print(f"  2. Re-verify canonical hashes on the FINAL commit before tagging:")
        print(f"       python3 scripts/release_prep.py verify-commit --version {args.version} --modules {','.join(selected) or 'none'}")
        print(f"  3. Create ALL release tags on that single commit:")
        for t in [args.version] + [f"{m}/{args.version}" for m in selected]:
            print(f"       git tag {t}")
        print(f"  4. Verify all tags point to the exact same commit:")
        print(f"       python3 scripts/release_prep.py verify-commit --version {args.version} --modules {','.join(selected) or 'none'} --check-tags")
        print(f"  5. Push main and all release tags together:")
        print(f"       git push origin main {tags_list}")
        print(f"  6. Verify public Go proxy & consumer builds after push:")
        print(f"       python3 scripts/release_prep.py verify-public --version {args.version} --modules {','.join(selected) or 'none'}")
        print("VERDICT: PASS")
        return 0
    finally:
        _make_writable_and_rmtree(tmp_root)


def cmd_verify_commit(args: argparse.Namespace) -> int:
    repo = git_root(args.repo)
    rev = git_rev_parse(repo, args.rev)
    validate_semver(args.version, ROOT_MODULE)
    selected = parse_modules_arg(args.modules)
    omitted = [m for m in SUBMODULES if m not in selected]
    errors: List[str] = []

    print(f"release_prep verify-commit: version={args.version} rev={rev} selected={selected or ['(root-only)']} omitted={omitted}")

    if args.rev == "HEAD" and not args.allow_dirty:
        dirty = _git_status_paths(repo)
        if dirty:
            errors.append(f"working tree is not clean ({len(dirty)} dirty path(s)): {', '.join(dirty[:5])}")

    root_gomod_text = run_cmd(["git", "show", f"{rev}:go.mod"], cwd=repo).stdout
    root_info = parse_gomod_info(root_gomod_text)
    root_aikit = root_info["requires"].get(AIKIT_MODULE)
    errors.extend(verify_gomod_text(root_gomod_text, "go.mod", expected_module=ROOT_MODULE, is_root=True))

    root_hashes = compute_module_hashes(repo, rev, ROOT_MODULE, args.version, subdir="")
    print(
        f"canonical root @ {rev[:12]}: files={root_hashes['file_count']}\n"
        f"  zip_h1   = {root_hashes['zip_h1']}\n"
        f"  gomod_h1 = {root_hashes['gomod_h1']}"
    )

    sub_zip_h1s: Dict[str, str] = {}
    for m in selected:
        spec = SUBMODULES[m]
        gomod_txt = run_cmd(["git", "show", f"{rev}:{m}/go.mod"], cwd=repo).stdout
        gosum_txt = run_cmd(["git", "show", f"{rev}:{m}/go.sum"], cwd=repo).stdout
        errors.extend(
            verify_gomod_text(
                gomod_txt,
                f"{m}/go.mod@{rev[:12]}",
                expected_module=spec.module_path,
                expected_root_version=args.version,
                expected_aikit=root_aikit,
                is_root=False,
            )
        )
        errors.extend(
            verify_gosum_entries(
                gosum_txt,
                ROOT_MODULE,
                args.version,
                root_hashes["zip_h1"],
                root_hashes["gomod_h1"],
                f"{m}/go.sum@{rev[:12]}",
            )
        )
        sh = compute_module_hashes(repo, rev, spec.module_path, args.version, subdir=spec.rel_dir)
        sub_zip_h1s[m] = sh["zip_h1"]
        print(f"  submodule {m} @ {rev[:12]}: files={sh['file_count']} zip_h1={sh['zip_h1']}")

    if args.check_tags:
        expected_tags = [args.version] + [f"{m}/{args.version}" for m in selected]
        for t in expected_tags:
            res = run_cmd(["git", "rev-parse", "-q", "--verify", f"refs/tags/{t}^{{commit}}"], cwd=repo, check=False)
            if res.returncode != 0:
                errors.append(f"missing expected local tag {t}")
            else:
                tag_sha = res.stdout.strip()
                if tag_sha != rev:
                    errors.append(f"tag {t} resolves to {tag_sha}, expected single release commit {rev}")
                else:
                    print(f"  [PASS] tag {t} -> {tag_sha[:12]}")
        for om in omitted:
            bad_tag = f"{om}/{args.version}"
            res = run_cmd(["git", "rev-parse", "-q", "--verify", f"refs/tags/{bad_tag}"], cwd=repo, check=False)
            if res.returncode == 0:
                errors.append(f"unselected backend tag {bad_tag} exists locally")

    if args.consumer_check and not errors:
        tmp_proxy = Path(tempfile.mkdtemp(prefix="goinfer-verify-proxy-"))
        try:
            root_meta = stage_proxy_module(tmp_proxy, repo, rev, ROOT_MODULE, args.version, subdir="")
            for m in selected:
                spec = SUBMODULES[m]
                stage_proxy_module(tmp_proxy, repo, rev, spec.module_path, args.version, subdir=spec.rel_dir)
            goproxy = _compose_staged_proxy(
                root_meta["proxy_url"],
                offline_deps=args.offline_deps,
                use_local_cache=not args.no_local_cache,
            )
            root_import = _detect_root_consumer_pkg(repo, rev)
            errors.extend(
                _run_consumer_suite(
                    version=args.version,
                    selected=selected,
                    root_import=root_import,
                    root_zip_h1=root_hashes["zip_h1"],
                    sub_zip_h1s=sub_zip_h1s,
                    goproxy=goproxy,
                    gosumdb="off" if args.offline_deps else None,
                    gonosumdb=None if args.offline_deps else f"{ROOT_MODULE},{ROOT_MODULE}/*",
                    gomaxprocs=args.gomaxprocs,
                )
            )
        finally:
            _make_writable_and_rmtree(tmp_proxy)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"VERDICT: FAIL ({len(errors)} error(s))")
        return 1
    print("VERDICT: PASS")
    return 0


def cmd_verify_public(args: argparse.Namespace) -> int:
    repo = git_root(args.repo)
    validate_semver(args.version, ROOT_MODULE)
    selected = parse_modules_arg(args.modules)
    omitted = [m for m in SUBMODULES if m not in selected]
    errors: List[str] = []

    print(
        f"release_prep verify-public: version={args.version} proxy={args.proxy} sumdb={args.sumdb} "
        f"selected={selected or ['(root-only)']} omitted={omitted}"
    )

    # 1. Verify remote tags all resolve to the exact same release commit.
    ls_out = run_cmd(["git", "ls-remote", "--tags", args.remote], cwd=repo).stdout
    tag_map = _parse_ls_remote_tags(ls_out)
    expected_tags = [args.version] + [f"{m}/{args.version}" for m in selected]
    commit_shas = set()
    for t in expected_tags:
        if t not in tag_map:
            errors.append(f"remote {args.remote!r} missing tag {t}")
        else:
            commit_shas.add(tag_map[t])
            print(f"  remote tag {t} -> {tag_map[t][:12]}")
    for om in omitted:
        bad_tag = f"{om}/{args.version}"
        if bad_tag in tag_map:
            errors.append(f"remote {args.remote!r} has unselected backend tag {bad_tag}")

    release_commit: Optional[str] = None
    if len(commit_shas) > 1:
        errors.append(f"release tags do not name a single commit: {sorted(commit_shas)}")
    elif len(commit_shas) == 1:
        release_commit = next(iter(commit_shas))
        if args.expected_commit:
            want_sha = git_rev_parse(repo, args.expected_commit)
            if release_commit != want_sha:
                errors.append(f"remote tags resolve to {release_commit}, expected {want_sha}")

    # 2. Compute canonical local git hashes if release_commit is available locally.
    expected_zips: Dict[str, str] = {}
    root_import = f"{ROOT_MODULE}/embeddinggemma2"
    if release_commit:
        have_local = run_cmd(["git", "cat-file", "-e", f"{release_commit}^{{commit}}"], cwd=repo, check=False)
        if have_local.returncode == 0:
            rh = compute_module_hashes(repo, release_commit, ROOT_MODULE, args.version, subdir="")
            expected_zips[ROOT_MODULE] = rh["zip_h1"]
            root_import = _detect_root_consumer_pkg(repo, release_commit)
            for m in selected:
                spec = SUBMODULES[m]
                sh = compute_module_hashes(repo, release_commit, spec.module_path, args.version, subdir=spec.rel_dir)
                expected_zips[spec.module_path] = sh["zip_h1"]
        else:
            errors.append(
                f"release commit {release_commit} not found locally (run `git fetch {args.remote}` to cross-check zip hashes)"
            )

    # 3. Resolve root and selected submodules strictly via the configured proxy & sumdb.
    tmp_root = Path(tempfile.mkdtemp(prefix="goinfer-verify-public-"))
    try:
        modcache = tmp_root / "modcache"
        modcache.mkdir()
        env = _build_go_env(
            gomodcache=modcache,
            goproxy=args.proxy,
            gosumdb=args.sumdb,
            gonosumdb="",
            gomaxprocs=args.gomaxprocs,
        )
        targets = [ROOT_MODULE] + [SUBMODULES[m].module_path for m in selected]
        downloaded_sums: Dict[str, str] = {}
        for mod in targets:
            try:
                listed = json.loads(
                    run_cmd(["go", "list", "-m", "-json", f"{mod}@{args.version}"], cwd=tmp_root, env=env).stdout
                )
                if listed.get("Version") != args.version:
                    errors.append(f"go list -m {mod}@{args.version}: got version {listed.get('Version')!r}")
                dl = json.loads(
                    run_cmd(["go", "mod", "download", "-json", f"{mod}@{args.version}"], cwd=tmp_root, env=env).stdout
                )
                if dl.get("Error"):
                    errors.append(f"go mod download {mod}@{args.version}: {dl['Error']}")
                    continue
                got_sum = dl.get("Sum", "")
                downloaded_sums[mod] = got_sum
                if mod in expected_zips and got_sum != expected_zips[mod]:
                    errors.append(
                        f"proxy zip checksum mismatch for {mod}@{args.version}: proxy={got_sum}, git({release_commit[:12]})={expected_zips[mod]}"
                    )
                else:
                    print(f"  [PASS] proxy {mod}@{args.version} Sum={got_sum}")
            except ReleasePrepError as exc:
                errors.append(str(exc))

        if not args.skip_consumer_build and not errors:
            sub_zip_h1s = {m: downloaded_sums[SUBMODULES[m].module_path] for m in selected}
            errors.extend(
                _run_consumer_suite(
                    version=args.version,
                    selected=selected,
                    root_import=root_import,
                    root_zip_h1=downloaded_sums[ROOT_MODULE],
                    sub_zip_h1s=sub_zip_h1s,
                    goproxy=args.proxy,
                    gosumdb=args.sumdb,
                    gonosumdb="",
                    gomaxprocs=args.gomaxprocs,
                )
            )
    finally:
        _make_writable_and_rmtree(tmp_root)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"VERDICT: FAIL ({len(errors)} error(s))")
        return 1
    print("VERDICT: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-commit release prep and verification for goinfer root + backend modules."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("stage-proxy", help="Stage canonical module zip(s) from git rev into a file:// proxy.")
    sp.add_argument("--version", required=True, help="Candidate semver tag (e.g. v0.6.2).")
    sp.add_argument("--out-dir", required=True, help="Directory to populate as a file:// GOPROXY.")
    sp.add_argument("--rev", default="HEAD", help="Git revision to archive (default: HEAD).")
    sp.add_argument("--repo", default=None, help="Git repository path (default: current repo).")
    sp.add_argument("--include-submodules", default="", help="Optional comma-separated backend modules to stage too.")
    sp.set_defaults(func=cmd_stage_proxy)

    pr = sub.add_parser("prep", help="Pin and verify selected submodule go.mod/go.sum outside go.work.")
    pr.add_argument("--version", required=True, help="Candidate semver tag (e.g. v0.6.2).")
    pr.add_argument(
        "--modules",
        default=",".join(DEFAULT_MODULES),
        help="Comma-separated backend modules to release (default: backend/cuda,backend/vulkan; or 'none').",
    )
    pr.add_argument("--rev", default="HEAD", help="Git revision to archive for the root zip (default: HEAD).")
    pr.add_argument("--repo", default=None, help="Git repository path (default: current repo).")
    pr.add_argument("--keep-proxy-dir", default=None, help="Keep the staged file:// proxy at this path.")
    pr.add_argument("--skip-build", action="store_true", help="Skip standalone go vet/go build (pin + hash + go mod verify only).")
    pr.add_argument("--offline-deps", action="store_true", help="Use staged proxy + local cache only (no network).")
    pr.add_argument("--no-local-cache", action="store_true", help="Do not chain local GOMODCACHE/cache/download.")
    pr.add_argument("--gomaxprocs", type=int, default=None, help="Optional GOMAXPROCS cap for go commands.")
    pr.set_defaults(func=cmd_prep)

    vc = sub.add_parser("verify-commit", help="Verify final release commit hashes, pins, tags, and consumer builds.")
    vc.add_argument("--version", required=True, help="Candidate semver tag (e.g. v0.6.2).")
    vc.add_argument(
        "--modules",
        default=",".join(DEFAULT_MODULES),
        help="Comma-separated backend modules in this release (default: backend/cuda,backend/vulkan; or 'none').",
    )
    vc.add_argument("--rev", default="HEAD", help="Git commit to verify (default: HEAD).")
    vc.add_argument("--repo", default=None, help="Git repository path (default: current repo).")
    vc.add_argument("--check-tags", action="store_true", help="Verify local root and submodule tags all resolve to --rev.")
    vc.add_argument("--consumer-check", action="store_true", help="Build/vet out-of-tree consumers against a staged file:// proxy.")
    vc.add_argument("--allow-dirty", action="store_true", help="Allow verifying HEAD with dirty working tree.")
    vc.add_argument("--offline-deps", action="store_true", help="Use staged proxy + local cache only during --consumer-check.")
    vc.add_argument("--no-local-cache", action="store_true", help="Do not chain local GOMODCACHE/cache/download.")
    vc.add_argument("--gomaxprocs", type=int, default=None, help="Optional GOMAXPROCS cap for go commands.")
    vc.set_defaults(func=cmd_verify_commit)

    vp = sub.add_parser("verify-public", help="Verify published tags, public Go proxy checksums, and consumer builds.")
    vp.add_argument("--version", required=True, help="Published semver tag (e.g. v0.6.2).")
    vp.add_argument(
        "--modules",
        default=",".join(DEFAULT_MODULES),
        help="Comma-separated backend modules in this release (default: backend/cuda,backend/vulkan; or 'none').",
    )
    vp.add_argument("--proxy", default="https://proxy.golang.org", help="Go proxy URL (default: https://proxy.golang.org).")
    vp.add_argument("--sumdb", default="sum.golang.org", help="Go checksum database (default: sum.golang.org).")
    vp.add_argument("--remote", default="origin", help="Git remote or URL to check published tags (default: origin).")
    vp.add_argument("--expected-commit", default=None, help="Optional expected commit SHA for the tags.")
    vp.add_argument("--repo", default=None, help="Git repository path (default: current repo).")
    vp.add_argument("--skip-consumer-build", action="store_true", help="Verify tags + proxy download/checksums only.")
    vp.add_argument("--gomaxprocs", type=int, default=None, help="Optional GOMAXPROCS cap for go commands.")
    vp.set_defaults(func=cmd_verify_public)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ReleasePrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("VERDICT: FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
