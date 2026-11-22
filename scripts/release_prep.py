#!/usr/bin/env python3
"""Stage a temporary file:// Go proxy to pin backend submodules on one release commit.

Uses Go's own canonical module zip for an already-pushed base commit, relabels
only the zip entry prefix to the candidate semver tag, verifies the relabeled
zip with `go mod download -json`, and updates/validates selected nested backend
submodules (`backend/cuda`, `backend/vulkan`, optionally `backend/metal`)
outside `go.work` using an isolated temporary GOMODCACHE.

Because Go's canonical module zip excludes nested directories that have their
own go.mod, editing only nested `backend/*/go.{mod,sum}` files after the pushed
base commit leaves the root zip hash bit-for-bit unchanged. Root and selected
submodule tags can therefore all name one commit.

This script NEVER creates or pushes git tags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_MODULE = "github.com/pkieltyka/goinfer"
AIKIT_MODULE = "github.com/pkieltyka/aikit"

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
class Submodule:
    rel_dir: str
    module_path: str
    goos: str
    goarch: str
    build_tags: str
    note: str


SUBMODULES: Dict[str, Submodule] = {
    "backend/cuda": Submodule(
        "backend/cuda",
        f"{ROOT_MODULE}/backend/cuda",
        "linux",
        "",
        "cuda",
        "Linux CUDA device gate (§C1-C) runs separately on real hardware",
    ),
    "backend/vulkan": Submodule(
        "backend/vulkan",
        f"{ROOT_MODULE}/backend/vulkan",
        "linux",
        "",
        "",
        "Discrete AMD Radeon Vulkan check runs separately on real hardware",
    ),
    "backend/metal": Submodule(
        "backend/metal",
        f"{ROOT_MODULE}/backend/metal",
        "darwin",
        "arm64",
        "metal",
        "Cross-builds darwin/arm64 on Linux; Metal device gate (§C1-M) runs on Apple Silicon",
    ),
}
DEFAULT_MODULES = ("backend/cuda", "backend/vulkan")


class PrepError(RuntimeError):
    pass


def rmtree_ro(path: Path) -> None:
    """Remove a temp tree even when Go marks GOMODCACHE files 0444/0555."""
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        os.chmod(root, 0o700)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o700)
        for f in files:
            os.chmod(os.path.join(root, f), 0o600)
    shutil.rmtree(path)


def run(cmd: Sequence[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, env=env, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise PrepError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stdout}{proc.stderr}".rstrip())
    return proc


def git_root(repo: Optional[str]) -> Path:
    base = Path(repo).resolve() if repo else Path.cwd()
    return Path(run(["git", "rev-parse", "--show-toplevel"], cwd=base).stdout.strip())


def git_commit(repo: Path, rev: str) -> str:
    return run(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"], cwd=repo).stdout.strip()


def validate_semver(version: str) -> None:
    m = SEMVER_RE.match(version)
    if not m:
        raise PrepError(f"invalid canonical Go semver {version!r} (expected e.g. v0.6.2 or v0.6.2-rc.1)")
    if int(m.group(1)) >= 2:
        raise PrepError(f"version {version!r} has major >= 2; {ROOT_MODULE} has no /vN suffix")


def parse_modules(raw: str) -> List[str]:
    s = raw.strip()
    if not s or s.lower() == "none":
        return []
    out: List[str] = []
    for item in s.split(","):
        m = item.strip().strip("/")
        if not m:
            continue
        if m not in SUBMODULES:
            raise PrepError(f"unknown backend module {m!r}; choose from {', '.join(SUBMODULES)} or 'none'")
        if m not in out:
            out.append(m)
    return out


def parse_gomod(text: str) -> Tuple[str, Dict[str, str], List[str]]:
    mod_path, reqs, replaces = "", {}, []
    in_req = in_rep = False
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line == ")":
            in_req = in_rep = False
            continue
        if in_rep:
            replaces.append(line)
            continue
        if in_req:
            parts = line.split()
            if len(parts) >= 2:
                reqs[parts[0]] = parts[1]
            continue
        if line.startswith("module "):
            mod_path = line.split()[1].strip('"')
        elif line.startswith("replace ") or line == "replace(":
            if line.endswith("("):
                in_rep = True
            else:
                replaces.append(line[len("replace "):].strip())
        elif line.startswith("require ") or line == "require(":
            if line.endswith("("):
                in_req = True
            else:
                parts = line[len("require "):].strip().split()
                if len(parts) >= 2:
                    reqs[parts[0]] = parts[1]
    return mod_path, reqs, replaces


def check_gomod(text: str, label: str, *, expected_mod: str, root_ver: Optional[str] = None, expected_aikit: Optional[str] = None, is_root: bool = False) -> List[str]:
    errs: List[str] = []
    mod_path, reqs, replaces = parse_gomod(text)
    if mod_path != expected_mod:
        errs.append(f"{label}: module is {mod_path!r}, expected {expected_mod!r}")
    if replaces:
        errs.append(f"{label}: forbidden replace directive(s): {replaces}")
    if is_root:
        for dep in reqs:
            if CYCLE_RE.search(dep):
                errs.append(f"{label}: root module must not require backend submodule {dep!r}")
    elif root_ver is not None and reqs.get(ROOT_MODULE) != root_ver:
        errs.append(f"{label}: requires {ROOT_MODULE} {reqs.get(ROOT_MODULE)!r}, expected {root_ver!r}")
    aikit = reqs.get(AIKIT_MODULE)
    if is_root:
        if not aikit:
            errs.append(f"{label}: missing {AIKIT_MODULE} require")
        elif PSEUDO_RE.search(aikit):
            errs.append(f"{label}: {AIKIT_MODULE} pin {aikit!r} is a pseudo-version")
    elif aikit:
        if PSEUDO_RE.search(aikit):
            errs.append(f"{label}: {AIKIT_MODULE} pin {aikit!r} is a pseudo-version")
        if expected_aikit and aikit != expected_aikit:
            errs.append(f"{label}: {AIKIT_MODULE} pin {aikit!r} disagrees with root {expected_aikit!r}")
    return errs


def check_gosum(text: str, version: str, zip_sum: str, mod_sum: str, label: str) -> List[str]:
    z_got, m_got = set(), set()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == ROOT_MODULE:
            if parts[1] == version:
                z_got.add(parts[2])
            elif parts[1] == f"{version}/go.mod":
                m_got.add(parts[2])
    errs = []
    if z_got != {zip_sum}:
        errs.append(f"{label}: {ROOT_MODULE} {version} hash is {sorted(z_got)}, expected [{zip_sum}]")
    if m_got != {mod_sum}:
        errs.append(f"{label}: {ROOT_MODULE} {version}/go.mod hash is {sorted(m_got)}, expected [{mod_sum}]")
    return errs


def go_env(*, modcache: Path, goproxy: str, gosumdb: Optional[str] = None, gonosumdb: Optional[str] = None, goos: str = "", goarch: str = "", gomaxprocs: Optional[int] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["GOWORK"] = "off"
    env["GOTOOLCHAIN"] = "local"
    env["CGO_ENABLED"] = "0"
    env["GOMODCACHE"] = str(modcache)
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
    if gomaxprocs is not None:
        env["GOMAXPROCS"] = str(gomaxprocs)
    return env


def local_cache_proxy() -> Optional[str]:
    p = run(["go", "env", "GOMODCACHE"], check=False)
    if p.returncode != 0:
        return None
    dl = Path(p.stdout.strip()) / "cache" / "download"
    return dl.resolve().as_uri() if dl.is_dir() else None


def relabel_canonical_zip(src_zip: Path, src_version: str, src_mod: Path, src_info: Optional[Path], dst_version: str, proxy_dir: Path) -> None:
    """Copy a Go-produced canonical root zip into a file:// proxy, relabeling only its version prefix."""
    src_prefix = f"{ROOT_MODULE}@{src_version}/"
    dst_prefix = f"{ROOT_MODULE}@{dst_version}/"
    vdir = proxy_dir / ROOT_MODULE / "@v"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "list").write_text(f"{dst_version}\n", encoding="utf-8")

    info = json.loads(src_info.read_text(encoding="utf-8")) if src_info and src_info.exists() else {}
    info["Version"] = dst_version
    info.setdefault("Time", "2026-01-01T00:00:00Z")
    (vdir / f"{dst_version}.info").write_text(json.dumps(info, separators=(",", ":")) + "\n", encoding="utf-8")
    shutil.copyfile(src_mod, vdir / f"{dst_version}.mod")

    with zipfile.ZipFile(src_zip, "r") as zin, zipfile.ZipFile(vdir / f"{dst_version}.zip", "w", zipfile.ZIP_DEFLATED) as zout:
        for zi in zin.infolist():
            if not zi.filename.startswith(src_prefix):
                raise PrepError(f"unexpected zip entry {zi.filename!r} (expected prefix {src_prefix!r})")
            nzi = zipfile.ZipInfo(dst_prefix + zi.filename[len(src_prefix):], date_time=zi.date_time)
            nzi.compress_type = zipfile.ZIP_DEFLATED
            nzi.external_attr = zi.external_attr
            zout.writestr(nzi, zin.read(zi))


def stage_root_from_go(
    *,
    base_sha: str,
    version: str,
    tmp: Path,
    fetch_proxy: str,
    gomaxprocs: Optional[int],
    source_zip: Optional[Path] = None,
    source_version: Optional[str] = None,
    source_mod: Optional[Path] = None,
) -> Tuple[Path, str, str]:
    """Fetch Go's canonical zip for base_sha (or use explicit source_zip), relabel to version, and verify with Go."""
    proxy_dir = tmp / "proxy"
    if source_zip is not None:
        if not source_version or source_mod is None:
            raise PrepError("--source-zip requires --source-version and --source-mod")
        relabel_canonical_zip(source_zip, source_version, source_mod, None, version, proxy_dir)
    else:
        fetch_mc = tmp / "fetch-mc"
        fetch_mc.mkdir()
        env = go_env(modcache=fetch_mc, goproxy=fetch_proxy, gomaxprocs=gomaxprocs)
        dl = json.loads(run(["go", "mod", "download", "-json", f"{ROOT_MODULE}@{base_sha}"], cwd=tmp, env=env).stdout)
        if dl.get("Error"):
            raise PrepError(f"go mod download {ROOT_MODULE}@{base_sha}: {dl['Error']}")
        relabel_canonical_zip(Path(dl["Zip"]), dl["Version"], Path(dl["GoMod"]), Path(dl["Info"]), version, proxy_dir)

    # Let Go itself validate the relabeled zip and compute the authoritative h1 sums.
    verify_mc = tmp / "verify-mc"
    verify_mc.mkdir()
    venv = go_env(
        modcache=verify_mc,
        goproxy=proxy_dir.resolve().as_uri(),
        gonosumdb=ROOT_MODULE,
        gomaxprocs=gomaxprocs,
    )
    staged = json.loads(run(["go", "mod", "download", "-json", f"{ROOT_MODULE}@{version}"], cwd=tmp, env=venv).stdout)
    if staged.get("Error"):
        raise PrepError(f"staged zip failed go mod download: {staged['Error']}")
    return proxy_dir, staged["Sum"], staged["GoModSum"]


def nested_module_dirs(repo: Path, rev: str) -> List[str]:
    files = run(["git", "ls-tree", "-r", "--name-only", rev], cwd=repo).stdout.splitlines()
    return [f[:-len("/go.mod")] for f in files if f.endswith("/go.mod")]


def root_module_diff(repo: Path, base_sha: str, head_sha: str) -> List[str]:
    """Return any changed files between base_sha and head_sha that belong to the root module."""
    changed = [p for p in run(["git", "diff", "--name-only", base_sha, head_sha], cwd=repo).stdout.splitlines() if p]
    if not changed:
        return []
    nested = set(nested_module_dirs(repo, base_sha)) & set(nested_module_dirs(repo, head_sha))
    return [p for p in changed if not any(p == d or p.startswith(d + "/") for d in nested)]


def dirty_paths(repo: Path) -> List[str]:
    out = run(["git", "status", "--porcelain"], cwd=repo).stdout.splitlines()
    return [line[3:].split(" -> ")[-1] for line in out if len(line) > 3]


def cmd_prep(args: argparse.Namespace) -> int:
    repo = git_root(args.repo)
    validate_semver(args.version)
    selected = parse_modules(args.modules)
    omitted = [m for m in SUBMODULES if m not in selected]
    base_sha = git_commit(repo, args.base)
    head_sha = git_commit(repo, "HEAD")

    # 1. Preconditions: no existing tags, no root drift since base, no dirty files outside selected go.{mod,sum}.
    for t in [args.version] + [f"{m}/{args.version}" for m in selected]:
        if run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{t}"], cwd=repo, check=False).returncode == 0:
            raise PrepError(f"tag {t} already exists locally")
    drift = root_module_diff(repo, base_sha, head_sha)
    if drift:
        raise PrepError(f"root-module files changed between --base ({base_sha[:12]}) and HEAD ({head_sha[:12]}): {', '.join(drift[:5])}")
    allowed = {f"{m}/{fn}" for m in selected for fn in ("go.mod", "go.sum")}
    bad_dirty = [p for p in dirty_paths(repo) if p not in allowed]
    if bad_dirty:
        raise PrepError(f"uncommitted changes outside selected submodule go.mod/go.sum: {', '.join(bad_dirty[:5])}")

    root_gomod = run(["git", "show", f"{base_sha}:go.mod"], cwd=repo).stdout
    _, root_reqs, _ = parse_gomod(root_gomod)
    root_aikit = root_reqs.get(AIKIT_MODULE)
    errs = check_gomod(root_gomod, "go.mod", expected_mod=ROOT_MODULE, is_root=True)
    if errs:
        raise PrepError("\n".join(errs))

    tmp = Path(tempfile.mkdtemp(prefix="goinfer-release-prep-"))
    try:
        proxy_dir, zip_sum, mod_sum = stage_root_from_go(
            base_sha=base_sha,
            version=args.version,
            tmp=tmp,
            fetch_proxy=args.fetch_proxy,
            gomaxprocs=args.gomaxprocs,
            source_zip=Path(args.source_zip).resolve() if args.source_zip else None,
            source_version=args.source_version,
            source_mod=Path(args.source_mod).resolve() if args.source_mod else None,
        )
        print(f"staged {ROOT_MODULE}@{args.version} from base {base_sha[:12]}:\n  Sum      = {zip_sum}\n  GoModSum = {mod_sum}")

        proxy_parts = [proxy_dir.resolve().as_uri()]
        if not args.no_local_cache:
            lc = local_cache_proxy()
            if lc:
                proxy_parts.append(lc)
        if not args.offline_deps:
            proxy_parts.extend(["https://proxy.golang.org", "direct"])
        goproxy = ",".join(proxy_parts)

        work_mc = tmp / "work-mc"
        work_mc.mkdir()
        for m in selected:
            spec = SUBMODULES[m]
            mdir = repo / spec.rel_dir
            env = go_env(
                modcache=work_mc,
                goproxy=goproxy,
                gosumdb="off" if args.offline_deps else None,
                gonosumdb=None if args.offline_deps else ROOT_MODULE,
                goos=spec.goos,
                goarch=spec.goarch,
                gomaxprocs=args.gomaxprocs,
            )
            run(["go", "mod", "edit", f"-require={ROOT_MODULE}@{args.version}"], cwd=mdir, env=env)
            run(["go", "mod", "tidy"], cwd=mdir, env=env)
            run(["go", "mod", "verify"], cwd=mdir, env=env)

            errs.extend(check_gomod((mdir / "go.mod").read_text(encoding="utf-8"), f"{m}/go.mod", expected_mod=spec.module_path, root_ver=args.version, expected_aikit=root_aikit))
            errs.extend(check_gosum((mdir / "go.sum").read_text(encoding="utf-8"), args.version, zip_sum, mod_sum, f"{m}/go.sum"))
            if not args.skip_build:
                targs = ["-tags", spec.build_tags] if spec.build_tags else []
                run(["go", "vet", *targs, "./..."], cwd=mdir, env=env)
                run(["go", "build", *targs, "./..."], cwd=mdir, env=env)
            print(f"  [PASS] {m}: pinned {args.version} + go mod verify" + (" (--skip-build)" if args.skip_build else " + standalone vet/build"))
            print(f"         {spec.note}")

        if omitted:
            print("omitted backend(s) (untouched; document omission in CHANGELOG and GitHub Release):")
            for om in omitted:
                om_mod = repo / om / "go.mod"
                cur = parse_gomod(om_mod.read_text(encoding="utf-8"))[1].get(ROOT_MODULE, "(none)") if om_mod.exists() else "(missing)"
                print(f"  - {om} (current root pin: {cur})")

        if errs:
            raise PrepError("\n".join(errs))

        files = " ".join(f"{m}/go.mod {m}/go.sum" for m in selected)
        tags = " ".join([args.version] + [f"{m}/{args.version}" for m in selected])
        print(
            "\nNext (no commit or tags created):\n"
            + (f"  git add -- {files}\n  git commit -m \"release: {args.version} (<summary>)\"\n" if selected else "")
            + f"  python3 scripts/release_prep.py check --version {args.version} --base {base_sha[:12]} --modules {','.join(selected) or 'none'}\n"
            + "".join(f"  git tag {t}\n" for t in [args.version] + [f"{m}/{args.version}" for m in selected])
            + f"  python3 scripts/release_prep.py check --version {args.version} --base {base_sha[:12]} --modules {','.join(selected) or 'none'} --check-tags\n"
            + f"  git push origin main {tags}"
        )
        print("VERDICT: PASS")
        return 0
    finally:
        rmtree_ro(tmp)


def cmd_check(args: argparse.Namespace) -> int:
    repo = git_root(args.repo)
    validate_semver(args.version)
    selected = parse_modules(args.modules)
    omitted = [m for m in SUBMODULES if m not in selected]
    base_sha = git_commit(repo, args.base)
    rev_sha = git_commit(repo, args.rev)
    errs: List[str] = []

    if args.rev == "HEAD" and not args.allow_dirty:
        dirty = dirty_paths(repo)
        if dirty:
            errs.append(f"working tree is not clean: {', '.join(dirty[:5])}")

    drift = root_module_diff(repo, base_sha, rev_sha)
    if drift:
        errs.append(f"root-module files changed between --base ({base_sha[:12]}) and {rev_sha[:12]}: {', '.join(drift[:5])}")

    root_gomod = run(["git", "show", f"{rev_sha}:go.mod"], cwd=repo).stdout
    _, root_reqs, _ = parse_gomod(root_gomod)
    root_aikit = root_reqs.get(AIKIT_MODULE)
    errs.extend(check_gomod(root_gomod, "go.mod", expected_mod=ROOT_MODULE, is_root=True))

    zip_sum = mod_sum = None
    if args.verify_sum or args.source_zip:
        tmp = Path(tempfile.mkdtemp(prefix="goinfer-release-check-"))
        try:
            _, zip_sum, mod_sum = stage_root_from_go(
                base_sha=base_sha,
                version=args.version,
                tmp=tmp,
                fetch_proxy=args.fetch_proxy,
                gomaxprocs=args.gomaxprocs,
                source_zip=Path(args.source_zip).resolve() if args.source_zip else None,
                source_version=args.source_version,
                source_mod=Path(args.source_mod).resolve() if args.source_mod else None,
            )
        finally:
            rmtree_ro(tmp)

    for m in selected:
        spec = SUBMODULES[m]
        gm = run(["git", "show", f"{rev_sha}:{m}/go.mod"], cwd=repo).stdout
        gs = run(["git", "show", f"{rev_sha}:{m}/go.sum"], cwd=repo).stdout
        errs.extend(check_gomod(gm, f"{m}/go.mod@{rev_sha[:12]}", expected_mod=spec.module_path, root_ver=args.version, expected_aikit=root_aikit))
        if zip_sum and mod_sum:
            errs.extend(check_gosum(gs, args.version, zip_sum, mod_sum, f"{m}/go.sum@{rev_sha[:12]}"))
        elif f"{ROOT_MODULE} {args.version} h1:" not in gs or f"{ROOT_MODULE} {args.version}/go.mod h1:" not in gs:
            errs.append(f"{m}/go.sum@{rev_sha[:12]}: missing h1 entries for {ROOT_MODULE} {args.version}")

    if args.check_tags:
        for t in [args.version] + [f"{m}/{args.version}" for m in selected]:
            p = run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{t}^{{commit}}"], cwd=repo, check=False)
            if p.returncode != 0:
                errs.append(f"missing local tag {t}")
            elif p.stdout.strip() != rev_sha:
                errs.append(f"tag {t} points to {p.stdout.strip()[:12]}, expected single release commit {rev_sha[:12]}")
            else:
                print(f"  [PASS] tag {t} -> {rev_sha[:12]}")
        for om in omitted:
            t = f"{om}/{args.version}"
            if run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{t}"], cwd=repo, check=False).returncode == 0:
                errs.append(f"unselected backend tag {t} exists locally")

    if errs:
        raise PrepError("\n".join(errs))
    print(f"verified single-commit release {args.version} at {rev_sha[:12]} (base {base_sha[:12]}, selected={selected or ['none']}, omitted={omitted})")
    print("VERDICT: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage Go's canonical root zip for single-commit backend releases.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--version", required=True, help="Candidate semver tag (e.g. v0.6.2).")
        sp.add_argument("--base", default="HEAD", help="Already-pushed base commit whose root tree is released (default: HEAD).")
        sp.add_argument("--modules", default=",".join(DEFAULT_MODULES), help="Comma-separated backend modules or 'none'.")
        sp.add_argument("--repo", default=None, help="Git repository path (default: current repo).")
        sp.add_argument("--fetch-proxy", default="https://proxy.golang.org,direct", help="GOPROXY for fetching base canonical zip.")
        sp.add_argument("--gomaxprocs", type=int, default=None, help="Optional GOMAXPROCS cap.")
        sp.add_argument("--source-zip", default=None, help=argparse.SUPPRESS)
        sp.add_argument("--source-version", default=None, help=argparse.SUPPRESS)
        sp.add_argument("--source-mod", default=None, help=argparse.SUPPRESS)

    pr = sub.add_parser("prep", help="Relabel Go's canonical base zip to candidate version and pin selected submodules.")
    add_common(pr)
    pr.add_argument("--skip-build", action="store_true", help="Skip standalone go vet/go build (pin + verify only).")
    pr.add_argument("--offline-deps", action="store_true", help="Resolve dependencies from staged proxy + local cache only.")
    pr.add_argument("--no-local-cache", action="store_true", help="Do not chain local GOMODCACHE/cache/download.")
    pr.set_defaults(func=cmd_prep)

    ck = sub.add_parser("check", help="Verify zero root drift since --base, submodule pins, and same-commit tags.")
    add_common(ck)
    ck.add_argument("--rev", default="HEAD", help="Final release commit to check (default: HEAD).")
    ck.add_argument("--check-tags", action="store_true", help="Verify root and selected backend tags all resolve to --rev.")
    ck.add_argument("--verify-sum", action="store_true", help="Re-fetch base zip via Go and confirm exact go.sum h1 hashes.")
    ck.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree when checking HEAD.")
    ck.set_defaults(func=cmd_check)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("VERDICT: FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
