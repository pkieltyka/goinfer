#!/usr/bin/env python3
"""Lean hermetic tests for scripts/release_prep.py (stdlib unittest, offline)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_prep as rp  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True).stdout.strip()


def _write_zip(path: Path, prefix: str, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in sorted(files.items()):
            zf.writestr(f"{prefix}/{name}", content.encode("utf-8"))


class TestReleasePrep(unittest.TestCase):
    def test_semver_and_gomod_invariants(self):
        rp.validate_semver("v0.6.2")
        rp.validate_semver("v0.6.2-rc.1")
        for bad in ("0.6.2", "v0.6", "v0.6.2+meta", "v2.0.0"):
            with self.assertRaises(rp.PrepError):
                rp.validate_semver(bad)

        good_root = f"module {rp.ROOT_MODULE}\n\ngo 1.27.0\n\nrequire {rp.AIKIT_MODULE} v1.79.0\n"
        self.assertEqual(rp.check_gomod(good_root, "go.mod", expected_mod=rp.ROOT_MODULE, is_root=True), [])

        bad_rep = f"module {rp.ROOT_MODULE}/backend/cuda\n\ngo 1.27.0\n\nrequire {rp.ROOT_MODULE} v0.6.2\nreplace {rp.ROOT_MODULE} => ../..\n"
        self.assertTrue(any("replace" in e for e in rp.check_gomod(bad_rep, "cuda", expected_mod=f"{rp.ROOT_MODULE}/backend/cuda", root_ver="v0.6.2")))

        bad_cycle = f"module {rp.ROOT_MODULE}\n\ngo 1.27.0\n\nrequire (\n\t{rp.AIKIT_MODULE} v1.79.0\n\t{rp.ROOT_MODULE}/backend/vulkan v0.6.2\n)\n"
        self.assertTrue(any("must not require" in e for e in rp.check_gomod(bad_cycle, "go.mod", expected_mod=rp.ROOT_MODULE, is_root=True)))

        bad_aikit = f"module {rp.ROOT_MODULE}/backend/vulkan\n\ngo 1.27.0\n\nrequire (\n\t{rp.AIKIT_MODULE} v1.78.0\n\t{rp.ROOT_MODULE} v0.6.2\n)\n"
        self.assertTrue(any("disagrees" in e for e in rp.check_gomod(bad_aikit, "vk", expected_mod=f"{rp.ROOT_MODULE}/backend/vulkan", root_ver="v0.6.2", expected_aikit="v1.79.0")))

        # GitHub Actions skips tag push events when >3 tags are pushed together.
        self.assertEqual(
            rp.format_push_commands(["v0.6.2"]),
            ["git push --atomic origin HEAD:main v0.6.2"],
        )
        self.assertEqual(
            rp.format_push_commands(["v0.6.2", "backend/cuda/v0.6.2", "backend/vulkan/v0.6.2"]),
            ["git push --atomic origin HEAD:main v0.6.2 backend/cuda/v0.6.2 backend/vulkan/v0.6.2"],
        )
        self.assertEqual(
            rp.format_push_commands(["v0.6.2", "backend/cuda/v0.6.2", "backend/vulkan/v0.6.2", "backend/metal/v0.6.2"]),
            [
                "git push --atomic origin HEAD:main v0.6.2 backend/cuda/v0.6.2 backend/vulkan/v0.6.2",
                "git push --atomic origin backend/metal/v0.6.2",
            ],
        )

        # Online go_env forces GOSUMDB=sum.golang.org even when caller has GOSUMDB=off.
        orig_sumdb = os.environ.get("GOSUMDB")
        os.environ["GOSUMDB"] = "off"
        try:
            env = rp.go_env(modcache=Path("/tmp/mc"), goproxy="https://proxy.golang.org,direct", gonosumdb=rp.ROOT_MODULE)
            self.assertEqual(env["GOSUMDB"], "sum.golang.org")
            self.assertEqual(env["GONOSUMDB"], rp.ROOT_MODULE)
            self.assertEqual(env["GOPRIVATE"], "")
        finally:
            if orig_sumdb is None:
                os.environ.pop("GOSUMDB", None)
            else:
                os.environ["GOSUMDB"] = orig_sumdb

        # Production CLI must not accept hidden archive-injection flags.
        with self.assertRaises(SystemExit):
            rp.build_parser().parse_args(["prep", "--version", "v0.6.2", "--source-zip", "/tmp/evil.zip"])

    def test_single_commit_prep_and_guards(self):
        tmp = Path(tempfile.mkdtemp(prefix="goinfer-rp-lean-"))
        try:
            # Stage a tiny fake aikit v1.79.0 in a local download cache so prep runs 100% offline.
            fake_dl = tmp / "fake-cache" / "cache" / "download"
            aikit_vdir = fake_dl / rp.AIKIT_MODULE / "@v"
            aikit_vdir.mkdir(parents=True)
            (aikit_vdir / "list").write_text("v1.79.0\n", encoding="utf-8")
            (aikit_vdir / "v1.79.0.info").write_text('{"Version":"v1.79.0","Time":"2026-11-22T00:00:00Z"}\n', encoding="utf-8")
            aikit_mod = f"module {rp.AIKIT_MODULE}\n\ngo 1.27.0\n"
            (aikit_vdir / "v1.79.0.mod").write_text(aikit_mod, encoding="utf-8")
            _write_zip(aikit_vdir / "v1.79.0.zip", f"{rp.AIKIT_MODULE}@v1.79.0", {"go.mod": aikit_mod, "aikit.go": 'package aikit\nconst Ver = "v1.79.0"\n'})

            # Build synthetic 4-module git repo (root + cuda + vulkan + metal).
            repo = tmp / "repo"
            repo.mkdir()
            _git(repo, "init", "-b", "main")
            _git(repo, "config", "user.name", "Test")
            _git(repo, "config", "user.email", "t@example.com")
            root_mod = f"module {rp.ROOT_MODULE}\n\ngo 1.27.0\n\nrequire {rp.AIKIT_MODULE} v1.79.0\n"
            root_api = f'package embeddinggemma2\nimport "{rp.AIKIT_MODULE}"\nfunc Mark() string {{ return aikit.Ver }}\n'
            (repo / "go.mod").write_text(root_mod, encoding="utf-8")
            (repo / "embeddinggemma2").mkdir()
            (repo / "embeddinggemma2" / "api.go").write_text(root_api, encoding="utf-8")
            for rel, tags, body_hdr in [
                ("backend/cuda", "cuda", "//go:build cuda\n\npackage cuda"),
                ("backend/vulkan", "", "package vulkan"),
                ("backend/metal", "metal", "//go:build darwin\n\npackage metal"),
            ]:
                d = repo / rel
                d.mkdir(parents=True)
                (d / "go.mod").write_text(f"module {rp.ROOT_MODULE}/{rel}\n\ngo 1.27.0\n\nrequire (\n\t{rp.AIKIT_MODULE} v1.79.0\n\t{rp.ROOT_MODULE} v0.6.1\n)\n", encoding="utf-8")
                (d / "mod.go").write_text(f'{body_hdr}\nimport "{rp.ROOT_MODULE}/embeddinggemma2"\nfunc M() string {{ return embeddinggemma2.Mark() }}\n', encoding="utf-8")
            _git(repo, "add", "go.mod", "embeddinggemma2/api.go", "backend/cuda", "backend/vulkan", "backend/metal")
            _git(repo, "commit", "-m", "base root release content")
            base_sha = _git(repo, "rev-parse", "HEAD")

            # Canonical zip for base_sha (simulating `go mod download -json ...@<base_sha>`).
            src_ver = f"v0.6.2-0.20261122000000-{base_sha[:12]}"
            src_zip = tmp / "base.zip"
            src_mod = tmp / "base.mod"
            src_mod.write_text(root_mod, encoding="utf-8")
            _write_zip(src_zip, f"{rp.ROOT_MODULE}@{src_ver}", {"go.mod": root_mod, "embeddinggemma2/api.go": root_api})

            orig_cache = rp.local_cache_proxy
            orig_fetch = rp.fetch_base_module
            rp.local_cache_proxy = lambda: fake_dl.resolve().as_uri()
            rp.fetch_base_module = lambda bsha, fproxy, tdir, gmp: (src_zip, src_ver, src_mod, None)
            try:
                # 1. Prep CUDA + Vulkan (Metal omitted).
                self.assertEqual(rp.main([
                    "prep", "--repo", str(repo), "--version", "v0.6.2", "--base", base_sha,
                    "--modules", "backend/cuda,backend/vulkan", "--offline-deps", "--gomaxprocs", "1",
                ]), 0)
                self.assertIn(f"{rp.ROOT_MODULE} v0.6.1", (repo / "backend/metal/go.mod").read_text())

                # Commit submodule pins & tag root + CUDA + Vulkan on ONE commit.
                _git(repo, "add", "backend/cuda/go.mod", "backend/cuda/go.sum", "backend/vulkan/go.mod", "backend/vulkan/go.sum")
                _git(repo, "commit", "-m", "release: v0.6.2")
                for t in ("v0.6.2", "backend/cuda/v0.6.2", "backend/vulkan/v0.6.2"):
                    _git(repo, "tag", t)

                self.assertEqual(rp.main([
                    "check", "--repo", str(repo), "--version", "v0.6.2", "--base", base_sha,
                    "--modules", "backend/cuda,backend/vulkan", "--check-tags", "--verify-sum", "--gomaxprocs", "1",
                ]), 0)

                # Negative control: unselected Metal tag rejected.
                _git(repo, "tag", "backend/metal/v0.6.2")
                self.assertEqual(rp.main([
                    "check", "--repo", str(repo), "--version", "v0.6.2", "--base", base_sha,
                    "--modules", "backend/cuda,backend/vulkan", "--check-tags",
                ]), 1)
                _git(repo, "tag", "-d", "backend/metal/v0.6.2")

                # 2. Metal included: prep + cross-build darwin/arm64 + verify same-commit tags.
                self.assertEqual(rp.main([
                    "prep", "--repo", str(repo), "--version", "v0.6.3", "--base", base_sha,
                    "--modules", "backend/cuda,backend/vulkan,backend/metal", "--offline-deps", "--gomaxprocs", "1",
                ]), 0)
                _git(repo, "add", "backend/cuda/go.mod", "backend/cuda/go.sum", "backend/vulkan/go.mod", "backend/vulkan/go.sum", "backend/metal/go.mod", "backend/metal/go.sum")
                _git(repo, "commit", "-m", "release: v0.6.3")
                for t in ("v0.6.3", "backend/cuda/v0.6.3", "backend/vulkan/v0.6.3", "backend/metal/v0.6.3"):
                    _git(repo, "tag", t)
                self.assertEqual(rp.main([
                    "check", "--repo", str(repo), "--version", "v0.6.3", "--base", base_sha,
                    "--modules", "backend/cuda,backend/vulkan,backend/metal", "--check-tags", "--verify-sum", "--gomaxprocs", "1",
                ]), 0)

                # Negative control: tagging on a side branch while local main stays behind fails --check-tags.
                rel_sha = _git(repo, "rev-parse", "HEAD")
                _git(repo, "checkout", "-b", "side-release")
                _git(repo, "branch", "-f", "main", base_sha)
                self.assertEqual(rp.main([
                    "check", "--repo", str(repo), "--version", "v0.6.3", "--base", base_sha,
                    "--modules", "backend/cuda,backend/vulkan,backend/metal", "--check-tags",
                ]), 1)
                _git(repo, "checkout", "-B", "main", rel_sha)

                # Negative control: moving an unchanged root file into an existing backend module
                # (R100 rename) removes it from the root zip and MUST be caught by root_module_diff + check.
                _git(repo, "mv", "embeddinggemma2/api.go", "backend/cuda/moved_api.go")
                _git(repo, "commit", "-m", "move root file into backend/cuda")
                self.assertIn("embeddinggemma2/api.go", rp.root_module_diff(repo, base_sha, "HEAD"))
                self.assertEqual(rp.main([
                    "check", "--repo", str(repo), "--version", "v0.6.3", "--base", base_sha,
                    "--modules", "backend/cuda,backend/vulkan,backend/metal",
                ]), 1)

                # Negative control: split two-commit tag rejected by --check-tags.
                _git(repo, "tag", "-f", "backend/cuda/v0.6.3", "HEAD")
                self.assertEqual(rp.main([
                    "check", "--repo", str(repo), "--version", "v0.6.3", "--base", base_sha,
                    "--modules", "backend/cuda,backend/vulkan,backend/metal", "--rev", "HEAD~1", "--check-tags",
                ]), 1)
            finally:
                rp.local_cache_proxy = orig_cache
                rp.fetch_base_module = orig_fetch
        finally:
            rp.rmtree_ro(tmp)


if __name__ == "__main__":
    unittest.main()
