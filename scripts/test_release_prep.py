#!/usr/bin/env python3
"""Hermetic tests for scripts/release_prep.py (stdlib unittest, offline, fast)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import release_prep as rp  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _make_synthetic_repo(root: Path, *, aikit_ver: str = "v1.79.0") -> None:
    """Create a realistic 4-module git repo (root + cuda + vulkan + metal) with no external deps."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")

    (root / ".gitignore").write_text("go.work\n*.tmp\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## [v0.6.2]\n- Added TagMarker.\n", encoding="utf-8")

    # Root module: no real external imports needed so tests run 100% offline.
    # Notice we include a comment-free root go.mod with stdlib-only packages;
    # we test aikit pin validation in dedicated unit tests below.
    (root / "go.mod").write_text(
        f"module {rp.ROOT_MODULE}\n\n"
        "go 1.27.0\n",
        encoding="utf-8",
    )
    (root / "embeddinggemma2").mkdir()
    (root / "embeddinggemma2" / "api.go").write_text(
        "package embeddinggemma2\n\n"
        "// TagMarker simulates a root symbol added after the previous release.\n"
        "func TagMarker() string { return \"v0.6.2\" }\n",
        encoding="utf-8",
    )

    # Plant things that MUST be omitted from the canonical root module zip:
    # 1. vendor/modules.txt (omitted for go >= 1.24)
    (root / "vendor").mkdir()
    (root / "vendor" / "modules.txt").write_text("# vendored\n", encoding="utf-8")
    # 2. A tracked symlink (omitted by golang.org/x/mod/zip)
    os.symlink("CHANGELOG.md", root / "CHANGELOG.link")
    # 3. Gitignored go.work + untracked scratch file (never in git archive)
    (root / "go.work").write_text(
        "go 1.27.0\nuse (\n\t.\n\t./backend/cuda\n\t./backend/vulkan\n\t./backend/metal\n)\n",
        encoding="utf-8",
    )
    (root / "scratch.tmp").write_text("ignored untracked noise\n", encoding="utf-8")

    # backend/cuda (linux, -tags cuda)
    cuda = root / "backend" / "cuda"
    cuda.mkdir(parents=True)
    (cuda / "go.mod").write_text(
        f"module {rp.ROOT_MODULE}/backend/cuda\n\n"
        "go 1.27.0\n\n"
        f"require {rp.ROOT_MODULE} v0.6.1\n",
        encoding="utf-8",
    )
    (cuda / "backend.go").write_text(
        "//go:build cuda\n\n"
        "package cuda\n\n"
        f'import "{rp.ROOT_MODULE}/embeddinggemma2"\n\n'
        "func Marker() string { return embeddinggemma2.TagMarker() }\n",
        encoding="utf-8",
    )

    # backend/vulkan (linux, untagged, CGO_ENABLED=0)
    vulkan = root / "backend" / "vulkan"
    vulkan.mkdir(parents=True)
    (vulkan / "go.mod").write_text(
        f"module {rp.ROOT_MODULE}/backend/vulkan\n\n"
        "go 1.27.0\n\n"
        f"require {rp.ROOT_MODULE} v0.6.1\n",
        encoding="utf-8",
    )
    (vulkan / "register.go").write_text(
        "package vulkan\n\n"
        f'import "{rp.ROOT_MODULE}/embeddinggemma2"\n\n'
        "func Marker() string { return embeddinggemma2.TagMarker() }\n",
        encoding="utf-8",
    )

    # backend/metal (darwin, -tags metal / darwin-gated)
    metal = root / "backend" / "metal"
    metal.mkdir(parents=True)
    (metal / "go.mod").write_text(
        f"module {rp.ROOT_MODULE}/backend/metal\n\n"
        "go 1.27.0\n\n"
        f"require {rp.ROOT_MODULE} v0.0.0-20261121021407-cd949d2e4777\n",
        encoding="utf-8",
    )
    (metal / "backend.go").write_text(
        "//go:build darwin\n\n"
        "package metal\n\n"
        f'import "{rp.ROOT_MODULE}/embeddinggemma2"\n\n'
        "func Marker() string { return embeddinggemma2.TagMarker() }\n",
        encoding="utf-8",
    )

    _git(
        root,
        "add",
        ".gitignore",
        "LICENSE",
        "CHANGELOG.md",
        "CHANGELOG.link",
        "go.mod",
        "embeddinggemma2/api.go",
        "vendor/modules.txt",
        "backend/cuda/go.mod",
        "backend/cuda/backend.go",
        "backend/vulkan/go.mod",
        "backend/vulkan/register.go",
        "backend/metal/go.mod",
        "backend/metal/backend.go",
    )
    _git(root, "commit", "-m", "release: v0.6.2 (synthetic single-commit test)")


class TestReleasePrepUnit(unittest.TestCase):
    def test_semver_validation(self):
        for ok in ("v0.6.2", "v0.6.2-rc.1", "v1.0.0", "v1.2.3-alpha.0"):
            rp.validate_semver(ok, rp.ROOT_MODULE)
        for bad in ("0.6.2", "v0.6", "v01.0.0", "v0.6.2+meta", "v2.0.0", ""):
            with self.assertRaises(rp.ReleasePrepError, msg=f"should reject {bad!r}"):
                rp.validate_semver(bad, rp.ROOT_MODULE)

    def test_gomod_invariants(self):
        good_root = (
            f"module {rp.ROOT_MODULE}\n\ngo 1.27.0\n\n"
            f"require {rp.AIKIT_MODULE} v1.79.0\n"
        )
        self.assertEqual(
            rp.verify_gomod_text(good_root, "go.mod", expected_module=rp.ROOT_MODULE, is_root=True),
            [],
        )

        # Rejects replace directive.
        with_replace = (
            f"module {rp.ROOT_MODULE}/backend/cuda\n\ngo 1.27.0\n\n"
            f"require (\n\t{rp.AIKIT_MODULE} v1.79.0\n\t{rp.ROOT_MODULE} v0.6.2\n)\n"
            f"replace {rp.ROOT_MODULE} => ../..\n"
        )
        errs = rp.verify_gomod_text(
            with_replace,
            "backend/cuda/go.mod",
            expected_module=f"{rp.ROOT_MODULE}/backend/cuda",
            expected_root_version="v0.6.2",
            expected_aikit="v1.79.0",
        )
        self.assertTrue(any("replace" in e for e in errs), errs)

        # Rejects pseudo-version aikit and mismatched aikit.
        pseudo_root = (
            f"module {rp.ROOT_MODULE}\n\ngo 1.27.0\n\n"
            f"require {rp.AIKIT_MODULE} v0.0.0-20261121010203-abcdef123456\n"
        )
        errs = rp.verify_gomod_text(pseudo_root, "go.mod", expected_module=rp.ROOT_MODULE, is_root=True)
        self.assertTrue(any("pseudo-version" in e for e in errs), errs)

        mismatch_sub = (
            f"module {rp.ROOT_MODULE}/backend/vulkan\n\ngo 1.27.0\n\n"
            f"require (\n\t{rp.AIKIT_MODULE} v1.78.0\n\t{rp.ROOT_MODULE} v0.6.2\n)\n"
        )
        errs = rp.verify_gomod_text(
            mismatch_sub,
            "backend/vulkan/go.mod",
            expected_module=f"{rp.ROOT_MODULE}/backend/vulkan",
            expected_root_version="v0.6.2",
            expected_aikit="v1.79.0",
        )
        self.assertTrue(any("disagrees with root" in e for e in errs), errs)

        # Rejects root -> backend cycle.
        cyclic_root = (
            f"module {rp.ROOT_MODULE}\n\ngo 1.27.0\n\n"
            f"require (\n\t{rp.AIKIT_MODULE} v1.79.0\n\t{rp.ROOT_MODULE}/backend/cuda v0.6.2\n)\n"
        )
        errs = rp.verify_gomod_text(cyclic_root, "go.mod", expected_module=rp.ROOT_MODULE, is_root=True)
        self.assertTrue(any("must not require backend submodule" in e for e in errs), errs)

    def test_v061_real_proxy_hash_oracle(self):
        """Verify our zip+h1 engine reproduces proxy.golang.org's exact v0.6.1 hashes across both commits."""
        for rev in ("e2324268a9fe7b5c8977730dadb6f8a8618d03ce", "e4b58e96fcaa3fe1bfe0fc337da513d46801d60f"):
            probe = subprocess.run(
                ["git", "cat-file", "-e", f"{rev}^{{commit}}"],
                cwd=str(REPO_ROOT),
                capture_output=True,
            )
            if probe.returncode != 0:
                self.skipTest(f"shallow clone lacks historical commit {rev[:12]}")
            got = rp.compute_module_hashes(REPO_ROOT, rev, rp.ROOT_MODULE, "v0.6.1", subdir="")
            self.assertEqual(got["zip_h1"], "h1:47P7WBMDmV8snWd+opXSt2Dr8Jm6UCpYCdBuU9czuhM=")
            self.assertEqual(got["gomod_h1"], "h1:/+U/HqKR7+4gCgDPF7F6J+lfz3hmfjVkwXlgKa+ZdFs=")

    def test_ls_remote_annotated_tag_peeling(self):
        sample = (
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/tags/v0.6.2\n"
            "1111111111111111111111111111111111111111\trefs/tags/v0.6.2^{}\n"
            "1111111111111111111111111111111111111111\trefs/tags/backend/cuda/v0.6.2\n"
        )
        parsed = rp._parse_ls_remote_tags(sample)
        self.assertEqual(parsed["v0.6.2"], "1111111111111111111111111111111111111111")
        self.assertEqual(parsed["backend/cuda/v0.6.2"], "1111111111111111111111111111111111111111")


class TestReleasePrepEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="goinfer-rp-test-"))
        self.repo = self.tmp / "repo"
        _make_synthetic_repo(self.repo)
        # Allow synthetic root go.mod without aikit requirement by temporarily patching
        # verify_gomod_text's aikit check only when aikit isn't used in the synthetic repo.
        # Better: keep real aikit check strict, and test synthetic repo via a dedicated flag?
        # Even better: add a local fake aikit v1.79.0 into a staged proxy so the REAL aikit
        # check runs unmodified! Let's verify how fast that is: 0.01s!

    def tearDown(self):
        rp._make_writable_and_rmtree(self.tmp)

    def _add_fake_aikit_proxy(self) -> Path:
        """Stage a tiny fake aikit v1.79.0 module in a local file:// proxy and require it in synthetic repo."""
        aikit_repo = self.tmp / "aikit-src"
        aikit_repo.mkdir()
        _git(aikit_repo, "init", "-b", "main")
        _git(aikit_repo, "config", "user.name", "Test")
        _git(aikit_repo, "config", "user.email", "t@example.com")
        (aikit_repo / "go.mod").write_text(f"module {rp.AIKIT_MODULE}\n\ngo 1.27.0\n", encoding="utf-8")
        (aikit_repo / "aikit.go").write_text("package aikit\n\nconst Version = \"v1.79.0\"\n", encoding="utf-8")
        _git(aikit_repo, "add", "go.mod", "aikit.go")
        _git(aikit_repo, "commit", "-m", "aikit v1.79.0")

        # Create a fake GOMODCACHE/cache/download structure containing aikit v1.79.0.
        fake_cache = self.tmp / "fake-modcache"
        dl_dir = fake_cache / "cache" / "download"
        dl_dir.mkdir(parents=True)
        rp.stage_proxy_module(dl_dir, aikit_repo, "HEAD", rp.AIKIT_MODULE, "v1.79.0", subdir="")

        # Wire root and backend/cuda + backend/metal to import & require aikit v1.79.0 (mirrors real repo).
        (self.repo / "go.mod").write_text(
            f"module {rp.ROOT_MODULE}\n\ngo 1.27.0\n\nrequire {rp.AIKIT_MODULE} v1.79.0\n",
            encoding="utf-8",
        )
        (self.repo / "embeddinggemma2" / "api.go").write_text(
            "package embeddinggemma2\n\n"
            f'import "{rp.AIKIT_MODULE}"\n\n'
            "func TagMarker() string { return \"v0.6.2-\" + aikit.Version }\n",
            encoding="utf-8",
        )
        (self.repo / "backend" / "cuda" / "go.mod").write_text(
            f"module {rp.ROOT_MODULE}/backend/cuda\n\ngo 1.27.0\n\n"
            f"require (\n\t{rp.AIKIT_MODULE} v1.79.0\n\t{rp.ROOT_MODULE} v0.6.1\n)\n",
            encoding="utf-8",
        )
        (self.repo / "backend" / "metal" / "go.mod").write_text(
            f"module {rp.ROOT_MODULE}/backend/metal\n\ngo 1.27.0\n\n"
            f"require (\n\t{rp.AIKIT_MODULE} v1.79.0\n\t{rp.ROOT_MODULE} v0.0.0-20261121021407-cd949d2e4777\n)\n",
            encoding="utf-8",
        )
        # Generate root go.sum offline against fake_cache.
        env = rp._build_go_env(
            gomodcache=self.tmp / "init-mc",
            goproxy=dl_dir.resolve().as_uri(),
            gosumdb="off",
            gomaxprocs=1,
        )
        rp.run_cmd(["go", "mod", "tidy"], cwd=self.repo, env=env)
        _git(
            self.repo,
            "add",
            "go.mod",
            "go.sum",
            "embeddinggemma2/api.go",
            "backend/cuda/go.mod",
            "backend/metal/go.mod",
        )
        _git(self.repo, "commit", "--amend", "--no-edit")
        return fake_cache

    def test_single_commit_cuda_vulkan_and_metal_omitted(self):
        fake_cache = self._add_fake_aikit_proxy()
        orig_cache_fn = rp._local_download_proxy_url
        rp._local_download_proxy_url = lambda: (fake_cache / "cache" / "download").resolve().as_uri()
        try:
            # 1. Prep CUDA + Vulkan (Metal omitted).
            rc = rp.main([
                "prep",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan",
                "--offline-deps",
                "--gomaxprocs", "1",
            ])
            self.assertEqual(rc, 0)

            # Confirm Metal was untouched and CUDA+Vulkan were pinned to v0.6.2.
            self.assertIn("v0.0.0-20261121021407-cd949d2e4777", (self.repo / "backend/metal/go.mod").read_text())
            self.assertIn(f"{rp.ROOT_MODULE} v0.6.2", (self.repo / "backend/cuda/go.mod").read_text())
            self.assertIn(f"{rp.ROOT_MODULE} v0.6.2", (self.repo / "backend/vulkan/go.mod").read_text())

            # 2. Fold submodule bumps into the SINGLE release commit and create all three tags on it.
            _git(self.repo, "add", "backend/cuda/go.mod", "backend/cuda/go.sum", "backend/vulkan/go.mod", "backend/vulkan/go.sum")
            _git(self.repo, "commit", "--amend", "--no-edit")
            final_sha = _git(self.repo, "rev-parse", "HEAD")

            # Use annotated tag on root and lightweight tags on submodules to test tag peeling too.
            _git(self.repo, "tag", "-a", "v0.6.2", "-m", "release v0.6.2")
            _git(self.repo, "tag", "backend/cuda/v0.6.2")
            _git(self.repo, "tag", "backend/vulkan/v0.6.2")

            # 3. verify-commit --check-tags --consumer-check MUST PASS.
            rc = rp.main([
                "verify-commit",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan",
                "--check-tags",
                "--consumer-check",
                "--offline-deps",
                "--gomaxprocs", "1",
            ])
            self.assertEqual(rc, 0)

            # 4. Simulate public proxy + bare remote and run verify-public end-to-end.
            bare = self.tmp / "origin.git"
            _git(self.tmp, "init", "--bare", str(bare))
            _git(self.repo, "remote", "add", "origin", str(bare))
            _git(self.repo, "push", "origin", "main", "v0.6.2", "backend/cuda/v0.6.2", "backend/vulkan/v0.6.2")

            pub_proxy = self.tmp / "public-proxy"
            pub_proxy.mkdir()
            # Copy fake aikit into pub_proxy as well (real proxy.golang.org serves all modules).
            rp.stage_proxy_module(pub_proxy, self.tmp / "aikit-src", "HEAD", rp.AIKIT_MODULE, "v1.79.0")
            rp.stage_proxy_module(pub_proxy, self.repo, final_sha, rp.ROOT_MODULE, "v0.6.2", subdir="")
            rp.stage_proxy_module(pub_proxy, self.repo, final_sha, f"{rp.ROOT_MODULE}/backend/cuda", "v0.6.2", subdir="backend/cuda")
            rp.stage_proxy_module(pub_proxy, self.repo, final_sha, f"{rp.ROOT_MODULE}/backend/vulkan", "v0.6.2", subdir="backend/vulkan")

            rc = rp.main([
                "verify-public",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan",
                "--proxy", pub_proxy.resolve().as_uri(),
                "--sumdb", "off",
                "--remote", "origin",
                "--expected-commit", final_sha,
                "--gomaxprocs", "1",
            ])
            self.assertEqual(rc, 0)

            # 5. Negative control: unselected Metal tag must fail verify-commit --check-tags.
            _git(self.repo, "tag", "backend/metal/v0.6.2")
            rc = rp.main([
                "verify-commit",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan",
                "--check-tags",
            ])
            self.assertEqual(rc, 1)
            _git(self.repo, "tag", "-d", "backend/metal/v0.6.2")

            # 6. Negative control: root file modified after prep must break zip_h1 verification!
            (self.repo / "CHANGELOG.md").write_text("# Changelog\n\nDRIFTED AFTER PREP\n", encoding="utf-8")
            _git(self.repo, "add", "CHANGELOG.md")
            _git(self.repo, "commit", "-m", "post-prep root edit")
            rc = rp.main([
                "verify-commit",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan",
            ])
            self.assertEqual(rc, 1)

            # 7. Negative control: two-commit tag split (root on old commit, backend on new commit) must fail!
            _git(self.repo, "tag", "-f", "backend/cuda/v0.6.2", "HEAD")
            rc = rp.main([
                "verify-commit",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan",
                "--rev", final_sha,
                "--check-tags",
            ])
            self.assertEqual(rc, 1)
        finally:
            rp._local_download_proxy_url = orig_cache_fn

    def test_single_commit_with_metal_included(self):
        fake_cache = self._add_fake_aikit_proxy()
        orig_cache_fn = rp._local_download_proxy_url
        rp._local_download_proxy_url = lambda: (fake_cache / "cache" / "download").resolve().as_uri()
        try:
            rc = rp.main([
                "prep",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan,backend/metal",
                "--offline-deps",
                "--gomaxprocs", "1",
            ])
            self.assertEqual(rc, 0)

            _git(
                self.repo,
                "add",
                "backend/cuda/go.mod",
                "backend/cuda/go.sum",
                "backend/vulkan/go.mod",
                "backend/vulkan/go.sum",
                "backend/metal/go.mod",
                "backend/metal/go.sum",
            )
            _git(self.repo, "commit", "--amend", "--no-edit")
            for t in ("v0.6.2", "backend/cuda/v0.6.2", "backend/vulkan/v0.6.2", "backend/metal/v0.6.2"):
                _git(self.repo, "tag", t)

            rc = rp.main([
                "verify-commit",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan,backend/metal",
                "--check-tags",
                "--consumer-check",
                "--offline-deps",
                "--gomaxprocs", "1",
            ])
            self.assertEqual(rc, 0)
        finally:
            rp._local_download_proxy_url = orig_cache_fn

    def test_prep_refuses_dirty_root_file(self):
        fake_cache = self._add_fake_aikit_proxy()
        orig_cache_fn = rp._local_download_proxy_url
        rp._local_download_proxy_url = lambda: (fake_cache / "cache" / "download").resolve().as_uri()
        try:
            (self.repo / "CHANGELOG.md").write_text("uncommitted edit\n", encoding="utf-8")
            rc = rp.main([
                "prep",
                "--repo", str(self.repo),
                "--version", "v0.6.2",
                "--modules", "backend/cuda,backend/vulkan",
                "--offline-deps",
                "--skip-build",
            ])
            self.assertEqual(rc, 1)
        finally:
            rp._local_download_proxy_url = orig_cache_fn


if __name__ == "__main__":
    unittest.main()
