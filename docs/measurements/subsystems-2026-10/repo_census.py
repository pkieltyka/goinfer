#!/usr/bin/env python3
"""repo_census.py — the git and source facts behind docs/completed/task-subsystems-2026-10.md §2, §6 and §7.

Read-only. Run from a goinfer checkout (a worktree of origin/main, so another session's tree is not read):

  python3 docs/measurements/subsystems-2026-10/repo_census.py [--since YYYY-MM-DD]

The donor-checkout mode was retired with goinfer#71: it read an external module's tree through `git archive`, and
that module is gone. Everything below runs on this repository alone; the retired mode's original is the file at
ab2721995e027c8ce6630396c922916cb2357a1d.
skipped.

Sections: commit rate and the most-touched files; the parity manifest's shared sets and how often their files change;
module-version bumps in goinfer by what drove them (by commit subject, so approximate); test and non-test files that import
`decoder`, per module; and the memory-related file groups.
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile

DECODER = '"github.com/townsendmerino/goinfer/decoder"'
MEMORY_GROUPS = {
    "residency": ["decoder/residency.go"],
    "fit guard": ["decoder/fitguard.go"],
    "fit plan, query, Metal fit": ["decoder/fitplan.go", "decoder/fitquery.go", "decoder/metal_fitctx.go"],
    "prefill scratch": ["decoder/prefill_budget.go", "decoder/scratch.go", "decoder/prefill_chunk_scratch.go"],
    "weight and resident bytes": ["decoder/weightbytes.go", "decoder/residentneed.go"],
    "KV plan": ["decoder/kvresolve.go", "decoder/kvgeometry.go", "decoder/kvpolicy.go", "decoder/kvscope.go"],
    "pagers": ["decoder/moepool.go", "decoder/moepaging.go", "decoder/moeworkingset.go", "decoder/layerpaging.go"],
    "probes": ["decoder/hostram_*.go", "decoder/memwatch_*.go", "decoder/swapwatch.go", "decoder/faultcount_*.go",
               "decoder/coldload*.go", "decoder/giwmap_*.go"],
    "internal/hostmem": ["internal/hostmem/*.go"],
    "internal/swapguard": ["internal/swapguard/*.go"],
    "internal/fitcmd": ["internal/fitcmd/*.go"],
    "internal/kvmgr": ["internal/kvmgr/*.go"],
}
BUMP_CLASSES = [
    ("towers (vision, audio)", r"vision|tower|siglip|pixtral|\bvl\b|image|audio|whisper|voxtral|asr|mmproj|bicubic"),
    ("GPU module", r"gpu/v|qwencuda|visioncuda|gpu v0|anncuda|annmetal"),
    ("linalg and kernels", r"linalg|kernel|quantiz|matmul|neon|avx|int4|int8|bf16|dotprod|fallback|scales|w4a8|widen|binary16"),
    ("embed, formats, mmap", r"embed|gguf|q4_k|q5k|q6k|dequant|safetensors|mmap|split"),
]
MOVING = ["vision", "audio", "gpu", "gpu/qwencuda", "gpu/qwenmetal", "gpu/visioncuda", "gpu/visionmetal", "gpu/webgpu",
          "gpu/anncuda", "gpu/annmetal", "gpu/enccuda", "gpu/encmetal"]


def git(*args, cwd="."):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout


def count_commits(since, paths, cwd="."):
    out = git("log", f"--since={since}", "--no-merges", "--full-history", "--format=%H", "--", *paths, cwd=cwd)
    return len([x for x in out.splitlines() if x])


def expand(pats):
    files = []
    for p in pats:
        files += [f for f in glob.glob(p) if not f.endswith("_test.go")]
    return sorted(set(files))


def lines(files):
    return sum(sum(1 for _ in open(f, errors="replace")) for f in files)


def go_files(root, test):
    for d, _, fs in os.walk(root):
        if "/." in d or "_to_delete" in d:
            continue
        for f in fs:
            if f.endswith(".go") and f.endswith("_test.go") == test:
                yield os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-26")
    a = ap.parse_args()
    head = git("log", "-1", "--format=%h %ad", "--date=short").strip()
    print(f"# repository census at {head}, window since {a.since}\n")

    shas = [x for x in git("log", f"--since={a.since}", "--no-merges", "--format=%ad", "--date=short").splitlines() if x]
    print(f"## rate: {len(shas)} non-merge commits; last ten days:")
    for day, n in sorted(collections.Counter(shas).items())[-10:]:
        print(f"  {day} {n}")
    names = [x for x in git("log", f"--since={a.since}", "--no-merges", "--name-only", "--format=").splitlines() if x]
    print("\n## most-touched files")
    for f, n in collections.Counter(names).most_common(25):
        print(f"  {n:4d}  {f}")

    m = json.load(open("testdata/parity_manifest.json"))
    uses = collections.Counter(u for v in m["families"].values() for u in v.get("uses", []))
    print(f"\n## parity manifest: {len(m['families'])} families; sets used: {dict(uses)}")
    for name, files in m["shared_sets"].items():
        for f in files:
            if os.path.exists(f):
                print(f"  {name:10s} {f:32s} {lines([f]):5d} lines {count_commits(a.since, [f]):4d} commits")
    adapters = len(re.findall(r"^func \w+Architecture\(cfg \*Config\)", open("decoder/registry.go").read(), re.M))
    print(f"  decoder/registry.go holds {adapters} family adapters")

    subjects = git("log", f"--since={a.since}", "--no-merges", "--full-history", "-G", "townsendmerino/aikit",
                   "--format=%s", "--", "go.mod").splitlines()
    cls = collections.Counter()
    for s in subjects:
        cls[next((k for k, rx in BUMP_CLASSES if re.search(rx, s.lower())), "other")] += 1
    print(f"\n## module requirement changes in go.mod: {len(subjects)}; by subject: {dict(cls)}")

    print("\n## files importing decoder")
    for mod in ("cuda", "metal", "gpu", "internal/serveapp"):
        t = list(go_files(mod, True))
        n = list(go_files(mod, False))
        ti = sum(1 for f in t if DECODER in open(f, errors="replace").read())
        ni = [f for f in n if DECODER in open(f, errors="replace").read()]
        ids = set()
        for f in n:
            ids |= set(re.findall(r"\bdecoder\.[A-Z]\w*", open(f, errors="replace").read()))
        print(f"  {mod:18s} test files {len(t):4d}, importing decoder {ti:4d}; non-test importing {len(ni):3d}; "
              f"distinct decoder identifiers {len(ids)}")

    print("\n## memory-related file groups")
    allf = []
    for name, pats in MEMORY_GROUPS.items():
        fs = expand(pats)
        allf += [f for f in fs if f.startswith("decoder/")]
        print(f"  {name:28s} {len(fs):3d} files {lines(fs):6d} lines {count_commits(a.since, fs):4d} commits")
    print(f"  decoder files above: {len(allf)}, {lines(allf)} lines, {count_commits(a.since, allf)} commits")

if __name__ == "__main__":
    main()
