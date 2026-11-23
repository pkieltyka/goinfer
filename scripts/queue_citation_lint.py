#!/usr/bin/env python3
"""queue_citation_lint.py — every citation in the queue must resolve, and its target must match.

The queue is queues/QUEUE.md in the private notes checkout (a sibling ../goinfer-notes, or --notes PATH) when that holds one, else
docs/QUEUE.md; with neither, the queue's commit citations are skipped and the run says so.

Commit citations (sha -> subject) are recorded in a generated index (citation-index.md, beside the queue) and compared on every run. Code
citations come in two forms since CC0 (docs/completed/task-code-comments-2026-10.md):

  - `path.go:Name` / `path.go:Type.Method` in a LIVE doc names a declaration, and the lint checks the file declares it NOW. Nothing is
    recorded and nothing goes stale when lines move above it.
  - `path:line` is accepted only in a PINNED record: a doc carrying `<!-- citations-at: <commit> -->` in its first lines. The lint
    checks each file and line in the local Git tree or in checksum-verified historical source recorded for that commit.
    It leaves the pinned document unchanged.
  - `path:line` in an unpinned doc is red: "pin this doc (`citations-at`) or name the declaration". A path under an allow-path
    comment (stdlib, llama.cpp) is not a citation into this project and is exempt.

The pre-CC0 line-number index (content keyed, with the "moved but unchanged" acceptance) is gone with the door it guarded; what
follows describes the commit side, which is unchanged.

WHY THIS EXISTS. `9e5f8fa` was cited in docs/QUEUE.md as the precedent for a goldens-gated deps_hash
refresh — "a metadata field addition re-staled weights.go and the refresh ran 19 goldens". It is
`fix(quant): reject --quant that conflicts with a prequant .giw at startup` and touches the manifest
not at all. The wrong SHA was then repeated several times, from the file, without anyone opening it.

That is what a state document is *for*, and why an error in one is worse than the same error in
conversation: nothing re-derives what a state document asserts. A claim made in passing gets
re-checked when it matters; a claim written into the queue gets cited. So the error propagated with
more confidence than it started with.

The remedy is mechanical. This lint extracts every SHA-looking token from the queue, resolves it, and
compares its subject against a recorded index at the bottom of the file. `--update` regenerates the
index. Three failure modes are covered, and all three were mutation-checked:

  - a SHA that does not resolve  -> fabricated, mistyped, or from a dropped branch
  - a SHA whose subject changed  -> the index is stale, or the citation was always wrong
  - a SHA reachable from NO ref  -> amended away or rebased out; see below

THE THIRD ONE WAS A HOLE IN THIS LINT, found 2026-08-12 by walking into it. `git log -1 <sha>`
succeeds for any object still in the local store, including one no branch or tag points at. Amending
a commit leaves the original in the reflog, so every citation to the old SHA kept passing green while
pointing at history that existed on exactly one machine. Turning the check on found FOUR such
citations already in the queue, all from this campaign's own rebases and amends — they had been green
the whole time.

Worse than a plain false green: it was MACHINE-DEPENDENT. The same citation is a hard red in a fresh
clone, where the orphan was never fetched. A gate whose answer depends on which box ran it is the
Environment class (docs/parity-coverage-policy.md) wearing a gate's clothes — and this one was the
gate written to stop exactly this kind of drift, which is the G-01 shape one level up: the instrument
exempting itself from the property it enforces.

The check is reachability from any ref, not ancestry of HEAD, so a commit legitimately living on
another branch or a tag still passes.

A citation that cannot resolve here but is still legitimate — another repository, a commit that
never left a machine — is declared explicitly:

    <!-- sha-lint: allow <sha> <reason> -->

The reason is mandatory and is printed by the lint. That is the whole mechanism: it does not make
the citation correct, it makes the fact that nobody has verified it VISIBLE, which is the state
9e5f8fa was in for weeks while reading as authoritative.

Deliberately NOT checking that the surrounding prose describes the commit accurately — that is not
mechanisable. It checks the citation, which is the part that was wrong.

SECOND INCIDENT, 2026-09-13, on the PATH side. `docs/task-work-queue-2026-09.md` was written and
left untracked for a few hours — this repo's normal resting state for a fresh design record —
which reded this lint, because live_docs() used to walk the filesystem (`ROOT.rglob`) rather than
git. The only remedy on offer was to remove the file from the tree, which is what happened, under
time pressure, to unblock an unrelated push; it was never restored, and the doc is gone from every
branch, stash, dangling object and backup. The fix (J0) is the same shape as the SHA-lint's own
`docs/internal/` exclusion, generalised from a directory to a property: `live_docs()` now skips
anything `git ls-files` does not resolve, tracked and untracked alike, and prints what it skipped
rather than silencing it — see `untracked_live_docs()` and its doc comment.
"""

import collections
import functools
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUEUE = ROOT / "docs" / "QUEUE.md"


def index_path():
    """The generated citation index's own file. It lived INSIDE docs/QUEUE.md until 2026-09-24,
    where every --update rewrote hundreds of rows in the file people also hand-edit: 383 of the 490
    commits touching QUEUE.md since 2026-08-24 changed nothing but index rows, and it was the repo's
    merge hot spot. The index and the checks are unchanged; only where it is stored moved. Derived
    from QUEUE at call time so a test that points QUEUE at a scratch repo moves the index with it.
    A QUEUE.md that still carries the old in-file block is migrated by the next --update (the block
    is moved out; everything around it, including anything after it, is kept)."""
    return (queue_file() or QUEUE).parent / "citation-index.md"


def notes_root():
    """The private notes checkout, or None: `--notes PATH` on the command line, else a sibling `../goinfer-notes`. No
    environment variable: where a checkout sits is a property of the command, not of the shell. The notes hold the task, queue,
    plan and scoping docs once they leave this repository; CI has no checkout, so anything CI runs must pass without it."""
    argv = sys.argv
    if "--notes" in argv and argv.index("--notes") + 1 < len(argv):
        cand = pathlib.Path(argv[argv.index("--notes") + 1]).expanduser()
    else:
        cand = ROOT.parent / "goinfer-notes"
    return cand.resolve() if cand.is_dir() else None


def queue_file():
    """The queue the SHA index is checked against: queues/QUEUE.md in the notes checkout when it has one, else docs/QUEUE.md
    here, else None (the queue lives only in the notes and no checkout is present: the commit citations in it are skipped, and
    main() says so)."""
    n = notes_root()
    if n and (n / "queues" / "QUEUE.md").is_file():
        return n / "queues" / "QUEUE.md"
    return QUEUE if QUEUE.exists() else None


def doc_label(f):
    """A document's name in a message: its path under ROOT, or `notes:` and its path under the notes checkout."""
    try:
        return str(f.relative_to(ROOT))
    except ValueError:
        return "notes:" + str(f.relative_to(notes_root()))


_notes_tracked_cache = {}


def _is_tracked(f):
    """Is f a git-tracked file of the repository it sits in? True when git cannot say (the pre-J0 degrade)."""
    n = notes_root()
    if n and (n in f.parents):
        if n not in _notes_tracked_cache:
            try:
                out = subprocess.run(["git", "ls-files", "-z"], cwd=n, capture_output=True, check=True, env=_repo_env())
                _notes_tracked_cache[n] = {p for p in out.stdout.decode("utf-8", "replace").split("\0") if p}
            except Exception:
                _notes_tracked_cache[n] = None
        t, rel = _notes_tracked_cache[n], str(f.relative_to(n))
    else:
        t, rel = _tracked_files(), str(f.relative_to(ROOT))
    return t is None or rel in t


_TRACKED_SENTINEL = object()
_tracked_cache = _TRACKED_SENTINEL


def _tracked_files():
    """Every path `git ls-files` reports for this repo, as a set of `ROOT`-relative strings — one
    subprocess call, cached for the process (J0, task-work-queue-2026-09.md). None means git itself
    is unusable here (no repository, git missing): callers must degrade to their pre-J0 behavior
    (lint everything) and say so, never fail silently shut or silently open.
    """
    global _tracked_cache
    if _tracked_cache is not _TRACKED_SENTINEL:
        return _tracked_cache
    try:
        # -z / null-split, not the default newline-delimited output: git quotes paths containing
        # spaces or non-ASCII bytes in that mode, which would silently mismatch every str(Path)
        # comparison below for such a file. No filename in this repo needs it today, but the
        # subprocess call is cheap either way and a doc title is not this lint's business to ban.
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        _tracked_cache = {p for p in out.stdout.decode("utf-8", "replace").split("\0") if p}
    except Exception:
        _tracked_cache = None
    return _tracked_cache


def _candidate_docs():
    """The rglob scan shared by live_docs() and untracked_live_docs(), before the tracked-file
    split — one filesystem walk instead of two, and the single place the docs/internal/ and
    testdata/ exclusions live."""
    out = []
    q = queue_file()
    n = notes_root()
    # the notes checkout's docs are live documents too, except its archive (completed/), which ages like docs/completed/
    notes_docs = [f for f in sorted(n.rglob("*.md")) if f.relative_to(n).parts[0] != "completed"] if n else []
    for f in sorted(ROOT.rglob("*.md")) + notes_docs:
        # the index is GENERATED from citations; linting its rows as citations would be circular
        if "/.git/" in str(f) or "docs/completed/" in str(f) or f in (QUEUE, q) or f in (QUEUE.parent / "citation-index.md", index_path()):
            continue
        # docs/internal/ is EXCLUDED, and the reason is not tidiness. That directory is
        # DELIBERATELY UNCOMMITTED (.gitignore), so every path inside it resolves on exactly one
        # machine and on none of the clones CI or anyone else runs. Linting it produces failures
        # that are structurally unfixable from anywhere but here -- it has already generated
        # local-only false positives -- and a lint that reds on files CI cannot see is a lint people
        # stop reading. That is a worse outcome than the coverage it buys. Now belt-and-braces once
        # untracked() below also excludes it (it is .gitignore'd, hence untracked) — left in place so
        # the reasoning stays attached to the example it was written for.
        if "docs/internal/" in str(f):
            continue
        # testdata/ files are gate test inputs (prose prompts), not documentation.
        # Their internal citations are not maintainable as code citations and must not be linted.
        if "/testdata/" in str(f):
            continue
        out.append(f)
    return out


def live_docs():
    """Every LIVE, GIT-TRACKED markdown document, i.e. everything outside docs/completed/.

    The burial folder is deliberately excluded: those are historical records, and their citations are
    SUPPOSED to age with the code they described. Linting them would fight the archival rule and bury
    the live signal in ~300 lines of expected noise. QUEUE.md is checked first because it hosts the
    index.

    UNTRACKED FILES ARE EXCLUDED (J0, 2026-09-13). `docs/task-work-queue-2026-09.md` was written,
    left untracked for a few hours — this repo's normal resting state for a fresh design record —
    and reded the pre-push lint, because this function used to walk the filesystem rather than git.
    The only remedy the lint offered was to remove the file from the tree; that is what happened,
    under time pressure, to unblock an unrelated push, and it was never restored. The exclusion
    already written for docs/internal/ applies word for word: an untracked file's citations resolve
    on exactly one machine, and a lint that reds on files CI cannot see is a lint people stop
    reading. See untracked_live_docs() for the ones this skips, and main()'s startup note, which
    prints that set rather than silencing it — a false red traded for an invisible hole would repeat
    the same failure one door down.
    """
    q = queue_file()
    out = [q] if q else []
    for f in _candidate_docs():
        if not _is_tracked(f):
            continue
        out.append(f)
    return out


def untracked_live_docs():
    """The live-doc-shaped .md files live_docs() is skipping because git does not track them —
    printed once by main(), never silently. Empty when git is unusable (there is nothing to skip on
    that basis; the degrade-to-everything note covers that case instead)."""
    if _tracked_files() is None:
        return []
    return [f for f in _candidate_docs() if not _is_tracked(f)]
MARK_BEGIN = "<!-- CITATION-INDEX: generated by scripts/queue_citation_lint.py --update; do not edit by hand -->"
MARK_END = "<!-- /CITATION-INDEX -->"

# 7-40 hex chars delimited by non-word characters. Requires at least one digit AND one letter, which
# is what separates a commit id from an English word like "deffff" or a decimal figure like 1234567.
SHA_RE = re.compile(r"(?<![0-9a-zA-Z_])([0-9a-f]{7,40})(?![0-9a-zA-Z_])")


def looks_like_sha(tok: str) -> bool:
    return any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)


# GOINFER_SHA_LINT_REPOS can add explicit Git checkouts for commit and live
# citations. Pinned source checks always use this repository and its evidence.
class SearchError(Exception):
    """A repository could not be searched — distinct from a SHA not being found in it."""


# A commit that still resolves locally but is reachable from no ref: amended away, rebased out, or
# on a deleted branch. Distinct from "does not resolve" because the remedy differs — the citation is
# not fabricated, it points at history that no longer exists anywhere others can see.
ORPHANED = object()

# Count the checkout paths that resolved during this run.
RESOLVED_FROM = collections.Counter()

# Cited commits reachable only from a LOCAL ref — reported as a warning, never a hard failure here.
LOCAL_ONLY = []


def sibling_repos():
    root = QUEUE.parent.parent
    env = os.environ.get("GOINFER_SHA_LINT_REPOS")
    if env:
        # Explicit search roots must be usable Git repositories.
        out = [str(root)]
        for c in env.split(":"):
            pp = pathlib.Path(c).expanduser()
            if not (pp / ".git").exists():
                raise SearchError(f"GOINFER_SHA_LINT_REPOS names {pp}, which is not a git repository")
            out.append(str(pp))
        return out
    return [str(root)]


# Git hooks export the pushing repository's environment. Clear it when looking
# up citations whose repository is selected by cwd.
_GIT_REPO_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
                  "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_PREFIX")


def _repo_env() -> dict:
    """The environment for a git command that names its repository by cwd."""
    return {k: v for k, v in os.environ.items() if k not in _GIT_REPO_VARS}


def _commit_count(repo: str) -> str:
    r = subprocess.run(["git", "rev-list", "--count", "HEAD"], capture_output=True, text=True, cwd=repo, env=_repo_env())
    return r.stdout.strip() or "?"


def subject_of(sha: str):
    """Return the subject, or None if the SHA is genuinely absent from every searchable repo.

    Raises SearchError if a repo could not be searched at all. "Could not look" and "looked and did
    not find" are different answers, and conflating them produces a false RED on a correct citation —
    which costs more than a false green, because it prompts someone to change something accurate.
    """
    for repo in sibling_repos():
        probe = subprocess.run(["git", "rev-parse", "--git-dir"],
                               capture_output=True, text=True, cwd=repo, env=_repo_env())
        if probe.returncode != 0:
            raise SearchError(f"{repo}: not a usable git repository ({probe.stderr.strip()})")
        # A SHALLOW clone cannot answer this question at all. It holds one commit, so every cited
        # SHA "does not resolve" — and reporting 35 accurate citations as fabricated is the exact
        # false red this file's own docstring calls worse than a false green, because it tells
        # someone to go and change references that were right.
        #
        # This is the "could not look" versus "looked and did not find" distinction the rest of
        # this function already makes, applied to the one case that actually shipped: the lint ran
        # in CI under a depth-1 checkout and main was red for 15 consecutive runs.
        shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"],
                                 capture_output=True, text=True, cwd=repo, env=_repo_env())
        if shallow.stdout.strip() == "true":
            raise SearchError(
                f"{repo}: SHALLOW clone — it contains {_commit_count(repo)} commit(s), so no "
                f"citation can be resolved and every one would be reported as fabricated.\n"
                f"  This lint needs full history. In GitHub Actions set:\n"
                f"      - uses: actions/checkout@v4\n"
                f"        with:\n"
                f"          fetch-depth: 0\n"
                f"  Locally: git fetch --unshallow")
        r = subprocess.run(
            ["git", "log", "-1", "--format=%s", sha],
            capture_output=True, text=True, cwd=repo, env=_repo_env(),
        )
        if r.returncode == 0:
            # RESOLVING IS NOT ENOUGH — the commit must be REACHABLE FROM A REF.
            #
            # `git log -1 <sha>` succeeds for any object still in the local store, including one
            # that no branch or tag points at: an amended-away or rebased-out commit survives in
            # the reflog and keeps resolving for as long as gc spares it. Found the honest way —
            # this file's own bump commit was amended, every citation to the old sha kept passing
            # green, and the old sha was by then reachable from nothing.
            #
            # That green was also MACHINE-DEPENDENT, which is the worse half: the same citation is
            # a hard red in a fresh clone, where the orphan was never fetched. A lint whose answer
            # depends on which box ran it is the Environment class (docs/parity-coverage-policy.md)
            # wearing a gate's clothes.
            #
            # So: reachable from any ref, or it does not count. Checked with rev-list rather than
            # `merge-base --is-ancestor HEAD`, because a legitimately cited commit may live on
            # another branch or a tag and must still pass.
            reach = subprocess.run(
                ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
                capture_output=True, text=True, cwd=repo, env=_repo_env(),
            )
            if reach.returncode != 0:
                anyref = subprocess.run(
                    ["git", "for-each-ref", "--contains", sha, "--format=%(refname)"],
                    capture_output=True, text=True, cwd=repo, env=_repo_env(),
                )
                refs = [r for r in anyref.stdout.split() if r]
                if not refs:
                    return ORPHANED
                # Reachable — but from WHAT? A commit on a local-only branch is exactly as invisible
                # to everyone else as an orphan, and passes here for the same bad reason: this box
                # has a ref nobody else does. c8b65ba did precisely that — green locally, "DOES NOT
                # RESOLVE" in CI, one push after the orphan check was added to stop this very shape.
                #
                # NOT fatal, deliberately. Citing a commit in the same push that creates it is normal
                # and would fail every pre-push run, which is how a lint gets disabled. CI is the
                # enforcement — it has only the pushed refs, so it fails there if the commit never
                # left. This names them so the failure is predicted rather than discovered.
                if not any(r.startswith("refs/remotes/") or r.startswith("refs/tags/") for r in refs):
                    LOCAL_ONLY.append((sha, refs[0]))
            subj = r.stdout.strip()
            if repo != str(QUEUE.parent.parent):
                subj = f"[{pathlib.Path(repo).name}] {subj}"
            return subj
        err = r.stderr.lower()
        # "unknown revision" is a real negative. Anything else — a corrupt or unreadable ref store,
        # a permissions problem — means the search did not happen and must not read as absence.
        if "unknown revision" not in err and "bad object" not in err and "ambiguous argument" not in err:
            raise SearchError(f"{repo}: git failed while resolving {sha}: {r.stderr.strip()}")
    return None


# A path citation: an optional repo-relative directory chain ending in a Go/shell/python file and a
# line number. Deliberately not matching bare filenames — those carry no line to check against, and
# counting them as validated would be the unqualified-green shape this lint exists to avoid.
PATH_RE = re.compile(r"(?<![\w/])((?:[a-z0-9_]+/)*[a-z0-9_]+\.(?:go|sh|py)):(\d+)(?![\d])")


def path_repos():
    """Search the local project and explicitly selected checkouts for live citations."""
    return [(repo_name(repo), pathlib.Path(repo)) for repo in sibling_repos()]


def repo_name(path) -> str:
    """The repo's NAME, from its go.mod module line — not from the directory it happens to sit in.

    A clone into a directory not called "goinfer" made path_repo_present("goinfer") false, so every
    goinfer-recorded citation was SKIPPED: 116 of them, reported as a clean green. CI checks out into
    a directory matching the repo, so it was correct there and wrong anywhere else — one more verdict
    that depended on the environment rather than on the citations.

    `module github.com/pkieltyka/goinfer` is in go.mod regardless of what the checkout is
    called, so the last element of the module path is the stable answer. Directory name remains the
    fallback for a root with no go.mod.
    """
    p = pathlib.Path(path)
    gm = p / "go.mod"
    try:
        m = re.search(r"^module\s+(\S+)", gm.read_text(), re.M)
        if m:
            return m.group(1).rstrip("/").split("/")[-1]
    except OSError:
        pass
    name = p.name
    return name.split("@")[0]


def path_repo_present(name: str) -> bool:
    """Can path citations into this repository be checked here?"""
    return any(n == name for n, _ in path_repos())


def commit_repo_present(name: str) -> bool:
    """Can Git history for this repository be checked here?"""
    return any(repo_name(r) == name and (pathlib.Path(r) / ".git").exists()
               for r in sibling_repos())


# A content key only works if the content DISCRIMINATES. `//`, `}`, a blank line or a lone brace
# appear hundreds of times, so "the recorded content is now at line 114" is a coincidence rather than
# a move. Such citations are indexed as UNKEYABLE and counted, not silently mis-matched — the failure
# mode otherwise is a confident wrong answer, which is worse than no answer.
def discriminating(content: str) -> bool:
    c = content.strip()
    return len(c) >= 12 and any(ch.isalnum() for ch in c)


# When the cited line is not discriminating, key on the nearest ENCLOSING DECLARATION instead. A
# citation at a bare `//` is pointing at a line; what it MEANS is almost always a nameable thing —
# allocSlots' call site, a const, a type — and a name is discriminating by construction and survives
# reformatting inside the body. The citation then verifies "this declaration still exists" rather
# than "this line still reads the same", which is the claim the prose was making anyway.
DECL_RE = re.compile(r"^(?:func|type|const|var)\s")


def anchor_for(lines, line: int):
    for i in range(min(line, len(lines)) - 1, -1, -1):
        if DECL_RE.match(lines[i]):
            return "anchor: " + lines[i].strip()[:80]
    return None


def resolve_base(rel: str):
    """The (name, base) resolve_path() reads from — the FIRST root containing rel, same order.

    Exists because the "did the content move?" search used to re-walk EVERY root instead of the one
    the citation actually resolved against, and then reported a line number from a different file as
    if it were a shift in this one. See the SHIFTED branch for what that produced.
    """
    for name, base in path_repos():
        if (base / rel).is_file():
            return name, base
    return None, None


def moved_to(rel: str, want: str):
    """Where the recorded content `want` sits now in the RESOLVING copy of rel: (line, count).

    ACCEPTED SINCE 2026-09-24 (owner decision): a cited line that MOVED but did not CHANGE is not a
    broken citation, it is a stale line number — the claim the prose makes still holds, word for
    word. It was red, and every edit above a cited line forced hand-rewrites across unrelated docs
    (~70 in one bug-fix commit). Only an EXACTLY-ONE match is accepted: count 0 is CONTENT ABSENT /
    GONE (still red — the file no longer supports the claim) and count > 1 is AMBIGUOUS (red —
    picking one would be a guess). Scoped to the resolving base, never another repo or version."""
    _rn, rbase = resolve_base(rel)
    if rbase is None or not want:
        return None, 0
    hits = [i for i, l in enumerate((rbase / rel).read_text(errors="replace").split("\n"), 1)
            if l.strip()[:88] == want]
    return (hits[0] if len(hits) == 1 else None), len(hits)


def repoint_citation(doc: pathlib.Path, rel: str, old: int, new: int) -> int:
    """Rewrite `rel:old` (and a range `rel:old-end`, keeping its length) to the new line IN THE
    DOCUMENT'S PROSE — the part --update used to be unable to fix, which is why it refused. Returns
    the number of occurrences rewritten."""
    text = doc.read_text()
    pat = re.compile(re.escape(rel) + ":" + str(old) + r"(?:-(\d+))?(?!\d)")

    def sub(m):
        if m.group(1):
            return f"{rel}:{new}-{int(m.group(1)) + new - old}"
        return f"{rel}:{new}"
    out, n = pat.subn(sub, text)
    if n:
        doc.write_text(out)
    return n


def resolve_path(rel: str, line: int):
    """Return (repo, content) for a cited path:line, or ("", None) when the file exists nowhere.

    A file present but SHORTER than the cited line is not an error here — it is a citation whose
    content is absent, which the caller reports as such.
    """
    for name, base in path_repos():
        f = base / rel
        if f.is_file():
            RESOLVED_FROM[("checkout", name)] += 1
            lines = f.read_text(errors="replace").split("\n")
            if line <= len(lines):
                c = lines[line - 1].strip()
                if discriminating(c):
                    return name, c
                a = anchor_for(lines, line)
                return name, (a if a else c)
            return name, ""
    return "", None


# A BARE file reference: same shape, no line number. Existence is checkable without a content key —
# it catches a file that moved packages or was deleted, which is most of the way to keeping a
# reference honest and costs nothing to record.
BARE_RE = re.compile(r"(?<![\w/])((?:[a-z0-9_]+/)*[a-z0-9_]+\.(?:go|sh|py))(?![\w:])")


# ---- SYMBOL CITATIONS (CC0.b, docs/completed/task-code-comments-2026-10.md) ----
#
# A live doc names a declaration, not a line: `decoder/forwardn.go:Model.forwardN`, `gpu/residency.go:residentDecoder.PrefillLast`,
# `scripts/x.py:Tool.run`. The check is that the file DECLARES that name. It survives every edit that does not rename the declaration
# or move it to another file, and a rename is exactly when the doc should go red. There is no content key and no index row: the
# declaration is the key, and it is read from the file each run.
#
# Never colliding with the other two forms: PATH_RE needs DIGITS after the colon, BARE_RE refuses a following colon, and a symbol
# starts with a letter or underscore. Only .go and .py have a symbol form (`def` and `class` count in Python); there is no .sh form,
# so a script is cited by name, or by line in a PINNED record (the tree carried zero live .sh line citations when CC0 was written).
SYMBOL_RE = re.compile(r"(?<![\w/])((?:[a-z0-9_]+/)*[a-z0-9_]+\.(?:go|py)):([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)(?!\w)")

# A line citation as the docs write it: `x.go:10`, `x.go:10-20`, `x.go:10–20` (an EN DASH) and `x.go:10,34,56` (a comma list of more lines in
# the same file). PATH_RE keeps only the first number, which is all a CONTENT key needs but not what a pinned record needs: EVERY number it
# names must lie inside the file at its commit. (A range or list whose later numbers were never read is how an en-dash range hid past EOF.)
PATH_RANGE_RE = re.compile(r"(?<![\w/])((?:[a-z0-9_]+/)*[a-z0-9_]+\.(?:go|sh|py)):(\d+)(?:[-–](\d+))?((?:,\d+(?:[-–]\d+)?)*)(?![\d])")


def _go_line_starts_in_code(lines):
    """For each line, True when it BEGINS outside a raw string and outside a block comment.

    Declarations are found by their first column, so a line that merely begins inside a raw string must not count: this repo embeds
    MSL, CUDA and WGSL kernel sources in Go raw strings, and WGSL has `const NAME: u32 = 64;` at column 0. A scanner that read those
    as Go declarations would let a doc cite a name that exists only inside a shader and call it declared."""
    in_raw = in_block = False
    out = []
    for ln in lines:
        out.append(not in_raw and not in_block)
        i, n = 0, len(ln)
        while i < n:
            if in_raw:
                j = ln.find("`", i)
                if j < 0:
                    break
                in_raw, i = False, j + 1
            elif in_block:
                j = ln.find("*/", i)
                if j < 0:
                    break
                in_block, i = False, j + 2
            else:
                c = ln[i]
                if c == "/" and ln[i + 1:i + 2] == "/":
                    break
                if c == "/" and ln[i + 1:i + 2] == "*":
                    in_block, i = True, i + 2
                elif c == "`":
                    in_raw, i = True, i + 1
                elif c in "\"'":
                    i += 1
                    while i < n and ln[i] != c:
                        i += 2 if ln[i] == "\\" else 1
                    i += 1
                else:
                    i += 1
    return out


# func [receiver] Name[typeparams](  — receivers `(s *T)`, `(s T)`, `(*T)`, `(T)`, `(s *T[K, V])`; the receiver's TYPE is what a citation names.
_GO_FUNC = re.compile(r"^func\s+(?:\(\s*(?:[A-Za-z_]\w*\s+)?\*?\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*\)\s*)?([A-Za-z_]\w*)\s*[\[(]")
_GO_TYPE = re.compile(r"^type\s+([A-Za-z_]\w*)")
_GO_VALUE = re.compile(r"^(?:const|var)\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)")
_GO_BLOCK = re.compile(r"^(const|var|type)\s*\(\s*(?://.*)?$")
_GO_SPEC = re.compile(r"^\t([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)(?=[\s=,]|$)")
_GO_TOPLEVEL = re.compile(r"^(?:func|type|const|var|import|package)\b")

Decl = collections.namedtuple("Decl", "name kind start end doc_start")  # 1-based inclusive lines; start is the keyword/spec line


def go_decls(lines):
    """Every top-level Go declaration of a file as a Decl, one per NAME (a multi-name spec yields one each, sharing its span).

    Funcs and methods (`Type.Method`, the receiver's type with pointer and type parameters removed), types, consts and vars, including
    each spec of a grouped `const (`/`var (`/`type (` block. Only a column-0 keyword or a one-tab spec inside such a block counts, so a
    local `var x` in a function body, a struct field and a comment all stay out. gofmt'd source is assumed (CI refuses anything else).

    The span is what migration needs to say which declaration a LINE is inside: from the keyword to the last line before the next
    column-0 non-closer, trailing blank lines trimmed; doc_start reaches up over the contiguous `//` run directly above."""
    mask = _go_line_starts_in_code(lines)
    n = len(lines)
    out = []

    def doc_start(i):  # 0-based index of the declaration line -> 1-based start of its contiguous comment run
        j = i
        while j > 0 and mask[j - 1] and lines[j - 1].startswith("//"):
            j -= 1
        return j + 1

    def extent_end(i):  # last 0-based line index belonging to the top-level declaration that starts at i
        j = i + 1
        while j < n:
            ln = lines[j]
            if mask[j] and ln and ln[0] not in " \t})":
                break
            j += 1
        j -= 1
        while j > i and not lines[j].strip():
            j -= 1
        return j

    i = 0
    while i < n:
        ln = lines[i]
        if not mask[i] or not ln or ln[0] in " \t":
            i += 1
            continue
        m = _GO_BLOCK.match(ln)
        if m:
            kind, end = m.group(1), extent_end(i)
            specs = [j for j in range(i + 1, end + 1) if mask[j] and _GO_SPEC.match(lines[j])]
            for k, j in enumerate(specs):
                names = [x.strip() for x in _GO_SPEC.match(lines[j]).group(1).split(",")]
                nxt = specs[k + 1] if k + 1 < len(specs) else end  # the block's closing paren belongs to nobody
                # a comment run directly above the NEXT spec is that spec's doc, not this one's tail
                stop = nxt - 1
                while stop > j and lines[stop].startswith("\t//"):
                    stop -= 1
                while stop > j and not lines[stop].strip():
                    stop -= 1
                ds = j
                while ds > i + 1 and lines[ds - 1].startswith("\t//"):
                    ds -= 1
                for nm in names:
                    if nm != "_":
                        out.append(Decl(nm, kind, j + 1, stop + 1, ds + 1))
            i = end + 1
            continue
        m = _GO_FUNC.match(ln)
        if m:
            recv, name = m.group(1), m.group(2)
            if name != "_":
                out.append(Decl(f"{recv}.{name}" if recv else name, "func", i + 1, extent_end(i) + 1, doc_start(i)))
            i = extent_end(i) + 1
            continue
        m = _GO_TYPE.match(ln)
        if m:
            if m.group(1) != "_":
                out.append(Decl(m.group(1), "type", i + 1, extent_end(i) + 1, doc_start(i)))
            i = extent_end(i) + 1
            continue
        m = _GO_VALUE.match(ln)
        if m:
            e = extent_end(i)
            for nm in (x.strip() for x in m.group(1).split(",")):
                if nm != "_":
                    out.append(Decl(nm, "value", i + 1, e + 1, doc_start(i)))
            i = e + 1
            continue
        i += 1
    return out


@functools.lru_cache(maxsize=256)
def go_declarations(text: str):
    """The set of names a `path.go:Name` citation may use for this source."""
    return frozenset(d.name for d in go_decls(text.split("\n")))


_PY_DECL = re.compile(r"^(\s*)(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


@functools.lru_cache(maxsize=256)
def py_declarations(text: str):
    """`def` and `class` names; a def indented under a class is `Class.method`. Nested helpers deeper than one level are not citable."""
    out, cls = set(), None
    for ln in text.split("\n"):
        m = _PY_DECL.match(ln)
        if not m:
            continue
        indent, name = len(m.group(1)), m.group(2)
        if indent == 0:
            cls = name if ln.lstrip().startswith("class") else None
            out.add(name)
        elif cls and indent <= 8:
            out.add(f"{cls}.{name}")
    return frozenset(out)


def declared_names(rel: str, text: str):
    return py_declarations(text) if rel.endswith(".py") else go_declarations(text)


def enclosing_declaration(lines, line: int):
    """The Decl a 1-based line lies inside (its doc comment included), or None when it is outside every declaration: the package
    clause, the import block, a file-level comment, a blank line between declarations. The innermost wins (a block spec, not the
    block). This is what anchor_for used to approximate by scanning up for a `func|type|const|var` line, which also claimed the
    gap lines after a closing brace for the function above."""
    best = None
    for d in go_decls(lines):
        if d.doc_start <= line <= d.end and (best is None or d.start >= best.start):
            best = d
    return best


# ---- PINNED RECORDS (CC0.a) ----
#
# A dated record — an audit, a review, a measurement write-up — describes code at a point in time and is not maintained as current
# state. It says `<!-- citations-at: <commit> -->` near its top, and every path:line in it is resolved AT THAT COMMIT: the file existed
# there and has that many lines. No content is recorded (the record's claim was true of that tree by construction), and --update never
# rewrites a pinned doc, because rewriting it to match HEAD would make it describe code it never audited.
CITATIONS_AT_RE = re.compile(r"<!--\s*citations-at:\s*([0-9a-f]{7,40})\s*-->")
CITATIONS_AT_ANY_RE = re.compile(r"<!--\s*citations-at:")
PIN_NEAR_TOP = 40  # the marker must sit in the first N lines; a marker buried in a doc is invisible to a reader

# (repo root, commit, rel) -> (label, text) | None
PINNED_FILE_CACHE = {}

# Set True by step CC0.c (the door): a path:line in an UNPINNED doc is then a red. Off while live docs are being migrated.
CLOSE_THE_DOOR = True
DOOR_MESSAGE = "pin this doc (`citations-at`) or name the declaration"


_FENCE_RE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.M | re.S)
_CODE_SPAN_RE = re.compile(r"`[^`\n]*`")


def _without_code(text: str) -> str:
    """The prose of a markdown doc: fenced blocks and inline code spans blanked, newlines kept so line numbers hold. A marker QUOTED in
    code is notation (the task doc that introduces `citations-at` quotes it in running text), and reading it as a marker would red the
    doc that explains the scheme."""
    text = _FENCE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    return _CODE_SPAN_RE.sub(lambda m: " " * len(m.group(0)), text)


def pinned_marker(text: str):
    """(commit, None) for a pinned doc, (None, None) for an unpinned one, (None, error) for a marker that is present but not near the top."""
    prose = _without_code(text)
    head = "\n".join(prose.split("\n")[:PIN_NEAR_TOP])
    m = CITATIONS_AT_RE.search(head)
    if m:
        return m.group(1), None
    if CITATIONS_AT_ANY_RE.search(prose):
        return None, f"a `citations-at` marker exists but not near the top (first {PIN_NEAR_TOP} lines), where a reader would see it"
    return None, None


def commit_resolves(commit: str) -> bool:
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
                       capture_output=True, text=True, cwd=str(ROOT), env=_repo_env())
    return r.returncode == 0


def _git_show(commit: str, rel: str):
    r = subprocess.run(["git", "show", f"{commit}:{rel}"], capture_output=True, cwd=str(ROOT), env=_repo_env())
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


class CannotSearch(Exception):
    """Historical source evidence is unavailable or corrupt."""


def sealed_file_at(commit: str, rel: str):
    directory = ROOT / "testdata" / "citation-history"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        full = subprocess.check_output(["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT,
                                       env=_repo_env(), text=True).strip()
        entry = entries.get(full, {}).get(rel)
        if entry is None:
            return None
        digest = entry["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid SHA-256")
        data = (directory / (digest + ".txt")).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("SHA-256 mismatch")
        text = data.decode("utf-8", "replace")
        if len(text.splitlines()) != entry["lines"]:
            raise ValueError("line count differs")
        return "sealed source", text
    except (AttributeError, KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as err:
        raise CannotSearch(f"sealed source {commit}:{rel}: {err}") from err


def file_at(commit: str, rel: str):
    """Read the file from its original Git tree or the sealed historical source."""
    key = (str(ROOT), commit, rel)
    if key not in PINNED_FILE_CACHE:
        text = _git_show(commit, rel)
        PINNED_FILE_CACHE[key] = (repo_name(ROOT), text) if text is not None else sealed_file_at(commit, rel)
    return PINNED_FILE_CACHE[key]


def check_pinned(doc_rel: str, commit: str, text: str, allow):
    """Red lines for one pinned doc: every path:line must exist at the commit (the file, and the line, and a range's end), and every
    symbol citation must name a declaration the file had there. Nothing is recorded."""
    bad = []
    seen = set()
    for m in PATH_RANGE_RE.finditer(text):
        rel, a, b, extra = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if rel in allow or rel.split("/")[0] in allow or (rel, m.group(0)) in seen:
            continue
        seen.add((rel, m.group(0)))
        cite = m.group(0)
        numbers = [a] + ([int(b)] if b else []) + [int(n) for n in re.findall(r"\d+", extra)]
        try:
            got = file_at(commit, rel)
        except CannotSearch as e:
            bad.append(f"  {doc_rel}|{cite}  CANNOT SEARCH — {e}")
            continue
        if got is None:
            bad.append(f"  {doc_rel}|{cite}  PINNED PATH ABSENT — {rel} is absent at {commit[:10]}, the commit this record is pinned to")
            continue
        nlines = len(got[1].split("\n")) - (1 if got[1].endswith("\n") else 0)
        if min(numbers) < 1 or max(numbers) > nlines:
            bad.append(f"  {doc_rel}|{cite}  PINNED LINE past the end of {rel} at {commit[:10]} ({nlines} lines)")
    for m in SYMBOL_RE.finditer(text):
        rel, name = m.group(1), m.group(2)
        if rel in allow or rel.split("/")[0] in allow or (rel, name) in seen:
            continue
        seen.add((rel, name))
        try:
            got = file_at(commit, rel)
        except CannotSearch as e:
            bad.append(f"  {doc_rel}|{rel}:{name}  CANNOT SEARCH — {e}")
            continue
        if got is None:
            bad.append(f"  {doc_rel}|{rel}:{name}  PINNED PATH ABSENT — {rel} is absent at {commit[:10]}, the commit this record is pinned to")
        elif name not in declared_names(rel, got[1]):
            bad.append(f"  {doc_rel}|{rel}:{name}  PINNED SYMBOL GONE — {rel} declared no `{name}` at {commit[:10]}")
    return bad


def check_symbol(rel: str, name: str):
    """('ok'|'gone'|'nofile', repo) for a `path:Name` citation in an UNPINNED doc, resolved through the same roots as a line citation
    (the current checkout, then explicitly named checkouts)."""
    for rname, base in path_repos():
        f = base / rel
        if f.is_file():
            RESOLVED_FROM[("checkout", rname)] += 1
            return ("ok" if name in declared_names(rel, f.read_text(errors="replace")) else "gone"), rname
    return "nofile", ""


ALLOW_RE = re.compile(r"<!--\s*sha-lint:\s*allow\s+([0-9a-f]{7,40})\s+(.+?)\s*-->")
# Path citations into a project that is not checked out here — the Go stdlib, llama.cpp's gguf-py.
# Same mechanism and same reasoning as the commit allowlist: it does not make the citation correct,
# it makes "this points outside every repo we have" explicit instead of indistinguishable from a typo.
ALLOWPATH_RE = re.compile(r"<!--\s*citation-lint:\s*allow-path\s+(\S+)\s+(.+?)\s*-->")


def path_allowlist():
    out = {}
    for doc in live_docs():
        for m in ALLOWPATH_RE.finditer(doc.read_text(errors="replace")):
            out[m.group(1)] = m.group(2)
    return out


def allowlist(text: str):
    return {m.group(1): m.group(2) for m in ALLOW_RE.finditer(text)}


def body_without_index(text: str) -> str:
    if MARK_BEGIN in text:
        return text.split(MARK_BEGIN, 1)[0]
    return text


def tail_after_index(text: str) -> str:
    """Whatever follows the generated index block — content --update MUST NOT destroy.

    THIS FUNCTION EXISTS BECAUSE --update ATE THREE QUEUE ENTRIES. `--update` rebuilt the file as
    body_without_index(text) + <fresh index> and dropped everything after MARK_END. Appending an
    entry to the end of QUEUE.md — the obvious way to add one — put it after the index, so the next
    `--update` silently deleted it. B11, B12 and B13 were each written, committed as "filed", and
    erased by the following lint run; B14 survived only because it happened to be inserted ABOVE the
    marker.

    The failure was invisible from both ends: the write succeeded, the commit contained the entry
    (until the lint ran again), and the lint reported success while destroying content. A tool that
    silently deletes what it was pointed at is worse than one that refuses to run.
    """
    if MARK_END in text:
        return text.split(MARK_END, 1)[1]
    return ""


# ---- GITIGNORED DESTINATIONS: forbidden in a committed document ----
#
# The lint treats a gitignored path in TWO DISTINCT ROLES, and they get opposite treatment:
#
#   as a SOURCE      -- excluded (see live_docs). Its own citations resolve on one machine, so CI
#                       cannot check them and a red there is unfixable from anywhere else.
#   as a DESTINATION -- FORBIDDEN. A committed document citing a path under a gitignored directory
#                       is red, ALWAYS, and WITHOUT AN EXISTENCE TEST. The gitignore status decides.
#
# WHY NO EXISTENCE TEST. Two reasons, and the second is the one that makes this absolute:
#   1. No reader outside this machine can reach it. The pointer is dead on arrival in every clone.
#   2. An uncommitted target HAS NO HISTORY, so nobody can ever establish whether it existed. The
#      five notes this rule was written after are unauditable BY CONSTRUCTION -- git cannot say
#      whether they were deleted or never written, and no amount of care later recovers that.
# A rule that tested existence would pass today and rot silently; the gitignore status is a property
# of the repository, checkable identically in every clone.
#
# SCOPE, stated because it is narrower than it sounds: this decides by THIS repository's gitignore.
# A path into another repository's gitignored directory is not ignored here and
# this check cannot see it. Those are handled by not writing them -- see the cleanup in this commit.
#
# Concrete file paths only. A directory mention (`docs/internal/`) or a glob (`docs/internal/
# RELEASE-*.md`) names a CLASS, not a document, and is how the convention gets described at all --
# RELEASING.md and roadmap.md both legitimately do so.
IGNORED_DEST_RE = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*/[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)`")


def gitignored_destinations():
    """Every (doc, path) where a COMMITTED document cites a concrete path that git ignores."""
    import subprocess
    tracked = set()
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
        tracked = set(out.stdout.split())
    except Exception:
        return []
    cands, hits = {}, []
    for doc in live_docs():
        if ROOT not in doc.parents:
            continue  # a notes-checkout document is not this repository's: its own gitignore is not ours to apply
        rel = str(doc.relative_to(ROOT))
        if rel not in tracked:
            continue  # only COMMITTED documents make this claim
        for m in IGNORED_DEST_RE.finditer(doc.read_text(errors="replace")):
            path = m.group(1)
            if "*" in path or path in tracked:
                continue
            cands.setdefault(path, []).append(rel)
    if not cands:
        return []
    try:
        proc = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(cands),
                              capture_output=True, text=True)
        ignored = set(proc.stdout.split())
    except Exception:
        return []
    for path in sorted(ignored):
        for doc in cands.get(path, []):
            hits.append((doc, path))
    return hits


_REPOINT_PASSES = 0


def main() -> int:
    """The entry point. The re-point recursion below is capped at 3 passes PER INVOCATION; the counter is module state, so a caller that
    runs main() repeatedly in one process (the tests, a harness) must start each invocation from zero, or the fourth re-point anywhere
    in the process reads "citations still shifting after 3 passes" (found 2026-10-10 mutating the lint's own tests)."""
    global _REPOINT_PASSES
    _REPOINT_PASSES = 0
    rc = _main()
    sys.stderr.flush()
    print(verdict_line(rc), flush=True)
    return rc


def verdict_line(rc: int) -> str:
    """The last line of every run, so `| tail` shows the outcome; the exit code (which the pre-push hook
    reads) is unchanged. Exit 2 is a search that could not run (CANNOT SEARCH, a path that resolves in no repository)."""
    if rc == 0:
        return "VERDICT: PASS — citation lint: every citation resolves"
    why = "a search could not run" if rc == 2 else "citations do not resolve"
    return f"VERDICT: FAIL — citation lint: {why} (exit {rc}; the reasons are printed above); the pre-push hook refuses this push"


def _main() -> int:
    update = "--update" in sys.argv
    queue = queue_file()
    notes = notes_root()
    # Said on every run, so a skip and a pass never look alike (CI has no notes checkout).
    if notes is None:
        print("queue_citation_lint: no notes checkout (a sibling ../goinfer-notes, or --notes PATH): the task, queue, plan and "
              "scoping docs kept there were NOT examined here.")
    else:
        print(f"queue_citation_lint: notes checkout {notes}: its markdown documents are examined with this repository's "
              f"(its completed/ is excluded like docs/completed/).")
    if queue is None:
        print("queue_citation_lint: PRIVATE ROWS SKIPPED — no queue file (docs/QUEUE.md is absent and no notes checkout holds "
              "queues/QUEUE.md): the queue's commit citations were NOT checked here.")
        if update:
            print("queue_citation_lint --update: no queue, so no index to write.")
            return 0
    text = queue.read_text() if queue else ""
    body = body_without_index(text)
    # itext is the generated index: its own file, or (before the migration's first --update) the
    # legacy block still inside QUEUE.md. Every index READ below goes through itext.
    ipath = index_path()
    itext = ipath.read_text() if ipath.exists() else text

    # J0 (task-work-queue-2026-09.md): say what live_docs() is skipping, and why, up front — a
    # silent skip trades a false red for an invisible hole, the same shape this file's own module
    # docstring names for the SHA-lint allowlist. Printed once here regardless of how many times
    # live_docs()/untracked_live_docs() run later in this invocation.
    if _tracked_files() is None:
        print("queue_citation_lint: git ls-files unavailable here (no repository, or git missing) — "
              "linting every markdown file in the tree, tracked or not, matching pre-J0 behavior.")
    else:
        untracked = untracked_live_docs()
        if untracked:
            print(f"queue_citation_lint: note: {len(untracked)} untracked doc(s) not linted (commit "
                  f"them to bring their citations under the gate):")
            for f in untracked:
                print(f"    {doc_label(f)}")

    found = []
    for m in SHA_RE.finditer(body):
        tok = m.group(1)
        if looks_like_sha(tok) and tok not in found:
            found.append(tok)
    found.sort()

    if not found and queue is not None:
        # Zero citations is indistinguishable from "the regexp stopped matching". Refuse.
        sys.stderr.write("queue_sha_lint: found ZERO SHA citations — the queue always cites some, "
                         "so the extractor is broken rather than the file being clean.\n")
        return 1

    paths, symbols, pre_bad, pinned_docs = [], [], [], {}
    pallow = path_allowlist()
    for doc in live_docs():
        dtxt = doc.read_text(errors="replace")
        dbody = body if doc == queue else dtxt
        rel_doc = doc_label(doc)
        # A PINNED record is checked at its commit and then left alone: its citations never enter the path index, --update never
        # reaches it (repoint_citation is only ever called for a key in `paths`), and HEAD's version of the code is irrelevant to it.
        commit, perr = pinned_marker(dbody)
        if perr:
            pre_bad.append(f"  {rel_doc}  {perr}")
        if commit:
            if commit_resolves(commit):
                pre_bad += check_pinned(rel_doc, commit, dbody, pallow)
                pinned_docs[rel_doc] = commit
            else:
                pre_bad.append(f"  {rel_doc}  pinned to {commit}, which does not resolve in this repository (a rebased-away or mistyped commit)")
            continue
        for m in PATH_RE.finditer(dbody):
            key = f"{rel_doc}|{m.group(1)}:{m.group(2)}"
            if key not in [p[0] for p in paths]:
                paths.append((key, m.group(1), int(m.group(2))))
        for m in SYMBOL_RE.finditer(dbody):
            if (rel_doc, m.group(1), m.group(2)) not in symbols:
                symbols.append((rel_doc, m.group(1), m.group(2)))
    if CLOSE_THE_DOOR:
        # CC0.c: the door is shut. A line number in an unpinned doc is a pointer that goes stale on the next edit above it.
        per_doc = collections.defaultdict(list)
        for key, rel, ln in paths:
            if rel in pallow or rel.split("/")[0] in pallow:
                continue
            per_doc[key.split("|", 1)[0]].append(f"{rel}:{ln}")
        for d, cs in sorted(per_doc.items()):
            pre_bad.append(f"  {d}  {len(cs)} path:line citation(s) in an unpinned doc ({', '.join(cs[:6])}"
                           f"{', ...' if len(cs) > 6 else ''}) — {DOOR_MESSAGE}")
    if pre_bad:
        sys.stderr.write("queue_citation_lint: docs/QUEUE.md citations are not sound:\n")
        sys.stderr.write("\n".join(pre_bad) + "\n")
        return 1

    bares = []
    for m in BARE_RE.finditer(body):   # bare refs: QUEUE.md only, where they carry decisions
        rel = m.group(1)
        if rel not in bares:
            bares.append(rel)
    bares.sort()

    allowed = allowlist(text)
    resolved, unresolved, orphaned = {}, [], []
    for sha in found:
        try:
            s = subject_of(sha)
        except SearchError as e:
            sys.stderr.write(f"queue_sha_lint: CANNOT SEARCH — {e}\n")
            sys.stderr.write("  This is not a missing SHA. Fix the repository path and re-run; "
                             "reporting citations as unresolved from here would be a false red.\n")
            return 2
        if s is ORPHANED:
            # An allowlisted orphan is exactly the c8b65ba shape the annotation exists for: it
            # resolves locally (this machine fetched the branch before it was rebased away) but
            # is unreachable from any ref, on ANY machine — a fresh CI clone never fetches it at
            # all and reports it as unresolved (None) instead, which the `elif s is None` branch
            # below already checks against `allowed`. Before this fix, ORPHANED skipped that check
            # entirely, so the same allowlisted SHA read green in a fresh clone (None -> allowed)
            # and red on any machine that happened to have the orphan in its object store — the
            # exact machine-dependent false-red this lint's own docstring warns is worse than a
            # false green (found via c8b65ba itself, allowlisted since 2026-08-12 but never
            # actually honored here).
            if sha in allowed:
                resolved[sha] = f"(unresolvable here) {allowed[sha]}"
                continue
            orphaned.append(sha)
        elif s is None:
            if sha in allowed:
                resolved[sha] = f"(unresolvable here) {allowed[sha]}"
                continue
            unresolved.append(sha)
        else:
            resolved[sha] = s

    pidx0 = {}
    if MARK_BEGIN in itext:
        for line in itext.split(MARK_BEGIN, 1)[1].splitlines():
            m = re.match(r"\|\s*`([^`]+:\d+)`\s*\|\s*([^|]*?)\s*\|\s*`(.*)`\s*\|$", line.strip())
            if m:
                pidx0[m.group(1)] = m.group(2)
    # The index AS IT STANDS, read before --update rewrites it — the only record of what each
    # citation was keyed to before this run, and what the launder check compares against.
    pindex_prior = {}
    if MARK_BEGIN in itext:
        for line in itext.split(MARK_BEGIN, 1)[1].splitlines():
            mm = re.match(r"\|\s*`([^`]+:\d+)`\s*\|\s*([^|]*?)\s*\|\s*`(.*)`\s*\|$", line.strip())
            if mm:
                pindex_prior[mm.group(1)] = mm.group(3).replace(chr(92) + "|", "|")
    presolved, pmissing = {}, []
    for key, rel, ln in paths:
        if rel in pallow or rel.split("/")[0] in pallow:
            continue
        repo, content = resolve_path(rel, ln)
        if content is None:
            if pidx0.get(key) and not path_repo_present(pidx0[key]):
                presolved[key] = (pidx0[key], None)  # foreign repo absent; skipped below
                continue
            pmissing.append((key, rel))
        else:
            presolved[key] = (repo, content)
    sym_bad = []
    for rel_doc, rel, name in symbols:
        if rel in pallow or rel.split("/")[0] in pallow:
            continue
        st, _repo = check_symbol(rel, name)
        if st == "nofile":
            pmissing.append((f"{rel_doc}|{rel}:{name}", rel))
        elif st == "gone":
            sym_bad.append(f"  {rel_doc}|{rel}:{name}  SYMBOL GONE — {rel} declares no `{name}` (renamed, moved to another file, or deleted); "
                           f"name the declaration it became, or pin this record (`citations-at`)")
    if pmissing:
        sys.stderr.write("queue_citation_lint: path citation(s) resolve in NO repository:\n")
        for key, rel in pmissing:
            sys.stderr.write(f"  {key}  — searched: {', '.join(n for n, _ in path_repos())}\n")
        sys.stderr.write("  A path that resolves nowhere is a hard error, not a not-found: either the\n"
                         "  citation names no real file, or it omits the repo it belongs to.\n")
        return 2
    if sym_bad:
        sys.stderr.write("queue_citation_lint: docs/QUEUE.md citations are not sound:\n")
        sys.stderr.write("\n".join(sym_bad) + "\n")
        return 1

    bresolved, bmissing = {}, []
    for rel in bares:
        hit = ""
        for name, base in path_repos():
            if (base / rel).is_file():
                hit = name
                break
        if hit:
            bresolved[rel] = hit
        else:
            bmissing.append(rel)

    if update:
        # REFUSE TO LAUNDER A SHIFTED CITATION.
        #
        # --update rewrites the index from whatever is at the cited line NOW. When that line is no
        # longer discriminating, resolve_path() substitutes the enclosing declaration as an anchor --
        # and the rewrite then records that anchor against the STALE line number. The check passes
        # afterwards (the declaration still exists), so the gate goes green while the prose still
        # cites a line that no longer holds what the sentence claims.
        #
        # That is not hypothetical: `decoder/serialize_test.go:443` cited a specific assertion; an
        # edit moved it to :436; --update re-keyed :443 to `anchor: func
        # TestSerializeWeightsTo_matchesBuffer(` and went green, with the prose still saying :443.
        # It was caught by hand, which is the wrong mechanism.
        #
        # --update CANNOT fix this: the line number lives in the prose, and this tool rewrites only
        # the generated index. So it refuses, names the new line, and makes the author edit the
        # sentence. Weakening a key is a decision, never a side effect of regenerating an index.
        launder, repointed = [], []
        for key in sorted(presolved):
            repo, content = presolved[key]
            oldrec = pindex_prior.get(key)
            if content is None or not oldrec or not discriminating(oldrec):
                continue
            if content[:88] == oldrec:
                continue
            rel = key.split("|", 1)[1].rsplit(":", 1)[0]
            _rn, rbase = resolve_base(rel)
            if rbase is None:
                continue
            at, count = moved_to(rel, oldrec)
            if at is not None:
                # Moved, unchanged, found exactly once: fix the PROSE (so it no longer cites the stale
                # line) and rebuild the index on a second pass below. The refusal this replaces
                # existed only because the prose was out of reach.
                doc = ROOT / key.split("|", 1)[0]
                old = int(key.rsplit(":", 1)[1])
                if doc.is_file() and repoint_citation(doc, rel, old, at):
                    repointed.append(f"  {key} -> :{at}")
                    continue
                launder.append(f"  {key}  SHIFTED to line {at}, but the citation could not be found "
                               f"in {doc} to re-point. Fix it by hand, then re-run --update.")
            elif count > 1:
                launder.append(f"  {key}  AMBIGUOUS — the recorded content `{oldrec[:48]}` now appears "
                               f"{count} times in {rel}; re-pointing would be a guess. Cite the "
                               f"intended line by hand, then re-run --update.")
            else:
                # N-51 (docs/audit-2026-09-10.md): CONTENT-ABSENT, not just shifted. The old
                # content isn't a shift victim — it does not exist ANYWHERE in the current file
                # (deleted, rewritten, or the cited line now runs past EOF, resolve_path's ""
                # case). The SHIFTED branch above is this same refusal for "moved"; this is its
                # missing twin for "gone", and without it --update silently re-keyed the citation
                # to whatever (if anything) now sits at the stale line — exactly the "weakening a
                # key... as a side effect" this whole guard exists to refuse.
                got = content[:48] if content else "(line does not exist in the file)"
                launder.append(f"  {key}  CONTENT GONE — `{oldrec[:48]}` no longer appears "
                               f"anywhere in {rel} (not shifted, deleted or rewritten). --update "
                               f"would re-key this to `{got}` — whatever happens to be there now — "
                               f"instead of refusing. Point the citation at real, current content, "
                               f"then re-run --update.")
        if launder:
            sys.stderr.write("queue_citation_lint --update: refusing to weaken shifted citation(s):\n")
            sys.stderr.write("\n".join(launder) + "\n")
            return 1
        if repointed:
            global _REPOINT_PASSES
            print(f"queue_citation_lint --update: re-pointed {len(repointed)} moved-but-unchanged "
                  f"citation(s) in the prose:")
            print("\n".join(repointed))
            if _REPOINT_PASSES < 3:
                _REPOINT_PASSES += 1
                return _main()  # rebuild the index from the corrected documents
            sys.stderr.write("queue_citation_lint --update: citations still shifting after 3 passes\n")
            return 1
        lines = [MARK_BEGIN, "", "## SHA index", "",
                 "Generated. Every commit id cited above, with the subject it resolved to at the time",
                 "of generation. Regenerate with `scripts/queue_sha_lint.py --update`.", "",
                 "| sha | subject |", "|---|---|"]
        for sha in sorted(resolved):
            if sha in allowed:
                continue
            lines.append(f"| `{sha}` | {resolved[sha].replace('|', '\\|')} |")
        for sha in unresolved:
            lines.append(f"| `{sha}` | **UNRESOLVED** |")
        # CC0.c: with the door shut there is no "Path index": a path:line in an unpinned doc is red, so nothing is keyed against a line
        # number. Symbol citations are checked against the declarations the file has, pinned records at their commit, and neither records
        # anything. The section is still written when the door is OPEN, which only the legacy tests do (the line-index machinery above is
        # unreachable in production now and is the next thing to delete).
        if not CLOSE_THE_DOOR:
            lines += ["", "## Path index", "",
                      "Generated. Every `file:line` cited above, the repo it resolved in, and the trimmed",
                      "content of that line. A line that MOVED is reported with its new number; content that",
                      "has VANISHED is red, because the citation then claims something the file no longer",
                      "supports.", "",
                      "| doc \\| path:line | repo | line content |", "|---|---|---|"]
            for key in sorted(presolved):
                repo, content = presolved[key]
                if content is None:
                    continue
                keyable = discriminating(content) or content.startswith("anchor: ")
                # TRUNCATE FIRST, THEN ESCAPE. The reverse order silently broke every
                # citation whose line contains a pipe -- i.e. every Go boolean
                # condition with `||`. Escaping doubles each pipe's width, so slicing
                # the ESCAPED string to 88 stored fewer than 88 real characters, while
                # the checker compares against `content[:88]` of the raw line. They can
                # never be equal, and the failure reports as "CONTENT ABSENT -- the line
                # now reads X and the recorded content X is nowhere in the file", with X
                # identical on both sides because the message truncates for display.
                # Found 2026-09-01 when four audit citations stayed red across repeated
                # --update runs; three of the four lines were unique in the tree, so an
                # anchor collision could not explain it.
                body_txt = content[:88].replace('|', chr(92) + '|') if keyable else "UNKEYABLE"
                lines.append(f"| `{key}` | {repo} | `{body_txt}` |")
        lines += ["", "## Bare file index", "",
                  "Generated. Every file referenced WITHOUT a line number, and the repo it resolves in.",
                  "Existence only — there is no line to key content against, which is recorded rather",
                  "than papered over.", "",
                  "| file | repo |", "|---|---|"]
        for rel in sorted(bresolved):
            lines.append(f"| `{rel}` | {bresolved[rel]} |")
        for rel in bmissing:
            lines.append(f"| `{rel}` | **RESOLVES NOWHERE** |")
        lines += ["", MARK_END, ""]
        # PRESERVE THE TAIL. See tail_after_index: dropping it deleted three filed queue entries.
        ipath.write_text(
            "# Citation index\n\n"
            "Generated by `scripts/queue_citation_lint.py --update` from every citation in the live docs "
            "(`docs/QUEUE.md` and the rest). Do not edit by hand, and do not cite this file. It moved out of "
            "`docs/QUEUE.md` on 2026-09-24 so the index churn stops colliding with hand-written entries.\n\n"
            + "\n".join(lines))
        if MARK_BEGIN in text:
            # One-time migration of the legacy in-file block. PRESERVE THE TAIL (see
            # tail_after_index: dropping it once deleted three filed queue entries).
            tail = tail_after_index(text)
            out = body.rstrip("\n") + "\n"
            if tail.strip():
                out = out + "\n" + tail.lstrip("\n")
            queue.write_text(out)
        print(f"queue_sha_lint: indexed {len(resolved)} SHAs ({len(unresolved)} unresolved)")
        return 1 if unresolved else 0

    if MARK_END not in itext and queue is not None:
        sys.stderr.write("queue_sha_lint: no citation index (docs/citation-index.md) — run with --update\n")
        return 1

    index = {}
    for line in (itext.split(MARK_BEGIN, 1)[1].splitlines() if MARK_BEGIN in itext else []):
        m = re.match(r"\|\s*`([0-9a-f]{7,40})`\s*\|\s*(.*?)\s*\|$", line.strip())
        if m:
            index[m.group(1)] = m.group(2)

    bad = []
    shifted_ok = []  # moved-but-unchanged, found exactly once: accepted, reported below
    unkeyable = []
    skipped_foreign = []
    for sha in unresolved:
        # The index records a foreign subject as "[repo] subject". If that repo is not checked out
        # here — CI clones goinfer alone — the citation is UNCHECKED, not fabricated. Reporting it as
        # fabricated would misstate the evidence.
        rec = index.get(sha, "")
        m = re.match(r"\[([a-z0-9_-]+)\]", rec)
        if m and not commit_repo_present(m.group(1)):
            skipped_foreign.append((sha, m.group(1)))
            continue
        bad.append(f"  {sha}  DOES NOT RESOLVE — fabricated, mistyped, or from a dropped branch")
    for sha in orphaned:
        bad.append(f"  {sha}  ORPHANED — it resolves in this local object store but is reachable "
                   f"from NO ref.\n      An amended, rebased or deleted-branch commit. It survives "
                   f"here in the reflog until gc, and is a hard failure in any fresh clone, so a "
                   f"green on this machine would not travel.\n      Re-point the citation at the "
                   f"commit that replaced it.")
    for sha, subj in resolved.items():
        if sha in allowed:
            continue  # declared unresolvable, with a stated reason; nothing to compare against
        rec = index.get(sha)
        if rec is None:
            bad.append(f"  {sha}  cited but absent from the index — run --update")
        elif rec.replace("\\|", "|") != subj:
            bad.append(f"  {sha}  subject changed\n      indexed: {rec}\n      actual:  {subj}")
    for sha in index:
        if sha not in resolved and sha not in unresolved and sha not in allowed:
            bad.append(f"  {sha}  indexed but no longer cited — run --update")

    pindex = {}
    for line in (itext.split(MARK_BEGIN, 1)[1].splitlines() if MARK_BEGIN in itext else []):
        m = re.match(r"\|\s*`([^`]+:\d+)`\s*\|\s*([^|]*?)\s*\|\s*`(.*)`\s*\|$", line.strip())
        if m:
            pindex[m.group(1)] = (m.group(2), m.group(3))
    for key, rel, ln in paths:
        if rel in pallow or rel.split("/")[0] in pallow:
            continue
        rec0 = pindex.get(key)
        if rec0 is not None and rec0[0] and not path_repo_present(rec0[0]):
            # The index records which repo this resolved in. If that repo is not checked out here —
            # CI clones goinfer alone — the citation is UNCHECKED, not broken. Counted and named, so
            # a skip and a pass do not look the same.
            skipped_foreign.append((key, rec0[0]))
            continue
        repo, content = presolved[key]
        rec = rec0
        if rec is None:
            bad.append(f"  {key}  cited but absent from the path index — run --update")
            continue
        want = rec[1].replace(chr(92) + "|", "|")
        if want == "UNKEYABLE" or not (discriminating(content) or content.startswith("anchor: ")):
            unkeyable.append(key)
            continue
        if content[:88] == want:
            continue
        if want.startswith("anchor: "):
            # The anchor is a declaration; look for it anywhere in the file before calling it gone.
            decl = want[len("anchor: "):]
            for _n, _b in path_repos():
                _f = _b / rel
                if _f.is_file() and any(l.strip()[:80] == decl for l in _f.read_text(errors="replace").split("\n")):
                    break
            else:
                bad.append(f"  {key}  ANCHOR GONE — the citation pointed at `{decl[:52]}`, which is "
                           f"no longer in the file")
            continue
        # The content moved or vanished. Look for it elsewhere IN THE SAME FILE before calling it red.
        #
        # Search for shifted lines only in the checkout used to resolve this path.
        rname, rbase = resolve_base(rel)
        found_at, found_n = moved_to(rel, want)
        # Evidence in another checkout can explain a mismatch, but gives no line number here.
        elsewhere = []
        if found_at is None and found_n == 0 and want:
            for name, base in path_repos():
                if base == rbase:
                    continue
                f = base / rel
                if f.is_file() and any(l.strip()[:88] == want
                                       for l in f.read_text(errors="replace").split("\n")):
                    elsewhere.append(str(base))
        # No found_at == ln guard: with the search scoped to the resolving base it is UNREACHABLE.
        # Reaching here means the cited line's content differs from `want`; if resolve_path()
        # substituted an anchor it did so because that line is NOT discriminating, and `want` is,
        # so `want` cannot match at that same line. A guard for an impossible state would silently
        # pass whatever future bug produced it -- better that "now at line {ln}" show up as the
        # visible absurdity it would be.
        if found_at:
            # Accepted (see moved_to): the claim still holds; only the number is stale.
            shifted_ok.append(f"  {key} -> :{found_at}")
        elif found_n > 1:
            bad.append(f"  {key}  AMBIGUOUS — the recorded content now appears {found_n} times in "
                       f"{rel}; which one the prose means is a guess. Cite the intended line and re-run "
                       f"--update.")
        elif elsewhere:
            bad.append(f"  {key}  VERSION MISMATCH — the recorded content is NOT in the {rname} copy "
                       f"this repo resolves against, but IS in: {', '.join(elsewhere)}. The citation "
                       f"is keyed to a different version than goinfer requires; re-cite against the "
                       f"required one. NO line number is offered — it would name a line in another "
                       f"file.")
        else:
            bad.append(f"  {key}  CONTENT ABSENT — the line now reads {content[:60]!r}, and the "
                       f"recorded content {want[:60]!r} is nowhere in the file. The citation claims "
                       f"something the file no longer supports.")

    bindex = {}
    for line in (itext.split(MARK_BEGIN, 1)[1].splitlines() if MARK_BEGIN in itext else []):
        m = re.match(r"\|\s*`([^`]+\.(?:go|sh|py))`\s*\|\s*([^|]*?)\s*\|$", line.strip())
        if m:
            bindex[m.group(1)] = m.group(2)
    still = []
    for rel in bmissing:
        rec = bindex.get(rel, "")
        if rec and rec != "**RESOLVES NOWHERE**" and not path_repo_present(rec):
            skipped_foreign.append((rel, rec))
            continue
        still.append(rel)
    for rel in still:
        bad.append(f"  {rel}  bare reference resolves in NO repository "
                   f"({', '.join(n for n, _ in path_repos())}) — moved package, deleted, or a typo")

    if shifted_ok:
        print(f"queue_citation_lint: {len(shifted_ok)} cited line(s) MOVED but are unchanged — accepted; "
              f"`--update` re-points the prose to the new line numbers:")
        print("\n".join(shifted_ok[:20]) + (f"\n  … and {len(shifted_ok) - 20} more" if len(shifted_ok) > 20 else ""))
    if bad:
        sys.stderr.write("queue_citation_lint: docs/QUEUE.md citations are not sound:\n")
        sys.stderr.write("\n".join(bad) + "\n")
        return 1
    if allowed:
        print(f"queue_sha_lint: {len(allowed)} citation(s) DECLARED unresolvable here:")
        for sha, why in sorted(allowed.items()):
            print(f"    {sha}  {why}")
    if unkeyable:
        print(f"queue_citation_lint: {len(unkeyable)} citation(s) UNKEYABLE — the cited line's content "
              f"does not discriminate (a bare `//`, a brace, a blank), so a match would be coincidence:")
        for k in sorted(unkeyable):
            print(f"    {k}")
    if skipped_foreign:
        print(f"queue_citation_lint: SKIPPED {len(skipped_foreign)} citation(s) — the repo they "
              f"resolved in is not checked out here:")
        for k, r in sorted(skipped_foreign):
            print(f"    {k}  (recorded in: {r})")
    ndocs = len({k.split("|", 1)[0] for k, _, _ in paths})
    print(f"queue_citation_lint: {len(symbols)} symbol citation(s) (path.go:Name) across "
          f"{len({d for d, _, _ in symbols})} live document(s), each checked against the declarations the file has now; "
          f"{len(pinned_docs)} PINNED record(s) checked at their commits (citations-at), never against HEAD and never rewritten.")
    # DENOMINATOR, stated every run alongside the numerator. "VALIDATED N" alone cannot distinguish
    # a clean tree from a scanner whose universe shrank — an rglob that stops matching, or a doc that
    # moved under an excluded prefix, both report a smaller green N.
    _all_md = [f for f in ROOT.rglob("*.md") if "/.git/" not in str(f)]
    _buried = [f for f in _all_md if "docs/completed/" in str(f)]
    _internal = [f for f in _all_md if "docs/internal/" in str(f)]
    print(f"queue_citation_lint: EXAMINED {len(live_docs())} live markdown document(s) of "
          f"{len(_all_md)} in the tree ({len(_buried)} excluded as docs/completed/ by the archival "
          f"rule below; {len(_internal)} excluded as docs/internal/, which is gitignored and so "
          f"resolves on this machine only). A document that leaves this denominator leaves the "
          f"lint silently.")
    # SCOPE STATED PRECISELY (N-41, audit-2026-09-02.md, found 2026-09-11): commit-SHA validation
    # reads docs/QUEUE.md ONLY (see `text = QUEUE.read_text()` above) — it does NOT cover the ~51
    # other SHA citations in docs/queue-engineering.md / docs/queue-release.md. Only the path:line
    # check runs across every live document. The old wording ("N commit citation(s), M path:line
    # citation(s) across K live document(s)") let "across K live documents" read as covering both
    # clauses; it covers only the second.
    print(f"queue_citation_lint: VALIDATED {len(resolved) - len(allowed)} commit citation(s) in "
          f"the queue, {len(paths) - len(unkeyable)} path:line citation(s) across {ndocs} live "
          f"document(s), and {len(bresolved)} bare file reference(s) in the queue (existence only).")
    if unkeyable:
        print(f"  {len(unkeyable)} path citation(s) are EXISTENCE-CHECKED ONLY — their line content "
              f"does not discriminate, so a shift in them is invisible.")
    if LOCAL_ONLY:
        print(f"  {len(LOCAL_ONLY)} citation(s) resolve here ONLY VIA A LOCAL REF — they will fail in CI "
              f"until pushed, because CI has only the published refs:")
        for sha, ref in sorted(set(LOCAL_ONLY)):
            print(f"    {sha}  reachable from {ref} — push it, or allowlist it with a reason")
        print("    Not failed here on purpose: citing a commit in the same push that creates it is "
              "normal. CI is the enforcement; this is the advance warning.")
    paths_from_checkout = sum(RESOLVED_FROM.values())
    print(f"  PATH RESOLUTION: {paths_from_checkout} lookup(s) answered from a checkout. Pinned external paths use sealed sources.")
    print("  docs/completed/ is EXCLUDED on purpose — archived records are supposed to age with the "
          "code they described, and linting them would fight the archival rule.")
    print("  NOT validated: RELEVANCE. A commit citation is checked to resolve and to match its "
          "recorded subject; nothing checks that the commit SUPPORTS the claim made about it.")
    print("    9e5f8fa passed green through two documents for weeks as \"the goldens-refresh "
          "precedent\" while touching the manifest not at all — a resolving sha with a matching "
          "subject and a false claim attached.")
    print("    Source citations resolve declarations or historical lines. The surrounding prose still needs review.")
    print("  Also NOT validated: every test name, tag, env var and URL — and for the bare file "
          "references, WHAT IS SAID ABOUT THEM, since there is no line to key against.")
    print("  A lint on a document raises trust in EVERY citation in it, not only the kinds it checks — "
          "so the scope is printed with the green.")

    # DESTINATION RULE. Red, always, no existence test — the gitignore status decides.
    ignored_dests = gitignored_destinations()
    if ignored_dests:
        sys.stderr.write("queue_citation_lint: FORBIDDEN DESTINATION — a committed document cites a "
                         "path under a gitignored directory:\n")
        for doc, path in ignored_dests:
            sys.stderr.write(f"    {doc}  ->  {path}\n")
        sys.stderr.write("  No reader outside this machine can reach it, and an uncommitted target "
                         "has NO HISTORY — nobody can ever establish whether it existed. Either move "
                         "the target to a committed location, or drop the pointer and keep the "
                         "claim's real support (the commit).\n")
        return 1
    else:
        print(f"  GITIGNORED DESTINATIONS: none. A committed document citing a path under a "
              f"gitignored directory is red without an existence test — the gitignore status "
              f"decides, because an uncommitted target has no history to audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
