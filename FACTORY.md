# goinfer factory brief

## Project

goinfer is a Go inference library, CLI, and HTTP server for running local models on CPU and through CUDA, Metal, and WebGPU. It also includes a served web UI and a Go-built documentation site. Read `AGENTS.md` and `CLAUDE.md` in full before changing the tree. Use `docs/layout.md` to locate modules and `scripts/make_work.sh` to prepare a Go workspace for cross-module work.

The team's focus is pbrain's Markdown indexing and retrieval workload. pbrain uses BM25, EmbeddingGemma 2 from goinfer, and BGE-small from aikit alongside SQLite and sqlite-vec. Keep the Go inference core and use independent reference vectors and retrieval fixtures to check changes. Prioritize EmbeddingGemma 2 on NVIDIA CUDA, profile CPU separately, and investigate AMD support as its own track.

Use the frozen Markdown fixture directly for EmbeddingGemma 2 experiments in goinfer. On 2026-11-21 the maintainer authorized enabling EmbeddingGemma 2 for the isolated pbrain CPU/RTX benchmark with fresh indexes and caches. Use a process-local setting for that run.

goinfer and aikit evolve as independent forks. Review relevant changes in `townsendmerino/goinfer` and `townsendmerino/aikit`, record upstream SHAs and provenance, and bring over focused fixes with fork-specific tests. Use llama.cpp as a reference for algorithms and performance experiments.

## Tasks

The PM owns GitHub Issues on `pkieltyka/goinfer` and project `pkieltyka/4`. Use `todos skill` to learn the CLI, `todos doctor --json` to check access, and `todos board --json` to inspect work. Pass `--repo pkieltyka/goinfer --project pkieltyka/4` whenever the current checkout or saved board selection differs. Pass `--json` on agent operations.

Use this tracker for both goinfer and aikit. Name the affected repository in each issue. The operator may assign direct changes in the sibling aikit checkout when the work needs them. Follow that repository's instructions and keep its branch and PR separate. Link cross-repository dependencies and verify callers against the proposed aikit revision.

The maintainer moves scoped work to Todo. The operator assigns it when its dependencies are clear. Move assigned work to In Progress and leave discoveries without a status until triaged. Record owner, acceptance criteria, branch or commit, verdict, and next step on the issue. Verify uncertain writes before retrying. Link the PR and close the issue after confirming merge. Use close to retire an issue.

Link the project task ID in the issue. The maintainer's task ledger governs landing and closure under `CLAUDE.md`. If the private checkout is unavailable, record any outstanding ledger update on the issue without copying private notes into GitHub.

## Workflow

The operator assigns work and resolves architecture decisions after consulting the architects. Developers use separate worktrees and keep parallel assignments on disjoint paths or coordinate dependencies through the operator. The PM updates the GitHub tracker. The reviewer records findings and verdicts against the exact paths or commit reviewed. Escalate unresolved findings to the operator after three review rounds.

The operator delegates implementation to lead developers and uses architects for design when needed. The operator coordinates, reviews, and integrates the work. Direct implementation by the operator requires explicit direction from the maintainer.

Use the repository's verification rules in `CLAUDE.md`. For GPU work, record device tests and hardware used, or record that no device tests ran. The maintainer decides gates, defaults, guard thresholds, and tags.

## Review

Begin with the reviewer's own review of the working diff or branch. At PR readiness, run Codegenie locally against the committed head and the agreed base from the repository checkout. Check availability with `codegenie --help` and see flags with `codegenie review --help`.

```sh
codegenie review --no-progress --base <base-sha> --head <head-sha>
```

Allow up to 30 minutes. Verify findings against the code, inspect paths the tool skipped, and record tool coverage with the review verdict. Keep the tool output local. For uncommitted changes in a shared checkout, do the manual review first and have the operator commit at the normal integration point.

## Pull requests

Product changes reach `main` through GitHub pull requests on `pkieltyka/goinfer`. Follow the two direct-push exceptions in `CLAUDE.md`. Check GitHub access with `gh auth status` and see CLI options with `gh pr --help`.

- Before pushing, fetch and merge `origin/main` and run the checks in `CLAUDE.md`. Stage explicit paths. Use the branch's named commit for a background push.
- Push the branch and open a PR with `gh pr create --base main --head <branch>`. Subsequent pushes update the PR.
- Immediately run `gh pr merge <number> --repo pkieltyka/goinfer --auto --merge` as a standalone command. Print `BRANCHES READY: #<number> <head-sha>`. If auto-merge is denied, record it in the PR and follow `CLAUDE.md`.
- Read PR state and CI at the next natural checkpoint as directed by `CLAUDE.md`. Use `gh pr view <number> --json state,headRefOid,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup` for state and `gh pr view <number> --comments` for discussion. For inline threads, open the PR's Files changed tab.

## Browser

Use the repository's Chrome DevTools browser gates for web UI changes. Check prerequisites with `node --version` and `google-chrome --version`. See `scripts/webui_app_gate.mjs` for the app gate and follow the timing-lock and test rules in `CLAUDE.md`.

## Notes

Use the maintainer's existing private checkout for queues, task ledgers, and plans as described in `docs/README.md`. From a member worktree, locate it relative to the primary checkout and use the scripts' `--notes` option where available. Search with `rg`.
