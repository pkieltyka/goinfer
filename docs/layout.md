# Repository layout

A map of the tree. Paths are relative to the repository root; every row was checked against the tree.

## Moved

- `cuda/` -> `backend/cuda/` (`module github.com/pkieltyka/goinfer/cuda` -> `github.com/pkieltyka/goinfer/backend/cuda`)
- `metal/` -> `backend/metal/` (`module github.com/pkieltyka/goinfer/metal` -> `github.com/pkieltyka/goinfer/backend/metal`)
- `gpu/` -> `backend/webgpu/` (`module github.com/pkieltyka/goinfer/gpu` -> `github.com/pkieltyka/goinfer/backend/webgpu`, `package gpu` -> `package webgpu`)
- `internal/{serveapp,chatapp,cliutil,fitcmd,decidecmd,pullcmd,inproc,connect,loadflags}` -> `internal/app/<pkg>` (package clause unchanged)
- `internal/{swapguard,prezerolint,paritykey,movedpaths,servecheck,memgate}` -> `internal/guard/<pkg>` (package clause unchanged)
- `internal/{hwcensus,hwreport,kernelprint,sb2census,towerid}` -> `internal/diag/<pkg>` (package clause unchanged)
- `internal/{giw,mem,kvmgr,f16,mamba2,whisper,scheduler,adapter,hybridattn,program,modelload,prequant,diskfree,sampling}` -> `internal/runtime/<pkg>` (`internal/mem/probe` -> `internal/runtime/mem/probe`; package clause unchanged)

## Modules and the workspace

Four modules are developed together; a separate `go.mod`, `.github/actions/staticcheck/`, only pins the CI linter build
(`go run ./cmd/lint` uses it) and is not part of the workspace.

| Path | Module path | Published |
|---|---|---|
| `.` | `github.com/pkieltyka/goinfer` | tagged `vX.Y.Z` |
| `backend/cuda/` | `.../goinfer/backend/cuda` | tagged `backend/cuda/vX.Y.Z` (formerly `cuda/vX.Y.Z`) |
| `backend/metal/` | `.../goinfer/backend/metal` | tagged `backend/metal/vX.Y.Z` (formerly `metal/vX.Y.Z`) |
| `backend/vulkan/` | `.../goinfer/backend/vulkan` | tagged `backend/vulkan/vX.Y.Z` |

**`go.work` is gitignored and required for cross-module work.** The submodule `go.mod` files carry no `replace`, so a
submodule built without a workspace resolves the root from the proxy at its last published tag (`RELEASING.md`). Create it
with `scripts/make_work.sh` (copies `go.work.example`, which lists all four modules, to `go.work` when absent) and it builds
from a clean clone. Without a `go.work` in the repo, Go walks up and uses one in a parent directory, and a `GOWORK` environment variable overrides both; `make_work.sh` warns about the second and fixes the first.

Untagged, `backend/cuda/` does not build on its own (`backend/cuda/internal/gpudev` needs the `cuda` build tag); see CLAUDE.md for the tags.

## Library packages (root module)

| Path | What it is |
|---|---|
| `decoder/` | the decoder-only transformer runtime: loaders, forward passes, KV cache, fit guards, the `.giw` sidecar |
| `decoder/config/` | checkpoint `Config` JSON loading, family defaults and validation; a leaf importing no goinfer package |
| `decoder/fit/` | the types the fit guard and the placement planner hand their callers (`Plan`, `PlanRequest`, `Placement`, `FitAnswer`, `FitDeclineError`); a leaf over `internal/runtime/mem` |
| `decoder/resident/` | the contract a resident backend implements: the four subsystem interfaces, `SubsystemProvider`, `Forward` (the base every resident implements), the drafter host and the small data their signatures carry; a leaf over `internal/cost` and `internal/runtime/adapter`, aliased from `decoder` as `decoder.Resident*` |
| `decoder/quant/` | safetensors checkpoint quantization config parsing and weight reconstruction (`Reconstructor`, `RowStreamReconstructor` for FP8, GPTQ, AWQ, NVFP4, MLX); a leaf over `internal/modelcore/embed` for `SafetensorsFile` (rule `quant-leaf` in `internal/layout/imports.json`) |
| `engine/` | what the serving shell asks of a model: a generation from a prepared request |
| `server/` | the HTTP API (OpenAI and Anthropic routes, web UI) as a library handler |
| `chat/` | chat templates and tool-call rendering and parsing, no Jinja engine |
| `constrain/` | constrained decoding: JSON Schema and grammar logit masks |
| `tokenizer/` | the BPE tokenizers the models ship |
| `multimodal/` | vision and audio towers and preprocessing |
| `embeddinggemma2/` | the EmbeddingGemma 2 text encoder |
| `pull/` | resumable checkpoint download from HuggingFace |
| `internal/modelcore/` | private CPU tensor kernels, model-file readers, memory mapping, and serving encoder and WordPiece code |
| `internal/` | packages private to the root module, grouped under `internal/app/` (CLI/server apps and wiring), `internal/guard/` (lint/CI/memory/parity guards), `internal/diag/` (hardware census and reports), and `internal/runtime/` (execution, memory, KV, scheduler, model loading, sampling, and runtime kernels), plus top-level domain packages (`audio`, `batchio`, `clef`, `codec`, `confidence`, `config`, `cost`, `decide`, `fidelity`, `layout`, `vision`, `wrappedupload`), and `testgate`, the skip gates tests share (imported by `_test.go` files only); `internal/layout` holds repository-shape checks as tests |

## Tier 3: subpackages of `decoder/`

`decoder/` is one package of ~800 files. Tier 3 splits it one subsystem at a time, `decoder/fit` first (Francis, 2026-11-05;
`docs/releases/v0.25.0-plan.md` §2, ledger row T3-fit in `docs/completed/task-subsystems-2026-10.md`). A subpackage lives
**under** `decoder/`, named for the subsystem.

**The rule for what may import what.** A subpackage never imports `decoder` (that is a cycle: the loader calls it).
`decoder` imports the subpackage and re-exports its public types as aliases (`type Plan = fit.Plan`), so a move changes no
caller until the caller chooses to name the subpackage, and a backend that does so stops depending on the loader (the
`decoderRatchet` in `internal/layout`). Each subpackage has a rule in `internal/layout/imports.json` naming the goinfer
packages it may import; `TestLayout_arrows` enforces it. Aliases are deleted when the last caller has moved.

**The subpackages so far.** `decoder/fit` (the Order below), `decoder/config` (DC1) and `decoder/quant`. `quant` has no Order
of its own: it holds the FP8, GPTQ, AWQ, NVFP4 and MLX config parsing and reconstruction behind `Reconstructor` and
`RowStreamReconstructor`, and `decoder` calls it directly, through `decoder/gptq.go`'s config parsing. The rules are
`fit-leaf` (may import `internal/runtime/mem`), `config-leaf` (standard library), and
`quant-leaf` (may import `internal/modelcore/embed` for safetensors).

**Why `fit` cannot move whole.** The three files (`fitguard.go`, `fitplan.go`, `fitquery.go`) are the loader's front
end, not a leaf. Moving them and reading the compiler's errors names 43 identifiers they need from `decoder`:
`Config`, `Options`, `Model` and `quantMode` (43 uses); `resolveArchitecture`, `ResolveKV`, `kvBytesForCtx`, `buildDemandFor`,
`Admit`, `loadKnob`; the loader's own quantizer (`quantizeWM`, `wmBytes`, `parseQuant`, `mlaProjInt8`, `mambaProjInt8`);
and the checkpoint openers (`openCheckpointMmap`, `OpenGGUFMmap`, `ggufConfig`, `loadConfig`, `mapGIW`, `readGIWHeadReader`).
In the other direction 13 non-test `decoder` files call it (`model.go`, `prefill_budget.go`, `sidecar.go`, `weights.go`,
`metal_fitctx.go`, `residency.go`, ...), and 40 files outside `decoder/` (17 non-test) name its exported API. The decisions are already in `internal/runtime/mem`, which imports no goinfer package.

**Order.**
1. **Types** (done): `Placement`, `PlanRequest`, `Plan`, `FitAnswer`, `FitDeclineError`, `ErrWontFitResident` in `decoder/fit`, aliased
   in `decoder/fitalias.go`. No behavior change; the files outside `decoder/` that name them compile unchanged.
2. **Callers' imports** (done): backends, `internal/app/*`, `modelload` and `memgate` name `fit.X` instead of `decoder.X`.
3. **Price from neutral inputs** (done as far as it can go without a larger move). The checkpoint-byte estimators, the flat KV rate and the
   tensor-name classifiers are in `decoder/fit/weights.go` over tensor names, shapes and `fit.Rates`. The guard's *decisions* were
   never in `decoder`: `fitCheck.check()` hands plain numbers to `mem.HostCheck`, `decideGIWFitAt` to `mem.SolveGIW`, and
   `internal/runtime/mem` imports no goinfer package. What is left in `fitguard.go`, `fitplan.go` and `fitquery.go` is the adapter
   from `Config`, `Options`, `Architecture`, `BuildDemand` and `ResolveKV` to those numbers (`fitCheckFor`, `priceCtxAndKV`,
   `kvPricingFor`, `denseStreamable`, `estimateKVBytes`, `fitRates` and `quantBytesPerElem`, which measure the per-element prices through
   the loader's quantizer, `Model.Plan`, `FitQuery`) plus the stderr line `guardFit` prints. Moving the
   `fitCheck` struct to `fit` would only relocate fields that `mem.HostLoad` already carries, at the price of rewriting about ten
   test files that build it by field name, so it is not planned. The adapter stays in `decoder`: `Config` has moved to
   `decoder/config` (DC1), and moving `Options` and `Architecture` would not free it, since it also needs the registry and the
   backends' demand reports.
4. **The aliases stay.** `decoder.Plan`, `PlanRequest`, `Placement*`, `FitAnswer`, `FitDeclineError` and `ErrWontFitResident` are the library's
   published names (`Model.Plan` returns a `Plan`), and `decoder/fitalias.go` keeps them (Francis, 2026-11-06). `Config` left `decoder` for
   `decoder/config`; `Options` and `Architecture` stay (Francis, 2026-11-06; the record is `docs/completed/task-decoder-config-2026-11.md`).
   Moving `Options` is a new task when a subsystem names a consumer for it; `Architecture` leaves `decoder` as data, through
   `internal/runtime/program`, not as a package move.

## GPU backend modules

| Path | What it is |
|---|---|
| `backend/cuda/` | the CUDA backend (cgo-free, tag `cuda`), `.cu` kernels and `backend/cuda/cmd/{chat,serve}` |
| `backend/metal/` | the Metal backend (tag `metal`), `backend/metal/cmd/{chat,serve}` |
| `backend/vulkan/` | the optional cgo-free Vulkan accelerator for EmbeddingGemma 2 text on Linux amd64 |

## Binaries

| Path | What it is | Module |
|---|---|---|
| `cmd/serve/` | `goinfer-serve`, the server | root |
| `cmd/gate/` | the gate runner (`go run ./cmd/gate`) | root |
| `cmd/lint/` | CI's staticcheck, pinned and verified | root |
| `cmd/hostmem/`, `cmd/prequant/` | prints the memory figure the fit guard uses; builds a `.giw` bundle from a GGUF | root |
| `cmd/claude-guard/` | the Claude Code PreToolUse hook that refuses commands the repo rules forbid | root |
| `demo/chat/` | `goinfer-chat`, the single-shot chat binary (`build-embed.sh` bakes a model in) | root |
| `backend/{cuda,metal}/cmd/{chat,serve}/` | the same two programs built with a backend | the backend module |
| `apps/examples/{confidence,embed,invoice,structured}/` | runnable library examples, each a `main` package with a test | root |

**Where each binary lives, and why** (decided in PR 3 of the Tier 2 layout move; a move would change the build and install
paths it names):

- CUDA and Metal each have backend-specific `cmd/` entrypoints. Root commands use the CPU.
- `demo/chat/` stays. It is a root-module `main` package, and `go install github.com/pkieltyka/goinfer/demo/chat@latest`
  is a documented install path (README, the release template, and the readme-smoke check). Moving it changes that path for
  every installer; if it moves later, the install line changes with it.
- `apps/examples/*` stays. Each example is linked from the docs and tested in place.

## Upstream components

The website, agent demo, SDK harness and WebGPU backend live in the [upstream archive](upstream-archive.md).
The fork still builds the chat CLI, server, web UI and library examples.

## Test assets

| Path | What it is |
|---|---|
| `testdata/` | tiny deterministic checkpoints (committed, see `.gitignore` exceptions), goldens (`*_golden.json[.gz]`), `assets.json` (heavy-test assets), `env_reads.txt`, `gate/`, `decisions/` |
| `decoder/selftestdata/` | embedded copies of three tiny checkpoints for the startup GPU self-test |
| `<pkg>/testdata/` | per-package fixtures (for example `chat/testdata/`) |
| `docs/measurements/`, `docs/completed/` | dated measurement records and archived task docs |

## Documentation

`docs/` holds design docs, `docs/book/` the inference primer, `docs/code-notes/` per-package notes, `docs/measurements/`
records. The queues, task docs, plans and the generated citation index are in a private notes checkout beside the repository (`../goinfer-notes`: `queues/`, `tasks/`, `plans/`; see `docs/README.md`). `changelog.d/` holds unreleased changelog fragments (folded into
`CHANGELOG.md` at release).

## scripts/

About 316 files, referenced by path from CI, skills, the night runner and docs: do not rename or move them. By prefix:

| Prefix or name | What it is |
|---|---|
| `pin_*` (about 160) | regenerate a tiny fixture or a golden from the HuggingFace reference; the file names the fixture |
| `bench_*` | benchmark drivers; `bench_peer.py` is the same-session peer comparison, `bench_compare.sh` goinfer-versus-goinfer |
| `test_*` | tests of the scripts themselves (Python) |
| `touched.py`, `touched_scans.py` | the day loop: build, vet and test what the diff touches |
| `timing_lock.py`, `night.py`, `install-night-agent.sh` | the timing lock and the night queue |
| `hostmem.py`, `mem_darwin.py` | memory: `cmd/hostmem`'s figures for the scripts, and the Mac memory instrument (a served model through two long prompts) |
| `queue_citation_lint.py`, `book_link_lint.py`, `comment_census.py`, `readme_counts_check.py`, `doc_id_collisions.py`, `ci_checks.py` | lints (some run from the pre-push hook, `pre-push.sh`) |
| `task_docs.py`, `doc_review_staleness.py` | task-doc ledgers and archiving (the docs are read from the notes checkout) |
| `webui*` | browser gates for the web UI |
| `s14*`, `s896*`, `g31*`, `q3vlmoe*`, `gemma4*` and other task-named files | one-off graders and probes from a named task |
| `make_work.sh` | creates `go.work` from `go.work.example` |

## Release files

| Path | What it is |
|---|---|
| `RELEASING.md` | the release procedure and the module-tag rules |
| `CHANGELOG.md`, `changelog.d/` | the changelog and its pending fragments |
| `.github/workflows/` | CI (`ci.yml`), coverage, vulnerability, release-asset and weekly workflows |
| `.ken-go.json` | the build configurations (GOOS, GOARCH, tags) that ken's Go tools ask gopls under, and the timing-lock wrapper each run goes through; three configurations compile 98.9% of the Go files, the host default 64% |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md` | licensing |
