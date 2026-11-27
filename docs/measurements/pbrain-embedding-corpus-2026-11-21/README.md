# PBE1 shared Markdown corpus CPU baseline

The workload measures EmbeddingGemma 2 text embeddings and whole-document
cosine retrieval over the pbrain shared fixture. The Markdown input includes
front matter and body verbatim. Document and query roles select the checkpoint
prompts. The native encoder applies its ordinary BOS/EOS framing and runs in
FP32. File I/O for the fixture, token verification, vector validation, cosine
scoring, and JSON output are outside each embedding duration.

## Frozen inputs and scoring

[manifest.json](manifest.json) pins the pbrain revision, Markdown byte hashes,
stable item IDs, labeled question file, and checkpoint files. The source is
the retrieval fixture in `pkieltyka/pbrain` at
`b4def8fad8ea34101c369777604144fbbec2182a`. Its item-level attribution and
all question labels are preserved in the pinned source. The source MIT license
travels with this record. [LICENSE](LICENSE)

The manifest records all 35 shared Markdown files. The current view embeds the
32 files left after the three `replaces` targets are removed. The 107 queries
are the 47 answerable and two 30-query unanswerable halves. The eight
personal-only questions remain shared-view negatives. Corpus materialization
reads only the shared files, labels, and license. The fixture clock is
2026-11-21T18:00:00Z. This encoder-only scorer has no clock-dependent behavior.

The released pbrain calibrator requires the personal-overlay replay and an
EmbeddingGemma 2 derived pack. This harness uses the pinned Hugging Face text
snapshot and implements whole-document cosine ranking. It omits pbrain's
chunk preparation, SQLite, BM25, hybrid scoring, and evidence delivery.

Sort by descending cosine and ascending item ID for ties. Report the
unfiltered first expected rank, per-query recall at 1 and 5 across all labeled
expected IDs, and the raw five highest scores. The optional-result decision
abstains when the highest cosine is below 0.72, the existing cutoff recorded
by pbrain in its calibration README at the source revision. It is a descriptive
read on this direct retrieval workload. The probe does not tune a threshold
or establish a gate.

The first timed call is `a001`. Then embed the document list in manifest order,
followed by the other queries in source order. Each input is timed once using
`Encoder.EmbedText`. `wall_ns` is elapsed wall time. `allocated_bytes` is the
change in Go's cumulative allocated bytes for the operation. The preflight and
CPU report carry each ID, role, token count and SHA-256 of framed uint32
little-endian token IDs. The report carries the SHA-256 of each little-endian
FP32 output vector, plus query ranks and scores. The [matched CPU and RTX
baseline](../pbrain-embedding-cpu-rtx-2026-11-21/README.md) reuses this exact
manifest, input order, prompts, token digests, ranking, and timing units.
The digests identify exact equality. Compare numeric outputs separately when
backends round differently.

The prior six-vector CPU smoke in [the 2026-11-21 initial native CPU
baseline](../pbrain-embedding-cpu-2026-11-21/README.md) supplies the
independent vector checks and prior timing/fit evidence. It used goinfer
`e33aa3893e51c15af4a5c5c256d40d728bb1b2f0` and the same checkpoint. The smoke
workload is separate from the corpus workload.

## Native CPU result

One corpus cell completed on 2026-11-21 at 15:28:10–15:28:37 EST in
26.96 seconds under the exclusive timing lock. All 139 inputs matched the
preflight token counts and digests. All embeddings had the expected 768
components and passed the unit-norm check. [Preflight](preflight.json)
[Per-input timings and query results](cpu.json) [Host record](host.json)

| Operation | Framed tokens | Time | Go allocation traffic |
| --- | ---: | ---: | ---: |
| Load checkpoint and tokenizer | | 908 ms | 2.394 GB |
| First query `a001` | 30 | 68.2 ms | 39.4 MB |
| 106 subsequent queries | 11–36 | 59.2 ms median, 6.304 s summed | 29.2 MB median, 2.956 GB summed |
| 32 whole documents | 128–2,655 | 19.435 s summed, 1.65 documents/s | 14.332 GB summed |
| 29 decisions, learnings and notes | 128–177 | 6.083 s summed, 4.77 documents/s | 5.428 GB summed |
| `discovery-field-guide` | 2,042 | 3.770 s | 2.573 GB |
| `storage-field-guide` | 2,372 | 4.467 s | 2.988 GB |
| `tokenizer-report` | 2,655 | 5.115 s | 3.343 GB |

Times for groups sum the per-input intervals. Rates divide the count by that
sum. Documents ran as consecutive single inputs. GB and MB are decimal units.
Allocation traffic counts bytes allocated over time. Kernel peak process RSS
was 2,787,224 KiB and sampled peak `VmSwap` was zero across 268 observations.

For the 47 answerable questions, an expected item ranked first for 40 and in
the top five for all 47. Mean recall@1 was 0.8298, mean recall@5 was 1.0000,
and mean reciprocal rank was 0.9085. Recall divides the expected IDs in the
top k by the number of labeled IDs for that question. Reciprocal rank uses
the first expected ID. Both measures average across answerable questions.
All six long-document questions reached their expected item in the top five.
For `a001`, `index-refresh` ranked first at cosine 0.733792.

At the existing 0.72 cutoff, the probe abstained on 24/30 calibration and
27/30 held-out unanswerables, including all eight personal-only controls.
Calibration false accepts were `u001`, `u003`, `u005`, `u006`, `u010`, and
`u012`. Held-out false accepts were `u032`, `u033`, and `u039`. All nine were
labeled near misses and `u001` was marked as a regression. Five answerable
questions also abstained (`a031`, `a035`, `a038`, `a040`, `a041`). These results
describe whole-document retrieval at the recorded cutoff.

- Hardware and software were an AMD Ryzen 9 7950X (16 physical cores, 32
  logical CPUs), Linux 7.3.4 amd64, Go 1.27.4, `CGO_ENABLED=0`, `GOMAXPROCS=4`,
  and aikit `v1.79.0`. CPU frequency scaling used `amd-pstate-epp` with the
  `powersave` governor. The CPU arm used no device backend.
- The driver source for the archived 2026-11-21 CPU record was pinned at
  historical goinfer `3580b186ddfde2aeb03929c722c24a20d6b100c4` (binary SHA-256
  `1ec8dfda3e0b9727eeb5a031e006f824f5eeb5dc72b910fe6acedacae4f5e80d`). On
  integration over `1a2604c10`, `testdata/main.go`, `testdata/main_test.go`,
  and `run_linux.py` were revalidated with `CGO_ENABLED=0` and produced
  byte-identical `preflight.json` against the pinned local snapshot; the
  historical `cpu.json`, `host.json`, and `preflight.json` records remain
  unchanged. Compared with the archived binary, the integrated runner
  additionally re-verifies all `manifest.model_files` off-clock before
  `LoadEncoder` (recording `model_verify_wall_ns` separately) and `run_linux.py`
  terminates and waits for its child process group on `SIGTERM`/`SIGHUP`/`SIGINT`
  so the outer timing lock is never released while a corpus child still runs.
  The model, fixture and interpretation pins are in the manifest.
  Inference is deterministic and uses no sampling.
- The verified snapshot was on local NVMe-backed Btrfs at the Hugging Face
  cache's `hub/models--google--embeddinggemma-2/snapshots/` path. For the
  archived 2026-11-21 run, preflight streamed the pinned files before the idle
  wait and launch-time page-cache residency was unmeasured. For future reruns
  with the integrated runner, the off-clock `verifyModel` pass immediately
  before `LoadEncoder` streams every pinned checkpoint file inside the scoring
  process, so future `load_wall_ns` readings reflect a warm OS page cache
  rather than cold-disk I/O.
- Start load was 0.929 and `MemAvailable` was 93,936,044 KiB. The CPU's
  `k10temp` Tctl reading was 42.125 °C at start and 68.375 °C at exit.
  The recorder collected endpoint temperatures. The waiter launched one
  valid CPU cell once the initial load condition cleared.

## Reproduce on Linux

Run from the goinfer root. Set `pbrain` to a local checkout containing the source
commit and `model` to the pinned snapshot on local disk. The following example
uses a new durable run directory. The preflight verifies all pinned files and
parses every input without a model forward, warming the checkpoint page cache.

```sh
set -euo pipefail
campaign="$PWD/docs/measurements/pbrain-embedding-corpus-2026-11-21"
pbrain=/path/to/pbrain
rev=b4def8fad8ea34101c369777604144fbbec2182a
prefix=internal/semantic/calibrate/testdata/retrieval
model="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e"
work=$(mktemp -d "${TMPDIR:-/tmp}/pbe1-corpus.XXXXXX")
mkdir "$work/fixture"
git -C "$pbrain" archive "$rev" LICENSE "$prefix/brain" "$prefix/questions.json" | tar -x -C "$work/fixture"
python3 scripts/timing_lock.py run --shared --label pbe1-corpus-build -- \
  env CGO_ENABLED=0 go build -o "$work/corpus" "$campaign/testdata/main.go"
python3 scripts/timing_lock.py run --shared --label pbe1-corpus-prepare -- \
  env GOMAXPROCS=4 "$work/corpus" -manifest "$campaign/manifest.json" \
  -fixture "$work/fixture" -model "$model" > "$work/preflight.json"
go version -m "$work/corpus" > "$work/build.txt"
```

Inspect the preflight token lengths and fit before launch. The reference run
used a conservative 8 GiB fit estimate. The largest framed input is 2,655
tokens, within the native limit. The prior 2,048-token observation recorded
2.58 GB of cumulative forward allocation and the load allocated 2.39 GB.
Scaling the former by the length ratio, adding the load and the entire 1.49 GB
checkpoint mapping, and rounding up gives room for the checkpoint and forward
working set. This is a fit estimate for the single-sequence workflow.

The Linux recorder samples load, available memory, temperature, and process
swap, and captures the child's kernel peak RSS. It checks the existing Linux
1.0 idle load limit and the repository heavy-load rule immediately before
launch. Set the physical core count for the actual host. Announce the launch
and estimated finish, then run the prebuilt binary under the exclusive lock.

```sh
env GOMAXPROCS=4 python3 scripts/timing_lock.py run --label pbe1-corpus-cpu -- \
  python3 "$campaign/run_linux.py" --output "$work/cpu-cell" --fit-gib 8 --physical-cores 16 -- \
  "$work/corpus" -manifest "$campaign/manifest.json" -fixture "$work/fixture" \
  -model "$model" -preflight "$work/preflight.json"
```
