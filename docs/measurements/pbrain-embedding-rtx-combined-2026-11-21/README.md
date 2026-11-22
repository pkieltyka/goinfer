# Combined EmbeddingGemma 2 CUDA experiment

The cgo-free Go build at local goinfer commit `92b93c7c7ebcfd08a4eb1a9bf05b246f360c728c` registers an opt-in accelerator named `cuda-eg2-combined` under `cuda,eg2_combined`. It composes adjacent sequence batching, transposed-key attention, and the existing tiled FP32 projections with explicit bias adds. Attention selects the transposed layout when an individual sequence has at least 128 tokens. GEMM selects the short tile only at up to 36 total rows, up to 2,048 outputs and aligned K. Those thresholds are experimental.

On 2026-11-21 the RTX PRO 6000 Blackwell Max-Q Workstation Edition, driver 595.112.07, passed tiny and real-checkpoint tests at both dispatch thresholds and around the 512-token attention window. Tests checked actual tiled and transposed launches and compared each batched hidden output bit for bit with its reference singleton. All 139 prepared ID and vector digests and all 107 retrieval scores and rankings matched in the frozen corpus. The combined build also ran the same corpus on a Linux RTX 5060 Laptop GPU with the same driver and produced the same digests and rankings as the workstation.

## Checkpoint and inputs

The native encoder runs used `google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e` from fast local storage. They widened BF16 weights to FP32 and used deterministic inference. The safetensors SHA-256 is `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`. The workstation read the checkpoint from its local Hugging Face snapshot. The laptop read a digest-verified copy on its local disk. The native harness used Go 1.27.3, `CGO_ENABLED=0` and `GOMAXPROCS=4`. The prepared IDs come from the [frozen Markdown and query corpus](../pbrain-embedding-peer-2026-11-21/README.md).

The workstation's native diagnostic reduced the summed embedding work for the 32 whole notes from 2.737 s to 0.803 s. Both arms ran in the same device window. The reference used one CUDA forward per note. The opt-in arm grouped adjacent notes with independent sequence bounds.

## First laptop diagnostic

The RTX 5060 Laptop GPU had 8,151 MiB total and 7,708 MiB available before the first process. The laptop reported battery power. Each cell used the same model and the same pinned binary as the workstation. Model file hashes were checked on the laptop. One-minute start load ranged from 2.797 to 4.119, so these are diagnostic A/B observations.

| Laptop encoder workload | Reference CUDA | Combined CUDA |
| --- | ---: | ---: |
| Sum of calls for 32 whole notes | 11.511 s | 3.318 s |
| Median of 107 warm singleton queries | 16.003 ms | 20.742 ms |
| Sampled peak VRAM in the 139-input corpus pass | 984 MiB | 1,296 MiB |

The short-query result motivates a separate plugged-in comparison. The combined layout and batch memory fit are demonstrated on this device. The observed workstation short-tile gain is device-specific until the laptop comparison is repeated.

## Plugged-in laptop query check

The first AC-connected pair used the same query binary, checkpoint, and 107 prepared IDs as the earlier laptop run. The platform profile was `performance` and both one-minute start loads were below 1.0. Median warm embedding took 14.295 ms with reference CUDA and 15.034 ms with combined CUDA. All 107 prepared IDs and vectors matched exactly. This pair measures the resident query path. The reverse-order pair timed out waiting for one-minute load to return below the registered 1.0 cap. Treat the result as a single matched observation. The raw results are in local `laptop-query-ac-first-pair/`.

## Workstation resident queries

On 2026-11-21 the RTX PRO 6000 ran five ABBA blocks of the 107 frozen prepared query IDs. Each cell loaded the checkpoint in a new process and timed the resident query forwards. The same executable selected the reference or combined CUDA backend. Every cell matched the ordered token IDs and output vector digests exactly.

| Per-cell query statistic | Reference CUDA | Combined CUDA | Median paired speedup | Pairs won |
| --- | ---: | ---: | ---: | ---: |
| Median | 15.637 ms | 10.482 ms | 1.490× | 10 / 10 |
| p95 | 15.872 ms | 10.733 ms | 1.479× | 10 / 10 |

Each arm column is the median of its ten cell statistics. Speedups are from adjacent A/B or B/A pairs on the same data. Start temperatures were 31–37 °C. The Linux one-minute load at each start was 0.792–0.986. The executable SHA-256 is `c503243067785c5ce92aa8e2dd0a7b84f4487a59a575523a629f5336dee0215b` and the prepared token export SHA-256 is `06287dc21059bf44d585dba53d4b12b1bc4aa97d74fe0eea2ba813fa7fa21e0a`. Local `query-cells/` preserves the launcher provenance, starts, outputs and summary.

## pbrain application on the workstation

The pbrain CLI comparison used pbrain `1f342f226b8535a265534724a6da6e9ac952f9e7`, goinfer `92b93c7c7ebcfd08a4eb1a9bf05b246f360c728c` and the digest-verified local pack derived from the same Google BF16 checkpoint. Both built with Go 1.27.3 and `CGO_ENABLED=0`. The singleton binary SHA-256 is `c1779bb18bad95f74709a48a32c73679bd50476384469ed8d61783a7ba7a04e6`, and the combined binary SHA-256 is `a7e90588e2d4da538d26d7e6d169a509c66b8534d6354ee4b8bfe4b6ee49208a`. The binary build metadata reports an untracked developer-tool directory in the pbrain worktree. The tracked source matched the recorded commit. Index and vector caches were isolated per run.

A diagnostic pair on the RTX PRO 6000 indexed the same 32 Markdown notes into fresh SQLite databases. Each arm processed all 59 passages and 9,271 prepared tokens, with zero reuse. The reference issued 59 forwards and took 2.716 s wall and 1.655 s in embedding. Combined issued 30 forwards, including 29 pairs, and took 1.645 s wall and 0.680 s in embedding. The one-minute start loads were 0.828 and 1.083 respectively. Both SQLite databases passed integrity checks and matched every prepared-ID and vector digest in order as well as the stored passage metadata and vectors. Local `app-diagnostic/` contains the cells and build hashes.

A follow-up of two ABBA blocks used a fresh index and an empty vector cache in every cell. Start load ranged from 0.939 to 0.992. All eight cells reproduced the exact ordered passage IDs, vector digests and SQLite rows. The combined build won all four adjacent matched pairs, with a median wall speedup of 1.577×.

| Median across four fresh index cells per arm | Reference CUDA | Combined CUDA |
| --- | ---: | ---: |
| Full index process wall time | 2.698 s | 1.707 s |
| Sum of embedding calls | 1.653 s | 0.678 s |
| CUDA forwards | 59 | 30 |

Local `app-cells/` records each cell's trace, SQLite output, start load and comparison summary. The two arms used the same driver, pack and binaries described above.

Three labeled questions also ran on each resulting index, in alternating arm order and with a new pbrain process per question. Results matched in order, returned the expected semantic hits, and produced the same query vector digests. Median wall time was 965.3 ms for the reference and 965.5 ms for combined. The median embedding phase fell from 16.18 to 10.96 ms. Weight verification occupied about 0.20 s, and opening the model occupied about 0.49 s, including CUDA attach at about 0.41 s. One-minute load ranged from 1.083 to 1.396. The OS page cache could be warm after indexing. Background work overlaps the scan, and CUDA attach is within model open, so phase durations should be read individually. Local `one-shot/` holds the traces and results.

The reviewed follow-up pbrain build at `4f2ebab6209a431d5a5f6cc81607d9303e131515` aligned phase clocks, tightened batch trace validation, and made the benchmark runner reject reused output directories. The RTX PRO 6000 live check reproduced all 59 prepared-ID and SQLite vector digests and all three exact query vectors and complete search results from the original runs. Median wall time across those three searches was 965.4 ms in both arms. This was a correctness replay under ordinary host load. The exact binary hashes are in local `app-reviewed-check/binary-digests.json` and the phase traces are in `one-shot-reviewed-check/`.

## pbrain application on the laptop

The same two pbrain binaries indexed the same 32 Markdown notes and 59 passages and searched the same three labeled questions on the RTX 5060 Laptop GPU. Both arms passed the SQLite check. Every stored vector and each query vector and result order matched the workstation. The laptop reported AC disconnected and a low-power platform profile. The one-minute load at readiness was 6.310 for the singleton arm and 6.631 for combined. GPU temperatures at readiness were 43 °C and 48 °C respectively.

| Laptop application diagnostic | Reference CUDA | Combined CUDA |
| --- | ---: | ---: |
| Full fresh index process | 6.595 s | 4.084 s |
| Sum of embedding calls | 4.445 s | 1.967 s |
| Sampled process peak VRAM | 712 MiB | 760 MiB |
| CUDA forwards | 59 | 30 |

These are the correctness passes from the portable recipe. The peak is a sampled lower bound. A separate transfer containing just the recipe, manifests and Python helpers also passed the combined device check using the same binaries and model pack. The index comparison was battery-powered. Local `laptop-pbrain-records/` holds the readiness, correctness and process evidence.

## Remaining workstation query cost

A separate CUDA event diagnostic used four query IDs from the frozen prepared-ID export. It preallocated events, marked a whole stream interval and individual launches in one sliding layer, one full layer, and the opening and final projections. Events on adjacent empty markers took about 0.74 µs in the calibration. Three unmarked and three marked runs per query produced the same vectors. Layers outside the marked subset contribute to the stream interval and host launch totals. Stream intervals can include idle gaps, and host submission overlaps device work.

| Query tokens | Median unmarked wall | Median marked wall | Median marked stream interval | Median host bias-add submission time |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 10.136 ms | 10.856 ms | 10.351 ms | 1.319 ms |
| 22 | 10.535 ms | 11.132 ms | 10.391 ms | 1.339 ms |
| 24 | 10.585 ms | 11.093 ms | 10.419 ms | 1.238 ms |
| 36 | 11.020 ms | 11.883 ms | 10.855 ms | 1.357 ms |

Each short query issued 217 tiled GEMMs, 217 bias adds, one wide reference GEMM, 24 attention launches, 194 normalization launches, and 241 elementwise launches. At 24 tokens, in-stream tiled GEMMs accounted for about 0.359 ms in the sampled sliding layer and 0.377 ms in the sampled full layer. Their bias adds accounted for about 0.041 and 0.031 ms respectively. The next focused kernel experiment can retain the winning tile and fold the explicit bias operation into its store. Its signed-zero behavior and raw FP32 output need the same exactness checks as the current two-kernel path. Host submission and device intervals are separate observations.

Local exactness, source, profiler output, and desktop/laptop artifact records are under `~/goinfer-logs/pbe6-rtx-combined-2026-11-21`.
