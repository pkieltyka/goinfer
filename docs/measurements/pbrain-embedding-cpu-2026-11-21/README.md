# EmbeddingGemma 2 CPU baseline for pbrain

A descriptive single-process baseline for PBE1 on 2026-11-21. The workload isolates goinfer's native text encoder. It excludes pbrain chunking, SQLite indexing, BM25, and hybrid retrieval.

| Workload | Wall time | Go heap allocations |
| --- | ---: | ---: |
| Load with the checkpoint files already in the OS page cache | 942 ms | 2.39 GB |
| First query, 15 tokens | 77.0 ms | 20.0 MB |
| Warm query, 15 tokens, median of five | 72.1 ms | 20.0 MB |
| Four documents, 9/398/496/660 tokens, median of three | 2.50 s (1.60 documents/s) | 1.98 GB |
| One document with prepared 2,048-token IDs | 4.18 s | 2.58 GB |

Allocation columns are deltas of Go's cumulative allocated bytes, rounded from the aligned median observations. Process `VmHWM` was 2,474,064 KiB. `VmSwap` was zero at exit. The sampled query and four document token sequences matched exactly. Their embeddings and the prepared long-input embedding passed the independent cosine ≥ 0.9999 and unit-norm checks.

## Provenance

- AMD Ryzen 9 7950X, Linux amd64, Go 1.27.4, `GOMAXPROCS=4`. goinfer `e33aa3893e51c15af4a5c5c256d40d728bb1b2f0`, aikit `v1.79.0`. The static binary used the native CPU path and no device backend.
- `google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e`. BF16 language-model tensors loaded into FP32. Local Btrfs Hugging Face snapshot under `~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/`. Weights SHA-256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`.
- `internal/embed/gemma2/testdata/reference.json` from pbrain `b4def8fad8ea34101c369777604144fbbec2182a`, unchanged at the run's checkout `b4f8aa1ec5fd389d93f6b64e9bfccb5604b0dd5e`. Fixture SHA-256 `43e1ccf6137f9eea80cc9fe4ed4e5acfbed15d1727a83c8630c35e30fe43a28c`. Inference is deterministic and uses no sampling.
- Exclusive project timing lock, local time 14:52:17–14:52:32 EST. One-minute load 0.93 and Linux `MemAvailable` 96,579,500 KiB at start. The CPU's k10temp control reading was 59.75 °C at start. No temperature trace was collected.

## Reproduction

Use zero-based rows from the pinned JSON. Load once with `embeddinggemma2.LoadEncoder`. Take the first query measurement with `EmbedText` on row 18 using its role, validate the first vector, then repeat the same query five times. For each of three observations, pass documents 9, 4, 3, and 0 in this order to `EncodeTasks` with the document role. Measure the prepared long input once through `encoder.Model().Embed` using row 10's `prepared_ids` and compare with its `prepared_vector`. This exposes the operation pbrain currently calls while excluding the adapter's preparation time. Wrap the prebuilt driver with the exclusive timing lock. Use a fit check, keep the file cache warm, and record the one-minute load, available memory, and thermal readings at the start.

This is an initial baseline. Collect profiles and more samples for any candidate comparison. The roughly 2 GB of allocation traffic in the document workload makes allocation profiling a direct next check. CUDA, AMD text, Metal, and peer timing require separate instrumented runs.
