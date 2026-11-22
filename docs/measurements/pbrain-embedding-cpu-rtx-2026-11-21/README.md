# EmbeddingGemma 2 corpus on CPU and RTX PRO 6000

On 2026-11-21 the two encoder-only cells ran sequentially on the same Linux host with four Go scheduler threads. Each encoded 32 current notes and 107 labeled queries from pbrain commit `b4def8fad8ea34101c369777604144fbbec2182a`. The two result files have the same corpus manifest SHA-256 `420988ad6653a9141a56ef61f12de91adc07255705426a3042f9a41cf413d732`. All 139 per-input token checksums, roles, IDs, and counts matched. The same top result, ordered top five and corpus-run abstention decision were observed for all 107 queries.

| Encoder metric | Ryzen 9 7950X CPU | RTX PRO 6000 Blackwell Max-Q |
| --- | ---: | ---: |
| Encode all 32 documents, summed call time | 19.435 s | 2.736 s |
| Documents per second, sequential calls | 1.65 | 11.69 |
| Median query latency across 107 inputs | 59.48 ms | 15.75 ms |
| Full driver elapsed time | 26.957 s | 6.076 s |
| Host peak RSS | 2,787,224 KiB | 1,874,600 KiB |
| Sampled peak device memory for the process | — | 1,432 MiB |

The 32 document lengths span 128–2,655 framed tokens; query lengths span 11–36. Calls use the native Go encoder's whole-file text preparation. The measuring driver includes tokenization in the per-input call. CUDA attachment added 290 ms after the 925 ms model load. CPU model load took 908 ms. The CPU run sampled no process swap. This is one starting baseline with sequential cells and different Go patch versions. It excludes pbrain's chunking, index writes, BM25 and hybrid query execution.

## Provenance

- CPU workload source goinfer `3580b186d`, Go 1.27.4. CUDA runtime additions from `91a2ae564`, with equivalent runtime behavior at parity head `331b065b5`, Go 1.27.3. Both linked aikit `v1.79.0` and ran with `CGO_ENABLED=0` and `GOMAXPROCS=4`.
- Both cells used `google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e` under the local Hugging Face cache at `~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/`. The local safetensors SHA-256 is `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`. The BF16 weights were expanded to FP32. The encoder is deterministic and uses no sampling.
- CPU ran 15:28:10–15:28:37 EST, with start one-minute load 0.929 and 93,936,044 KiB available. CPU k10temp control temperature was 42.125 °C at start and 68.375 °C at end.
- CUDA ran 15:28:52–15:28:58 EST on NVIDIA driver 595.112.07. The 31 °C GPU had 97,249 MiB free at start. Host load was 1.43 with 94,053,552 KiB available. The maintainer requested immediate execution on 2026-11-21 to collect the waiting baseline, so the default idle-start cutoff was waived for that cell. The recorded start passed the host fit and heavy-load checks. Both cells held the exclusive timing lock.
