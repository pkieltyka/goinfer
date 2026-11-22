# EmbeddingGemma 2 matched native and llama.cpp results

On 2026-11-21, this local in-process comparison used one sequence per forward with the same prepared IDs and the same Google BF16 text checkpoint widened to FP32. The 107 frozen queries and 32 current Markdown documents came from the same input manifest as the [earlier native baseline](../pbrain-embedding-cpu-rtx-2026-11-21/README.md). Each role on each backend ran four interleaved cells with zero void cells. All cells used four host threads and a resident model. The timed boundary ran from prepared IDs through the synchronized forward, per-sequence pooling, and L2 normalization. Model loading and untimed warmup preceded it.

| Backend | Metric | goinfer cell range | llama.cpp cell range |
| --- | --- | ---: | ---: |
| Ryzen 9 7950X | Median query latency | 59.22–60.84 ms | 24.16–24.25 ms |
| Ryzen 9 7950X | Document throughput | 1.66–1.69 documents/s | 2.27 documents/s |
| NVIDIA RTX PRO 6000 Blackwell Max-Q | Median query latency | 15.24–15.26 ms | 3.73–3.74 ms |
| NVIDIA RTX PRO 6000 Blackwell Max-Q | Document throughput | 11.93 documents/s | 82.39–82.66 documents/s |

Document throughput uses the sum across all 32 document intervals in each cell. On RTX the longest documents make the throughput gap larger than the 3.51× single-document paired median speed ratio. RTX per-query paired medians favored llama.cpp by 4.08–4.10×. The matching CPU ratios were 1.90–1.94× for documents and 2.35–2.40× for queries. Engine revisions and results are independently pinned per backend.

Native used core `6bea857d695b2291bfcf8e006080631784dabbd8` and, for RTX, CUDA `b9af8d68f264f69f57b3167599ef1ee3cdf6f9cf`. llama.cpp used CPU `e6d624b9a4b95c6b52f5ca384d70b75581f9821a` and RTX `362ac929749d56e433878585a8506f24968698b4` with driver 595.112.07, flash attention disabled, and TF32 disabled. The same-source FP32 GGUF SHA-256 was `1e61453cc7a9a5583dbd27b2d29a2347574814063347945d6d8e5d3ecbe58dfd`. Longest-shape native and llama.cpp vector parity reached cosine 0.999999982 on CPU and 0.999999999990 on RTX.

The comparison ran through `scripts/bench_peer.py --eg2 run-registered` on local commit `2ac439cb63757c85879e87bb03ea61e5f492e35a`. The exact executable digests, prepared workload digest, fit checks, per-input intervals, and paired summaries are in `~/goinfer-logs/pbe1-llamacpp-2026-11-21/registered-compare/{cpu,rtx}/results.jsonl`.
