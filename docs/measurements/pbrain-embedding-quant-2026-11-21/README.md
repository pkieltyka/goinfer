# EmbeddingGemma 2 GGUF retrieval evaluation on RTX

The 2026-11-21 quality sweep ran the direct llama.cpp b12613 CUDA peer on the RTX PRO 6000 Blackwell Max-Q Workstation Edition with driver 595.112.07. It encoded the same 32 frozen Markdown notes and 107 labeled queries from the [matched peer run](../pbrain-embedding-peer-2026-11-21/README.md) for each weight format. Each variant supplied its own document and query vectors. The evaluation kept the prepared IDs, pooling, labels, and corpus cutoff fixed. The F32 control came from the pinned Google snapshot. The Unsloth files matched the sizes and hashes in the [pinned model inventory](../../../specs/research/embeddinggemma2-inference/weight-quantization.md). Their exact upstream Google source revision is unverified.

| Weight file | File size | Sampled peak RTX memory | Recall@5 | MRR | Top result agrees with BF16 | Misses | False accepts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same-source F32 | 1,099.8 MB | 1,554 MiB | 100% | 0.909 | 107 / 107 | 5 | 9 |
| Unsloth BF16 | 558.0 MB | 1,318 MiB | 100% | 0.909 | 107 / 107 | 5 | 9 |
| Unsloth Q8_0 | 309.9 MB | 1,204 MiB | 100% | 0.898 | 105 / 107 | 5 | 10 |
| Unsloth UD-Q6_K_XL | 248.8 MB | 1,176 MiB | 100% | 0.913 | 106 / 107 | 4 | 8 |
| Unsloth UD-Q4_K_XL | 175.7 MB | 1,134 MiB | 97.9% | 0.884 | 93 / 107 | 7 | 8 |

There are 47 answerable questions and 60 unanswerable questions. Recall and MRR use the raw ranking of answerable questions. A miss counts an answerable question that abstained at the frozen cutoff or had no expected result in the top five. A false accept counts an unanswerable question that passed the cutoff. Top-result agreement uses all 107 queries before applying the cutoff. Q6 preserved every labeled top-five hit and Q4 lost one. The differences cover this corpus and cutoff.

The quality record is `~/goinfer-logs/pbe4-rtx-quant-2026-11-21/quality/quality.json`, with per-variant ranked outputs and vector archives beside it. The sampled memory readings are maxima from eight or nine process samples in the quality/fit sweep. The registered timing run records latency, throughput, and sampled memory separately. The quality run evaluates the frozen whole-note encoder workload. Application indexing and chunking use the [pbrain index fixture](../pbrain-embedding-index-2026-11-21/README.md).

## RTX singleton timing

The registered run completed all 16 cells on the same RTX with the pinned direct llama.cpp b12613 probe. Each role used a mirrored order, BF16, Q8, Q6, Q4, then Q4, Q6, Q8, BF16. There were no void cells. Cells used four host threads, one sequence per request, and the same prepared IDs with a resident model. Each timed call includes the synchronized forward and pooling. The two-cell ranges below describe this run.

| Unsloth weight file | Median query latency by cell | Documents/s by cell | Sampled peak query / document memory |
| --- | ---: | ---: | ---: |
| BF16 | 3.942–3.947 ms | 80.86–81.53 | 962 / 1,248 MiB |
| Q8_0 | 3.659–3.666 ms | 92.75–92.95 | 846 / 1,202 MiB |
| UD-Q6_K_XL | 3.751–3.775 ms | 89.07–89.17 | 818 / 1,176 MiB |
| UD-Q4_K_XL | 3.505–3.508 ms | 92.73–93.51 | 764 / 1,134 MiB |

Q6's whole-document throughput improved with the corpus's longest inputs, while the median paired individual document was about 1% slower than BF16. Q8 gave almost the same aggregate throughput as Q4, with 105 versus 93 top-result agreements out of 107. Q6 was smaller than Q8 while preserving all labeled top-five hits and 106 top-result agreements. These are properties of this model output and runtime on this device. The GGUF experiment used llama.cpp; the Go prototype measured separately uses the pinned Google weights.

The timing records and exact build/model provenance are in `~/goinfer-logs/pbe4-rtx-quant-2026-11-21/speed/results.jsonl`. The loader's reported artifact types include FP32 auxiliary tensors in all four formats. The artifact names describe their main weight formats.
