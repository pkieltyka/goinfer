# CUDA scratch reuse results

The 2026-11-21 [registered experiment](registration.md) completed all 20 cells in five ABBA blocks from 17:00:39 to 17:03:26 EST. The candidate shortened summed document time by about 0.6% and warm median query latency by about 3.4%. It won all ten pairs for each metric. All inputs, vectors, ranking orders, and abstention decisions matched. The document improvement missed the registered 1.05× threshold, so the candidate stays parked.

| Metric | Median paired reference time / candidate time | Pair range | Candidate wins |
| --- | ---: | ---: | ---: |
| Sum of 32 document calls | 1.0060× | 1.0027–1.0073× | 10 / 10 |
| Median of 107 warm query calls | 1.0354× | 1.0326–1.0379× | 10 / 10 |

Both arms used the pinned 139-input pbrain corpus, four Go scheduler threads, the same BF16 Google text weights widened to FP32, and the existing CUDA calculations on the RTX PRO 6000 Blackwell Max-Q Workstation Edition with driver 595.112.07. The one-minute start load ranged from 0.936 to 0.994. The reference was `7242e78b8` (binary SHA-256 `268711a19d252b0f7a1ca8fec1055aa1bbaaa4712d43de9955d453666dc02804`) and the candidate was `fc2de4234` (binary SHA-256 `e4a36c7b3c037c2d219389b1bdb086a04c45e93af747fe255474aac9d6efa8d3`).

The local `~/goinfer-logs/pbe2-rtx-scratch-2026-11-21/paired-cells/summary.json` records paired ratios and the per-cell times. Each numbered cell has separate input, vector, and start-condition evidence. The allocation trace was a separate run. At the maximum corpus length of 2,655 tokens it measured 356,558,320 charged device bytes retained by the candidate.
