# EmbeddingGemma 2 RTX forward profile

A 2026-11-21 diagnostic on the RTX PRO 6000 Blackwell Max-Q Workstation Edition with driver 595.112.07 profiled three prepared singleton inputs from the [matched native workload](../pbrain-embedding-peer-2026-11-21/README.md). Each shape had one warmup and three alternating uninstrumented and instrumented resident forwards. All instrumented projected hidden rows matched warmup bit for bit.

| Frozen input | Tokens | Uninstrumented median | Instrumented median | Attention class median | GEMM class median | Device copy class median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `u001` | 24 | 15.66 ms | 21.33 ms | 0.72 ms | 14.29 ms | 1.14 ms |
| `index-refresh` | 148 | 19.97 ms | 25.21 ms | 3.49 ms | 14.34 ms | 1.03 ms |
| `tokenizer-report` | 2,655 | 844.86 ms | 853.59 ms | 787.06 ms | 28.38 ms | 2.03 ms |

The 2,655-token input spent about 92% of instrumented wall time in the attention class. The three inputs over 2,000 tokens together accounted for about 80% of goinfer's summed document time in each matched RTX peer cell. This supports prototyping a long-input attention kernel first. The short-query profile points to GEMM and dispatch costs as a separate optimization lane.

The experiment used the corrected encoder at `44c5aa2ab3f76ec120307817534bad2b3b21d9e3`, with local diagnostic-only hooks, Go 1.27.3, four Go scheduler threads, the same derived FP32 calculations, and the Google checkpoint revision `f9972ac22e696a105c482202646979e33e4b5a1e`. Start load was 1.98, available host memory was 93,703,560 KiB, and the device had 97,249 MiB free. It held the exclusive timing lock.

The existing CUDA tower profiler drains the stream when the operation class changes. Each class includes host submission and synchronization, so the numbers are diagnostic wall intervals. The overhead is particularly visible for short queries. The run adds a class for strided device copies. The source diff, test, binary digest and raw observations are in `~/goinfer-logs/pbe2-rtx-profile-2026-11-21`.

## Transposed-key correctness and fit

A separate diagnostic prototype transposes each layer's keys so adjacent score threads read adjacent memory. It reads Q and V directly from their original grouped-head buffers, preserves each input's full and sliding ranges, and keeps the FP32 dot products, FP64 softmax reduction, and value accumulation order. The prototype uses existing per-forward scratch for the transpose. It compiles to PTX with NVRTC 12.9.86 and runs in a cgo-free Go build.

At 17:34 EST the RTX parity test passed every intermediate and projected row bit for bit from the alignment tails through the 2,655-token input. It also passed final-row parity on the three complete documents of 2,042, 2,372 and 2,655 tokens. The longest input took 843 ms on the packed kernel and 189 ms on the transposed kernel in the diagnostic forward without layer readbacks. Both measurements are single observations in `attention-parity.log`.

In an additional full-corpus fit check, the candidate dispatched the new kernel only on the three documents over 2,000 tokens. All 139 token and vector digests matched and all 107 query rankings, scores and abstention decisions were exact. The diagnostic document-call sum was 2.695 s for the packed reference and 1.040 s for the candidate. The registered repeated A/B uses the same qualification rule and same binary. The fit check establishes that the candidate and exact-output gate are ready for that comparison.

## Matched timing result

The [registered](registration.md) RTX run completed all 20 cells from 17:43:29 to 17:53:45 EST. Start loads ranged from 0.935 to 0.999. The packed arm's median sum for 32 documents was 2.695 s, and the candidate's was 1.038 s. The median paired speed ratio was **2.596×**, with individual ratios from 2.567× to 2.609× and all ten pairs favoring the candidate. All 139 output vector digests and all 107 rankings matched. The warm query medians were 15.727 ms and 15.739 ms, consistent with both arms using packed attention on queries.

The timing binary SHA-256 was `c758ccf3e9726ea16de3823968da2e7f1abb713b5cd18bcaee92e1dba847c2cd`. The matching per-input results, host observations, and summary are under `~/goinfer-logs/pbe2-rtx-profile-2026-11-21/attention-cells/`. The exact source snapshot is in the sibling `benchmark-source/` directory. The source-reviewed prototype is on local commit `5aa2349dc`. That commit adds guarded optional tests and error-path draining atop the timed successful path. The RTX subsequently passed the real-checkpoint parity test and a planted enqueue-failure test covering buffer retention and Close retry.
