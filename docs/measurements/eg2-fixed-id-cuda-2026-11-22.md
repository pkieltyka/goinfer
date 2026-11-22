# EmbeddingGemma 2 CUDA fixed-ID attention profile

This profile addresses [goinfer #30](https://github.com/pkieltyka/goinfer/issues/30).
It uses the cgo-free `cuda` selector at goinfer root v0.5.0
(`2b137944683166156e23a06a8c0cd4487948fdad`). The CUDA v0.5.0 submodule tag differs
in module metadata only. The run reads the official Google safetensors checkpoint
at Hugging Face revision `f9972ac22e696a105c482202646979e33e4b5a1e` from the local
Hugging Face cache on disk. Projection and attention ran in FP32 with the
existing double-precision softmax reduction. The host used a Ryzen 9 7950X,
RTX PRO 6000 Blackwell Max-Q, NVIDIA driver 595.112.07 and Go 1.27.3.
The run began at 13:47 EST on 2026-11-22 with a 32 °C GPU, 0% sampled GPU
utilization, a 0.74 one-minute load average, and 97,249 MiB of CUDA-visible free
device memory.

Each case uses deterministic valid token IDs generated with the same Go PCG
seeds as the benchmark in the CUDA package. Each shape totals 4,096 positions.
For three independent accelerator attachments, warm each shape once, time one
ordinary `EmbedIDsBatch` call, then time one call with CUDA events immediately
before and after each kernel launch. Report the median across attachments.
Ordinary wall time includes host token gathering, copying, forward and pooling.
The between-kernels interval includes interleaved copies, submission and idle
gaps. The GPU memory sample covers the whole process at 100 ms intervals.

| Inputs × tokens/input | Wall ms | Tokens/s | Passages/s | Attention ms | GEMM ms | Between kernels ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 × 64 | 104.8 | 39,071 | 610.5 | 21.5 | 38.2 | 2.6 |
| 8 × 512 | 561.0 | 7,302 | 14.3 | 478.5 | 38.3 | 2.5 |
| 2 × 2,048 | 1,137.2 | 3,602 | 1.8 | 1,054.6 | 38.3 | 2.6 |

Whole-device peak memory was 1,618 MiB. Kernel instrumentation made the median
wall time 1.5–2.5 ms longer across the three cases. The untagged combined-route
subtest could not start because it requires the `eg2_combined` build tag. The
three default-route profiles completed and the table uses only those profiles.

Attention took approximately 85% of the ordinary 8 × 512 wall time and 93% at
2 × 2,048. The constant-token experiment points to attention as the first place
to reduce long-input latency. It does not establish a complete-device throughput
gain or resolve the 10× target in the issue.

## Reproduce the uninstrumented benchmark

Set `GOINFER_EG2_DIR` to the snapshot directory and make the NVIDIA driver
library discoverable. From the repo root, the benchmark includes each 4,096-token
profile shape plus 4 × 2,048 and a separate 12,287-token ragged case.

```sh
CGO_ENABLED=0 go test -tags cuda ./backend/cuda -run '^$' \
  -bench '^BenchmarkEG2CUDALongBatch$' -benchtime=3x -count=3
```

The official-checkpoint aggregate test uses five 2,048-token inputs and one
2,047-token input as one batch. It checks pooled float32 bits against singleton,
split and reversed input order for both CUDA selectors. Use the same snapshot.

```sh
CGO_ENABLED=0 GOINFER_HEAVY_TESTS=1 go test -tags 'cuda eg2_combined' \
  ./backend/cuda -v -count=1 -run '^TestEG2CUDALargeRealBatch$'
```

## Paired transposed-attention change

The existing opt-in `cuda-eg2-combined` kernel scored one key at a time per
thread. The candidate interleaves two independent score chains and computes
two output features together when the head width permits. The original FMA
order for each score and each output component is preserved. Softmax uses the
same reduction tree and arithmetic as before.

The paired run used the same machine, driver, checkpoint and shape generator
and started at 13:51 EST with a 38 °C GPU, 0% sampled GPU utilization and a 0.97
one-minute load average. It warmed both versions of the transposed-attention
kernel, then alternated them on the same loaded encoder for 10 pairs per shape.
Only the attention kernel changed between arms. The table uses upper medians
of wall times and paired ratios. Pooled vectors from both arms matched the
default CUDA selector bit for bit at both shapes.

| Inputs × tokens/input | Previous combined | Revised combined | Paired speedup | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| 8 × 512 | 153.26 ms | 141.07 ms | 1.090× | 10/10 |
| 4 × 2,048 | 524.13 ms | 507.93 ms | 1.032× | 10/10 |

Scratch size and device allocations are the same. Offline CUDA 13.0 `ptxas`
for `sm_120` reported 48 registers instead of 40, the same 5,120 static shared
bytes and no stack or spills in either attention kernel. The optimized
kernel remains behind the `eg2_combined` build tag and selector.

## Final-build forward rates

The final code was measured on the same hardware, driver and checkpoint at
13:53 EST. GPU temperature was 37 °C, one-minute load average 0.97 and sampled
GPU utilization 0% at the start. Each entry is a median of three separately
warmed calls; the selectors ran in groups. Peak sampled whole-device memory
across all shapes was 2,668 MiB.

| Selector | Inputs × tokens/input | Wall ms | Tokens/s | Passages/s |
| --- | --- | ---: | ---: | ---: |
| cuda | 8 × 512 | 563.26 | 7,272 | 14.20 |
| cuda-eg2-combined | 8 × 512 | 138.09 | 29,662 | 57.93 |
| cuda | 4 × 2,048 | 2,250.25 | 3,640 | 1.78 |
| cuda-eg2-combined | 4 × 2,048 | 507.84 | 16,131 | 7.88 |
| cuda | 5 × 2,048 + 1 × 2,047 | 3,358.36 | 3,659 | 1.79 |
| cuda-eg2-combined | 5 × 2,048 + 1 × 2,047 | 785.88 | 15,635 | 7.64 |

The benchmark entry for the 12,287-token case is `6x2048-ragged`. Compile with
`-tags 'cuda eg2_combined'` to include both selectors. The 12,287-token shape
requires the independent CUDA aggregate limit in [PR #45](https://github.com/pkieltyka/goinfer/pull/45).
