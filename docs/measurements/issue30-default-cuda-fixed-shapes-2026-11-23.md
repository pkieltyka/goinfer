# EmbeddingGemma 2 default CUDA across v0.5.0 and v0.6.6

The four prepared-ID workloads ran at nearly the same rate across the two
default CUDA releases. At 8 × 512 on v0.6.6, sliding and full attention accounted
for 86.0% of summed CUDA event time. The comparison is descriptive. The formal
10× target under a one-minute load cap of 1.0 remains unassessed. [goinfer #30](https://github.com/pkieltyka/goinfer/issues/30)

## Public call wall time

Each shape is inputs × positions per input. Two fresh ABBA blocks per shape
yielded four adjacent pairs. Each process warmed once and then recorded three
synchronous `EmbedIDsBatch` calls without event hooks, including host ID
gathering, copies, forward, readout and pooling. Each arm's wall and rate
columns take the median of the four process means or corresponding rates. The
paired ratio takes the median of the four historical/current wall ratios.

| Shape | v0.5.0 ms/call | v0.6.6 ms/call | v0.5.0 positions/s | v0.6.6 positions/s | Matched throughput v0.6.6/v0.5.0 (range) | v0.6.6 faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 × 64 | 104.943 | 105.479 | 39,031 | 38,833 | 0.995649× (0.988610–1.004985) | 2/4 |
| 8 × 512 | 563.766 | 561.657 | 7,265 | 7,293 | 1.001648× (1.000868–1.007031) | 4/4 |
| 2 × 2,048 | 1,137.528 | 1,142.532 | 3,601 | 3,585 | 0.995607× (0.995197–0.997468) | 0/4 |
| 4 × 2,048 | 2,254.129 | 2,266.545 | 3,634 | 3,614 | 0.994517× (0.994319–0.994668) | 0/4 |

## Event attribution

Separate binaries captured one diagnostic forward per release and shape at
4,096 total positions. Each capture recorded 677 kernel launches, including 20
sliding-attention and four full-attention launches. Percentages use the sum of
all recorded CUDA operation spans, including copies, as their denominator. The
matching unstamped call wall is in the table above. Event sums describe the
instrumented forward. They do not measure exclusive shares of public wall or
hardware occupancy, and host waits can overlap device work.

| Shape | Release | Instrumented wall (ms) | Event sum (ms) | Sliding attention (ms, % of sum) | Full attention (ms, % of sum) |
| --- | --- | ---: | ---: | ---: | ---: |
| 64 × 64 | v0.5.0 | 108.295 | 99.982 | 15.364 (15.37%) | 6.036 (6.04%) |
| 64 × 64 | v0.6.6 | 107.895 | 99.988 | 15.358 (15.36%) | 6.047 (6.05%) |
| 8 × 512 | v0.5.0 | 564.650 | 556.844 | 347.212 (62.35%) | 131.835 (23.68%) |
| 8 × 512 | v0.6.6 | 564.924 | 556.606 | 345.671 (62.10%) | 133.095 (23.91%) |
| 2 × 2,048 | v0.5.0 | 1,139.219 | 1,131.682 | 588.528 (52.00%) | 465.423 (41.13%) |
| 2 × 2,048 | v0.6.6 | 1,144.913 | 1,137.017 | 585.083 (51.46%) | 473.775 (41.67%) |

The longer sequences shift most measured device-event time to attention even
at a fixed total position count. The 8,192-position row has public wall
measurements only.

## Qualification and provenance

- On 2026-11-23 the cgo-free Go 1.27.4 builds used Linux amd64, CPUs 0–3 and
  `GOMAXPROCS=4` on a Ryzen 9 7950X with an RTX PRO 6000 Blackwell Max-Q and
  NVIDIA driver 595.112.07. The pinned GPU was at PCI `0000:01:00.0`.
  Sampled GPU temperature was 39–66 °C in the wall phase and 42–52 °C in events.
- Historical root/CUDA commits were `2b137944683166156e23a06a8c0cd4487948fdad`
  and `e7482f8addf4469674de2864f2233072661d72db`. The v0.6.6 root and CUDA
  both used `c0a5159bbff1c2fe5c8624e1399aa7438c820677`. Both used the
  `cuda` selector and the same Google safetensors checkpoint on local disk at
  revision `f9972ac22e696a105c482202646979e33e4b5a1e`. Model SHA-256
  `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`.
  The sealed inputs include the four ordered-ID digests and all six Google
  files, including the tokenizer configuration and chat template.
- Both releases passed their CPU reference and CUDA fidelity checks. The
  8 × 512 and 4 × 2,048 certificates matched singleton, packed, split and
  reversed layouts bit for bit within each release. All four shapes matched
  FP32 output words across releases, with cosine 1.0 at 768 and 512 dimensions.
  Every wall and event output matched its same-arm certificate.
- Sustained ARM emulation had kept the 1.0 load window unavailable. On
  2026-11-23 the descriptive phases were authorized at load1 ≤ 6.0, with
  exclusive CUDA use and the original memory, fidelity and paging guards.
  All 12 qualification, 32 wall and six event cells passed on their first
  attempts under the final seal. Actual launch load was 1.099–3.996, minimum
  host availability was 103,730 MiB and peak sampled device use was 2,142 MiB.
  No foreign CUDA PID, child swap or new host pageout was observed. An earlier
  1.0-cap attempt ended in quiet wait, and an earlier descriptive attempt
  stopped when the sampler missed the short-lived CUDA child. Both are
  retained outside the accepted receipts.

| Retained artifact | SHA-256 |
| --- | --- |
| Runtime input seal | `600f0e9358d63d1301596b4e0be554cc6cad821080d03171523425e71b53136d` |
| Untimed qualification receipt | `b6382b9285f13143971ef7edf5d17aa317a10633c609bd840ecddeba0f52cdc9` |
| Public wall receipt | `d3202b7549a864bffbb174cd2cbd41bb4fb8a931df8e5288ed9b50edc18fad87` |
| Diagnostic event receipt | `ac92c2724980395b5924169cc64ffc97a1b2a17de18516eb34964a8153dca3c9` |
| Diagnostic event archive seal | `520899094fe0a3d4ad82f40b0d9334334cad00d672197444a4219c8650f0a21f` |

## Next experiments

1. Pair the shipped v0.6.6 `cuda-eg2-fp16` opt-in with v0.6.6 `cuda` on
   the same checkpoint, RTX and four shapes. Verify the FP16 route, its
   precision-specific reference gate and exact layout bits within the
   selector. Use matched public-call walls and capture profiles separately.
   The selector also changes FFN precision and uses fusions, so attributing
   a gain to packed attention would require a test-only arm that enables
   packed attention alone with the default precision and fusion settings
   elsewhere. Check its reference fidelity, exact within-arm layout bits
   and scratch residency before timing.
2. For a bit-preserving default-path prototype, first test coalesced K loads
   inside `attention_seg` with each lane's dot-product order preserved.
   Keep the FP64 softmax lane assignments, width and reduction tree, ordered
   V accumulation and per-sequence full/sliding bounds. Require stock
   output bits for singleton, packed, split and reordered inputs, including
   window boundaries and tails. Compare public walls and peak scratch on
   the same four shapes before considering removal of redundant barriers
   as a separate change. A variant that changes reduction order would need
   its own opt-in fidelity, layout and speed checks.

## Historical Q8 peer context

The earlier peer comparison used the Q8 GGUF checkpoint and the
`cuda-eg2-fp16` opt-in selector with a different workload. Its results are
separate from these default-CUDA safetensors measurements. [goinfer #67](https://github.com/pkieltyka/goinfer/issues/67)
