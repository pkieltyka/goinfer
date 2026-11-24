# EmbeddingGemma 2 attention routes on v0.6.6 CUDA

The shipped FP16 selector and a private packed-attention-only arm both reduced
warm public-call wall on the four prepared-ID shapes. At 8 × 512 the matched
throughput ratios against fresh default CUDA cells were 6.463× for the shipped
selector and 5.660× for packed attention alone. The earlier default-only
release comparison and its formal target remain separate.
[goinfer #30](https://github.com/pkieltyka/goinfer/issues/30) ·
[Default CUDA release comparison](issue30-default-cuda-fixed-shapes-2026-11-23.md)

## Setup and acceptance

On 2026-11-23 both arms of each comparison ran goinfer root and CUDA
v0.6.6 at `c0a5159bbff1c2fe5c8624e1399aa7438c820677`. The hardware was a
Ryzen 9 7950X and RTX PRO 6000 Blackwell Max-Q, CUDA capability 12.0,
PCI `0000:01:00.0`, NVIDIA driver 595.112.07. The cgo-free Linux amd64
binaries used Go 1.27.4. Each process used CPUs 0–3 and `GOMAXPROCS=4`.
The same Google safetensors checkpoint at revision
`f9972ac22e696a105c482202646979e33e4b5a1e` was read from local disk.
The model SHA-256 is
`197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`.
All six checkpoint and tokenizer inputs and the four ordered-ID digests
match the pinned inputs from the earlier comparison.

Each shape below is inputs × positions per input. Two fresh ABBA blocks
give four adjacent pairs. Each process warms once and records three
synchronous `EmbedIDsBatch` wall times. The clock includes public-call
gather, forward, readout and pooling. Checkpoint load, attachment, fit
admission, tracing and vector checks occur outside the clock. Reported
times and rates are medians across the four processes per arm. The paired
throughput column takes the median and range of stock-wall divided by
candidate-wall in the four adjacent pairs.

The guard verifies the pinned device and sole CUDA compute child under
the exclusive timing lock. It requires 10 GiB available host memory,
7 GiB GPU free at preflight and less than 6 GiB sampled device-wide use.
Live fit is checked before each forward. The accepted cells had zero child
swap and no new host pageout. Wall cells reproduced their own selector's
qualified output bits. Untimed certificates checked singleton, packed,
split and reversed bits within each selector at 8 × 512 and 4 × 2,048.
All four exact-ID CPU comparisons passed cosine 0.999 at 768 and 512
dimensions. The default output minimum was 0.999999999610 across widths.

## Shipped FP16 selector

The candidate used the shipped `cuda-eg2-fp16` factory with FP16 FFN,
packed attention and the GELU/PLE fusions. An explicit UUID attach on
the public path verified the selected route. Each untimed real-shape
witness observed 72 FP16 FFN, 24 packed attention and 24 launches of each
fusion. The released FP16 precision test passed. The minimum cosine
against the 48-row independent real reference was 0.999999960155 at 768
and 0.999999959894 at 512. The lowest same-ID CPU cosine was 0.99999233.

| Shape | Stock / FP16 ms/call | Stock / FP16 positions/s | Paired FP16 throughput ratio (range) | Peak GPU stock / FP16 MiB |
| --- | ---: | ---: | ---: | ---: |
| 64 × 64 | 104.795 / 72.704 | 39,086 / 56,338 | 1.441× (1.441–1.456) | 1,608 / 1,364 |
| 8 × 512 | 561.742 / 86.941 | 7,292 / 47,113 | 6.463× (6.429–6.489) | 1,610 / 1,366 |
| 2 × 2,048 | 1,143.331 / 111.996 | 3,583 / 36,574 | 10.209× (10.129–10.327) | 1,618 / 1,374 |
| 4 × 2,048 | 2,270.274 / 214.416 | 3,608 / 38,206 | 10.583× (10.577–10.602) | 2,142 / 1,794 |

FP16 was faster in all four pairs per shape. The 32 first-attempt wall cells
started at load1 0.598–0.987 under cap 1.0, with sampled GPU temperatures
33–58 °C and no voids. Peak host RSS across the cells was 1,882,980 KiB for
stock and 1,881,304 KiB for FP16. The planner recorded 547,139,584 B of
resident weights for stock and 396,144,640 B for FP16.

## Packed attention alone

The private arm selected the same packed-attention kernels and kept stock
FP32 FFN, unfused activations and the stock short-GEMM policy. Its untimed
route witnesses counted 20 sliding and four full packed-attention launches
per real forward with zero stock-attention launches. A tiny test checked
mixed-length layout bits, rejection of a stock attachment by the packed
route check, and nonfinite-input refusal followed by recovery. The
independent 48-row real reference minimum cosine was 0.999999999991 at
each width. The minimum same-ID CPU cosine was 0.999999999665.

| Shape | Start cap | Stock / packed ms/call | Stock / packed positions/s | Paired packed throughput ratio (range) | Peak GPU stock / packed MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 × 64 | 1.0 | 104.726 / 84.134 | 39,112 / 48,684 | 1.243× (1.206–1.252) | 1,608 / 1,514 |
| 8 × 512 | 1.0 | 561.530 / 99.162 | 7,294 / 41,306 | 5.660× (5.652–5.688) | 1,610 / 1,516 |
| 2 × 2,048 | 1.0 | 1,144.160 / 123.003 | 3,580 / 33,302 | 9.300× (9.218–9.400) | 1,618 / 1,524 |
| 4 × 2,048 | 4.0 | 2,275.442 / 237.576 | 3,600 / 34,482 | 9.581× (9.533–9.595) | 2,142 / 1,952 |

Packed attention was faster in all four pairs per shape. The original
1.0-cap run accepted 31 first-attempt cells and exhausted its 600-second
quiet budget before cell 32. All original starts were at load1 0.792–0.994.
On 2026-11-23 a 4.0 cap was approved for a fresh descriptive long-shape
block because the original quiet budget ran out. The last row uses only
those eight fresh cells, which started at load1 0.881–1.425. The seven
earlier 4 × 2,048 cells remain diagnostic. Both attempts had zero
execution voids. Sampled GPU temperatures were 31–55 °C in the original
run and 32–57 °C in the fallback.

Peak host RSS among the table's cells was 1,882,124 KiB stock and
1,881,212 KiB packed. Both arms held 547,139,584 B of resident weights.
At 4 × 2,048 the cold planner estimated an additional 1,111,621,632 B
for stock and 912,270,336 B for packed. The sampled device peak and planner
estimate describe separate quantities.

## Bit-preserving transposed attention

This private arm uses the released transpose producer and transposed
attention consumer, with stock FP32 FFN, unfused activations and short tiled
GEMM disabled. It preserves the default FP64 softmax reduction and output
bits. The 64 × 64 control stays on stock attention. The three long-shape
qualification witnesses each recorded 24 transposes and 24 transposed
attention launches, with zero stock or tiled launches. Stock bytes matched
on all four shapes, on both long-layout gates and across ten crossover and
tail cases. In the separate 11-position control both arms took the stock
route, matched stock bits and recorded six walls of 15.351–15.713 ms.

| Shape | Stock / transposed ms/call | Stock / transposed positions/s | Paired transposed throughput ratio (range) | Transposed faster |
| --- | ---: | ---: | ---: | ---: |
| 64 × 64 | 104.640 / 104.868 | 39,144 / 39,059 | 0.997× (0.994–1.007) | 1/4 |
| 8 × 512 | 561.576 / 135.599 | 7,294 / 30,207 | 4.141× (4.134–4.171) | 4/4 |
| 2 × 2,048 | 1,142.235 / 250.107 | 3,586 / 16,377 | 4.569× (4.526–4.586) | 4/4 |
| 4 × 2,048 | 2,268.932 / 505.948 | 3,611 / 16,191 | 4.482× (4.466–4.504) | 4/4 |

On 2026-11-23 the 4.0 cap was approved for this descriptive campaign to
avoid another extended wait at 1.0. All 32 wall cells passed on the first
attempt with load1 0.954–1.783, sampled GPU temperatures 36–62 °C and no
voids. Both arms peaked at 2,142 MiB sampled device use. Stock/transposed
peak RSS was 1,882,724 / 1,882,556 KiB. Both held 547,139,584 B of resident
weights and had matching cold planned footprints. The long-shape gain
belongs to the complete transposed route, which also changes shared Q
staging, score interleaving and V loads and avoids stock QKV packing.

## Evidence and scope

| Evidence retained by SHA-256 | Digest |
| --- | --- |
| Shipped-selector runtime seal | `516cdcdf311d6e07ec3a319dbfe0c721de4df9118d0da59a84fc1e3f5c71075e` |
| Shipped-selector qualification receipt | `65256e6e23cbf63dd9dede65ab46fc8ef41a5a07dbce3ca64d63c9753eed5597` |
| Shipped-selector wall receipt | `0880e8f2a43c82f31bca6d93c9a4d9b7393102bcb86ff2ac4d18c45cc35079e1` |
| Same-ID CPU reference | `cd4a1b6f7bad60bd821848463c109a22135d7275d306d116d5d24a1b8116cfc5` |
| Packed-only runtime seal | `b40145e3c822a8a025bbba2711e392036cea57932aadc22ce6f2eeb37ece5bf8` |
| Packed-only qualification receipt | `4f8f5c30417aff8d33f0268541931efcf3af9314fc5ace6c6ca538fb6b481e21` |
| Packed-only original parked archive | `119d68f57a181efdcfe1464bdd2d7c26ad03fc732d761f57d2b29d2c0c733867` |
| Packed-only complete 1.0-cap shapes | `871c93939e0d45065190449d1ba7c082e0c2e3286d3786f2f0805af0c580f9ae` |
| Fresh descriptive 4 × 2,048 wall receipt | `a5c8639444e742b521c385bb22eed11c0e1c0a0cfadcc61ebeb001cc5ce1735d` |
| Transposed runtime seal | `8b8369f29058ca7a5da1dde4f74a78fc5f640653084f90dadb630ac2764f95c7` |
| Transposed qualification receipt | `55249e3ae379bd28c05f04a208b11e1349bbc2799b050e6c64a569b57e985f1b` |
| Transposed wall receipt | `25438405cc06d0f8e45b03e2813d9f2445f4bb6108ae55ca9021ded08b72128f` |

Each comparison uses fresh stock pairs on the same RTX and Google checkpoint.
Packed attention accounts for most of the observed long-batch gain. Measuring
the separate contributions of FP16 FFN and fusions requires their own isolated
arms. The selector-constant formal target remains open.
