# EmbeddingGemma 2 CUDA composed FP16 qualification

On November 22, 2026, the RTX PRO 6000 Blackwell Max-Q Workstation Edition
with driver 595.112.07 ran goinfer composed product commit
`16d163cef9a4e094aa05a560880c89cf73fa69c8` (over `origin/main` base
`f69080de21606a8a052f6db24dfb9e8d1bfef0df`). The host was a Ryzen 9 7950X.
Both arms used one Go 1.27.3 linux/amd64 executable built with
`CGO_ENABLED=0 GOAMD64=v1` and `-tags 'cuda eg2_combined eg2_fp16'`, 32
logical CPUs and `GOMAXPROCS=32`. Checkpoints and prepared inputs were on
local disk.

## Selector, build tags, and artifact pins

Under `-tags 'cuda eg2_combined eg2_fp16'`, the existing opt-in
`cuda-eg2-fp16` selector composes three PTX modules on CUDA compute
capability 8.0 or later while leaving default `cuda` and `cuda-eg2-combined`
unchanged:

| Artifact | Source | NVRTC arch | SHA-256 |
| --- | --- | --- | --- |
| `backend/cuda/testdata/eg2_ffn_fp16.ptx` | `backend/cuda/eg2_ffn_fp16.cu` | `compute_80` | `a1265fd9e30f81984e8897da396e05537bb366a68a777dae6e4d04eed3efab31` |
| `backend/cuda/testdata/eg2_packed_attn.ptx` | `backend/cuda/eg2_packed_attn.cu` | `compute_75` | `d0ea4dac4c894d51c7df5e5632c69a7303a1b99dfb41e63d2d24ac3412ec01fa` |
| `backend/cuda/testdata/eg2_fusions.ptx` | `backend/cuda/eg2_fusions.cu` | `compute_75` | `446e34045eeafac5e229894d90f429379d0d4c8c1768c56515e1ae6210198c62` |

The inputs are the 200 passages from
[goinfer #63](https://github.com/pkieltyka/goinfer/issues/63), in their
original 13 groups, with 100,066 prepared positions:

- Prepared-ID TSV SHA-256: `97c426da8c155b04bf0b34e270dcfa5e8cf18bfc1343e926f63e638fea6bcd22`
- Google pack layer-file SHA-256: `28a9678df941bf3eb7acca4e27ffd2fba6eef24039374d62148d705fc83b6227`
- Unsloth Q8_0 GGUF SHA-256: `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc`

Keep both the build revision and the `cuda-eg2-fp16` selector fixed for
embeddings that share an index: prior FFN-only builds (such as
`e73cd96688711d1fda91cb8bba9fb5466b7fae59`) used the same selector name
with different attention arithmetic and produced different per-mode vector
digests.

## Composed paired ABBA throughput gate

For each format, the acceptance rule set before the run was at least 9 of 10
faster pairs and a median paired speedup of at least 1.05×. In each of five
ABBA blocks, A selected `cuda-eg2-combined` and B selected `cuda-eg2-fp16` on
the same explicit CUDA UUID. In a fresh process per cell, load and attach
once, check `BatchMemory` for each group, pass the 13 groups in order to
`EmbedIDsBatch`, and sum only those synchronous encoder-forward wall
intervals (excluding attach, checksums, and any application indexing). Pair
cells 1/2 and 4/3 within each block. The exclusive timing lock covered all
20 cells in each model run with zero environmental voids. Cell-start 1-minute
load ranged from 0.63 to 1.02 (Google) and 0.85 to 1.26 (Q8_0); cell-start
GPU temperatures ranged from 41 to 54 °C (Google) and 53 to 57 °C (Q8_0).
Process swap was 0 KiB and all cells ran within the 6 GiB fit envelope.

| Checkpoint | Combined positions/s | Composed FP16 positions/s | Combined forward ms | Composed FP16 forward ms | Paired median | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Google pack | 24,010.27 | 44,858.03 | 4,167.63 | 2,230.73 | 1.86594× | 10/10 |
| Q8_0 GGUF | 23,982.42 | 45,520.24 | 4,172.47 | 2,198.27 | 1.89579× | 10/10 |

The positions/s and forward-ms columns are arm medians across the 10 cells
per arm. The paired speedup ranges were 1.85698–1.87417× for Google and
1.86956–1.91656× for Q8_0. Every cell kept its exact per-arm output digest
across all 10 cells: `cuda-eg2-combined` matched the published-control replay
digests (`a09fcb1aa99a52daaaa3fd31f0f5c91a02ddd83dbdb9ff5dbe7778df08cb20c2`
for Google, `cca899e59ab394ce0f21c6b9cff10d291bc96026c53611b83a91333e16d7c468`
for Q8_0), while composed `cuda-eg2-fp16` held
`9ecc8b11e92ee972206df03f9ec864745405d908e7a8115e2ec70dcde21a3899` (Google)
and `97d6b1e8d2cca976fd69167370472196c1d00210bdc75f58de45826f316b1920` (Q8_0).

For both checkpoints:

- Tracked resident weights fell from `547,139,584` to `396,144,640` bytes.
- Maximum estimated additional device memory across the 13 requested groups
  fell from `1,098,102,880` to `881,976,368` bytes.
- Sampled peak GPU memory fell from `2,130` to `1,782` MiB.
- Peak process RSS rose from `1,852,000` to `1,994,628` KiB for Google and
  from `1,386,284` to `1,500,392` KiB for Q8_0, with 0 KiB process swap.

## Composed 81-case and 200-passage quality gate

Untimed validation compared each format against its own CPU vectors on the
same GPU (zero process swap) across the 81-case reference set (22 reference
texts and 59 passages, 2 groups) and the 200-passage slice (13 groups) at
both 768 and renormalized 512 dimensions. Every vector cleared 0.999 cosine
and matched its same-mode singleton by bits when packed, split, or reordered:

| Checkpoint | Case set | Min CPU cosine (768) | Min CPU cosine (512) | Singleton/pack/split/reorder bits |
| --- | --- | ---: | ---: | --- |
| Google pack | 81-case reference (2 groups) | 0.9999999459204383 | 0.9999999472227570 | exact |
| Google pack | 200-passage slice (13 groups) | 0.9999998178604141 | 0.9999998162982013 | exact |
| Q8_0 GGUF | 81-case reference (2 groups) | 0.9999998995980207 | 0.9999999008339214 | exact |
| Q8_0 GGUF | 200-passage slice (13 groups) | 0.9999994480482701 | 0.9999994598295197 | exact |

Across both case sets and both output dimensions, the overall minimum CPU
cosine was `0.9999998162982013` for the Google pack and `0.9999994480482701`
for Unsloth Q8_0 GGUF.

## Focused RTX smoke and launch/memory counter deltas

On the RTX PRO 6000 Blackwell Max-Q (driver 595.112.07, exclusive timing
lock, zero prior/residual compute apps, 0 skips across all 7 tests at
`16d163cef9a4e094aa05a560880c89cf73fa69c8`):

- `TestEG2FP16Encoder`: PASS (0.34s; raw singleton bit-identity; minimum tiny
  pooled CPU cosine `0.999949546364`).
- `TestEG2FP16FailedAttach`: PASS (0.26s).
- `TestEG2FP16PackedAttnContract`: PASS (0.23s; verified FP16 FFN, packed
  attention, and both fusion pipelines loaded; exact cold/warm lease peaks
  and cold RoPE retention on BM=32 and BM=32+16; M=36/37 shortTiled raw
  bit-identity; `sliding_window=4294967296` clamp; native cleanup precedence
  ahead of `errEG2FP16InputRange` across `fp16-ffn+packed+fusions`,
  `packed-attn-only`, and `fp16-ffn-only`).
- `TestEG2FusionsArithmetic`: PASS (0.18s; bit-identical finite/zero outputs
  and matching exceptional classes vs separate `gelu_tanh` + `tower_mul` /
  `Copy2DAsync`).
- `TestEG2PackedAttn_DeviceParityAndBitIdentity`: PASS 14/14 (3.46s;
  `local_bytes=0` across all 4 entry points `hd16`/`hd32`/`hd256`/`hd512`;
  worst sinusoidal FP32 CPU cosine `0.99999921`; masked nonfinite isolation,
  clipped endpoint 514/W=512, extreme negative score + large window,
  exceptional-V tiny FP32 weight, and finite half overflow flag).
- `TestEG2FusionsTiny` (3 inputs, 182 rows, PASS 0.72s) and
  `TestEG2FusionsReal` (first frozen group: 17 inputs, 7,924 positions on the
  pinned Google pack, PASS 4.95s): both arms kept raw, pooled, and
  singleton/split/reversed bits identical across the fusion toggle, and the
  composed fused arm matched public `cuda-eg2-fp16` bit-for-bit:

| Workload and mode | Kernel launches | D2D copies | Scratch allocs | Tracked peak bytes |
| --- | ---: | ---: | ---: | ---: |
| Tiny 3 inputs / 182 rows (`combined-fp32`, unfused → fused) | 179 → 167 | 18 → 12 | 28 → 27 | 1,135,712 → 1,124,064 |
| Tiny 3 inputs / 182 rows (`composed-fp16-packed`, unfused → fused) | 173 → 161 | 18 → 12 | 26 → 25 | 853,348 → 841,700 |
| Google 17 inputs / 7,924 rows (`combined-fp32`, unfused → fused) | 701 → 653 | 92 → 68 | 28 → 27 | 1,076,803,392 → 1,060,026,176 |
| Google 17 inputs / 7,924 rows (`composed-fp16-packed`, unfused → fused) | 677 → 629 | 92 → 68 | 29 → 28 | 881,650,628 → 864,873,412 |

On the 17-input, 7,924-position real group, composed `cuda-eg2-fp16` reduces
launches from 701 to 629 (−72), D2D copies from 92 to 68 (−24), and tracked
per-forward peak scratch from 1,076,803,392 to 864,873,412 bytes
(−211,929,980 bytes).

## Preserved independent component baselines

Each component was also verified independently prior to composition on the
same RTX PRO 6000 Blackwell Max-Q (driver 595.112.07):

1. **FP16 FFN only** (`e73cd96688711d1fda91cb8bba9fb5466b7fae59`, PR #64,
   [FP16 FFN qualification](eg2-fp16-ffn-2026-11-22.md)): paired ABBA median
   `1.07760×` (10/10, 23,907 → 25,735 pos/s) on Google pack and `1.08179×`
   (10/10, 24,016 → 25,985 pos/s) on Q8_0 GGUF; worst 768/512 CPU cosine
   `0.9999998630` (Google) and `0.9999994381` (Q8_0); resident weights
   `547,139,584 → 396,144,640` bytes; sampled peak GPU `2,130 → 1,988` MiB.
2. **Standalone three-term split-FP16 packed attention**:
   - Direct 24-layer attention microbenchmark at
     `51a932d9d32368ae6479362d106a8e8c59ca337e` vs `cuda-eg2-combined`
     (`TestEG2PackedAttn_VsCombinedBench`):
     - 17-input 7,924-token shape: `169.25 → 34.74 ms` (`4.87×`, sliding
       `6.27×`, full `3.25×`, `cos=1.00000000`)
     - `8×512` (4,096 tokens): `59.25 → 16.64 ms` (`3.56×`, `cos=1.00000000`)
     - `4×2048` (8,192 tokens): `347.41 → 79.80 ms` (`4.35×`, `cos=1.00000000`)
   - Standalone opt-in encoder qualification at
     `0cb9b4580c349ed863387ff3e6453c9b74adb5c9` across 81-case reference +
     200-passage slice: worst 768/512 CPU cosine `0.999999999969176` (Google)
     and `0.9999999999859042` (Q8_0), with exact singleton/packed/split/reordered
     bits.
3. **Standalone bit-exact GELU/PLE fusions**
   (`7a57ae3a716e07502b57398e307abb48369b2d3b` / admission-hardened
   `6614b9e5444515929e9d0bab73709db31bf92991`): bit-identical raw, pooled,
   and retained layer-state outputs vs two-launch baseline on tiny 182 rows
   and the 7,924-position Google group (`701 → 653` launches, `92 → 68` D2D
   copies, `28 → 27` allocations, `1,076,803,392 → 1,060,026,176` tracked
   peak bytes).

These timings cover only the synchronous prepared-ID encoder forward calls on
the fixed slice. A full application sync also prepares text, maintains caches,
and writes the SQLite index.
