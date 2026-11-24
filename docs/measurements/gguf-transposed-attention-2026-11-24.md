# GGUF same-file check — default transposed CUDA attention, 2026-11-24

**What was measured.** The default `cuda` EmbeddingGemma 2 route before (#79's parent, `105095440`) and after
(`449a96ef`, merged as #79 at `790194d`), on the two shipped GGUF files through the public encoder only. Every cell
loads one file from local disk, checks its byte length and SHA-256, and runs a fixed input set in both arms: the stock
binary is the pre-#79 build, the candidate binary is the post-#79 build. Machine: an NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition (96 GB), NVIDIA driver `595.112.07`, four cores, exclusive timing lease, load1 cap 1.0, no
concurrent device work. Run date 2026-11-24 (UTC).

| file | bytes | SHA-256 |
| --- | --- | --- |
| `embeddinggemma-2-Q8_0.gguf` | 309,855,520 | `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc` |
| `embeddinggemma-2-UD-Q6_K_XL.gguf` | 248,812,032 | `dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3` |

**Exact bits.** The candidate output bytes equal the stock output bytes in all 16 (file, shape) cells, and the
singleton, split and reversed-batch layouts are bit-identical to the batch call in each cell. Minimum same-file cosine
against a never-attached CPU oracle across the 32 default-CUDA certificate cells: **0.99999999997667** at 768 and
**0.99999999997689** at 512.

| shape | long route on the candidate | extra device bytes |
| --- | --- | --- |
| 127+96 | stock | 33,130,992 |
| 96+127 | stock | 33,130,992 |
| 96 | stock | 16,287,232 |
| 128+96 | 24 transpose + 24 transposed attention, 0 stock/tiled | 33,197,568 |
| 96+128 | 24 transpose + 24 transposed attention, 0 stock/tiled | 33,197,568 |
| 128 | 24 transpose + 24 transposed attention, 0 stock/tiled | 18,221,056 |
| 8×512 | 24 transpose + 24 transposed attention, 0 stock/tiled | 552,665,088 |
| 4×2048 | 24 transpose + 24 transposed attention, 0 stock/tiled | 1,111,621,632 |

Both files carry these route and fit values: the same 24-layer geometry, the same per-shape batches, and equal extra
bytes on both arms of every cell.

Resident weight bytes are 547,139,584 on both arms of every cell: the transposed route adds no weights and no planner
charge. The route gate is the longest individual input, so a batch containing a 128-token row and a 96-token row takes
the long route while the 127+96 and 96+127 controls and the 96-token row alone stay on stock attention.

**Timing.** One fresh ABBA per file and shape: two matched pairs of fresh processes, three timed forwards per process,
every call digest equal to the pre-clock warm call and to the stock certificate. Descriptive only, no threshold claim,
no cross-quant ratio.

| file | shape | stock ms | candidate ms | stock pos/s | candidate pos/s | matched ratio (median) | range | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q8_0 | 8×512 | 569.264 | 142.719 | 7,195 | 28,700 | 3.989× | 3.981–3.996 | 2/2 |
| Q8_0 | 4×2048 | 2,277.488 | 506.128 | 3,597 | 16,186 | 4.500× | 4.497–4.503 | 2/2 |
| UD-Q6_K_XL | 8×512 | 569.899 | 144.674 | 7,187 | 28,312 | 3.939× | 3.925–3.954 | 2/2 |
| UD-Q6_K_XL | 4×2048 | 2,278.469 | 515.368 | 3,595 | 15,895 | 4.421× | 4.409–4.433 | 2/2 |

**Memory.** A certificate cell opens one device context and one model; sampled peak device memory is 1,082–2,142 MiB
over the eight shapes and both files. The two timed shapes read 1,610 MiB at 8×512 and 2,142 MiB at 4×2048 on either
arm, and peak host RSS was 1,111–1,364 MiB.

**Optional linked FP16 applicability.** In a `cuda,eg2_combined,eg2_fp16` build the candidate's `cuda-eg2-fp16`
attaches for both files and stays on packed attention (24 packed launches, zero stock or transposed), with no refusal:

| file | shape | cosine 768 / 512 | resident bytes | extra bytes | peak device |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | 8×512 | 0.99999879 / 0.99999886 | 396,144,640 | 445,649,408 | 1,366 MiB |
| Q8_0 | 4×2048 | 0.99999981 / 0.99999982 | 396,144,640 | 895,493,120 | 1,794 MiB |
| UD-Q6_K_XL | 8×512 | 0.99999775 / 0.99999785 | 396,144,640 | 445,649,408 | 1,366 MiB |
| UD-Q6_K_XL | 4×2048 | 0.99999972 / 0.99999973 | 396,144,640 | 895,493,120 | 1,794 MiB |

All four executed FP16 cells passed the layout checks bit-identically: singleton, split and reversed batches match
the batch call. goinfer itself has no Q8-versus-Q6 selector rule. pbrain ships `cuda-eg2-fp16` for Q8 only; the Q6
reading is exploratory and is not a shipping change.

**Provenance.** Source seals: the two trees are `10509544077597772cdf32e56d16e5e3c59a8bab` and
`449a96ef765131176709a36a4ee66ae4af73c3b1`, both clean at build time. Binaries `42e66216…`, `bd5825a1…`,
`dd0ddbdc…`; runtime manifest `5b16cefd85c4303cbee4822831d7a4addbb79565df150b85d755ecc9a75f864d`; source manifest
`e8e1af3c0c228e17b67ffe5aee2ed5b8b07f2f099dae856471295aa9393efdc5`. Phase receipts: oracle `1ed8daf4…`,
certificate `605dd30b…`, wall `dd612b4a…`, fp16 `d73dc9dc…` (SHA-256 of each receipt JSON). 54 accepted process
starts (2 CPU oracle, 32 certificate, 16 wall, 4 fp16) and one childless admission refusal at load1 1.04, retried in
the same position: 55 attempts, zero execution voids, zero swap, zero pageout. All 48 timed calls reproduced their
registered output digest.

**Scope.** Same-file default-route exactness and descriptive timing on the two shipped GGUF files; linked
`cuda-eg2-fp16` applicability for both files, with Q6 exploratory; the formal #30 target remains open.
