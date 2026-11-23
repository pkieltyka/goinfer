# Radeon Vulkan projection tile padding (#76)

<!-- citations-at: 0aad368b14b66e569eebce021ee029eac10b3ca8 -->

**Parked.** On 2026-11-23, five fresh ABBA quads measured stock and padded shared
weight tiles in EmbeddingGemma 2. Neither arm reached the registered median
1.05× speedup and 9/10 paired wins on either shape.

## Method

Both cgo-free Go 1.27.4 test executables used goinfer
`0aad368b14b66e569eebce021ee029eac10b3ca8` and the same Go test overlay. Stock A
used the 16×16 shared weight tile. Padded B used 16×17, raising static
workgroup storage from 1,536 to 1,600 B. SPIR-V inspection verified the same
16×8 dispatch tile, three storage bindings, four shape words, barriers and
ascending-K FP32 FMA sequence.

The host had a Ryzen 9 7950X, 124.9 GiB RAM and Linux 7.3.5. The device was an
RX 9060 XT (RADV GFX1200), Mesa 26.3.1, with 17,095,983,104 B device-local
memory. Vulkan loader and Khronos validation were 1.4.363.0. Each fresh process
attached the pinned device `vulkan:000000000c0000000000000000000000`, used four
Go scheduler threads and loaded the same 309,855,520 B Q8_0 GGUF from local disk.

Each cell warmed the 11-position query `a035` and the four-document batch
`pack-startup`, `file-navigation`, `combined-gate`, `vector-sql`
(166/165/161/158 positions). It then measured three calls of each shape in the
same order. The query clock covered `EmbedIDs`; the document clock covered one
`EmbedIDsBatch`. Both ended on return of host-normalized vectors. Model load,
attachment, fit checks and vector validation were outside the clocks. These
are the prepared shapes from the earlier Radeon projection profile.
[Radeon profile and corpus evidence](issue39-radeon-v066-2026-11-23.md)

## Paired result

Ratios divide stock wall by padded wall within each adjacent pair, using the
sum of three calls per cell. Latency and rates in the table are medians across
the ten process measurements for each arm.

| Shape | Median A/B | Paired range | Wins A / B | Median ms/call A / B | Median positions/s A / B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query, 11 positions | 1.005404× | 0.959599–1.092210× | 4 / 6 | 22.104 / 21.957 | 497.7 / 501.0 |
| Documents, 650 positions | 0.999172× | 0.985142–1.033035× | 7 / 3 | 364.474 / 365.037 | 1,783.4 / 1,780.6 |

Padded query wall was lower by 0.54% at the paired median and higher in four
pairs. Padded document wall was higher by 0.083% at the paired median. The
five quads resolve the registered decision for these two Q8_0 forwards on the
qualified Radeon.

## Fidelity and guards

The six untimed qualification cells passed 50 named tests and subtests with
zero skips. Raw tails and canaries, tiny layer snapshots, Q8_0 batch layouts
and both real workloads matched exactly across arms. Minimum same-file CPU
cosine was 0.999999999997. The two clock-disabled fit cells passed on the
scored binaries. All 160 scored-process forwards, including warm-ups,
reproduced the frozen query and document FP32 bits. Query activity was
653 dispatches, two copies and three submit/wait pairs. Document activity was
1,162 dispatches, two copies and 154 submit/wait pairs.

The scored run accepted 20 cells, with one childless start refusal at load1
4.09082 and zero execution voids. The refusal was retried at the original
position. The scoped 4.0 cap was approved on 2026-11-23 because workstation
load blocked the earlier 1.0 window. The earlier qualification refusals and
the stopped 1.0 quiet wait are retained separately. Accepted launch load was
2.399414–3.978516 and cumulative quiet wait was 355.0 s. The exclusive guard
required 8 GiB host and 2 GiB Radeon free, with 4 GiB caps on process-attributed
and device-wide delta bytes. Peak attribution was 622,825,472 B in both arms.
Peak device-wide delta was 676,831,232 / 633,843,712 B (A/B), and host RSS
peaked at 941,268,992 / 941,244,416 B. Accepted cells recorded zero child
swap, new host pageout, resets or validation errors.

## Evidence fingerprints

SHA-256 values identify the retained inputs and receipts.

| Artifact | SHA-256 |
| --- | --- |
| Q8_0 GGUF | `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc` |
| Prepared-ID export | `06287dc21059bf44d585dba53d4b12b1bc4aa97d74fe0eea2ba813fa7fa21e0a` |
| Stock / padded SPIR-V | `aee8d9454fa3af610726ecdd54b0d4074a1809bc8eb47867ca956c2ec80b92dc` / `7f6f730c8ff2c330d8d1de0caea6fc14d31a0fae50b5f917441798601ce2025a` |
| Qualification / clock-disabled fit | `5b0c85395a8e2cf1c9f0888c9e9e22b241a2fae598fa0484a2ea8d2b06a8b275` / `9987e8c9957b83e0784492702a61a45c85b261640dfc07301472d63684a91b35` |
| Scored manifest | `148a177b701d08a2633259bd34892df32dbba945187ba4f5d5b1f172f83a44bc` |
| Scored summary | `13f0673e9b6962774a4b1b06c88fb670662439e8d51d8d92f20463f0d943568b` |
| Attempt ledger | `999f1557fdfdc06924c160b7182c070b283e44a9a122926e7f6906ed3fd551b2` |
| Run evidence hash list | `0fa058d62cfcc0d302d71e68a0bcaff281730d5bf2c4c4aa78524e6e52c79a70` |
