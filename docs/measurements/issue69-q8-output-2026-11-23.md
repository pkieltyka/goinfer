# #69 packed Q8_0 attention-output A/B measurement note

Record on current main `995971400`. The measured base is goinfer main
`b0dc987dcf9b6034ebd645a6a77fd53b4a886ad8`. The immutable v2 manifest is
`b57853e0…` and the qualification receipt is `8425662c…`. The registered 20-cell
campaign completed: all 20 cells were accepted on first attempt, and the
registered symmetric decision is **ambiguous**, so the private packed path stays
unshipped. Every pair, median/range, throughput, memory and parity figure below
comes from the retained scored receipt set after an independent read-only audit
reconstructed the group clocks and pair arithmetic. No partial run is used, no
product code or selector registration changes, and no benchmark or device rerun
was performed for this record.

## Immutable v2 and qualification

- Manifest `b57853e0cdf41efd92f08f2fcb6db0bab56a5cf83c0cc17bbfd243145efcc68e`;
  qualification receipt
  `8425662c2d08f924c5b0f62b347bbbcc8c6c29adbde0f54937aad33e10702f14`;
  qualification hash list
  `2b49c7e0a9896a92e2d3d7400b24ba80601f8a1fd56d559b3effcda87ba7c64b`.
- Runner `ee3b849a…`; test `2a1856cd…`; cell binary `a94960b8…`; qualifier
  `7e8a2fd0…`.
- Raw quality log `9897d4f0…`: PASS. `packed=false` (dense) and `packed=true`
  (packed) each made 24 O allocations, with requested and charged bytes equal
  within each arm: dense `58,720,256` B, packed `15,597,568` B. Resident
  `396,144,640` / `353,021,952` B. Cold additional A/B `8,191,076` /
  `8,191,076` B; warm additional `7,988,324` / `7,988,324` B.
- Replay log `1a77b99e…`: PASS. Reference-81 minimum CPU cosine 768/512
  `0.999999899598` / `0.999999900834`; frozen-200 `0.999999448048` /
  `0.999999459830`; layouts exact; both arms.
- Fit A result `80891110…` and fit B `8d610765…`: both completed, exit 0, with
  vector SHA `97d6b1e8…`, minimum CPU cosine 768/512
  `0.999999448048` / `0.999999459830`; cold and warm scratch arrays identical
  between arms; no swap growth and no pageout. Peak RSS A/B `1,801,512` /
  `1,640,200` KiB; peak GPU `1,782` / `1,752` MiB.

## A/B mechanisms

- **A** uses the existing `cuda-eg2-fp16` selector with dense attention O
  weights: `58,720,256` B raw/charged, `396,144,640` B resident.
- **B** uses the private `issue69-private-q8-output-timing` factory with only the
  24 Q8_0 O weights and their projection changed: `15,597,568` B raw/charged,
  `353,021,952` B resident.
- Both arms are one cgo-free scratch executable with the sealed private
  stage-two overlay; the prior reviewed O CUDA/PTX bytes are identical. B
  decodes signed bytes with one half scale per 32 values during the tile, uses
  one ascending-K FP32 FMA chain, then the existing positive-zero epilogue.
  Neither arm stages a new forward buffer.

## Fixture

The pinned Q8_0 GGUF is `6f1bd4ac…` (`309,855,520` B). Both arms use the same
200 prepared inputs in the exact 13 ordered physical groups, `100,066`
positions; prepared IDs `97c426da…` and CPU slice `fdb04440…`. A fresh process
attaches once and discards a 13-group warm-up. Each measured interval covers one
public `EmbedIDsBatch` call through device completion, full readback and
host-normalized vectors. Loading, attaching, plan computation, qualification,
digests, JSON and reporting fall outside those clocks; no retry or split enters
the measured API.

## Scope

CUDA only, on the NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
(driver 595.112.07, 97,887 MiB total, device
`GPU-d3752b75-6d35-9d0f-a67e-c201d9c27381`). The private overlay extends tests
only; there is no product, default, selector-registration or workspace change.
With an ambiguous speed decision the packed arm remains unshipped: this record
adds no product code and no selectable dense or packed default.

## Bit-identical and memory findings

- Both A and B reproduce the retained released raw FP32LE vector digest
  `97d6b1e8…` in their fit cells, and the raw/replay quality cells report exact
  layouts and the same minimum cosines. This is bit-identical output, not a
  timing result.
- The output representation and resident weights each fall by `43,122,688` B
  (`58,720,256` -> `15,597,568`; `396,144,640` -> `353,021,952`). Cold and warm
  scratch plans are identical between arms; the 13-group warm scratch peak was
  `876,618,920` B, and the fit guard retained 1 GiB of headroom beyond its 6 GiB
  physical cap. No new forward buffer is staged. Qualification fit peaks were
  A/B `1,801,512` / `1,640,200` KiB RSS and `1,782` / `1,752` MiB VRAM; the
  scored peaks are reported separately below and are not the same measurement.

## Timed comparison

Five ABBA quads produced 20 clean fresh-process cells and ten matched packed/dense
pairs in accepted order. All 20 were accepted on attempt one: no start void,
execution void or retry, and every process completed with exit 0. A is dense and
B is packed, so a paired ratio below `1` means the packed wall was lower.

| Pair | Cells (accepted order) | Dense wall (s) | Packed wall (s) | Packed / dense | Lower wall |
| --- | --- | --- | --- | --- | --- |
| 1 | 01-A, 02-B | 2.161014076 | 2.245811757 | 1.0392397634 | dense |
| 2 | 03-B, 04-A | 2.185859416 | 2.135446559 | 0.9769368256 | packed |
| 3 | 05-A, 06-B | 2.158374565 | 2.143950749 | 0.9933172786 | packed |
| 4 | 07-B, 08-A | 2.180927620 | 2.222796742 | 1.0191978503 | dense |
| 5 | 09-A, 10-B | 2.196483898 | 2.220513385 | 1.0109399787 | dense |
| 6 | 11-B, 12-A | 2.165167025 | 2.157190931 | 0.9963161761 | packed |
| 7 | 13-A, 14-B | 2.160032780 | 2.242546431 | 1.0382001846 | dense |
| 8 | 15-B, 16-A | 2.185141999 | 2.169280381 | 0.9927411500 | packed |
| 9 | 17-A, 18-B | 2.142667297 | 2.171403401 | 1.0134113700 | dense |
| 10 | 19-B, 20-A | 2.228956816 | 2.169981815 | 0.9735414340 | packed |

- Median paired packed/dense ratio `1.0036280774`; range `0.9735414340` to
  `1.0392397634`; dense had the lower wall in 5 pairs and packed in 5.
- Median wall was dense `2.1730473225` s and packed `2.170692608` s. Median
  measured throughput was dense `46,049.303627` and packed `46,098.655499`
  positions/s; observed per-arm ranges were `44,893.646787`–`46,701.604183` and
  `44,556.717493`–`46,859.519653` positions/s.
- The registered symmetric rule is: packed lower wall only if the median is at
  most `1/1.05` (`0.952381…`) and at least 9/10 pairs favor packed; dense lower
  wall only if the median is at least `1.05` and at least 9/10 pairs favor dense;
  otherwise ambiguous. Neither condition holds (median `1.003628…`, pairs split
  5/5), so the applied decision is **ambiguous** and the private packed path is
  not shipped.

**Parity.** All 20 cells returned FP32LE vectors that re-encode to the retained
released digest `97d6b1e8…`; the independent audit re-encoded all 4,000 vectors
and recomputed all 260 group clocks. Every cell used the frozen 13-group order,
200 inputs and `100,066` positions, and minimum CPU cosine was `0.999999448048`
at 768 and `0.999999459830` at 512 in every cell. Output is bit-identical for
both arms; this is not a timing result.

**Memory and guards.** Scored peak host RSS was dense `1,834,572` and packed
`1,812,192` KiB; sampled peak VRAM was `1,782` / `1,752` MiB. These are scored
peaks, distinct from the qualification fit peaks above. Launch load1 spanned
`0.7856`–`0.9912`, minimum available host memory was `104,919` MiB, and there
were zero child swap, zero new host pageout and zero timing voids. No cell
approached the 25-second per-cell cap.

## Evidence hashes

- Manifest `b57853e0cdf41efd92f08f2fcb6db0bab56a5cf83c0cc17bbfd243145efcc68e`;
  qualification receipt
  `8425662c2d08f924c5b0f62b347bbbcc8c6c29adbde0f54937aad33e10702f14`;
  qualification hash list
  `2b49c7e0a9896a92e2d3d7400b24ba80601f8a1fd56d559b3effcda87ba7c64b`.
- Runner `ee3b849a12322a039ee6170ab534ec4fec9c308c943db5d01760031a902b3e80`;
  test `2a1856cd78b724cc35eccebd5a6d7705d500f64006b35a65892c136054ddd9e8`;
  cell `a94960b8dd0b650e2bd9a4fff9990117a0b8c0bb2dd2d620882e62d55b7f5c7a`;
  qualifier `7e8a2fd0c756de609b61ecaba11ffa819bc1827bdfa2c63df3e275011e1a6041`.
- Raw log `9897d4f02aaabce2f495562b0829bb2522acb54531eee0e0504d88ba6d250468`;
  replay log `1a77b99e88ef01b9678b818dad9ff7cebd0d221a9d05a258a3baf3c693897e51`;
  fit A `80891110f95a3c1ea36816cd7a809ed443f22b9a83d57435c172af3d0b25ab66`;
  fit B `8d6107655b81261566b483fa320ea2519cd56b8a495b1a8a0c87959b719ac9f2`;
  GGUF `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc`;
  prepared IDs `97c426da8c155b04bf0b34e270dcfa5e8cf18bfc1343e926f63e638fea6bcd22`;
  CPU slice `fdb04440fe0d678cc36cb913d703b32f4f6a6d99ed943a30461166bd6ec68c8b`.
- Scored summary `5649f28e0f597cd3528a033f963b1bee7b341ca42306b40bce9577193ed5db17`;
  scored ledger `9569ac3e883d3f0a985ca4d01fa87b108ba6fb2333142008f45de35f213c8e37`;
  audit `705ccf208d05d16a4f24185c3ee1bc06647a1c7f5eaeff51512ca4f3ee880804`;
  audit log `e9654eda3d90d33ffcb1dc0d69dd806ad2c6b9931ce21c5ec15b179b795394fe`;
  score launch log `1b77008569450c37035ebc4fad02210b4c27435ec711ef2e9500a6a0eabd7a88`;
  full run seal `5ada26f9a019fab8151423ec87b8182a56481bb86f1fca51d48c0aee4197d136`
  (66 entries, all verified).
- Stable hashes only; no private session paths.
