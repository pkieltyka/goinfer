# EmbeddingGemma 2 owned Markdown replay on Apple M4

The run began on 2026-11-27 at 00:37:16 UTC on a base Apple M4 with macOS 27.2.1 (26B131). A cgo-free Darwin ARM64 test binary built with Go 1.27.3 exercised `fb1282740cc283b32116bc84c684ba7433565e84` (tree `9fa5f5f55e25503245d831784abb131f795640bb`). Both local checkpoints used the explicit `metal-eg2-batch` attachment and reported profile `metal-eg2-combined-v1`.

Use the [fixture and runnable commands](../../testdata/embeddinggemma2-markdown-replay/replay-v1/README.md) to reproduce the public prepared-ID replay. The corpus SHA-256 is `32cc8691001fc23a360b4bd264c1d1a3d1605295b98d2cabc9d52f4dee2b69d5`. The manifest SHA-256 is `52dbe674fe616b75a81eb589e3d5db4314f6a95790c27eef984d3bffd87fefdb`. Both formats matched the same ordered sequence of 219 prepared lengths and per-document ID digests in the manifest. Each replay covers 113,707 useful positions across 54 four-document groups and a final three-document group. The largest group contains 2,375 positions.

The test loaded and attached each format once, Q8_0 first. Off-clock singleton and packed calls established word equality at 768 and independently L2-normalized 512. The runner also checked rotated, split and changed-neighbour layouts on the two sentinel groups. After warming the largest group, it cleared RoPE tables before each of three replays and retained the row scratch and scalar arena. The reported public wall includes RoPE refill and pooling. Model verification, load, attachment, tokenization, output checks, and export are outside that wall.

| Pinned local checkpoint | Replay walls (s), in order | Median (s) | Range (ms) | Positions/s at median |
| --- | --- | ---: | ---: | ---: |
| GGUF Q8_0, revision `031f0d4b35536f69ab3509d4893c923264fcf253` | 21.345327, 21.358444, 21.356579 | 21.356579 | 13.117 | 5,324.21 |
| MLX8, revision `7505ef2f8ddef45efef6d060865f27989b3c9cec` | 21.399054, 21.393480, 21.398807 | 21.398807 | 5.573 | 5,313.71 |

Each replay completed 55 commands and 41,164 dispatches. The counters describe submitted and waited work. The walls are host observations for this workload, with command and dispatch counts as the device evidence. They establish no GPU phase attribution or comparative speed result.

| Checkpoint | Ordered raw FP32 768 SHA-256 | Ordered normalized FP32 512 SHA-256 |
| --- | --- | --- |
| Q8_0 | `548ea5a32914b20e2f9eb285dfc891d97955e399adf2f838cc7fbdd8ca992b92` | `7e37dc4037fc863d35e96623080fea8760ec9936f0ecee84dfe241b11c1e9a5d` |
| MLX8 | `fdfd6b306d5b9ffa066f9f27a749f49e76b71eb0bad8ba26b09b85c0353def52` | `ea02e17a0c5bb25b1e9329d1777ff07201d06385b4e63ddcb2a2127855eeb017` |

Both formats reproduced their own singleton words in all three replays. The 24-entry native manifest SHA-256 is `386589dcd5960298dc32ac700d6059f0f22b44a95abd24044c5da93b75512ebb`, covering the input and source bindings, raw arrays, cell receipts and native log. The independently reduced 512 prefixes reproduced the saved bytes.

The exclusive run ended at 00:41:32 UTC and both device ledgers closed. All 15 guard readings passed. One-minute load was 0.83–1.65 and reclaimable memory was at least 14,252,965,888 bytes. Swap stayed at 954,990,592 bytes and Swapouts at 2,990,026 pages. Current and peak task compression were zero, thermal readings were clean and the three sampled process censuses were empty. The batch fit observations apply to the admitted 2,375-position maximum of this fixture.
