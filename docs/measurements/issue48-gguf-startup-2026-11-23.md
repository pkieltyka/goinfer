# EmbeddingGemma 2 GGUF desktop CPU startup (#48)

On 2026-11-23, goinfer **v0.6.5** was measured on a desktop CPU with pbrain's
real verify → open → first query → close path and with goinfer's loader split
benchmark. This record is committed on **v0.6.6 / current main**; the measured
source is the v0.6.5 tag. `embeddinggemma2/gguf.go` and its test are unchanged
between v0.6.5 and v0.6.6, but no v0.6.6 process was run. The numbers below
prove v0.6.5 on this host and configuration, not v0.6.6 or another host.

## Host and method

- **Hardware:** AMD Ryzen 9 7950X, 16 physical cores / 32 logical CPUs,
  124.9 GiB RAM, Linux 7.3.5. The pbrain timed binary was built with Go 1.27.3;
  the goinfer split binary with Go 1.27.4.
- Timed commands ran on CPUs 0–3 with `GOMAXPROCS=4`, `CGO_ENABLED=0`,
  `GOWORK=off`. The run held the exclusive timing lock; every cell recorded
  load and `MemAvailable`. Load1 was 1.62–1.74 (void above 8.0) and
  `MemAvailable` 102.8–103.6 GiB (void below the fit + 2 GiB margin). No cell
  was void.
- **pbrain scratch:** throwaway worktree at pbrain `c51cc32a89e8c66686cb347762e751209b4dbd05`
  (v0.4.1) with only `go.mod`/`go.sum` bumped to goinfer v0.6.5. pbrain's
  release registry has Q8_0 only; the UD-Q6_K_XL row was added by a scratch
  entry (file name, size and SHA-256 below) applied through `-overlay`. The
  pbrain checkout was not edited.
- **Models** (Unsloth revision `031f0d4b35536f69ab3509d4893c923264fcf253`):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `embeddinggemma-2-Q8_0.gguf` | 309,855,520 | `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc` |
| `embeddinggemma-2-UD-Q6_K_XL.gguf` | 248,812,032 | `dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3` |

- **Rule 3:** five fresh processes per quant through `embed.Verify`, registry
  `Open`, one `Embed(Query, …)`, and `Close`. The sample is the benchmark's
  `ns/op`; peak RSS is GNU `time -v`'s maximum per process; settled RSS is
  `/proc/self/status` after the diagnostic query and GC with the model live.
- **Split decode:** goinfer v0.6.5 `BenchmarkGGUFVerifiedStartup`, five
  iterations per mode, `taskset -c 0-3`.

## Rule 3 (pbrain path)

| Quant | Samples (ms) | Median | Max | Peak RSS (MiB) | Settled RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Q8_0 | 570.995, 563.507, 558.497, 555.934, 569.384 | **563.507** | 570.995 | 1,032.4 | 1,032.9 |
| UD-Q6_K_XL | 511.324, 522.120, 519.313, 520.353, 534.512 | **520.353** | 534.512 | 943.4 | 933.1 |

One stage run per quant (ms): Q8_0 verify 211.19, decode runtime 273.28,
first query 75.76, close 0.02; UD-Q6_K_XL 172.18, 274.99, 73.41, 0.02.

## Split decode

The token-table and other-tensor jobs overlap in packed-parallel; the columns
are separately labelled and **not additive**. `weights` is the total
weight-decode wall for that mode. `load` is the whole loader call and contains
work beyond `weights` that this benchmark does not attribute: 131.6 ms for
Q8_0 and 131.1 ms for UD-Q6_K_XL. This record therefore does not claim that the
loader has no other cost.

| Quant | Mode | token-table ms | other-wall ms | weights ms | load ms | verify ms | first query ms | peak / settled MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | packed-parallel (shipped) | 19.98 | 54.65 | **54.66** | 186.3 | 115.3 | 69.85 | 1,395 / 1,033 |
| Q8_0 | packed-serial | 13.40 | 111.5 | 124.9 | 256.7 | 115.0 | 67.63 | 1,395 / 1,069 |
| Q8_0 | dense-serial (pre-packed control) | 142.2 | 138.9 | 281.1 | 413.4 | 115.3 | 67.95 | 1,396 / 1,395 |
| UD-Q6_K_XL | packed-parallel (shipped) | 16.11 | 70.08 | **70.09** | 201.2 | 92.46 | 68.66 | 1,451 / 939.3 |
| UD-Q6_K_XL | packed-serial | 17.66 | 180.6 | 198.3 | 338.8 | 92.35 | 66.52 | 1,451 / 1,009 |
| UD-Q6_K_XL | dense-serial (pre-packed control) | 225.6 | 187.0 | 412.6 | 548.3 | 92.32 | 66.53 | 1,452 / 1,362 |

The higher split-benchmark peaks include the raw file bytes and the model in
its own process; its settled figures (1,033 Q8_0, 939.3 MiB Q6_K_XL) match the
pbrain path.

## Targets

- **CPU rule 3 <600 ms:** PASS for both quants, every sample. Q8_0 worst 571.0
  (29 ms margin); UD-Q6_K_XL worst 534.5.
- **Decode ≤300 ms Q8_0 / ≤360 ms UD-Q6_K_XL:** PASS on the shipped
  packed-parallel path (54.66 / 70.09 ms) and on packed-serial (124.9 / 198.3).
  The dense-serial control misses only for UD-Q6_K_XL (412.6 ms); dense-serial
  is the pre-packed path, not the shipped loader.
- Q8_0 rule 3 is flat against the v0.6.2 pbrain median (562.9 → 563.5 ms);
  UD-Q6_K_XL improved (540.6 → 520.4 ms). Decode is no longer the dominant
  cost: verification, runtime setup and the 70–76 ms query account for most of
  rule 3.

## What this proves, and the applicable prior evidence

The v0.6.5 run directly proves, on the host and configuration above, that the
shipped packed-parallel loader meets the decode targets for both quants and
that the pbrain verify → open → first query → close path is under 600 ms for
every one of the ten fresh-process samples. It does not prove v0.6.6 process
timings, other hosts, cold-cache or download cost, GPU paths, or independent
tokenizer parity. The separate untimed v0.6.5 checks below cover the
loader/ReaderAt contract and same-ID numerical parity only.

Applicable existing evidence for the loader itself:

- **#50, merged `4b38d15bfeea3fd347ed9ef880fe6d294f8d9308`.** The packed loader
  joins in-flight reads before returning an error and has tests for packed-row
  ownership, byte parity, and refusal after accelerator close. The ReaderAt
  contract holds: the loader does not retain the supplied reader or backing
  bytes after construction, and the encoder exposes the tokenizer it uses.
- **CPU same-file parity at v0.6.2** (pbrain `d5129b3b880ed0f361f07f8876700ebd60ab0ddd`):
  goinfer's vectors on each quantized file against the pinned llama.cpp CPU probe
  on the **same file** — 22 reference texts and 59 passages, at 768 and normalized
  512 dimensions, 162 comparisons per quant. Minimum cosine Q8_0
  `0.9998506099`, UD-Q6_K_XL `0.9991069286`. This is the same-file comparator
  (goinfer against the reference implementation on the same quantized file), not
  a v0.6.1-vs-v0.6.2 cross-version comparison and not the quant-against-Google
  comparison. The latter, in the v0.6.1 table's 59-passage figures, was min
  0.999839 / median 0.999923 for Q8_0 and 0.998660 / 0.999423 for UD-Q6_K_XL; it
  measures how far quantization moves a vector, not parity. These are prior
  evidence at v0.6.2 (same-file) and v0.6.1 (Google).

### v0.6.5 untimed checks: same-ID parity and ReaderAt

- **ReaderAt/loader:** the focused `TestLoadGGUFEncoderReaderAtReal` passed for
  both GGUFs, with `TestLoadWeightsDrainsFailedReaderAt` (2.5 s). It checks the
  packed table is not widened, the tokenizer accessor, packed rows bit-identical
  to the dense loader, token IDs, and the failed-reader drain. This is loader
  evidence, not pbrain numerical parity.
- **Same-ID numerical parity, 22 reference texts + 59 passages:** 81 rows and
  162 comparisons per quant (768 and renormalized first-512, both sides
  normalized over the truncated vector, as in `quality.py`).

| Quant | Min 768, reference / small | Min first-512, reference / small | Below 0.999 | Prior v0.6.2 min |
| --- | --- | --- | ---: | ---: |
| Q8_0 | 0.9998516351 / 0.9998506099 | 0.9998598651 / 0.9998577758 | 0 | 0.9998506099 |
| UD-Q6_K_XL | 0.9996140114 / 0.9991069286 | 0.9996187330 / 0.9991266965 | 0 | 0.9991069286 |

  This is **same-ID** numerical parity: the 22 reference and 59 passage token
  IDs are generated by the v0.6.5 pbrain run and handed to the pinned probe.
  The 22 reference IDs are not compared to an independently pinned PreparedIDs
  set, so this does not establish tokenizer parity. `TestGemmaParity` on the
  intended pack is the separate tokenizer-ID check; it was not run here.
- **Provenance:** prepared-ID TSV SHA-256
  `c84d9f86a0bca593572b73c0c15cc1921f2c7617087efcc0dc1268cc68167081`
  (both quants); comparator SHA-256
  `afde877d5f3e98d4514deeae59b00b387e9bc8a8691b2bb03e390f8d0880ff56`;
  probe SHA-256
  `91f1116273f17d7ad3ab403a244903b223844035f3036b51fee01d150a27fc87`
  at llama.cpp `8d60873dfb8619f2b5ed31adfbab07cccb40db5c`; pbrain parity test
  binary SHA-256
  `f526ed4332dc022d3e2839da5a5812f6679ebf3002363027ca84e947b5028a4e`
  (Go 1.27.3); goinfer v0.6.5 `897943c7a`. Output hashes: Q8 vectors
  `f8e43571…`, reference `608fc323…`; Q6 vectors `de7a268c…`, reference
  `7d4d6481…`; `parity-summary.json` `25879268…`. Guard: load 1.33→3.45,
  MemAvailable 108.2→108.4 GiB, `SwapFree` +328 KiB. No paging-counter
  snapshot was retained, so the record makes no pageout claim.

## Reproduce

Build the split benchmark from the v0.6.5 tag and run it with the two files in
one directory:

```sh
git -C <goinfer> checkout v0.6.5
(cd <goinfer> && GOWORK=off CGO_ENABLED=0 go test -c -o eg2.test ./embeddinggemma2)
taskset -c 0-3 ./eg2.test -test.run '^$' -test.bench '^BenchmarkGGUFVerifiedStartup$' \
  -test.benchtime 5x -eg2-reader-gguf-dir=<dir with both GGUFs>
```

For the rule-3 path, use a disposable pbrain worktree at `c51cc32`, bump only
`go.mod`/`go.sum` to goinfer v0.6.5, add the Q6 row as a small scratch entry
with the file, size and digest above, apply it and the evaluator with
`go test -overlay`, then run five fresh processes per quant:

```sh
taskset -c 0-3 <pbrain-test> -test.run '^$' \
  -test.bench '^BenchmarkGemmaEvalFirstQuery$' -test.benchtime 1x
```

with `PBRAIN_EVAL_MODEL`, `PBRAIN_TEST_GEMMA2_PACK` and
`PBRAIN_ALLOW_GEMMA2=1` set per arm. The run retained its raw sample outputs,
cell load/memory, GNU `time -v` output and provenance; this record carries the
exact numbers. The retained aggregates are `samples.json` SHA-256
`cf3d1d0265699612e4117eae68e7b10ebb8c4ed540a788bc30eec784e16cb1b1` and
`split.out` SHA-256
`0fba6b5adc08a51079000eb5b89b87629fb3b684851c8b0cfb932a6e2603f0bf`.
