# EmbeddingGemma 2 pbrain index baseline

On 2026-11-21, pbrain indexed the same 32 current Markdown notes in fresh scratch brains. Both arms generated 59 passages and embedded 9,271 prepared tokens. The token-ID digests matched in order, all vectors were finite unit-length 768-dimensional vectors, and the minimum CPU/RTX cosine for corresponding cached vectors was 0.9999999999985. Both arms reported zero reused passages. The CUDA arm recorded 59 successful CUDA forwards. Both published SQLite databases passed `PRAGMA quick_check` and contained 32 items and 59 passages.

| Measured time | CPU | RTX |
| --- | ---: | ---: |
| Full `pbrain index --json` process | 18.933 s | 2.811 s |
| Sum of embedding calls | 18.481 s | 1.663 s |
| Model verification | 0.203 s | 0.197 s |
| Model open, including any device attachment | 0.086 s | 0.619 s |
| Accelerator close | — | 0.067 s |
| Token preparation | 0.018 s | 0.018 s |
| Vector-cache writes | 0.015 s | 0.017 s |
| SQLite item writes and index publish | 0.007 s | 0.007 s |

These are diagnostic timings from one CPU run and one RTX run, in that order, under the exclusive goinfer timing lock. At the start of each run the host had about 89 GiB of available memory. The one-minute load averages were 4.56 and 4.73 while a separate BGE-small index was running. The host used a Ryzen 9 7950X and an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition with driver 595.112.07. Each process used `GOMAXPROCS=4`. Model verification and startup are included in the process wall time. Both processes used the same already-verified files in the OS page cache. CUDA attachment accounted for 0.541 s of the RTX model open. Phase timings omit uninstrumented work and include trace overhead where applicable.

The notes came from `internal/semantic/calibrate/testdata/retrieval/brain` at pbrain commit `b4def8fad8ea34101c369777604144fbbec2182a`. Each matched its source digest in the manifest from the [native corpus run](../pbrain-embedding-cpu-rtx-2026-11-21/README.md). pbrain split the files into passages before applying its document prompt and token cap. The two-note preflight used `decisions/index-refresh.md` and `sources/storage-field-guide.md` and got 13 matching prepared inputs, 13 CUDA forwards, and minimum cached-vector cosine 0.9999999999993. The CUDA arm also failed visibly with `CUDA_ERROR_NO_DEVICE` when the accelerator was hidden.

The pbrain benchmark build used local commit `48874652e642c33a1f1f947e1a691f1565317a44` and a workspace pinned to goinfer root and CUDA modules at `44c5aa2ab3f76ec120307817534bad2b3b21d9e3`. Go 1.27.3 built the CPU arm with `CGO_ENABLED=0 -tags pbrain_bench` and the RTX arm with `CGO_ENABLED=1 -tags cuda,pbrain_bench_cuda,pbrain_bench`. Binary SHA-256s were `38fc16968a16e5f7c288098dc314b6a0385bfd97effa2684c5b04cdbf8d0b5f2` and `fbe4846dc7595aa88f68a1cb5e9aa5d54e0999bd48b1b520f32c62a76f54aed3` respectively. Both used the same derived, digest-verified pbrain pack from Google EmbeddingGemma 2 revision `f9972ac22e696a105c482202646979e33e4b5a1e`. The workload flag was set in each benchmark process. Indexes and caches lived in separate scratch directories.

The local run directory `~/goinfer-logs/pbe3-pbrain-index-2026-11-21` contains the launcher, verifier, immutable source manifest, per-arm CLI JSON and traces, binary build metadata, and `full-32-verified.json`. To repeat, give each build a fresh copy of the verified notes and an empty `PBRAIN_CACHE`, set the scratch brain's `semantic.model` and local `model_path`, and run `pbrain --brain <scratch> index --json` with `PBRAIN_ALLOW_GEMMA2=1` and a per-run `PBRAIN_BENCH_TRACE` path. On the measured NixOS host, set `LD_LIBRARY_PATH=/run/opengl-driver/lib` for the CUDA build. For a speed claim about a kernel change, use the registered repeated, interleaved experiment with the same backend and compare it against the corrected CUDA reference.

## Transposed-key attention application check

A 2026-11-21 diagnostic at 17:43 EST compared two fresh RTX indexes on the same 32 notes and model pack. It used pbrain `09c1dac49e76f6c988b24d74179daa84dd044b27` and the goinfer [transposed-key prototype](../pbrain-embedding-rtx-profile-2026-11-21/README.md) over `44c5aa2ab`. Both builds used Go 1.27.3, `GOAMD64=v1`, `CGO_ENABLED=0`, four Go scheduler threads, benchmark build tags, and the same verified model files. A temporary Go overlay selected the transposed kernel for passages of 128 tokens or longer in the candidate. That experimental cutoff covered 23 of the 59 passages. The documents were chunked identically in both arms.

| RTX singleton index | Packed attention | Transposed-key candidate |
| --- | ---: | ---: |
| Full process wall time | 2.700 s | 2.081 s |
| Sum of embedding calls | 1.634 s | 1.108 s |

The one-minute host load was 1.279 and 1.256 at the respective starts. Each run used a fresh index and cache, recorded 59 device forwards and zero reuse, and matched every prepared ID and vector digest in order. Both SQLite files passed `PRAGMA quick_check` and contained all 59 passages. This is a one-pair application diagnostic. The cutoff is local to the overlay check.

The local `attention-build/` and `attention-diagnostic/` directories record the source overlay, build evidence and hashes, per-cell traces, provenance and summary. The build SHA-256s were `c6c843a400f95081544cb3837eb0cce13a05ce542c9e9a459248815e0316516f` and `35b04ec183adaca9855f5836c11a95a0a7211aa3fdaf34f8a69b4924e5d46b6d` respectively.

## RTX passage pairing

The [registered fresh-index comparison](paired-registration.md) completed eight cells from 18:06:33 to 18:06:53 EST on 2026-11-21. The same RTX, Google model pack, 32-note fixture and scratch policy applied to both arms. Each run embedded 59 passages and 9,271 prepared tokens with zero cache reuse. Every SQLite index passed integrity checks. Prepared-ID hashes, passage offsets and digest, and the vector bytes stored in SQLite matched across all eight cells.

| Median across four cells per arm | Singleton CUDA | Paired CUDA |
| --- | ---: | ---: |
| Full process wall time | 2.650 s | 2.188 s |
| Sum of embedding call times | 1.650 s | 1.216 s |
| CUDA forward dispatches | 59 | 30 |

The paired arm issued 29 two-passage forwards and one singleton. The four adjacent wall-time ratios were 1.272×, 1.203×, 1.194×, and 1.212×, with a median of 1.208×. Both builds ran at four Go scheduler threads, and one-minute start loads ranged from 0.681 to 0.811.

Both binaries used pbrain `dd0b4153477fe883048a0d95429585c635fc77f3` and goinfer `688afa19d4ba6533597bae8a5c2615d7b3b4331d`, Go 1.27.3, `GOAMD64=v1`, and `CGO_ENABLED=0`. The paired build added `pbrain_bench_batch` to the singleton tags. The binary hashes were `8e9ac031444aa5adaa80ff53c1e38dfc3d0494737892fb573de2ad0676d07e88` and `2d7fec97628dfa26b0300d7e9bd948bb232bf038a99dd35f3d17eaa0106e9d69` respectively. The [native CUDA pairing experiment](../pbrain-embedding-rtx-batch-2026-11-21/README.md) measures the whole-note path separately. Each pbrain cell and its source manifest, device identity, build hashes, traces, SQLite index and summary are under the local `batch-cells/` directory.
