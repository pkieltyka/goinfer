# EmbeddingGemma 2 Q8 and Q6 on CUDA

Both Unsloth files passed the [registered correctness checks](registration.md) on
November 22, 2026. Goinfer expands their weights to FP32 at load time. Their
downloads are smaller. Observed startup was shorter for both GGUFs, and warm
CUDA latencies were similar to the original Google checkpoint.

## Correctness and retrieval

The cgo-free runs used a Ryzen 9 7950X and an NVIDIA RTX PRO 6000 Blackwell Max-Q,
driver 595.112.07, with four Go scheduler threads. Goinfer was built at
`f38f5431bfd9a845d5a5829ddb4a9fc539b02eca` using Go 1.27.3. The executable
SHA-256 was `ad5383cf30927704eb349b47af6a081f736a1a2789b62462fac4087420096708`.
The local Hugging Face cache held both files at
[Unsloth revision 031f0d4](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/tree/031f0d4b35536f69ab3509d4893c923264fcf253).
The corpus manifest SHA-256 was
`420988ad6653a9141a56ef61f12de91adc07255705426a3042f9a41cf413d732`.
It contains 32 Markdown notes and 107 queries. The verified Q8_0 file was
309,855,520 bytes with SHA-256
`6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc`.
UD-Q6_K_XL was 248,812,032 bytes with SHA-256
`dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3`.
The encoder is deterministic and
uses no sampling. [Corpus and timing setup](../pbrain-embedding-cpu-rtx-2026-11-21/README.md)

| Check | Q8_0 | UD-Q6_K_XL |
|---|---:|---:|
| Exact prepared IDs versus Google tokenizer | 139/139 | 139/139 |
| CPU–CUDA minimum cosine, corpus | 0.9999999999987 | 0.9999999999989 |
| CPU–CUDA largest coordinate difference, corpus | 2.243e-7 | 2.050e-7 |
| CUDA minimum cosine against same-file llama.cpp b12613 | 0.999845 | 0.999825 |
| CPU–CUDA boundary and long inputs passed | 14/14 | 14/14 |
| CUDA packed groups matching singletons bit-for-bit | 9/9 | 9/9 |

The boundary inputs included 511–514 tokens and a 2,655-token note. Packed groups
reached 5,027 tokens. Hiding the device produced a CUDA attachment error. The
separate `TestEG2GGUFCUDA` device test accepts `-eg2-gguf-dir` pointing at the
pinned files and checks both digests before running.

| Model | Labeled top-five hits before cutoff | MRR | Misses at 0.72 | False accepts at 0.72 |
|---|---:|---:|---:|---:|
| Google BF16 reference | 47/47 | 0.909 | 5 | 9 |
| Unsloth Q8_0, CUDA | 47/47 | 0.910 | 5 | 10 |
| Unsloth UD-Q6_K_XL, CUDA | 47/47 | 0.913 | 4 | 8 |

A miss is an answerable query that abstained or lacked a labeled hit in the top
five. False accepts count unanswerable queries that passed the cutoff. Of the
107 queries, 47 were answerable and 60 were unanswerable. Q6 showed fewer cutoff
errors on this corpus. The pbrain model selection remains outside this evaluation.

## Observed goinfer timings

| Model | CPU, 32 notes | RTX, 32 notes | RTX median warm query | RTX load + attach |
|---|---:|---:|---:|---:|
| Google original BF16 | 19.97 s | 2.74 s | 15.77 ms | 1.29 s |
| Unsloth Q8_0 | 21.23 s | 2.77 s | 15.75 ms | 0.86 s |
| Unsloth UD-Q6_K_XL | 20.90 s | 2.76 s | 15.78 ms | 1.09 s |

Each document entry sums the per-note embedding calls. Warm-query medians use
the 106 queries after the first. These encoder-only readings exclude SQLite,
chunking, and hybrid search. Load and attach use a fresh process with local
cache files. OS file-cache state was uncontrolled.

The Google values are medians from two cells per backend in the separate
November 22 three-device run at goinfer `299ac790cf9c379977145b9d06d94ed346aa9e42`.
That run timed tokenization and embedding together, using Google revision
`f9972ac22e696a105c482202646979e33e4b5a1e`. CUDA cells started at one-minute
loads of 0.87–1.00. The GGUF readings come from one correctness cell per
artifact and backend with tokenization outside the timed call. One-minute host
load at variant start was 5.05 for Q8 and 4.40 for Q6. The device probe recorded
31 °C at the start of GGUF verification. These conditions support a descriptive
comparison. An interleaved run with a shared timing boundary would be needed
to resolve small differences.

The existing matched RTX llama.cpp study describes quantized execution
separately. Q8 had the fastest queries and highest aggregate document throughput
of its BF16, Q8 and Q6 files. Q6 had the lowest sampled peak device memory of
those three. [Matched llama.cpp quantization benchmark](../pbrain-embedding-quant-2026-11-21/README.md)
