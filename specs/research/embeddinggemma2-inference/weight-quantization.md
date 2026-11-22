# EmbeddingGemma 2 weight quantization for pbrain

## Checkpoint and variants

The local Google snapshot `f9972ac22e696a105c482202646979e33e4b5a1e` contains a 1,488,915,288-byte `model.safetensors`. Its header lists all 413 text-model tensors as BF16, with 542,005,296 bytes of text weights by their offsets. Goinfer's current text encoder loads those weights as FP32 for compute. Google's model card describes 740M parameters for the full multimodal model and 270M for text alone. It recommends BF16 or FP32 compute because FP16 activations can overflow. [Google model card](https://huggingface.co/google/embeddinggemma-2#4-numerical-precision)

These are text artifacts from Unsloth revision `031f0d4b35536f69ab3509d4893c923264fcf253`. Sizes are exact bytes reported by the model repository's LFS metadata as checked on 2026-11-21. The modality projector is separate and pbrain does not need it. The saved size includes tokenizer metadata and is distinct from the live memory required by a runtime. [Pinned Unsloth artifacts](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/tree/031f0d4b35536f69ab3509d4893c923264fcf253)

| Text GGUF | Bytes | LFS SHA-256 |
| --- | ---: | --- |
| BF16 | 557,950,240 | `f315cbbb30dd487e44d501c8902abe88808755e43753a96beed1964f0a48aa4f` |
| Q8_0 | 309,855,520 | `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc` |
| UD-Q6_K_XL | 248,812,032 | `dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3` |
| UD-Q5_K_XL | 210,055,680 | `a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41` |
| UD-Q4_K_XL | 175,673,856 | `ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493` |

## Quality evidence

AtomicChat's model authors compared Unsloth variants on 1,000 held-out multilingual query/passage pairs cut from 81 Wikipedia articles. In that run, the top result matched the BF16 ranking for Q8_0 on 98.7% of queries, for Unsloth Q6 on 97.0%, Q5 on 93.9%, and Q4 on 87.1%. They report that the three vendors' Q8_0 tensors match. Agreement with BF16 is a ranking stability metric. A different top result can still be relevant. AtomicChat separately reported similar labeled recall for its own Q4 and BF16 layouts. Different tensor layouts and a Wikipedia-derived evaluation make that evidence suggestive for pbrain. [AtomicChat measurements and method](https://huggingface.co/AtomicChat/embeddinggemma-2-GGUF#how-these-compare-to-other-builds)

The same report observed no speed advantage for its smaller files in its llama.cpp run on an M4 Max with flash attention off. This is a reason to measure, especially on the RTX PRO 6000 where all candidates fit. A low-bit file can save disk and device memory, and any speed gain depends on the kernel's ability to operate on packed weights with a suitable activation precision. Goinfer's current EmbeddingGemma 2 loader takes Hugging Face safetensors. A matching GGUF type in another goinfer model family does not provide EmbeddingGemma 2 GGUF loading or packed math automatically. [AtomicChat runtime setup](https://huggingface.co/AtomicChat/embeddinggemma-2-GGUF#how-we-measured-it)

## Experiment on the frozen pbrain corpus

1. First prove full-precision CUDA and record its unchanged measurement for [#12](https://github.com/pkieltyka/goinfer/issues/12). Profile and consider native BF16 compute before investing in low-bit kernels for this card.
2. For [#13](https://github.com/pkieltyka/goinfer/issues/13), pin llama.cpp, input preparation, dimensions and artifact digests. Establish the BF16 GGUF baseline with the same source revision or document any mismatch. Test Q8_0 as the low-risk candidate, Q4 as the compression bound, and use Q6 to evaluate the middle of the tradeoff. Add Q5 if the decision needs it. Keep weight and activation precision separate in each record.
3. Rebuild each arm's SQLite vectors and query with those same weights. Record and compare per-query Recall@k, MRR, wrong accepts and misses on never-tuned queries. Check drift and nearest-score margins as diagnostics. Compare frozen-cutoff behavior before considering recalibration. Require index identity to encode the model variant, digest, and dimension if any quantized model is adopted.
4. Record both singleton latency and document throughput on the same RTX, along with model load and actual peak memory. Separate model comparisons from backend comparisons. Preserve results even if a candidate saves memory and costs latency. Keep a native packed-kernel effort in goinfer or aikit behind a measured quality and performance decision.

## Local result

The [RTX retrieval and timing record](../../../docs/measurements/pbrain-embedding-quant-2026-11-21/README.md) completes the frozen whole-note encoder comparison. Q6 had the highest top-one agreement with BF16 among the three compressed files and kept every labeled top-five hit. Q8 and Q4 had similar aggregate document throughput. Q4 changed 14 top results and lost one labeled top-five hit. Q6 saved about 309 MB of disk and had lower sampled device memory than BF16. Its aggregate throughput improved, while its median individual document call was slightly slower. Each weight file generated all of the document and query vectors for its arm.

All four variants fit easily on the RTX. A native packed-format implementation still needs its own Go loading and compute tests with variant-specific index identity. The current results supply an independent quantization reference for that decision.
