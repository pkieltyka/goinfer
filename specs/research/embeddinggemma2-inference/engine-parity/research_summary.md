# Engine support and evaluation equivalence

## Bottom line

Use PyTorch Transformers plus Sentence Transformers as the independent semantic reference, and pinned llama.cpp as the first cross-device performance peer. vLLM, SGLang and MLX-VLM have model-specific implementations for `google/embeddinggemma-2`. They provide useful scheduling and batch-validation examples, although support on a particular backend must be proven separately. Preserve pbrain's prepared-token contract and start from a verified full-precision comparison before evaluating reduced precision or native batching.

## Key findings

| Engine | Evidence and role |
| --- | --- |
| Transformers and Sentence Transformers | Transformers v5.19.0 documents the model-specific bidirectional encoder and learned projection. Sentence Transformers 6.1.0 or later applies model prompts, mask-aware mean pooling and normalization. Use the same HF snapshot and an FP32 CPU run as the independent reference. [Versioned model documentation](https://huggingface.co/docs/transformers/v5.19.0/en/model_doc/embedding_gemma2) |
| llama.cpp | Model support merged in `4fbc76d` on 2026-10-06. The inspected local peer, `e6d624b9a4b95c6b52f5ca384d70b75581f9821a`, has the dedicated symmetric, no-cache architecture and a normalized embeddings endpoint. Official GGUF releases are BF16 and Q8_0. Use a pinned conversion of the same source checkpoint to construct an F32 comparison arm. [Merge](https://github.com/ggml-org/llama.cpp/pull/30054) [Pinned model graph](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/src/models/gemma-embedding2.cpp) [GGUF files](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) |
| vLLM | The model-specific pooling implementation merged as `02b8391` on 2026-10-06. Its PR includes text-only serving and singleton-versus-batch checks. The model-specific configuration documents Triton attention as its validated default for heterogeneous head dimensions. That is a scheduling and correctness reference, with AMD results dependent on the specific runtime and device. [Merge and tests](https://github.com/vllm-project/vllm/pull/60254) [Attention selection](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/config.py) |
| SGLang | Dedicated support merged in `d1ba943` on 2026-10-08. The merge shows a failed AMD ROCm CI entry, which does not establish whether that failure came from this model. Use the implementation as an additional batching/serving reference. [Model PR and CI](https://github.com/sgl-project/sglang/pull/42782) |
| MLX-VLM and vLLM Metal | Native MLX-VLM support merged as `fec3f50` on 2026-10-06. Its pinned README documents output semantics, text-only loading and Apple M5 Max batch and precision validation. vLLM Metal documents a separate unquantized text-only path. Both are Apple references with no local hardware verification here. [MLX-VLM merge](https://github.com/Neywa-Labs/mlx-vlm/pull/2446) [Pinned MLX model documentation](https://raw.githubusercontent.com/Neywa-Labs/mlx-vlm/fec3f50/mlx_vlm/models/embedding_gemma2/README.md) [vLLM Metal documentation](https://docs.vllm.ai/projects/vllm-metal/en/stable/text_embedding_pooling/) |
| Ollama and LiteRT | The Ollama catalog includes both MLX-tagged and unmarked text variants, so fix the tag, artifact digest and backend when using it. Google documents LiteRT's model-specific edge support and adaptive graph sizing as design prior art. Keep quantized edge results separate from native desktop GPU timing. [Ollama tags](https://ollama.com/library/embeddinggemma-2/tags) [Google AI Edge implementation](https://developers.googleblog.com/google-ai-edge-with-embeddinggemma-2/) |

The core comparison needs exactly matched role bytes, BOS/EOS, window/masks, effective sequence lengths, text-only tower loading, pooling and normalization, precision, and per-request shape. Use a prepared-ID fidelity lane and a separate served timing lane. pbrain deliberately prepares its own token IDs, including a 2,048-token limit, and currently calls `Model.Embed` directly. Wire the goinfer accelerator at that boundary before measuring application gain. [pbrain adapter](https://github.com/pkieltyka/pbrain/blob/b4f8aa1ec5fd389d93f6b64e9bfccb5604b0dd5e/internal/embed/gemma2/gemma2.go) [Google developer guide](https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/)

## Details

- [Peer protocol](./peer-protocol.md) covers exact preparation, compute precision, batching, and how to extend the project harness.

## Open questions and gaps

- No engines or parity fixtures were executed in this source review. There is no new peer speed result.
- Parity of a locally produced F32 GGUF with the installed Hugging Face snapshot remains to be checked.
- The public support evidence does not establish a working ROCm path for this exact Radeon, or a speed ranking on any of the target devices.

## Sources

- [Google model card](https://huggingface.co/google/embeddinggemma-2) and [EmbeddingGemma 2 paper, v2](https://arxiv.org/abs/2610.37364v2) establish model identity and architecture.
- [Pinned llama.cpp embedding API](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/tools/server/README.md#post-v1embeddings-openai-compatible-embeddings-api) documents mean-pooled service comparison mechanics.
- [Original EmbeddingGemma model card](https://huggingface.co/google/embeddinggemma-300m) establishes the distinct Gemma 3-backed model.
