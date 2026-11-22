# GGUF reader

This reader validates the little-endian GGUF v3 index and decodes F32, BF16,
Q8_0 and Q6_K tensors. Callers check architecture and validate model-specific
metadata and tensor names. Tokenizer arrays are skipped, while scalar metadata
such as tokenizer model and BOS/EOS IDs is retained. Metadata arrays use Go
slices of the corresponding wire scalar type. Nested arrays use slices of
individually typed values. Metadata keys accept hyphens used by the
llama.cpp EmbeddingGemma 2 writer, including the `gemma-embedding2` namespace.

The parser bounds the index and retained metadata to 64 MiB each. It also
bounds entry counts, string and array lengths, nesting, and decoded tensor
sizes. Tensor reads use block-aligned chunks of at most 64 KiB.

The [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
defines the index layout and type IDs. The Q8_0 and Q6_K block layout and
dequantization were verified against the ggml authors' MIT-licensed
[block definitions](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/ggml/src/ggml-common.h)
and [reference decoders](https://github.com/ggml-org/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/ggml/src/ggml-quants.c).
