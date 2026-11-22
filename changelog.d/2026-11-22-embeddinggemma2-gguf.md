- **EmbeddingGemma 2 text GGUF.** Go applications can open Unsloth's Q8_0 and UD-Q6_K_XL
  text checkpoints with `embeddinggemma2.LoadGGUFEncoder` and attach CUDA for embedding
  and prepared-ID batching. The loader decodes the file's mixed tensor formats in Go,
  with FP32 weights for execution.
