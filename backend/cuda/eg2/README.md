# CUDA text embeddings with EmbeddingGemma 2

For a Go application that needs only CUDA text embeddings, build with the `cuda` tag and blank-import
`github.com/pkieltyka/goinfer/backend/cuda/eg2` in place of `github.com/pkieltyka/goinfer/backend/cuda`.
The text-only package belongs to the existing CUDA Go module and links into your application binary.

The leaf registers CUDA text accelerators only. Image and audio towers can run on the CPU if the application
loads a compatible checkpoint with the required tower weights. The tested text-only GGUF and MLX checkpoints
cannot supply those towers. [Server embedding inputs](../../../docs/server.md).

In a stripped, cgo-free Linux/amd64 build of pbrain, substituting just the import reduced the executable from
49.4 MB to 20.1 MB, about 29 MB. pbrain's text index still uses the full CUDA import.
[Build-size measurement in PR #119](https://github.com/pkieltyka/goinfer/pull/119).
