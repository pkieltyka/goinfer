- **EmbeddingGemma 2 on AMD.** The optional Linux amd64 Vulkan backend runs text embedding
  and independent document batches with `CGO_ENABLED=0`. `Encoder.UseAutoAccelerator` lets
  an application opt into trying its linked CUDA and Vulkan backends, with CPU available
  when neither attaches.
