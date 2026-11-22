- **EmbeddingGemma 2 Vulkan memory and attach errors.** Use `vulkan.Probe` with `ProbeOptions.IncludeMemory` to
  request estimated headroom before loading a model. An attached Vulkan accelerator implements the optional
  `embeddinggemma2.DeviceMemoryReporter`. A bare unlinked backend now returns `ErrBackendUnavailable`.
  Vulkan also distinguishes no match, multiple matches and native allocation failure.
