# EmbeddingGemma 2 inference for goinfer

Research snapshot from 2026-11-21, before the CUDA integration. Subsequent parity and timing results are in the [combined CUDA measurements](../../../docs/measurements/pbrain-embedding-rtx-combined-2026-11-21/README.md).

## Bottom line

First freeze pbrain's prepared-token workload and validate it against FP32 Transformers with Sentence Transformers. Use pinned llama.cpp with a same-snapshot F32 GGUF as the first performance peer. For the RTX PRO 6000, start with a resident FP32 CUDA text encoder that reuses goinfer's existing accelerator interface, tower primitives, and the Metal encoder's attention schedule. Wire it through pbrain's prepared-token path, which currently calls `Model.Embed` directly and bypasses the accelerator. On the Radeon RX 9060 XT, build on the verified WebGPU/Vulkan compute path and establish real encoder parity before choosing a shipping backend. For macOS, use the existing cgo-free Metal accelerator as the starting point when Apple hardware is available. [Engine parity](./engine-parity/research_summary.md) · [NVIDIA CUDA](./nvidia-cuda/research_summary.md) · [AMD and Metal](./amd-metal/research_summary.md)

## Key findings

- **Measure the same encoder graph and workload.** The checkpoint uses full and symmetric sliding, non-causal grouped-query attention with 256- and 512-wide heads. The pinned llama.cpp build implements this graph without a persistent KV cache. Match checkpoint revision, text-only weights, precision, role prompts, BOS/EOS and truncation, pooling, and normalization. Check exact prepared IDs and vector parity separately from served timing. vLLM and SGLang offer model-specific scheduling references, and MLX-VLM offers a model-specific Apple reference. [Pinned llama.cpp graph](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/src/models/gemma-embedding2.cpp) · [Engine parity](./engine-parity/research_summary.md)
- **Establish the native CUDA path before specializing.** The RTX PRO 6000 is SM120. In pinned llama.cpp, its FP16 MMA path at head width 512 requires the optimized GQA dispatch conditions. FlashInfer's SM120 designs offer kernel ideas, with cited sliding-window and 512-wide examples using causal workloads. Treat device-side pooling, packed documents with isolated attention, graph replay for repeat shapes, and specialized GEMM or attention as candidates to test after measuring transfers, launch gaps, and kernel time. [Pinned llama.cpp dispatch](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a/ggml/src/ggml-cuda/fattn.cu) · [FlashInfer mask implementation](https://github.com/flashinfer-ai/flashinfer/pull/3518) · [NVIDIA CUDA](./nvidia-cuda/research_summary.md)
- **AMD has a verified route to device compute.** On the RX 9060 XT with RADV Mesa 26.3.1, explicitly selected WebGPU/Vulkan passed all four embedded decoder fixtures and separate compute checks after the installed Vulkan loader was made visible. These results cover generic backend execution. The optional wgpu-native binding uses cgo. Reusing it preserves the pure-Go core, while a cgo-free AMD binary would require a different binding or backend. [AMD tracking issue](https://github.com/pkieltyka/goinfer/issues/11) · [AMD and Metal](./amd-metal/research_summary.md)
- **Metal already provides a reference implementation.** Its FP32 path keeps weights and scratch resident and normally submits one command buffer per forward pass. Short-row GEMM, fused attention, and per-segment pooling are candidates for future tests. llama.cpp's shape-aware dispatch and Apple’s pipeline limits inform the experiments. [Pinned llama.cpp Metal dispatch](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/ggml/src/ggml-metal/ggml-metal-ops.cpp) · [AMD and Metal](./amd-metal/research_summary.md)

## Implications

1. **Baseline.** The initial native CPU run measured encoder-only timings for a short query, a mixed-length document batch, and a long prepared input. All six vectors passed the independent cosine and unit-norm checks. Next, measure broader pbrain retrieval and extend the peer harness for EmbeddingGemma 2. Establish F32 GGUF parity before collecting matched llama.cpp timings. [CPU measurement](../../../docs/measurements/pbrain-embedding-cpu-2026-11-21/README.md) · [Baseline issue #9](https://github.com/pkieltyka/goinfer/issues/9) · [Engine parity](./engine-parity/research_summary.md)
2. **CUDA and CPU.** Bring up one resident CUDA forward pass with CPU parity, then run the same prepared IDs through pbrain with GPU execution verified. Profile short queries and indexing independently. Use those profiles to choose CPU work and CUDA follow-ups, and involve aikit if a shared primitive is the bottleneck. [CUDA and CPU issue #10](https://github.com/pkieltyka/goinfer/issues/10) · [NVIDIA CUDA](./nvidia-cuda/research_summary.md)
3. **AMD.** Inventory missing WebGPU text operations and check both head geometries, the inclusive window edge, future-token influence, layer parity, and the pooled vector on the selected Radeon. Revisit direct Vulkan or HIP if the prototype reveals a concrete compatibility, deployment, or performance gap. [AMD issue #11](https://github.com/pkieltyka/goinfer/issues/11) · [AMD and Metal](./amd-metal/research_summary.md)

## Conflicts and uncertainty

Published engine support applies to particular versions, attention backends, precisions, and device targets. In particular, the causal kernel measurements do not establish symmetric-window behavior, and an AMD CI entry does not establish model support on the tested Radeon. The experiment order above is an inference from source and device evidence. [Engine parity](./engine-parity/research_summary.md) · [NVIDIA CUDA](./nvidia-cuda/research_summary.md) · [AMD and Metal](./amd-metal/research_summary.md)

## Gaps

The research left same-snapshot F32 GGUF parity, matched peer timings, and real-checkpoint device parity and performance on NVIDIA and AMD open. Metal findings are source-based and need an Apple hardware run before choosing an optimization. [Engine parity](./engine-parity/research_summary.md) · [NVIDIA CUDA](./nvidia-cuda/research_summary.md) · [AMD and Metal](./amd-metal/research_summary.md)

## Subtopics

- [Engine support and evaluation equivalence](./engine-parity/research_summary.md) covers the model-specific implementations and the peer protocol.
- [NVIDIA CUDA implementation and performance](./nvidia-cuda/research_summary.md) maps the resident encoder, attention constraints, and profiling experiments.
- [AMD Vulkan/HIP and macOS Metal](./amd-metal/research_summary.md) covers backend selection, packaging boundaries, and device validation.
- [Weight quantization](./weight-quantization.md) records the pinned Unsloth artifacts, published retrieval comparisons, and the pbrain evaluation order.
