# NVIDIA CUDA implementation and performance

## Bottom line

Start with a resident FP32 EmbeddingGemma 2 text accelerator in goinfer's optional CUDA module. Reuse its tower GEMM and executor and the Metal encoder's blocked grouped-query attention schedule, with the CPU graph as the numerical reference. The pbrain adapter currently reaches `Model.Embed` directly, so an accelerator-aware prepared-token entry point is needed to reach the intended workload. Establish parity and profile separate query and indexing shapes before choosing between segmented batching, graph replay and specialized kernels. [CUDA tower](https://github.com/pkieltyka/goinfer/blob/e33aa3893/backend/cuda/tower_base.go) · [Metal path](https://github.com/pkieltyka/goinfer/blob/e33aa3893/backend/metal/embeddinggemma2.go) · [pbrain adapter](https://github.com/pkieltyka/pbrain/blob/b4f8aa1/internal/embed/gemma2/gemma2.go)

## Key findings

- **The defining constraint is non-causal attention at both head sizes.** The pinned checkpoint has sliding layers with 256-wide heads and full layers with 512-wide heads. The llama.cpp graph uses a symmetric window, sequence-aware masks and no persistent K/V cache. Its CUDA FP16 MMA selector reaches head size 512 only with the optimized GQA mask, padding and alignment conditions. This gives a kernel design reference and a concrete dispatch check for a peer run. [Encoder graph](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a/src/models/gemma-embedding2.cpp) · [No-cache graph](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a/src/llama-graph.cpp) · [CUDA selector](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a/ggml/src/ggml-cuda/fattn.cu)
- **Use external kernel work as reference evidence.** FlashInfer v0.6.14 illustrates 256/512 execution on SM120, which NVIDIA identifies as the RTX PRO 6000 Blackwell architecture. Its FMHAv2 sliding window is causal, and the cited 512-wide Gemma 4 measurements use a different paged causal workload. FlashAttention-2 documents head dimensions up to 256. [NVIDIA device matrix](https://developer.nvidia.com/cuda/gpus) · [FlashInfer release](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.14) · [FMHAv2 mask change](https://github.com/flashinfer-ai/flashinfer/pull/3518) · [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- **Favor application-level experiments.** Batch document rows across dense layers while keeping per-document positions, attention bounds and pooling. Evaluate device pooling or pooling before the final linear projection as an independent optimization, with a precision check against CPU mean and normalization. Use the existing graph API for repeat shapes only after a trace establishes launch gaps. Compare vendor GEMM in an isolated reference harness if actual projection shapes dominate. [Accelerator contract](https://github.com/pkieltyka/goinfer/blob/e33aa3893/embeddinggemma2/accel.go) · [Graph binding](https://github.com/pkieltyka/goinfer/blob/e33aa3893/backend/cuda/internal/gpudev/cuda.go) · [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)

## Details

- [Implementation notes](implementation-notes.md) give the source mapping, parity constraints, experiment order and evidence that would redirect each experiment.

## Open questions

- The balance between GPU work, launch gaps, transfers, and CPU preparation on this checkpoint and hardware remains unmeasured.
- Exact pbrain packed workload shapes need to be included in the final benchmark. Kernel eligibility, automatic selection and symmetric-window parity on SM120 must be checked on hardware.
- No device measurements or end-to-end speedups were produced by this source study.
