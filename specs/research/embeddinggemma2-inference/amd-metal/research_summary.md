# AMD Vulkan/HIP and macOS Metal

## Bottom line

Start AMD bring-up on the existing WebGPU/Vulkan path with explicit Radeon adapter selection and an execution test, then implement the float32 encoder through the existing accelerator interface. WebGPU can preserve the pure-Go core as an optional backend. Its current native bridge uses cgo, so a cgo-free AMD binary is a distinct requirement. Keep direct Vulkan and HIP as alternatives for a demonstrated capability, deployment or performance gap. For macOS, goinfer already has a cgo-free Metal text accelerator. Use its float32 path as the reference and study llama.cpp's shape-aware matrix and attention dispatch before selecting changes for future Apple hardware tests. [goinfer text accelerator contract](https://github.com/pkieltyka/goinfer/blob/e33aa3893e51c15af4a5c5c256d40d728bb1b2f0/embeddinggemma2/accel.go) · [WebGPU native binding](https://github.com/oliverbestmann/webgpu/blob/v1.38.0/wgpu/instance.go) · [goinfer Metal reference](https://github.com/pkieltyka/goinfer/blob/e33aa3893e51c15af4a5c5c256d40d728bb1b2f0/backend/metal/embeddinggemma2.go) · [llama.cpp Metal dispatch](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/ggml/src/ggml-metal/ggml-metal-ops.cpp)

## Key findings

- **Adapter selection.** A high-performance preference is insufficient when AMD, NVIDIA, integrated and software adapters can all be present. The native Go binding already supports adapter enumeration and structured IDs. Use process-local Vulkan loader and Mesa filters for the first proof. Check the actual adapter reported by the test process, then require real dispatch and readback before calling the generic path verified. [goinfer WebGPU context](https://github.com/pkieltyka/goinfer/blob/e33aa3893e51c15af4a5c5c256d40d728bb1b2f0/backend/webgpu/internal/kernels/gpu.go) · [Native adapter enumeration](https://github.com/oliverbestmann/webgpu/blob/v1.38.0/wgpu/instance.go) · [Mesa device selection](https://docs.mesa3d.org/envvars.html#envvar-DRI_PRIME)
- **Encoder-specific gates.** EmbeddingGemma 2 needs non-causal full and symmetric sliding grouped-query attention with 256- and 512-wide heads. A generic compute smoke is one prerequisite. Next verify masking at the inclusive window edge, future-token influence, layer parity and the pooled vector. Validate packed sequence isolation before claiming batching. [Pinned llama.cpp encoder graph](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/src/models/gemma-embedding2.cpp) · [goinfer CPU model](https://github.com/pkieltyka/goinfer/blob/e33aa3893e51c15af4a5c5c256d40d728bb1b2f0/embeddinggemma2/model.go)
- **Vulkan and HIP.** llama.cpp's Vulkan implementation chooses kernels based on shape, types, subgroup support and shared memory pressure. Cooperative matrices are a separate capability. Its HIP build recompiles shared ggml sources with HIP and depends on HIP, hipBLAS and rocBLAS. Treat it as a potential reference arm before investing in another backend ABI. AMD documents the Radeon RX 9060 XT as `gfx1200` under compatible release and OS combinations. [Vulkan implementation](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/ggml/src/ggml-vulkan/ggml-vulkan.cpp) · [HIP build](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/ggml/src/ggml-hip/CMakeLists.txt) · [AMD ROCm requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.2/reference/system-requirements.html)
- **Metal.** The existing accelerator already keeps weights and scratch resident and normally encodes a forward into one command buffer. Plausible next experiments are a short-row GEMM specialization, a fused attention path that avoids staging intermediates, and on-device per-segment pooling. Gate selection by per-pipeline limits and actual device family. Preserve normalization and measure short queries separately from document batches. [goinfer Metal implementation](https://github.com/pkieltyka/goinfer/blob/e33aa3893e51c15af4a5c5c256d40d728bb1b2f0/backend/metal/embeddinggemma2.go) · [Apple threadgroup sizing](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes) · [Apple Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

## Details

- [AMD Linux feasibility](amd-linux.md) covers process-level adapter selection, WebGPU versus Vulkan/HIP, and a staged encoder proof.
- [Metal reference designs](macos-metal.md) maps llama.cpp and Apple documentation to specific experiments on the existing accelerator.

## Open questions

- The viability of cgo-free AMD packaging depends on the binding or backend chosen.
- The AMD shader features, operation residency and real encoder parity need device evidence. The separate generic hardware smoke does not settle those questions.
- Metal specialization winners and practical tile choices depend on the Apple hardware and workload. No Metal runs or platform speed comparisons were performed in this research.

## Sources

- [llama.cpp source at e6d624b9a4b9](https://github.com/ggml-org/llama.cpp/tree/e6d624b9a4b95c6b52f5ca384d70b75581f9821a) pins the model and backend references.
- [goinfer source at e33aa3893](https://github.com/pkieltyka/goinfer/tree/e33aa3893e51c15af4a5c5c256d40d728bb1b2f0) pins the baseline implementation.
- [WebGPU candidate recommendation draft, 16 November 2026](https://www.w3.org/TR/2026/CRD-webgpu-20261116/) defines the optional features, adapter and device contracts.
- [AMD ROCm Linux system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.2/reference/system-requirements.html) give the device and OS compatibility framework.
