// Package eg2 registers the cgo-free CUDA text accelerator for EmbeddingGemma 2
// without importing goinfer's decoder or generation kernels.
//
// Import this package for its side effects in a text-only client:
//
//	import _ "github.com/pkieltyka/goinfer/backend/cuda/eg2"
//
// Build with `-tags cuda` for the baseline "cuda" selector, `-tags cuda,eg2_combined`
// to add "cuda-eg2-combined", `-tags cuda,eg2_combined,eg2_fp16` to add "cuda-eg2-fp16",
// and `-tags cuda,eg2_fusions` for the standalone activation-fusion kernels.
//
// Package eg2 exports no identifiers. A client that links only this package has no
// CUDA vision or audio tower accelerator, so EnableVision and EnableAudio run those
// towers on the CPU. Full multimodal or generation clients should continue to import
// github.com/pkieltyka/goinfer/backend/cuda, which blank-imports this package.
package eg2
