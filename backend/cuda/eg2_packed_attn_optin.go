//go:build cuda && eg2_combined

package cuda

import "github.com/pkieltyka/goinfer/embeddinggemma2"

func init() {
	embeddinggemma2.RegisterAccelerator("cuda-eg2-packed-attn", bareEG2FactoryMode(eg2ModePackedAttn))
	embeddinggemma2.RegisterDeviceAccelerator("cuda-eg2-packed-attn", "cuda", selectedEG2FactoryMode(eg2ModePackedAttn))
}
