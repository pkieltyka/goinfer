//go:build cuda && eg2_combined

package eg2

import "github.com/pkieltyka/goinfer/embeddinggemma2"

func init() {
	embeddinggemma2.RegisterAccelerator("cuda-eg2-combined", bareEG2Factory(true))
	embeddinggemma2.RegisterDeviceAcceleratorWithSupport("cuda-eg2-combined", "cuda", selectedEG2Factory(true), cudaSupportsEG2)
}
