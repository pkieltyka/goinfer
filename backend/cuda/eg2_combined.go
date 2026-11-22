//go:build cuda && eg2_combined

package cuda

import "github.com/pkieltyka/goinfer/embeddinggemma2"

func init() {
	embeddinggemma2.RegisterAccelerator("cuda-eg2-combined", func(model *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return newEG2Accel(model, true)
	})
	embeddinggemma2.RegisterSelectedAccelerator("cuda-eg2-combined", selectedEG2Factory(true))
}
