//go:build cuda && eg2_combined && eg2_fp16

package cuda

import (
	_ "embed"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

//go:embed testdata/eg2_ffn_fp16.ptx
var eg2FFNFP16PTX []byte

//go:embed testdata/eg2_packed_attn.ptx
var eg2PackedAttnPTX []byte

const eg2FP16Selector = "cuda-eg2-fp16"

func init() {
	attach := func(m *embeddinggemma2.Model, open func() (*Device, error)) (*eg2Accel, error) {
		return newEG2AccelWithFP16(m, true, eg2FFNFP16PTX, eg2PackedAttnPTX, open)
	}
	embeddinggemma2.RegisterAccelerator(eg2FP16Selector, bareEG2FactoryOn(attach))
	embeddinggemma2.RegisterDeviceAccelerator(eg2FP16Selector, "cuda", selectedEG2FactoryOn(attach))
}
