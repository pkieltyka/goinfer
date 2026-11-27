//go:build cuda && eg2_combined && eg2_fp16

package eg2

import (
	_ "embed"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

//go:embed testdata/eg2_ffn_fp16.ptx
var embeddedEG2FFNFP16PTX []byte

//go:embed testdata/eg2_packed_attn.ptx
var embeddedEG2PackedAttnPTX []byte

//go:embed testdata/eg2-sm120/ffn.sm120.cubin
var embeddedEG2FFNSM120 []byte

//go:embed testdata/eg2-sm120/packed.sm120.cubin
var embeddedEG2PackedSM120 []byte

func init() {
	eg2FFNFP16PTX, eg2PackedAttnPTX = embeddedEG2FFNFP16PTX, embeddedEG2PackedAttnPTX
	eg2FFNSM120, eg2PackedSM120 = embeddedEG2FFNSM120, embeddedEG2PackedSM120
	attach := func(m *embeddinggemma2.Model, open func() (*gpu.Device, error)) (*eg2Accel, error) {
		return newEG2AccelConfigured(m, true, eg2FFNFP16PTX, eg2PackedAttnPTX, eg2FusionsPTX, open)
	}
	embeddinggemma2.RegisterAccelerator(eg2FP16Selector, bareEG2FactoryOn(attach))
	embeddinggemma2.RegisterDeviceAcceleratorWithSupport(eg2FP16Selector, "cuda", selectedEG2FactoryOn(attach), cudaSupportsEG2FP16)
}

func cudaSupportsEG2FP16(d embeddinggemma2.DeviceInfo) bool {
	return cudaSupportsEG2(d) && eg2FP16Supported(d.ComputeCapability.Major)
}
