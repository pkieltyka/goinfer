//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"math"

	"github.com/pkieltyka/goinfer/backend/cuda/internal/wpack"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

var errEG2FP16InputRange = errors.New("cuda: EmbeddingGemma 2 FP16 input is outside the finite FP16 range")

const (
	eg2FP16MinMajor = 8
	eg2FP16Selector = "cuda-eg2-fp16"
)

func eg2FP16Weights(v []float32) ([]uint16, error) {
	packed := make([]uint16, len(v))
	for i, value := range v {
		half := wpack.F32ToF16RNE(value)
		if half&0x7c00 == 0x7c00 {
			return nil, fmt.Errorf("cuda: EmbeddingGemma 2 FFN weight %d is outside the finite FP16 range", i)
		}
		packed[i] = half
	}
	return packed, nil
}

func eg2FFNGeometry(hidden, intermediate int) bool {
	return hidden > 0 && intermediate > 0 && hidden%16 == 0 && intermediate%16 == 0 &&
		hidden <= math.MaxInt32/intermediate
}

func cudaSupportsEG2FP16(d embeddinggemma2.DeviceInfo) bool {
	return cudaSupportsEG2(d) && eg2FP16Supported(d.ComputeCapability.Major)
}

func eg2FP16Supported(major int) bool {
	return major >= eg2FP16MinMajor
}

func requireEG2FP16Capability(capability embeddinggemma2.CUDAComputeCapability, err error) error {
	if err != nil {
		return fmt.Errorf("cuda: FP16 FFN device capability: %w", err)
	}
	major, minor := capability.Major, capability.Minor
	if !cudaSupportsPTX(major, minor) {
		return fmt.Errorf("%w: CUDA compute capability %d.%d is below the default kernel target %d.%d",
			embeddinggemma2.ErrNoDevice, major, minor, ptxTargetMajor, ptxTargetMinor)
	}
	if !eg2FP16Supported(major) {
		return fmt.Errorf("%w: %q needs CUDA compute capability %d.0, device has %d.%d",
			embeddinggemma2.ErrUnsupportedSelector, eg2FP16Selector, eg2FP16MinMajor, major, minor)
	}
	return nil
}

func eg2FP16Pipeline(dev *Device, ptx []byte, capability embeddinggemma2.CUDAComputeCapability, capabilityErr error) (Pipeline, error) {
	if err := requireEG2FP16Capability(capability, capabilityErr); err != nil {
		return Pipeline{}, err
	}
	lib, err := dev.CompileLibrary(ptx)
	if err != nil {
		return Pipeline{}, fmt.Errorf("cuda: load FP16 FFN: %w", err)
	}
	return dev.NewComputePipeline(lib, "eg2_ffn_fp16")
}

func (a *eg2Accel) ffnGEMM(x, w, out, invalidInput Buffer, m, n, k int) {
	o := a.ops
	if a.ffnFP16 == (Pipeline{}) {
		o.gemm(x, w, out, m, n, k)
		return
	}
	o.cls(clsGEMM)
	o.profGEMM(m, n, k)
	o.launch(a.ffnFP16, LaunchConfig{
		GridX: uint32((n + 63) / 64), GridY: uint32((m + 31) / 32), GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1,
	}, Arg(x), Arg(w), Arg(out), Arg(invalidInput), i32(int32(m)), i32(int32(n)), i32(int32(k)))
}
