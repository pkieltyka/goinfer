//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"math"

	"github.com/pkieltyka/goinfer/backend/cuda/internal/wpack"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

var errEG2FP16InputRange = errors.New("cuda: EmbeddingGemma 2 FFN input is outside the finite FP16 range")

const eg2FP16MinMajor = 8

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

func eg2FP16Pipeline(dev *Device, ptx []byte) (Pipeline, error) {
	major, minor, err := dev.Context().Device().ComputeCapability()
	if err != nil {
		return Pipeline{}, fmt.Errorf("cuda: FP16 FFN device capability: %w", err)
	}
	if major < eg2FP16MinMajor {
		return Pipeline{}, fmt.Errorf("%w: CUDA FP16 FFN needs compute capability 8.0, device has %d.%d",
			embeddinggemma2.ErrNoDevice, major, minor)
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
