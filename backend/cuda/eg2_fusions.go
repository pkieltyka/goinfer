//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
)

type eg2Fusions struct {
	geluMul Pipeline
	geluPLE Pipeline
}

var eg2FusionsPTX []byte

func loadEG2Fusions(dev *Device, ptx []byte) (*eg2Fusions, error) {
	lib, err := dev.CompileLibrary(ptx)
	if err != nil {
		return nil, fmt.Errorf("cuda: load EmbeddingGemma 2 activation fusions: %w", err)
	}
	ffn, err := dev.NewComputePipeline(lib, "eg2_gelu_mul")
	if err != nil {
		return nil, err
	}
	ple, err := dev.NewComputePipeline(lib, "eg2_gelu_mul_ple")
	if err != nil {
		return nil, err
	}
	return &eg2Fusions{geluMul: ffn, geluPLE: ple}, nil
}

func (a *eg2Accel) ffnActivation(gate, up Buffer, n int) {
	o := a.ops
	if a.fusions == nil {
		o.geluTanh(gate, n)
		o.mulVec(gate, up, n)
		return
	}
	o.cls(clsElem)
	o.launch(a.fusions.geluMul, gpu.Grid1D(n, 256), Arg(gate), Arg(up), i32(int32(n)))
}

func (a *eg2Accel) pleActivation(gate, table, gathered Buffer, rows, layer int) {
	o := a.ops
	p := a.cfg.PLEDim
	stride := a.cfg.Layers * p
	layerTable := table.At(layer * p * 4)
	if a.fusions == nil {
		o.geluTanh(gate, rows*p)
		if o.err == nil {
			o.err = o.q.Copy2DAsync(gathered, p*4, layerTable, stride*4, p*4, rows)
		}
		o.mulVec(gate, gathered, rows*p)
		return
	}
	o.cls(clsElem)
	o.launch(a.fusions.geluPLE, gpu.Grid1D(rows*p, 256),
		Arg(gate), Arg(layerTable), i32(int32(p)), i64(int64(stride)), i32(int32(rows*p)))
}
