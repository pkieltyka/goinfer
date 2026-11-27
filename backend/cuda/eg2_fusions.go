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

func loadEG2Fusions(ops *towerOps, ptx []byte) (*eg2Fusions, error) {
	lib, err := ops.LoadLibrary(eg2ModuleFusion, ptx)
	if err != nil {
		return nil, fmt.Errorf("cuda: load EmbeddingGemma 2 activation fusions: %w", err)
	}
	ffn, err := ops.Dev.NewComputePipeline(lib, "eg2_gelu_mul")
	if err != nil {
		return nil, err
	}
	ple, err := ops.Dev.NewComputePipeline(lib, "eg2_gelu_mul_ple")
	if err != nil {
		return nil, err
	}
	return &eg2Fusions{geluMul: ffn, geluPLE: ple}, nil
}

func (a *eg2Accel) ffnActivation(gate, up Buffer, n int) {
	o := a.ops
	if a.fusions == nil {
		o.GeluTanh(gate, n)
		o.MulVec(gate, up, n)
		return
	}
	o.Cls(clsElem)
	o.Launch(a.fusions.geluMul, gpu.Grid1D(n, 256), Arg(gate), Arg(up), i32(int32(n)))
}

func (a *eg2Accel) pleActivation(gate, table, gathered Buffer, rows, layer int) {
	o := a.ops
	p := a.cfg.PLEDim
	stride := a.cfg.Layers * p
	layerTable := table.At(layer * p * 4)
	if a.fusions == nil {
		o.GeluTanh(gate, rows*p)
		if o.Err == nil {
			o.Err = o.Q.Copy2DAsync(gathered, p*4, layerTable, stride*4, p*4, rows)
		}
		o.MulVec(gate, gathered, rows*p)
		return
	}
	o.Cls(clsElem)
	o.Launch(a.fusions.geluPLE, gpu.Grid1D(rows*p, 256),
		Arg(gate), Arg(layerTable), i32(int32(p)), i64(int64(stride)), i32(int32(rows*p)))
}
