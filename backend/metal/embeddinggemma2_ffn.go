//go:build darwin

package metal

import (
	"fmt"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

const (
	eg2StagedThreads = 128
	eg2StagedTGBytes = 13824 // (32 input rows + 64 weight rows) × padded K stride 36 × 4 bytes
	eg2StagedMinRows = 31
	eg2StagedMaxRows = 2042
)

// The measured tile replaces the dense gate/up projections on the base M4
// combined graph. The architecture check keeps unknown Metal implementations
// and other GPU families on their existing kernels.
func eg2StagedFFNEligible(device, arch, profile string, c embeddinggemma2.Config) bool {
	return device == "Apple M4" && arch == "applegpu_g16g" && profile == eg2MatrixProfile &&
		c.Hidden == 512 && c.Intermediate == 2048 && c.Layers == 24
}

func eg2StagedFFNRows(rows int) bool { return rows >= eg2StagedMinRows && rows <= eg2StagedMaxRows }

func newEG2StagedFFN(d *Device) (Pipeline, error) {
	if d.MaxThreadgroupMemoryLength() < eg2StagedTGBytes {
		return Pipeline{}, nil
	}
	lib, err := d.CompileLibraryPrecise(eg2FFNStagedMSL, MSL3_1)
	if err != nil {
		return Pipeline{}, fmt.Errorf("metal: compile staged FFN gate/up: %w", err)
	}
	defer d.ReleaseObject(lib)
	p, err := d.NewComputePipeline(lib, "eg2_ffn_gate_up")
	if err != nil {
		return Pipeline{}, err
	}
	if p.ThreadExecutionWidth() != 32 || p.MaxTotalThreadsPerThreadgroup() < eg2StagedThreads ||
		p.StaticThreadgroupMemoryLength() != eg2StagedTGBytes {
		d.ReleasePipeline(p)
		return Pipeline{}, nil
	}
	return p, nil
}

// gemmStagedFFN consumes the same contiguous FP32 A[M,K], W[N,K] and C[M,N]
// as w32. The caller fixes N=2048 and K=512 before choosing this dispatch.
func (a *eg2Accel) gemmStagedFFN(e *Encoder, A, W, C Buffer, rows, width, inner int) {
	e.Dispatch2D(a.stagedFFN, width/64, (rows+31)/32, eg2StagedThreads, 1, A, W, C,
		a.u32(uint32(rows)), a.u32(uint32(width)), a.u32(uint32(inner)))
}
