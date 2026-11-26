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

// newEG2StagedFFN builds the released staged gate/up pipeline and, when the device
// supports it, the optional fused up-store pipeline. The fused pipeline is an
// optimisation for the explicit batch attachment; if it cannot be built the
// released staged pipeline is still returned and the batch attachment is unchanged.
func newEG2StagedFFN(d *Device, wantFused bool) (Pipeline, Pipeline, error) {
	var none Pipeline
	if d.MaxThreadgroupMemoryLength() < eg2StagedTGBytes {
		return none, none, nil
	}
	lib, err := d.CompileLibraryPrecise(eg2FFNStagedMSL, MSL3_1)
	if err != nil {
		return none, none, fmt.Errorf("metal: compile staged FFN gate/up: %w", err)
	}
	defer d.ReleaseObject(lib)
	staged, err := d.NewComputePipeline(lib, "eg2_ffn_gate_up")
	if err != nil {
		return none, none, err
	}
	if staged.ThreadExecutionWidth() != 32 || staged.MaxTotalThreadsPerThreadgroup() < eg2StagedThreads ||
		staged.StaticThreadgroupMemoryLength() != eg2StagedTGBytes {
		d.ReleasePipeline(staged)
		return none, none, nil
	}
	if !wantFused {
		return staged, none, nil
	}
	// The fused up-store belongs to the explicit batch attachment. It is optional:
	// a compile or resource failure leaves the released staged pipeline and the
	// batch attachment usable.
	fusedLib, err := d.CompileLibraryPrecise(eg2FFNStagedGeGLUMSL, MSL3_1)
	if err != nil {
		return staged, none, nil
	}
	defer d.ReleaseObject(fusedLib)
	fused, err := d.NewComputePipeline(fusedLib, "eg2_ffn_up_geglu")
	if err != nil {
		return staged, none, nil
	}
	if fused.ThreadExecutionWidth() != 32 || fused.MaxTotalThreadsPerThreadgroup() < eg2StagedThreads ||
		fused.StaticThreadgroupMemoryLength() != eg2StagedTGBytes {
		d.ReleasePipeline(fused)
		return staged, none, nil
	}
	return staged, fused, nil
}

// gemmStagedFFN consumes the same contiguous FP32 A[M,K], W[N,K] and C[M,N]
// as w32. The caller fixes N=2048 and K=512 before choosing this dispatch.
func (a *eg2Accel) gemmStagedFFN(e *Encoder, A, W, C Buffer, rows, width, inner int) {
	e.Dispatch2D(a.stagedFFN, width/64, (rows+31)/32, eg2StagedThreads, 1, A, W, C,
		a.u32(uint32(rows)), a.u32(uint32(width)), a.u32(uint32(inner)))
}

// gemmStagedFFNUpGeGLU is the fused up projection: the same staged tile and K order
// as gemmStagedFFN, with C pointing at the completed gate buffer. The multiply writes
// eg2_gelu(gate) * up back over the gate rows and leaves the separate up buffer
// untouched.
func (a *eg2Accel) gemmStagedFFNUpGeGLU(e *Encoder, A, W, C Buffer, rows, width, inner int) {
	e.Dispatch2D(a.stagedFFNGeGLU, width/64, (rows+31)/32, eg2StagedThreads, 1, A, W, C,
		a.u32(uint32(rows)), a.u32(uint32(width)), a.u32(uint32(inner)))
}
