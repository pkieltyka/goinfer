//go:build cuda && eg2_combined && eg2_fp16

package cuda

import (
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
	"github.com/pkieltyka/aikit/linalg"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/wpack"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestEG2FP16Geometry(t *testing.T) {
	for _, shape := range [][2]int{{64, 96}, {512, 2048}, {16, 16}} {
		if !eg2FFNGeometry(shape[0], shape[1]) {
			t.Errorf("refused %v", shape)
		}
	}
	for _, shape := range [][2]int{{0, 16}, {16, -16}, {32, 15}, {1 << 20, 1 << 20}} {
		if eg2FFNGeometry(shape[0], shape[1]) {
			t.Errorf("admitted %v", shape)
		}
	}
	if names := embeddinggemma2.DeviceSelectors("cuda"); !slices.Contains(names, eg2FP16Selector) {
		t.Fatalf("missing %s selector in %v", eg2FP16Selector, names)
	}
	acc, err := selectedEG2FactoryOn(eg2CUDAAttach(true, eg2FFNFP16PTX))(nil, "invalid")
	if acc != nil || !errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) {
		t.Fatalf("invalid explicit device = %v, %v", acc, err)
	}
}

func TestEG2FP16Weights(t *testing.T) {
	values := []float32{65504, 65519, -65504, -65519, 0, math.Float32frombits(1 << 31), 0x1p-24, 0x1p-25, 1 + 0x1p-11}
	want := []uint16{0x7bff, 0x7bff, 0xfbff, 0xfbff, 0, 0x8000, 1, 0, 0x3c00}
	if got, err := eg2FP16Weights(values); err != nil || !slices.Equal(got, want) {
		t.Fatalf("weights = %#x, %v, want %#x", got, err, want)
	}
	for _, bad := range []float32{65520, -65520, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if packed, err := eg2FP16Weights([]float32{1, bad}); packed != nil || err == nil {
			t.Errorf("invalid weight %g returned %v, %v", bad, packed, err)
		}
	}
}

func eg2FP16TestAdapter(t *testing.T) gpu.Adapter {
	t.Helper()
	id := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if id == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID to run FP16 FFN device tests")
	}
	devices, err := gpu.Adapters()
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := resolveCUDADevice(id, devices)
	if err != nil {
		t.Fatal(err)
	}
	if adapter.Major < 8 {
		t.Fatalf("FP16 FFN device test requires SM80 or newer, got %d.%d", adapter.Major, adapter.Minor)
	}
	return adapter
}

func TestEG2FP16Projection(t *testing.T) {
	adapter := eg2FP16TestAdapter(t)
	ops, err := newTowerOpsOn(func() (*Device, error) { return gpu.CreateSelectedDevice(adapter) }, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ops.closeChecked(); err != nil {
			t.Error(err)
		}
	})
	var p Pipeline
	if err := ops.do(func() error {
		var err error
		p, err = eg2FP16Pipeline(ops.dev, eg2FFNFP16PTX)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	a := &eg2Accel{ops: ops, ffnFP16: p}
	rng := rand.New(rand.NewPCG(63, 16))
	for _, shape := range [][2]int{{67, 16}, {63, 80}, {96, 512}, {512, 2048}} {
		n, k := shape[0], shape[1]
		w := make([]uint16, n*k)
		for i := range w {
			w[i] = wpack.F32ToF16RNE(rng.Float32()*2 - 1)
		}
		x := make([]float32, 65*k)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		var wb Buffer
		if err := ops.do(func() error {
			wb = gpu.NewBufferOf(ops.dev, w)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		run := func(x []float32, wantInvalid bool) []float32 {
			t.Helper()
			m := len(x) / k
			out := make([]float32, m*n)
			var status [1]uint32
			err := ops.do(func() error {
				xb := ops.af(len(x))
				yb := ops.af(len(out))
				invalid := ops.ai(1)
				if err := gpu.Upload(xb, x); err != nil {
					return err
				}
				if err := ops.q.ZeroAsync(invalid, 4); err != nil {
					return err
				}
				a.ffnGEMM(xb, wb, yb, invalid, m, n, k)
				if err := ops.finish(); err != nil {
					return err
				}
				if err := gpu.Download(invalid, status[:]); err != nil {
					return err
				}
				return gpu.Download(yb, out)
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := ops.do(ops.releaseScratchChecked); err != nil {
				t.Fatal(err)
			}
			if (status[0] != 0) != wantInvalid {
				t.Fatalf("range status %#x, want invalid=%t", status[0], wantInvalid)
			}
			return out
		}
		got := run(x, false)
		for r := 0; r < 65; r++ {
			for c := 0; c < n; c++ {
				var sum float64
				for d := 0; d < k; d++ {
					rx := linalg.F16ToF32(wpack.F32ToF16RNE(x[r*k+d]))
					sum += float64(rx) * float64(linalg.F16ToF32(w[c*k+d]))
				}
				if delta := math.Abs(float64(got[r*n+c]) - sum); !(delta <= 3e-4) {
					t.Fatalf("N=%d K=%d row %d col %d got %g want %g (abs error %g)", n, k, r, c, got[r*n+c], sum, delta)
				}
			}
		}
		for _, r := range []int{0, 1, 14, 15, 16, 30, 31, 32, 63, 64} {
			single := run(x[r*k:(r+1)*k], false)
			if d := eg2FirstDiff(single, got[r*n:(r+1)*n], n); d != "" {
				t.Fatalf("N=%d K=%d row %d changed in singleton: %s", n, k, r, d)
			}
		}
		if n == 67 {
			boundary := []struct {
				value   float32
				invalid bool
			}{{65504, false}, {-65519, false}, {0x1p-24, false}, {65520, true},
				{-65520, true}, {float32(math.NaN()), true}, {float32(math.Inf(1)), true}}
			late := slices.Clone(x[:33*k])
			for i, tc := range boundary {
				// Alternate the low and high half of a pair in the final row.
				late[(len(late)-2)+(i%2)] = tc.value
				run(late, tc.invalid)
				late[(len(late)-2)+(i%2)] = 0
			}
		}
		if err := ops.do(func() error { return ops.dev.ReleaseBufChecked(wb) }); err != nil {
			t.Fatal(err)
		}
	}
}

func TestEG2FP16Encoder(t *testing.T) {
	device := eg2FP16TestAdapter(t)
	m := eg2TinyModel(t)
	rec := &eg2RecordingLeaser{}
	origLeaser := gpu.NewLeaser
	gpu.NewLeaser = func() gpu.Leaser { return rec }
	acc, err := m.NewAccelerator(eg2FP16Selector + ":" + device.UUID)
	gpu.NewLeaser = origLeaser
	if err != nil {
		t.Fatal(err)
	}
	a := acc.(*eg2Accel)
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Error(err)
		}
	})
	if a.ffnFP16 == (Pipeline{}) || a.AttachedDevice().ID != "cuda:"+device.UUID {
		t.Fatalf("FP16 pipeline/device not selected: %s", a.AttachedDevice().ID)
	}
	maxHD, maxN := 0, max(a.cfg.Layers*a.cfg.PLEDim, a.cfg.Intermediate, a.cfg.Hidden, a.cfg.EmbeddingDim)
	for _, ly := range a.layers {
		maxHD = max(maxHD, ly.w.HeadDim)
		maxN = max(maxN, a.cfg.Heads*ly.w.HeadDim)
	}
	est, err := a.BatchMemory([]int{17, 31})
	if err != nil {
		t.Fatal(err)
	}
	if want := rec.devLive - gpu.ChargedBytes(int64(maxHD)*4) - gpu.ChargedBytes(int64(maxN)*4); est.ResidentWeightBytes != want {
		t.Fatalf("resident weight estimate %d, want %d charged bytes", est.ResidentWeightBytes, want)
	}
	cfg := a.cfg
	cfg.EmbeddingDim = 64
	plain := &eg2Accel{cfg: cfg, layers: a.layers}
	withStatus := &eg2Accel{cfg: cfg, layers: a.layers, ffnFP16: a.ffnFP16}
	refPlan, err := plain.scratchPlan(8192)
	if err != nil {
		t.Fatal(err)
	}
	fp16Plan, err := withStatus.scratchPlan(8192)
	if err != nil {
		t.Fatal(err)
	}
	if fp16Plan.res != refPlan.res+1 || fp16Plan.chargedBytes-refPlan.chargedBytes != gpu.AllocQuantumBytes {
		t.Fatalf("status crosses a charge boundary: elements %d to %d, charge %d to %d",
			refPlan.res, fp16Plan.res, refPlan.chargedBytes, fp16Plan.chargedBytes)
	}
	old := gpu.TraceForTest
	t.Cleanup(func() { gpu.TraceForTest = old })
	launches := 0
	gpu.TraceForTest = func(event gpu.TraceEvent) {
		if event.Op == gpu.TraceLaunch && event.Pipe == a.ffnFP16.Ptr() {
			launches++
		}
	}
	inputs := make([][]int, 0, 8)
	for _, n := range []int{1, 15, 16, 17, 31, 32, 33, 65} {
		inputs = append(inputs, eg2SeqIDs(n, n+1, m.Config().VocabSize))
	}
	singles := make([][]float32, len(inputs))
	minCosine := 1.0
	for i, ids := range inputs {
		h, _, err := a.Forward(ids, false)
		if err != nil {
			t.Fatal(err)
		}
		cp, err := m.Embed(ids)
		if err != nil {
			t.Fatal(err)
		}
		v := eg2PoolNorm(h, len(ids), m.Dim(), m.Dim())
		if c := eg2Cos(v, cp); !eg2Finite(v) || !(c >= .999) {
			t.Fatalf("input %d length %d CPU cosine %g", i, len(ids), c)
		} else {
			minCosine = min(minCosine, c)
		}
		singles[i] = h
	}
	for _, order := range [][]int{{0, 1, 2, 3, 4, 5, 6, 7}, {7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3}, {4, 5, 6, 7}, {7, 2}, {0, 2}} {
		group := make([][]int, len(order))
		for i, pos := range order {
			group[i] = inputs[pos]
		}
		before := launches
		out, err := a.ForwardBatch(group)
		if err != nil {
			t.Fatal(err)
		}
		if launches-before != 3*m.Config().Layers {
			t.Fatalf("order %v: FP16 launches %d", order, launches-before)
		}
		for i, pos := range order {
			if d := eg2FirstDiff(out[i], singles[pos], m.Dim()); d != "" {
				t.Fatalf("order %v input %d: %s", order, pos, d)
			}
		}
	}
	t.Logf("raw rows match singleton bits; minimum pooled CPU cosine %.12f", minCosine)

	// A nonfinite value in the later prepared member invalidates the entire packed forward.
	x, err := m.EmbedTokens(append(slices.Clone(inputs[0]), inputs[1]...))
	if err != nil {
		t.Fatal(err)
	}
	x[len(x)-1] = float32(math.Inf(1))
	h, layers, err := a.forwardRows(x, []int{len(inputs[0]), len(inputs[1])}, true)
	if h != nil || layers != nil || !errors.Is(err, errEG2FP16InputRange) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("invalid later input returned %d rows, %d layers, %v", len(h), len(layers), err)
	}
	base, allocs := rec.resetWindow()
	est, err = a.BatchMemory([]int{len(inputs[0]), len(inputs[1])})
	if err != nil {
		t.Fatal(err)
	}
	out, err := a.ForwardBatch(inputs[:2])
	if err != nil {
		t.Fatal(err)
	}
	peak, end, _ := rec.windowDelta(base, allocs)
	if peak > est.AdditionalPeakBytes || end != 0 {
		t.Fatalf("recovery scratch peak=%d, estimate=%d, end=%d", peak, est.AdditionalPeakBytes, end)
	}
	for i := range out {
		if d := eg2FirstDiff(out[i], singles[i], m.Dim()); d != "" {
			t.Fatalf("after range failure input %d: %s", i, d)
		}
	}

	oom := gc.ErrOutOfMemory
	a.ops.dev.SetAllocHookForTest(func(string, int) error { return oom })
	out, err = a.ForwardBatch(inputs[:2])
	a.ops.dev.SetAllocHookForTest(nil)
	if out != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || !errors.Is(err, oom) {
		t.Fatalf("forward allocation failure returned %d outputs, %v", len(out), err)
	}
	recovered, err := a.ForwardBatch(inputs[:2])
	if err != nil {
		t.Fatal(err)
	}
	for i := range recovered {
		if d := eg2FirstDiff(recovered[i], singles[i], m.Dim()); d != "" {
			t.Fatalf("after allocation retry input %d: %s", i, d)
		}
	}

	fatal := &cudaresult.Error{Code: cudasys.CUDA_ERROR_ASSERT, Op: "injected FFN execution failure"}
	a.defect.failForward, a.defect.failForwardErr = true, fatal
	beforeFatal := launches
	out, err = a.ForwardBatch(inputs[:2])
	if out != nil || !errors.Is(err, fatal) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || launches <= beforeFatal {
		t.Fatalf("post-FFN execution failure returned %d outputs, %v, launched %d FP16 kernels", len(out), err, launches-beforeFatal)
	}
	beforeFatal = launches
	out, err = a.ForwardBatch(inputs[:2])
	if out != nil || !errors.Is(err, fatal) || launches != beforeFatal {
		t.Fatalf("poisoned forward returned %d outputs, %v, launched %d FP16 kernels", len(out), err, launches-beforeFatal)
	}
}

func TestEG2FP16FailedAttach(t *testing.T) {
	adapter := eg2FP16TestAdapter(t)
	m := eg2TinyModel(t)
	bytes := m.Config().Hidden * m.Config().Intermediate * 2
	var dev *Device
	var count int
	acc, err := newEG2AccelWithFFN(m, true, eg2FFNFP16PTX, func() (*Device, error) {
		d, err := gpu.CreateSelectedDevice(adapter)
		if err != nil {
			return nil, err
		}
		dev = d
		d.SetAllocHookForTest(func(_ string, n int) error {
			if n == bytes {
				count++
				if count == 2 {
					return gc.ErrOutOfMemory
				}
			}
			return nil
		})
		return d, nil
	})
	if acc != nil || !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) ||
		!errors.Is(err, gc.ErrOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("failed attach = %v, %v", acc, err)
	}
	if dev == nil || count != 2 {
		t.Fatalf("device %v allocated %d FP16-size matrices before failure", dev != nil, count)
	}
	if allocs, objects := dev.LedgerLen(); allocs != 0 || objects != 0 {
		t.Fatalf("failed attach retains %d allocations and %d objects", allocs, objects)
	}
	m.Weights().Layers[0].Up[0] = 65520
	acc, err = newEG2AccelWithFFN(m, true, eg2FFNFP16PTX, func() (*Device, error) {
		d, err := gpu.CreateSelectedDevice(adapter)
		dev = d
		return d, err
	})
	if acc != nil || err == nil || !strings.Contains(err.Error(), "outside the finite FP16 range") ||
		errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("attach with invalid weight = %v, %v", acc, err)
	}
	if dev == nil {
		t.Fatal("device did not open for the invalid-weight test")
	}
	if allocs, objects := dev.LedgerLen(); allocs != 0 || objects != 0 {
		t.Fatalf("invalid-weight attach retains %d allocations and %d objects", allocs, objects)
	}
}
