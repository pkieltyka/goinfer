//go:build cuda

package cuda

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"sync"
	"testing"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

const eg2TinyDir = "../../testdata/embeddinggemma2-tiny"

// eg2TinyModel is the tiny fixture (sliding radius 3; head dims 16 with 2 KV heads and 32 with 1) with every norm weight
// redrawn in [0.5, 1.5) through Weights' aliasing slices, so a norm given the wrong weight cannot hide behind the
// fixture's own.
func eg2TinyModel(t *testing.T) *embeddinggemma2.Model {
	t.Helper()
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatalf("load the tiny fixture: %v", err)
	}
	w := m.Weights()
	r := rand.New(rand.NewPCG(11, 12))
	redraw := func(vs ...[]float32) {
		for _, v := range vs {
			for i := range v {
				v[i] = 0.5 + r.Float32()
			}
		}
	}
	redraw(w.PLENorm, w.FinalNorm)
	for _, l := range w.Layers {
		redraw(l.InNorm, l.QNorm, l.KNorm, l.PostAttnNorm, l.PreFFNorm, l.PostFFNorm, l.PLEPostNorm)
	}
	return m
}

func newEG2ForTest(t *testing.T, m *embeddinggemma2.Model) *eg2Accel {
	t.Helper()
	acc, err := m.NewAccelerator("cuda")
	if err != nil {
		t.Skipf("no CUDA device for EmbeddingGemma 2: %v", err)
	}
	t.Cleanup(func() { _ = acc.Close() })
	return acc.(*eg2Accel)
}

// eg2TinyInputs are the golden's four cases (1, 2, 9 and 17 tokens) and random inputs whose lengths put rows at both
// window edges and in the middle: 7 = 2W+1 (one row reaches both edges), 8 and 64, and 300.
func eg2TinyInputs(t *testing.T, vocab int) [][]int {
	t.Helper()
	raw, err := os.ReadFile(eg2TinyDir + "/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Cases []struct {
			IDs []int `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	var in [][]int
	for _, c := range g.Cases {
		in = append(in, c.IDs)
	}
	for _, n := range []int{7, 8, 64, 300} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7919 + 13 + n) % vocab
		}
		in = append(in, ids)
	}
	return in
}

func eg2Cos(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func eg2Finite(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

// eg2PoolNorm is sentence-transformers' mean pool then L2 normalisation over the first d of each row's E values.
func eg2PoolNorm(h []float32, T, E, d int) []float32 {
	mean := make([]float64, d)
	for t := range T {
		for j := range d {
			mean[j] += float64(h[t*E+j])
		}
	}
	var ss float64
	for j := range mean {
		mean[j] /= float64(T)
		ss += mean[j] * mean[j]
	}
	out := make([]float32, d)
	for j := range mean {
		out[j] = float32(mean[j] / math.Max(math.Sqrt(ss), 1e-12))
	}
	return out
}

// eg2Worst is the lowest cosine between the accelerator's and the CPU's states over every input: each layer's input and
// the last layer's output, the projected last hidden state and the pooled embedding. It fails the test on a non-finite
// value or a shape that differs.
func eg2Worst(t *testing.T, m *embeddinggemma2.Model, a *eg2Accel, inputs [][]int) float64 {
	t.Helper()
	worst := 1.0
	E := m.Dim()
	for ci, ids := range inputs {
		cpuLast, cpuLayers, err := m.ForwardCPU(ids, true)
		if err != nil {
			t.Fatal(err)
		}
		last, layers, err := a.Forward(ids, true)
		if err != nil {
			t.Fatalf("case %d (%d tokens): cuda forward: %v", ci, len(ids), err)
		}
		if len(layers) != len(cpuLayers) || len(last) != len(cpuLast) {
			t.Fatalf("case %d: %d layer states and %d values, CPU %d and %d", ci, len(layers), len(last), len(cpuLayers), len(cpuLast))
		}
		lw, lwAt := 1.0, -1
		for li := range cpuLayers {
			if !eg2Finite(layers[li]) {
				t.Fatalf("case %d: layer state %d is not finite", ci, li)
			}
			if c := eg2Cos(layers[li], cpuLayers[li]); c < lw {
				lw, lwAt = c, li
			}
		}
		if !eg2Finite(last) {
			t.Fatalf("case %d: the last hidden state is not finite", ci)
		}
		lc := eg2Cos(last, cpuLast)
		ec := eg2Cos(eg2PoolNorm(last, len(ids), E, E), eg2PoolNorm(cpuLast, len(ids), E, E))
		t.Logf("case %d (%d tokens): worst layer state %.9f (index %d), last hidden %.9f, embedding %.9f", ci, len(ids), lw, lwAt, lc, ec)
		worst = min(worst, lw, lc, ec)
	}
	return worst
}

// TestEG2CUDA_matchesCPU: on the tiny fixture, the CUDA accelerator's forward matches the CPU forward to cosine 0.9999
// on every layer's input, the last layer's output, the projected last hidden state and the pooled embedding, at both
// head geometries and both attention kinds.
func TestEG2CUDA_matchesCPU(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	if w := eg2Worst(t, m, a, eg2TinyInputs(t, m.Config().VocabSize)); w < 0.9999 {
		t.Fatalf("worst cosine %.9f, under 0.9999", w)
	}
}

// TestEG2CUDA_defectsNoticed: each planted defect alone (a sliding window one wider or one narrower than the radius,
// v normed with a weight) takes some state of some tiny input under the bar TestEG2CUDA_matchesCPU holds.
func TestEG2CUDA_defectsNoticed(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	inputs := eg2TinyInputs(t, m.Config().VocabSize)
	for _, d := range []struct {
		name string
		d    eg2Defect
	}{{"window W+1", eg2Defect{windowWide: true}}, {"window W-1", eg2Defect{windowNarrow: true}}, {"weighted v norm", eg2Defect{vNormWeight: true}}} {
		a.defect = d.d
		w := eg2Worst(t, m, a, inputs)
		t.Logf("%s: worst cosine %.9f", d.name, w)
		if w >= 0.9999 {
			t.Errorf("%s: worst cosine %.9f; the defect went unnoticed", d.name, w)
		}
	}
	a.defect = eg2Defect{}
}

// TestEG2CUDA_runsOnDevice: a forward is device work. Each layer launches attention_seg once on the device the
// accelerator opened, at its own key ranges, and the strided copies gather q, k and v; the test logs the device.
func TestEG2CUDA_runsOnDevice(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	c := m.Config()
	var mu sync.Mutex
	var launches, attn, copies int
	attnPtr := a.ops.vit.AttentionSeg.Ptr()
	gpu.TraceForTest = func(e gpu.TraceEvent) {
		mu.Lock()
		defer mu.Unlock()
		switch e.Op {
		case gpu.TraceLaunch:
			launches++
			if e.Pipe == attnPtr {
				attn++
			}
		case gpu.TraceCopy2D:
			copies++
		}
	}
	defer func() { gpu.TraceForTest = nil }()
	if _, _, err := a.Forward([]int{2, 17, 33, 41, 5, 99, 120, 7, 1}, false); err != nil {
		t.Fatal(err)
	}
	gpu.TraceForTest = nil
	var name string
	if err := a.ops.do(func() error { name = a.ops.dev.Name(); return nil }); err != nil {
		t.Fatal(err)
	}
	t.Logf("device %q: %d launches, %d attention_seg, %d strided copies over %d layers", name, launches, attn, copies, c.Layers)
	if attn != c.Layers {
		t.Errorf("%d attention_seg launches, want one per layer (%d)", attn, c.Layers)
	}
	if want := c.Layers * (2 + 2*c.Heads); copies != want {
		t.Errorf("%d strided copies, want %d (q, each head's k and v, and the layer's per-layer input)", copies, want)
	}
	if launches < 20*c.Layers {
		t.Errorf("%d launches for %d layers", launches, c.Layers)
	}
}

// TestEG2CUDA_bounds: a sliding row's key range is every key within the radius, the radius itself included, clipped to
// the input; a full row's is the whole input.
func TestEG2CUDA_bounds(t *testing.T) {
	lo, hi := eg2Bounds(10, 3)
	for i := range 10 {
		for j := range 10 {
			in := int32(j) >= lo[i] && int32(j) < hi[i]
			if want := j-i <= 3 && i-j <= 3; in != want {
				t.Errorf("row %d key %d: in range %v, want %v", i, j, in, want)
			}
		}
	}
	lo, hi = eg2Bounds(5, -1)
	for i := range 5 {
		if lo[i] != 0 || hi[i] != 5 {
			t.Errorf("full row %d: [%d, %d), want [0, 5)", i, lo[i], hi[i])
		}
	}
}

// TestEG2CUDA_contract: ForwardEmbeds leaves its input alone; a repeated forward gives the same values bit for bit,
// also when several goroutines share the accelerator; over-long, empty and out-of-vocabulary inputs are refused; and
// after Close (twice) a forward is an error.
func TestEG2CUDA_contract(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	ids := []int{2, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 1}
	x, err := m.EmbedTokens(ids)
	if err != nil {
		t.Fatal(err)
	}
	keep := slices.Clone(x)
	ref, _, err := a.ForwardEmbeds(x, len(ids), false)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(x, keep) {
		t.Fatal("ForwardEmbeds wrote its input")
	}
	var wg sync.WaitGroup
	errs := make([]error, 4)
	for g := range errs {
		wg.Go(func() {
			for range 3 {
				got, _, err := a.Forward(ids, false)
				if err == nil && !slices.Equal(got, ref) {
					err = errors.New("a repeated forward differs")
				}
				if err != nil {
					errs[g] = err
					return
				}
			}
		})
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
	if _, _, err := a.Forward(nil, false); err == nil {
		t.Error("an empty input was accepted")
	}
	if _, _, err := a.Forward([]int{m.Config().VocabSize}, false); err == nil {
		t.Error("an out-of-vocabulary id was accepted")
	}
	H := m.Config().Hidden
	if _, _, err := a.ForwardEmbeds(make([]float32, (eg2MaxTokens+1)*H), eg2MaxTokens+1, false); err == nil {
		t.Error("an input over the accelerator's length was accepted")
	}
	if err := errors.Join(a.Close(), a.Close()); err != nil {
		t.Fatal(err)
	}
	if _, _, err := a.Forward(ids, false); !errors.Is(err, errEG2Closed) {
		t.Fatalf("forward after Close: %v, want errEG2Closed", err)
	}
}
