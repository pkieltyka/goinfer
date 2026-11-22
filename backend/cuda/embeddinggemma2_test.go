//go:build cuda

package cuda

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/tokenizer"
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

// eg2Counts is what the device saw while f ran: launches (attention_seg among them), strided copies and allocations.
type eg2Counts struct{ launches, attn, copies, allocs int }

func eg2Count(t *testing.T, a *eg2Accel, f func()) eg2Counts {
	t.Helper()
	var mu sync.Mutex
	var n eg2Counts
	attnPtr := a.ops.vit.AttentionSeg.Ptr()
	gpu.TraceForTest = func(e gpu.TraceEvent) {
		mu.Lock()
		defer mu.Unlock()
		switch e.Op {
		case gpu.TraceLaunch:
			n.launches++
			if e.Pipe == attnPtr {
				n.attn++
			}
		case gpu.TraceCopy2D:
			n.copies++
		case gpu.TraceAlloc:
			n.allocs++
		}
	}
	defer func() { gpu.TraceForTest = nil }()
	f()
	return n
}

// eg2FirstDiff is the first value at which got and want differ bit for bit, as row and column of width w, or "".
func eg2FirstDiff(got, want []float32, w int) string {
	if len(got) != len(want) {
		return fmt.Sprintf("%d values, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			return fmt.Sprintf("row %d column %d: %#08x (%v), want %#08x (%v)", i/w, i%w, math.Float32bits(got[i]), got[i],
				math.Float32bits(want[i]), want[i])
		}
	}
	return ""
}

func eg2SeqIDs(n, seed, vocab int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*2654435761 + seed*40503 + 17) % vocab
		if ids[i] < 0 {
			ids[i] = -ids[i]
		}
	}
	return ids
}

// TestEG2CUDA_batchMatchesSingles: a batch of inputs of mixed lengths (1 to 300 tokens; both sides of the radius-3
// window, both head geometries) gives each input's projected rows bit for bit what Forward gives that input alone, in
// either order, and each is within the CPU bar.
func TestEG2CUDA_batchMatchesSingles(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	V, E := m.Config().VocabSize, m.Dim()
	var inputs [][]int
	for i, n := range []int{1, 2, 7, 9, 64, 300} {
		inputs = append(inputs, eg2SeqIDs(n, i+1, V))
	}
	singles := make([][]float32, len(inputs))
	for i, ids := range inputs {
		h, _, err := a.Forward(ids, false)
		if err != nil {
			t.Fatal(err)
		}
		singles[i] = h
		cpu, _, err := m.ForwardCPU(ids, false)
		if err != nil {
			t.Fatal(err)
		}
		if c := eg2Cos(h, cpu); !(c >= 0.9999) {
			t.Fatalf("input %d (%d tokens) alone: cosine %.9f with the CPU", i, len(ids), c)
		}
	}
	for _, order := range [][]int{{0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}, {3, 0, 5, 1, 4, 2}} {
		batch := make([][]int, len(order))
		for j, i := range order {
			batch[j] = inputs[i]
		}
		got, err := a.ForwardBatch(batch)
		if err != nil {
			t.Fatal(err)
		}
		for j, i := range order {
			if d := eg2FirstDiff(got[j], singles[i], E); d != "" {
				t.Fatalf("order %v, input %d (%d tokens): batch differs from the input alone at %s", order, i, len(inputs[i]), d)
			}
		}
	}
}

// TestEG2CUDA_batchNoLeakage: what sits beside an input in a batch — other tokens, another length, before or after it,
// one or two neighbours — never changes that input's rows by a bit.
func TestEG2CUDA_batchNoLeakage(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	V, E := m.Config().VocabSize, m.Dim()
	b := eg2SeqIDs(9, 99, V)
	want, _, err := a.Forward(b, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{1, 3, 4, 8, 64} {
		for seed := range 2 {
			other := eg2SeqIDs(n, 1000+seed, V)
			for _, c := range []struct {
				name  string
				batch [][]int
				at    int
			}{{"after", [][]int{other, b}, 1}, {"before", [][]int{b, other}, 0}, {"between", [][]int{other, b, other}, 1}} {
				got, err := a.ForwardBatch(c.batch)
				if err != nil {
					t.Fatal(err)
				}
				if d := eg2FirstDiff(got[c.at], want, E); d != "" {
					t.Fatalf("neighbour of %d tokens (seed %d) %s: the 9-token input differs at %s", n, seed, c.name, d)
				}
			}
		}
	}
}

// TestEG2CUDA_batchRopeSegments: the batch's RoPE tables hold, for every input, the cached table's leading rows bit for
// bit: each input starts again at position 0, also after a nonempty one, for both (head dim, theta) pairs.
func TestEG2CUDA_batchRopeSegments(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	lens := []int{5, 9, 3}
	if err := a.ops.do(func() error {
		defer a.ops.releaseScratch()
		var pending []Buffer
		var relErr error
		tabs, err := a.batchRope(lens, &pending, &relErr)
		if err != nil {
			return err
		}
		if err := a.ops.finish(); err != nil {
			return err
		}
		if len(tabs) != 2 {
			return fmt.Errorf("%d (head dim, theta) pairs, want the fixture's 2", len(tabs))
		}
		for k, tb := range tabs {
			cs, sn, err := a.rope(9, k.hd, k.theta, &pending, &relErr)
			if err != nil {
				return err
			}
			for j, src := range []Buffer{cs, sn} {
				ref := make([]float32, 9*k.hd)
				got := make([]float32, 17*k.hd)
				if err := errors.Join(gpu.Download(src, ref), gpu.Download(tb[j], got)); err != nil {
					return err
				}
				off := 0
				for b, n := range lens {
					if d := eg2FirstDiff(got[off*k.hd:(off+n)*k.hd], ref[:n*k.hd], k.hd); d != "" {
						return fmt.Errorf("hd %d theta %g table %d, input %d: %s", k.hd, k.theta, j, b, d)
					}
					off += n
				}
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// TestEG2CUDA_batchContract: a batch's outputs are the caller's (appending to one leaves the next alone; a later
// forward changes neither); a batch totaling the single-input cap runs; an empty batch, an empty, out-of-vocabulary
// or oversized member (in the last slot) are refused before any device allocation, copy or launch; a
// batch of one issues what Forward issues; a batch of N issues one forward's launches (one attention_seg per layer)
// and Forward's copies plus two per input for each RoPE pair; and after Close a batch is an error.
func TestEG2CUDA_batchContract(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	c := m.Config()
	V, E := c.VocabSize, m.Dim()
	in := [][]int{eg2SeqIDs(7, 1, V), eg2SeqIDs(5, 2, V), eg2SeqIDs(12, 3, V), eg2SeqIDs(2, 4, V)}
	got, err := a.ForwardBatch(in)
	if err != nil {
		t.Fatal(err)
	}
	keep := slices.Clone(got[1])
	_ = append(got[0], 1, 2, 3)
	if _, err := a.ForwardBatch(in[:2]); err != nil {
		t.Fatal(err)
	}
	if d := eg2FirstDiff(got[1], keep, E); d != "" {
		t.Fatalf("an append to one output or a later forward changed the next output at %s", d)
	}
	if _, err := a.ForwardBatch([][]int{eg2SeqIDs(eg2MaxTokens-5, 5, V), eg2SeqIDs(5, 6, V)}); err != nil {
		t.Fatalf("a batch of exactly %d positions: %v", eg2MaxTokens, err)
	}
	for _, bad := range []struct {
		name  string
		batch [][]int
	}{
		{"empty batch", nil},
		{"empty last member", [][]int{in[0], {}}},
		{"out-of-vocabulary last member", [][]int{in[0], {1, V}}},
		{"oversized member", [][]int{in[0], eg2SeqIDs(eg2MaxTokens+1, 6, V)}},
	} {
		var err error
		n := eg2Count(t, a, func() { _, err = a.ForwardBatch(bad.batch) })
		if err == nil || n != (eg2Counts{}) {
			t.Errorf("%s: error %v, device work %+v; want a refusal and none", bad.name, err, n)
		}
	}
	// Refused before the gather too: an out-of-vocabulary id at the end of a batch at the cap allocates far less host
	// memory than the gather buffer would take.
	capBatch := [][]int{eg2SeqIDs(eg2MaxTokens-5, 5, V), append(eg2SeqIDs(4, 6, V), V)}
	var ms0, ms1 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	_, cerr := a.ForwardBatch(capBatch)
	runtime.ReadMemStats(&ms1)
	if gather := uint64(eg2MaxTokens * c.Hidden * 4); cerr == nil || ms1.TotalAlloc-ms0.TotalAlloc >= gather/2 {
		t.Errorf("an out-of-vocabulary id at the cap: error %v, %d host bytes allocated (the gather buffer is %d)", cerr,
			ms1.TotalAlloc-ms0.TotalAlloc, gather)
	}
	single := eg2Count(t, a, func() {
		if _, _, err := a.Forward(in[2], false); err != nil {
			t.Fatal(err)
		}
	})
	one := eg2Count(t, a, func() {
		if _, err := a.ForwardBatch(in[2:3]); err != nil {
			t.Fatal(err)
		}
	})
	if one != single {
		t.Errorf("a batch of one: %+v, Forward: %+v", one, single)
	}
	var cat []int
	for _, ids := range in {
		cat = append(cat, ids...)
	}
	flat := eg2Count(t, a, func() {
		if _, _, err := a.Forward(cat, false); err != nil {
			t.Fatal(err)
		}
	})
	batch := eg2Count(t, a, func() {
		if _, err := a.ForwardBatch(in); err != nil {
			t.Fatal(err)
		}
	})
	t.Logf("%d tokens as one input: %+v; as %d inputs: %+v", len(cat), flat, len(in), batch)
	if batch.launches != flat.launches || batch.attn != c.Layers || batch.copies != flat.copies+2*2*len(in) {
		t.Errorf("a batch of %d: %+v; want %d launches with %d attention_seg and %d copies", len(in), batch, flat.launches,
			c.Layers, flat.copies+2*2*len(in))
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ForwardBatch(in); !errors.Is(err, errEG2Closed) {
		t.Fatalf("a batch after Close: %v, want errEG2Closed", err)
	}
}

// TestEG2CUDA_failedDrain: a failed forward is drained before its scratch is freed; when that drain fails the
// accelerator refuses forwards and batches and keeps its device memory, and Close frees nothing until a drain
// succeeds, then empties the ledger; a second Close does nothing.
func TestEG2CUDA_failedDrain(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	V := m.Config().VocabSize
	in := [][]int{eg2SeqIDs(7, 1, V), eg2SeqIDs(5, 2, V)}
	if _, err := a.ForwardBatch(in); err != nil {
		t.Fatal(err)
	}
	dev := a.ops.dev
	held, _ := dev.LedgerLen()
	a.defect.failForward = true
	if _, err := a.ForwardBatch(in); err == nil {
		t.Fatal("the planted failure was not reported")
	}
	if n, _ := dev.LedgerLen(); n != held || a.broken != nil {
		t.Fatalf("after a drained failure: %d allocations (want %d), broken %v", n, held, a.broken)
	}
	a.defect.drainFails = 1
	if _, err := a.ForwardBatch(in); err == nil {
		t.Fatal("the planted failure was not reported")
	}
	a.defect.failForward = false
	kept, _ := dev.LedgerLen()
	if a.broken == nil || kept <= held {
		t.Fatalf("after an undrained failure: broken %v, %d allocations (held %d before)", a.broken, kept, held)
	}
	if _, err := a.ForwardBatch(in); !errors.Is(err, a.broken) {
		t.Fatalf("a batch on a broken accelerator: %v", err)
	}
	if _, _, err := a.Forward(in[0], false); !errors.Is(err, a.broken) {
		t.Fatalf("a forward on a broken accelerator: %v", err)
	}
	a.defect.drainFails = 1
	if err := a.Close(); err == nil {
		t.Fatal("Close returned nil though its drain failed")
	}
	if n, _ := dev.LedgerLen(); n != kept || a.closed {
		t.Fatalf("after a failed Close: %d allocations (want %d), closed %v", n, kept, a.closed)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close once the drain recovers: %v", err)
	}
	if allocs, objs := dev.LedgerLen(); allocs != 0 || objs != 0 {
		t.Fatalf("after Close the ledger holds %d allocations and %d objects", allocs, objs)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("a second Close: %v", err)
	}
}

// TestEG2CUDA_realBatchParity: on the real checkpoint (GOINFER_HEAVY_TESTS=1, GOINFER_EG2_DIR), ForwardBatch gives every
// input bit for bit what Forward gives it alone: nine inputs of 1 to 2,048 tokens around the 512-radius window in one
// batch (6,673 rows); a 1,025-token input beside neighbours of 1, 512, 513 and 1,500 tokens; and the golden's 48 texts
// in batches of up to 8,192 rows, each pooled embedding at cosine >= 0.9999 with the sentence-transformers reference.
// Two inputs alone are also checked against the CPU forward.
func TestEG2CUDA_realBatchParity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" || os.Getenv("GOINFER_EG2_DIR") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and GOINFER_EG2_DIR (loads the real checkpoint)")
	}
	dir := os.Getenv("GOINFER_EG2_DIR")
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not local disk (CLAUDE.md)", dir)
	}
	m, err := embeddinggemma2.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("cuda")
	if err != nil {
		t.Fatalf("the cuda accelerator: %v", err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	var name string
	if err := a.ops.do(func() error { name = a.ops.dev.Name(); return nil }); err != nil {
		t.Fatal(err)
	}
	c := m.Config()
	V, E := c.VocabSize, m.Dim()
	t.Logf("device %q; %s; window radius %d", name, dir, c.SlidingWindow)
	single := map[string][]float32{}
	alone := func(key string, ids []int) []float32 {
		if h, ok := single[key]; ok {
			return h
		}
		h, _, err := a.Forward(ids, false)
		if err != nil {
			t.Fatalf("%s alone: %v", key, err)
		}
		single[key] = h
		return h
	}
	check := func(what string, batch [][]int, keys []string) {
		t.Helper()
		got, err := a.ForwardBatch(batch)
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		for i, ids := range batch {
			if d := eg2FirstDiff(got[i], alone(keys[i], ids), E); d != "" {
				t.Fatalf("%s, input %d (%d tokens): differs from the input alone at %s", what, i, len(ids), d)
			}
		}
	}
	lens := []int{1, 13, 511, 512, 513, 1024, 1025, 1026, 2048}
	var mixed [][]int
	var keys []string
	for i, n := range lens {
		mixed = append(mixed, eg2SeqIDs(n, 7+i, V))
		keys = append(keys, fmt.Sprintf("len%d", n))
	}
	check("nine lengths around the window", mixed, keys)
	target := mixed[6] // 1,025 tokens
	for _, n := range []int{1, 512, 513, 1500} {
		other := eg2SeqIDs(n, 500+n, V)
		on := fmt.Sprintf("nb%d", n)
		check(fmt.Sprintf("1,025 tokens after %d", n), [][]int{other, target}, []string{on, "len1025"})
		check(fmt.Sprintf("1,025 tokens before %d", n), [][]int{target, other}, []string{"len1025", on})
	}
	for _, i := range []int{4, 6} { // 513 and 1,025 tokens alone, against the CPU
		cpu, _, err := m.ForwardCPU(mixed[i], false)
		if err != nil {
			t.Fatal(err)
		}
		if cs := eg2Cos(single[keys[i]], cpu); !(cs >= 0.9999) {
			t.Fatalf("%d tokens alone: cosine %.9f with the CPU", lens[i], cs)
		}
	}
	raw, err := os.ReadFile("../../testdata/embeddinggemma2-real/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Items []eg2GoldenItem `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	worst, groups := 1.0, 0
	for i := 0; i < len(g.Items); {
		var batch [][]int
		var bk []string
		total := 0
		start := i
		for ; i < len(g.Items) && total+len(g.Items[i].IDs) <= eg2MaxTokens; i++ {
			batch, bk, total = append(batch, g.Items[i].IDs), append(bk, fmt.Sprintf("golden%d", i)), total+len(g.Items[i].IDs)
		}
		got, err := a.ForwardBatch(batch)
		if err != nil {
			t.Fatal(err)
		}
		groups++
		for j, ids := range batch {
			if d := eg2FirstDiff(got[j], alone(bk[j], ids), E); d != "" {
				t.Fatalf("golden item %d in a batch: differs from the item alone at %s", start+j, d)
			}
			cs := eg2Cos(eg2PoolNorm(got[j], len(ids), E, E), g.Items[start+j].Embedding)
			worst = min(worst, cs)
			if !(cs >= 0.9999) {
				t.Errorf("golden item %d: cosine %.9f with the reference", start+j, cs)
			}
		}
	}
	t.Logf("RESULT bit-identical to singletons in every batch; golden: %d texts in %d batches, worst cosine %.12f", len(g.Items), groups, worst)
}

// TestEG2CUDA_encoderBatch: Encoder.EmbedIDsBatch on the CUDA accelerator runs the batch, one forward with one
// attention_seg per layer for the whole batch, and gives every input bit for bit what EmbedIDs gives it alone.
func TestEG2CUDA_encoderBatch(t *testing.T) {
	m := eg2TinyModel(t)
	vocab := map[string]int{"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
	raw, err := json.Marshal(map[string]any{
		"added_tokens": []map[string]any{{"id": 0, "content": "<pad>"}, {"id": 1, "content": "<eos>"}, {"id": 2, "content": "<bos>"},
			{"id": 3, "content": "<unk>"}},
		"model": map[string]any{"type": "BPE", "unk_token": "<unk>", "vocab": vocab, "merges": [][2]string{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadJSONBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := embeddinggemma2.NewEncoder(m, tok, nil)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := enc.UseAccelerator("cuda")
	if err != nil {
		t.Skipf("no CUDA device for EmbeddingGemma 2: %v", err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	V := m.Config().VocabSize
	in := [][]int{eg2SeqIDs(9, 1, V), eg2SeqIDs(64, 2, V), eg2SeqIDs(2, 3, V)}
	var got [][]float32
	n := eg2Count(t, a, func() {
		if got, err = enc.EmbedIDsBatch(in); err != nil {
			t.Fatal(err)
		}
	})
	if n.attn != m.Config().Layers {
		t.Fatalf("EmbedIDsBatch: %d attention_seg launches, want one forward's %d", n.attn, m.Config().Layers)
	}
	for i, ids := range in {
		want, err := enc.EmbedIDs(ids)
		if err != nil {
			t.Fatal(err)
		}
		if d := eg2FirstDiff(got[i], want, len(want)); d != "" {
			t.Fatalf("input %d: EmbedIDsBatch differs from EmbedIDs at %s", i, d)
		}
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

// eg2GoldenItem is one text of the real golden: its prompt name, the reference's ids and its 768-wide embedding.
type eg2GoldenItem struct {
	Prompt    string    `json:"prompt"`
	IDs       []int     `json:"ids"`
	Embedding []float32 `json:"embedding"`
}

// TestEG2CUDA_realParity: the real google/embeddinggemma-2 (GOINFER_EG2_DIR, a local-disk checkpoint) on the CUDA
// accelerator, through Accelerator.Forward on the committed sentence-transformers golden's own ids
// (testdata/embeddinggemma2-real/golden.json: 48 texts, 10 to 1,771 tokens, so the longest reach past both edges of the
// 512-radius window). Every pooled, normalised embedding is finite and at cosine >= 0.9999 with the reference, at 768
// and at its 512-wide Matryoshka prefix. Three inputs (the shortest, the longest under the window's width and the
// longest) are also run on the CPU forward, and every layer state and the projected hidden state must match it at
// 0.9999. Wall time is logged for reading, not as a speed result.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_EG2_DIR=<checkpoint> go test -tags cuda -count=1 -run '^TestEG2CUDA_realParity$' -v ./backend/cuda/
func TestEG2CUDA_realParity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and GOINFER_EG2_DIR (loads the real checkpoint)")
	}
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not local disk (CLAUDE.md)", dir)
	}
	raw, err := os.ReadFile("../../testdata/embeddinggemma2-real/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Revision string          `json:"model_revision"`
		Items    []eg2GoldenItem `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	m, err := embeddinggemma2.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	tLoad := time.Since(t0)
	acc, err := m.NewAccelerator("cuda")
	if err != nil {
		t.Fatalf("the cuda accelerator: %v", err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	tAttach := time.Since(t0) - tLoad
	var name string
	if err := a.ops.do(func() error { name = a.ops.dev.Name(); return nil }); err != nil {
		t.Fatal(err)
	}
	c := m.Config()
	t.Logf("device %q; checkpoint %s (golden revision %s); %d layers, window radius %d; host load %.1f s, device upload %.1f s",
		name, dir, g.Revision, c.Layers, c.SlidingWindow, tLoad.Seconds(), tAttach.Seconds())
	E := m.Dim()
	const prefix = 512
	worst, worstP := 1.0, 1.0
	var total time.Duration
	for i, it := range g.Items {
		s := time.Now()
		h, _, err := acc.Forward(it.IDs, false)
		if err != nil {
			t.Fatalf("item %d (%d tokens): %v", i, len(it.IDs), err)
		}
		d := time.Since(s)
		total += d
		v := eg2PoolNorm(h, len(it.IDs), E, E)
		if !eg2Finite(v) {
			t.Fatalf("item %d: the embedding is not finite", i)
		}
		cs := eg2Cos(v, it.Embedding)
		cp := eg2Cos(eg2PoolNorm(h, len(it.IDs), E, prefix), it.Embedding[:prefix])
		worst, worstP = min(worst, cs), min(worstP, cp)
		t.Logf("item %2d %-15q %5d tokens: 1-cosine %.2e, %d-prefix %.2e; %.1f ms", i, it.Prompt, len(it.IDs), 1-cs, prefix, 1-cp, d.Seconds()*1e3)
		if !(cs >= 0.9999) || !(cp >= 0.9999) {
			t.Errorf("item %d (%d tokens): cosine %.9f, %d-prefix %.9f, under 0.9999", i, len(it.IDs), cs, prefix, cp)
		}
	}
	t.Logf("RESULT %d texts: worst cosine %.12f, worst %d-prefix %.12f; device forwards %.2f s in all", len(g.Items), worst, prefix, worstP, total.Seconds())
	order := slices.Clone(g.Items)
	slices.SortFunc(order, func(x, y eg2GoldenItem) int { return len(x.IDs) - len(y.IDs) })
	var underWindow []int
	for _, it := range order {
		if len(it.IDs) <= 2*c.SlidingWindow+1 {
			underWindow = it.IDs
		}
	}
	var launches, attn int
	attnPtr := a.ops.vit.AttentionSeg.Ptr()
	gpu.TraceForTest = func(e gpu.TraceEvent) {
		if e.Op == gpu.TraceLaunch {
			launches++
			if e.Pipe == attnPtr {
				attn++
			}
		}
	}
	_, _, err = acc.Forward(order[len(order)-1].IDs, false)
	gpu.TraceForTest = nil
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("the %d-token forward: %d device launches, %d of them attention_seg", len(order[len(order)-1].IDs), launches, attn)
	if attn != c.Layers {
		t.Errorf("%d attention_seg launches, want one per layer (%d)", attn, c.Layers)
	}
	for _, ids := range [][]int{order[0].IDs, underWindow, order[len(order)-1].IDs} {
		cpuLast, cpuLayers, err := m.ForwardCPU(ids, true)
		if err != nil {
			t.Fatal(err)
		}
		last, layers, err := acc.Forward(ids, true)
		if err != nil {
			t.Fatal(err)
		}
		lw, lwAt := 1.0, -1
		for li := range cpuLayers {
			if cc := eg2Cos(layers[li], cpuLayers[li]); cc < lw {
				lw, lwAt = cc, li
			}
		}
		lc := eg2Cos(last, cpuLast)
		t.Logf("against the CPU forward, %d tokens: worst layer state 1-cosine %.2e (index %d), last hidden %.2e", len(ids), 1-lw, lwAt, 1-lc)
		if !(lw >= 0.9999) || !(lc >= 0.9999) {
			t.Errorf("%d tokens: worst layer state %.9f at %d, last hidden %.9f, under 0.9999", len(ids), lw, lwAt, lc)
		}
	}
}

func TestEG2CombinedTiny(t *testing.T) {
	if !slices.Contains(embeddinggemma2.Accelerators(), "cuda-eg2-combined") {
		t.Skip("requires eg2_combined build tag")
	}
	checkCombined(t, eg2TinyModel(t), [][]int{{36}, {37}, {12, 24}, {12, 25}, {127, 10}, {64, 64}, {5, 128}, {129, 1}})
}

func TestEG2CombinedReal(t *testing.T) {
	if !slices.Contains(embeddinggemma2.Accelerators(), "cuda-eg2-combined") {
		t.Skip("requires eg2_combined build tag")
	}
	dir := os.Getenv("GOINFER_EG2_DIR")
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" || dir == "" {
		t.Skip("requires opt-in real checkpoint")
	}
	model, err := embeddinggemma2.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	checkCombined(t, model, [][]int{{36}, {37}, {7, 127}, {128, 9}, {511, 13}, {31, 513}})
}

func checkCombined(t *testing.T, model *embeddinggemma2.Model, shapes [][]int) {
	t.Helper()
	accel, err := model.NewAccelerator("cuda-eg2-combined")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := accel.Close(); err != nil {
			t.Error(err)
		}
	})
	a := accel.(*eg2Accel)
	transpose, attention := a.combinedTranspose, a.combinedAttention
	for _, shape := range shapes {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			a.combinedTranspose, a.combinedAttention, a.ops.shortTiled = Pipeline{}, Pipeline{}, false
			ids := make([][]int, len(shape))
			want := make([][]float32, len(shape))
			total, longest := 0, 0
			for i, n := range shape {
				ids[i] = eg2SeqIDs(n, 13+i, model.Config().VocabSize)
				total, longest = total+n, max(longest, n)
				out, _, err := a.Forward(ids[i], false)
				if err != nil {
					t.Fatal(err)
				}
				want[i] = out
			}
			a.combinedTranspose, a.combinedAttention, a.ops.shortTiled = transpose, attention, true
			var tiled, transposes, attn int
			gpu.TraceForTest = func(event gpu.TraceEvent) {
				if event.Op != gpu.TraceLaunch {
					return
				}
				switch event.Pipe {
				case a.ops.vit.GEMMF32Tiled.Ptr():
					tiled++
				case transpose.Ptr():
					transposes++
				case attention.Ptr():
					attn++
				}
			}
			defer func() { gpu.TraceForTest = nil }()
			out, err := a.ForwardBatch(ids)
			if err != nil {
				t.Fatal(err)
			}
			for i, got := range out {
				if diff := eg2FirstDiff(got, want[i], model.Dim()); diff != "" {
					t.Fatalf("input %d differs at %s", i, diff)
				}
			}
			if (tiled > 0) != (total <= 36) {
				t.Errorf("total rows %d produced %d short tiled launches", total, tiled)
			}
			wantAttn := 0
			if longest >= 128 {
				wantAttn = model.Config().Layers
			}
			if attn != wantAttn || transposes != wantAttn {
				t.Errorf("longest input %d, transposes %d attention %d, wanted %d each", longest, transposes, attn, wantAttn)
			}
		})
	}
}

func TestEG2CUDA_largeAggregateBatch(t *testing.T) {
	m := eg2TinyModel(t)
	ids := make([][]int, 6)
	for i, length := range []int{2048, 2048, 2048, 2048, 2048, 2047} {
		ids[i] = eg2SeqIDs(length, i+1, m.Config().VocabSize)
	}
	for _, selector := range []string{"cuda", "cuda-eg2-combined"} {
		t.Run(selector, func(t *testing.T) {
			a := newEG2ForTest(t, m)
			if selector == "cuda-eg2-combined" {
				if err := a.enableCombined(); err != nil {
					t.Fatal(err)
				}
			}
			batch, err := a.ForwardBatch(ids)
			if err != nil {
				t.Fatal(err)
			}
			for i, input := range ids {
				single, _, err := a.Forward(input, false)
				if err != nil {
					t.Fatal(err)
				}
				if diff := eg2FirstDiff(batch[i], single, m.Dim()); diff != "" {
					t.Fatalf("input %d differs at %s", i, diff)
				}
			}
		})
	}
}

func TestEG2CUDA_aggregateIndexLimit(t *testing.T) {
	a := &eg2Accel{cfg: embeddinggemma2.Config{Hidden: 65536, VocabSize: 1, MaxPositions: eg2MaxTokens}}
	input := make([]int, eg2MaxTokens)
	// This crosses signed element indexing even though every member is valid.
	if _, err := a.ForwardBatch([][]int{input, input, input, input}); err == nil {
		t.Fatal("CUDA accepted an aggregate that would overflow its indexing")
	}
}

func TestEG2CUDA_refusesOversizedIDsBeforeGather(t *testing.T) {
	const width = 512
	cfg := embeddinggemma2.Config{Hidden: width, VocabSize: 1, MaxPositions: eg2MaxTokens + 10}
	a := eg2Accel{cfg: cfg}
	ids := make([]int, eg2MaxTokens+1)
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	_, _, err := a.Forward(ids, false)
	runtime.ReadMemStats(&after)
	if err == nil || !strings.Contains(err.Error(), "over the accelerator") {
		t.Fatalf("oversized input: %v", err)
	}
	gatherBytes := uint64(len(ids) * width * 4)
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated >= gatherBytes {
		t.Fatalf("refused input allocated %d bytes, enough for the %d-byte gather", allocated, gatherBytes)
	}
}

func TestEG2CUDA_closedRejectsIDsBeforeGather(t *testing.T) {
	a := eg2Accel{closed: true}
	if _, _, err := a.Forward([]int{0}, false); err != errEG2Closed {
		t.Fatalf("closed accelerator returned %v, want %v", err, errEG2Closed)
	}
}

func TestEG2CUDA_configuredPositions(t *testing.T) {
	a := newEG2ForTest(t, eg2TinyModel(t))
	a.cfg.MaxPositions = 4
	ids := make([]int, 5)
	_, _, singleErr := a.Forward(ids, false)
	_, _, embedsErr := a.ForwardEmbeds(make([]float32, len(ids)*a.cfg.Hidden), len(ids), false)
	_, batchErr := a.ForwardBatch([][]int{ids[:4], ids})
	for name, err := range map[string]error{"IDs": singleErr, "rows": embedsErr, "batch": batchErr} {
		if err == nil || !strings.Contains(err.Error(), "over max_position_embeddings 4") {
			t.Errorf("%s: expected model-position limit error, got %v", name, err)
		}
	}
	if got, err := a.ForwardBatch([][]int{ids[:4], ids[:4]}); err != nil || len(got) != 2 {
		t.Fatalf("two valid sequences whose total exceeds the per-input cap: %d results, %v", len(got), err)
	}
}

var eg2GGUFDir = flag.String("eg2-gguf-dir", "", "directory containing the pinned EmbeddingGemma 2 Q8_0 and UD-Q6_K_XL text GGUF files")

func TestEG2GGUFCUDA(t *testing.T) {
	if *eg2GGUFDir == "" {
		t.Skip("set -eg2-gguf-dir to run the GGUF CUDA device test")
	}
	files := []struct{ suffix, digest string }{
		{"Q8_0", "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"},
		{"UD-Q6_K_XL", "dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3"},
	}
	for _, model := range files {
		t.Run(model.suffix, func(t *testing.T) {
			path := filepath.Join(*eg2GGUFDir, "embeddinggemma-2-"+model.suffix+".gguf")
			verifyEG2GGUFHash(t, path, model.digest)
			enc, err := embeddinggemma2.LoadGGUFEncoder(path, map[string]string{
				"query": "task: search result | query: ",
			})
			if err != nil {
				t.Fatal(err)
			}
			query, err := enc.Tokenize("Where is the index stored?", "query")
			if err != nil {
				t.Fatal(err)
			}
			inputs := [][]int{query, eg2GGUFIDs(511), eg2GGUFIDs(514)}
			cpu, err := enc.EmbedIDsBatch(inputs)
			if err != nil {
				t.Fatal(err)
			}
			tooLong := eg2GGUFIDs(embeddinggemma2.MaxTokens + 1)
			if vec, err := enc.EmbedIDs(tooLong); vec != nil || err == nil {
				t.Fatalf("CPU accepted %d positions", len(tooLong))
			}

			accel, err := enc.UseAccelerator("cuda")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := accel.Close(); err != nil {
					t.Error(err)
				}
			})
			if accel.Name() != "cuda" {
				t.Fatalf("attached accelerator %q, want cuda", accel.Name())
			}
			single, err := enc.EmbedIDs(query)
			if err != nil {
				t.Fatal(err)
			}
			batch, err := enc.EmbedIDsBatch(inputs)
			if err != nil {
				t.Fatal(err)
			}
			if len(batch) != len(inputs) || len(cpu) != len(inputs) {
				t.Fatalf("batch sizes CPU=%d CUDA=%d, want %d", len(cpu), len(batch), len(inputs))
			}
			for i, got := range batch {
				t.Logf("batch input with %d prepared IDs", len(inputs[i]))
				checkEG2GGUFVector(t, got, cpu[i])
			}
			checkEG2GGUFVector(t, single, cpu[0])
		})
	}
}

func verifyEG2GGUFHash(t *testing.T, path, want string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != want {
		t.Fatalf("%s SHA-256 %s, want %s", path, got, want)
	}
}

func eg2GGUFIDs(n int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (7919*i+41)%200000 + 5
	}
	ids[0], ids[n-1] = 2, 1
	return ids
}

func checkEG2GGUFVector(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != 768 || len(want) != 768 || !eg2Finite(got) || !eg2Finite(want) {
		t.Fatal("expected finite 768-dimensional CPU and CUDA vectors")
	}
	var maxError, gotNorm, wantNorm float64
	for i, x := range got {
		a, b := float64(x), float64(want[i])
		maxError = max(maxError, math.Abs(a-b))
		gotNorm += a * a
		wantNorm += b * b
	}
	if math.Abs(math.Sqrt(gotNorm)-1) > 1e-5 || math.Abs(math.Sqrt(wantNorm)-1) > 1e-5 {
		t.Fatal("expected unit-length CPU and CUDA vectors")
	}
	cosine := eg2Cos(got, want)
	t.Logf("CUDA vs CPU cosine %.12f, max coordinate error %.3g", cosine, maxError)
	if !(cosine >= 0.99999) || maxError > 3e-5 {
		t.Fatal("CUDA vector differs from the same-artifact CPU result")
	}
}

func BenchmarkEG2CUDALongBatch(b *testing.B) {
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		b.Skip("set GOINFER_EG2_DIR to the EmbeddingGemma 2 checkpoint")
	}
	e, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		b.Fatal(err)
	}
	for _, selector := range []string{"cuda", "cuda-eg2-combined"} {
		if !slices.Contains(embeddinggemma2.Accelerators(), selector) {
			continue
		}
		b.Run(selector, func(b *testing.B) {
			a, err := e.UseAccelerator(selector)
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(func() {
				if err := a.Close(); err != nil {
					b.Error(err)
				}
			})
			for _, shape := range [][2]int{{64, 64}, {8, 512}, {2, 2048}, {4, 2048}} {
				count, length := shape[0], shape[1]
				b.Run(fmt.Sprintf("%dx%d", count, length), func(b *testing.B) {
					benchmarkEG2Batch(b, e, eg2FixedBatch(e, count, length))
				})
			}
			b.Run("6x2048-ragged", func(b *testing.B) {
				ids := eg2FixedBatch(e, 6, 2048)
				ids[5] = ids[5][:2047]
				benchmarkEG2Batch(b, e, ids)
			})
		})
	}
}

func benchmarkEG2Batch(b *testing.B, e *embeddinggemma2.Encoder, batch [][]int) {
	var tokens int
	for _, input := range batch {
		tokens += len(input)
	}
	if _, err := e.EmbedIDsBatch(batch); err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := e.EmbedIDsBatch(batch); err != nil {
			b.Fatal(err)
		}
	}
	passes := float64(b.N) / b.Elapsed().Seconds()
	b.ReportMetric(float64(len(batch))*passes, "passages/s")
	b.ReportMetric(float64(tokens)*passes, "tokens/s")
}

// TestEG2CUDALargeRealBatch verifies pooled output across singleton, packed,
// split and reordered inputs on the official checkpoint.
func TestEG2CUDALargeRealBatch(t *testing.T) {
	dir := os.Getenv("GOINFER_EG2_DIR")
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" || dir == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and GOINFER_EG2_DIR")
	}
	e, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids := eg2FixedBatch(e, 6, 2048)
	ids[5] = ids[5][:2047]
	for _, selector := range []string{"cuda", "cuda-eg2-combined"} {
		t.Run(selector, func(t *testing.T) {
			if !slices.Contains(embeddinggemma2.Accelerators(), selector) {
				t.Skip("requires eg2_combined build tag")
			}
			acc, err := e.UseAccelerator(selector)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := acc.Close(); err != nil {
					t.Error(err)
				}
			})
			packed, err := e.EmbedIDsBatch(ids)
			if err != nil {
				t.Fatal(err)
			}
			reversed := slices.Clone(ids)
			slices.Reverse(reversed)
			reordered, err := e.EmbedIDsBatch(reversed)
			if err != nil {
				t.Fatal(err)
			}
			slices.Reverse(reordered)
			for i, input := range ids {
				single, err := e.EmbedIDs(input)
				if err != nil {
					t.Fatal(err)
				}
				if diff := eg2FirstDiff(packed[i], single, e.HiddenDim()); diff != "" {
					t.Errorf("input %d differs from singleton at %s", i, diff)
				}
			}
			var split [][]float32
			for _, inputs := range [][][]int{ids[:2], ids[2:]} {
				part, err := e.EmbedIDsBatch(inputs)
				if err != nil {
					t.Fatal(err)
				}
				split = append(split, part...)
			}
			for _, variant := range []struct {
				name string
				vecs [][]float32
			}{{"split", split}, {"reordered", reordered}} {
				for i, vec := range variant.vecs {
					if diff := eg2FirstDiff(packed[i], vec, e.HiddenDim()); diff != "" {
						t.Errorf("input %d differs from %s at %s", i, variant.name, diff)
					}
				}
			}
		})
	}
}

func eg2FixedBatch(e *embeddinggemma2.Encoder, count, length int) [][]int {
	batch := make([][]int, count)
	for i := range batch {
		batch[i] = eg2SeqIDs(length, i+1, e.Model().Config().VocabSize)
	}
	return batch
}

func TestEG2CUDA_RetryOOM_ClassificationContract(t *testing.T) {
	oomPanic := fmt.Errorf("cuda: executor job panicked: %w", gpu.AllocFailedPanic("typed-len", 4096, gc.ErrOutOfMemory))
	nonOOMAlloc := fmt.Errorf("cuda: executor job panicked: %w", gpu.AllocFailedPanic("typed-len", 4096, gc.ErrInvalidValue))
	latched := errors.New("cuda: latched launch error")
	drainErr := errors.New("cuda: drain failed")
	relErr := errors.New("cuda: close failed")
	broken := fmt.Errorf("cuda: broken: %w", drainErr)

	// Clean recovered CUDA OOM authorizes retry.
	got := classifyEG2ForwardErr(oomPanic, nil, nil, nil, nil, false)
	if !errors.Is(got, embeddinggemma2.ErrBatchOutOfMemory) ||
		!errors.Is(got, embeddinggemma2.ErrDeviceOutOfMemory) ||
		!errors.Is(got, gc.ErrOutOfMemory) {
		t.Fatalf("clean OOM = %v, want ErrBatchOutOfMemory + ErrDeviceOutOfMemory + gc.ErrOutOfMemory", got)
	}
	var af *gpu.AllocFailure
	if !errors.As(got, &af) {
		t.Fatalf("clean OOM lost *gpu.AllocFailure: %v", got)
	}

	// Non-OOM AllocFailure matches neither retry nor common OOM.
	gotNonOOM := classifyEG2ForwardErr(nonOOMAlloc, nil, nil, nil, nil, false)
	if errors.Is(gotNonOOM, embeddinggemma2.ErrBatchOutOfMemory) || errors.Is(gotNonOOM, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(gotNonOOM, gc.ErrInvalidValue) {
		t.Fatalf("non-OOM alloc = %v", gotNonOOM)
	}

	// Any latched, drain, release, broken, or closed condition suppresses ErrBatchOutOfMemory
	// while keeping common OOM, native OOM cause, and the fatal cause visible.
	for _, tc := range []struct {
		name                        string
		latched, drain, rel, broken error
		closed                      bool
		wantFatal                   error
	}{
		{"latched", latched, nil, nil, nil, false, latched},
		{"drain", nil, drainErr, nil, broken, false, drainErr},
		{"release", nil, nil, relErr, fmt.Errorf("rollback: %w", relErr), false, relErr},
		{"broken-no-drain", nil, nil, nil, broken, false, broken},
		{"closed", nil, nil, nil, nil, true, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := classifyEG2ForwardErr(oomPanic, tc.latched, tc.drain, tc.rel, tc.broken, tc.closed)
			if errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("%s must not match ErrBatchOutOfMemory: %v", tc.name, err)
			}
			if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) {
				t.Fatalf("%s lost common/native OOM: %v", tc.name, err)
			}
			if tc.wantFatal != nil && !errors.Is(err, tc.wantFatal) {
				t.Fatalf("%s lost fatal cause %v: %v", tc.name, tc.wantFatal, err)
			}
		})
	}

	// Attach classification maps OOM to common ErrDeviceOutOfMemory only, never ErrBatchOutOfMemory.
	m := eg2TinyModel(t)
	if _, err := newEG2AccelWithDevice(m, false, func() (*Device, error) {
		return nil, gpu.AllocFailedPanic("attach", 1024, gc.ErrOutOfMemory)
	}); !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("attach OOM = %v, want ErrDeviceOutOfMemory + gc.ErrOutOfMemory without ErrBatchOutOfMemory", err)
	}
	unrelated := errors.New("cuda: synthetic unrelated attach failure")
	if _, err := newEG2AccelWithDevice(m, false, func() (*Device, error) {
		return nil, unrelated
	}); !errors.Is(err, unrelated) || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("unrelated attach = %v", err)
	}
}

func TestEG2CUDA_RetryOOM_DeviceAcceptance(t *testing.T) {
	probe, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	probe.ReleaseObjects()

	for _, combined := range []bool{false, true} {
		arm := "packed"
		if combined {
			arm = "combined"
		}
		t.Run(arm, func(t *testing.T) {
			enc, err := embeddinggemma2.LoadEncoder(eg2TinyDir)
			if err != nil {
				t.Fatal(err)
			}
			m := enc.Model()
			V := m.Config().VocabSize
			orig := [][]int{
				eg2SeqIDs(6, 1, V),
				eg2SeqIDs(8, 2, V),
				eg2SeqIDs(5, 3, V),
				eg2SeqIDs(7, 4, V),
			}

			// Helper that attaches a fresh eg2Accel on enc with the chosen selector.
			attachFresh := func(t *testing.T) (*embeddinggemma2.Encoder, *eg2Accel) {
				t.Helper()
				e, err := embeddinggemma2.LoadEncoder(eg2TinyDir)
				if err != nil {
					t.Fatal(err)
				}
				name := fmt.Sprintf("cuda-issue25-test-%s-%d", arm, time.Now().UnixNano())
				embeddinggemma2.RegisterAccelerator(name, func(mod *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
					return newEG2Accel(mod, combined)
				})
				acc, err := e.UseAccelerator(name)
				if err != nil {
					t.Fatalf("attach %s: %v", arm, err)
				}
				return e, acc.(*eg2Accel)
			}

			// Baseline outputs on a healthy encoder for exact bit comparison.
			eBase, aBase := attachFresh(t)
			baseBatch, err := eBase.EmbedIDsBatch(orig)
			if err != nil {
				t.Fatalf("baseline EmbedIDsBatch: %v", err)
			}
			baseSingle, err := eBase.EmbedIDs(orig[0])
			if err != nil {
				t.Fatalf("baseline EmbedIDs: %v", err)
			}
			if err := aBase.Close(); err != nil {
				t.Fatalf("baseline Close: %v", err)
			}

			// (1) Pristine multi-input OOM before submit, then split retry on the same encoder.
			t.Run("pristine-batch-oom-and-split-retry", func(t *testing.T) {
				e, a := attachFresh(t)
				defer func() {
					if err := a.Close(); err != nil {
						t.Errorf("Close: %v", err)
					}
				}()
				dev := a.ops.dev
				heldBefore, _ := dev.LedgerLen()
				allocs := 0
				dev.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 5 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				in := make([][]int, len(orig))
				for i := range orig {
					in[i] = slices.Clone(orig[i])
				}
				got, err := e.EmbedIDsBatch(in)
				dev.SetAllocHookForTest(nil)
				if got != nil {
					t.Fatalf("pristine OOM returned non-nil outputs")
				}
				if !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) ||
					!errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) ||
					!errors.Is(err, gc.ErrOutOfMemory) {
					t.Fatalf("pristine OOM err = %v", err)
				}
				for i := range orig {
					if !slices.Equal(in[i], orig[i]) {
						t.Fatalf("input %d mutated", i)
					}
				}
				if heldAfter, _ := dev.LedgerLen(); heldAfter != heldBefore {
					t.Fatalf("scratch leaked on pristine OOM: ledger %d, want %d", heldAfter, heldBefore)
				}
				// Halve between complete inputs on the same encoder and check bit-exact match.
				left, err := e.EmbedIDsBatch(in[:2])
				if err != nil {
					t.Fatalf("left split retry: %v", err)
				}
				right, err := e.EmbedIDsBatch(in[2:])
				if err != nil {
					t.Fatalf("right split retry: %v", err)
				}
				combinedVecs := append(left, right...)
				for i := range orig {
					if diff := eg2FirstDiff(combinedVecs[i], baseBatch[i], m.Dim()); diff != "" {
						t.Fatalf("split output %d differed from baseline at %s", i, diff)
					}
				}
			})

			// (2a) Post-enqueue OOM (fail 2nd batchRope scratch pair after 1st key's Copy2DAsync queued).
			// (2b) Repeatedly fail 2nd RoPE allocation (sin) to prove zero partial-pair leak.
			t.Run("post-enqueue-oom-and-rope-second-alloc-leak", func(t *testing.T) {
				e, a := attachFresh(t)
				defer func() {
					if err := a.Close(); err != nil {
						t.Errorf("Close: %v", err)
					}
				}()
				dev := a.ops.dev
				// First prove repeated 2nd RoPE allocation failure leaks 0 buffers.
				// Singleton allocates 20 base scratch buffers, then rope() allocates cos (#21) and sin (#22).
				heldCold, _ := dev.LedgerLen()
				for attempt := range 5 {
					allocs := 0
					dev.SetAllocHookForTest(func(what string, nBytes int) error {
						allocs++
						if allocs == 22 { // 20 scratch + cos (#21), fail sin (#22)
							return gc.ErrOutOfMemory
						}
						return nil
					})
					_, err := e.EmbedIDs(orig[0])
					dev.SetAllocHookForTest(nil)
					if !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
						t.Fatalf("attempt %d: 2nd RoPE alloc OOM = %v, want ErrBatchOutOfMemory", attempt, err)
					}
					if n, _ := dev.LedgerLen(); n != heldCold {
						t.Fatalf("attempt %d: partial RoPE pair leaked: ledger %d, want %d", attempt, n, heldCold)
					}
					if len(a.ropes) != 0 {
						t.Fatalf("attempt %d: partial RoPE pair published in a.ropes: %d", attempt, len(a.ropes))
					}
				}
				// Warm up caches on orig[0] (length 6) so subsequent singleton at length 10 grows RoPE
				// AFTER layer 0 has queued work on the sliding head_dim and hits the full layer's RoPE growth.
				v0, err := e.EmbedIDs(orig[0])
				if err != nil {
					t.Fatalf("post-recovery singleton: %v", err)
				}
				if diff := eg2FirstDiff(v0, baseSingle, m.Dim()); diff != "" {
					t.Fatalf("post-recovery singleton differed at %s", diff)
				}
				// Now run a multi-input batch whose caches are already warm at >=8:
				// Warm caches up to 8 first.
				if _, err := e.EmbedIDsBatch(orig[:2]); err != nil {
					t.Fatalf("warm batch: %v", err)
				}
				heldWarm, _ := dev.LedgerLen()
				lBefore, wBefore := dev.Activity()
				// On warm batch: 20 base scratch + 2 batchRope buffers for key 0 (#21,#22) + Copy2DAsync queued,
				// then fail key 1's batchRope buffer (#23) AFTER copies were enqueued.
				allocs := 0
				dev.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 23 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				got, err := e.EmbedIDsBatch(orig)
				dev.SetAllocHookForTest(nil)
				_, wAfter := dev.Activity()
				if got != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("post-enqueue OOM = (%v, %v), want nil + ErrBatchOutOfMemory", got, err)
				}
				if wAfter <= wBefore {
					t.Fatalf("post-enqueue OOM did not drain queue: waits %d -> %d (launches before %d)", wBefore, wAfter, lBefore)
				}
				if n, _ := dev.LedgerLen(); n != heldWarm {
					t.Fatalf("post-enqueue OOM leaked buffers: ledger %d, want %d", n, heldWarm)
				}
				retryOut, err := e.EmbedIDsBatch(orig)
				if err != nil {
					t.Fatalf("retry after post-enqueue OOM: %v", err)
				}
				for i := range orig {
					if diff := eg2FirstDiff(retryOut[i], baseBatch[i], m.Dim()); diff != "" {
						t.Fatalf("post-enqueue retry output %d differed at %s", i, diff)
					}
				}
			})

			// (3) Fatal combinations: OOM + latched launch error, OOM + failed drain, OOM + failed free.
			t.Run("fatal-combinations-never-retry", func(t *testing.T) {
				// 3a: OOM + latched launch error (plant failForward on layer 0, then OOM on layer 1's RoPE alloc).
				// eg2TinyModel has layer 0 sliding (hd=16) and a full layer (hd=32), so singleton hits layer 0 first
				// (latching failForward) and allocates the full layer's RoPE later.
				eLatch, aLatch := attachFresh(t)
				devLatch := aLatch.ops.dev
				aLatch.defect.failForward = true
				allocs := 0
				devLatch.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 23 { // 20 scratch + layer-0 cos(#21), sin(#22); fail full-layer cos(#23)
						return gc.ErrOutOfMemory
					}
					return nil
				})
				_, err := eLatch.EmbedIDs(orig[0])
				devLatch.SetAllocHookForTest(nil)
				aLatch.defect.failForward = false
				if err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("OOM+latched must not match ErrBatchOutOfMemory: %v", err)
				}
				if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || !strings.Contains(err.Error(), "planted forward failure") {
					t.Fatalf("OOM+latched missing OOM or latched cause: %v", err)
				}
				if err := aLatch.Close(); err != nil {
					t.Fatalf("aLatch.Close: %v", err)
				}

				// 3b: OOM + failed drain -> no retry, sticky broken, no premature free, Close obeys drain contract.
				eDrain, aDrain := attachFresh(t)
				devDrain := aDrain.ops.dev
				heldBefore, _ := devDrain.LedgerLen()
				aDrain.defect.drainFails = 1
				allocs = 0
				devDrain.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 5 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				_, err = eDrain.EmbedIDsBatch(orig)
				devDrain.SetAllocHookForTest(nil)
				if err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("OOM+failed-drain must not match ErrBatchOutOfMemory: %v", err)
				}
				if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) {
					t.Fatalf("OOM+failed-drain lost OOM cause: %v", err)
				}
				kept, _ := devDrain.LedgerLen()
				if aDrain.broken == nil || kept <= heldBefore {
					t.Fatalf("OOM+failed-drain did not retain pending buffers: broken=%v ledger=%d before=%d", aDrain.broken, kept, heldBefore)
				}
				if _, err := eDrain.EmbedIDsBatch(orig[:1]); !errors.Is(err, aDrain.broken) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("post-broken batch = %v, want aDrain.broken without retry", err)
				}
				aDrain.defect.drainFails = 1
				if err := aDrain.Close(); err == nil {
					t.Fatal("Close succeeded despite planted drain failure")
				}
				if n, _ := devDrain.LedgerLen(); n != kept {
					t.Fatalf("failed Close altered ledger: %d vs %d", n, kept)
				}
				if err := aDrain.Close(); err != nil {
					t.Fatalf("Close retry after drain recovery: %v", err)
				}
				if allocsLeft, objsLeft := devDrain.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
					t.Fatalf("Close left ledger (%d, %d)", allocsLeft, objsLeft)
				}

				// 3c: OOM + failed rollback free -> no retry, sticky broken, ledger retains failed buffer.
				eFree, aFree := attachFresh(t)
				devFree := aFree.ops.dev
				heldBeforeFree, _ := devFree.LedgerLen()
				freeErr := errors.New("cuda: planted buffer close failure")
				closeCalls := 0
				devFree.SetCloseBufHookForTest(func(closeReal func() error) error {
					closeCalls++
					if closeCalls == 1 {
						return freeErr // leave real buffer open so Close->ReleaseAll frees it cleanly
					}
					return closeReal()
				})
				allocs = 0
				devFree.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 5 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				_, err = eFree.EmbedIDsBatch(orig)
				devFree.SetAllocHookForTest(nil)
				devFree.SetCloseBufHookForTest(nil)
				if err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("OOM+failed-free must not match ErrBatchOutOfMemory: %v", err)
				}
				if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || !errors.Is(err, freeErr) {
					t.Fatalf("OOM+failed-free missing cause: %v", err)
				}
				if aFree.broken == nil {
					t.Fatal("OOM+failed-free did not poison accelerator")
				}
				if n, _ := devFree.LedgerLen(); n != heldBeforeFree+1 {
					t.Fatalf("failed rollback free dropped ledger ownership: got %d, want %d", n, heldBeforeFree+1)
				}
				if _, err := eFree.EmbedIDsBatch(orig[:1]); !errors.Is(err, aFree.broken) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("subsequent call after failed free = %v", err)
				}
				if err := aFree.Close(); err != nil {
					t.Fatalf("aFree.Close: %v", err)
				}
				if allocsLeft, objsLeft := devFree.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
					t.Fatalf("aFree.Close left ledger (%d, %d)", allocsLeft, objsLeft)
				}
			})

			// (4) Singleton EmbedIDs OOM + attach partial-weight-upload OOM + unrelated attach error.
			t.Run("singleton-and-attach-oom", func(t *testing.T) {
				e, a := attachFresh(t)
				defer func() {
					if err := a.Close(); err != nil {
						t.Errorf("Close: %v", err)
					}
				}()
				dev := a.ops.dev
				allocs := 0
				dev.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 4 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				in := slices.Clone(orig[0])
				v, err := e.EmbedIDs(in)
				dev.SetAllocHookForTest(nil)
				if v != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || !slices.Equal(in, orig[0]) {
					t.Fatalf("singleton OOM = (%v, %v)", v, err)
				}
				retryV, err := e.EmbedIDs(in)
				if err != nil {
					t.Fatalf("singleton retry: %v", err)
				}
				if diff := eg2FirstDiff(retryV, baseSingle, m.Dim()); diff != "" {
					t.Fatalf("singleton retry differed at %s", diff)
				}

				// Attach OOM after several real weight uploads -> ErrDeviceOutOfMemory only, 0 leaks.
				var attachDev *Device
				attachAllocs := 0
				gotAcc, err := newEG2AccelWithDevice(m, combined, func() (*Device, error) {
					d, err := CreateSystemDefaultDevice()
					if err != nil {
						return nil, err
					}
					attachDev = d
					d.SetAllocHookForTest(func(what string, nBytes int) error {
						attachAllocs++
						if attachAllocs == 8 {
							return gc.ErrOutOfMemory
						}
						return nil
					})
					return d, nil
				})
				if gotAcc != nil || !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("attach partial-upload OOM = (%v, %v)", gotAcc, err)
				}
				if allocsLeft, objsLeft := attachDev.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
					t.Fatalf("attach OOM leaked ledger (%d, %d)", allocsLeft, objsLeft)
				}

				// Unrelated attach allocation error (gc.ErrInvalidValue) -> neither OOM sentinel, 0 leaks.
				attachAllocs = 0
				gotAcc, err = newEG2AccelWithDevice(m, combined, func() (*Device, error) {
					d, err := CreateSystemDefaultDevice()
					if err != nil {
						return nil, err
					}
					attachDev = d
					d.SetAllocHookForTest(func(what string, nBytes int) error {
						attachAllocs++
						if attachAllocs == 8 {
							return gc.ErrInvalidValue
						}
						return nil
					})
					return d, nil
				})
				if gotAcc != nil || err == nil || !errors.Is(err, gc.ErrInvalidValue) ||
					errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("unrelated attach failure = (%v, %v)", gotAcc, err)
				}
				if allocsLeft, objsLeft := attachDev.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
					t.Fatalf("unrelated attach failure leaked ledger (%d, %d)", allocsLeft, objsLeft)
				}
			})

			// (5) Non-OOM errors stay distinguishable from retry-safe OOM.
			t.Run("non-oom-distinctions", func(t *testing.T) {
				e, a := attachFresh(t)
				checkNotOOM := func(label string, err error) {
					t.Helper()
					if err == nil {
						t.Fatalf("%s: expected error", label)
					}
					if errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) {
						t.Fatalf("%s must not match OOM sentinels: %v", label, err)
					}
				}
				_, err := e.EmbedIDsBatch(nil)
				checkNotOOM("empty batch", err)
				_, err = e.EmbedIDsBatch([][]int{{1}, {}})
				checkNotOOM("empty member", err)
				_, err = e.EmbedIDsBatch([][]int{{1, 2}, {3, V}})
				checkNotOOM("invalid last ID", err)
				overMaxPos := make([]int, m.Config().MaxPositions+1)
				_, err = e.EmbedIDsBatch([][]int{overMaxPos})
				checkNotOOM("over MaxPositions", err)
				overAgg := make([][]int, a.batchRowLimit()+1)
				for i := range overAgg {
					overAgg[i] = []int{1}
				}
				_, err = e.EmbedIDsBatch(overAgg)
				checkNotOOM("over batchRowLimit", err)

				a.defect.failForward = true
				_, err = e.EmbedIDsBatch(orig[:1])
				a.defect.failForward = false
				checkNotOOM("ordinary execution failure", err)

				if err := a.Close(); err != nil {
					t.Fatalf("Close: %v", err)
				}
				_, err = e.EmbedIDsBatch(orig[:1])
				checkNotOOM("closed encoder", err)
			})
		})
	}
}
