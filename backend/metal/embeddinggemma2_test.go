//go:build darwin

package metal

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"

	gpu "github.com/pkieltyka/goinfer/backend/metal/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

const eg2TinyDir = "../../testdata/embeddinggemma2-tiny"

func eg2Cos(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func eg2Pool(h []float32, T, d int) []float32 {
	out := make([]float32, d)
	for t := range T {
		for j := range d {
			out[j] += h[t*d+j] / float32(T)
		}
	}
	return out
}

// TestEG2Metal_matchesCPU is Phase M's Gate 1: on the tiny fixture, the Metal
// accelerator's forward matches the CPU forward (itself gated against transformers, Gate 1 of the CPU path) to cosine
// 0.9999 on every layer's input, on the projected last hidden state and on the pooled embedding. Cases: the golden's
// four (1, 2, 9 and 17 tokens; sliding radius 3 and full layers of different head dims and KV-head counts) and two
// random inputs of 300 and 700 tokens, each at attention blocks of 256 (the default), 4 and 5 query rows, so blocks
// cut through the radius-3 window and leave partial last blocks. The tiny fixture's head dims are 16 and 32; the real
// checkpoint's 256 and 512 are Gate 2's.
func TestEG2Metal_matchesCPU(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatalf("load the tiny fixture: %v", err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	defer acc.Close()
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
	var inputs [][]int
	for _, c := range g.Cases {
		inputs = append(inputs, c.IDs)
	}
	for _, n := range []int{300, 700} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7919 + 13) % m.Config().VocabSize
		}
		inputs = append(inputs, ids)
	}
	E := m.Dim()
	defer func(b int) { eg2Block = b }(eg2Block)
	for _, blk := range []int{256, 4, 5} {
		eg2Block = blk
		for ci, ids := range inputs {
			cpuLast, cpuLayers, err := m.ForwardCPU(ids, true)
			if err != nil {
				t.Fatal(err)
			}
			gpuLast, gpuLayers, err := acc.Forward(ids, true)
			if err != nil {
				t.Fatalf("case %d: metal forward: %v", ci, err)
			}
			if len(gpuLayers) != len(cpuLayers) {
				t.Fatalf("case %d: %d layer states, CPU %d", ci, len(gpuLayers), len(cpuLayers))
			}
			worst, worstAt := 1.0, -1
			for li := range cpuLayers {
				if c := eg2Cos(gpuLayers[li], cpuLayers[li]); c < worst {
					worst, worstAt = c, li
				}
			}
			lc := eg2Cos(gpuLast, cpuLast)
			ec := eg2Cos(eg2Pool(gpuLast, len(ids), E), eg2Pool(cpuLast, len(ids), E))
			t.Logf("block %d, case %d (%d tokens): worst layer-input cosine %.9f (index %d), last hidden %.9f, embedding %.9f", blk, ci, len(ids), worst, worstAt, lc, ec)
			if worst < 0.9999 || lc < 0.9999 || ec < 0.9999 {
				t.Errorf("block %d, case %d (%d tokens): worst layer %.9f at index %d, last hidden %.9f, embedding %.9f (bar 0.9999)", blk, ci, len(ids), worst, worstAt, lc, ec)
			}
		}
	}
}

func TestMetalInventoryMemory(t *testing.T) {
	oldRAM, oldCeiling := physicalRAM, metalMemoryCeilingForTest
	t.Cleanup(func() { physicalRAM, metalMemoryCeilingForTest = oldRAM, oldCeiling })
	physicalRAM = func() (uint64, error) { return 4096, nil }
	metalMemoryCeilingForTest = 1024
	for _, tc := range []struct {
		unified, include bool
		kind             embeddinggemma2.MemoryKind
		memory           embeddinggemma2.DeviceMemory
	}{
		{true, false, embeddinggemma2.MemoryShared, embeddinggemma2.DeviceMemory{TotalBytes: 4096}},
		{true, true, embeddinggemma2.MemoryShared, embeddinggemma2.DeviceMemory{TotalBytes: 4096, AvailableBytes: 1024, Basis: embeddinggemma2.MemoryEstimated}},
		{false, true, embeddinggemma2.MemoryDedicated, embeddinggemma2.DeviceMemory{}},
	} {
		got := describeMetalDevice("GPU", tc.unified, tc.include)
		if got.Backend != "metal" || got.Name != "GPU" || got.ID != "" || got.Alias != "" ||
			got.TextBatching || got.MemoryKind != tc.kind || got.Memory != tc.memory {
			t.Errorf("unified %t include %t: %+v", tc.unified, tc.include, got)
		}
	}
	physicalRAM = func() (uint64, error) { return 0, errors.New("unreadable") }
	if got := describeMetalDevice("GPU", true, true).Memory; got != (embeddinggemma2.DeviceMemory{}) {
		t.Fatalf("unavailable physical RAM should be unknown: %+v", got)
	}
}

func TestMetalInventoryAndAttachAcquisitionErrors(t *testing.T) {
	for _, tt := range []struct{ native, category error }{
		{gpu.ErrNoDevice, embeddinggemma2.ErrNoDevice},
		{errors.New("framework missing"), embeddinggemma2.ErrBackendUnavailable},
	} {
		open := func() (*Device, error) { return nil, tt.native }
		ops, err := newEG2OpsOn(open, "", nil)
		if ops.d != nil || !errors.Is(err, tt.category) || !errors.Is(err, tt.native) {
			t.Fatalf("attach %v: %+v, %v", tt.native, ops.d, err)
		}
		rows, err := probeMetalDeviceOn(open, embeddinggemma2.ProbeOptions{})
		if len(rows) != 0 ||
			tt.category == embeddinggemma2.ErrNoDevice && err != nil ||
			tt.category == embeddinggemma2.ErrBackendUnavailable && (!errors.Is(err, tt.category) || !errors.Is(err, tt.native)) {
			t.Fatalf("inventory %v: rows %+v, error %v", tt.native, rows, err)
		}
	}
}

// #84 scalar-arena sizing and growth regressions.

func eg2GoogleKVHeads() []int {
	// Google geometry: every sixth layer is full attention with one KV head,
	// the other twenty are sliding with two.
	out := make([]int, 24)
	for i := range out {
		out[i] = 2
		if (i+1)%6 == 0 {
			out[i] = 1
		}
	}
	return out
}

func TestEG2ArenaSlotsGoogle(t *testing.T) {
	cfg := embeddinggemma2.Config{Heads: 4, Layers: 24}
	kv := eg2GoogleKVHeads()
	for _, tc := range []struct{ tokens, want int }{
		{1536, 14512}, {1537, 16556}, {2048, 18600}, {8192, 67656},
	} {
		got, err := eg2ArenaSlotsFor(cfg, kv, tc.tokens, false)
		if err != nil {
			t.Fatalf("%d tokens: %v", tc.tokens, err)
		}
		if got != tc.want {
			t.Fatalf("%d tokens: %d slots, want %d", tc.tokens, got, tc.want)
		}
	}
	// The initial arena must not already cover the observed 1,537 boundary, so
	// the growth path is the one exercised there.
	slots, err := eg2ArenaSlotsFor(cfg, kv, 1537, false)
	if err != nil {
		t.Fatal(err)
	}
	if slots <= eg2ArenaSlots {
		t.Fatalf("1,537 needs %d slots, initial arena already holds %d", slots, eg2ArenaSlots)
	}
}

func TestEG2ArenaSlotsKeepLayers(t *testing.T) {
	cfg := embeddinggemma2.Config{Heads: 4, Layers: 24}
	kv := eg2GoogleKVHeads()
	got, err := eg2ArenaSlotsFor(cfg, kv, 1536, true)
	if err != nil {
		t.Fatal(err)
	}
	// Largest interval is the opening plus the first (sliding, two-KV-head)
	// layer: 6 blocks of (19*4 + 5*2) plus 8 + 93.
	if want := 8 + 93 + 6*(19*4+5*2); got != want {
		t.Fatalf("keepLayers interval %d slots, want %d", got, want)
	}
	if got >= 14512 {
		t.Fatalf("keepLayers interval %d should be far below the whole forward", got)
	}
}

func TestEG2ArenaSlotsInvalid(t *testing.T) {
	cfg := embeddinggemma2.Config{Heads: 4, Layers: 24}
	kv := eg2GoogleKVHeads()
	if _, err := eg2ArenaSlotsFor(cfg, kv, 0, false); err == nil {
		t.Fatal("zero rows accepted")
	}
	if _, err := eg2ArenaSlotsFor(cfg, kv, -1, false); err == nil {
		t.Fatal("negative rows accepted")
	}
	if _, err := eg2ArenaSlotsFor(cfg, kv[:23], 128, false); err == nil {
		t.Fatal("short KV slice accepted")
	}
	bad := append([]int(nil), kv...)
	bad[3] = 0
	if _, err := eg2ArenaSlotsFor(cfg, bad, 128, false); err == nil {
		t.Fatal("zero KV heads accepted")
	}
	if _, err := eg2ArenaSlotsFor(cfg, kv, math.MaxInt/2, false); err == nil {
		t.Fatal("addressable-range overflow accepted")
	}
	old := eg2Block
	eg2Block = 0
	defer func() { eg2Block = old }()
	if _, err := eg2ArenaSlotsFor(cfg, kv, 128, false); err == nil {
		t.Fatal("zero block accepted")
	}
}

// TestEG2Metal_growsScalarArena is the red-first caller regression: with a
// one-slot arena the old fixed-capacity guard panicked; the sized path grows
// before encoding and reproduces the normal run's bytes.
func TestEG2Metal_growsScalarArena(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	ids := make([]int, 300)
	for i := range ids {
		ids[i] = (i*7919 + 13) % m.Config().VocabSize
	}
	want, _, err := a.Forward(ids, false)
	if err != nil {
		t.Fatal(err)
	}
	short := make([]int, 17)
	for i := range short {
		short[i] = (i*104729 + 7) % m.Config().VocabSize
	}
	shortWant, _, err := a.Forward(short, false)
	if err != nil {
		t.Fatal(err)
	}
	a.d.ReleaseBuf(a.arena)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	got, _, err := a.Forward(ids, false)
	if err != nil {
		t.Fatalf("grown forward: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("%d values, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("value %d changed after growing the arena", i)
		}
	}
	if a.arena.Len() <= 256 {
		t.Fatalf("arena did not grow: %d bytes", a.arena.Len())
	}
	shortGot, _, err := a.Forward(short, false)
	if err != nil {
		t.Fatalf("short forward after growth: %v", err)
	}
	for i := range shortWant {
		if math.Float32bits(shortGot[i]) != math.Float32bits(shortWant[i]) {
			t.Fatalf("short output %d changed after growth", i)
		}
	}
}

// TestEG2Metal_arenaGrowRefusalKeepsArena proves a refused growth returns an
// error before encoding, keeps the old arena, and a later retry succeeds with
// the ledger back to its starting length.
// TestEG2Metal_longForwardAfterClose is the caller regression for the closed
// path: a long ForwardEmbeds whose arena requirement exceeds the retained one
// must return errTowerClosed, not dereference a released device.
func TestEG2Metal_longForwardAfterClose(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	a := acc.(*eg2Accel)
	// Shrink the retained arena so the long requirement exceeds it after Close.
	a.d.ReleaseBuf(a.arena)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	if err := acc.Close(); err != nil {
		t.Fatal(err)
	}
	H := m.Config().Hidden
	x := make([]float32, 300*H)
	for i := range x {
		x[i] = float32(i%17) / 17
	}
	if _, _, err := a.ForwardEmbeds(x, 300, false); !errors.Is(err, errTowerClosed) {
		t.Fatalf("long ForwardEmbeds after Close: %v, want errTowerClosed", err)
	}
}

func TestEG2Metal_arenaGrowRefusalKeepsArena(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	ids := make([]int, 300)
	for i := range ids {
		ids[i] = (i*7919 + 13) % m.Config().VocabSize
	}
	if _, _, err := a.Forward(ids, false); err != nil { // warm rope and scratch
		t.Fatal(err)
	}
	a.d.ReleaseBuf(a.arena)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	old := a.arena
	allocs, _ := a.d.LedgerLen()
	eg2ArenaAllocForTest = func(int) (Buffer, bool) { return Buffer{}, false }
	_, _, err = a.Forward(ids, false)
	eg2ArenaAllocForTest = nil
	if err == nil {
		t.Fatal("refused growth did not fail the forward")
	}
	if a.arena.Len() != old.Len() {
		t.Fatalf("refusal replaced the arena: %d bytes, was %d", a.arena.Len(), old.Len())
	}
	if _, _, err := a.Forward(ids, false); err != nil {
		t.Fatalf("retry after a refused growth: %v", err)
	}
	if a.arena.Len() <= old.Len() {
		t.Fatal("retry did not grow the arena")
	}
	if got, _ := a.d.LedgerLen(); got != allocs {
		t.Fatalf("ledger length %d after grow, want %d", got, allocs)
	}
}

func TestEG2AttentionResolution(t *testing.T) {
	hard := errors.New("command failed")
	for _, tc := range []struct {
		name    string
		request eg2AttentionRequest
		shape   bool
		setup   error
		calls   int
		profile string
		err     error
	}{
		{"default eligible", eg2AttentionDefault, true, nil, 1, eg2MatrixProfile, nil},
		{"default geometry decline", eg2AttentionDefault, false, nil, 0, eg2StockProfile, nil},
		{"default capability decline", eg2AttentionDefault, true, embeddinggemma2.ErrUnsupportedSelector, 1, eg2StockProfile, nil},
		{"default native failure", eg2AttentionDefault, true, hard, 1, "", hard},
		{"default OOM", eg2AttentionDefault, true, embeddinggemma2.ErrDeviceOutOfMemory, 1, "", embeddinggemma2.ErrDeviceOutOfMemory},
		{"forced matrix decline", eg2AttentionMatrix, true, embeddinggemma2.ErrUnsupportedSelector, 1, "", embeddinggemma2.ErrUnsupportedSelector},
		{"forced matrix geometry decline", eg2AttentionMatrix, false, nil, 0, "", embeddinggemma2.ErrUnsupportedSelector},
		{"forced matrix eligible", eg2AttentionMatrix, true, nil, 1, eg2MatrixProfile, nil},
		{"forced stock bypasses setup", eg2AttentionStock, true, hard, 0, eg2StockProfile, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			got, err := resolveEG2Attention(tc.request, tc.shape, func() error {
				calls++
				return tc.setup
			})
			if got != tc.profile || !errors.Is(err, tc.err) || calls != tc.calls {
				t.Fatalf("profile=%q err=%v setup calls=%d", got, err, calls)
			}
		})
	}
}

func TestEG2MatrixGeometry(t *testing.T) {
	w := embeddinggemma2.Weights{
		Config: embeddinggemma2.Config{Heads: 4, Layers: 24, SlidingWindow: 512},
		Layers: make([]embeddinggemma2.LayerWeights, 24),
	}
	for i := range w.Layers {
		w.Layers[i] = embeddinggemma2.LayerWeights{HeadDim: 256, KVHeads: 2}
		if i%6 == 5 {
			w.Layers[i] = embeddinggemma2.LayerWeights{Full: true, HeadDim: 512, KVHeads: 1}
		}
	}
	if !eg2MatrixGeometry(w) {
		t.Fatal("eligible graph declined")
	}
	for _, change := range []func(*embeddinggemma2.Weights){
		func(w *embeddinggemma2.Weights) { w.Config.Heads = 2 },
		func(w *embeddinggemma2.Weights) { w.Config.SlidingWindow = 511 },
		func(w *embeddinggemma2.Weights) { w.Layers = w.Layers[:23] },
		func(w *embeddinggemma2.Weights) { w.Layers[5].Full = false },
		func(w *embeddinggemma2.Weights) { w.Layers[11].HeadDim = 256 },
		func(w *embeddinggemma2.Weights) { w.Layers[0].KVHeads = 1 },
	} {
		copy := w
		copy.Layers = append([]embeddinggemma2.LayerWeights(nil), w.Layers...)
		change(&copy)
		if eg2MatrixGeometry(copy) {
			t.Fatal("unsupported graph accepted")
		}
	}
}

func TestEG2MatrixLengthAdmission(t *testing.T) {
	for _, rawCap := range []int{0, 4096, 262144} {
		limit := embeddinggemma2.MaxTokens
		if rawCap > 0 {
			limit = min(limit, rawCap)
		}
		// Only the rejection paths below use Forward. A zero Device lets
		// the guard run while making any unintended device work fail.
		a := &eg2Accel{
			eg2Ops:  eg2Ops{d: &Device{}},
			cfg:     embeddinggemma2.Config{Hidden: 1, MaxPositions: rawCap},
			profile: eg2MatrixProfile,
		}
		est, err := a.BatchMemory([]int{limit})
		if err != nil || est.Quality != embeddinggemma2.BatchMemoryUnknown {
			t.Fatalf("max %d memory admission: %+v, %v", limit, est, err)
		}
		_, err = a.BatchMemory([]int{limit + 1})
		_, _, idErr := a.Forward(make([]int, limit+1), false)
		_, _, rowErr := a.ForwardEmbeds(make([]float32, limit+1), limit+1, false)
		for _, err := range []error{err, idErr, rowErr} {
			if !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
				t.Fatalf("max+1 at %d: %v", limit, err)
			}
		}
		if _, _, err := a.ForwardEmbeds(nil, limit+1, false); err == nil || errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
			t.Fatalf("malformed prepared input: %v", err)
		}
	}
}

func TestEG2MatrixCanaryOracle(t *testing.T) {
	for _, dims := range [][2]int{{512, 1}, {256, 2}} {
		hd, kv := dims[0], dims[1]
		rows := eg2ProbeRows + eg2ProbeTail
		q, k, v := make([]float32, rows*4*hd), make([]float32, rows*kv*hd), make([]float32, rows*kv*hd)
		eg2ProbeInputs(q, k, v, hd, kv)
		out := make([]float32, len(q))
		for i := eg2ProbeRows * 4 * hd; i < len(out); i++ {
			out[i] = eg2ProbeMark
		}
		// Direct FP64 Q·K, softmax and P·V, independent of the tagged-logit shortcut.
		for row := range eg2ProbeRows {
			for h := range 4 {
				group := h / (4 / kv)
				var sum float64
				var p [eg2ProbeRows]float64
				for key := range eg2ProbeRows {
					var dot float64
					for c := range hd {
						dot += float64(q[(row*4+h)*hd+c]) * float64(k[(key*kv+group)*hd+c])
					}
					p[key] = math.Exp(dot)
					sum += p[key]
				}
				for c := range hd {
					var x float64
					for key := range eg2ProbeRows {
						x += p[key] * float64(v[(key*kv+group)*hd+c])
					}
					out[(row*4+h)*hd+c] = float32(x / sum)
				}
			}
		}
		if err := eg2ProbeOutput(out, v, hd, kv); err != nil {
			t.Fatal(err)
		}
		out[0], out[1] = out[1], out[0]
		if err := eg2ProbeOutput(out, v, hd, kv); !errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) {
			t.Fatalf("permuted words: %v", err)
		}
		out[len(out)-1] = 0
		if err := eg2ProbeOutput(out, v, hd, kv); err == nil || errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) {
			t.Fatalf("guard write with a simultaneous numerical mismatch must fail hard: %v", err)
		}
	}
}

func TestEG2MetalMatrixNativeCanary(t *testing.T) {
	ops, err := newEG2Ops("", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.closeDevice()
	if !ops.d.HasUnifiedMemory() || !ops.d.SupportsApple7() {
		t.Skip("Apple7+ with unified memory required")
	}
	beforeBuffers, beforeObjects := ops.d.LedgerLen()
	_, err = ops.newEG2MatrixPipes()
	if err != nil {
		t.Fatal(err)
	}
	if buffers, objects := ops.d.LedgerLen(); buffers != beforeBuffers || objects != beforeObjects+2 || ops.slot != 0 {
		t.Fatalf("temporary canary resources retained: buffers %d/%d objects %d/%d scalar slot %d", buffers, beforeBuffers, objects, beforeObjects+2, ops.slot)
	}
}

func TestEG2MetalStockSelector(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, selector := range []string{"metal", "metal-eg2-stock"} {
		a, err := m.NewAccelerator(selector)
		if err != nil {
			t.Fatal(err)
		}
		defer a.Close()
		if reporter, ok := a.(embeddinggemma2.AttachedProfileReporter); !ok || reporter.AttachedProfile() != eg2StockProfile || a.Name() != "metal" {
			t.Fatalf("%s: %s has no resolved stock profile", selector, a.Name())
		}
	}
	if a, err := m.NewAccelerator("metal-eg2-combined"); !errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) {
		if a != nil {
			a.Close()
		}
		t.Fatalf("strict matrix selector accepted tiny geometry: %v", err)
	}
}

func TestEG2StagedFFNEligibility(t *testing.T) {
	c := embeddinggemma2.Config{Hidden: 512, Intermediate: 2048, Layers: 24}
	if !eg2StagedFFNEligible("Apple M4", "applegpu_g16g", eg2MatrixProfile, c) {
		t.Fatal("measured device, graph and profile rejected")
	}
	for _, test := range []struct {
		name, device, arch, profile string
		cfg                         embeddinggemma2.Config
	}{
		{"unknown arch", "Apple M4", "", eg2MatrixProfile, c},
		{"other arch", "Apple M4", "applegpu_g15g", eg2MatrixProfile, c},
		{"other model", "Apple M4 Pro", "applegpu_g16g", eg2MatrixProfile, c},
		{"stock", "Apple M4", "applegpu_g16g", eg2StockProfile, c},
		{"other graph", "Apple M4", "applegpu_g16g", eg2MatrixProfile, embeddinggemma2.Config{Hidden: 512, Intermediate: 4096, Layers: 24}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if eg2StagedFFNEligible(test.device, test.arch, test.profile, test.cfg) {
				t.Fatal("unqualified graph or device admitted")
			}
		})
	}
	for _, tt := range []struct {
		rows int
		want bool
	}{{1, false}, {30, false}, {31, true}, {36, true}, {879, true}, {2042, true}, {2043, false}, {8192, false}} {
		if got := eg2StagedFFNRows(tt.rows); got != tt.want {
			t.Errorf("rows %d: staged=%t, want %t", tt.rows, got, tt.want)
		}
	}
}
