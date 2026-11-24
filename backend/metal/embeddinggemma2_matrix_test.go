//go:build darwin

package metal

import (
	"errors"
	"math"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

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
