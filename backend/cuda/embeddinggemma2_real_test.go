//go:build cuda

package cuda

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

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
