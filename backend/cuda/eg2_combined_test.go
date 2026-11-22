//go:build cuda && eg2_combined

package cuda

import (
	"fmt"
	"os"
	"testing"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestEG2CombinedTiny(t *testing.T) {
	checkCombined(t, eg2TinyModel(t), [][]int{{36}, {37}, {12, 24}, {12, 25}, {127, 10}, {64, 64}, {5, 128}, {129, 1}})
}

func TestEG2CombinedReal(t *testing.T) {
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
	a := newEG2ForTest(t, model)
	if err := a.enableCombined(); err != nil {
		t.Fatal(err)
	}
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
