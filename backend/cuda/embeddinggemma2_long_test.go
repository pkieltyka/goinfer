//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

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
