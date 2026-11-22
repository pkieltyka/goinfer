//go:build cuda && realckpt

package cuda

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

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
