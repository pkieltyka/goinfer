package vulkan

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

var ggufDir = flag.String("eg2-vulkan-gguf-dir", "", "directory containing the pinned EmbeddingGemma 2 GGUFs")

func TestVulkanGGUFPreparedIDs(t *testing.T) {
	if *ggufDir == "" {
		t.Skip("set -eg2-vulkan-gguf-dir to the pinned GGUF snapshots")
	}
	id := requiredDevice(t)
	for _, model := range []struct{ file, sha string }{
		{"embeddinggemma-2-Q8_0.gguf", "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"},
		{"embeddinggemma-2-UD-Q6_K_XL.gguf", "dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3"},
	} {
		t.Run(model.file, func(t *testing.T) {
			path := filepath.Join(*ggufDir, model.file)
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			hash := sha256.New()
			_, readErr := io.Copy(hash, f)
			closeErr := f.Close()
			if readErr != nil || closeErr != nil || fmt.Sprintf("%x", hash.Sum(nil)) != model.sha {
				t.Fatalf("checkpoint verification failed: read=%v close=%v sha=%x", readErr, closeErr, hash.Sum(nil))
			}
			e, err := embeddinggemma2.LoadGGUFEncoder(path, nil)
			if err != nil {
				t.Fatal(err)
			}

			query := []int{2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 1}
			doc := []int{2, 3250, 236787, 7293, 1109, 1816, 236787, 87215, 1}
			window := e.Model().Config().SlidingWindow
			ids := [][]int{query, doc}
			for _, length := range []int{window - 1, window, window + 1} {
				row := make([]int, length)
				for i := 1; i < len(row)-1; i++ {
					row[i] = query[1+i%(len(query)-2)]
				}
				row[0], row[len(row)-1] = 2, 1
				ids = append(ids, row)
			}
			want, err := e.EmbedIDsBatch(ids)
			if err != nil {
				t.Fatal(err)
			}

			acc, err := e.UseAccelerator("vulkan")
			if err != nil {
				t.Fatal(err)
			}
			a := acc.(*encoder)
			checkDevice(t, a, id)
			before := a.device.Activity()
			got, err := e.EmbedIDsBatch(ids)
			if err != nil {
				t.Fatal(err)
			}
			rows := 0
			for _, input := range ids {
				rows += len(input)
			}
			checkedActivity(t, a, before, rows, false)
			if len(got) != len(want) {
				t.Fatalf("batch returned %d vectors, want %d", len(got), len(want))
			}
			for i, vector := range got {
				single, err := e.EmbedIDs(ids[i])
				if err != nil {
					t.Fatal(err)
				}
				label := fmt.Sprintf("%d-token input", len(ids[i]))
				checkRows(t, label+" CPU parity", vector, want[i], e.HiddenDim())
				checkRows(t, label+" batch isolation", vector, single, e.HiddenDim())
			}
		})
	}
}
