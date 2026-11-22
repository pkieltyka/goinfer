package vulkan

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestVulkanBatchLayoutBytes(t *testing.T) {
	id := requiredDevice(t)
	for _, checkpoint := range []struct {
		name string
		load func(*testing.T) *embeddinggemma2.Encoder
	}{
		{"official", func(t *testing.T) *embeddinggemma2.Encoder {
			dir := os.Getenv("GOINFER_EG2_DIR")
			if dir == "" {
				t.Skip("set GOINFER_EG2_DIR to the pinned local checkpoint")
			}
			return loadPinnedOfficial(t, dir)
		}},
		{"q8", func(t *testing.T) *embeddinggemma2.Encoder {
			return loadPinnedGGUF(t, pinnedGGUFs[0])
		}},
		{"q6", func(t *testing.T) *embeddinggemma2.Encoder {
			return loadPinnedGGUF(t, pinnedGGUFs[1])
		}},
	} {
		if !t.Run(checkpoint.name, func(t *testing.T) {
			e := checkpoint.load(t)
			a := attachParityDevice(t, e, id)
			checkBatchLayout(t, e, a)
		}) {
			return
		}
	}
}

func attachParityDevice(t *testing.T, e *embeddinggemma2.Encoder, id uint32) *encoder {
	t.Helper()
	devices, err := Devices()
	if err != nil {
		t.Fatal(err)
	}
	var selectors []string
	for _, device := range devices {
		if device.DeviceID == id && device.ID != "" {
			selectors = append(selectors, device.ID)
		}
	}
	if len(selectors) != 1 {
		t.Fatalf("expected exactly one stable Vulkan identity for PCI product %04x, got %v", id, selectors)
	}
	acc, err := e.UseAccelerator(selectors[0])
	if err != nil {
		t.Fatal(err)
	}
	a := acc.(*encoder)
	checkDevice(t, a, id)
	if device, ok := AttachedDevice(acc); !ok || device.ID != selectors[0] {
		t.Fatalf("requested %s, attached %+v", selectors[0], device)
	}
	t.Logf("selector %s", selectors[0])
	return a
}

func checkBatchLayout(t *testing.T, e *embeddinggemma2.Encoder, a *encoder) {
	t.Helper()
	query := []int{2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 1217, 1440, 740, 2536, 577, 1680, 901, 974, 3463, 1135, 1}
	doc := []int{2, 3250, 236787, 7293, 1109, 1816, 236787, 87215, 1}
	inputs := [][]int{query, doc}
	for _, n := range []int{127, 128, e.Model().Config().SlidingWindow + 2} {
		row := make([]int, n)
		for i := 1; i < n-1; i++ {
			row[i] = query[1+(i+n)%(len(query)-2)]
		}
		row[0], row[n-1] = 2, 1
		inputs = append(inputs, row)
	}
	original := make([][]int, len(inputs))
	want := make([][]float32, len(inputs))
	for i, ids := range inputs {
		original[i] = slices.Clone(ids)
		before := a.device.Activity()
		vector, err := e.EmbedIDs(ids)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, original[i]) {
			t.Fatalf("singleton %d changed its prepared input", i)
		}
		checkedActivity(t, a, before, []int{len(ids)}, false)
		if len(vector) != e.Model().Dim() {
			t.Fatalf("singleton %d returned %d components", i, len(vector))
		}
		var norm float64
		for _, value := range vector {
			norm += float64(value) * float64(value)
		}
		if !(math.Abs(norm-1) < 1e-6) {
			t.Fatalf("singleton %d squared norm %g", i, norm)
		}
		t.Logf("singleton %d: %d tokens, pooled SHA-256 %x", i, len(ids), pooledDigest([][]float32{vector}))
		want[i] = vector
	}
	for _, layout := range []struct {
		name  string
		order []int
	}{
		{"two-short", []int{0, 1}},
		{"across-block", []int{2, 1}},
		{"all", []int{0, 1, 2, 3, 4}},
		{"reversed", []int{4, 3, 2, 1, 0}},
		{"first-half", []int{0, 1, 2}},
		{"second-half", []int{3, 4}},
	} {
		if !t.Run(layout.name, func(t *testing.T) {
			batch := make([][]int, len(layout.order))
			lengths := make([]int, len(batch))
			for i, j := range layout.order {
				batch[i] = inputs[j]
				lengths[i] = len(batch[i])
			}
			before := a.device.Activity()
			got, err := e.EmbedIDsBatch(batch)
			if err != nil {
				t.Fatal(err)
			}
			checkedActivity(t, a, before, lengths, false)
			if len(got) != len(batch) {
				t.Fatalf("got %d vectors, want %d", len(got), len(batch))
			}
			for i, j := range layout.order {
				if !slices.Equal(batch[i], original[j]) {
					t.Fatalf("changed prepared input at output %d (singleton %d)", i, j)
				}
				checkPooledBits(t, fmt.Sprintf("output %d (singleton %d, %d tokens)", i, j, len(batch[i])), got[i], want[j])
			}
		}) {
			return
		}
	}
	for _, offset := range []int{1, 63, 64, 65} {
		if !t.Run(fmt.Sprintf("offset-%d", offset), func(t *testing.T) {
			prefix := slices.Clone(inputs[2][:offset])
			originalPrefix := slices.Clone(prefix)
			before := a.device.Activity()
			got, err := e.EmbedIDsBatch([][]int{prefix, doc})
			if err != nil {
				t.Fatal(err)
			}
			checkedActivity(t, a, before, []int{offset, len(doc)}, false)
			if len(got) != 2 || !slices.Equal(prefix, originalPrefix) || !slices.Equal(doc, original[1]) {
				t.Fatal("packed call changed the prepared inputs or output count")
			}
			checkPooledBits(t, "shifted document", got[1], want[1])
		}) {
			return
		}
	}
}

func pooledDigest(vectors [][]float32) [32]byte {
	var payload []byte
	for _, vector := range vectors {
		for _, value := range vector {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(value))
		}
	}
	return sha256.Sum256(payload)
}

func checkPooledBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("%s: got %d components, want %d", label, len(got), len(want))
	}
	for i, v := range got {
		if g, w := math.Float32bits(v), math.Float32bits(want[i]); g != w {
			t.Errorf("%s component %d: bits %08x, want %08x", label, i, g, w)
			break
		}
	}
}

func verifyPinnedCheckpoint(t *testing.T, path, want string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.New()
	_, readErr := io.Copy(hash, f)
	closeErr := f.Close()
	if readErr != nil || closeErr != nil || fmt.Sprintf("%x", hash.Sum(nil)) != want {
		t.Fatalf("cannot verify %s: read=%v close=%v SHA-256=%x, want %s", filepath.Base(path), readErr, closeErr, hash.Sum(nil), want)
	}
	t.Logf("%s SHA-256 %s", filepath.Base(path), want)
}
