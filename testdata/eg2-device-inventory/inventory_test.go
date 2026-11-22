//go:build linux && amd64 && cuda

package inventory_test

import (
	"errors"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	_ "github.com/pkieltyka/goinfer/backend/cuda"
	_ "github.com/pkieltyka/goinfer/backend/vulkan"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/tokenizer"
)

// TestMixedDeviceInventory links both native backends in one cgo-free binary.
// Run explicitly through the workspace; testdata is excluded from ./... .
func TestMixedDeviceInventory(t *testing.T) {
	cudaUUID, vulkanProduct := os.Getenv("GOINFER_TEST_CUDA_UUID"), os.Getenv("GOINFER_TEST_VULKAN_DEVICE")
	if cudaUUID == "" || vulkanProduct == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID and GOINFER_TEST_VULKAN_DEVICE to the GPU UUID and Radeon PCI product ID")
	}
	product, err := strconv.ParseUint(vulkanProduct, 16, 32)
	if err != nil {
		t.Fatal(err)
	}
	list := func(opts embeddinggemma2.ProbeOptions) []embeddinggemma2.DeviceInfo {
		t.Helper()
		rows, failures := embeddinggemma2.Devices(opts)
		if len(failures) != 0 {
			t.Fatalf("inventory: %+v", failures)
		}
		return rows
	}
	cards := map[string]embeddinggemma2.DeviceInfo{}
	for _, row := range list(embeddinggemma2.ProbeOptions{}) {
		switch {
		case row.Backend == "cuda" && strings.EqualFold(row.ID, "cuda:"+cudaUUID) && row.VendorID == 0x10de,
			row.Backend == "vulkan" && row.VendorID == 0x1002 && row.ProductID == uint32(product):
			if cards[row.Backend].ID != "" {
				t.Fatalf("several %s rows matched the requested device", row.Backend)
			}
			cards[row.Backend] = row
		}
	}
	for _, backend := range []string{"cuda", "vulkan"} {
		row := cards[backend]
		if row.ID == "" || row.Alias == "" || !row.TextBatching || row.MemoryKind != embeddinggemma2.MemoryDedicated ||
			row.Memory.Basis != embeddinggemma2.MemoryUnknown || row.Memory.AvailableBytes != 0 || row.Memory.TotalBytes <= 0 {
			t.Fatalf("%s capacity-only inventory: %+v", backend, row)
		}
		t.Logf("inventory: %+v, linked selectors %v", row, embeddinggemma2.DeviceSelectors(backend))
	}
	if cards["cuda"].ID == cards["vulkan"].ID {
		t.Fatal("CUDA and Vulkan device IDs collide")
	}

	seen := map[string]bool{}
	for _, row := range list(embeddinggemma2.ProbeOptions{IncludeMemory: true}) {
		if row.ID != cards[row.Backend].ID || row.ID == "" {
			continue
		}
		seen[row.Backend] = true
		checkMemory(t, row.Backend, row.Memory)
		t.Logf("%s inventory memory: %+v", row.ID, row.Memory)
	}
	if !seen["cuda"] || !seen["vulkan"] {
		t.Fatalf("memory inventory lost selected devices: %v", seen)
	}

	m, err := embeddinggemma2.Load("../embeddinggemma2-tiny")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadJSONBytes([]byte(`{"model":{"type":"BPE","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3},"merges":[]}}`))
	if err != nil {
		t.Fatal(err)
	}
	e, err := embeddinggemma2.NewEncoder(m, tok, nil)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{2, 10, 35, 17, 52, 18, 1}
	want, err := e.EmbedIDs(ids)
	if err != nil {
		t.Fatal(err)
	}
	if acc, err := e.UseDevice(cards["vulkan"].ID, "cuda"); acc != nil || !errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) ||
		errors.Is(err, embeddinggemma2.ErrBackendUnavailable) || e.Accelerator() != nil {
		t.Fatalf("cross-backend selector: %v, %v", acc, err)
	}
	for _, backend := range []string{"cuda", "vulkan"} {
		row := cards[backend]
		selectors := embeddinggemma2.DeviceSelectors(backend)
		if !slices.Contains(selectors, backend) {
			t.Fatalf("%s default absent from %v", backend, selectors)
		}
		for _, selector := range selectors {
			if !t.Run(selector, func(t *testing.T) {
				acc, err := e.UseDevice(row.ID, selector)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := acc.Close(); err != nil {
						t.Error(err)
					}
				})
				report, ok := acc.(embeddinggemma2.AttachedDeviceReporter)
				if !ok {
					t.Fatalf("%s cannot report attachment", selector)
				}
				attached := report.AttachedDevice()
				if attached.ID != row.ID || attached.Backend != backend || attached.VendorID != row.VendorID ||
					attached.ProductID != row.ProductID || acc.Name() != backend {
					t.Fatalf("attached %+v, accelerator %q; want %+v", attached, acc.Name(), row)
				}
				memory, ok := acc.(embeddinggemma2.DeviceMemoryReporter)
				if !ok {
					t.Fatalf("%s cannot report attached memory", selector)
				}
				reading, err := memory.DeviceMemory()
				if err != nil {
					t.Fatal(err)
				}
				checkMemory(t, backend, reading)
				got, err := e.EmbedIDs(ids)
				if err != nil {
					t.Fatal(err)
				}
				cos := cosine(t, got, want)
				if math.IsNaN(cos) || cos < 0.9999 {
					t.Fatalf("CPU cosine %.12f for %s", cos, selector)
				}
				t.Logf("attached %s (%s), memory %+v, CPU cosine %.12f", attached.ID, attached.Name, reading, cos)
			}) {
				return
			}
		}
	}
}

func checkMemory(t *testing.T, backend string, got embeddinggemma2.DeviceMemory) {
	t.Helper()
	want := embeddinggemma2.MemoryLive
	if backend == "vulkan" {
		want = embeddinggemma2.MemoryEstimated
	}
	if got.Basis != want || got.TotalBytes <= 0 || got.AvailableBytes < 0 || got.AvailableBytes > got.TotalBytes {
		t.Fatalf("%s memory %+v, want %s basis with valid capacity", backend, got, want)
	}
}

func cosine(t *testing.T, a, b []float32) float64 {
	t.Helper()
	if len(a) != len(b) || len(a) == 0 {
		t.Fatalf("vector size %d, want %d", len(a), len(b))
	}
	var dot, an, bn float64
	for i, v := range a {
		x, y := float64(v), float64(b[i])
		dot, an, bn = dot+x*y, an+x*x, bn+y*y
	}
	return dot / math.Sqrt(an*bn)
}
