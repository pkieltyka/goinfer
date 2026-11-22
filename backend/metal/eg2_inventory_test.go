//go:build darwin

package metal

import (
	"errors"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

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
