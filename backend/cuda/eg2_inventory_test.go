//go:build cuda

package cuda

import (
	"errors"
	"slices"
	"testing"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestCUDAInventoryRows(t *testing.T) {
	adapters := []gpu.Adapter{
		{Ordinal: 0, UUID: "Old", Major: 7},
		{Ordinal: 1, UUID: "First", Name: "GPU", Major: 7, Minor: 5, TotalBytes: 2048},
		{Ordinal: 2, Major: 8, TotalBytes: 1024},
		{Ordinal: 3, UUID: "Third", Name: "GPU", Major: 12, TotalBytes: 4096},
		{Ordinal: 4, UUID: "Duplicate", Major: 8},
		{Ordinal: 5, UUID: "duplicate", Major: 8},
	}
	var sampled []int
	failure := errors.New("memory unavailable")
	read := func(a gpu.Adapter) (embeddinggemma2.DeviceMemory, error) {
		sampled = append(sampled, a.Ordinal)
		if a.Ordinal == 3 {
			return embeddinggemma2.DeviceMemory{}, failure
		}
		return embeddinggemma2.DeviceMemory{TotalBytes: int64(a.TotalBytes), AvailableBytes: 512, Basis: embeddinggemma2.MemoryLive}, nil
	}
	cheap, err := describeCUDADevices(adapters, embeddinggemma2.ProbeOptions{}, read)
	if err != nil || len(sampled) != 0 || len(cheap) != 5 || cheap[0].Alias != "cuda:1" || cheap[2].Alias != "cuda:3" ||
		cheap[0].ID != "cuda:First" || cheap[0].Backend != "cuda" || cheap[0].MemoryKind != embeddinggemma2.MemoryDedicated ||
		cheap[0].VendorID != 0x10de || !cheap[0].TextBatching || cheap[0].Memory.TotalBytes != 2048 ||
		cheap[0].Memory.Basis != embeddinggemma2.MemoryUnknown {
		t.Fatalf("cheap inventory %+v, queries %v, error %v", cheap, sampled, err)
	}
	for _, i := range []int{1, 3, 4} {
		if cheap[i].ID != "" || cheap[i].Alias != "" {
			t.Fatalf("unverified identifier at %d: %+v", i, cheap[i])
		}
	}
	live, err := describeCUDADevices(adapters, embeddinggemma2.ProbeOptions{IncludeMemory: true}, read)
	if !errors.Is(err, failure) || !slices.Equal(sampled, []int{1, 3}) || len(live) != len(cheap) ||
		live[0].Memory.Basis != embeddinggemma2.MemoryLive || live[0].Memory.AvailableBytes != 512 || live[2] != cheap[2] {
		t.Fatalf("live inventory %+v, queries %v, error %v", live, sampled, err)
	}
}
