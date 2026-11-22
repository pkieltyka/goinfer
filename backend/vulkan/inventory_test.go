package vulkan

import (
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestVulkanInventoryBridge(t *testing.T) {
	src := DeviceInfo{
		ID: "vulkan:00000000090000000000000000000000", Alias: "vulkan:1", Name: "GPU",
		VendorID: 0x1002, DeviceID: 0x7590,
		Memory: embeddinggemma2.DeviceMemory{TotalBytes: 1024, AvailableBytes: 0, Basis: embeddinggemma2.MemoryEstimated},
	}
	got := inventoryDevice(src)
	if got.ID != src.ID || got.Alias != src.Alias || got.Name != src.Name || got.Backend != "vulkan" ||
		got.VendorID != src.VendorID || got.ProductID != src.DeviceID || got.Memory != src.Memory ||
		got.MemoryKind != embeddinggemma2.MemoryDedicated || !got.TextBatching {
		t.Fatalf("converted %+v from %+v", got, src)
	}
	src.ID, src.Alias = "", ""
	got = inventoryDevice(src)
	if got.ID != "" || got.Alias != "" {
		t.Fatalf("UUID-less row acquired identity: %+v", got)
	}
}
