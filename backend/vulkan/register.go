// Package vulkan registers the direct Vulkan EmbeddingGemma 2 text accelerator.
// Importing the package registers a factory. The system loader and AMD device are
// opened when the caller inventories devices or attaches the accelerator.
package vulkan

import "github.com/pkieltyka/goinfer/embeddinggemma2"

//go:generate sh shaders/generate.sh

func init() {
	embeddinggemma2.RegisterAccelerator("vulkan", openAccelerator)
	embeddinggemma2.RegisterSelectedAccelerator("vulkan", openSelectedAccelerator)
	embeddinggemma2.RegisterDeviceProber("vulkan", probeInventory)
}

func probeInventory(opts embeddinggemma2.ProbeOptions) ([]embeddinggemma2.DeviceInfo, error) {
	adapters, err := Probe(opts)
	rows := make([]embeddinggemma2.DeviceInfo, 0, len(adapters))
	for _, adapter := range adapters {
		rows = append(rows, inventoryDevice(adapter))
	}
	return rows, err
}

func inventoryDevice(device DeviceInfo) embeddinggemma2.DeviceInfo {
	return embeddinggemma2.DeviceInfo{
		ID: device.ID, Alias: device.Alias, Backend: "vulkan", Name: device.Name,
		VendorID: device.VendorID, ProductID: device.DeviceID, MemoryKind: embeddinggemma2.MemoryDedicated,
		Memory: device.Memory, TextBatching: true,
	}
}

// DeviceInfo describes a discrete AMD Vulkan device. ID is based on the Vulkan
// device UUID and is independent of discovery order. The UUID identifies the device on
// this host. Re-inventory after hardware or driver changes. Alias is an ordinal
// in the current inventory and can move when the visible device set changes.
// A UUID-less device has an empty ID and alias, but occupies its inventory
// position. It can attach only as the sole match of a bare Vulkan selector.
// TotalLocalBytes is the sum of device-local heap sizes.
// Memory can include estimated headroom when Probe requests it.
// APIVersion and DriverVersion contain Vulkan's original packed version words.
// The driver version's encoding is vendor-specific.
type DeviceInfo struct {
	ID, Alias, Name                               string
	TotalLocalBytes                               uint64
	Memory                                        embeddinggemma2.DeviceMemory
	VendorID, DeviceID, APIVersion, DriverVersion uint32
	Type                                          string
}

// AttachedDevice returns a copy of the identity captured by this backend at
// attachment. It remains available after Close. Other accelerators return false.
// Read a fresh memory snapshot through embeddinggemma2.DeviceMemoryReporter.
func AttachedDevice(acc embeddinggemma2.Accelerator) (DeviceInfo, bool) {
	a, ok := acc.(interface{ attachedDevice() (DeviceInfo, bool) })
	if !ok {
		return DeviceInfo{}, false
	}
	return a.attachedDevice()
}
