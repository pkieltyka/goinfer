// Package vulkan registers the direct Vulkan EmbeddingGemma 2 text accelerator.
// Importing the package registers a factory. The system loader and AMD device are
// opened only when the caller selects "vulkan".
package vulkan

import "github.com/pkieltyka/goinfer/embeddinggemma2"

//go:generate sh shaders/generate.sh

func init() { embeddinggemma2.RegisterAccelerator("vulkan", openAccelerator) }

// DeviceInfo describes the physical Vulkan device selected at attachment.
// APIVersion and DriverVersion contain Vulkan's original packed version words.
// The driver version's encoding is vendor-specific.
type DeviceInfo struct {
	Name                                          string
	VendorID, DeviceID, APIVersion, DriverVersion uint32
	Type                                          string
}

// AttachedDevice returns a copy of the identity captured by this backend at
// attachment. It remains available after Close. Other accelerators return false.
func AttachedDevice(acc embeddinggemma2.Accelerator) (DeviceInfo, bool) {
	a, ok := acc.(interface{ attachedDevice() (DeviceInfo, bool) })
	if !ok {
		return DeviceInfo{}, false
	}
	return a.attachedDevice()
}
