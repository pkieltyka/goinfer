package vulkan

import (
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"testing"
)

type foreignAccelerator struct{ embeddinggemma2.Accelerator }

func TestAttachedDeviceRequiresVulkan(t *testing.T) {
	for _, foreign := range []embeddinggemma2.Accelerator{nil, foreignAccelerator{}} {
		if got, ok := AttachedDevice(foreign); ok || got != (DeviceInfo{}) {
			t.Fatalf("foreign accelerator returned %+v, %v", got, ok)
		}
	}
}
