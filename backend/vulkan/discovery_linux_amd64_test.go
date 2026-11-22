package vulkan

import (
	"errors"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestVulkanDiscoveryAndAttach(t *testing.T) {
	product := requiredDevice(t)
	devices, err := Devices()
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	m, err := embeddinggemma2.Load("../../testdata/embeddinggemma2-tiny")
	if err != nil {
		t.Fatal(err)
	}
	for _, device := range devices {
		if device.DeviceID != product {
			continue
		}
		found = true
		if device.ID == "" || device.Alias == "" || device.TotalLocalBytes == 0 {
			t.Fatalf("incomplete identity: %+v", device)
		}
		for _, selector := range []string{device.ID, device.Alias} {
			accel, err := m.NewAccelerator(selector)
			if err != nil {
				t.Fatal(err)
			}
			checkDevice(t, accel.(*encoder), product)
			bound, ok := AttachedDevice(accel)
			if !ok || bound.ID != device.ID {
				t.Fatalf("selector %q bound to %+v, want %q", selector, bound, device.ID)
			}
			if _, _, err := accel.Forward([]int{1, 2}, false); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("discovered %s (%s, %s) with %d device-local bytes", device.Name, device.ID, device.Alias, device.TotalLocalBytes)
	}
	if !found {
		t.Fatalf("PCI product %04x absent from %+v", product, devices)
	}
	_, err = m.NewAccelerator("vulkan:ffffffffffffffffffffffffffffffff")
	if !errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) {
		t.Fatalf("missing device: %v", err)
	}
}
