package vulkan

import (
	"errors"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestVulkanDiscoveryAndAttach(t *testing.T) {
	product := requiredDevice(t)
	devices, err := Probe(embeddinggemma2.ProbeOptions{IncludeMemory: true})
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
		if device.Memory.Basis != embeddinggemma2.MemoryEstimated || device.Memory.TotalBytes != int64(device.TotalLocalBytes) ||
			device.Memory.AvailableBytes < 0 || device.Memory.AvailableBytes > device.Memory.TotalBytes {
			t.Fatalf("invalid estimated memory: %+v", device.Memory)
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
			reporter := accel.(embeddinggemma2.DeviceMemoryReporter)
			memory, err := reporter.DeviceMemory()
			if err != nil || memory.Basis != embeddinggemma2.MemoryEstimated || memory.TotalBytes != device.Memory.TotalBytes ||
				memory.AvailableBytes < 0 || memory.AvailableBytes > memory.TotalBytes || bound.Memory.Basis != embeddinggemma2.MemoryUnknown {
				t.Fatalf("attach memory %+v, identity snapshot %+v: %v", memory, bound.Memory, err)
			}
			t.Logf("%s attached memory %+v", selector, memory)
			if err := accel.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err := reporter.DeviceMemory(); err == nil {
				t.Fatal("closed accelerator reported memory")
			}
		}
		t.Logf("discovered %s (%s, %s) with %d device-local bytes and %d estimated available", device.Name, device.ID, device.Alias, device.TotalLocalBytes, device.Memory.AvailableBytes)
	}
	if !found {
		t.Fatalf("PCI product %04x absent from %+v", product, devices)
	}
	_, err = m.NewAccelerator("vulkan:ffffffffffffffffffffffffffffffff")
	if !errors.Is(err, embeddinggemma2.ErrNoDevice) {
		t.Fatalf("missing device: %v", err)
	}
}
