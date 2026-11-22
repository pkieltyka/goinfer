//go:build darwin

package metal

import (
	"errors"
	"fmt"

	gpu "github.com/pkieltyka/goinfer/backend/metal/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func probeMetalDevice(opts embeddinggemma2.ProbeOptions) ([]embeddinggemma2.DeviceInfo, error) {
	return probeMetalDeviceOn(CreateSystemDefaultDevice, opts)
}

func acquireMetalDevice(open func() (*Device, error)) (*Device, error) {
	device, err := open()
	if errors.Is(err, gpu.ErrNoDevice) {
		return nil, fmt.Errorf("%w: %w", embeddinggemma2.ErrNoDevice, err)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %w", embeddinggemma2.ErrBackendUnavailable, err)
	}
	return device, nil
}

func probeMetalDeviceOn(open func() (*Device, error), opts embeddinggemma2.ProbeOptions) ([]embeddinggemma2.DeviceInfo, error) {
	device, err := acquireMetalDevice(open)
	if errors.Is(err, embeddinggemma2.ErrNoDevice) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer device.ReleaseObjects()
	return []embeddinggemma2.DeviceInfo{describeMetalDevice(device.Name(), device.HasUnifiedMemory(), opts.IncludeMemory)}, nil
}

func describeMetalDevice(name string, unified, includeMemory bool) embeddinggemma2.DeviceInfo {
	info := embeddinggemma2.DeviceInfo{Backend: "metal", Name: name, MemoryKind: embeddinggemma2.MemoryDedicated}
	if unified {
		info.MemoryKind = embeddinggemma2.MemoryShared
		info.Memory = metalSharedMemory(includeMemory)
	}
	return info
}

func metalSharedMemory(includeMemory bool) embeddinggemma2.DeviceMemory {
	ram, err := physicalRAM()
	if err != nil || ram == 0 {
		return embeddinggemma2.DeviceMemory{}
	}
	memory := embeddinggemma2.DeviceMemory{TotalBytes: int64(ram)}
	if includeMemory {
		memory.AvailableBytes = metalMemoryCeiling(ram)
		memory.Basis = embeddinggemma2.MemoryEstimated
	}
	return memory
}

var _ embeddinggemma2.AttachedDeviceReporter = (*eg2Accel)(nil)
var _ embeddinggemma2.DeviceMemoryReporter = (*eg2Accel)(nil)

// AttachedDevice returns the system default Metal device captured at attach.
func (a *eg2Accel) AttachedDevice() embeddinggemma2.DeviceInfo { return a.identity }

// DeviceMemory estimates headroom when the attached device shares host memory.
func (a *eg2Accel) DeviceMemory() (embeddinggemma2.DeviceMemory, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.open(); err != nil {
		return embeddinggemma2.DeviceMemory{}, err
	}
	if a.identity.MemoryKind == embeddinggemma2.MemoryShared {
		return metalSharedMemory(true), nil
	}
	return a.identity.Memory, nil
}
