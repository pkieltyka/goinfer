package vulkan

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

// Devices lists visible discrete AMD adapters with a Vulkan compute queue.
func Devices() ([]DeviceInfo, error) { return Probe(embeddinggemma2.ProbeOptions{}) }

// Probe lists eligible adapters without creating a logical device or loading a
// model. IncludeMemory requests a driver budget estimate for each adapter.
func Probe(opts embeddinggemma2.ProbeOptions) ([]DeviceInfo, error) {
	adapters, err := vk.Devices(opts.IncludeMemory)
	if err != nil {
		return nil, classify(err)
	}
	return describeDevices(adapters), nil
}

func describeDevices(adapters []vk.Info) []DeviceInfo {
	devices := make([]DeviceInfo, 0, len(adapters))
	for _, adapter := range adapters {
		devices = append(devices, describeDevice(adapter))
	}
	sort.SliceStable(devices, func(i, j int) bool { return devices[i].ID < devices[j].ID })
	for i := range devices {
		if devices[i].ID != "" {
			devices[i].Alias = "vulkan:" + strconv.Itoa(i)
		}
	}
	return devices
}

func describeDevice(info vk.Info) DeviceInfo {
	var id string
	if info.UUID != ([16]byte{}) {
		id = "vulkan:" + hex.EncodeToString(info.UUID[:])
	}
	return DeviceInfo{
		ID: id, Name: info.Name, VendorID: info.VendorID, DeviceID: info.DeviceID,
		APIVersion: info.APIVersion, DriverVersion: info.DriverVersion, Type: "discrete-gpu",
		TotalLocalBytes: info.TotalLocalBytes, Memory: deviceMemory(info.TotalLocalBytes, info.AvailableLocalBytes, info.BudgetKnown),
	}
}

func deviceMemory(total, available uint64, known bool) embeddinggemma2.DeviceMemory {
	memory := embeddinggemma2.DeviceMemory{TotalBytes: int64(min(total, math.MaxInt64))}
	if known {
		memory.AvailableBytes = int64(min(total, available, math.MaxInt64))
		memory.Basis = embeddinggemma2.MemoryEstimated
	}
	return memory
}

func openSelectedAccelerator(m *embeddinggemma2.Model, id string) (embeddinggemma2.Accelerator, error) {
	selector, err := resolveSelector(id, Devices)
	if err != nil {
		return nil, err
	}
	return attach(m, selector)
}

func resolveSelector(id string, inventory func() ([]DeviceInfo, error)) (vk.Selector, error) {
	const prefix = "vulkan:"
	var selector vk.Selector
	if len(id) != hex.EncodedLen(len(selector.UUID)) {
		if id == "" {
			return selector, fmt.Errorf("%w: empty Vulkan selector", embeddinggemma2.ErrDeviceUnavailable)
		}
		for _, r := range id {
			if r < '0' || r > '9' {
				return selector, fmt.Errorf("%w: invalid Vulkan selector %q", embeddinggemma2.ErrDeviceUnavailable, id)
			}
		}
		devices, err := inventory()
		if err != nil {
			return selector, err
		}
		alias := prefix + id
		id = ""
		for _, device := range devices {
			if device.Alias == alias {
				id, _ = strings.CutPrefix(device.ID, prefix)
				break
			}
		}
		if id == "" {
			return selector, fmt.Errorf("%w: Vulkan alias %q was not found", embeddinggemma2.ErrNoDevice, alias)
		}
	}
	if len(id) != hex.EncodedLen(len(selector.UUID)) {
		return selector, fmt.Errorf("%w: invalid Vulkan UUID %q", embeddinggemma2.ErrDeviceUnavailable, id)
	}
	if _, err := hex.Decode(selector.UUID[:], []byte(id)); err != nil || selector.UUID == ([16]byte{}) {
		return vk.Selector{}, fmt.Errorf("%w: invalid Vulkan UUID %q", embeddinggemma2.ErrDeviceUnavailable, id)
	}
	return selector, nil
}

func classify(err error) error {
	var category error
	switch {
	case errors.Is(err, vk.ErrDeviceLost):
		return &embeddinggemma2.DeviceLostError{Cause: err}
	case errors.Is(err, vk.ErrOutOfMemory):
		category = embeddinggemma2.ErrDeviceOutOfMemory
	case errors.Is(err, vk.ErrRuntimeUnavailable):
		category = embeddinggemma2.ErrBackendUnavailable
	case errors.Is(err, vk.ErrNoAdapter):
		category = embeddinggemma2.ErrNoDevice
	case errors.Is(err, vk.ErrAmbiguousAdapter):
		category = embeddinggemma2.ErrAmbiguousDevice
	default:
		return err
	}
	return fmt.Errorf("%w: %w", category, err)
}
