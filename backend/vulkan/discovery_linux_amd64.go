package vulkan

import (
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

// Devices lists visible discrete AMD adapters with a Vulkan compute queue. It
// creates a temporary Vulkan instance without allocating device buffers or
// loading a model. EmbeddingGemma 2 text batching is supported on these devices.
func Devices() ([]DeviceInfo, error) {
	adapters, err := vk.Devices()
	if err != nil {
		return nil, deviceError(err)
	}
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
	return devices, nil
}

func describeDevice(info vk.Info) DeviceInfo {
	var id string
	if info.UUID != ([16]byte{}) {
		id = "vulkan:" + hex.EncodeToString(info.UUID[:])
	}
	return DeviceInfo{
		ID: id, Name: info.Name, VendorID: info.VendorID, DeviceID: info.DeviceID,
		APIVersion: info.APIVersion, DriverVersion: info.DriverVersion, Type: "discrete-gpu",
		TotalLocalBytes: info.TotalLocalBytes,
	}
}

func openSelectedAccelerator(m *embeddinggemma2.Model, id string) (embeddinggemma2.Accelerator, error) {
	if len(id) != hex.EncodedLen(16) {
		devices, err := Devices()
		if err != nil {
			return nil, err
		}
		for _, device := range devices {
			if device.Alias == "vulkan:"+id {
				id = strings.TrimPrefix(device.ID, "vulkan:")
				break
			}
		}
	}
	var selector vk.Selector
	if len(id) != hex.EncodedLen(len(selector.UUID)) {
		return nil, fmt.Errorf("%w: invalid Vulkan device %q", embeddinggemma2.ErrDeviceUnavailable, id)
	}
	if _, err := hex.Decode(selector.UUID[:], []byte(id)); err != nil || selector.UUID == ([16]byte{}) {
		return nil, fmt.Errorf("%w: invalid Vulkan device %q", embeddinggemma2.ErrDeviceUnavailable, id)
	}
	return attach(m, selector)
}

func deviceError(err error) error {
	if errors.Is(err, vk.ErrDeviceUnavailable) {
		return errors.Join(embeddinggemma2.ErrDeviceUnavailable, err)
	}
	return err
}
