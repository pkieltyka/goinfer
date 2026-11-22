//go:build linux

package gpudev

import (
	"errors"
	"fmt"
	"strings"

	gc "github.com/eitamring/gocudrv/cuda"
)

// Adapter describes a visible CUDA device. Ordinal is the driver's ordinal.
type Adapter struct {
	Ordinal      int
	UUID         string
	Name         string
	PCIBusID     string
	TotalBytes   uint64
	Major, Minor int
}

var (
	ErrDriverInit     = errors.New("cuda: driver init")
	ErrAdapterChanged = errors.New("cuda: the selected ordinal no longer identifies the requested device")
)

type adapterHandle interface {
	Ordinal() int
	UUID() (string, error)
	Name() (string, error)
	PCIBusID() (string, error)
	TotalMemory() (uint64, error)
	ComputeCapability() (int, int, error)
}

// Adapters lists driver-visible devices without retaining a CUDA context.
func Adapters() ([]Adapter, error) {
	return enumerateAdapters(gc.Init, gc.DeviceCount, func(ordinal int) (adapterHandle, error) {
		return gc.GetDevice(ordinal)
	})
}

func enumerateAdapters(init func() error, count func() (int, error), get func(int) (adapterHandle, error)) ([]Adapter, error) {
	if err := init(); err != nil {
		if errors.Is(err, gc.ErrNoDevice) {
			return nil, nil
		}
		return nil, fmt.Errorf("%w: %w", ErrDriverInit, err)
	}
	n, err := count()
	if err != nil {
		return nil, fmt.Errorf("cuda: device count: %w", err)
	}
	var adapters []Adapter
	var failures []error
	for ordinal := range n {
		dev, err := get(ordinal)
		if err == nil {
			var adapter Adapter
			adapter, err = describeAdapter(dev)
			if err == nil {
				adapters = append(adapters, adapter)
				continue
			}
		}
		failures = append(failures, fmt.Errorf("cuda: ordinal %d: %w", ordinal, adapterAccessError(err)))
	}
	return adapters, errors.Join(failures...)
}

func adapterAccessError(err error) error {
	if errors.Is(err, gc.ErrInvalidOrdinal) || errors.Is(err, gc.ErrInvalidDevice) || errors.Is(err, gc.ErrNoDevice) {
		return fmt.Errorf("%w: %w", ErrAdapterChanged, err)
	}
	return err
}

func describeAdapter(dev adapterHandle) (Adapter, error) {
	a := Adapter{Ordinal: dev.Ordinal()}
	var err error
	if a.UUID, err = dev.UUID(); err != nil && !errors.Is(err, gc.ErrSymbolUnavailable) {
		return Adapter{}, fmt.Errorf("device UUID: %w", err)
	}
	if a.PCIBusID, err = dev.PCIBusID(); err != nil && !errors.Is(err, gc.ErrSymbolUnavailable) {
		return Adapter{}, fmt.Errorf("PCI bus ID: %w", err)
	}
	if a.Name, err = dev.Name(); err != nil {
		return Adapter{}, fmt.Errorf("device name: %w", err)
	}
	if a.TotalBytes, err = dev.TotalMemory(); err != nil {
		return Adapter{}, fmt.Errorf("device capacity: %w", err)
	}
	if a.Major, a.Minor, err = dev.ComputeCapability(); err != nil {
		return Adapter{}, fmt.Errorf("compute capability: %w", err)
	}
	return a, nil
}

// Adapter reports identity from the same handle that retained this context.
// Unavailable details are left empty.
func (d *Device) Adapter() Adapter {
	dev := d.cx.Device()
	name, _ := dev.Name()
	uuid, _ := dev.UUID()
	pci, _ := dev.PCIBusID()
	total, _ := dev.TotalMemory()
	return Adapter{Ordinal: dev.Ordinal(), UUID: uuid, Name: name, PCIBusID: pci, TotalBytes: total}
}

// CreateSelectedDevice retains the primary context only after checking the UUID
// on the same handle. A change in ordinal mapping cannot redirect the attach.
func CreateSelectedDevice(a Adapter) (*Device, error) {
	if a.UUID == "" {
		return nil, ErrAdapterChanged
	}
	return createDevice(a.Ordinal, a.UUID)
}

func retainPrimary(dev interface {
	UUID() (string, error)
	Primary() (*gc.Context, error)
}, expectedUUID string) (*gc.Context, error) {
	if expectedUUID != "" {
		uuid, err := dev.UUID()
		if err != nil {
			return nil, fmt.Errorf("cuda: verify selected device UUID: %w", adapterAccessError(err))
		}
		if !strings.EqualFold(uuid, expectedUUID) {
			return nil, fmt.Errorf("%w: wanted %s, found %s", ErrAdapterChanged, expectedUUID, uuid)
		}
	}
	cx, err := dev.Primary()
	if err != nil {
		return nil, fmt.Errorf("cuda: retain primary context: %w", adapterAccessError(err))
	}
	return cx, nil
}
