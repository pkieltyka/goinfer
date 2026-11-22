//go:build linux

package gpudev

import (
	"errors"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

type fakeAdapterHandle struct {
	Adapter
	uuidErr error
	retains int
}

func (d *fakeAdapterHandle) Ordinal() int                 { return d.Adapter.Ordinal }
func (d *fakeAdapterHandle) UUID() (string, error)        { return d.Adapter.UUID, d.uuidErr }
func (d *fakeAdapterHandle) Name() (string, error)        { return d.Adapter.Name, nil }
func (d *fakeAdapterHandle) PCIBusID() (string, error)    { return d.Adapter.PCIBusID, nil }
func (d *fakeAdapterHandle) TotalMemory() (uint64, error) { return d.TotalBytes, nil }
func (d *fakeAdapterHandle) ComputeCapability() (int, int, error) {
	return d.Major, d.Minor, nil
}
func (d *fakeAdapterHandle) Primary() (*gc.Context, error) {
	d.retains++
	return nil, nil
}

func TestCUDAAdapterEnumeration(t *testing.T) {
	devs := []fakeAdapterHandle{
		{Adapter: Adapter{Ordinal: 0, UUID: "first", Name: "A", PCIBusID: "0000:01:00.0", TotalBytes: 1024, Major: 7, Minor: 5}},
		{Adapter: Adapter{Ordinal: 1, UUID: "second", Name: "B", PCIBusID: "0000:02:00.0", TotalBytes: 2048, Major: 12}},
		{Adapter: Adapter{Ordinal: 2, Name: "C", TotalBytes: 512, Major: 7}, uuidErr: gc.ErrSymbolUnavailable},
	}
	init := func() error { return nil }
	count := func() (int, error) { return len(devs), nil }
	get := func(ordinal int) (adapterHandle, error) { return &devs[ordinal], nil }
	rows, err := enumerateAdapters(init, count, get)
	if err != nil || len(rows) != 3 {
		t.Fatalf("enumerate: rows %+v, error %v", rows, err)
	}
	for i, row := range rows {
		if row != devs[i].Adapter || devs[i].retains != 0 {
			t.Fatalf("ordinal %d: got %+v, %d retains; want %+v, no retains", i, row, devs[i].retains, devs[i].Adapter)
		}
	}
	failure := errors.New("driver read failed")
	devs[1].uuidErr = failure
	rows, err = enumerateAdapters(init, count, get)
	if len(rows) != 2 || rows[1].Ordinal != 2 || !errors.Is(err, failure) {
		t.Fatalf("partial enumeration: rows %+v, error %v", rows, err)
	}
	rows, err = enumerateAdapters(func() error { return gc.ErrNoDevice }, nil, nil)
	if len(rows) != 0 || err != nil {
		t.Fatalf("empty enumeration: rows %+v, error %v", rows, err)
	}
	rows, err = enumerateAdapters(init, func() (int, error) { return 0, nil }, nil)
	if len(rows) != 0 || err != nil {
		t.Fatalf("zero-device enumeration: rows %+v, error %v", rows, err)
	}
	rows, err = enumerateAdapters(func() error { return failure }, nil, nil)
	if len(rows) != 0 || !errors.Is(err, ErrDriverInit) || !errors.Is(err, failure) {
		t.Fatalf("init failure: rows %+v, error %v", rows, err)
	}
	for _, native := range []error{gc.ErrInvalidOrdinal, gc.ErrInvalidDevice} {
		rows, err := enumerateAdapters(init, func() (int, error) { return 1, nil }, func(int) (adapterHandle, error) {
			return nil, native
		})
		if len(rows) != 0 || !errors.Is(err, ErrAdapterChanged) || !errors.Is(err, native) {
			t.Fatalf("vanished ordinal: rows %+v, error %v (%v)", rows, err, native)
		}
	}
	for _, native := range []error{gc.ErrNoDevice, gc.ErrInvalidDevice} {
		rows, err := enumerateAdapters(init, func() (int, error) { return 0, native }, nil)
		if len(rows) != 0 || !errors.Is(err, ErrAdapterChanged) || !errors.Is(err, native) {
			t.Fatalf("failed count: rows %+v, error %v (%v)", rows, err, native)
		}
	}
}

func TestCUDAAdapterRetainsOnlyVerifiedHandle(t *testing.T) {
	first := &fakeAdapterHandle{Adapter: Adapter{Ordinal: 0, UUID: "first"}}
	second := &fakeAdapterHandle{Adapter: Adapter{Ordinal: 1, UUID: "second"}}
	if _, err := retainPrimary(first, "second"); !errors.Is(err, ErrAdapterChanged) || first.retains != 0 {
		t.Fatalf("changed adapter: error %v, %d retains", err, first.retains)
	}
	if _, err := retainPrimary(second, "SECOND"); err != nil || second.retains != 1 {
		t.Fatalf("matching adapter: error %v, %d retains", err, second.retains)
	}
	second.uuidErr = gc.ErrSymbolUnavailable
	if _, err := retainPrimary(second, "second"); !errors.Is(err, gc.ErrSymbolUnavailable) || second.retains != 1 {
		t.Fatalf("unverified adapter: error %v, %d retains", err, second.retains)
	}
	second.uuidErr = gc.ErrInvalidDevice
	if _, err := retainPrimary(second, "second"); !errors.Is(err, ErrAdapterChanged) || !errors.Is(err, gc.ErrInvalidDevice) || second.retains != 1 {
		t.Fatalf("vanished adapter: error %v, %d retains", err, second.retains)
	}
	if _, err := retainPrimary(first, ""); err != nil || first.retains != 1 {
		t.Fatalf("bare selection: error %v, %d retains", err, first.retains)
	}
}
