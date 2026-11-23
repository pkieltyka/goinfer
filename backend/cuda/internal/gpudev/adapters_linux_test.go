//go:build linux

package gpudev

import (
	"errors"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

type fakeAdapterHandle struct {
	Adapter
	uuidErr, capabilityErr   error
	retains, capabilityReads int
}

func (d *fakeAdapterHandle) Ordinal() int                 { return d.Adapter.Ordinal }
func (d *fakeAdapterHandle) UUID() (string, error)        { return d.Adapter.UUID, d.uuidErr }
func (d *fakeAdapterHandle) Name() (string, error)        { return d.Adapter.Name, nil }
func (d *fakeAdapterHandle) PCIBusID() (string, error)    { return d.Adapter.PCIBusID, nil }
func (d *fakeAdapterHandle) TotalMemory() (uint64, error) { return d.TotalBytes, nil }
func (d *fakeAdapterHandle) ComputeCapability() (int, int, error) {
	d.capabilityReads++
	return d.Major, d.Minor, d.capabilityErr
}
func (d *fakeAdapterHandle) Attribute(attr gc.DeviceAttribute) (int, error) {
	if attr != gc.DeviceAttributeIntegrated || !d.IntegratedKnown {
		return 0, gc.ErrNotSupported
	}
	if d.Integrated {
		return 1, nil
	}
	return 0, nil
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
	devs[0].IntegratedKnown = true // discrete
	devs[1].Integrated, devs[1].IntegratedKnown = true, true
	init := func() error { return nil }
	count := func() (int, error) { return len(devs), nil }
	get := func(ordinal int) (adapterHandle, error) { return &devs[ordinal], nil }
	rows, err := enumerateAdapters(init, count, get)
	if err != nil || len(rows) != 3 {
		t.Fatalf("enumerate: rows %+v, error %v", rows, err)
	}
	for i, row := range rows {
		if row != devs[i].Adapter || devs[i].retains != 0 || devs[i].capabilityReads != 1 {
			t.Fatalf("ordinal %d: got %+v, %d retains; want %+v, no retains", i, row, devs[i].retains, devs[i].Adapter)
		}
	}
	failure := errors.New("driver read failed")
	devs[1].uuidErr = failure
	rows, err = enumerateAdapters(init, count, get)
	if len(rows) != 2 || rows[1].Ordinal != 2 || !errors.Is(err, failure) {
		t.Fatalf("partial enumeration: rows %+v, error %v", rows, err)
	}
	devs[1].uuidErr = nil
	devs[1].capabilityErr = failure
	rows, err = enumerateAdapters(init, count, get)
	if !errors.Is(err, failure) || len(rows) != 2 || rows[0] != devs[0].Adapter || rows[1] != devs[2].Adapter {
		t.Fatalf("capability failure lost sibling rows or native error: rows %+v, error %v", rows, err)
	}
	for _, dev := range devs {
		if dev.retains != 0 {
			t.Fatalf("cheap enumeration retained ordinal %d", dev.Adapter.Ordinal)
		}
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

func TestCUDAAttachedAdapterPreservesVerifiedIdentity(t *testing.T) {
	d := &fakeAdapterHandle{
		Adapter: Adapter{Ordinal: 2, UUID: "actual", Major: 7, Minor: 5, Integrated: true, IntegratedKnown: true},
		uuidErr: gc.ErrNotSupported,
	}
	got, err := attachedAdapter(d, "actual", nil)
	if err != nil || got != d.Adapter || d.capabilityReads != 1 {
		t.Fatalf("attachment got %+v, %v, %d capability reads; want %+v", got, err, d.capabilityReads, d.Adapter)
	}
	d.capabilityErr = gc.ErrInvalidDevice
	got, err = attachedAdapter(d, "actual", nil)
	want := d.Adapter
	want.Major, want.Minor = 0, 0
	if !errors.Is(err, d.capabilityErr) || got != want || d.capabilityReads != 2 {
		t.Fatalf("failed capability read got %+v, %v, %d capability reads; want %+v", got, err, d.capabilityReads, want)
	}

	d.uuidErr, d.capabilityErr = nil, nil
	list := func() ([]Adapter, error) {
		return []Adapter{d.Adapter, {Ordinal: 3, UUID: "ACTUAL"}}, nil
	}
	if got, err := attachedAdapter(d, "", list); err != nil || got.UUID != "" || got.Major != 7 || got.Minor != 5 {
		t.Fatalf("bare attachment did not preserve capability while suppressing an ambiguous ID: %+v, %v", got, err)
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
