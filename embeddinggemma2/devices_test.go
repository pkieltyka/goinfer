package embeddinggemma2

import (
	"errors"
	"slices"
	"testing"
)

func TestDevicesReturnsOrderedPartialInventory(t *testing.T) {
	const a, b = "fake-a", "fake-b"
	failure := errors.New("adapter query failed")
	var called []string
	RegisterDeviceProber(b, func(opts ProbeOptions) ([]DeviceInfo, error) {
		called = append(called, b)
		if !opts.IncludeMemory {
			t.Error("probe did not receive IncludeMemory")
		}
		return []DeviceInfo{{ID: b + ":abc", Backend: b, Name: "Device", ProductID: 1234}}, failure
	})
	RegisterDeviceProber(a, func(ProbeOptions) ([]DeviceInfo, error) {
		called = append(called, a)
		DeviceSelectors(a) // registration lookup must not deadlock inside a probe
		return []DeviceInfo{{ID: a + ":abc", Backend: a, Name: "Device", ProductID: 1234}}, nil
	})
	t.Cleanup(func() {
		accelMu.Lock()
		delete(deviceProbes, a)
		delete(deviceProbes, b)
		accelMu.Unlock()
	})
	devices, failures := Devices(ProbeOptions{IncludeMemory: true})
	if !slices.Equal(called, []string{a, b}) || len(devices) != 2 || devices[0].ID != a+":abc" || devices[1].ID != b+":abc" ||
		len(failures) != 1 || failures[0].Backend != b || !errors.Is(failures[0].Err, failure) {
		t.Fatalf("called %v; inventory %+v; failures %+v", called, devices, failures)
	}
}

func TestUseDeviceBindsOnlyRequestedSelectorAndToken(t *testing.T) {
	const backend, override, other = "fake-dev", "fake-override", "fake-other"
	var called []string
	failure := errors.New("attach failed")
	registerTempAccelerator(t, backend, fakeText{"bare"})
	open := func(selector string) func(*Model, string) (Accelerator, error) {
		return func(_ *Model, token string) (Accelerator, error) {
			called = append(called, selector+":"+token)
			if token == "fail" {
				return nil, failure
			}
			return fakeText{selector}, nil
		}
	}
	RegisterSelectedAccelerator(backend, open(backend))
	RegisterDeviceAccelerator(override, backend, open(override))
	RegisterSelectedAccelerator(other, open(other))
	t.Cleanup(func() {
		accelMu.Lock()
		defer accelMu.Unlock()
		for _, name := range []string{backend, override, other} {
			delete(selectedAccels, name)
		}
	})
	if got := DeviceSelectors(backend); !slices.Equal(got, []string{backend, override}) {
		t.Fatalf("selectors: %v", got)
	}
	e := &Encoder{m: &Model{}, vis: &visionTower{}, aud: &audioTower{}}
	for _, tt := range []struct {
		id, selector, want string
	}{
		{backend + ":abcd", "", backend},
		{backend + ":0", override, override},
	} {
		acc, err := e.UseDevice(tt.id, tt.selector)
		if err != nil || acc != e.Accelerator() || acc.Name() != tt.want ||
			e.VisionDevice() != selectedDeviceTowerReason || e.AudioDevice() != selectedDeviceTowerReason {
			t.Fatalf("UseDevice(%q, %q): %v %v, vision %q, audio %q", tt.id, tt.selector, acc, err, e.VisionDevice(), e.AudioDevice())
		}
	}
	attached := e.Accelerator()
	for _, tt := range []struct {
		id, selector string
		want         error
	}{
		{backend, "", ErrDeviceUnavailable},
		{backend + ":", "", ErrDeviceUnavailable},
		{backend + ":abcd", other, ErrDeviceUnavailable},
		{backend + ":abcd", "unlinked", ErrBackendUnavailable},
		{"unlinked:abcd", "", ErrBackendUnavailable},
		{backend + ":fail", override, failure},
	} {
		acc, err := e.UseDevice(tt.id, tt.selector)
		if acc != nil || !errors.Is(err, tt.want) || e.Accelerator() != attached ||
			tt.want == ErrDeviceUnavailable && errors.Is(err, ErrBackendUnavailable) {
			t.Fatalf("UseDevice(%q, %q): %v %v; attached %v", tt.id, tt.selector, acc, err, e.Accelerator())
		}
	}
	wantCalls := []string{backend + ":abcd", override + ":0", override + ":fail"}
	if !slices.Equal(called, wantCalls) {
		t.Fatalf("called %v, want %v", called, wantCalls)
	}
	if acc, err := e.UseAccelerator(backend); err != nil || acc.Name() != "bare" {
		t.Fatalf("bare behavior changed: %v %v", acc, err)
	}
}
