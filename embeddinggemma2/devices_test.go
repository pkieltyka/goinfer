package embeddinggemma2

import (
	"errors"
	"fmt"
	"slices"
	"testing"
)

func TestDevicesReturnsOrderedPartialInventory(t *testing.T) {
	const a, b = "fake-a", "fake-b"
	const missing = "fake-absent"
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
	RegisterDeviceProber(missing, nil)
	t.Cleanup(func() {
		accelMu.Lock()
		delete(deviceProbes, a)
		delete(deviceProbes, b)
		delete(deviceProbes, missing)
		accelMu.Unlock()
	})
	devices, failures := Devices(ProbeOptions{IncludeMemory: true})
	if !slices.Equal(called, []string{a, b}) || len(devices) != 2 || devices[0].ID != a+":abc" || devices[1].ID != b+":abc" ||
		len(failures) != 2 || failures[0].Backend != missing || !errors.Is(failures[0].Err, ErrBackendUnavailable) ||
		failures[1].Backend != b || !errors.Is(failures[1].Err, failure) {
		t.Fatalf("called %v; inventory %+v; failures %+v", called, devices, failures)
	}
}

func TestUseDeviceBindsOnlyRequestedSelectorAndToken(t *testing.T) {
	const backend, override, other = "fake-dev", "fake-override", "fake-other"
	var called []string
	failure := errors.New("attach failed")
	unsupported := fmt.Errorf("selector hardware: %w", ErrUnsupportedSelector)
	registerTempAccelerator(t, backend, fakeText{"bare"})
	open := func(selector string) func(*Model, string) (Accelerator, error) {
		return func(_ *Model, token string) (Accelerator, error) {
			called = append(called, selector+":"+token)
			switch token {
			case "fail":
				return nil, failure
			case "unsupported":
				return nil, unsupported
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
		{backend + ":unsupported", override, ErrUnsupportedSelector},
	} {
		acc, err := e.UseDevice(tt.id, tt.selector)
		if acc != nil || !errors.Is(err, tt.want) || e.Accelerator() != attached ||
			tt.want == ErrDeviceUnavailable && errors.Is(err, ErrBackendUnavailable) {
			t.Fatalf("UseDevice(%q, %q): %v %v; attached %v", tt.id, tt.selector, acc, err, e.Accelerator())
		}
	}
	wantCalls := []string{backend + ":abcd", override + ":0", override + ":fail", override + ":unsupported"}
	if !slices.Equal(called, wantCalls) {
		t.Fatalf("called %v, want %v", called, wantCalls)
	}
	if acc, err := e.UseAccelerator(backend); err != nil || acc.Name() != "bare" {
		t.Fatalf("bare behavior changed: %v %v", acc, err)
	}
}

func TestDeviceRunnableSelectors(t *testing.T) {
	const backend = "cuda"
	const (
		first   = "test-capability-a"
		last    = "test-capability-z"
		legacy  = "test-capability-legacy"
		denied  = "test-capability-denied"
		nilOpen = "test-capability-nil"
		other   = "test-capability-other"
	)
	t.Cleanup(func() {
		accelMu.Lock()
		defer accelMu.Unlock()
		for _, name := range []string{"", first, last, legacy, denied, nilOpen, other} {
			delete(selectedAccels, name)
		}
	})
	open := func(*Model, string) (Accelerator, error) {
		t.Error("inventory must not open a device")
		return nil, nil
	}
	var calls, otherCalls int
	supports := func(d DeviceInfo) bool {
		calls++
		return d.ComputeCapability.Major >= 8 && slices.Contains(DeviceSelectors(backend), first)
	}
	RegisterDeviceAcceleratorWithSupport("", backend, open, func(DeviceInfo) bool {
		t.Error("empty selector name ran a predicate")
		return true
	})
	RegisterDeviceAcceleratorWithSupport(last, backend, open, supports)
	RegisterDeviceAcceleratorWithSupport(first, backend, open, supports)
	RegisterDeviceAccelerator(legacy, backend, open)
	RegisterDeviceAcceleratorWithSupport(denied, backend, open, func(DeviceInfo) bool { return false })
	RegisterDeviceAcceleratorWithSupport(nilOpen, backend, nil, func(DeviceInfo) bool {
		t.Error("ran predicate without a factory")
		return true
	})
	RegisterDeviceAcceleratorWithSupport(other, "test-non-cuda", open, func(DeviceInfo) bool {
		otherCalls++
		return true
	})

	d := DeviceInfo{ID: "cuda:GPU-abcd1234-1111-2222-3333-444444444444", Alias: "cuda:0", Backend: backend,
		ComputeCapability: CUDAComputeCapability{Major: 8}}
	different := d
	different.ComputeCapability.Minor++
	seen := map[DeviceInfo]bool{d: true}
	if !seen[d] || seen[different] {
		t.Fatal("DeviceInfo lost value semantics")
	}
	got := d.RunnableSelectors()
	if !slices.Equal(got, []string{first, last}) || calls != 2 {
		t.Fatalf("runnable selectors = %v, %d predicate calls", got, calls)
	}
	got[0] = "caller mutation"
	if next := d.RunnableSelectors(); !slices.Equal(next, []string{first, last}) {
		t.Fatalf("caller changed later snapshot: %v", next)
	}
	if linked := DeviceSelectors(backend); !slices.Equal(linked, []string{"", first, denied, legacy, last}) {
		t.Fatalf("linked selectors = %v", linked)
	}
	for _, id := range []string{"", "abc", "cuda:", "cuda::0", "cuda:12", "cuda:bad token", "metal:abc", d.Alias} {
		invalid := d
		invalid.ID = id
		before := calls
		if got := invalid.RunnableSelectors(); got != nil || calls != before {
			t.Errorf("unusable ID %q: selectors %v, predicate calls %d", id, got, calls-before)
		}
	}
	nonCUDA := DeviceInfo{ID: "test-non-cuda:abc", Backend: "test-non-cuda"}
	if got := nonCUDA.RunnableSelectors(); got != nil || otherCalls != 0 {
		t.Fatalf("non-CUDA: selectors %v, calls %d", got, otherCalls)
	}
	if got := DeviceSelectors(nonCUDA.Backend); !slices.Equal(got, []string{other}) {
		t.Fatalf("non-CUDA linked selectors = %v", got)
	}
	RegisterDeviceAccelerator(first, backend, open)
	RegisterDeviceAcceleratorWithSupport(last, backend, open, func(DeviceInfo) bool { return false })
	if got := d.RunnableSelectors(); got != nil {
		t.Fatalf("overwritten factories retained support: %v", got)
	}
}
