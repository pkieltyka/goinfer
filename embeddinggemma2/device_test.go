package embeddinggemma2

import (
	"errors"
	"fmt"
	"testing"
)

func TestDeviceErrorCategories(t *testing.T) {
	categories := []error{ErrBackendUnavailable, ErrNoDevice, ErrAmbiguousDevice, ErrUnsupportedSelector, ErrDeviceOutOfMemory}
	for _, err := range categories {
		wrapped := fmt.Errorf("attach: %w", err)
		if errors.Is(wrapped, ErrDeviceUnavailable) != (err != ErrDeviceOutOfMemory) {
			t.Errorf("%v has the wrong umbrella category", wrapped)
		}
		for _, other := range categories {
			if errors.Is(wrapped, other) != (err == other) {
				t.Errorf("errors.Is(%v, %v) = %t", err, other, errors.Is(wrapped, other))
			}
		}
	}
	for _, name := range []string{"test-device-absent", "test-device-absent:0"} {
		if _, err := (&Model{}).NewAccelerator(name); !errors.Is(err, ErrBackendUnavailable) {
			t.Errorf("%q: wanted unlinked-backend error, got %v", name, err)
		}
	}
}

type memoryTestAccelerator struct {
	*fakeEmbedAccel
	memory DeviceMemory
	calls  int
}

func (a *memoryTestAccelerator) DeviceMemory() (DeviceMemory, error) {
	a.calls++
	return a.memory, nil
}

func TestOptionalDeviceMemoryReporter(t *testing.T) {
	for i, want := range []string{"unknown", "live", "estimated"} {
		if got := MemoryBasis(i).String(); got != want {
			t.Errorf("basis %d = %q, want %q", i, got, want)
		}
	}
	if (DeviceMemory{}).Basis != MemoryUnknown {
		t.Fatal("zero value must have unknown basis")
	}
	e := newTinyEncoderForTest(t)
	a := &memoryTestAccelerator{fakeEmbedAccel: &fakeEmbedAccel{name: "test-a"}, memory: DeviceMemory{AvailableBytes: 2, TotalBytes: 8, Basis: MemoryLive}}
	b := &memoryTestAccelerator{fakeEmbedAccel: &fakeEmbedAccel{name: "test-b"}, memory: DeviceMemory{AvailableBytes: 7, TotalBytes: 9, Basis: MemoryEstimated}}
	for _, a := range []*memoryTestAccelerator{a, b} {
		registerTempAccelerator(t, a.Name(), a)
	}
	if _, err := e.UseAccelerator("test-b"); err != nil {
		t.Fatal(err)
	}
	reporter, ok := e.Accelerator().(DeviceMemoryReporter)
	if !ok {
		t.Fatal("attached accelerator lost its optional reporter")
	}
	for _, available := range []int64{7, 4, 0} {
		b.memory.AvailableBytes = available
		if got, err := reporter.DeviceMemory(); err != nil || got != b.memory {
			t.Fatalf("attached memory = %+v, %v, want %+v", got, err, b.memory)
		}
	}
	if a.calls != 0 || b.calls != 3 {
		t.Fatalf("memory reads routed to wrong accelerator: first=%d attached=%d", a.calls, b.calls)
	}

	const plain = "test-no-memory"
	registerTempAccelerator(t, plain, &fakeEmbedAccel{name: plain})
	if _, err := newTinyEncoderForTest(t).UseAccelerator(plain); err != nil {
		t.Fatal(err)
	}
}
