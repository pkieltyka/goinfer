package embeddinggemma2

import (
	"errors"
	"fmt"
	"testing"
)

func TestDeviceErrorCategories(t *testing.T) {
	categories := []error{ErrBackendUnavailable, ErrNoDevice, ErrAmbiguousDevice, ErrDeviceOutOfMemory}
	for i, err := range categories {
		wrapped := fmt.Errorf("attach: %w", err)
		if errors.Is(wrapped, ErrDeviceUnavailable) != (i != 3) {
			t.Errorf("%v has the wrong umbrella category", wrapped)
		}
		for j, other := range categories {
			if errors.Is(wrapped, other) != (i == j) {
				t.Errorf("errors.Is(%v, %v) = %t", err, other, i != j)
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
}

func (a *memoryTestAccelerator) DeviceMemory() (DeviceMemory, error) { return a.memory, nil }

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
	for _, available := range []int64{7, 4} {
		b.memory.AvailableBytes = available
		if got, err := reporter.DeviceMemory(); err != nil || got != b.memory {
			t.Fatalf("attached memory = %+v, %v, want %+v", got, err, b.memory)
		}
	}

	const plain = "test-no-memory"
	registerTempAccelerator(t, plain, &fakeEmbedAccel{name: plain})
	if _, err := newTinyEncoderForTest(t).UseAccelerator(plain); err != nil {
		t.Fatal(err)
	}
}
