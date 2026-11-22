//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestResolveCUDADevice(t *testing.T) {
	const first = "GPU-abcd1234-1111-2222-3333-444444444444"
	const second = "GPU-abcd1234-1111-2222-3333-555555555555"
	const old = "GPU-abcd1234-1111-2222-3333-666666666666"
	adapters := []gpu.Adapter{
		{Ordinal: 0, UUID: first, Major: 7, Minor: 5},
		{Ordinal: 1, UUID: old, Major: 7},
		{Ordinal: 2, UUID: second, Major: 12},
		{Ordinal: 3, Major: 8},
	}
	for _, tt := range []struct {
		token   string
		ordinal int
		want    error
	}{
		{first, 0, nil}, {strings.ToUpper(second), 2, nil}, {"2", 2, nil},
		{old, 0, embeddinggemma2.ErrNoDevice}, {"1", 0, embeddinggemma2.ErrNoDevice},
		{"02", 0, embeddinggemma2.ErrNoDevice}, {"3", 0, embeddinggemma2.ErrNoDevice},
		{"", 0, embeddinggemma2.ErrDeviceUnavailable}, {"zz", 0, embeddinggemma2.ErrDeviceUnavailable},
		{"GPU-xyz", 0, embeddinggemma2.ErrDeviceUnavailable},
	} {
		got, err := resolveCUDADevice(tt.token, adapters)
		if !errors.Is(err, tt.want) || err == nil && got.Ordinal != tt.ordinal ||
			tt.want == embeddinggemma2.ErrDeviceUnavailable && errors.Is(err, embeddinggemma2.ErrNoDevice) {
			t.Errorf("resolve %q: got %+v, %v; want ordinal %d, %v", tt.token, got, err, tt.ordinal, tt.want)
		}
	}
	adapters[3].UUID = second
	for _, token := range []string{second, "2", "3"} {
		if _, err := resolveCUDADevice(token, adapters); !errors.Is(err, embeddinggemma2.ErrAmbiguousDevice) {
			t.Errorf("duplicate UUID, select %q: %v", token, err)
		}
	}
}

func TestCUDAAttachErrorCategories(t *testing.T) {
	invalidAlloc := gpu.AllocFailedPanic("weight", 4096, gc.ErrInvalidValue)
	textOOM := errors.New("OUT_OF_MEMORY without a native result")
	for _, tt := range []struct {
		native error
		want   error
	}{
		{fmt.Errorf("%w: %w", gpu.ErrDriverInit, gc.ErrOutOfMemory), embeddinggemma2.ErrDeviceOutOfMemory},
		{fmt.Errorf("%w: %w", gpu.ErrDriverInit, gc.ErrNoDevice), embeddinggemma2.ErrNoDevice},
		{fmt.Errorf("%w: %w", gpu.ErrDriverInit, gc.ErrStubLibrary), embeddinggemma2.ErrBackendUnavailable},
		{fmt.Errorf("%w: %w", gpu.ErrAdapterChanged, gc.ErrInvalidOrdinal), embeddinggemma2.ErrNoDevice},
		{fmt.Errorf("%w: %w", gpu.ErrAdapterChanged, gc.ErrInvalidDevice), embeddinggemma2.ErrNoDevice},
		{fmt.Errorf("%w: %w", gpu.ErrAdapterChanged, gc.ErrNoDevice), embeddinggemma2.ErrNoDevice},
		{gpu.AllocFailedPanic("weight", 4096, gc.ErrOutOfMemory), embeddinggemma2.ErrDeviceOutOfMemory},
		{invalidAlloc, invalidAlloc},
		{textOOM, textOOM},
		{gc.ErrInvalidDevice, gc.ErrInvalidDevice},
		{gc.ErrInvalidPTX, gc.ErrInvalidPTX},
	} {
		got := classifyCUDAAttach(tt.native)
		if !errors.Is(got, tt.want) || !errors.Is(got, tt.native) ||
			tt.want == embeddinggemma2.ErrDeviceOutOfMemory && errors.Is(got, embeddinggemma2.ErrDeviceUnavailable) {
			t.Errorf("classify %v: got %v, want %v preserving native cause", tt.native, got, tt.want)
		}
		if tt.native == tt.want && got != tt.native {
			t.Errorf("unclassified %v unexpectedly wrapped: %v", tt.native, got)
		}
	}
}

func TestCUDAEG2OpenerFailure(t *testing.T) {
	m := eg2TinyModel(t)
	for _, combined := range []bool{false, true} {
		calls := 0
		got, err := newEG2AccelOn(m, combined, func() (*Device, error) {
			calls++
			return nil, gc.ErrOutOfMemory
		})
		if got != nil || calls != 1 || !errors.Is(err, gc.ErrOutOfMemory) ||
			!errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) {
			t.Fatalf("combined %t: got %v, %v, %d open calls", combined, got, err, calls)
		}
	}
}

func TestCUDAEG2SelectedDevice(t *testing.T) {
	uuid := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if uuid == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID to the target CUDA device's GPU UUID")
	}
	adapters, err := gpu.Adapters()
	if err != nil {
		t.Fatal(err)
	}
	target, err := resolveCUDADevice(uuid, adapters)
	if err != nil {
		t.Fatal(err)
	}
	m := eg2TinyModel(t)
	ids := eg2TinyInputs(t, m.Config().VocabSize)[2]
	want, _, err := m.ForwardCPU(ids, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, selector := range embeddinggemma2.Accelerators() {
		if selector != "cuda" && selector != "cuda-eg2-combined" {
			continue
		}
		for _, token := range []string{uuid, strconv.Itoa(target.Ordinal)} {
			if !t.Run(selector+"/"+token, func(t *testing.T) {
				acc, err := m.NewAccelerator(selector + ":" + token)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := acc.Close(); err != nil {
						t.Error(err)
					}
				})
				a := acc.(*eg2Accel)
				native := a.ops.dev.Context().Device()
				gotUUID, err := native.UUID()
				if err != nil || !strings.EqualFold(gotUUID, uuid) || native.Ordinal() != target.Ordinal {
					t.Fatalf("attached UUID %q ordinal %d, error %v; want %q ordinal %d", gotUUID, native.Ordinal(), err, uuid, target.Ordinal)
				}
				result := make(chan error, 1)
				go func() {
					got, _, err := acc.Forward(ids, false)
					if err == nil && (len(got) != len(want) || !eg2Finite(got) || eg2Cos(got, want) < 0.9999) {
						err = fmt.Errorf("CUDA/CPU projected rows diverged")
					}
					result <- err
				}()
				reading, readErr := a.DeviceMemory()
				if err := <-result; err != nil {
					t.Fatal(err)
				}
				if readErr != nil || reading.Basis != embeddinggemma2.MemoryLive || reading.AvailableBytes < 0 ||
					reading.AvailableBytes > reading.TotalBytes || reading.TotalBytes <= 0 {
					t.Fatalf("memory %+v: %v", reading, readErr)
				}
				launches, _ := a.ops.dev.Activity()
				if launches == 0 {
					t.Fatal("no native CUDA launches")
				}
				t.Logf("%s PCI %s ordinal %d UUID %s: %d launches, memory %+v", target.Name, target.PCIBusID, target.Ordinal, gotUUID, launches, reading)
				if err := acc.Close(); err != nil {
					t.Fatal(err)
				}
				if _, err := a.DeviceMemory(); err == nil {
					t.Fatal("memory read succeeded after close")
				}
			}) {
				return
			}
		}
	}
	if _, err := m.NewAccelerator("cuda:GPU-00000000-0000-0000-0000-000000000000"); !errors.Is(err, embeddinggemma2.ErrNoDevice) {
		t.Fatalf("absent UUID: %v", err)
	}
}
