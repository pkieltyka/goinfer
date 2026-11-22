package vulkan

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestVulkanAliasesAndExactUUIDs(t *testing.T) {
	a, b := vk.Info{UUID: [16]byte{0x11}}, vk.Info{UUID: [16]byte{0xab}}
	input := []vk.Info{b, {}, a}
	want := describeDevices(input)
	slices.Reverse(input)
	if again := describeDevices(input); !slices.Equal(want, again) {
		t.Fatalf("enumeration order changed aliases: %v vs %v", want, again)
	}
	if want[0].ID != "" || want[0].Alias != "" || want[1].Alias != "vulkan:1" || want[2].Alias != "vulkan:2" {
		t.Fatalf("unexpected UUID-less or ordinal identity: %+v", want)
	}
	for _, tc := range []struct {
		id        string
		want      [16]byte
		inventory bool
		err       error
	}{
		{"1", a.UUID, true, nil},
		{"2", b.UUID, true, nil},
		{"01", [16]byte{}, true, embeddinggemma2.ErrNoDevice},
		{"0", [16]byte{}, true, embeddinggemma2.ErrNoDevice},
		{"7", [16]byte{}, true, embeddinggemma2.ErrNoDevice},
		{strings.TrimPrefix(want[1].ID, "vulkan:"), a.UUID, false, nil},
		{"99000000000000000000000000000000", [16]byte{0x99}, false, nil},
		{strings.ToUpper(strings.TrimPrefix(want[2].ID, "vulkan:")), b.UUID, false, nil},
		{"00000000000000000000000000000000", [16]byte{}, false, embeddinggemma2.ErrDeviceUnavailable},
		{"zz", [16]byte{}, false, embeddinggemma2.ErrDeviceUnavailable},
		{"", [16]byte{}, false, embeddinggemma2.ErrDeviceUnavailable},
	} {
		t.Run(tc.id, func(t *testing.T) {
			calls := 0
			got, err := resolveSelector(tc.id, func() ([]DeviceInfo, error) {
				calls++
				return want, nil
			})
			if got.UUID != tc.want || !errors.Is(err, tc.err) || (calls == 1) != tc.inventory {
				t.Fatalf("selector=%+v err=%v, alias lookups=%d", got, err, calls)
			}
			if tc.err == embeddinggemma2.ErrDeviceUnavailable && errors.Is(err, embeddinggemma2.ErrNoDevice) {
				t.Fatalf("malformed selector reported as absent: %v", err)
			}
		})
	}
}

func TestVulkanClassifiesNativeErrors(t *testing.T) {
	for _, tc := range []struct{ cause, want error }{
		{vk.ErrRuntimeUnavailable, embeddinggemma2.ErrBackendUnavailable},
		{vk.ErrNoAdapter, embeddinggemma2.ErrNoDevice},
		{vk.ErrAmbiguousAdapter, embeddinggemma2.ErrAmbiguousDevice},
		{&vk.ResultError{Op: "allocate", Code: -2}, embeddinggemma2.ErrDeviceOutOfMemory},
	} {
		got := classify(tc.cause)
		if !errors.Is(got, tc.want) || !errors.Is(got, tc.cause) ||
			errors.Is(got, embeddinggemma2.ErrDeviceUnavailable) != (tc.want != embeddinggemma2.ErrDeviceOutOfMemory) {
			t.Fatalf("classify(%v) = %v, wanted %v", tc.cause, got, tc.want)
		}
	}
	unrelated := &vk.ResultError{Op: "create queue", Code: -3}
	if classify(unrelated) != unrelated {
		t.Fatal("unclassified error was changed")
	}
}

func TestVulkanDeviceLossPreservesNativeCause(t *testing.T) {
	cause := &vk.ResultError{Op: "submit command buffer", Code: -4}
	err := classify(cause)
	var typed *embeddinggemma2.DeviceLostError
	var native *vk.ResultError
	if !errors.Is(err, embeddinggemma2.ErrDeviceLost) || !errors.As(err, &typed) ||
		!errors.As(err, &native) || native != cause || typed.Unwrap() != cause ||
		errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) {
		t.Fatalf("wrong classification for %v (typed %v, native %v)", err, typed, native)
	}
}

func TestVulkanDeviceMemoryConversion(t *testing.T) {
	known := deviceMemory(math.MaxUint64, math.MaxUint64, true)
	if known != (embeddinggemma2.DeviceMemory{TotalBytes: math.MaxInt64, AvailableBytes: math.MaxInt64, Basis: embeddinggemma2.MemoryEstimated}) {
		t.Fatalf("unclamped snapshot %+v", known)
	}
	unknown := deviceMemory(7, 12, false)
	if unknown != (embeddinggemma2.DeviceMemory{TotalBytes: 7}) {
		t.Fatalf("unexpected unknown snapshot %+v", unknown)
	}
	var iface embeddinggemma2.DeviceMemoryReporter = &encoder{}
	if mem, err := iface.DeviceMemory(); err == nil || mem != (embeddinggemma2.DeviceMemory{}) {
		t.Fatalf("closed encoder returned memory %v, %v", mem, err)
	}
}
