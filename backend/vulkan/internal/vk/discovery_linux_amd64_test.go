package vk

import (
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
	"unsafe"
)

type fakeAdapter struct {
	kind, vendor, version uint32
	uuid                  [16]byte
	compute               bool
	extension             bool
	memory                memoryProperties
	budget                memoryBudget
}

type fakeDriver struct {
	order                       []handle
	adapters                    map[handle]*fakeAdapter
	createInstance, enumerate   result
	createDevice, listExtension result
	created, destroyed          int
	selected, sampled           []handle
}

func (f *fakeDriver) install(t *testing.T, instance11 bool) *Device {
	t.Helper()
	d := &Device{}
	if instance11 {
		d.enumerateVersion = func(out *uint32) result { *out = apiVersion11; return success }
	}
	d.api.createInstance = func(_ *instanceInfo, _ unsafe.Pointer, out *handle) result {
		if f.createInstance == success {
			*out = 101
			f.created++
		}
		return f.createInstance
	}
	d.api.destroyInstance = func(handle, unsafe.Pointer) { f.destroyed++ }
	d.api.enumeratePhysical = func(_ handle, count *uint32, out *handle) result {
		if f.enumerate != success {
			return f.enumerate
		}
		if out != nil {
			copy(unsafe.Slice(out, *count), f.order)
		}
		*count = uint32(len(f.order))
		return success
	}
	d.api.physicalProperties = func(h handle, out *physicalProperties) {
		card := f.adapters[h]
		out.Kind, out.Vendor, out.Device, out.API = card.kind, card.vendor, 0x7590, card.version
	}
	d.properties2 = func(h handle, out *physicalProperties2) {
		(*physicalIDProperties)(out.Next).DeviceUUID = f.adapters[h].uuid
	}
	d.api.queueFamilies = func(h handle, count *uint32, out *queueFamily) {
		*count = 1
		if out != nil && f.adapters[h].compute {
			out.Flags, out.Count = queueCompute, 1
		}
	}
	d.api.memoryProperties = func(h handle, out *memoryProperties) { *out = f.adapters[h].memory }
	d.api.enumerateExtensions = func(h handle, layer *byte, count *uint32, out *extensionProperties) result {
		if layer != nil {
			t.Error("unexpected layer-specific extension query")
		}
		if f.listExtension != success {
			return f.listExtension
		}
		if !f.adapters[h].extension {
			*count = 0
			return success
		}
		*count = 1
		if out != nil {
			copy(out.Name[:], "VK_EXT_memory_budget")
		}
		return success
	}
	d.memoryProperties2 = func(h handle, out *memoryProperties2) {
		budget := (*memoryBudget)(out.Next)
		if out.Type != stMemoryProperties2 || budget.Type != stMemoryBudget {
			t.Error("incorrect budget query chain")
		}
		f.sampled = append(f.sampled, h)
		budget.Budget, budget.Usage = f.adapters[h].budget.Budget, f.adapters[h].budget.Usage
	}
	d.api.createDevice = func(h handle, _ *deviceInfo, _ unsafe.Pointer, out *handle) result {
		f.selected = append(f.selected, h)
		if f.createDevice == success {
			*out = 102
		}
		return f.createDevice
	}
	d.api.destroyDevice = func(handle, unsafe.Pointer) {}
	d.api.getQueue = func(_ handle, _, _ uint32, out *handle) { *out = 103 }
	d.api.createCmdPool = func(_ handle, _ *cmdPoolInfo, _ unsafe.Pointer, out *handle) result { *out = 104; return success }
	d.api.destroyCmdPool = func(handle, handle, unsafe.Pointer) {}
	d.api.allocateCmds = func(_ handle, _ *cmdAlloc, out *handle) result { *out = 105; return success }
	d.api.createDescPool = func(_ handle, _ *descPoolInfo, _ unsafe.Pointer, out *handle) result { *out = 106; return success }
	d.api.destroyDescPool = func(handle, handle, unsafe.Pointer) {}
	d.api.createFence = func(_ handle, _ *fenceInfo, _ unsafe.Pointer, out *handle) result { *out = 107; return success }
	d.api.destroyFence = func(handle, handle, unsafe.Pointer) {}
	prev := loadLibrary
	loadLibrary = func() (*Device, error) { return d, nil }
	t.Cleanup(func() { loadLibrary = prev })
	return d
}

func testDriver() *fakeDriver {
	amd := func(id byte) *fakeAdapter {
		return &fakeAdapter{
			kind: deviceDiscrete, vendor: amdVendor, version: apiVersion11,
			uuid: [16]byte{id}, compute: true, extension: true,
			memory: memoryProperties{Count: 1, HeapCount: 3, Heaps: [16]memoryHeap{{100, heapDeviceLocal}, {200, 0}, {300, heapDeviceLocal}}},
			budget: memoryBudget{Budget: [16]uint64{85, 180, 270}, Usage: [16]uint64{15, 0, 40}},
		}
	}
	nvidia, integrated, cpu, noCompute := amd(3), amd(4), amd(5), amd(6)
	nvidia.vendor, integrated.kind, cpu.kind, noCompute.compute = 0x10de, 1, 4, false
	return &fakeDriver{
		order:    []handle{1, 2, 3, 4, 5, 6},
		adapters: map[handle]*fakeAdapter{1: amd(1), 2: amd(2), 3: nvidia, 4: integrated, 5: cpu, 6: noCompute},
	}
}

func TestVulkanStartupErrors(t *testing.T) {
	for _, tc := range []struct {
		name                string
		instance, enumerate result
		category            error
		code                int32
	}{
		{"incompatible", -9, 0, ErrRuntimeUnavailable, -9},
		{"init", -3, 0, ErrRuntimeUnavailable, -3},
		{"host memory", outOfHost, 0, ErrOutOfMemory, -1},
		{"device memory", outOfDevice, 0, ErrOutOfMemory, -2},
		{"enumerate memory", 0, outOfHost, ErrOutOfMemory, -1},
		{"enumerate init", 0, -3, ErrRuntimeUnavailable, -3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := testDriver()
			f.createInstance, f.enumerate = tc.instance, tc.enumerate
			f.install(t, true)
			_, err := Devices(false)
			var native *ResultError
			if !errors.Is(err, tc.category) || !errors.As(err, &native) || native.Code != tc.code ||
				errors.Is(err, ErrOutOfMemory) == errors.Is(err, ErrRuntimeUnavailable) ||
				!strings.Contains(err.Error(), "VK_ERROR_") || f.created != f.destroyed {
				t.Fatalf("error=%v, native=%+v, instance creates=%d destroys=%d", err, native, f.created, f.destroyed)
			}
		})
	}
	t.Run("loader", func(t *testing.T) {
		f := testDriver()
		f.install(t, true)
		loaderErr := errors.New("loader missing")
		loadLibrary = func() (*Device, error) { return nil, loaderErr }
		_, err := Open(Selector{})
		if !errors.Is(err, ErrRuntimeUnavailable) || !errors.Is(err, loaderErr) {
			t.Fatal(err)
		}
	})
}

func TestVulkanInventoryAndBudget(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		instance11, device11 bool
		include, extension   bool
		budgets, usage       [16]uint64
		listErr              result
		want                 uint64
		known                bool
	}{
		{"default", true, true, false, true, [16]uint64{85, 180, 270}, [16]uint64{15, 0, 40}, 0, 0, false},
		{"mixed heaps", true, true, true, true, [16]uint64{85, 180, 270}, [16]uint64{15, 0, 40}, 0, 300, true},
		{"zero budget", true, true, true, true, [16]uint64{85, 180, 0}, [16]uint64{}, 0, 0, false},
		{"below usage", true, true, true, true, [16]uint64{1, 0, 1}, [16]uint64{9, 0, 9}, 0, 0, true},
		{"clamped", true, true, true, true, [16]uint64{math.MaxUint64, 0, 301}, [16]uint64{1, 0, 23}, 0, 376, true},
		{"missing extension", true, true, true, false, [16]uint64{85, 180, 270}, [16]uint64{}, 0, 0, false},
		{"old instance", false, true, true, true, [16]uint64{85, 180, 270}, [16]uint64{}, 0, 0, false},
		{"old device", true, false, true, true, [16]uint64{85, 180, 270}, [16]uint64{}, 0, 0, false},
		{"query failure", true, true, true, true, [16]uint64{}, [16]uint64{}, -3, 0, false},
		{"query memory failure", true, true, true, true, [16]uint64{}, [16]uint64{}, outOfDevice, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := testDriver()
			f.listExtension = tc.listErr
			for _, h := range []handle{1, 2} {
				card := f.adapters[h]
				if !tc.device11 {
					card.version = 1 << 22
				}
				card.extension = tc.extension
				card.budget.Budget, card.budget.Usage = tc.budgets, tc.usage
			}
			for range 3 {
				f.install(t, tc.instance11)
				got, err := Devices(tc.include)
				if tc.listErr == outOfDevice {
					if !errors.Is(err, ErrOutOfMemory) || errors.Is(err, ErrRuntimeUnavailable) {
						t.Fatal(err)
					}
				} else {
					if err != nil || len(got) != 2 {
						t.Fatalf("inventory %v: %v", got, err)
					}
					for _, info := range got {
						if info.BudgetKnown != tc.known || info.AvailableLocalBytes != tc.want || info.TotalLocalBytes != 400 {
							t.Fatalf("unexpected memory: %+v", info)
						}
					}
				}
				slices.Reverse(f.order)
			}
			if f.created != 3 || f.destroyed != 3 || len(f.selected) != 0 {
				t.Fatalf("probe lifetime creates=%d destroys=%d logical=%v", f.created, f.destroyed, f.selected)
			}
		})
	}
}

func TestVulkanCompleteInventory(t *testing.T) {
	for _, complete := range []bool{false, true} {
		t.Run(map[bool]string{false: "keeps changing", true: "grows"}[complete], func(t *testing.T) {
			f := testDriver()
			d := f.install(t, true)
			counts, lists := 0, 0
			d.api.enumeratePhysical = func(_ handle, n *uint32, out *handle) result {
				if out == nil {
					counts++
					*n = 1
					if complete && lists > 0 {
						*n = 2
					}
					return success
				}
				lists++
				if complete && lists > 1 {
					copy(unsafe.Slice(out, *n), []handle{1, 2})
					return success
				}
				*out = 1
				return incomplete
			}
			_, err := Open(Selector{})
			if complete {
				if !errors.Is(err, ErrAmbiguousAdapter) || counts != 2 || lists != 2 {
					t.Fatalf("should discover both adapters on retry: %v (%d,%d)", err, counts, lists)
				}
			} else {
				var native *ResultError
				if !errors.Is(err, ErrRuntimeUnavailable) || !errors.As(err, &native) || native.Code != int32(incomplete) || counts != 3 || lists != 3 {
					t.Fatalf("want incomplete diagnostic: %v (%d,%d)", err, counts, lists)
				}
			}
			if len(f.selected) != 0 || f.created != f.destroyed {
				t.Fatalf("partial inventory attached a device or leaked the instance: %+v", f)
			}
		})
	}
}

func TestVulkanInventoryWithoutEligibleCards(t *testing.T) {
	f := testDriver()
	f.order = []handle{3, 4, 5, 6}
	f.install(t, true)
	found, err := Devices(false)
	if err != nil || len(found) != 0 || f.created != 1 || f.destroyed != 1 || len(f.selected) != 0 {
		t.Fatalf("empty inventory %v: %v; lifecycle %+v", found, err, f)
	}
}

func TestVulkanAttachedBudget(t *testing.T) {
	for _, order := range [][]handle{{1, 2}, {2, 1}} {
		f := testDriver()
		f.order = order
		f.install(t, true)
		d, err := Open(Selector{UUID: [16]byte{2}})
		if err != nil {
			t.Fatal(err)
		}
		if info := d.Info(); info.BudgetKnown || info.AvailableLocalBytes != 0 {
			t.Fatalf("attach should leave the identity budget unqueried: %+v", info)
		}
		for _, used := range []uint64{15, 17} {
			f.adapters[2].budget.Usage[0] = used
			total, free, known, err := d.Memory()
			if err != nil || !known || total != 400 || free != 315-used {
				t.Fatalf("attached memory: total %d, available %d, known %t, err %v", total, free, known, err)
			}
		}
		must(t, d.Close())
		if _, _, _, err := d.Memory(); err == nil {
			t.Fatal("closed device reported memory")
		}
		if f.created != 1 || f.destroyed != 1 || !reflect.DeepEqual(f.selected, []handle{2}) ||
			!reflect.DeepEqual(f.sampled, []handle{2, 2}) {
			t.Fatalf("unexpected logical and physical activity: %+v", f)
		}
	}
}

func TestVulkanSelectorAndDeviceCreationFailures(t *testing.T) {
	for _, tc := range []struct {
		name      string
		order     []handle
		selector  Selector
		duplicate bool
		fail      result
		want      error
	}{
		{"no adapters", []handle{3, 4, 5, 6}, Selector{}, false, 0, ErrNoAdapter},
		{"missing UUID", []handle{1, 2}, Selector{UUID: [16]byte{9}}, false, 0, ErrNoAdapter},
		{"ambiguous", []handle{1, 2}, Selector{}, false, 0, ErrAmbiguousAdapter},
		{"duplicate UUID", []handle{1, 2}, Selector{UUID: [16]byte{1}}, true, 0, ErrAmbiguousAdapter},
		{"device allocation", []handle{1}, Selector{}, false, outOfDevice, ErrOutOfMemory},
		{"device init", []handle{1}, Selector{}, false, -3, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := testDriver()
			f.order, f.createDevice = tc.order, tc.fail
			if tc.duplicate {
				f.adapters[2].uuid = f.adapters[1].uuid
			}
			f.install(t, true)
			_, err := Open(tc.selector)
			if err == nil || tc.want != nil && !errors.Is(err, tc.want) || errors.Is(err, ErrRuntimeUnavailable) ||
				(tc.fail == success) != (len(f.selected) == 0) || f.created != f.destroyed {
				t.Fatalf("error=%v, want %v; lifecycle %+v", err, tc.want, f)
			}
			if tc.fail != success {
				var native *ResultError
				if !errors.As(err, &native) || native.Code != int32(tc.fail) || errors.Is(err, ErrOutOfMemory) != (tc.fail == outOfDevice) {
					t.Fatalf("native device error lost: %v", err)
				}
			}
		})
	}
}
