package vk

import (
	_ "embed"
	"errors"
	"os"
	"slices"
	"strconv"
	"testing"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.0 testdata/step.comp -o testdata/step.spv
//go:embed testdata/step.spv
var stepSPIRV []byte

// Layout values are from the Vulkan C headers on Linux x86_64.
func TestNativeABI(t *testing.T) {
	for _, tc := range []struct {
		name      string
		got, want uintptr
	}{
		{"VkInstanceCreateInfo", unsafe.Sizeof(instanceInfo{}), 64},
		{"VkDeviceQueueCreateInfo", unsafe.Sizeof(queueInfo{}), 40},
		{"VkDeviceCreateInfo", unsafe.Sizeof(deviceInfo{}), 72},
		{"VkPhysicalDeviceProperties", unsafe.Sizeof(physicalProperties{}), 824},
		{"VkPhysicalDeviceProperties.deviceName", unsafe.Offsetof(physicalProperties{}.Name), 20},
		{"VkPhysicalDeviceProperties.limits", unsafe.Offsetof(physicalProperties{}.Limits), 296},
		{"VkPhysicalDeviceProperties.sparseProperties", unsafe.Offsetof(physicalProperties{}.Sparse), 800},
		{"VkPhysicalDeviceLimits", unsafe.Sizeof(physicalLimits{}), 504},
		{"VkPhysicalDeviceLimits.maxStorageBufferRange", unsafe.Offsetof(physicalLimits{}.MaxStorageBufferRange), 28},
		{"VkPhysicalDeviceLimits.maxPushConstantsSize", unsafe.Offsetof(physicalLimits{}.MaxPushConstantsSize), 32},
		{"VkPhysicalDeviceLimits.maxMemoryAllocationCount", unsafe.Offsetof(physicalLimits{}.MaxMemoryAllocationCount), 36},
		{"VkPhysicalDeviceLimits.maxPerStageDescriptorStorageBuffers", unsafe.Offsetof(physicalLimits{}.MaxPerStageStorageBuffers), 76},
		{"VkPhysicalDeviceLimits.maxDescriptorSetStorageBuffers", unsafe.Offsetof(physicalLimits{}.MaxDescriptorSetStorageBuffers), 108},
		{"VkPhysicalDeviceLimits.maxComputeWorkGroupCount", unsafe.Offsetof(physicalLimits{}.MaxWorkGroupCount), 220},
		{"VkQueueFamilyProperties", unsafe.Sizeof(queueFamily{}), 24},
		{"VkPhysicalDeviceMemoryProperties", unsafe.Sizeof(memoryProperties{}), 520},
		{"VkBufferCreateInfo", unsafe.Sizeof(bufferInfo{}), 56},
		{"VkMemoryRequirements", unsafe.Sizeof(memoryRequirements{}), 24},
		{"VkMemoryAllocateInfo", unsafe.Sizeof(memoryInfo{}), 32},
		{"VkShaderModuleCreateInfo", unsafe.Sizeof(shaderInfo{}), 40},
		{"VkDescriptorSetLayoutBinding", unsafe.Sizeof(binding{}), 24},
		{"VkDescriptorSetLayoutCreateInfo", unsafe.Sizeof(setLayoutInfo{}), 32},
		{"VkPipelineLayoutCreateInfo", unsafe.Sizeof(layoutInfo{}), 48},
		{"VkPipelineLayoutCreateInfo.pPushConstantRanges", unsafe.Offsetof(layoutInfo{}.Ranges), 40},
		{"VkPushConstantRange", unsafe.Sizeof(pushRange{}), 12},
		{"VkPipelineShaderStageCreateInfo", unsafe.Sizeof(stageInfo{}), 48},
		{"VkComputePipelineCreateInfo", unsafe.Sizeof(pipelineInfo{}), 96},
		{"VkDescriptorPoolSize", unsafe.Sizeof(poolSize{}), 8},
		{"VkDescriptorPoolCreateInfo", unsafe.Sizeof(descPoolInfo{}), 40},
		{"VkDescriptorSetAllocateInfo", unsafe.Sizeof(setInfo{}), 40},
		{"VkDescriptorBufferInfo", unsafe.Sizeof(descriptor{}), 24},
		{"VkWriteDescriptorSet", unsafe.Sizeof(writeInfo{}), 64},
		{"VkWriteDescriptorSet.pBufferInfo", unsafe.Offsetof(writeInfo{}.Buffers), 48},
		{"VkCommandPoolCreateInfo", unsafe.Sizeof(cmdPoolInfo{}), 24},
		{"VkCommandBufferAllocateInfo", unsafe.Sizeof(cmdAlloc{}), 32},
		{"VkCommandBufferBeginInfo", unsafe.Sizeof(cmdBegin{}), 32},
		{"VkSubmitInfo", unsafe.Sizeof(submitInfo{}), 72},
		{"VkFenceCreateInfo", unsafe.Sizeof(fenceInfo{}), 24},
		{"VkMemoryBarrier", unsafe.Sizeof(memoryBarrier{}), 24},
		{"VkBufferCopy", unsafe.Sizeof(bufferCopy{}), 24},
	} {
		if tc.got != tc.want {
			t.Errorf("%s: got %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}

func TestSPIRVAndRanges(t *testing.T) {
	code, err := spirvWords(stepSPIRV)
	if err != nil || len(code)*4 != len(stepSPIRV) || code[1] != 0x00010000 {
		t.Fatalf("invalid SPIR-V 1.0 sentinel: %v", err)
	}
	for _, invalid := range [][]byte{nil, stepSPIRV[:19], stepSPIRV[:21], make([]byte, 20)} {
		if _, err := spirvWords(invalid); err == nil {
			t.Error("accepted invalid SPIR-V header/size")
		}
	}
	if !copyRange(8, 4, 4) || copyRange(8, 0, 0) || copyRange(8, 4, 8) ||
		copyRange(8, 1, 4) || copyRange(8, 0, 6) || copyRange(8, ^uint64(0)-3, 8) {
		t.Error("accepted an invalid range or rejected a valid one")
	}
}

func TestFenceWaitFailurePreservesInFlightObjects(t *testing.T) {
	const unresolved result = -2
	for _, drainResult := range []result{success, deviceLost, unresolved} {
		d := &Device{device: 1, queue: 2, fence: 3, cmd: 4}
		d.api.resetCmdPool = func(handle, handle, uint32) result { return success }
		d.api.resetDescPool = func(handle, handle, uint32) result { return success }
		d.api.beginCmd = func(handle, *cmdBegin) result { return success }
		d.api.endCmd = func(handle) result { return success }
		d.api.resetFences = func(handle, uint32, *handle) result { return success }
		d.api.queueSubmit = func(handle, uint32, *submitInfo, handle) result { return success }
		d.api.waitFences = func(handle, uint32, *handle, uint32, uint64) result { return unresolved }
		d.api.deviceWaitIdle = func(handle) result { return drainResult }
		freed := 0
		d.api.destroyFence = func(handle, handle, unsafe.Pointer) { freed++ }
		d.api.destroyBuffer = d.api.destroyFence
		d.api.destroyPipeline = d.api.destroyFence
		d.api.destroyDevice = func(handle, unsafe.Pointer) { freed++ }

		if err := d.Run(func(*Commands) error { return nil }); err == nil || d.inFlight != (drainResult == unresolved) {
			t.Fatalf("drain result %d: incorrect fence failure state: inFlight=%t, err=%v", drainResult, d.inFlight, err)
		}
		if err := d.Run(func(*Commands) error { t.Error("device accepted another recorder"); return nil }); err == nil {
			t.Fatal("device remained usable after failed fence")
		}
		b := &Buffer{d: d, buffer: 5}
		p := &Pipeline{d: d, pipeline: 6}
		if drainResult == unresolved {
			if b.Close() == nil || p.Close() == nil || d.Close() == nil || freed != 0 {
				t.Fatal("destroyed objects while work may still be running")
			}
			d.api.deviceWaitIdle = func(handle) result { return success }
		}
		if err := errors.Join(b.Close(), p.Close(), d.Close()); err != nil || freed != 4 {
			t.Fatalf("drain result %d: cleanup freed %d handles: %v", drainResult, freed, err)
		}
	}
}

func TestDiscreteAMDGroupedCommands(t *testing.T) {
	pin := os.Getenv("GOINFER_TEST_VULKAN_DEVICE")
	if pin == "" {
		t.Skip("set GOINFER_TEST_VULKAN_DEVICE to the expected hexadecimal AMD PCI device ID")
	}
	id, err := strconv.ParseUint(pin, 16, 16)
	if err != nil || id == 0 {
		t.Fatalf("invalid GOINFER_TEST_VULKAN_DEVICE=%q", pin)
	}
	expect := uint32(id)
	d, err := Open(Selector{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		must(t, d.Close())
	})
	pinned, err := Open(Selector{DeviceID: expect})
	if err != nil {
		t.Fatal(err)
	}
	if info := pinned.Info(); info != d.Info() || info.VendorID != amdVendor || info.DeviceID != expect {
		t.Errorf("pinned %v and unpinned %v must select discrete AMD %04x", info, d.Info(), expect)
	}
	must(t, pinned.Close())
	if other, err := Open(Selector{DeviceID: 0x10000}); err == nil {
		must(t, other.Close())
		t.Fatal("nonexistent PCI device passed selection")
	}
	t.Logf("adapter %+v", d.Info())

	const n = 3
	in, err := d.NewBuffer(4*(n+1), HostVisible)
	must(t, err)
	out, err := d.NewBuffer(4*(n+1), HostVisible)
	must(t, err)
	first, err := d.NewBuffer(4*n, DeviceLocal)
	must(t, err)
	second, err := d.NewBuffer(4*n, DeviceLocal)
	must(t, err)
	p, err := d.NewPipeline(stepSPIRV, 2, 4)
	must(t, err)

	if err := first.WriteF32(0, []float32{1}); err == nil {
		t.Fatal("device-only buffer is mapped as host-visible")
	}
	if err := in.WriteF32(2, []float32{1}); err == nil {
		t.Fatal("accepted unaligned mapped write")
	}
	cancel := errors.New("discard recorded commands")
	var abandoned *Commands
	err = d.Run(func(c *Commands) error {
		abandoned = c
		if err := c.Copy(in, first, 4, 0, 4*n); err != nil {
			return err
		}
		return cancel
	})
	if !errors.Is(err, cancel) || d.Activity() != (Activity{}) {
		t.Fatalf("discarded recording submitted work: %v, %+v", err, d.Activity())
	}
	if err := abandoned.Copy(in, first, 4, 0, 4*n); err == nil {
		t.Fatal("recorder usable after callback returned")
	}

	for run, input := range [][]float32{{3.5, -5, 8}, {0, 0.25, -104}} {
		must(t, in.WriteF32(4, input))
		must(t, d.Run(func(c *Commands) error {
			if err := c.Dispatch(p, []*Buffer{first, second}, []uint32{3}, [3]uint32{0, 1, 1}); err == nil {
				return errors.New("accepted invalid grid")
			}
			if err := c.Copy(first, first, 0, 4, 8); err == nil {
				return errors.New("accepted overlapping copy")
			}
			if err := c.Copy(in, first, 4, 0, 4*n); err != nil {
				return err
			}
			if err := c.Dispatch(p, []*Buffer{first, second}, []uint32{3}, [3]uint32{n, 1, 1}); err != nil {
				return err
			}
			if err := c.Dispatch(p, []*Buffer{second, first}, []uint32{5}, [3]uint32{n, 1, 1}); err != nil {
				return err
			}
			return c.Copy(first, out, 0, 4, 4*n)
		}))
		var output [n]float32
		must(t, out.ReadF32(4, output[:]))
		for i := range input {
			input[i] += 8
		}
		if !slices.Equal(output[:], input) {
			t.Fatalf("group %d: readback %v, expected %v", run, output, input)
		}
		steps := uint64(run + 1)
		if got, want := d.Activity(), (Activity{2 * steps, 2 * steps, steps, steps}); got != want {
			t.Fatalf("group %d: activity %+v, expected %+v", run, got, want)
		}
		t.Logf("group %d readback %v, activity %+v", run, output, d.Activity())
	}
	must(t, first.Close())
	must(t, first.Close())
	must(t, p.Close())
	must(t, d.Close())
	must(t, out.Close())
	if err := d.Run(func(*Commands) error { return nil }); err == nil {
		t.Fatal("closed device accepted a submission")
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
