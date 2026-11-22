package vk

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
)

type result int32

const (
	success     result = 0
	incomplete  result = 5
	outOfHost   result = -1
	outOfDevice result = -2
	deviceLost  result = -4
)

var (
	ErrRuntimeUnavailable = errors.New("vulkan: runtime unavailable")
	ErrNoAdapter          = errors.New("vulkan: no discrete AMD adapter matches the selection")
	ErrAmbiguousAdapter   = errors.New("vulkan: several discrete AMD adapters match the selection")
	ErrOutOfMemory        = errors.New("vulkan: native allocation failed")
)

type ResultError struct {
	Op   string
	Code int32
}

func (e *ResultError) Error() string {
	names := [...]string{
		"VK_ERROR_OUT_OF_HOST_MEMORY",
		"VK_ERROR_OUT_OF_DEVICE_MEMORY",
		"VK_ERROR_INITIALIZATION_FAILED",
		"VK_ERROR_DEVICE_LOST",
		"VK_ERROR_MEMORY_MAP_FAILED",
		"VK_ERROR_LAYER_NOT_PRESENT",
		"VK_ERROR_EXTENSION_NOT_PRESENT",
		"VK_ERROR_FEATURE_NOT_PRESENT",
		"VK_ERROR_INCOMPATIBLE_DRIVER",
		"VK_ERROR_TOO_MANY_OBJECTS",
		"VK_ERROR_FORMAT_NOT_SUPPORTED",
		"VK_ERROR_FRAGMENTED_POOL",
		"VK_ERROR_UNKNOWN",
	}
	switch {
	case e.Code <= -1 && e.Code >= -int32(len(names)):
		return fmt.Sprintf("vulkan: %s: %s (%d)", e.Op, names[-e.Code-1], e.Code)
	case result(e.Code) == incomplete:
		return fmt.Sprintf("vulkan: %s: VK_INCOMPLETE (%d)", e.Op, e.Code)
	default:
		return fmt.Sprintf("vulkan: %s: VkResult %d", e.Op, e.Code)
	}
}

func (e *ResultError) Is(target error) bool {
	return target == ErrOutOfMemory && (result(e.Code) == outOfHost || result(e.Code) == outOfDevice)
}

func (r result) check(op string) error {
	if r == success {
		return nil
	}
	return &ResultError{Op: op, Code: int32(r)}
}

// Object creation can leave the output undefined on failure.
func createHandle[T any](owner handle, info *T, op string, create func(handle, *T, unsafe.Pointer, *handle) result) (handle, error) {
	var h handle
	if err := create(owner, info, nil, &h).check(op); err != nil {
		return 0, err
	}
	return h, nil
}

type api struct {
	createInstance      func(*instanceInfo, unsafe.Pointer, *handle) result
	destroyInstance     func(handle, unsafe.Pointer)
	enumeratePhysical   func(handle, *uint32, *handle) result
	enumerateExtensions func(handle, *byte, *uint32, *extensionProperties) result
	physicalProperties  func(handle, *physicalProperties)
	queueFamilies       func(handle, *uint32, *queueFamily)
	memoryProperties    func(handle, *memoryProperties)
	createDevice        func(handle, *deviceInfo, unsafe.Pointer, *handle) result
	destroyDevice       func(handle, unsafe.Pointer)
	getQueue            func(handle, uint32, uint32, *handle)
	deviceWaitIdle      func(handle) result
	createBuffer        func(handle, *bufferInfo, unsafe.Pointer, *handle) result
	destroyBuffer       func(handle, handle, unsafe.Pointer)
	bufferRequirements  func(handle, handle, *memoryRequirements)
	allocateMemory      func(handle, *memoryInfo, unsafe.Pointer, *handle) result
	freeMemory          func(handle, handle, unsafe.Pointer)
	bindMemory          func(handle, handle, handle, uint64) result
	mapMemory           func(handle, handle, uint64, uint64, uint32, *unsafe.Pointer) result
	unmapMemory         func(handle, handle)
	createShader        func(handle, *shaderInfo, unsafe.Pointer, *handle) result
	destroyShader       func(handle, handle, unsafe.Pointer)
	createSetLayout     func(handle, *setLayoutInfo, unsafe.Pointer, *handle) result
	destroySetLayout    func(handle, handle, unsafe.Pointer)
	createLayout        func(handle, *layoutInfo, unsafe.Pointer, *handle) result
	destroyLayout       func(handle, handle, unsafe.Pointer)
	createPipelines     func(handle, handle, uint32, *pipelineInfo, unsafe.Pointer, *handle) result
	destroyPipeline     func(handle, handle, unsafe.Pointer)
	createDescPool      func(handle, *descPoolInfo, unsafe.Pointer, *handle) result
	destroyDescPool     func(handle, handle, unsafe.Pointer)
	resetDescPool       func(handle, handle, uint32) result
	allocateSets        func(handle, *setInfo, *handle) result
	updateSets          func(handle, uint32, *writeInfo, uint32, unsafe.Pointer)
	createCmdPool       func(handle, *cmdPoolInfo, unsafe.Pointer, *handle) result
	destroyCmdPool      func(handle, handle, unsafe.Pointer)
	resetCmdPool        func(handle, handle, uint32) result
	allocateCmds        func(handle, *cmdAlloc, *handle) result
	beginCmd            func(handle, *cmdBegin) result
	endCmd              func(handle) result
	createFence         func(handle, *fenceInfo, unsafe.Pointer, *handle) result
	destroyFence        func(handle, handle, unsafe.Pointer)
	resetFences         func(handle, uint32, *handle) result
	waitFences          func(handle, uint32, *handle, uint32, uint64) result
	bindPipeline        func(handle, uint32, handle)
	bindSets            func(handle, uint32, handle, uint32, uint32, *handle, uint32, *uint32)
	pushConstants       func(handle, handle, uint32, uint32, uint32, *uint32)
	dispatch            func(handle, uint32, uint32, uint32)
	copyBuffer          func(handle, handle, handle, uint32, *bufferCopy)
	barrier             func(handle, uint32, uint32, uint32, uint32, *memoryBarrier, uint32, unsafe.Pointer, uint32, unsafe.Pointer)
	queueSubmit         func(handle, uint32, *submitInfo, handle) result
}

func (a *api) load(lib uintptr) error {
	for _, fn := range []struct {
		name string
		slot any
	}{
		{"CreateInstance", &a.createInstance},
		{"DestroyInstance", &a.destroyInstance},
		{"EnumeratePhysicalDevices", &a.enumeratePhysical},
		{"EnumerateDeviceExtensionProperties", &a.enumerateExtensions},
		{"GetPhysicalDeviceProperties", &a.physicalProperties},
		{"GetPhysicalDeviceQueueFamilyProperties", &a.queueFamilies},
		{"GetPhysicalDeviceMemoryProperties", &a.memoryProperties},
		{"CreateDevice", &a.createDevice},
		{"DestroyDevice", &a.destroyDevice},
		{"GetDeviceQueue", &a.getQueue},
		{"DeviceWaitIdle", &a.deviceWaitIdle},
		{"CreateBuffer", &a.createBuffer},
		{"DestroyBuffer", &a.destroyBuffer},
		{"GetBufferMemoryRequirements", &a.bufferRequirements},
		{"AllocateMemory", &a.allocateMemory},
		{"FreeMemory", &a.freeMemory},
		{"BindBufferMemory", &a.bindMemory},
		{"MapMemory", &a.mapMemory},
		{"UnmapMemory", &a.unmapMemory},
		{"CreateShaderModule", &a.createShader},
		{"DestroyShaderModule", &a.destroyShader},
		{"CreateDescriptorSetLayout", &a.createSetLayout},
		{"DestroyDescriptorSetLayout", &a.destroySetLayout},
		{"CreatePipelineLayout", &a.createLayout},
		{"DestroyPipelineLayout", &a.destroyLayout},
		{"CreateComputePipelines", &a.createPipelines},
		{"DestroyPipeline", &a.destroyPipeline},
		{"CreateDescriptorPool", &a.createDescPool},
		{"DestroyDescriptorPool", &a.destroyDescPool},
		{"ResetDescriptorPool", &a.resetDescPool},
		{"AllocateDescriptorSets", &a.allocateSets},
		{"UpdateDescriptorSets", &a.updateSets},
		{"CreateCommandPool", &a.createCmdPool},
		{"DestroyCommandPool", &a.destroyCmdPool},
		{"ResetCommandPool", &a.resetCmdPool},
		{"AllocateCommandBuffers", &a.allocateCmds},
		{"BeginCommandBuffer", &a.beginCmd},
		{"EndCommandBuffer", &a.endCmd},
		{"CreateFence", &a.createFence},
		{"DestroyFence", &a.destroyFence},
		{"ResetFences", &a.resetFences},
		{"WaitForFences", &a.waitFences},
		{"CmdBindPipeline", &a.bindPipeline},
		{"CmdBindDescriptorSets", &a.bindSets},
		{"CmdPushConstants", &a.pushConstants},
		{"CmdDispatch", &a.dispatch},
		{"CmdCopyBuffer", &a.copyBuffer},
		{"CmdPipelineBarrier", &a.barrier},
		{"QueueSubmit", &a.queueSubmit},
	} {
		name := "vk" + fn.name
		addr, err := purego.Dlsym(lib, name)
		if err != nil {
			return fmt.Errorf("vulkan: load %s: %w", name, err)
		}
		purego.RegisterFunc(fn.slot, addr)
	}
	return nil
}
