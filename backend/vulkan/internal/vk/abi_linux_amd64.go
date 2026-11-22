package vk

import "unsafe"

// Vulkan dispatchable and non-dispatchable handles occupy one machine word
// on this platform. Go pointers in create infos live through the native call.
type handle uintptr

type header struct {
	Type uint32
	Next unsafe.Pointer
}

type instanceInfo struct {
	header
	Flags          uint32
	App            unsafe.Pointer
	Layers         uint32
	LayerNames     unsafe.Pointer
	Extensions     uint32
	ExtensionNames unsafe.Pointer
}

type queueInfo struct {
	header
	Flags, Family, Count uint32
	Priorities           *float32
}

type deviceInfo struct {
	header
	Flags, Count             uint32
	Queues                   *queueInfo
	Layers                   uint32
	LayerNames               unsafe.Pointer
	Extensions               uint32
	ExtensionNames, Features unsafe.Pointer
}

type bufferInfo struct {
	header
	Flags                 uint32
	Size                  uint64
	Usage, Sharing, Count uint32
	Families              *uint32
}

type memoryRequirements struct {
	Size, Alignment uint64
	Bits            uint32
}

type memoryInfo struct {
	header
	Size  uint64
	Index uint32
}

type memoryType struct{ Flags, Heap uint32 }

type memoryHeap struct {
	Size  uint64
	Flags uint32
}

type memoryProperties struct {
	Count     uint32
	Types     [32]memoryType
	HeapCount uint32
	Heaps     [16]memoryHeap
}

type queueFamily struct {
	Flags, Count, TimestampBits uint32
	Granularity                 [3]uint32
}

// Opaque spans retain the Vulkan C layout for fields the runtime does not use.
type physicalLimits struct {
	_                              [0]uint64
	_                              [28]byte
	MaxStorageBufferRange          uint32
	MaxPushConstantsSize           uint32
	MaxMemoryAllocationCount       uint32
	_                              [36]byte
	MaxPerStageStorageBuffers      uint32
	_                              [28]byte
	MaxDescriptorSetStorageBuffers uint32
	_                              [108]byte
	MaxWorkGroupCount              [3]uint32
	MaxWorkGroupInvocations        uint32
	MaxWorkGroupSize               [3]uint32
	_                              [256]byte
}

type physicalProperties struct {
	API, Driver, Vendor, Device, Kind uint32
	Name                              [256]byte
	PipelineCacheUUID                 [16]byte
	Limits                            physicalLimits
	Sparse                            [5]uint32
}

type shaderInfo struct {
	header
	Flags    uint32
	CodeSize uintptr
	Code     *uint32
}

type binding struct {
	Number, Kind, Count, Stages uint32
	Samplers                    *handle
}

type setLayoutInfo struct {
	header
	Flags, Count uint32
	Bindings     *binding
}

type pushRange struct{ Stages, Offset, Size uint32 }

type layoutInfo struct {
	header
	Flags, Count uint32
	Sets         *handle
	RangeCount   uint32
	Ranges       *pushRange
}

type stageInfo struct {
	header
	Flags, Stage   uint32
	Module         handle
	Name           *byte
	Specialization unsafe.Pointer
}

type pipelineInfo struct {
	header
	Flags        uint32
	Stage        stageInfo
	Layout, Base handle
	BaseIndex    int32
}

type poolSize struct{ Kind, Count uint32 }

type descPoolInfo struct {
	header
	Flags, MaxSets, SizeCount uint32
	Sizes                     *poolSize
}

type setInfo struct {
	header
	Pool    handle
	Count   uint32
	Layouts *handle
}

type descriptor struct {
	Buffer        handle
	Offset, Range uint64
}

type writeInfo struct {
	header
	Set                           handle
	Binding, Element, Count, Kind uint32
	Images                        unsafe.Pointer
	Buffers                       *descriptor
	Views                         *handle
}

type cmdPoolInfo struct {
	header
	Flags, Family uint32
}

type cmdAlloc struct {
	header
	Pool         handle
	Level, Count uint32
}

type cmdBegin struct {
	header
	Flags       uint32
	Inheritance unsafe.Pointer
}

type memoryBarrier struct {
	header
	Src, Dst uint32
}

type bufferCopy struct{ SrcOff, DstOff, Size uint64 }

type fenceInfo struct {
	header
	Flags uint32
}

type submitInfo struct {
	header
	WaitCount   uint32
	Waits       *handle
	WaitStages  *uint32
	CmdCount    uint32
	Cmds        *handle
	SignalCount uint32
	Signals     *handle
}

func st(typ uint32) header { return header{Type: typ} }

const (
	stInstance       = 1
	stDeviceQueue    = 2
	stDevice         = 3
	stSubmit         = 4
	stMemoryAllocate = 5
	stFence          = 8
	stBuffer         = 12
	stShader         = 16
	stPipelineStage  = 18
	stPipeline       = 29
	stPipelineLayout = 30
	stSetLayout      = 32
	stDescriptorPool = 33
	stSetAllocate    = 34
	stWriteSet       = 35
	stCommandPool    = 39
	stCommandAlloc   = 40
	stCommandBegin   = 42
	stMemoryBarrier  = 46

	deviceDiscrete    = 2
	queueCompute      = 0x2
	memoryDeviceLocal = 0x1
	memoryHostVisible = 0x2
	memoryCoherent    = 0x4
	memoryHostCached  = 0x8
	usageTransferSrc  = 0x1
	usageTransferDst  = 0x2
	usageStorage      = 0x20
	descriptorStorage = 7
	shaderCompute     = 0x20
	bindCompute       = 1
	commandOneTime    = 0x1

	stageCompute    = 0x800
	stageTransfer   = 0x1000
	stageHost       = 0x4000
	accessShaderR   = 0x20
	accessShaderW   = 0x40
	accessTransferR = 0x800
	accessTransferW = 0x1000
	accessHostR     = 0x2000
)
