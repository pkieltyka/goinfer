// Package vk owns the native Vulkan objects used by the Linux/amd64 backend.
package vk

import (
	"bytes"
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	apiVersion11    = 1<<22 | 1<<12
	amdVendor       = 0x1002
	heapDeviceLocal = 0x1
	poolSets        = 4096
	poolDescriptors = 32768
)

// Selector filters by PCI product ID, physical device UUID, or both.
type Selector struct {
	DeviceID uint32
	UUID     [16]byte
}

type Info struct {
	Name                      string
	UUID                      [16]byte
	TotalLocalBytes           uint64
	AvailableLocalBytes       uint64
	BudgetKnown               bool
	VendorID, DeviceID        uint32
	APIVersion, DriverVersion uint32
	MaxStorageRange           uint64
	MaxGroupCount             [3]uint32
	MaxPushBytes              uint32
}

type Activity struct{ Dispatches, Copies, Submissions, Waits uint64 }

type Device struct {
	mu                      sync.Mutex
	api                     api
	properties2             func(handle, *physicalProperties2)
	memoryProperties2       func(handle, *memoryProperties2)
	enumerateVersion        func(*uint32) result
	getProc                 func(handle, string) uintptr
	instanceVersion         uint32
	physical                handle
	lib                     uintptr
	instance, device, queue handle
	cmdPool, cmd, descPool  handle
	fence                   handle
	info                    Info
	properties              memoryProperties
	maxBindings             uint32
	maxAllocations          uint32
	buffers                 map[*Buffer]struct{}
	pipelines               map[*Pipeline]struct{}
	activity                Activity
	inFlight                bool
	broken                  error
	allocMemHook            func(Memory, uint64) result
	closeBufHook            func() error
}

// SetAllocMemoryHookForTest installs an optional per-device hook invoked right
// at vkAllocateMemory inside Device.NewBuffer. Returning 0 delegates to the
// underlying allocateMemory function; any nonzero VkResult fails the call.
func (d *Device) SetAllocMemoryHookForTest(fn func(Memory, uint64) int32) {
	d.mu.Lock()
	if fn == nil {
		d.allocMemHook = nil
	} else {
		d.allocMemHook = func(m Memory, size uint64) result { return result(fn(m, size)) }
	}
	d.mu.Unlock()
}

// SetCloseBufferHookForTest installs an optional per-device hook invoked at
// Buffer.Close before releasing the buffer. Returning a non-nil error leaves
// the buffer open (and poisons the device if the error matches ErrDeviceLost).
func (d *Device) SetCloseBufferHookForTest(fn func() error) {
	d.mu.Lock()
	d.closeBufHook = fn
	d.mu.Unlock()
}

// LiveBuffersForTest reports how many Buffer handles this Device currently owns.
func (d *Device) LiveBuffersForTest() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.buffers)
}

// FakeHooksForTest configures deterministic native VkResult injections for
// NewFakeDeviceForTest. Nil hooks succeed (0).
type FakeHooksForTest struct {
	MaxAllocations uint32
	CreateBuffer   func(size uint64) int32
	AllocateMemory func(memory Memory, size uint64) int32
	BindMemory     func() int32
	MapMemory      func() int32
	QueueSubmit    func() int32
	WaitFences     func() int32
	DeviceWaitIdle func() int32
}

// NewFakeDeviceForTest constructs a CPU-only Device backed by in-memory fake
// Vulkan entry points so package vulkan tests can inject deterministic native
// results without opening libvulkan.so.1 or touching hardware.
func NewFakeDeviceForTest(h *FakeHooksForTest) *Device {
	if h == nil {
		h = &FakeHooksForTest{}
	}
	maxAlloc := h.MaxAllocations
	if maxAlloc == 0 {
		maxAlloc = 4096
	}
	var nextHandle handle = 100
	newHandle := func() handle {
		nextHandle++
		return nextHandle
	}
	bufSizes := map[handle]uint64{}
	memBacking := map[handle][]byte{}
	d := &Device{
		instance:       1,
		physical:       2,
		device:         3,
		queue:          4,
		cmdPool:        5,
		cmd:            6,
		descPool:       7,
		fence:          8,
		maxBindings:    16,
		maxAllocations: maxAlloc,
		buffers:        make(map[*Buffer]struct{}),
		pipelines:      make(map[*Pipeline]struct{}),
		info: Info{
			Name:            "fake-vulkan-cpu",
			UUID:            [16]byte{0xde, 0xad, 0xbe, 0xef},
			VendorID:        amdVendor,
			DeviceID:        0x744c,
			MaxStorageRange: 1 << 30,
			MaxGroupCount:   [3]uint32{65535, 65535, 65535},
			MaxPushBytes:    128,
		},
	}
	d.properties.Count = 2
	d.properties.Types[0].Flags = memoryDeviceLocal
	d.properties.Types[1].Flags = memoryHostVisible | memoryCoherent | memoryHostCached
	d.api = api{
		destroyInstance: func(handle, unsafe.Pointer) {},
		destroyDevice:   func(handle, unsafe.Pointer) {},
		deviceWaitIdle: func(handle) result {
			if h.DeviceWaitIdle != nil {
				return result(h.DeviceWaitIdle())
			}
			return success
		},
		createBuffer: func(_ handle, info *bufferInfo, _ unsafe.Pointer, out *handle) result {
			if h.CreateBuffer != nil {
				if r := result(h.CreateBuffer(info.Size)); r != success {
					return r
				}
			}
			id := newHandle()
			bufSizes[id] = info.Size
			*out = id
			return success
		},
		destroyBuffer: func(_ handle, b handle, _ unsafe.Pointer) { delete(bufSizes, b) },
		bufferRequirements: func(_ handle, b handle, req *memoryRequirements) {
			*req = memoryRequirements{Size: bufSizes[b], Alignment: 16, Bits: 0x3}
		},
		allocateMemory: func(_ handle, info *memoryInfo, _ unsafe.Pointer, out *handle) result {
			mem := DeviceLocal
			if info.Index == 1 {
				mem = HostVisible
			}
			if h.AllocateMemory != nil {
				if r := result(h.AllocateMemory(mem, info.Size)); r != success {
					return r
				}
			}
			id := newHandle()
			memBacking[id] = make([]byte, max(info.Size, 1))
			*out = id
			return success
		},
		freeMemory: func(_ handle, m handle, _ unsafe.Pointer) { delete(memBacking, m) },
		bindMemory: func(_, _, _ handle, _ uint64) result {
			if h.BindMemory != nil {
				return result(h.BindMemory())
			}
			return success
		},
		mapMemory: func(_ handle, m handle, _, _ uint64, _ uint32, out *unsafe.Pointer) result {
			if h.MapMemory != nil {
				if r := result(h.MapMemory()); r != success {
					return r
				}
			}
			b := memBacking[m]
			*out = unsafe.Pointer(&b[0])
			return success
		},
		unmapMemory: func(handle, handle) {},
		createShader: func(_ handle, _ *shaderInfo, _ unsafe.Pointer, out *handle) result {
			*out = newHandle()
			return success
		},
		destroyShader: func(handle, handle, unsafe.Pointer) {},
		createSetLayout: func(_ handle, _ *setLayoutInfo, _ unsafe.Pointer, out *handle) result {
			*out = newHandle()
			return success
		},
		destroySetLayout: func(handle, handle, unsafe.Pointer) {},
		createLayout: func(_ handle, _ *layoutInfo, _ unsafe.Pointer, out *handle) result {
			*out = newHandle()
			return success
		},
		destroyLayout: func(handle, handle, unsafe.Pointer) {},
		createPipelines: func(_ handle, _ handle, _ uint32, _ *pipelineInfo, _ unsafe.Pointer, out *handle) result {
			*out = newHandle()
			return success
		},
		destroyPipeline: func(handle, handle, unsafe.Pointer) {},
		destroyDescPool: func(handle, handle, unsafe.Pointer) {},
		resetDescPool:   func(handle, handle, uint32) result { return success },
		allocateSets:    func(_ handle, _ *setInfo, out *handle) result { *out = newHandle(); return success },
		updateSets:      func(handle, uint32, *writeInfo, uint32, unsafe.Pointer) {},
		destroyCmdPool:  func(handle, handle, unsafe.Pointer) {},
		resetCmdPool:    func(handle, handle, uint32) result { return success },
		beginCmd:        func(handle, *cmdBegin) result { return success },
		endCmd:          func(handle) result { return success },
		destroyFence:    func(handle, handle, unsafe.Pointer) {},
		resetFences:     func(handle, uint32, *handle) result { return success },
		waitFences: func(handle, uint32, *handle, uint32, uint64) result {
			if h.WaitFences != nil {
				return result(h.WaitFences())
			}
			return success
		},
		bindPipeline:  func(handle, uint32, handle) {},
		bindSets:      func(handle, uint32, handle, uint32, uint32, *handle, uint32, *uint32) {},
		pushConstants: func(handle, handle, uint32, uint32, uint32, *uint32) {},
		dispatch:      func(handle, uint32, uint32, uint32) {},
		copyBuffer:    func(handle, handle, handle, uint32, *bufferCopy) {},
		barrier: func(handle, uint32, uint32, uint32, uint32, *memoryBarrier, uint32, unsafe.Pointer, uint32, unsafe.Pointer) {
		},
		queueSubmit: func(handle, uint32, *submitInfo, handle) result {
			if h.QueueSubmit != nil {
				return result(h.QueueSubmit())
			}
			return success
		},
	}
	return d
}

var loadLibrary = loadNativeLibrary

func loadNativeLibrary() (*Device, error) {
	lib, err := purego.Dlopen("libvulkan.so.1", purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("vulkan: open libvulkan.so.1: %w", err)
	}
	d := &Device{lib: lib}
	if err := d.api.load(lib); err != nil {
		return nil, errors.Join(err, d.Close())
	}
	if addr, err := purego.Dlsym(lib, "vkEnumerateInstanceVersion"); err == nil {
		purego.RegisterFunc(&d.enumerateVersion, addr)
	}
	if addr, err := purego.Dlsym(lib, "vkGetInstanceProcAddr"); err == nil {
		purego.RegisterFunc(&d.getProc, addr)
	}
	return d, nil
}

func runtimeFailure(err error) error {
	if errors.Is(err, ErrOutOfMemory) {
		return err
	}
	return fmt.Errorf("%w: %w", ErrRuntimeUnavailable, err)
}

func openInstance() (_ *Device, err error) {
	d, err := loadLibrary()
	if err != nil {
		return nil, runtimeFailure(err)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, d.Close())
		}
	}()
	info := instanceInfo{header: st(stInstance)}
	if d.enumerateVersion != nil {
		var version uint32
		if err := d.enumerateVersion(&version).check("enumerate instance version"); err != nil {
			return nil, runtimeFailure(err)
		}
		if version >= apiVersion11 {
			d.instanceVersion = apiVersion11
			info.App = &applicationInfo{APIVersion: d.instanceVersion}
		}
	}
	var instance handle
	if err = d.api.createInstance(&info, nil, &instance).check("create instance"); err != nil {
		return nil, runtimeFailure(err)
	}
	d.instance = instance
	if info.App != nil && d.getProc != nil {
		if addr := d.getProc(instance, "vkGetPhysicalDeviceProperties2"); addr != 0 {
			purego.RegisterFunc(&d.properties2, addr)
		}
		if addr := d.getProc(instance, "vkGetPhysicalDeviceMemoryProperties2"); addr != 0 {
			purego.RegisterFunc(&d.memoryProperties2, addr)
		}
	}
	return d, nil
}

// Open loads Vulkan and selects the unique eligible discrete AMD adapter.
func Open(selector Selector) (_ *Device, err error) {
	d, err := openInstance()
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, d.Close())
		}
	}()
	candidates, err := d.adapters(false)
	if err != nil {
		return nil, err
	}
	selected, err := choosePhysical(candidates, selector)
	if err != nil {
		return nil, err
	}
	physical, family := selected.handle, selected.family
	d.physical = physical
	d.info = selected.info
	d.maxBindings, d.maxAllocations = selected.maxBindings, selected.maxAllocations
	d.buffers = make(map[*Buffer]struct{})
	d.pipelines = make(map[*Pipeline]struct{})
	d.api.memoryProperties(physical, &d.properties)
	if d.properties.Count == 0 || d.properties.Count > uint32(len(d.properties.Types)) {
		return nil, fmt.Errorf("vulkan: invalid memory type count %d", d.properties.Count)
	}
	priority := float32(1)
	queues := queueInfo{header: st(stDeviceQueue), Family: family, Count: 1, Priorities: &priority}
	info := deviceInfo{header: st(stDevice), Count: 1, Queues: &queues}
	if d.device, err = createHandle(physical, &info, "create device", d.api.createDevice); err != nil {
		return nil, err
	}
	d.api.getQueue(d.device, family, 0, &d.queue)
	if err = d.createRunObjects(family); err != nil {
		return nil, err
	}
	return d, nil
}

type adapter struct {
	handle                      handle
	family                      uint32
	info                        Info
	maxBindings, maxAllocations uint32
}

// Devices inventories eligible adapters without creating a logical device.
func Devices(includeBudget bool) (_ []Info, err error) {
	d, err := openInstance()
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, d.Close()) }()
	adapters, err := d.adapters(includeBudget)
	if err != nil {
		return nil, err
	}
	out := make([]Info, 0, len(adapters))
	for _, adapter := range adapters {
		out = append(out, adapter.info)
	}
	return out, nil
}

func (d *Device) adapters(includeBudget bool) ([]adapter, error) {
	devices, err := enumerate("enumerate physical devices", func(n *uint32, dst *handle) result {
		return d.api.enumeratePhysical(d.instance, n, dst)
	})
	if err != nil {
		return nil, runtimeFailure(err)
	}
	var out []adapter
	for _, physical := range devices {
		var p physicalProperties
		d.api.physicalProperties(physical, &p)
		if p.Kind != deviceDiscrete || p.Vendor != amdVendor {
			continue
		}
		family, err := d.computeFamily(physical)
		if err != nil {
			continue
		}
		var id physicalIDProperties
		if d.properties2 != nil && p.API >= apiVersion11 {
			id.header = st(stPhysicalIDProperties)
			properties := physicalProperties2{header: st(stPhysicalProperties2)}
			properties.Next = unsafe.Pointer(&id)
			d.properties2(physical, &properties)
		}
		var memory memoryProperties
		d.api.memoryProperties(physical, &memory)
		total := localCapacity(&memory)
		var available uint64
		var known bool
		if includeBudget {
			available, known, err = d.availableMemory(physical, p.API, &memory)
			if err != nil {
				return nil, err
			}
		}
		name, _, _ := bytes.Cut(p.Name[:], []byte{0})
		out = append(out, adapter{
			handle: physical, family: family,
			info: Info{
				Name: string(name), UUID: id.DeviceUUID, VendorID: p.Vendor, DeviceID: p.Device,
				TotalLocalBytes: total, AvailableLocalBytes: available, BudgetKnown: known, APIVersion: p.API, DriverVersion: p.Driver,
				MaxStorageRange: uint64(p.Limits.MaxStorageBufferRange),
				MaxGroupCount:   p.Limits.MaxWorkGroupCount, MaxPushBytes: p.Limits.MaxPushConstantsSize,
			},
			maxBindings:    min(p.Limits.MaxPerStageStorageBuffers, p.Limits.MaxDescriptorSetStorageBuffers),
			maxAllocations: p.Limits.MaxMemoryAllocationCount,
		})
	}
	return out, nil
}

func choosePhysical(candidates []adapter, selector Selector) (adapter, error) {
	var selected adapter
	for _, candidate := range candidates {
		if selector.DeviceID != 0 && candidate.info.DeviceID != selector.DeviceID ||
			selector.UUID != ([16]byte{}) && candidate.info.UUID != selector.UUID {
			continue
		}
		if selected.handle != 0 {
			return adapter{}, ErrAmbiguousAdapter
		}
		selected = candidate
	}
	if selected.handle == 0 {
		return adapter{}, ErrNoAdapter
	}
	return selected, nil
}

func (d *Device) computeFamily(physical handle) (uint32, error) {
	var count uint32
	d.api.queueFamilies(physical, &count, nil)
	if count != 0 {
		families := make([]queueFamily, count)
		d.api.queueFamilies(physical, &count, &families[0])
		for i, family := range families[:count] {
			if family.Flags&queueCompute != 0 && family.Count != 0 {
				return uint32(i), nil
			}
		}
	}
	return 0, errors.New("vulkan: selected device has no compute queue")
}

func (d *Device) createRunObjects(family uint32) (err error) {
	a := &d.api
	cmdPool := cmdPoolInfo{header: st(stCommandPool), Family: family}
	if d.cmdPool, err = createHandle(d.device, &cmdPool, "create command pool", a.createCmdPool); err != nil {
		return err
	}
	var cmd handle
	if err := a.allocateCmds(d.device, &cmdAlloc{header: st(stCommandAlloc), Pool: d.cmdPool, Count: 1}, &cmd).check("allocate command buffer"); err != nil {
		return err
	}
	d.cmd = cmd
	size := poolSize{Kind: descriptorStorage, Count: poolDescriptors}
	pool := descPoolInfo{header: st(stDescriptorPool), MaxSets: poolSets, SizeCount: 1, Sizes: &size}
	if d.descPool, err = createHandle(d.device, &pool, "create descriptor pool", a.createDescPool); err != nil {
		return err
	}
	d.fence, err = createHandle(d.device, &fenceInfo{header: st(stFence)}, "create fence", a.createFence)
	return err
}

func (d *Device) Info() Info { return d.info }

// Activity returns cumulative native work since Open.
func (d *Device) Activity() Activity {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.activity
}

func (d *Device) usable() error {
	if d.device == 0 {
		return errors.New("vulkan: device is closed")
	}
	return d.broken
}

// Status reports whether commands and allocations may proceed.
func (d *Device) Status() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.usable()
}

// LossCause reports the first recorded device loss, including after Close.
func (d *Device) LossCause() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if errors.Is(d.broken, ErrDeviceLost) {
		return d.broken
	}
	return nil
}

func (d *Device) poisonOnLoss(err error) {
	if !errors.Is(d.broken, ErrDeviceLost) && errors.Is(err, ErrDeviceLost) {
		d.broken = err
	}
}

// drain permits destruction only once Vulkan confirms idle or device loss.
func (d *Device) drain() error {
	if !d.inFlight {
		return nil
	}
	switch r := d.api.deviceWaitIdle(d.device); r {
	case success:
		d.inFlight = false
		return nil
	case deviceLost:
		d.inFlight = false
		d.poisonOnLoss(r.check("drain submitted work"))
		return nil
	default:
		return r.check("drain submitted work")
	}
}

func (d *Device) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := d.drain(); err != nil {
		return err
	}
	if d.device != 0 {
		a := &d.api
		if d.cmdPool != 0 {
			a.destroyCmdPool(d.device, d.cmdPool, nil)
		}
		if d.descPool != 0 {
			a.destroyDescPool(d.device, d.descPool, nil)
		}
		if d.fence != 0 {
			a.destroyFence(d.device, d.fence, nil)
		}
		for p := range d.pipelines {
			p.release()
		}
		for b := range d.buffers {
			b.release()
		}
		a.destroyDevice(d.device, nil)
		d.device, d.physical = 0, 0
	}
	if d.instance != 0 {
		d.api.destroyInstance(d.instance, nil)
		d.instance = 0
	}
	if d.lib != 0 {
		err := purego.Dlclose(d.lib)
		d.lib = 0
		return err
	}
	return nil
}
