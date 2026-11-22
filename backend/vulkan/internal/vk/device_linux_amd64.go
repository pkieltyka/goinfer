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
