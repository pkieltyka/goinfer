// Package vk owns the native Vulkan objects used by the Linux/amd64 backend.
package vk

import (
	"bytes"
	"errors"
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

const (
	amdVendor       = 0x1002
	poolSets        = 4096
	poolDescriptors = 32768
)

// Selector optionally pins the PCI device ID of a discrete AMD adapter.
type Selector struct{ DeviceID uint32 }

type Info struct {
	Name                      string
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

// Open loads Vulkan and selects the unique eligible discrete AMD adapter.
func Open(selector Selector) (_ *Device, err error) {
	lib, err := purego.Dlopen("libvulkan.so.1", purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("vulkan: open libvulkan.so.1: %w", err)
	}
	d := &Device{lib: lib, buffers: make(map[*Buffer]struct{}), pipelines: make(map[*Pipeline]struct{})}
	defer func() {
		if err != nil {
			err = errors.Join(err, d.Close())
		}
	}()
	if err = d.api.load(lib); err != nil {
		return nil, err
	}
	var instance handle
	if err = d.api.createInstance(&instanceInfo{header: st(stInstance)}, nil, &instance).check("create instance"); err != nil {
		return nil, err
	}
	d.instance = instance
	physical, err := d.choosePhysical(selector)
	if err != nil {
		return nil, err
	}
	family, err := d.computeFamily(physical)
	if err != nil {
		return nil, err
	}
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

func (d *Device) choosePhysical(selector Selector) (handle, error) {
	var count uint32
	if err := d.api.enumeratePhysical(d.instance, &count, nil).check("count physical devices"); err != nil {
		return 0, err
	}
	if count == 0 {
		return 0, errors.New("vulkan: no physical devices")
	}
	devices := make([]handle, count)
	if err := d.api.enumeratePhysical(d.instance, &count, &devices[0]).check("enumerate physical devices"); err != nil {
		return 0, err
	}
	var selected handle
	for _, physical := range devices[:count] {
		var p physicalProperties
		d.api.physicalProperties(physical, &p)
		if p.Kind != deviceDiscrete || p.Vendor != amdVendor || selector.DeviceID != 0 && p.Device != selector.DeviceID {
			continue
		}
		if selected != 0 {
			if selector.DeviceID == 0 {
				return 0, errors.New("vulkan: multiple discrete AMD adapters found")
			}
			return 0, fmt.Errorf("vulkan: multiple discrete AMD devices match PCI device ID %04x", selector.DeviceID)
		}
		selected = physical
		name, _, _ := bytes.Cut(p.Name[:], []byte{0})
		d.info = Info{
			Name: string(name), VendorID: p.Vendor, DeviceID: p.Device,
			APIVersion: p.API, DriverVersion: p.Driver, MaxStorageRange: uint64(p.Limits.MaxStorageBufferRange),
			MaxGroupCount: p.Limits.MaxWorkGroupCount, MaxPushBytes: p.Limits.MaxPushConstantsSize,
		}
		d.maxBindings = min(p.Limits.MaxPerStageStorageBuffers, p.Limits.MaxDescriptorSetStorageBuffers)
		d.maxAllocations = p.Limits.MaxMemoryAllocationCount
	}
	if selected == 0 {
		if selector.DeviceID == 0 {
			return 0, errors.New("vulkan: no discrete AMD adapter found")
		}
		return 0, fmt.Errorf("vulkan: no discrete AMD device matches PCI device ID %04x", selector.DeviceID)
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

// drain permits destruction only once Vulkan confirms idle or device loss.
func (d *Device) drain() error {
	if !d.inFlight {
		return nil
	}
	switch r := d.api.deviceWaitIdle(d.device); r {
	case success, deviceLost:
		d.inFlight = false
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
		d.device = 0
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
