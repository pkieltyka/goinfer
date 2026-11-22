package vk

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

type Memory uint8

const (
	DeviceLocal Memory = iota
	HostVisible
)

type Buffer struct {
	d      *Device
	buffer handle
	memory handle
	mapped unsafe.Pointer
	size   uint64
}

func (d *Device) NewBuffer(size uint64, memory Memory) (_ *Buffer, err error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if err = d.usable(); err != nil {
		return nil, err
	}
	if size == 0 || size > math.MaxInt {
		return nil, fmt.Errorf("vulkan: invalid buffer size %d", size)
	}
	if memory != DeviceLocal && memory != HostVisible {
		return nil, fmt.Errorf("vulkan: invalid memory placement %d", memory)
	}
	if len(d.buffers) >= int(d.maxAllocations) {
		return nil, errors.New("vulkan: device memory allocation count exhausted")
	}

	b := &Buffer{d: d, size: size}
	defer func() {
		if err != nil {
			b.release()
		}
	}()
	a := &d.api
	info := bufferInfo{header: st(stBuffer), Size: size, Usage: usageStorage | usageTransferSrc | usageTransferDst}
	if err = a.createBuffer(d.device, &info, nil, &b.buffer).check("create buffer"); err != nil {
		return nil, err
	}
	var requirements memoryRequirements
	a.bufferRequirements(d.device, b.buffer, &requirements)
	index, err := d.memoryIndex(requirements.Bits, memory)
	if err != nil {
		return nil, err
	}
	alloc := memoryInfo{header: st(stMemoryAllocate), Size: requirements.Size, Index: index}
	if err = a.allocateMemory(d.device, &alloc, nil, &b.memory).check("allocate buffer memory"); err != nil {
		return nil, err
	}
	if err = a.bindMemory(d.device, b.buffer, b.memory, 0).check("bind buffer memory"); err != nil {
		return nil, err
	}
	if memory == HostVisible {
		if err = a.mapMemory(d.device, b.memory, 0, size, 0, &b.mapped).check("map buffer memory"); err != nil {
			return nil, err
		}
	}
	d.buffers[b] = struct{}{}
	return b, nil
}

func (d *Device) memoryIndex(bits uint32, memory Memory) (uint32, error) {
	var wanted, preferred uint32 = memoryDeviceLocal, memoryDeviceLocal
	if memory == HostVisible {
		wanted = memoryHostVisible | memoryCoherent
		preferred = wanted | memoryHostCached
	}
	var index uint32
	found := false
	for i, t := range d.properties.Types[:d.properties.Count] {
		if bits&(1<<i) == 0 || t.Flags&wanted != wanted {
			continue
		}
		if t.Flags&preferred == preferred {
			return uint32(i), nil
		}
		if !found {
			index, found = uint32(i), true
		}
	}
	if found {
		return index, nil
	}
	return 0, fmt.Errorf("vulkan: no memory type with required flags %#x for buffer", wanted)
}

func (b *Buffer) Size() uint64 { return b.size }

func (b *Buffer) WriteF32(offset uint64, src []float32) error {
	b.d.mu.Lock()
	defer b.d.mu.Unlock()
	m, err := b.mappedWords(offset, len(src))
	if err != nil {
		return err
	}
	copy(m, src)
	return nil
}

func (b *Buffer) ReadF32(offset uint64, dst []float32) error {
	b.d.mu.Lock()
	defer b.d.mu.Unlock()
	m, err := b.mappedWords(offset, len(dst))
	if err != nil {
		return err
	}
	copy(dst, m)
	return nil
}

func (b *Buffer) mappedWords(offset uint64, words int) ([]float32, error) {
	if err := b.d.usable(); err != nil {
		return nil, err
	}
	if b.buffer == 0 || b.mapped == nil {
		return nil, errors.New("vulkan: buffer is closed or not host-visible")
	}
	if offset > b.size || offset%4 != 0 || uint64(words) > (b.size-offset)/4 {
		return nil, errors.New("vulkan: invalid mapped buffer range")
	}
	if words == 0 {
		return nil, nil
	}
	return unsafe.Slice((*float32)(unsafe.Add(b.mapped, offset)), words), nil
}

func (b *Buffer) Close() error {
	b.d.mu.Lock()
	defer b.d.mu.Unlock()
	if b.buffer == 0 {
		return nil
	}
	if err := b.d.drain(); err != nil {
		return err
	}
	b.release()
	return nil
}

func (b *Buffer) release() {
	if b.mapped != nil {
		b.d.api.unmapMemory(b.d.device, b.memory)
		b.mapped = nil
	}
	if b.buffer != 0 {
		b.d.api.destroyBuffer(b.d.device, b.buffer, nil)
		b.buffer = 0
	}
	if b.memory != 0 {
		b.d.api.freeMemory(b.d.device, b.memory, nil)
		b.memory = 0
	}
	delete(b.d.buffers, b)
}
