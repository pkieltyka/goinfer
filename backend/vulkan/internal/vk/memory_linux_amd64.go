package vk

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// enumerate obtains a complete snapshot before returning a list. Selecting
// from a partial physical-device list could mistake an ambiguous match for one.
func enumerate[T any](op string, list func(*uint32, *T) result) ([]T, error) {
	for range 3 {
		var count uint32
		if err := list(&count, nil).check(op); err != nil {
			return nil, err
		}
		if count == 0 {
			return nil, nil
		}
		if count > 1<<16 {
			return nil, fmt.Errorf("vulkan: %s: invalid count %d", op, count)
		}
		items := make([]T, count)
		r := list(&count, &items[0])
		if r == incomplete {
			continue
		}
		if err := r.check(op); err != nil {
			return nil, err
		}
		if count > uint32(len(items)) {
			return nil, fmt.Errorf("vulkan: %s: invalid written count %d", op, count)
		}
		return items[:count], nil
	}
	return nil, incomplete.check("obtain complete snapshot: " + op)
}

func addSaturated(a, b uint64) uint64 {
	return a + min(b, math.MaxUint64-a)
}

func localCapacity(memory *memoryProperties) uint64 {
	var total uint64
	for _, heap := range memory.Heaps[:min(memory.HeapCount, uint32(len(memory.Heaps)))] {
		if heap.Flags&heapDeviceLocal != 0 {
			total = addSaturated(total, heap.Size)
		}
	}
	return total
}

func (d *Device) availableMemory(physical handle, version uint32, memory *memoryProperties) (uint64, bool, error) {
	if d.instanceVersion < apiVersion11 || version < apiVersion11 || d.memoryProperties2 == nil ||
		localCapacity(memory) == 0 || memory.HeapCount > uint32(len(memory.Heaps)) {
		return 0, false, nil
	}
	extensions, err := enumerate("enumerate device extensions", func(n *uint32, dst *extensionProperties) result {
		return d.api.enumerateExtensions(physical, nil, n, dst)
	})
	if err != nil {
		if errors.Is(err, ErrOutOfMemory) {
			return 0, false, err
		}
		return 0, false, nil
	}
	for _, ext := range extensions {
		name, _, _ := bytes.Cut(ext.Name[:], []byte{0})
		if string(name) != "VK_EXT_memory_budget" {
			continue
		}
		budget := memoryBudget{header: st(stMemoryBudget)}
		props := memoryProperties2{header: st(stMemoryProperties2)}
		props.Next = unsafe.Pointer(&budget)
		d.memoryProperties2(physical, &props)
		available, known := localBudget(memory, &budget)
		return available, known, nil
	}
	return 0, false, nil
}

func localBudget(memory *memoryProperties, budget *memoryBudget) (uint64, bool) {
	var available uint64
	for i, heap := range memory.Heaps[:memory.HeapCount] {
		if heap.Flags&heapDeviceLocal == 0 {
			continue
		}
		if budget.Budget[i] == 0 {
			return 0, false
		}
		ceiling := min(budget.Budget[i], heap.Size)
		available = addSaturated(available, ceiling-min(ceiling, budget.Usage[i]))
	}
	return available, true
}

// Memory samples the budget for this device's physical adapter.
func (d *Device) Memory() (total, available uint64, known bool, err error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := d.usable(); err != nil {
		return 0, 0, false, err
	}
	available, known, err = d.availableMemory(d.physical, d.info.APIVersion, &d.properties)
	return d.info.TotalLocalBytes, available, known, err
}
