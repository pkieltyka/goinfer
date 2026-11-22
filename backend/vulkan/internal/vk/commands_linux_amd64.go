package vk

import (
	"errors"
	"fmt"
	"math"
)

type Commands struct {
	d           *Device
	active      bool
	dispatches  uint64
	copies      uint64
	descriptors int
}

// Run records a group, submits it to one queue, and waits for completion.
// Record synchronously and use only the supplied Commands inside the callback.
func (d *Device) Run(record func(*Commands) error) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := d.usable(); err != nil {
		return err
	}
	if record == nil {
		return errors.New("vulkan: nil command recorder")
	}
	a := &d.api
	if err := a.resetCmdPool(d.device, d.cmdPool, 0).check("reset command pool"); err != nil {
		return err
	}
	if err := a.resetDescPool(d.device, d.descPool, 0).check("reset descriptor pool"); err != nil {
		return err
	}
	if err := a.beginCmd(d.cmd, &cmdBegin{header: st(stCommandBegin), Flags: commandOneTime}).check("begin command buffer"); err != nil {
		return err
	}
	c := &Commands{d: d, active: true}
	if err := c.record(record); err != nil {
		return err
	}
	if err := a.endCmd(d.cmd).check("end command buffer"); err != nil {
		return err
	}
	if err := a.resetFences(d.device, 1, &d.fence).check("reset fence"); err != nil {
		return err
	}
	info := submitInfo{header: st(stSubmit), CmdCount: 1, Cmds: &d.cmd}
	if r := a.queueSubmit(d.queue, 1, &info, d.fence); r != success {
		err := r.check("submit command buffer")
		if r == deviceLost {
			d.broken = err
		}
		return err
	}
	d.inFlight = true
	d.activity.Submissions++
	d.activity.Copies += c.copies
	d.activity.Dispatches += c.dispatches
	d.activity.Waits++
	switch r := a.waitFences(d.device, 1, &d.fence, 1, math.MaxUint64); r {
	case success:
		d.inFlight = false
		return nil
	case deviceLost:
		d.inFlight = false
		d.broken = r.check("wait for fence")
		return d.broken
	default:
		d.broken = r.check("wait for fence")
		return errors.Join(d.broken, d.drain())
	}
}

func (c *Commands) record(record func(*Commands) error) error {
	defer func() { c.active = false }()
	return record(c)
}

func (c *Commands) Dispatch(p *Pipeline, bindings []*Buffer, push []uint32, grid [3]uint32) error {
	if !c.active {
		return errors.New("vulkan: inactive command recorder")
	}
	d := c.d
	if p == nil || p.d != d || p.pipeline == 0 {
		return errors.New("vulkan: missing, foreign, or closed compute pipeline")
	}
	if len(bindings) != p.bindings || len(push) != int(p.pushBytes/4) {
		return errors.New("vulkan: dispatch does not match pipeline bindings or push constants")
	}
	for i, count := range grid {
		if count == 0 || count > d.info.MaxGroupCount[i] {
			return fmt.Errorf("vulkan: dispatch grid axis %d count %d exceeds valid range 1..%d", i, count, d.info.MaxGroupCount[i])
		}
	}
	if c.dispatches >= poolSets || len(bindings) > poolDescriptors-c.descriptors {
		return errors.New("vulkan: descriptor capacity exhausted for this submission")
	}
	for i, b := range bindings {
		if !c.validBuffer(b) || b.size > d.info.MaxStorageRange {
			return fmt.Errorf("vulkan: invalid storage buffer at binding %d", i)
		}
	}

	a := &d.api
	var set handle
	if err := a.allocateSets(d.device, &setInfo{header: st(stSetAllocate), Pool: d.descPool, Count: 1, Layouts: &p.set}, &set).check("allocate descriptor set"); err != nil {
		return err
	}
	descriptors := make([]descriptor, len(bindings))
	writes := make([]writeInfo, len(bindings))
	for i, b := range bindings {
		descriptors[i] = descriptor{Buffer: b.buffer, Range: b.size}
		writes[i] = writeInfo{header: st(stWriteSet), Set: set, Binding: uint32(i), Count: 1, Kind: descriptorStorage, Buffers: &descriptors[i]}
	}
	a.updateSets(d.device, uint32(len(writes)), &writes[0], 0, nil)
	a.bindPipeline(d.cmd, bindCompute, p.pipeline)
	a.bindSets(d.cmd, bindCompute, p.layout, 0, 1, &set, 0, nil)
	if len(push) > 0 {
		a.pushConstants(d.cmd, p.layout, shaderCompute, 0, p.pushBytes, &push[0])
	}
	a.dispatch(d.cmd, grid[0], grid[1], grid[2])
	c.sync(stageCompute, accessShaderW)
	c.dispatches++
	c.descriptors += len(bindings)
	return nil
}

func (c *Commands) Copy(src, dst *Buffer, srcOff, dstOff, size uint64) error {
	if !c.active {
		return errors.New("vulkan: inactive command recorder")
	}
	if !c.validBuffer(src) || !c.validBuffer(dst) {
		return errors.New("vulkan: missing, foreign, or closed copy buffer")
	}
	if !copyRange(src.size, srcOff, size) || !copyRange(dst.size, dstOff, size) {
		return errors.New("vulkan: invalid copy range")
	}
	if src == dst && srcOff < dstOff+size && dstOff < srcOff+size {
		return errors.New("vulkan: copy ranges overlap in the same buffer")
	}
	d := c.d
	d.api.copyBuffer(d.cmd, src.buffer, dst.buffer, 1, &bufferCopy{SrcOff: srcOff, DstOff: dstOff, Size: size})
	c.sync(stageTransfer, accessTransferW)
	c.copies++
	return nil
}

func copyRange(bufferSize, offset, size uint64) bool {
	return size > 0 && size%4 == 0 && offset%4 == 0 && offset <= bufferSize && size <= bufferSize-offset
}

func (c *Commands) validBuffer(b *Buffer) bool {
	return b != nil && b.d == c.d && b.buffer != 0
}

// The execution dependency also orders reads that precede a later write.
func (c *Commands) sync(stage, access uint32) {
	const stages = stageCompute | stageTransfer | stageHost
	const accesses = accessShaderR | accessShaderW | accessTransferR | accessTransferW | accessHostR
	d := c.d
	barrier := memoryBarrier{header: st(stMemoryBarrier), Src: access, Dst: accesses}
	d.api.barrier(d.cmd, stage, stages, 0, 1, &barrier, 0, nil, 0, nil)
}
