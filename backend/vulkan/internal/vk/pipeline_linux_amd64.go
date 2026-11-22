package vk

import (
	"encoding/binary"
	"errors"
	"fmt"
)

type Pipeline struct {
	d                     *Device
	pipeline, set, layout handle
	bindings              int
	pushBytes             uint32
}

func spirvWords(spirv []byte) ([]uint32, error) {
	if len(spirv) < 20 || len(spirv)%4 != 0 || binary.LittleEndian.Uint32(spirv) != 0x07230203 {
		return nil, errors.New("vulkan: invalid little-endian SPIR-V header or size")
	}
	code := make([]uint32, len(spirv)/4)
	for i := range code {
		code[i] = binary.LittleEndian.Uint32(spirv[i*4:])
	}
	return code, nil
}

func (d *Device) NewPipeline(spirv []byte, bindings int, pushBytes uint32) (_ *Pipeline, err error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if err = d.usable(); err != nil {
		return nil, err
	}
	if bindings <= 0 || bindings > int(d.maxBindings) || bindings > poolDescriptors {
		return nil, fmt.Errorf("vulkan: invalid storage binding count %d", bindings)
	}
	if pushBytes%4 != 0 || pushBytes > d.info.MaxPushBytes {
		return nil, fmt.Errorf("vulkan: invalid push constant byte count %d", pushBytes)
	}
	code, err := spirvWords(spirv)
	if err != nil {
		return nil, err
	}

	a := &d.api
	shader, err := createHandle(d.device, &shaderInfo{header: st(stShader), CodeSize: uintptr(len(spirv)), Code: &code[0]}, "create shader", a.createShader)
	if err != nil {
		return nil, err
	}
	defer a.destroyShader(d.device, shader, nil)

	p := &Pipeline{d: d, bindings: bindings, pushBytes: pushBytes}
	defer func() {
		if err != nil {
			p.release()
		}
	}()
	slots := make([]binding, bindings)
	for i := range slots {
		slots[i] = binding{Number: uint32(i), Kind: descriptorStorage, Count: 1, Stages: shaderCompute}
	}
	set := setLayoutInfo{header: st(stSetLayout), Count: uint32(bindings), Bindings: &slots[0]}
	if p.set, err = createHandle(d.device, &set, "create descriptor set layout", a.createSetLayout); err != nil {
		return nil, err
	}
	layout := layoutInfo{header: st(stPipelineLayout), Count: 1, Sets: &p.set}
	if pushBytes != 0 {
		layout.RangeCount = 1
		layout.Ranges = &pushRange{Stages: shaderCompute, Size: pushBytes}
	}
	if p.layout, err = createHandle(d.device, &layout, "create pipeline layout", a.createLayout); err != nil {
		return nil, err
	}
	entry := [5]byte{'m', 'a', 'i', 'n'}
	stage := stageInfo{header: st(stPipelineStage), Stage: shaderCompute, Module: shader, Name: &entry[0]}
	info := pipelineInfo{header: st(stPipeline), Stage: stage, Layout: p.layout, BaseIndex: -1}
	// Pipeline creation guarantees that any returned nonzero handle needs destruction.
	if err = a.createPipelines(d.device, 0, 1, &info, nil, &p.pipeline).check("create compute pipeline"); err != nil {
		return nil, err
	}
	d.pipelines[p] = struct{}{}
	return p, nil
}

func (p *Pipeline) Close() error {
	p.d.mu.Lock()
	defer p.d.mu.Unlock()
	if p.pipeline == 0 {
		return nil
	}
	if err := p.d.drain(); err != nil {
		return err
	}
	p.release()
	return nil
}

func (p *Pipeline) release() {
	d := p.d
	if p.pipeline != 0 {
		d.api.destroyPipeline(d.device, p.pipeline, nil)
		p.pipeline = 0
	}
	if p.layout != 0 {
		d.api.destroyLayout(d.device, p.layout, nil)
		p.layout = 0
	}
	if p.set != 0 {
		d.api.destroySetLayout(d.device, p.set, nil)
		p.set = 0
	}
	delete(d.pipelines, p)
}
