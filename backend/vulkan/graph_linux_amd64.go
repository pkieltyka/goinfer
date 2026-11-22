package vulkan

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

// A forward owns its temporaries through the final fence, including snapshots
// requested by layer-difference tests. Only projected rows normally return to Go.
type forward struct {
	a                                                                                                          *encoder
	rows, maxGroupX                                                                                            int
	lengths                                                                                                    []int
	buffers                                                                                                    []*vk.Buffer
	err                                                                                                        error
	input, x, normalized, branch, q, k, v, context, gate, up, pleGate, ple, scores, output, download, segments *vk.Buffer
	tables                                                                                                     map[rotation][2]*vk.Buffer
	snapshots                                                                                                  []*vk.Buffer
}

func (f *forward) buffer(mem vk.Memory, dims ...int) *vk.Buffer {
	if f.err != nil {
		return nil
	}
	bytes, err := byteSize(f.a.device.Info().MaxStorageRange, dims...)
	if err != nil {
		f.err = err
		return nil
	}
	b, err := f.a.device.NewBuffer(bytes, mem)
	if err != nil {
		f.err = err
		return nil
	}
	f.buffers = append(f.buffers, b)
	return b
}
func (f *forward) mapped(values []float32) *vk.Buffer {
	b := f.buffer(vk.HostVisible, len(values))
	if f.err == nil {
		f.err = b.WriteF32(0, values)
	}
	return b
}
func (f *forward) close() error {
	var err error
	for _, b := range f.buffers {
		err = errors.Join(err, b.Close())
	}
	return err
}

func (a *encoder) forwardRows(x []float32, lengths []int, keep bool) (out []float32, hidden [][]float32, err error) {
	t := len(x) / a.cfg.Hidden
	f := forward{a: a, rows: t, maxGroupX: int(a.device.Info().MaxGroupCount[0]), lengths: lengths, tables: make(map[rotation][2]*vk.Buffer)}
	beforeSub := a.device.Activity().Submissions
	submitted := false
	defer func() {
		closeErr := f.close()
		if closeErr != nil {
			out, hidden = nil, nil
			err = errors.Join(err, closeErr)
		}
		status := a.device.Status()
		if errors.Is(status, vk.ErrDeviceLost) && !errors.Is(err, vk.ErrDeviceLost) {
			err = fmt.Errorf("%w (forward returned %v)", status, err)
			out, hidden = nil, nil
		}
		if err == nil {
			return
		}
		out, hidden = nil, nil
		if errors.Is(err, vk.ErrDeviceLost) || errors.Is(status, vk.ErrDeviceLost) {
			err = classify(err)
			return
		}
		if !submitted && a.device.Activity().Submissions == beforeSub && closeErr == nil && status == nil && errors.Is(err, vk.ErrBufferDeviceOutOfMemory) {
			err = fmt.Errorf("%w: %w", embeddinggemma2.ErrBatchOutOfMemory, err)
			return
		}
		if errors.Is(err, vk.ErrOutOfMemory) {
			err = classify(err)
		}
	}()
	cfg := a.cfg
	f.input = f.mapped(x)
	f.x = f.buffer(vk.DeviceLocal, t, cfg.Hidden)
	f.normalized = f.buffer(vk.DeviceLocal, t, cfg.Hidden)
	f.branch = f.buffer(vk.DeviceLocal, t, cfg.Hidden)
	f.q = f.buffer(vk.DeviceLocal, t, a.maxQ)
	f.context = f.buffer(vk.DeviceLocal, t, a.maxQ)
	f.k = f.buffer(vk.DeviceLocal, t, a.maxKV)
	f.v = f.buffer(vk.DeviceLocal, t, a.maxKV)
	f.gate = f.buffer(vk.DeviceLocal, t, cfg.Intermediate)
	f.up = f.buffer(vk.DeviceLocal, t, cfg.Intermediate)
	f.pleGate = f.buffer(vk.DeviceLocal, t, cfg.PLEDim)
	f.ple = f.buffer(vk.DeviceLocal, t, cfg.Layers, cfg.PLEDim)
	longest := slices.Max(lengths)
	f.scores = f.buffer(vk.DeviceLocal, cfg.Heads, min(longest, queryBlock), longest)
	f.output = f.buffer(vk.DeviceLocal, t, cfg.EmbeddingDim)
	f.download = f.buffer(vk.HostVisible, t, cfg.EmbeddingDim)
	if keep {
		for range cfg.Layers + 1 {
			f.snapshots = append(f.snapshots, f.buffer(vk.HostVisible, t, cfg.Hidden))
		}
	}
	if f.err != nil {
		return nil, nil, f.err
	}
	segments := make([]float32, 2*t)
	start := 0
	for _, n := range lengths {
		end := start + n
		for row := start; row < end; row++ {
			segments[2*row] = float32(start)
			segments[2*row+1] = float32(end)
		}
		start = end
	}
	f.segments = f.mapped(segments)
	for _, l := range a.layers {
		if _, ok := f.tables[l.rotation]; !ok {
			cs, sn := embeddinggemma2.RopeTables(longest, l.dim, l.theta)
			f.tables[l.rotation] = [2]*vk.Buffer{f.mapped(cs), f.mapped(sn)}
		}
	}
	if f.err != nil {
		return nil, nil, f.err
	}
	plan := f.plan(keep)
	if plan.err != nil {
		return nil, nil, plan.err
	}
	for i, group := range plan.groups {
		submitted = true
		if err := a.device.Run(func(c *vk.Commands) error {
			for _, command := range group.commands {
				if err := command(c); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return nil, nil, fmt.Errorf("vulkan: submission %d/%d (%d estimated work units): %w", i+1, len(plan.groups), group.units, err)
		}
	}
	out = make([]float32, t*cfg.EmbeddingDim)
	if err := f.download.ReadF32(0, out); err != nil {
		return nil, nil, err
	}
	for _, b := range f.snapshots {
		state := make([]float32, t*cfg.Hidden)
		if err := b.ReadF32(0, state); err != nil {
			return nil, nil, err
		}
		hidden = append(hidden, state)
	}
	return out, hidden, nil
}

func (f *forward) plan(keep bool) workPlan {
	a, t, cfg := f.a, f.rows, f.a.cfg
	r := recording{f: f, plan: workPlan{budget: a.dispatchBudget}}
	stateBytes := product(t, cfg.Hidden, 4)
	r.copy(f.input, f.x, stateBytes)
	if keep {
		r.copy(f.x, f.snapshots[0], stateBytes)
	}
	r.project(f.x, a.pleProj, f.ple, cfg.Hidden, cfg.Layers*cfg.PLEDim)
	r.mix(f.ple, a.ones, t*cfg.Layers*cfg.PLEDim, scale, float32(1/math.Sqrt(float64(cfg.Hidden))), 0)
	r.norm(f.ple, a.pleNorm, f.ple, t*cfg.Layers, cfg.PLEDim)
	for i := range cfg.Layers {
		r.layer(i)
		if keep {
			r.copy(f.x, f.snapshots[i+1], stateBytes)
		}
	}
	r.norm(f.x, a.finalNorm, f.normalized, t, cfg.Hidden)
	r.project(f.normalized, a.projection, f.output, cfg.Hidden, cfg.EmbeddingDim)
	r.copy(f.output, f.download, product(t, cfg.EmbeddingDim, 4))
	return r.plan
}

type recording struct {
	f    *forward
	plan workPlan
}

func (r *recording) copy(src, dst *vk.Buffer, bytes uint64) {
	r.plan.add("copy", bytes, false, func(c *vk.Commands) error {
		return c.Copy(src, dst, 0, 0, bytes)
	})
}
func ceilDiv(n, div int) int     { return 1 + (n-1)/div }
func grid(x, y, z int) [3]uint32 { return [3]uint32{uint32(x), uint32(y), uint32(z)} }
func (r *recording) linear(groups int) [3]uint32 {
	x := min(groups, r.f.maxGroupX)
	return grid(x, ceilDiv(groups, x), 1)
}
func (r *recording) dispatch(name string, units uint64, shape [3]uint32, push []uint32, buffers ...*vk.Buffer) {
	r.plan.add(name, units, true, func(c *vk.Commands) error {
		if err := c.Dispatch(r.f.a.pipes[name], buffers, push, shape); err != nil {
			return fmt.Errorf("vulkan: %s: %w", name, err)
		}
		return nil
	})
}
func (r *recording) project(src, w, dst *vk.Buffer, in, out int) {
	perGroup := product(8, tile(out, 16), tile(in, 16))
	maxRows := 8 * int(submissionWork/perGroup)
	if maxRows == 0 {
		if r.plan.err == nil {
			r.plan.err = fmt.Errorf("vulkan: one projection row exceeds the submission work limit")
		}
		return
	}
	for start := 0; start < r.f.rows; start += maxRows {
		rows := min(maxRows, r.f.rows-start)
		r.dispatch("project", product(tile(rows, 8), tile(out, 16), tile(in, 16)),
			grid(ceilDiv(out, 16), ceilDiv(rows, 8), 1),
			[]uint32{uint32(start), uint32(rows), uint32(out), uint32(in)}, src, w, dst)
	}
}
func (r *recording) norm(src, w, dst *vk.Buffer, rows, width int) {
	eps := math.Float32bits(float32(r.f.a.cfg.RMSNormEps))
	g := r.linear(rows)
	r.dispatch("norm", product(int(g[0]), int(g[1]), 4*tile(width, 64)+64), g, []uint32{uint32(rows), uint32(width), eps}, src, w, dst)
}

const (
	scale = iota
	residual
	geluProduct
	pleProduct
)

func (r *recording) mix(dst, src *vk.Buffer, n, mode int, factor float32, layer int) {
	cfg := r.f.a.cfg
	g := r.linear(ceilDiv(n, 64))
	perLane := 3
	if mode == geluProduct || mode == pleProduct {
		perLane = 12
	}
	r.dispatch("mix", product(int(g[0]), int(g[1]), 64, perLane), g, []uint32{uint32(n), uint32(mode), math.Float32bits(factor), uint32(cfg.PLEDim), uint32(cfg.Layers), uint32(layer)}, dst, src)
}
func (r *recording) rotate(x *vk.Buffer, l layer, heads int) {
	table := r.f.tables[l.rotation]
	pairs := r.f.rows * heads * (l.dim / 2)
	g := r.linear(ceilDiv(pairs, 64))
	r.dispatch("rope", product(int(g[0]), int(g[1]), 64, 8), g, []uint32{uint32(pairs), uint32(heads), uint32(l.dim)}, x, table[0], table[1], r.f.segments)
}
func (r *recording) layer(i int) {
	f := r.f
	l := f.a.layers[i]
	cfg := f.a.cfg
	t, h, p := f.rows, cfg.Hidden, cfg.PLEDim
	qd, kvd := cfg.Heads*l.dim, l.kv*l.dim
	r.norm(f.x, l.inNorm, f.normalized, t, h)
	r.project(f.normalized, l.q, f.q, h, qd)
	r.project(f.normalized, l.k, f.k, h, kvd)
	r.project(f.normalized, l.v, f.v, h, kvd)
	r.norm(f.q, l.qNorm, f.q, t*cfg.Heads, l.dim)
	r.norm(f.k, l.kNorm, f.k, t*l.kv, l.dim)
	r.norm(f.v, f.a.ones, f.v, t*l.kv, l.dim)
	r.rotate(f.q, l, cfg.Heads)
	r.rotate(f.k, l, l.kv)
	radius := min(t, cfg.SlidingWindow)
	if l.full {
		radius = t
	}
	err := attentionBlocks(f.lengths, cfg.Heads, l.dim, radius, func(b attentionBlock) {
		args := []uint32{uint32(b.query), uint32(b.rows), uint32(cfg.Heads), uint32(l.kv), uint32(l.dim),
			uint32(b.key), uint32(b.keys), uint32(b.sequence), uint32(radius)}
		r.dispatch("scores", b.scores(cfg.Heads, l.dim), grid(ceilDiv(b.keys, 16), ceilDiv(b.rows, 8), cfg.Heads), args, f.q, f.k, f.scores)
		r.dispatch("softmax", b.softmax(cfg.Heads), grid(b.rows, cfg.Heads, 1), args, f.scores)
		r.dispatch("values", b.values(cfg.Heads, l.dim), grid(ceilDiv(l.dim, 16), ceilDiv(b.rows, 8), cfg.Heads), args, f.scores, f.v, f.context)
	})
	if err != nil && r.plan.err == nil {
		r.plan.err = err
	}
	r.project(f.context, l.o, f.branch, qd, h)
	r.norm(f.branch, l.postAttn, f.branch, t, h)
	r.mix(f.x, f.branch, t*h, residual, 1, 0)

	r.norm(f.x, l.preFF, f.normalized, t, h)
	r.project(f.normalized, l.gate, f.gate, h, cfg.Intermediate)
	r.project(f.normalized, l.up, f.up, h, cfg.Intermediate)
	r.mix(f.gate, f.up, t*cfg.Intermediate, geluProduct, 1, 0)
	r.project(f.gate, l.down, f.branch, cfg.Intermediate, h)
	r.norm(f.branch, l.postFF, f.branch, t, h)
	r.mix(f.x, f.branch, t*h, residual, 1, 0)

	r.project(f.x, l.pleGate, f.pleGate, h, p)
	r.mix(f.pleGate, f.ple, t*p, pleProduct, 1, i)
	r.project(f.pleGate, l.pleProj, f.branch, p, h)
	r.norm(f.branch, l.plePost, f.branch, t, h)
	r.mix(f.x, f.branch, t*h, residual, l.scalar, 0)
}
