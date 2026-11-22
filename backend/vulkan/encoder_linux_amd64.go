package vulkan

import (
	"embed"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

//go:embed shaders/*.spv
var shaders embed.FS

type encoder struct {
	mu                                            sync.Mutex // all forwards and Close share the device and its command pool
	model                                         *embeddinggemma2.Model
	cfg                                           embeddinggemma2.Config
	device                                        *vk.Device
	broken                                        error // first non-lost rollback failure that left a temporary buffer unreleased
	identity                                      DeviceInfo
	pipes                                         map[string]*vk.Pipeline
	pleProj, pleNorm, finalNorm, projection, ones *vk.Buffer
	layers                                        []layer
	maxQ, maxKV                                   int
	dispatchBudget                                int // descriptor sets available in one command buffer
}
type rotation struct {
	dim   int
	theta float64
}
type layer struct {
	rotation
	full                                       bool
	kv                                         int
	scalar                                     float32
	inNorm, q, k, v, o, qNorm, kNorm, postAttn *vk.Buffer
	preFF, gate, up, down, postFF              *vk.Buffer
	pleGate, pleProj, plePost                  *vk.Buffer
}

func openAccelerator(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
	return attach(m, vk.Selector{})
}

func attach(m *embeddinggemma2.Model, selector vk.Selector) (embeddinggemma2.Accelerator, error) {
	return attachWithOpener(m, func() (*vk.Device, error) { return vk.Open(selector) })
}

func attachWithOpener(m *embeddinggemma2.Model, openDev func() (*vk.Device, error)) (embeddinggemma2.Accelerator, error) {
	a, err := newEncoderWithOpener(m, openDev)
	if err != nil {
		return nil, classify(err)
	}
	return a, nil
}
func newEncoder(m *embeddinggemma2.Model, selector vk.Selector) (*encoder, error) {
	return newEncoderWithOpener(m, func() (*vk.Device, error) { return vk.Open(selector) })
}
func newEncoderWithOpener(m *embeddinggemma2.Model, openDev func() (*vk.Device, error)) (*encoder, error) {
	if m == nil {
		return nil, errors.New("vulkan: nil model")
	}
	w := m.TransformerWeights()
	cfg := w.Config
	if cfg.Hidden <= 0 || cfg.PLEDim <= 0 || cfg.Intermediate <= 0 || cfg.Heads <= 0 || len(w.Layers) == 0 || cfg.Layers != len(w.Layers) || cfg.SlidingWindow < 0 {
		return nil, errors.New("vulkan: invalid encoder geometry")
	}
	a := &encoder{model: m, cfg: cfg, pipes: make(map[string]*vk.Pipeline), dispatchBudget: 4096}
	maxDim := 0
	for i, l := range w.Layers {
		if l.HeadDim <= 0 || l.HeadDim%2 != 0 || l.KVHeads <= 0 || cfg.Heads%l.KVHeads != 0 || l.RopeTheta <= 0 {
			return nil, fmt.Errorf("vulkan: invalid head geometry in layer %d", i)
		}
		maxDim = max(maxDim, l.HeadDim)
		a.maxQ = max(a.maxQ, cfg.Heads*l.HeadDim)
		a.maxKV = max(a.maxKV, l.KVHeads*l.HeadDim)
	}
	d, err := openDev()
	if err != nil {
		return nil, err
	}
	a.device = d
	info := d.Info()
	a.identity = describeDevice(info)
	fail := func(err error) (*encoder, error) { return nil, errors.Join(err, d.Close()) }
	for _, s := range []struct {
		name            string
		bindings, words int
	}{
		{"project", 3, 4}, {"norm", 3, 3}, {"mix", 2, 6}, {"rope", 4, 3}, {"scores", 3, 9}, {"softmax", 1, 9}, {"values", 3, 9},
	} {
		code, err := shaders.ReadFile("shaders/" + s.name + ".spv")
		if err != nil {
			return fail(err)
		}
		p, err := d.NewPipeline(code, s.bindings, uint32(4*s.words))
		if err != nil {
			return fail(fmt.Errorf("vulkan: %s pipeline: %w", s.name, err))
		}
		a.pipes[s.name] = p
	}
	ones := make([]float32, maxDim)
	for i := range ones {
		ones[i] = 1
	}
	if err := a.upload([]weight{{&a.ones, ones}, {&a.pleProj, w.PLEProj}, {&a.pleNorm, w.PLENorm}, {&a.finalNorm, w.FinalNorm}, {&a.projection, w.Projection}}); err != nil {
		return fail(err)
	}
	for i, l := range w.Layers {
		g := layer{rotation: rotation{l.HeadDim, l.RopeTheta}, full: l.Full, kv: l.KVHeads, scalar: l.LayerScalar}
		err := a.upload([]weight{
			{&g.inNorm, l.InNorm}, {&g.q, l.Q}, {&g.k, l.K}, {&g.v, l.V}, {&g.o, l.O}, {&g.qNorm, l.QNorm}, {&g.kNorm, l.KNorm}, {&g.postAttn, l.PostAttnNorm},
			{&g.preFF, l.PreFFNorm}, {&g.gate, l.Gate}, {&g.up, l.Up}, {&g.down, l.Down}, {&g.postFF, l.PostFFNorm},
			{&g.pleGate, l.PLEGate}, {&g.pleProj, l.PLEProj}, {&g.plePost, l.PLEPostNorm},
		})
		if err != nil {
			return fail(fmt.Errorf("vulkan: layer %d upload: %w", i, err))
		}
		a.layers = append(a.layers, g)
	}
	return a, nil
}

type weight struct {
	dst    **vk.Buffer
	values []float32
}

// Upload a group per layer so staging dies at the end of its fence instead of
// retaining a second copy of the whole checkpoint.
func (a *encoder) upload(group []weight) (err error) {
	var staging []*vk.Buffer
	defer func() {
		for _, b := range staging {
			err = errors.Join(err, b.Close())
		}
	}()
	for _, w := range group {
		size := uint64(len(w.values)) * 4
		if size == 0 || size > a.device.Info().MaxStorageRange {
			return fmt.Errorf("vulkan: weight of %d bytes exceeds storage range or is empty", size)
		}
		dst, err := a.device.NewBuffer(size, vk.DeviceLocal)
		if err != nil {
			return err
		}
		*w.dst = dst // device owns child buffers and pipelines until Close
		src, err := a.device.NewBuffer(size, vk.HostVisible)
		if err != nil {
			return err
		}
		staging = append(staging, src)
		if err := src.WriteF32(0, w.values); err != nil {
			return err
		}
	}
	return a.device.Run(func(c *vk.Commands) error {
		for i, w := range group {
			if err := c.Copy(staging[i], *w.dst, 0, 0, staging[i].Size()); err != nil {
				return err
			}
		}
		return nil
	})
}

func (*encoder) Name() string { return "vulkan" }

func (a *encoder) attachedDevice() (DeviceInfo, bool) {
	if a == nil {
		return DeviceInfo{}, false
	}
	return a.identity, true
}

func (a *encoder) DeviceMemory() (embeddinggemma2.DeviceMemory, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil {
		return embeddinggemma2.DeviceMemory{}, errors.New("vulkan: encoder is closed")
	}
	total, available, known, err := a.device.Memory()
	if err != nil {
		return embeddinggemma2.DeviceMemory{}, classify(err)
	}
	return deviceMemory(total, available, known), nil
}

func (a *encoder) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil {
		return nil
	}
	if err := a.device.Close(); err != nil {
		return err
	}
	a.device = nil
	a.broken = nil
	a.model = nil
	a.layers = nil
	a.pipes = nil
	return nil
}

func (a *encoder) validateRows(rows int) error {
	if rows <= 0 || rows > embeddinggemma2.MaxTokens || (a.cfg.MaxPositions > 0 && rows > a.cfg.MaxPositions) {
		return fmt.Errorf("vulkan: unsupported input length %d", rows)
	}
	return nil
}

func (a *encoder) status() error {
	if a.device == nil {
		return errors.New("vulkan: encoder is closed")
	}
	if err := a.device.Status(); err != nil {
		if errors.Is(err, vk.ErrDeviceLost) {
			return classify(err)
		}
		if a.broken != nil {
			return a.broken
		}
		return classify(err)
	}
	if a.broken != nil {
		return a.broken
	}
	return nil
}

func (a *encoder) Forward(ids []int, keep bool) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.status(); err != nil {
		return nil, nil, err
	}
	if err := a.validateRows(len(ids)); err != nil {
		return nil, nil, err
	}
	x, err := a.model.EmbedTokens(ids)
	if err != nil {
		return nil, nil, err
	}
	return a.forwardRows(x, []int{len(ids)}, keep)
}

func (a *encoder) ForwardEmbeds(x []float32, rows int, keep bool) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.status(); err != nil {
		return nil, nil, err
	}
	if err := a.validateRows(rows); err != nil {
		return nil, nil, err
	}
	if len(x) != rows*a.cfg.Hidden {
		return nil, nil, fmt.Errorf("vulkan: got %d values for %d input rows", len(x), rows)
	}
	return a.forwardRows(x, []int{rows}, keep)
}

// ForwardBatch packs the independent sequences into one row-major forward.
// RoPE positions restart at each boundary; every attention row masks foreign keys.
func (a *encoder) ForwardBatch(ids [][]int) ([][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.status(); err != nil {
		return nil, err
	}
	if len(ids) == 0 || len(ids) > embeddinggemma2.MaxTokens {
		return nil, fmt.Errorf("vulkan: invalid batch size %d", len(ids))
	}
	lengths := make([]int, len(ids))
	total := 0
	for i, in := range ids {
		if err := a.validateRows(len(in)); err != nil {
			return nil, fmt.Errorf("vulkan: batch input %d: %w", i, err)
		}
		if len(in) > embeddinggemma2.MaxTokens-total {
			return nil, fmt.Errorf("vulkan: batch exceeds %d positions", embeddinggemma2.MaxTokens)
		}
		lengths[i] = len(in)
		total += len(in)
	}
	flat := make([]int, 0, total)
	for _, in := range ids {
		flat = append(flat, in...)
	}
	x, err := a.model.EmbedTokens(flat)
	if err != nil {
		return nil, err
	}
	rows, _, err := a.forwardRows(x, lengths, false)
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(ids))
	start := 0
	for i, n := range lengths {
		end := start + n*a.cfg.EmbeddingDim
		out[i] = rows[start:end:end]
		start = end
	}
	return out, nil
}

// byteSize keeps all GLSL float indices representable and checks the signed host product.
func byteSize(maxBytes uint64, dims ...int) (uint64, error) {
	n := uint64(4)
	limit := min(maxBytes, uint64(math.MaxUint32)*4)
	for _, d := range dims {
		if d <= 0 || uint64(d) > limit/n {
			return 0, fmt.Errorf("vulkan: tensor %v exceeds storage range %d or has an empty dimension", dims, maxBytes)
		}
		n *= uint64(d)
	}
	return n, nil
}
