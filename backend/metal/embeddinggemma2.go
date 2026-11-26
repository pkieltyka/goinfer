//go:build darwin

package metal

import (
	"errors"
	"fmt"
	"math"
	"sync"

	gpu "github.com/pkieltyka/goinfer/backend/metal/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

// EmbeddingGemma 2's text encoder on Metal: the CPU forward of
// embeddinggemma2/model.go in float32 on the GPU, one command buffer per input. The GEMMs and the row RMSNorm are
// the device layer's ViT kernels (an f32 simdgroup GEMM of A·Bᵀ with B as nn.Linear stores it, [N, K]); attention runs as those
// GEMMs over blocks of query rows (attention, below), and this file adds the windowed row softmax and the data
// movement around it, RoPE from the CPU's own tables, the GELU-tanh multiplies and the residual adds. The embedding
// gather and pooling stay on the host. Registered as the "metal" accelerator.

func init() {
	embeddinggemma2.RegisterAccelerator("metal", func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return newEG2Accel(m)
	})
	embeddinggemma2.RegisterAccelerator("metal-eg2-stock", func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return newEG2AccelWithAttention(m, eg2AttentionStock)
	})
	embeddinggemma2.RegisterAccelerator("metal-eg2-combined", func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return newEG2AccelWithAttention(m, eg2AttentionMatrix)
	})
	embeddinggemma2.RegisterAccelerator("metal-eg2-batch", func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		a, err := newEG2AccelWithAttention(m, eg2AttentionBatch)
		if err != nil {
			return nil, err
		}
		return &eg2BatchAccel{eg2Accel: a}, nil
	})
	embeddinggemma2.RegisterDeviceProber("metal", probeMetalDevice)
}

const eg2TG = 128

type eg2Pipes struct{ geluMul, geluMulPLE, add, addScale, rope, headMajor, vtBlock, softmax, scatter Pipeline }

// eg2Block is how many query rows one attention block takes. A var only so a test can make blocks smaller than the
// tiny fixture's inputs and the sliding window.
var eg2Block = 256

type eg2Layer struct {
	w                                  embeddinggemma2.LayerWeights
	inNorm, q, k, v, o, qNorm, kNorm   Buffer
	postAttnNorm, preFFNorm, gate, up  Buffer
	down, postFFNorm, pleGate, pleProj Buffer
	plePostNorm                        Buffer
}

// eg2Ops is what the text encoder and the vision tower (gemma4_vision.go) share: the device, a queue, the device layer's ViT
// kernels, this file's pipelines and the scalar arena.
type eg2Ops struct {
	d     *Device
	q     Queue
	vit   gpu.ViT
	p     eg2Pipes
	arena Buffer // one scalar per 256-byte slot, so no two dispatches in a command buffer share one
	slot  int
	eps   float32
	w32   Pipeline // tower_gemm_w32: gemm's kernel
	// vitGemmForTest runs gemm on gpudev's GEMMF32Plan kernel, which tower_gemm_w32 matches byte for byte: the old arm of
	// its A/B.
	vitGemmForTest bool
}

// eg2AttnBufs are attention's inputs, scratch and output: q, k, v [T, heads, hd]; their head-major copies; a block's
// scores and transposed values and output; ctx [T, nH*hd].
type eg2AttnBufs struct{ q, k, v, qh, kh, vh, sc, vt, ob, ctx Buffer }

// newEG2Ops opens the device and compiles this file's kernels with extra (more kernels in the same precise library),
// whose pipelines it builds into more, by kernel name.
func newEG2Ops(extra string, more map[string]*Pipeline) (eg2Ops, error) {
	return newEG2OpsOn(CreateSystemDefaultDevice, extra, more)
}

func newEG2OpsOn(open func() (*Device, error), extra string, more map[string]*Pipeline) (_ eg2Ops, err error) {
	d, err := acquireMetalDevice(open)
	if err != nil {
		return eg2Ops{}, err
	}
	defer func() {
		if err != nil {
			d.ReleaseAll()
			d.ReleaseObjects()
		}
	}()
	vit, err := d.NewViT()
	if err != nil {
		return eg2Ops{}, err
	}
	lib, err := d.CompileLibraryPrecise(eg2MSL+towerGemmMSL+extra, MSL3_1)
	if err != nil {
		return eg2Ops{}, fmt.Errorf("metal: compile the EmbeddingGemma 2 kernels: %w", err)
	}
	o := eg2Ops{d: d, q: d.NewCommandQueue(), vit: vit}
	for _, b := range []struct {
		name string
		dst  *Pipeline
	}{{"eg2_gelu_mul", &o.p.geluMul}, {"eg2_gelu_mul_ple", &o.p.geluMulPLE}, {"eg2_add", &o.p.add},
		{"eg2_add_scale", &o.p.addScale}, {"eg2_rope", &o.p.rope}, {"eg2_headmajor", &o.p.headMajor},
		{"eg2_vt_block", &o.p.vtBlock}, {"eg2_softmax_rows", &o.p.softmax}, {"eg2_scatter_head", &o.p.scatter},
		{"tower_gemm_w32", &o.w32}} {
		if *b.dst, err = d.NewComputePipeline(lib, b.name); err != nil {
			return eg2Ops{}, err
		}
	}
	for name, dst := range more {
		if *dst, err = d.NewComputePipeline(lib, name); err != nil {
			return eg2Ops{}, err
		}
	}
	var ok bool
	o.arena, ok = d.TryNewBufferBytesOn(mem.LineCompanion, eg2ArenaSlots*256)
	if !ok {
		return eg2Ops{}, fmt.Errorf("%w: Metal EG2 scalar arena", embeddinggemma2.ErrDeviceOutOfMemory)
	}
	return o, nil
}

// errTowerClosed is what a forward returns after Close.
var errTowerClosed = errors.New("metal: the accelerator is closed")

// open reports errTowerClosed once closeDevice has run. The caller holds the accelerator's lock.
func (o *eg2Ops) open() error {
	if o.d == nil {
		return errTowerClosed
	}
	return nil
}

// release frees each buffer from the device ledger and zeroes it. The caller holds the accelerator's lock, and every
// forward waits for its command buffers, so no GPU work reads them.
func (o *eg2Ops) release(bufs ...*Buffer) {
	for _, b := range bufs {
		o.d.ReleaseBuf(*b)
		*b = Buffer{}
	}
}

// closeDevice releases every buffer and object on the device ledger (weights, scratch, pipelines, the queue) and the
// device handle. Idempotent; the caller holds the accelerator's lock.
func (o *eg2Ops) closeDevice() {
	if o.d == nil {
		return
	}
	o.d.ReleaseAll()
	o.d.ReleaseObjects()
	o.d = nil
}

// eg2Accel is the Metal accelerator. One forward at a time (mu): its scratch and scalar arena are shared.
type eg2Accel struct {
	mu sync.Mutex
	eg2Ops
	cfg         embeddinggemma2.Config
	model       *embeddinggemma2.Model
	identity    embeddinggemma2.DeviceInfo
	profile     string
	matrix      eg2MatrixPipes
	stagedFFN   Pipeline
	pleProj     Buffer // scaled by hidden^-0.5 at upload, as the CPU scales the product
	pleNorm     Buffer
	finalNorm   Buffer
	proj        Buffer
	ones        Buffer // the unweighted v norm
	layers      []eg2Layer
	ropes       map[eg2RopeKey][2]Buffer // immutable cos/sin tables, by input length, head dim and theta
	cap         int                      // scratch rows allocated
	scr         eg2Scratch
	kvHeads     []int // per-layer KV heads, for scalar-arena sizing
	weightBytes int64 // actual device charges for the batch selector's uploaded model tensors
}

type eg2Scratch struct{ x, xn, q, k, v, ctx, o, g, u, f, gg, pr, ple, out, qh, kh, vh, sc, vt, ob Buffer }

type eg2RopeKey struct {
	T, hd int
	theta float64
}

const eg2ArenaSlots = 1 << 14 // initial slots: 4 MiB, enough for short forwards

// Text-forward scalar-arena sizing. The counts are the production tower_gemm_w32
// command's slots, so they are a conservative upper bound over the test-only
// 3-scalar GEMM. One slot is 256 bytes.
const (
	eg2ArenaOpeningSlots = 8  // prologue: PLE projection gemm
	eg2ArenaClosingSlots = 8  // epilogue: final norm and projection
	eg2ArenaLayerSlots   = 93 // a layer outside its attention blocks
	eg2ArenaVTSlots      = 5  // vtBlock, per KV group per query block
	eg2ArenaHeadSlots    = 19 // two W32 gemms (5 each), softmax (4), scatter (5)
)

// eg2ArenaAllocForTest, when set, replaces the device allocation in ensureArena
// so a test can inject a refusal. Test seam, nil in production.
var eg2ArenaAllocForTest func(bytes int) (Buffer, bool)

func eg2ArenaCheckedAdd(a, b int64) (int64, error) {
	if b < 0 || a > math.MaxInt64-b {
		return 0, fmt.Errorf("metal: EmbeddingGemma 2 scalar arena size overflows")
	}
	return a + b, nil
}

func eg2ArenaCheckedMul(a, b int64) (int64, error) {
	if a < 0 || b < 0 || (a != 0 && b > math.MaxInt64/a) {
		return 0, fmt.Errorf("metal: EmbeddingGemma 2 scalar arena size overflows")
	}
	return a * b, nil
}

// eg2ArenaSlotsFor returns the scalar slots one text forward of T rows needs on
// the production W32 path. keepLayers bounds a single command interval instead
// of the whole forward: opening plus the first layer, each later layer alone,
// or the closing segment.
func eg2ArenaSlotsFor(c embeddinggemma2.Config, kvHeads []int, T int, keepLayers bool) (int, error) {
	if T <= 0 {
		return 0, fmt.Errorf("metal: EmbeddingGemma 2 scalar arena needs a positive row count, got %d", T)
	}
	if eg2Block <= 0 || c.Heads <= 0 || c.Layers <= 0 || len(kvHeads) != c.Layers {
		return 0, fmt.Errorf("metal: EmbeddingGemma 2 scalar arena geometry: block %d, heads %d, layers %d, kv heads %d",
			eg2Block, c.Heads, c.Layers, len(kvHeads))
	}
	blocks, err := eg2ArenaCheckedAdd(1, int64((T-1)/eg2Block))
	if err != nil {
		return 0, err
	}
	head, err := eg2ArenaCheckedMul(eg2ArenaHeadSlots, int64(c.Heads))
	if err != nil {
		return 0, err
	}
	total := int64(0)
	opening, closing := int64(eg2ArenaOpeningSlots), int64(eg2ArenaClosingSlots)
	var first, maxLayer int64
	for i, kv := range kvHeads {
		if kv <= 0 {
			return 0, fmt.Errorf("metal: EmbeddingGemma 2 scalar arena layer %d has %d KV heads", i, kv)
		}
		vt, err := eg2ArenaCheckedMul(eg2ArenaVTSlots, int64(kv))
		if err != nil {
			return 0, err
		}
		perBlock, err := eg2ArenaCheckedAdd(head, vt)
		if err != nil {
			return 0, err
		}
		blocksCost, err := eg2ArenaCheckedMul(blocks, perBlock)
		if err != nil {
			return 0, err
		}
		layer, err := eg2ArenaCheckedAdd(eg2ArenaLayerSlots, blocksCost)
		if err != nil {
			return 0, err
		}
		if i == 0 {
			first = layer
		}
		if layer > maxLayer {
			maxLayer = layer
		}
		total, err = eg2ArenaCheckedAdd(total, layer)
		if err != nil {
			return 0, err
		}
	}
	if !keepLayers {
		total, err = eg2ArenaCheckedAdd(total, opening)
		if err != nil {
			return 0, err
		}
		total, err = eg2ArenaCheckedAdd(total, closing)
		if err != nil {
			return 0, err
		}
	} else {
		openFirst, err := eg2ArenaCheckedAdd(opening, first)
		if err != nil {
			return 0, err
		}
		total = openFirst
		if maxLayer > total {
			total = maxLayer
		}
		if closing > total {
			total = closing
		}
	}
	if total > int64(math.MaxInt)/256 {
		return 0, fmt.Errorf("metal: EmbeddingGemma 2 scalar arena needs %d slots, over the addressable range", total)
	}
	return int(total), nil
}

// ensureArena grows the scalar arena to slots when the current buffer is
// smaller. It must run under the accelerator mutex before any encoding. A
// refusal leaves the current arena in place and returns an error, so a smaller
// forward can still run.
func (a *eg2Ops) ensureArena(slots int) error {
	if slots <= 0 {
		return fmt.Errorf("metal: EmbeddingGemma 2 scalar arena needs a positive slot count, got %d", slots)
	}
	need64, err := eg2ArenaCheckedMul(int64(slots), 256)
	if err != nil || need64 > int64(math.MaxInt) {
		return fmt.Errorf("metal: EmbeddingGemma 2 scalar arena %d slots is over the addressable range", slots)
	}
	need := int(need64)
	if a.arena.Len() >= need {
		return nil
	}
	var next Buffer
	var ok bool
	if eg2ArenaAllocForTest != nil {
		next, ok = eg2ArenaAllocForTest(need)
	} else {
		next, ok = a.d.TryNewBufferBytesOn(mem.LineCompanion, need)
	}
	if !ok || next.Len() < need {
		if next.Len() > 0 {
			a.d.ReleaseBuf(next)
		}
		return fmt.Errorf("%w: Metal EG2 needs %d scalar slots (%d bytes)", embeddinggemma2.ErrDeviceOutOfMemory, slots, need)
	}
	old := a.arena
	a.arena = next
	if old.Len() > 0 {
		a.d.ReleaseBuf(old)
	}
	return nil
}

func newEG2Accel(m *embeddinggemma2.Model) (*eg2Accel, error) {
	return newEG2AccelWithAttention(m, eg2AttentionDefault)
}

func newEG2AccelWithAttention(m *embeddinggemma2.Model, request eg2AttentionRequest) (_ *eg2Accel, err error) {
	w := m.TransformerWeights()
	for _, ly := range w.Layers {
		if ly.HeadDim > 512 {
			return nil, fmt.Errorf("metal: EmbeddingGemma 2 head dim %d, over the attention kernel's 512", ly.HeadDim)
		}
	}
	ops, err := newEG2Ops("", nil)
	if err != nil {
		return nil, err
	}
	d := ops.d
	a := &eg2Accel{eg2Ops: ops, identity: describeMetalDevice(d.Name(), d.HasUnifiedMemory(), false), ropes: map[eg2RopeKey][2]Buffer{}}
	defer func() {
		if err != nil {
			a.closeDevice()
		}
	}()
	a.profile, err = resolveEG2Attention(request, eg2MatrixGeometry(w), func() error {
		var setupErr error
		a.matrix, setupErr = a.newEG2MatrixPipes()
		return setupErr
	})
	if err != nil {
		return nil, err
	}
	// Matrix attachment already established the unified-memory Apple7 capability.
	if a.profile == eg2MatrixProfile && eg2StagedFFNEligible(d.Name(), d.ArchitectureName(), a.profile, w.Config) {
		a.stagedFFN, err = newEG2StagedFFN(d)
		if err != nil {
			return nil, err
		}
	}
	if request == eg2AttentionBatch && !eg2BatchGeometry(a.stagedFFN != (Pipeline{}), w) {
		return nil, fmt.Errorf("%w: Metal batching requires the base Apple M4 combined text graph", embeddinggemma2.ErrUnsupportedSelector)
	}
	a.cfg, a.model = w.Config, m
	a.eps = float32(a.cfg.RMSNormEps)
	up := func(v []float32) Buffer {
		if err != nil {
			return Buffer{}
		}
		b, ok := gpu.TryNewBufferOfOn(d, mem.LineCompanion, v)
		if !ok {
			err = fmt.Errorf("%w: Metal EG2 weight upload (%d float32 values)", embeddinggemma2.ErrDeviceOutOfMemory, len(v))
		}
		if request == eg2AttentionBatch {
			a.weightBytes += b.AllocatedBytes()
		}
		return b
	}
	ps := float32(1 / math.Sqrt(float64(a.cfg.Hidden)))
	scaled := make([]float32, len(w.PLEProj))
	for i, x := range w.PLEProj {
		scaled[i] = x * ps
	}
	a.pleProj, a.pleNorm, a.finalNorm, a.proj = up(scaled), up(w.PLENorm), up(w.FinalNorm), up(w.Projection)
	ones := make([]float32, 512)
	for i := range ones {
		ones[i] = 1
	}
	a.ones = up(ones)
	if request == eg2AttentionBatch {
		a.weightBytes -= a.ones.AllocatedBytes()
	}
	for _, ly := range w.Layers {
		a.kvHeads = append(a.kvHeads, ly.KVHeads)
		a.layers = append(a.layers, eg2Layer{w: ly, inNorm: up(ly.InNorm), q: up(ly.Q), k: up(ly.K), v: up(ly.V), o: up(ly.O),
			qNorm: up(ly.QNorm), kNorm: up(ly.KNorm), postAttnNorm: up(ly.PostAttnNorm), preFFNorm: up(ly.PreFFNorm),
			gate: up(ly.Gate), up: up(ly.Up), down: up(ly.Down), postFFNorm: up(ly.PostFFNorm), pleGate: up(ly.PLEGate),
			pleProj: up(ly.PLEProj), plePostNorm: up(ly.PLEPostNorm)})
	}
	if err != nil {
		return nil, err
	}
	return a, nil
}

func (a *eg2Accel) Name() string { return "metal" }

var _ embeddinggemma2.AttachedProfileReporter = (*eg2Accel)(nil)
var _ embeddinggemma2.BatchMemoryEstimator = (*eg2Accel)(nil)

// AttachedProfile identifies the numerical route fixed at attachment.
func (a *eg2Accel) AttachedProfile() string { return a.profile }

func (a *eg2Accel) checkLength(T int) error {
	limit := embeddinggemma2.MaxTokens
	if a.cfg.MaxPositions > 0 {
		limit = min(limit, a.cfg.MaxPositions)
	}
	if T > limit {
		return fmt.Errorf("%w: Metal EG2 input has %d tokens, over limit %d", embeddinggemma2.ErrUnsupportedBatchShape, T, limit)
	}
	return nil
}

// BatchMemory checks singleton admission and reports unknown allocation costs.
func (a *eg2Accel) BatchMemory(lengths []int) (embeddinggemma2.BatchMemoryEstimate, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.open(); err != nil {
		return embeddinggemma2.BatchMemoryEstimate{}, err
	}
	if len(lengths) == 0 || lengths[0] <= 0 {
		return embeddinggemma2.BatchMemoryEstimate{}, errors.New("metal: EG2 needs a positive input length")
	}
	if len(lengths) != 1 {
		return embeddinggemma2.BatchMemoryEstimate{}, fmt.Errorf("%w: Metal EG2 runs single inputs", embeddinggemma2.ErrUnsupportedBatchShape)
	}
	if err := a.checkLength(lengths[0]); err != nil {
		return embeddinggemma2.BatchMemoryEstimate{}, err
	}
	return embeddinggemma2.BatchMemoryEstimate{Quality: embeddinggemma2.BatchMemoryUnknown}, nil
}

// Close frees the encoder's device memory; a forward after it is an error.
func (a *eg2Accel) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.closeDevice()
	a.layers, a.ropes, a.cap, a.scr = nil, nil, 0, eg2Scratch{}
	a.model = nil
	return nil
}

// u32 and f32 write one scalar into its own arena slot and return the slot to bind.
func (a *eg2Ops) u32(v uint32) Buffer {
	if cap := a.arena.Len() / 256; a.slot >= cap {
		panic("metal: EmbeddingGemma 2 scalar arena exhausted")
	}
	a.arena.U32s()[a.slot*64] = v
	b := a.arena.At(a.slot * 256)
	a.slot++
	return b
}
func (a *eg2Ops) f32(v float32) Buffer { return a.u32(math.Float32bits(v)) }

func (s *eg2Scratch) buffers() [20]*Buffer {
	return [20]*Buffer{&s.x, &s.xn, &s.q, &s.k, &s.v, &s.ctx, &s.o, &s.g, &s.u, &s.f,
		&s.gg, &s.pr, &s.ple, &s.out, &s.qh, &s.kh, &s.vh, &s.sc, &s.vt, &s.ob}
}

func (a *eg2Accel) scratchWords(rows int) [20]int {
	c := a.cfg
	maxQD, maxKVD, maxHD := 0, 0, 0
	for _, ly := range a.layers {
		maxQD = max(maxQD, c.Heads*ly.w.HeadDim)
		maxKVD = max(maxKVD, ly.w.KVHeads*ly.w.HeadDim)
		maxHD = max(maxHD, ly.w.HeadDim)
	}
	h, i, p, d := rows*c.Hidden, rows*c.Intermediate, rows*c.PLEDim, rows*c.EmbeddingDim
	q, kv, hd := rows*maxQD, rows*maxKVD, rows*maxHD
	block := min(eg2Block, rows)
	return [20]int{h, h, q, kv, kv, q, h, i, i, h, p, h, p * c.Layers, d, q, kv, kv, block * rows, hd, block * maxHD}
}

func (a *eg2Accel) releaseScratch(s *eg2Scratch) {
	bufs := s.buffers()
	a.release(bufs[:]...)
}

func (a *eg2Accel) newScratch(words [20]int, alloc func(int) (Buffer, bool)) (eg2Scratch, error) {
	var s eg2Scratch
	for i, b := range s.buffers() {
		var ok bool
		*b, ok = alloc(words[i])
		if !ok {
			a.releaseScratch(&s)
			return eg2Scratch{}, fmt.Errorf("%w: Metal text scratch buffer of %d float32 values", embeddinggemma2.ErrDeviceOutOfMemory, words[i])
		}
	}
	return s, nil
}

// grow sizes singleton scratch for T rows, releasing what it replaces.
func (a *eg2Accel) grow(T int) error {
	if err := a.open(); err != nil {
		return err
	}
	if T <= a.cap {
		return nil
	}
	a.releaseScratch(&a.scr)
	a.cap = 0
	next, err := a.newScratch(a.scratchWords(T), func(words int) (Buffer, bool) {
		return a.d.TryNewBufferLenOn(mem.LineCompanion, words)
	})
	if err != nil {
		return err
	}
	a.scr, a.cap = next, T
	return nil
}

// gemm is C[M,N] = A[M,K]·Bᵀ in f32 through tower_gemm_w32 (its bias bound but unread).
func (a *eg2Ops) gemm(e *Encoder, A, B, C Buffer, M, N, K int) {
	if a.vitGemmForTest {
		p, gx, gy, tgx, tgy := a.vit.GEMMF32Plan(M, N, K)
		e.Dispatch2D(p, gx, gy, tgx, tgy, A, B, C, a.u32(uint32(M)), a.u32(uint32(N)), a.u32(uint32(K)))
		return
	}
	e.Dispatch2D(a.w32, (N+63)/64, (M+63)/64, 512, 1, A, B, C, B, a.u32(uint32(M)), a.u32(uint32(N)), a.u32(uint32(K)),
		a.u32(0), a.u32(0))
}

func (a *eg2Ops) rms(e *Encoder, x, w, out Buffer, rows, dim int) {
	a.rmsEps(e, x, w, out, rows, dim, a.eps)
}

// rmsEps is rms at eps.
func (a *eg2Ops) rmsEps(e *Encoder, x, w, out Buffer, rows, dim int, eps float32) {
	e.Dispatch(a.vit.RMSNorm, rows*256, 256, x, w, out, a.u32(uint32(rows)), a.u32(uint32(dim)), a.f32(eps))
}

// Forward runs the encoder over ids (see embeddinggemma2.Accelerator): the embedding gather on the host, then
// ForwardEmbeds.
func (a *eg2Accel) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.open(); err != nil {
		return nil, nil, err
	}
	if a.profile == eg2MatrixProfile {
		if err := a.checkLength(len(ids)); err != nil {
			return nil, nil, err
		}
	}
	x, err := a.model.EmbedTokens(ids)
	if err != nil {
		return nil, nil, err
	}
	return a.forwardEmbeds(x, len(ids), keepLayers)
}

// ForwardEmbeds runs the encoder from T prepared input rows (see embeddinggemma2.Accelerator).
func (a *eg2Accel) ForwardEmbeds(x0 []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.forwardEmbeds(x0, T, keepLayers)
}

func (a *eg2Accel) forwardEmbeds(x0 []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	if err := a.open(); err != nil {
		return nil, nil, err
	}
	c := a.cfg
	H, E := c.Hidden, c.EmbeddingDim
	if T <= 0 || H <= 0 || len(x0)/H != T || len(x0)%H != 0 {
		return nil, nil, fmt.Errorf("metal: %d input values for %d rows of %d", len(x0), T, H)
	}
	if a.profile == eg2MatrixProfile {
		if err := a.checkLength(T); err != nil {
			return nil, nil, err
		}
	}
	slots, err := eg2ArenaSlotsFor(c, a.kvHeads, T, keepLayers)
	if err != nil {
		return nil, nil, err
	}
	if err := a.ensureArena(slots); err != nil {
		return nil, nil, err
	}
	if err := a.grow(T); err != nil {
		return nil, nil, err
	}
	s := &a.scr
	x := s.x.Floats()[:T*H]
	copy(x, x0)
	layers, err := a.encodeTextRows(T, keepLayers, nil, nil)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), s.out.Floats()[:T*E]...), layers, nil
}

// encodeTextRows records the common row-local graph. A packed plan changes
// only the RoPE and attention spans and the per-document gate/up routing.
func (a *eg2Accel) encodeTextRows(T int, keepLayers bool, batch *eg2BatchPlan, tables *[eg2RopeCacheLimit][2]Buffer) ([][]float32, error) {
	c := a.cfg
	H, I, P, L, E := c.Hidden, c.Intermediate, c.PLEDim, c.Layers, c.EmbeddingDim
	s := &a.scr
	var layers [][]float32
	if keepLayers {
		layers = append(layers, append([]float32(nil), s.x.Floats()[:T*H]...))
	}
	gemmFFN := a.gemm
	if a.stagedFFN != (Pipeline{}) && eg2StagedFFNRows(T) && !a.vitGemmForTest {
		gemmFFN = a.gemmStagedFFN
	}
	a.slot = 0
	e := a.q.Begin()
	a.gemm(e, s.x, a.pleProj, s.ple, T, L*P, H)
	a.rms(e, s.ple, a.pleNorm, s.ple, T*L, P)
	for li := range a.layers {
		ly := &a.layers[li]
		hd, nKV := ly.w.HeadDim, ly.w.KVHeads
		qd, kvd := c.Heads*hd, nKV*hd
		// attention
		a.rms(e, s.x, ly.inNorm, s.xn, T, H)
		a.gemm(e, s.xn, ly.q, s.q, T, qd, H)
		a.gemm(e, s.xn, ly.k, s.k, T, kvd, H)
		a.gemm(e, s.xn, ly.v, s.v, T, kvd, H)
		a.rms(e, s.q, ly.qNorm, s.q, T*c.Heads, hd)
		a.rms(e, s.k, ly.kNorm, s.k, T*nKV, hd)
		a.rms(e, s.v, a.ones, s.v, T*nKV, hd)
		if batch != nil {
			pair := tables[batch.layerRope[li]]
			for _, doc := range batch.docs[:batch.docCount] {
				q, ctx := s.q.At(4*doc.start*qd), s.ctx.At(4*doc.start*qd)
				k, v := s.k.At(4*doc.start*kvd), s.v.At(4*doc.start*kvd)
				a.encodeRoPE(e, q, pair, doc.rows, c.Heads, hd)
				a.encodeRoPE(e, k, pair, doc.rows, nKV, hd)
				a.dispatchEG2Matrix(e, a.matrix, q, k, v, ctx, doc.rows, ly.w.Full)
			}
		} else {
			cb, sb := a.ropeTables(T, hd, ly.w.RopeTheta)
			pair := [2]Buffer{cb, sb}
			a.encodeRoPE(e, s.q, pair, T, c.Heads, hd)
			a.encodeRoPE(e, s.k, pair, T, nKV, hd)
			if a.profile == eg2MatrixProfile {
				a.dispatchEG2Matrix(e, a.matrix, s.q, s.k, s.v, s.ctx, T, ly.w.Full)
			} else {
				win := uint32(c.SlidingWindow)
				if ly.w.Full {
					win = 0
				}
				a.attention(e, eg2AttnBufs{q: s.q, k: s.k, v: s.v, qh: s.qh, kh: s.kh, vh: s.vh, sc: s.sc, vt: s.vt, ob: s.ob, ctx: s.ctx},
					T, c.Heads, hd, nKV, win)
			}
		}
		a.gemm(e, s.ctx, ly.o, s.o, T, H, qd)
		a.rms(e, s.o, ly.postAttnNorm, s.o, T, H)
		e.Dispatch(a.p.add, T*H, 256, s.x, s.o, a.u32(uint32(T*H)))
		// MLP
		a.rms(e, s.x, ly.preFFNorm, s.xn, T, H)
		if batch == nil {
			gemmFFN(e, s.xn, ly.gate, s.g, T, I, H)
			gemmFFN(e, s.xn, ly.up, s.u, T, I, H)
		} else {
			for _, run := range batch.runs[:batch.runCount] {
				gemm := a.gemm
				if run.staged {
					gemm = a.gemmStagedFFN
				}
				xn := s.xn.At(4 * run.start * H)
				gemm(e, xn, ly.gate, s.g.At(4*run.start*I), run.rows, I, H)
				gemm(e, xn, ly.up, s.u.At(4*run.start*I), run.rows, I, H)
			}
		}
		e.Dispatch(a.p.geluMul, T*I, 256, s.g, s.u, a.u32(uint32(T*I)))
		a.gemm(e, s.g, ly.down, s.f, T, H, I)
		a.rms(e, s.f, ly.postFFNorm, s.f, T, H)
		e.Dispatch(a.p.add, T*H, 256, s.x, s.f, a.u32(uint32(T*H)))
		// PLE block, then the layer scalar
		a.gemm(e, s.x, ly.pleGate, s.gg, T, P, H)
		e.Dispatch(a.p.geluMulPLE, T*P, 256, s.gg, s.ple, a.u32(uint32(T*P)), a.u32(uint32(P)), a.u32(uint32(L)), a.u32(uint32(li)))
		a.gemm(e, s.gg, ly.pleProj, s.pr, T, H, P)
		a.rms(e, s.pr, ly.plePostNorm, s.pr, T, H)
		e.Dispatch(a.p.addScale, T*H, 256, s.x, s.pr, a.u32(uint32(T*H)), a.f32(ly.w.LayerScalar))
		if keepLayers { // a command buffer per layer, so the host can read the layer's output
			e.End()
			if err := e.Err(); err != nil {
				return nil, err
			}
			layers = append(layers, append([]float32(nil), s.x.Floats()[:T*H]...))
			a.slot = 0
			e = a.q.Begin()
		}
	}
	a.rms(e, s.x, a.finalNorm, s.xn, T, H)
	a.gemm(e, s.xn, a.proj, s.out, T, E, H)
	e.End()
	if err := e.Err(); err != nil {
		return nil, err
	}
	return layers, nil
}

func (a *eg2Accel) encodeRoPE(e *Encoder, x Buffer, pair [2]Buffer, rows, heads, hd int) {
	pairs := rows * heads * hd / 2
	e.Dispatch(a.p.rope, pairs, 64, x, pair[0], pair[1], a.u32(uint32(heads)), a.u32(uint32(hd)), a.u32(uint32(pairs)))
}

// attention is one layer's bidirectional grouped-query attention over b.q/b.k/b.v into b.ctx, as matmuls: q, k and v
// laid out head-major, then for each block of eg2Block query rows the scores Q·Kᵀ over only the keys the block can
// reach (the window plus the block on a sliding layer), a windowed row softmax, and the output scores × V against the
// block's values transposed (the GEMM is A·Bᵀ, so V must be [hd, keys]). Memory is a block's scores, never T x T.
func (a *eg2Ops) attention(e *Encoder, s eg2AttnBufs, T, nH, hd, nKV int, win uint32) {
	e.Dispatch(a.p.headMajor, T*nH*hd, 256, s.q, s.qh, a.u32(uint32(T)), a.u32(uint32(nH)), a.u32(uint32(hd)))
	e.Dispatch(a.p.headMajor, T*nKV*hd, 256, s.k, s.kh, a.u32(uint32(T)), a.u32(uint32(nKV)), a.u32(uint32(hd)))
	e.Dispatch(a.p.headMajor, T*nKV*hd, 256, s.v, s.vh, a.u32(uint32(T)), a.u32(uint32(nKV)), a.u32(uint32(hd)))
	B, W := min(eg2Block, T), int(win)
	group := nH / nKV
	for g := range nKV {
		for i0 := 0; i0 < T; i0 += B {
			i1 := min(T, i0+B)
			rows := i1 - i0
			klo, khi := 0, T
			if W > 0 {
				klo, khi = max(0, i0-W), min(T, i1+W)
			}
			kc := khi - klo
			e.Dispatch(a.p.vtBlock, hd*kc, 256, s.vh, s.vt, a.u32(uint32(T)), a.u32(uint32(hd)), a.u32(uint32(g)),
				a.u32(uint32(klo)), a.u32(uint32(kc)))
			for h := g * group; h < (g+1)*group; h++ {
				a.gemm(e, s.qh.At(((h*T)+i0)*hd*4), s.kh.At(((g*T)+klo)*hd*4), s.sc, rows, kc, hd)
				e.Dispatch(a.p.softmax, rows*eg2TG, eg2TG, s.sc, a.u32(uint32(kc)), a.u32(uint32(i0)), a.u32(uint32(klo)), a.u32(win))
				a.gemm(e, s.sc, s.vt, s.ob, rows, hd, kc)
				e.Dispatch(a.p.scatter, rows*hd, 256, s.ob, s.ctx, a.u32(uint32(rows)), a.u32(uint32(hd)), a.u32(uint32(nH)),
					a.u32(uint32(h)), a.u32(uint32(i0)))
			}
		}
	}
}

// ropeTables returns the CPU's own cos/sin tables for T positions at this head dim and theta, uploaded once and never
// written again, so every layer and every in-flight command buffer can read the same buffers. At most 16 lengths are
// kept; the 17th clears the cache and releases its tables (a forward holds the lock, so no command buffer is using a
// dropped table).
func (a *eg2Accel) ropeTables(T, hd int, theta float64) (Buffer, Buffer) {
	k := eg2RopeKey{T, hd, theta}
	if b, ok := a.ropes[k]; ok {
		return b[0], b[1]
	}
	if len(a.ropes) >= eg2RopeCacheLimit {
		for k, b := range a.ropes {
			a.release(&b[0], &b[1])
			delete(a.ropes, k)
		}
	}
	cs, sn := embeddinggemma2.RopeTables(T, hd, theta)
	b := [2]Buffer{NewBufferFloatsOn(mem.LineCompanion, a.d, cs), NewBufferFloatsOn(mem.LineCompanion, a.d, sn)}
	a.ropes[k] = b
	return b[0], b[1]
}
