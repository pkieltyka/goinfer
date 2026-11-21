//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"math"
	"sync"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

// EmbeddingGemma 2's text encoder on CUDA: the CPU forward of embeddinggemma2/model.go in float32 with every weight
// resident on the device, on the tower base (tower_base.go). The token-row gather is the host's, as Metal's is; every
// layer runs here. Attention is aikit's attention_seg, which takes each query row's own key range: [i-W, i+W] clipped
// to the input on a sliding layer, the whole input on a full one. It reads one fused [T, 3, nH, hd] buffer, so q, k and
// v are normed and rotated in their own buffers and gathered into it by strided device copies (Queue.Copy2DAsync), each
// query head taking its KV head's columns. Registered as the "cuda" accelerator; nothing selects it by default.

func init() {
	embeddinggemma2.RegisterAccelerator("cuda", func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return newEG2Accel(m)
	})
}

// eg2MaxTokens is the longest input the accelerator takes: a full layer's score row is T float32 in attention_seg's
// dynamic shared memory, inside the 48 KB a block gets without an opt-in. It is the encoder's own MaxTokens.
const eg2MaxTokens = embeddinggemma2.MaxTokens

type eg2Layer struct {
	w                                 embeddinggemma2.LayerWeights
	inNorm, q, k, v, o, qNorm, kNorm  Buffer
	postAttnNorm, preFFNorm, gate, up Buffer
	down, postFFNorm, pleGate         Buffer
	pleProj, plePostNorm              Buffer
}

// eg2Rope is one (head dim, theta) pair's cos and sin tables, [rows, hd] each with both halves of a row holding the
// CPU's [hd/2] values (tower_rope_half's layout). A row depends only on its position, so the first T rows serve any
// input of T tokens.
type eg2Rope struct {
	hd       int
	theta    float64
	rows     int
	cos, sin Buffer
}

// eg2Defect plants one defect for a test to show it is noticed; zero in production.
type eg2Defect struct {
	windowWide   bool // a sliding row reaches distance W+1
	windowNarrow bool // a sliding row stops at distance W-1
	vNormWeight  bool // v's norm takes the k weight instead of none
}

// eg2Accel is the CUDA accelerator. One forward at a time (mu); towerOps runs every device call on its executor.
type eg2Accel struct {
	mu     sync.Mutex
	ops    *towerOps
	closed bool
	cfg    embeddinggemma2.Config
	embed  []float32 // the host token table, aliasing the model's

	pleProj, pleNorm, finalNorm, proj, ones Buffer
	layers                                  []eg2Layer
	ropes                                   []eg2Rope
	defect                                  eg2Defect
}

func newEG2Accel(m *embeddinggemma2.Model) (*eg2Accel, error) {
	w := m.Weights()
	c := w.Config
	maxN := max(c.Layers*c.PLEDim, c.Intermediate, c.Hidden, c.EmbeddingDim)
	maxHD := 0
	for i, ly := range w.Layers {
		if ly.HeadDim%2 != 0 || ly.KVHeads <= 0 || c.Heads%ly.KVHeads != 0 {
			return nil, fmt.Errorf("cuda: EmbeddingGemma 2 layer %d: %d heads over %d KV heads, head dim %d", i, c.Heads, ly.KVHeads, ly.HeadDim)
		}
		maxN = max(maxN, c.Heads*ly.HeadDim)
		maxHD = max(maxHD, ly.HeadDim)
	}
	ops, err := newTowerOps(maxN)
	if err != nil {
		return nil, err
	}
	a := &eg2Accel{ops: ops, cfg: c, embed: w.Embed}
	if err := ops.do(func() error {
		up := ops.up
		a.pleProj, a.pleNorm, a.finalNorm, a.proj = up(w.PLEProj), up(w.PLENorm), up(w.FinalNorm), up(w.Projection)
		ones := make([]float32, maxHD)
		for i := range ones {
			ones[i] = 1
		}
		a.ones = up(ones)
		for _, ly := range w.Layers {
			a.layers = append(a.layers, eg2Layer{w: ly, inNorm: up(ly.InNorm), q: up(ly.Q), k: up(ly.K), v: up(ly.V), o: up(ly.O),
				qNorm: up(ly.QNorm), kNorm: up(ly.KNorm), postAttnNorm: up(ly.PostAttnNorm), preFFNorm: up(ly.PreFFNorm),
				gate: up(ly.Gate), up: up(ly.Up), down: up(ly.Down), postFFNorm: up(ly.PostFFNorm), pleGate: up(ly.PLEGate),
				pleProj: up(ly.PLEProj), plePostNorm: up(ly.PLEPostNorm)})
		}
		return ops.finish()
	}); err != nil {
		ops.close()
		return nil, fmt.Errorf("cuda: EmbeddingGemma 2 upload: %w", err)
	}
	return a, nil
}

func (a *eg2Accel) Name() string { return "cuda" }

// Close frees the encoder's device memory and stops its executor; a forward after it is an error. Idempotent.
func (a *eg2Accel) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.closed {
		a.ops.close()
		a.closed, a.layers, a.ropes = true, nil, nil
	}
	return nil
}

var errEG2Closed = errors.New("cuda: the EmbeddingGemma 2 accelerator is closed")

// Forward gathers the scaled token rows on the host, then runs ForwardEmbeds (embeddinggemma2.Accelerator).
func (a *eg2Accel) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	c := a.cfg
	H := c.Hidden
	if len(ids) == 0 {
		return nil, nil, fmt.Errorf("cuda: no input ids")
	}
	x := make([]float32, len(ids)*H)
	scale := float32(math.Sqrt(float64(H)))
	for t, id := range ids {
		if id < 0 || id >= c.VocabSize {
			return nil, nil, fmt.Errorf("cuda: token id %d out of range [0, %d)", id, c.VocabSize)
		}
		for j, v := range a.embed[id*H : (id+1)*H] {
			x[t*H+j] = v * scale
		}
	}
	return a.ForwardEmbeds(x, len(ids), keepLayers)
}

// rope returns the tables for at least T rows at this head dim and theta, rebuilding them from the CPU's own values
// when T is past what is held. The caller is on the executor.
func (a *eg2Accel) rope(T, hd int, theta float64) (Buffer, Buffer) {
	for i := range a.ropes {
		r := &a.ropes[i]
		if r.hd != hd || r.theta != theta {
			continue
		}
		if r.rows >= T {
			return r.cos, r.sin
		}
		a.ops.dev.ReleaseBuf(r.cos)
		a.ops.dev.ReleaseBuf(r.sin)
		a.ropes = append(a.ropes[:i], a.ropes[i+1:]...)
		break
	}
	cs, sn := embeddinggemma2.RopeTables(T, hd, theta)
	half := hd / 2
	cf, sf := make([]float32, T*hd), make([]float32, T*hd)
	for t := range T {
		for i := range half {
			cf[t*hd+i], cf[t*hd+half+i] = cs[t*half+i], cs[t*half+i]
			sf[t*hd+i], sf[t*hd+half+i] = sn[t*half+i], sn[t*half+i]
		}
	}
	r := eg2Rope{hd: hd, theta: theta, rows: T, cos: a.ops.up(cf), sin: a.ops.up(sf)}
	a.ropes = append(a.ropes, r)
	return r.cos, r.sin
}

// bounds is each query row's key range [lo, hi) on a sliding layer of radius W (|i - j| <= W) or, with W < 0, a full one.
func eg2Bounds(T, W int) (lo, hi []int32) {
	lo, hi = make([]int32, T), make([]int32, T)
	for i := range T {
		if W < 0 {
			lo[i], hi[i] = 0, int32(T)
			continue
		}
		lo[i], hi[i] = int32(max(0, i-W)), int32(min(T, i+W+1))
	}
	return lo, hi
}

// ForwardEmbeds runs the encoder from T prepared input rows (embeddinggemma2.Accelerator); x0 is not modified.
func (a *eg2Accel) ForwardEmbeds(x0 []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, nil, errEG2Closed
	}
	c := a.cfg
	H, I, P, L, E, nH := c.Hidden, c.Intermediate, c.PLEDim, c.Layers, c.EmbeddingDim, c.Heads
	if T == 0 || len(x0) != T*H {
		return nil, nil, fmt.Errorf("cuda: %d input values for %d rows of %d", len(x0), T, H)
	}
	if T > eg2MaxTokens {
		return nil, nil, fmt.Errorf("cuda: %d positions, over the accelerator's %d", T, eg2MaxTokens)
	}
	W := c.SlidingWindow
	if a.defect.windowWide {
		W++
	}
	if a.defect.windowNarrow {
		W--
	}
	slLo, slHi := eg2Bounds(T, W)
	fuLo, fuHi := eg2Bounds(T, -1)
	maxQD, maxKVD := 0, 0
	for _, ly := range a.layers {
		maxQD = max(maxQD, nH*ly.w.HeadDim)
		maxKVD = max(maxKVD, ly.w.KVHeads*ly.w.HeadDim)
	}
	var out []float32
	var layers [][]float32
	if keepLayers {
		layers = append(layers, append([]float32(nil), x0...))
	}
	err := a.ops.do(func() error {
		o := a.ops
		defer o.releaseScratch()
		af := o.af
		x, xn := af(T*H), af(T*H)
		q, k, v := af(T*maxQD), af(T*maxKVD), af(T*maxKVD)
		qkv, ctx := af(T*3*maxQD), af(T*maxQD)
		ob, g, u, f := af(T*H), af(T*I), af(T*I), af(T*H)
		ple, pleL, gg, pr := af(T*L*P), af(T*P), af(T*P), af(T*H)
		res := af(T * E)
		slLoB, slHiB, fuLoB, fuHiB := o.ai(T), o.ai(T), o.ai(T), o.ai(T)
		if e := gpu.Upload(x, x0); e != nil {
			return e
		}
		for _, b := range []struct {
			dst Buffer
			v   []int32
		}{{slLoB, slLo}, {slHiB, slHi}, {fuLoB, fuLo}, {fuHiB, fuHi}} {
			if e := gpu.Upload(b.dst, b.v); e != nil {
				return e
			}
		}
		eps := float32(c.RMSNormEps)
		// Per-layer inputs, as the CPU forms them: RMSNorm(W · x0 · hidden^-0.5) over [T, L, P].
		o.gemm(x, a.pleProj, ple, T, L*P, H)
		o.scaleVec(ple, T*L*P, float32(1/math.Sqrt(float64(H))))
		o.rms(ple, a.pleNorm, ple, T*L, P, eps)
		for li := range a.layers {
			ly := &a.layers[li]
			hd, nKV := ly.w.HeadDim, ly.w.KVHeads
			qd, kvd, group := nH*hd, nKV*hd, nH/nKV
			// attention
			o.rms(x, ly.inNorm, xn, T, H, eps)
			o.gemm(xn, ly.q, q, T, qd, H)
			o.gemm(xn, ly.k, k, T, kvd, H)
			o.gemm(xn, ly.v, v, T, kvd, H)
			o.rms(q, ly.qNorm, q, T*nH, hd, eps)
			o.rms(k, ly.kNorm, k, T*nKV, hd, eps)
			vw := a.ones
			if a.defect.vNormWeight {
				vw = ly.kNorm
			}
			o.rms(v, vw, v, T*nKV, hd, eps)
			cs, sn := a.rope(T, hd, ly.w.RopeTheta)
			o.ropeHalfTo(q, cs, sn, T, nH, hd)
			o.ropeHalfTo(k, cs, sn, T, nKV, hd)
			if e := a.pack(qkv, q, k, v, T, nH, hd, group); e != nil {
				return e
			}
			lo, hi, maxSeg := slLoB, slHiB, min(T, 2*W+1)
			if ly.w.Full {
				lo, hi, maxSeg = fuLoB, fuHiB, T
			}
			o.cls(clsAttn)
			o.launch(o.vit.AttentionSeg, gpu.SegAttentionGrid(T, nH, maxSeg), Arg(qkv), Arg(ctx), Arg(lo), Arg(hi),
				i32(int32(T)), i32(int32(nH)), i32(int32(hd)), f32v(1))
			o.gemm(ctx, ly.o, ob, T, H, qd)
			o.rms(ob, ly.postAttnNorm, ob, T, H, eps)
			o.addVec(x, ob, T*H)
			// MLP
			o.rms(x, ly.preFFNorm, xn, T, H, eps)
			o.gemm(xn, ly.gate, g, T, I, H)
			o.gemm(xn, ly.up, u, T, I, H)
			o.geluTanh(g, T*I)
			o.mulVec(g, u, T*I)
			o.gemm(g, ly.down, f, T, H, I)
			o.rms(f, ly.postFFNorm, f, T, H, eps)
			o.addVec(x, f, T*H)
			// PLE block, then the layer scalar over the whole residual sum
			o.gemm(x, ly.pleGate, gg, T, P, H)
			o.geluTanh(gg, T*P)
			if o.err == nil {
				o.err = o.q.Copy2DAsync(pleL, P*4, ple.At(li*P*4), L*P*4, P*4, T)
			}
			o.mulVec(gg, pleL, T*P)
			o.gemm(gg, ly.pleProj, pr, T, H, P)
			o.rms(pr, ly.plePostNorm, pr, T, H, eps)
			o.addVec(x, pr, T*H)
			o.scaleVec(x, T*H, ly.w.LayerScalar)
			if keepLayers {
				if e := o.finish(); e != nil {
					return fmt.Errorf("layer %d: %w", li, e)
				}
				h := make([]float32, T*H)
				if e := gpu.Download(x, h); e != nil {
					return e
				}
				layers = append(layers, h)
			}
		}
		o.rms(x, a.finalNorm, xn, T, H, eps)
		o.gemm(xn, a.proj, res, T, E, H)
		if e := o.finish(); e != nil {
			return e
		}
		out = make([]float32, T*E)
		return gpu.Download(res, out)
	})
	if err != nil {
		return nil, nil, fmt.Errorf("cuda: EmbeddingGemma 2 forward: %w", err)
	}
	return out, layers, nil
}

// pack gathers q [T, nH*hd], k and v [T, nKV*hd] into qkv [T, 3, nH, hd], attention_seg's layout, by strided device
// copies: q whole, then each query head h's k and v from KV head h/group. It latches the first error as a launch does.
func (a *eg2Accel) pack(qkv, q, k, v Buffer, T, nH, hd, group int) error {
	o := a.ops
	if o.err != nil {
		return nil
	}
	qd, kvd := nH*hd, nH/group*hd
	row := 3 * qd * 4
	if e := o.q.Copy2DAsync(qkv, row, q, qd*4, qd*4, T); e != nil {
		return e
	}
	for h := range nH {
		g := h / group
		if e := o.q.Copy2DAsync(qkv.At((qd+h*hd)*4), row, k.At(g*hd*4), kvd*4, hd*4, T); e != nil {
			return e
		}
		if e := o.q.Copy2DAsync(qkv.At((2*qd+h*hd)*4), row, v.At(g*hd*4), kvd*4, hd*4, T); e != nil {
			return e
		}
	}
	return nil
}
