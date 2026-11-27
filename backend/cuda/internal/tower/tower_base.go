//go:build cuda

package tower

import (
	_ "embed"
	"errors"
	"fmt"
	"runtime"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

// BasePTX holds tower_pos_add / tower_clamp / tower_clamp_copy / tower_rope_axial /
// tower_rope_half / tower_mul / tower_scale / tower_attn_hd{64,72,80}, the CUDA
// tower base's own kernels, used alongside gpudev's ViT kernels.
// Built at NVRTC 12.9.86.
//
//go:embed testdata/tower_base.ptx
var BasePTX []byte

// The CUDA vision-tower base (docs/measurements/multimodal-support-2026-10/s4-gate0-cuda-towers.md) uses gpudev's ViT kernels (f32 GEMMs with bias and
// bias+residual epilogues, LayerNorm, RMSNorm, GELU, scaled attention, a NeoX RoPE) plus the kernels of
// tower_base.cu, behind one executor goroutine that owns the device (CUDA contexts are thread-affine, as in
// vision_encoder.go) and a launch helper that latches the first error. It is the counterpart of Metal's eg2Ops, not a
// port: CUDA kernels take their scalars as launch arguments, and attention uses the CUDA device kernel (np <= 12288 by its
// shared-memory row).
//
// Every method except New/Close/Do must run inside Do(), which turns a failed allocation (a panic inside the executor,
// gpu.NewBufferLenOf's contract) into an error, so a tower's factory and Hidden return errors and serve can name the CPU
// fallback.

type Ops struct {
	Dev *gpu.Device
	Q   gpu.Queue
	ViT gpu.ViT

	LoadLibrary ModuleLoader

	posAdd, clamp, clampCopy, ropeAxial, ropeHalf, mul, scale gpu.Pipeline

	zeroBias gpu.Buffer // zeros, as long as the widest bias-free projection: lets GEMM run the register-blocked bias kernel at any M

	Scratch    []gpu.Buffer         // per-call buffers, released by ReleaseScratch
	Err        error                // the first launch error of the current call
	ShortTiled bool                 // opt-in EmbeddingGemma 2 query experiment
	prof       *Prof                // the per-class profile; nil in production (tower_prof.go)
	ownAttn    map[int]gpu.Pipeline // the tower's own fused attention by head dim (64, 72, 80); a head dim not in it takes the reference kernel

	ReqCh chan func() error
	AckCh chan error
}

// New creates the device and loads the kernels on a pinned executor goroutine. maxN is the widest output of any projection run without a bias.
func New(maxN int) (*Ops, error) {
	return NewOn(gpu.CreateSystemDefaultDevice, maxN)
}

// NewOn creates Ops using an explicit non-nil device opener on the executor goroutine.
func NewOn(open func() (*gpu.Device, error), maxN int) (t *Ops, err error) {
	t, err = NewOnUnclosed(open, maxN, nil)
	if err != nil {
		if t != nil {
			t.Close()
		}
		return nil, err
	}
	return t, nil
}

// NewOnUnclosed builds Ops without running unchecked t.Close() on
// failure, returning the partial *Ops alongside the error so EG2's
// serialized attach path owns checked rollback.
func NewOnUnclosed(open func() (*gpu.Device, error), maxN int, moduleLoader func(*gpu.Device) ModuleLoader) (t *Ops, err error) {
	if open == nil {
		return nil, errors.New("cuda tower: nil device opener")
	}
	t = &Ops{ReqCh: make(chan func() error), AckCh: make(chan error)}
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for j := range t.ReqCh {
			t.AckCh <- RunJob(j)
		}
	}()
	err = t.Do(func() error {
		var e error
		if t.Dev, e = open(); e != nil {
			return e
		}
		t.Q = t.Dev.NewCommandQueue()
		if moduleLoader == nil {
			t.LoadLibrary = PTXLoader(t.Dev.CompileLibrary)
		} else {
			t.LoadLibrary = moduleLoader(t.Dev)
		}
		if t.ViT, e = t.Dev.NewViTWithLoader(func(ptx []byte) (gpu.Library, error) {
			return t.LoadLibrary(ModuleViT, ptx)
		}); e != nil {
			return fmt.Errorf("cuda tower: aikit's ViT kernels: %w", e)
		}
		lib, e := t.LoadLibrary(ModuleBase, BasePTX)
		if e != nil {
			return fmt.Errorf("cuda tower: compile tower_base: %w", e)
		}
		for _, b := range []struct {
			name string
			dst  *gpu.Pipeline
		}{{"tower_pos_add", &t.posAdd}, {"tower_clamp", &t.clamp}, {"tower_clamp_copy", &t.clampCopy},
			{"tower_rope_axial", &t.ropeAxial}, {"tower_rope_half", &t.ropeHalf}, {"tower_mul", &t.mul}, {"tower_scale", &t.scale}} {
			if *b.dst, e = t.Dev.NewComputePipeline(lib, b.name); e != nil {
				return fmt.Errorf("cuda tower: pipeline %s: %w", b.name, e)
			}
		}
		t.ownAttn = map[int]gpu.Pipeline{}
		for _, hd := range []int{64, 72, 80} {
			p, e := t.Dev.NewComputePipeline(lib, fmt.Sprintf("tower_attn_hd%d", hd))
			if e != nil {
				return fmt.Errorf("cuda tower: pipeline tower_attn_hd%d: %w", hd, e)
			}
			t.ownAttn[hd] = p
		}
		t.zeroBias = gpu.NewBufferOfOn(t.Dev, mem.LineCompanion, make([]float32, max(maxN, 1)))
		return nil
	})
	if err != nil {
		return t, err
	}
	return t, nil
}

// Do runs j on the executor goroutine; a panic inside it (an allocation failure) comes back as an error.
func (t *Ops) Do(j func() error) error {
	t.ReqCh <- j
	return AckJob(<-t.AckCh)
}

// Close releases everything the device holds and stops the executor. Safe to call once.
func (t *Ops) Close() {
	if t.ReqCh == nil {
		return
	}
	_ = t.Do(func() error {
		if t.Dev != nil {
			t.ReleaseScratch()
			t.Dev.ReleaseObjects()
		}
		return nil
	})
	close(t.ReqCh)
	t.ReqCh = nil
}

// CloseChecked retires every tracked buffer, object, and the primary context
// via ReleaseObjectsChecked before stopping the executor. If any native close
// fails, the executor and context stay alive with the failed handles still
// owned in the ledger so a later Close or cleanup drain can retry without
// losing leases or handles.
func (t *Ops) CloseChecked() error {
	if t.ReqCh == nil {
		return nil
	}
	if err := t.Do(func() error {
		if t.Dev == nil {
			return nil
		}
		if err := t.Dev.ReleaseObjectsChecked(); err != nil {
			return err
		}
		t.Scratch = nil
		return nil
	}); err != nil {
		return err
	}
	close(t.ReqCh)
	t.ReqCh = nil
	return nil
}

// Up uploads a float32 slice as a persistent buffer (weights, tables).
func (t *Ops) Up(x []float32) gpu.Buffer {
	if len(x) == 0 {
		return gpu.Buffer{}
	}
	return gpu.NewBufferOfOn(t.Dev, mem.LineCompanion, x)
}

// Af allocates a float32 scratch buffer of n elements, released by ReleaseScratch.
func (t *Ops) Af(n int) gpu.Buffer {
	b := gpu.NewBufferLenOfOn[float32](t.Dev, mem.LineCompanion, max(n, 1))
	t.Scratch = append(t.Scratch, b)
	return b
}

// Ai allocates an int32 scratch buffer.
func (t *Ops) Ai(n int) gpu.Buffer {
	b := gpu.NewBufferLenOfOn[int32](t.Dev, mem.LineCompanion, max(n, 1))
	t.Scratch = append(t.Scratch, b)
	return b
}

func (t *Ops) ReleaseScratch() {
	for _, b := range t.Scratch {
		t.Dev.ReleaseBuf(b)
	}
	t.Scratch = t.Scratch[:0]
}

// ReleaseScratchChecked frees each scratch buffer via ReleaseBufChecked,
// retaining any buffer whose native close fails so ownership and ledger
// charges are never discarded.
func (t *Ops) ReleaseScratchChecked() error {
	var err error
	var kept []gpu.Buffer
	for _, b := range t.Scratch {
		if e := t.Dev.ReleaseBufChecked(b); e != nil {
			err = errors.Join(err, e)
			kept = append(kept, b)
		}
	}
	t.Scratch = kept
	return err
}

// Launch enqueues one kernel and latches the first error.
func (t *Ops) Launch(p gpu.Pipeline, cfg gpu.LaunchConfig, args ...gpu.KernelArg) {
	if t.Err != nil {
		return
	}
	t.Err = t.Q.Launch(p, cfg, args...)
}

// Finish waits for the queue and returns the call's first error, clearing it.
func (t *Ops) Finish() error {
	t.profFlush()
	if t.Err == nil {
		t.Err = t.Q.Sync()
	}
	err := t.Err
	t.Err = nil
	return err
}

var i32 = gpu.ArgValue[int32]
var f32v = gpu.ArgValue[float32]
var i64 = gpu.ArgValue[int64]

// GEMM is out[M,N] = a[M,K] · w[N,K]ᵀ. With K a multiple of 16 it runs the bias kernel with a zero bias, which takes any M and N at the register-blocked
// speed; otherwise the device layer's plan (the register kernel when M and N are multiples of 64, else the tiled one).
func (t *Ops) GEMM(a, w, out gpu.Buffer, M, N, K int) {
	t.Cls(ClsGEMM)
	t.ProfGEMM(M, N, K)
	// Exclude the large initial PLE projection and preserve the reference bias epilogue.
	if t.ShortTiled && M <= 36 && N <= 2048 && K%16 == 0 && N <= t.zeroBias.Len() {
		t.Launch(t.ViT.GEMMF32Tiled, gpu.TileGrid(M, N), gpu.Arg(a), gpu.Arg(w), gpu.Arg(out),
			i32(int32(M)), i32(int32(N)), i32(int32(K)))
		t.Launch(t.ViT.AddBias, gpu.Grid1D(M*N, 256), gpu.Arg(out), gpu.Arg(t.zeroBias),
			i32(int32(M)), i32(int32(N)))
		return
	}
	if K%16 == 0 && N <= t.zeroBias.Len() {
		p, cfg := t.ViT.GEMMF32BiasPlan(M, N, K)
		t.Launch(p, cfg, gpu.Arg(a), gpu.Arg(w), gpu.Arg(t.zeroBias), gpu.Arg(out), i32(int32(M)), i32(int32(N)), i32(int32(K)))
		return
	}
	p, cfg := t.ViT.GEMMF32Plan(M, N, K)
	t.Launch(p, cfg, gpu.Arg(a), gpu.Arg(w), gpu.Arg(out), i32(int32(M)), i32(int32(N)), i32(int32(K)))
}

// GEMMBias is out[M,N] = a[M,K] · w[N,K]ᵀ + bias[N]. A K that is not a multiple of 16 (real GLM-OCR's patch embed, K = 1176) takes the tiled GEMM plus add_bias.
func (t *Ops) GEMMBias(a, w, bias, out gpu.Buffer, M, N, K int) {
	t.Cls(ClsGEMM)
	t.ProfGEMM(M, N, K)
	if K%16 == 0 {
		p, cfg := t.ViT.GEMMF32BiasPlan(M, N, K)
		t.Launch(p, cfg, gpu.Arg(a), gpu.Arg(w), gpu.Arg(bias), gpu.Arg(out), i32(int32(M)), i32(int32(N)), i32(int32(K)))
		return
	}
	p, cfg := t.ViT.GEMMF32Plan(M, N, K)
	t.Launch(p, cfg, gpu.Arg(a), gpu.Arg(w), gpu.Arg(out), i32(int32(M)), i32(int32(N)), i32(int32(K)))
	t.Launch(t.ViT.AddBias, gpu.Grid1D(M*N, 256), gpu.Arg(out), gpu.Arg(bias), i32(int32(M)), i32(int32(N)))
}

// GEMMBiasAdd is resid[M,N] += a[M,K] · w[N,K]ᵀ + bias[N]. A K that is not a multiple of 16 goes through tmp[M*N] and add_vec.
func (t *Ops) GEMMBiasAdd(a, w, bias, resid, tmp gpu.Buffer, M, N, K int) {
	t.Cls(ClsGEMM)
	t.ProfGEMM(M, N, K)
	if K%16 == 0 {
		p, cfg := t.ViT.GEMMF32BiasAddPlan(M, N, K)
		t.Launch(p, cfg, gpu.Arg(a), gpu.Arg(w), gpu.Arg(bias), gpu.Arg(resid), i32(int32(M)), i32(int32(N)), i32(int32(K)))
		return
	}
	t.GEMMBias(a, w, bias, tmp, M, N, K)
	t.AddVec(resid, tmp, M*N)
}

// RMS is out = rmsnorm(x) * w over rows of dim (in place when out is x), eps as given.
func (t *Ops) RMS(x, w, out gpu.Buffer, rows, dim int, eps float32) {
	t.Cls(ClsNorm)
	t.Launch(t.ViT.RMSNorm, gpu.RowGrid(rows), gpu.Arg(x), gpu.Arg(w), gpu.Arg(out), i32(int32(rows)), i32(int32(dim)), f32v(eps))
}

// LayerNorm is out = layernorm(x) * w + b over rows of dim.
func (t *Ops) LayerNorm(x, w, b, out gpu.Buffer, rows, dim int, eps float32) {
	t.Cls(ClsNorm)
	t.Launch(t.ViT.LayerNorm, gpu.RowGrid(rows), gpu.Arg(x), gpu.Arg(w), gpu.Arg(b), gpu.Arg(out), i32(int32(rows)), i32(int32(dim)), f32v(eps))
}

func (t *Ops) AddVec(x, y gpu.Buffer, n int) {
	t.Cls(ClsElem)
	t.Launch(t.ViT.AddVec, gpu.Grid1D(n, 256), gpu.Arg(x), gpu.Arg(y), i32(int32(n)))
}

func (t *Ops) GeluTanh(x gpu.Buffer, n int) {
	t.Cls(ClsElem)
	t.Launch(t.ViT.GELUTanh, gpu.Grid1D(n, 256), gpu.Arg(x), i32(int32(n)))
}

// MulVec is x *= u, elementwise.
func (t *Ops) MulVec(x, u gpu.Buffer, n int) {
	t.Cls(ClsElem)
	t.Launch(t.mul, gpu.Grid1D(n, 256), gpu.Arg(x), gpu.Arg(u), i64(int64(n)))
}

// ScaleVec is x *= s.
func (t *Ops) ScaleVec(x gpu.Buffer, n int, s float32) {
	t.Cls(ClsElem)
	t.Launch(t.scale, gpu.Grid1D(n, 256), gpu.Arg(x), i64(int64(n)), f32v(s))
}

// ClampInPlace clamps x[:n] to [lo, hi]; ClampCopyTo writes the clamped x into out.
func (t *Ops) ClampInPlace(x gpu.Buffer, n int, lo, hi float32) {
	t.Cls(ClsElem)
	t.Launch(t.clamp, gpu.Grid1D(n, 256), gpu.Arg(x), i64(int64(n)), f32v(lo), f32v(hi))
}

func (t *Ops) ClampCopyTo(x, out gpu.Buffer, n int, lo, hi float32) {
	t.Cls(ClsElem)
	t.Launch(t.clampCopy, gpu.Grid1D(n, 256), gpu.Arg(x), gpu.Arg(out), i64(int64(n)), f32v(lo), f32v(hi))
}

// PosAddTo is x[i, d] += X[pos[i].x, d] + Y[pos[i].y, d] over rows of width H (pos is [rows, 2] int32).
func (t *Ops) PosAddTo(x, X, Y, pos gpu.Buffer, rows, H int) {
	t.Cls(ClsElem)
	n := rows * H
	t.Launch(t.posAdd, gpu.Grid1D(n, 256), gpu.Arg(x), gpu.Arg(X), gpu.Arg(Y), gpu.Arg(pos), i32(int32(H)), i64(int64(n)))
}

// RopeAxialTo is Gemma 4's axial 2-D RoPE in place over x [T, heads, hd] from cos/sin [T, hd].
func (t *Ops) RopeAxialTo(x, cs, sn gpu.Buffer, T, heads, hd int) {
	t.Cls(ClsElem)
	n := T * heads * hd / 2
	t.Launch(t.ropeAxial, gpu.Grid1D(n, 256), gpu.Arg(x), gpu.Arg(cs), gpu.Arg(sn), i32(int32(heads)), i32(int32(hd)), i64(int64(n)))
}

// RopeHalfTo is the NeoX rotate-half RoPE in place over x [T, heads, hd] from cos/sin [T, hd] (Qwen3.5 and GLM-OCR's tables).
func (t *Ops) RopeHalfTo(x, cs, sn gpu.Buffer, T, heads, hd int) {
	t.Cls(ClsElem)
	n := T * heads * hd / 2
	t.Launch(t.ropeHalf, gpu.Grid1D(n, 256), gpu.Arg(x), gpu.Arg(cs), gpu.Arg(sn), i32(int32(heads)), i32(int32(hd)), i64(int64(n)))
}

// Attention is the in-tree bidirectional multi-head self-attention over np patches, q/k/v/out all [np, nH*hd] with the heads in place, at the given scale. The
// untiled kernel keeps a score row in shared memory (np*4 bytes, np <= 12288 at 48 KB) and the tiled one (from np 3072, hd <= 128) does not; an image neither can
// take is refused by name, not truncated.
func (t *Ops) Attention(q, k, v, out gpu.Buffer, np, nH, hd int, scale float32) error {
	t.Cls(ClsAttn)
	t.profAttn(np, nH, hd)
	if p, ok := t.ownAttn[hd]; ok && !AttnAikit {
		// The tower's own fused float32 kernel (tower_base.cu). No score row in shared memory, so no np ceiling; 64 queries per block, one block per head.
		cfg := gpu.LaunchConfig{GridX: uint32((np + 63) / 64), GridY: uint32(nH), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		t.Launch(p, cfg, gpu.Arg(q), gpu.Arg(k), gpu.Arg(v), gpu.Arg(out), i32(int32(np)), i32(int32(nH)), f32v(scale*1.4426950408889634), i32(int32(AttnDefect)))
		return nil
	}
	p, cfg := t.ViT.AttentionPlan(np, nH, hd)
	if p != t.ViT.AttentionTiled && np > 12288 {
		// the untiled kernel keeps a score row in shared memory; the query-tiled online-softmax kernel (taken from np 3072 when the head dim allows) does not
		return fmt.Errorf("cuda tower: %d patches exceed the attention kernel's shared-memory row (12288) and the head dim %d has no tiled kernel", np, hd)
	}
	t.Launch(p, cfg, gpu.Arg(q), gpu.Arg(k), gpu.Arg(v), gpu.Arg(out), i32(int32(np)), i32(int32(nH)), i32(int32(hd)), f32v(scale))
	return nil
}

// AttnAikit routes the towers back to the reference attention kernel (the A/B arm of the whole-tower comparison and the
// fallback's own test); AttnDefect plants the fused kernel's defects for its tests (1: no rescale when the running
// max moves; 2: the key mask one past the segment). Both are zero in production.
var (
	AttnAikit  bool
	AttnDefect int
)
