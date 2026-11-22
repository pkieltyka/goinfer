//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

// EmbeddingGemma 2's text encoder on CUDA: the CPU forward of embeddinggemma2/model.go in float32 with every weight
// resident on the device, on the tower base (tower_base.go). The token-row gather is the host's, as Metal's is; every
// layer runs here. Attention is aikit's attention_seg, which takes each query row's own key range: [i-W, i+W] clipped
// to the input on a sliding layer, the whole input on a full one. It reads one fused [T, 3, nH, hd] buffer, so q, k and
// v are normed and rotated in their own buffers and gathered into it by strided device copies (Queue.Copy2DAsync), each
// query head taking its KV head's columns. Registered as the "cuda" accelerator; nothing selects it by default.

func init() {
	embeddinggemma2.RegisterAccelerator("cuda", func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return newEG2Accel(m, false)
	})
	embeddinggemma2.RegisterSelectedAccelerator("cuda", selectedEG2Factory(false))
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
	windowWide     bool  // a sliding row reaches distance W+1
	windowNarrow   bool  // a sliding row stops at distance W-1
	vNormWeight    bool  // v's norm takes the k weight instead of none
	failForward    bool  // a forward fails after its first layer is queued
	failForwardErr error // optional typed error for failForward (nil uses default string error)
	drainFails     int   // the next drainFails queue drains fail
}

// eg2Accel is the CUDA accelerator. One forward at a time (mu); towerOps runs every device call on its executor.
type eg2Accel struct {
	mu     sync.Mutex
	ops    *towerOps
	closed bool
	cfg    embeddinggemma2.Config
	model  *embeddinggemma2.Model

	pleProj, pleNorm, finalNorm, proj, ones Buffer
	residentWeightBytes                     int64
	layers                                  []eg2Layer
	ropes                                   []eg2Rope
	broken                                  error // set when a failed forward could not be drained; every later forward returns it
	combinedTranspose, combinedAttention    Pipeline
	defect                                  eg2Defect
}

// isSingleCUDAErrOutOfMemory reports whether err is a single-unwrap chain
// terminating at native gc.ErrOutOfMemory with no joined sibling errors.
func isSingleCUDAErrOutOfMemory(err error) bool {
	for cur := err; cur != nil; {
		if _, multi := cur.(interface{ Unwrap() []error }); multi {
			return false
		}
		if cur == gc.ErrOutOfMemory {
			return true
		}
		u, ok := cur.(interface{ Unwrap() error })
		if !ok {
			return errors.Is(cur, gc.ErrOutOfMemory)
		}
		cur = u.Unwrap()
	}
	return false
}

// isPureCUDAAllocOOM reports whether err is a single-wrapped *gpu.AllocFailure
// whose own Err is a single-wrapped native gc.ErrOutOfMemory with no joined
// sibling errors anywhere outside or inside AllocFailure. Only this pure
// allocation-only cause can certify forward retry safety.
func isPureCUDAAllocOOM(err error) bool {
	for cur := err; cur != nil; {
		if _, multi := cur.(interface{ Unwrap() []error }); multi {
			return false
		}
		if af, ok := cur.(*gpu.AllocFailure); ok {
			return isSingleCUDAErrOutOfMemory(af.Err)
		}
		u, ok := cur.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		cur = u.Unwrap()
	}
	return false
}

// isCUDACommonOOM reports whether err carries a typed native gc.ErrOutOfMemory
// anywhere in its error tree (never matching non-OOM AllocFailure or raw text).
func isCUDACommonOOM(err error) bool {
	return errors.Is(err, gc.ErrOutOfMemory)
}

func isCUDAFatalCode(code cudasys.CUresult) bool {
	switch code {
	case cudasys.CUDA_ERROR_ECC_UNCORRECTABLE,
		cudasys.CUDA_ERROR_CONTEXT_IS_DESTROYED,
		cudasys.CUDA_ERROR_ASSERT,
		cudasys.CUDA_ERROR_HARDWARE_STACK_ERROR,
		cudasys.CUDA_ERROR_ILLEGAL_INSTRUCTION,
		cudasys.CUDA_ERROR_MISALIGNED_ADDRESS,
		cudasys.CUDA_ERROR_INVALID_ADDRESS_SPACE,
		cudasys.CUDA_ERROR_INVALID_PC,
		cudasys.CUDA_ERROR_ILLEGAL_ADDRESS,
		cudasys.CUDA_ERROR_LAUNCH_TIMEOUT,
		cudasys.CUDA_ERROR_LAUNCH_FAILED,
		cudasys.CUDA_ERROR_INVALID_CONTEXT,
		cudasys.CUDA_ERROR_DEINITIALIZED,
		cudasys.CUDA_ERROR_NOT_INITIALIZED,
		cudasys.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
		cudasys.CUDA_ERROR_UNKNOWN:
		return true
	}
	return false
}

// firstCUDAFatalExec returns the first fatal native *cudaresult.Error anywhere
// in err's unwrap tree (walking single and joined errors), or nil if none.
func firstCUDAFatalExec(err error) *cudaresult.Error {
	if err == nil {
		return nil
	}
	if ce, ok := err.(*cudaresult.Error); ok && isCUDAFatalCode(ce.Code) {
		return ce
	}
	if multi, ok := err.(interface{ Unwrap() []error }); ok {
		for _, sub := range multi.Unwrap() {
			if fe := firstCUDAFatalExec(sub); fe != nil {
				return fe
			}
		}
		return nil
	}
	if single, ok := err.(interface{ Unwrap() error }); ok {
		return firstCUDAFatalExec(single.Unwrap())
	}
	return nil
}

func isCUDAFatalExec(err error) bool { return firstCUDAFatalExec(err) != nil }

// fatalFirst ensures errors.As(..., &ce) returns fatal before any earlier
// non-fatal *cudaresult.Error (such as CUDA_ERROR_OUT_OF_MEMORY in an OOM-first
// joined tree) while keeping the full original error chain intact.
func fatalFirst(fatal *cudaresult.Error, err error) error {
	if fatal == nil || err == nil {
		return err
	}
	var first *cudaresult.Error
	if errors.As(err, &first) && first == fatal {
		return err
	}
	return errors.Join(fatal, err)
}

// classifyEG2ForwardErr certifies ErrBatchOutOfMemory ONLY when err is a pure
// single-chain call-owned CUDA allocation OOM (no joined sibling errors), no
// launch/copy error was latched, drain and checked rollback release both
// succeeded, and the accelerator stays healthy. On any fatal, joined, or
// unrecovered combination it never attaches ErrBatchOutOfMemory regardless of
// join order, and orders any fatal native CUDA execution cause ahead of OOM
// so errors.As yields the fatal native code first while preserving all causes.
func classifyEG2ForwardErr(err, latchedErr, drainErr, releaseErr, broken error, closed bool) error {
	if err == nil && latchedErr == nil && drainErr == nil && releaseErr == nil {
		return nil
	}
	if err != nil && isPureCUDAAllocOOM(err) && latchedErr == nil && drainErr == nil && releaseErr == nil && broken == nil && !closed {
		return fmt.Errorf("cuda: EmbeddingGemma 2 forward: %w: %w", embeddinggemma2.ErrBatchOutOfMemory, err)
	}
	var fatal *cudaresult.Error
	switch {
	case firstCUDAFatalExec(broken) != nil:
		fatal = firstCUDAFatalExec(broken)
	case firstCUDAFatalExec(latchedErr) != nil:
		fatal = firstCUDAFatalExec(latchedErr)
	case firstCUDAFatalExec(err) != nil:
		fatal = firstCUDAFatalExec(err)
	}
	var parts []error
	brokenAdded := false
	latchedAdded := false
	if fatal != nil {
		switch {
		case broken != nil && firstCUDAFatalExec(broken) == fatal:
			parts = append(parts, fatalFirst(fatal, broken))
			brokenAdded = true
		case latchedErr != nil && firstCUDAFatalExec(latchedErr) == fatal:
			parts = append(parts, fatalFirst(fatal, latchedErr))
			latchedAdded = true
		default:
			parts = append(parts, fatal)
		}
	}
	if err != nil {
		if isCUDACommonOOM(err) {
			parts = append(parts, fmt.Errorf("%w: %w", embeddinggemma2.ErrDeviceOutOfMemory, err))
		} else {
			parts = append(parts, err)
		}
	}
	if latchedErr != nil && !latchedAdded && latchedErr != err && !errors.Is(broken, latchedErr) {
		parts = append(parts, latchedErr)
	}
	if broken != nil && !brokenAdded {
		parts = append(parts, broken)
	}
	if drainErr != nil && !errors.Is(broken, drainErr) {
		parts = append(parts, drainErr)
	}
	if releaseErr != nil && !errors.Is(broken, releaseErr) {
		parts = append(parts, releaseErr)
	}
	return fmt.Errorf("cuda: EmbeddingGemma 2 forward: %w", errors.Join(parts...))
}

func newEG2Accel(m *embeddinggemma2.Model, combined bool) (*eg2Accel, error) {
	return newEG2AccelOn(m, combined, CreateSystemDefaultDevice)
}

// newEG2AccelOn is the single classified EmbeddingGemma 2 CUDA attach path
// shared by legacy default attach, the combined attention selector, and
// explicitly selected device openers: open acquires the device while this
// constructor owns setup, partial-upload rollback, and OOM classification.
func newEG2AccelOn(m *embeddinggemma2.Model, combined bool, open func() (*Device, error)) (_ *eg2Accel, err error) {
	defer func() { err = classifyCUDAAttach(err) }()
	if open == nil {
		return nil, errors.New("cuda: EmbeddingGemma 2 nil device opener")
	}
	w := m.TransformerWeights()
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
	towerCleanup.mu.Lock()
	defer towerCleanup.mu.Unlock()
	ops, err := newTowerOpsOnLocked(open, maxN)
	if err != nil {
		return nil, err
	}
	a := &eg2Accel{ops: ops, cfg: c, model: m}
	if err := ops.do(func() error {
		var weightBytes int64
		upWeight := func(v []float32) Buffer {
			if len(v) > 0 {
				weightBytes += gpu.ChargedBytes(int64(len(v)) * 4)
			}
			return ops.up(v)
		}
		a.pleProj, a.pleNorm, a.finalNorm, a.proj = upWeight(w.PLEProj), upWeight(w.PLENorm), upWeight(w.FinalNorm), upWeight(w.Projection)
		ones := make([]float32, maxHD)
		for i := range ones {
			ones[i] = 1
		}
		a.ones = ops.up(ones) // synthetic helper, excluded from residentWeightBytes
		for _, ly := range w.Layers {
			a.layers = append(a.layers, eg2Layer{w: ly, inNorm: upWeight(ly.InNorm), q: upWeight(ly.Q), k: upWeight(ly.K), v: upWeight(ly.V), o: upWeight(ly.O),
				qNorm: upWeight(ly.QNorm), kNorm: upWeight(ly.KNorm), postAttnNorm: upWeight(ly.PostAttnNorm), preFFNorm: upWeight(ly.PreFFNorm),
				gate: upWeight(ly.Gate), up: upWeight(ly.Up), down: upWeight(ly.Down), postFFNorm: upWeight(ly.PostFFNorm), pleGate: upWeight(ly.PLEGate),
				pleProj: upWeight(ly.PLEProj), plePostNorm: upWeight(ly.PLEPostNorm)})
		}
		a.residentWeightBytes = weightBytes
		return ops.finish()
	}); err != nil {
		if cerr := retireTowerAttachLocked(ops); cerr != nil {
			return nil, fmt.Errorf("cuda: EmbeddingGemma 2 upload: %w", errors.Join(err, cerr))
		}
		return nil, fmt.Errorf("cuda: EmbeddingGemma 2 upload: %w", err)
	}
	if combined {
		if err := a.enableCombined(); err != nil {
			if cerr := retireTowerAttachLocked(a.ops); cerr != nil {
				return nil, errors.Join(err, cerr)
			}
			return nil, err
		}
	}
	return a, nil
}

func (a *eg2Accel) enableCombined() error {
	if err := a.ops.do(func() error {
		lib, err := a.ops.dev.CompileLibrary(eg2CombinedAttentionPTX)
		if err != nil {
			return err
		}
		a.combinedTranspose, err = a.ops.dev.NewComputePipeline(lib, "eg2_transpose_keys")
		if err != nil {
			return err
		}
		a.combinedAttention, err = a.ops.dev.NewComputePipeline(lib, "eg2_attention_transposed")
		return err
	}); err != nil {
		return fmt.Errorf("cuda: load EmbeddingGemma 2 transposed attention: %w", err)
	}
	a.ops.shortTiled = true
	return nil
}

func (a *eg2Accel) Name() string { return "cuda" }

// Close frees the encoder's device memory and stops its executor; a forward after it is an error. On an accelerator
// a failed forward left undrained (broken), it drains the queue first: if that fails too it frees nothing, returns the
// error and leaves the accelerator as it was, so a later Close can try again. Buffer teardown runs through
// closeChecked so a failed native cuMemFree preserves remaining ledger ownership and leases for a later Close retry.
// Once closed, Close does nothing.
func (a *eg2Accel) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	if a.broken != nil {
		if err := a.ops.do(a.drainQueue); err != nil {
			return fmt.Errorf("cuda: EmbeddingGemma 2 Close could not drain the queue, so nothing was freed: %w", err)
		}
	}
	if err := a.ops.closeChecked(); err != nil {
		if a.broken == nil {
			a.broken = fmt.Errorf("cuda: EmbeddingGemma 2 Close release failed: %w", err)
		}
		return fmt.Errorf("cuda: EmbeddingGemma 2 Close could not release device memory: %w", err)
	}
	a.closed, a.layers, a.ropes, a.model = true, nil, nil, nil
	return nil
}

// drainQueue waits for every queued launch and copy. The caller is on the executor.
func (a *eg2Accel) drainQueue() error {
	if a.defect.drainFails > 0 {
		a.defect.drainFails--
		return errors.New("cuda: planted drain failure")
	}
	return a.ops.q.Sync()
}

var errEG2Closed = errors.New("cuda: the EmbeddingGemma 2 accelerator is closed")

// checkPositions applies both position limits to one independent sequence.
func (a *eg2Accel) checkPositions(n int) error {
	if n > eg2MaxTokens {
		return fmt.Errorf("cuda: %w: %d positions, over the accelerator's %d", embeddinggemma2.ErrUnsupportedBatchShape, n, eg2MaxTokens)
	}
	if limit := a.cfg.MaxPositions; limit > 0 && n > limit {
		return fmt.Errorf("cuda: %w: %d positions, over max_position_embeddings %d", embeddinggemma2.ErrUnsupportedBatchShape, n, limit)
	}
	return nil
}

// validateSingleLength checks a single prepared sequence length for Forward
// and singleton BatchMemory without imposing ForwardBatch's aggregate row limit.
func (a *eg2Accel) validateSingleLength(n int) error {
	if n <= 0 {
		return fmt.Errorf("cuda: input has invalid length %d", n)
	}
	return a.checkPositions(n)
}

// validateBatchLengths checks prepared sequence lengths for ForwardBatch and
// multi-input BatchMemory before any allocation or arithmetic overflow.
func (a *eg2Accel) validateBatchLengths(lens []int) (total, maxT int, err error) {
	if len(lens) == 0 {
		return 0, 0, errors.New("cuda: an empty batch")
	}
	for b, n := range lens {
		if n <= 0 {
			return 0, 0, fmt.Errorf("cuda: batch input %d has invalid length %d", b, n)
		}
		if n > math.MaxInt-total {
			return 0, 0, errors.New("cuda: batch token sum overflows int")
		}
		total += n
		if n > maxT {
			maxT = n
		}
	}
	rowLimit := a.batchRowLimit()
	if len(lens) > rowLimit || total > rowLimit {
		return 0, 0, fmt.Errorf("cuda: %w: batch exceeds the launch limit of %d total token rows", embeddinggemma2.ErrUnsupportedBatchShape, rowLimit)
	}
	for b, n := range lens {
		if err := a.checkPositions(n); err != nil {
			return 0, 0, fmt.Errorf("cuda: batch input %d: %w", b, err)
		}
	}
	return total, maxT, nil
}

func checkedBufCharged(rows, cols, elemBytes int) (elems int, charged int64, err error) {
	if rows <= 0 || cols < 0 || elemBytes <= 0 {
		return 0, 0, fmt.Errorf("cuda: invalid buffer dimensions (%d, %d)", rows, cols)
	}
	r64, c64 := int64(rows), int64(cols)
	if c64 > 0 && r64 > math.MaxInt/c64 {
		return 0, 0, errors.New("cuda: buffer element count overflows int")
	}
	n64 := max(r64*c64, int64(1))
	maxRaw := int64(math.MaxInt64 - gpu.AllocQuantumBytes)
	if n64 > maxRaw/int64(elemBytes) {
		return 0, 0, errors.New("cuda: buffer byte size overflows int64")
	}
	return int(r64 * c64), gpu.ChargedBytes(n64 * int64(elemBytes)), nil
}

func addCharged(sum, delta int64) (int64, error) {
	if delta < 0 || delta > math.MaxInt64-sum {
		return 0, errors.New("cuda: workspace byte total overflows int64")
	}
	return sum + delta, nil
}

type eg2ScratchPlan struct {
	x, xn, q, k, v, qkv, ctx, ob, g, u, f, ple, pleL, gg, pr, res, bounds int
	chargedBytes                                                          int64
}

// scratchPlan computes the element counts and total charged device bytes for
// the 20 base per-forward scratch buffers (16 float32 and 4 int32 bounds).
// Shared by forwardRows and BatchMemory so buffer dimensions stay aligned.
func (a *eg2Accel) scratchPlan(T int) (eg2ScratchPlan, error) {
	c := a.cfg
	maxQD, maxKVD := 0, 0
	for _, ly := range a.layers {
		maxQD = max(maxQD, c.Heads*ly.w.HeadDim)
		maxKVD = max(maxKVD, ly.w.KVHeads*ly.w.HeadDim)
	}
	if c.Layers < 0 || c.PLEDim < 0 || (c.PLEDim > 0 && c.Layers > math.MaxInt/c.PLEDim) {
		return eg2ScratchPlan{}, errors.New("cuda: PLE geometry overflows int")
	}
	if maxQD > math.MaxInt/3 {
		return eg2ScratchPlan{}, errors.New("cuda: QKV geometry overflows int")
	}
	var p eg2ScratchPlan
	add := func(dst *int, cols, count, elemBytes int) error {
		elems, charged, err := checkedBufCharged(T, cols, elemBytes)
		if err != nil {
			return err
		}
		*dst = elems
		for range count {
			p.chargedBytes, err = addCharged(p.chargedBytes, charged)
			if err != nil {
				return err
			}
		}
		return nil
	}
	for _, spec := range []struct {
		dst   *int
		cols  int
		count int
		bytes int
	}{
		{&p.x, c.Hidden, 1, 4},
		{&p.xn, c.Hidden, 1, 4},
		{&p.q, maxQD, 1, 4},
		{&p.k, maxKVD, 1, 4},
		{&p.v, maxKVD, 1, 4},
		{&p.qkv, 3 * maxQD, 1, 4},
		{&p.ctx, maxQD, 1, 4},
		{&p.ob, c.Hidden, 1, 4},
		{&p.g, c.Intermediate, 1, 4},
		{&p.u, c.Intermediate, 1, 4},
		{&p.f, c.Hidden, 1, 4},
		{&p.ple, c.Layers * c.PLEDim, 1, 4},
		{&p.pleL, c.PLEDim, 1, 4},
		{&p.gg, c.PLEDim, 1, 4},
		{&p.pr, c.Hidden, 1, 4},
		{&p.res, c.EmbeddingDim, 1, 4},
		{&p.bounds, 1, 4, 4}, // slLoB, slHiB, fuLoB, fuHiB
	} {
		if err := add(spec.dst, spec.cols, spec.count, spec.bytes); err != nil {
			return eg2ScratchPlan{}, err
		}
	}
	return p, nil
}

func (a *eg2Accel) distinctRopeKeys() []eg2RopeKey {
	seen := make(map[eg2RopeKey]struct{}, len(a.layers))
	var keys []eg2RopeKey
	for _, ly := range a.layers {
		k := eg2RopeKey{ly.w.HeadDim, ly.w.RopeTheta}
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		keys = append(keys, k)
	}
	return keys
}

func (a *eg2Accel) ropeCachedRows(k eg2RopeKey) int {
	for i := range a.ropes {
		r := &a.ropes[i]
		if r.hd == k.hd && r.theta == k.theta {
			return r.rows
		}
	}
	return 0
}

// BatchMemory estimates the device working memory for the next forward of
// preparedLengths (EmbedIDs for one length, EmbedIDsBatch for multiple,
// keepLayers=false) without running a forward or allocating device memory.
//
// Quality is BatchMemoryUpperBound over explicitly tracked gpudev buffer
// allocations priced with gpudev.ChargedBytes. It does not bound physical
// cuMemGetInfo deltas, which also reflect driver page packing, context, and
// external process memory. When a cached RoPE table grows to
// max(preparedLengths), the full new pair is counted conservatively even though
// the older pair is released first. Both combinedAttention selectors reuse qkv
// and zeroBias without extra buffers.
func (a *eg2Accel) BatchMemory(preparedLengths []int) (embeddinggemma2.BatchMemoryEstimate, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return embeddinggemma2.BatchMemoryEstimate{}, errEG2Closed
	}
	if a.broken != nil {
		return embeddinggemma2.BatchMemoryEstimate{}, a.broken
	}
	var total, maxT int
	if len(preparedLengths) == 0 {
		return embeddinggemma2.BatchMemoryEstimate{}, errors.New("cuda: an empty batch")
	}
	if len(preparedLengths) == 1 {
		if err := a.validateSingleLength(preparedLengths[0]); err != nil {
			return embeddinggemma2.BatchMemoryEstimate{}, err
		}
		total, maxT = preparedLengths[0], preparedLengths[0]
	} else {
		var err error
		total, maxT, err = a.validateBatchLengths(preparedLengths)
		if err != nil {
			return embeddinggemma2.BatchMemoryEstimate{}, err
		}
	}
	plan, err := a.scratchPlan(total)
	if err != nil {
		return embeddinggemma2.BatchMemoryEstimate{}, err
	}
	peak := plan.chargedBytes
	multi := len(preparedLengths) > 1
	for _, k := range a.distinctRopeKeys() {
		if a.ropeCachedRows(k) < maxT {
			_, ropeBuf, err := checkedBufCharged(maxT, k.hd, 4)
			if err != nil {
				return embeddinggemma2.BatchMemoryEstimate{}, err
			}
			for range 2 {
				peak, err = addCharged(peak, ropeBuf)
				if err != nil {
					return embeddinggemma2.BatchMemoryEstimate{}, err
				}
			}
		}
		if multi {
			_, batchBuf, err := checkedBufCharged(total, k.hd, 4)
			if err != nil {
				return embeddinggemma2.BatchMemoryEstimate{}, err
			}
			for range 2 {
				peak, err = addCharged(peak, batchBuf)
				if err != nil {
					return embeddinggemma2.BatchMemoryEstimate{}, err
				}
			}
		}
	}
	return embeddinggemma2.BatchMemoryEstimate{
		AdditionalPeakBytes: peak,
		ResidentWeightBytes: a.residentWeightBytes,
		Quality:             embeddinggemma2.BatchMemoryUpperBound,
	}, nil
}

// Forward gathers the scaled token rows on the host, then runs the encoder.
func (a *eg2Accel) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, nil, errEG2Closed
	}
	if a.broken != nil {
		return nil, nil, a.broken
	}
	if err := a.validateSingleLength(len(ids)); err != nil {
		return nil, nil, err
	}
	x, err := a.model.EmbedTokens(ids)
	if err != nil {
		return nil, nil, err
	}
	return a.forwardRows(x, []int{len(ids)}, keepLayers)
}

// rope returns the tables for at least T rows at this head dim and theta,
// rebuilding them from the CPU's own values when T is past what is held. Each
// handle is registered in pending before upload and before allocating its
// sibling so a mid-pair allocation or upload failure never leaks an unowned
// buffer; only complete cos/sin pairs are published into a.ropes. Retiring an
// older smaller table uses ReleaseBufChecked so a failed native close preserves
// ownership and records releaseErr. The caller is on the executor.
func (a *eg2Accel) rope(T, hd int, theta float64, pending *[]Buffer, releaseErr *error) (Buffer, Buffer, error) {
	for i := range a.ropes {
		r := &a.ropes[i]
		if r.hd != hd || r.theta != theta {
			continue
		}
		if r.rows >= T {
			return r.cos, r.sin, nil
		}
		if r.cos != (Buffer{}) {
			if err := a.ops.dev.ReleaseBufChecked(r.cos); err != nil {
				*releaseErr = errors.Join(*releaseErr, fmt.Errorf("cuda: release retired RoPE cos table: %w", err))
				return Buffer{}, Buffer{}, *releaseErr
			}
			r.cos = Buffer{}
		}
		if r.sin != (Buffer{}) {
			if err := a.ops.dev.ReleaseBufChecked(r.sin); err != nil {
				*releaseErr = errors.Join(*releaseErr, fmt.Errorf("cuda: release retired RoPE sin table: %w", err))
				return Buffer{}, Buffer{}, *releaseErr
			}
			r.sin = Buffer{}
		}
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
	cos := gpu.NewBufferLenOfOn[float32](a.ops.dev, mem.LineCompanion, len(cf))
	*pending = append(*pending, cos)
	if err := gpu.Upload(cos, cf); err != nil {
		return Buffer{}, Buffer{}, fmt.Errorf("cuda: upload RoPE cos table: %w", err)
	}
	sin := gpu.NewBufferLenOfOn[float32](a.ops.dev, mem.LineCompanion, len(sf))
	*pending = append(*pending, sin)
	if err := gpu.Upload(sin, sf); err != nil {
		return Buffer{}, Buffer{}, fmt.Errorf("cuda: upload RoPE sin table: %w", err)
	}
	*pending = (*pending)[:len(*pending)-2]
	r := eg2Rope{hd: hd, theta: theta, rows: T, cos: cos, sin: sin}
	a.ropes = append(a.ropes, r)
	return r.cos, r.sin, nil
}

// eg2Bounds is each query row's key range [lo, hi) on a sliding layer of radius W (|i - j| <= W) or, with W < 0, a full one.
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

// eg2BatchBounds is eg2Bounds for inputs of the given lengths laid end to end: each row's range is its own input's,
// shifted to where that input starts, so no row reaches another input's keys.
func eg2BatchBounds(lens []int, W int) (lo, hi []int32) {
	off := 0
	for _, n := range lens {
		l, h := eg2Bounds(n, W)
		for i := range l {
			lo, hi = append(lo, l[i]+int32(off)), append(hi, h[i]+int32(off))
		}
		off += n
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
	if a.broken != nil {
		return nil, nil, a.broken
	}
	if err := a.checkPositions(T); err != nil {
		return nil, nil, err
	}
	if H := a.cfg.Hidden; T <= 0 || len(x0) != T*H {
		return nil, nil, fmt.Errorf("cuda: %d input values for %d rows of %d", len(x0), T, H)
	}
	return a.forwardRows(x0, []int{T}, keepLayers)
}

// batchRowLimit bounds the row grid of the smallest GEMM tile and the signed
// element indices used by the kernels. Leave room for a partial final block
// and for rounding the transposed attention pitch up to 32 rows.
func (a *eg2Accel) batchRowLimit() int {
	const maxGridY = 65535
	c := a.cfg
	stride := max(1, c.Hidden, c.Intermediate, c.Layers*c.PLEDim, c.EmbeddingDim)
	for _, layer := range a.layers {
		stride = max(stride, 3*c.Heads*layer.w.HeadDim)
	}
	return min(maxGridY*gpu.GEMMTile, (math.MaxInt32-255)/stride-31)
}

// ForwardBatch runs the encoder over several inputs in one forward: their token rows laid end to end, each attending
// only within itself at its own positions 0..T_b-1, every row-wise step over all of them at once. It returns each
// input's projected last hidden state, [len(ids[b]), E] row-major, in input order, for the caller to pool.
// Each input has its own position cap. A batch of one runs exactly what Forward runs.
func (a *eg2Accel) ForwardBatch(ids [][]int) ([][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, errEG2Closed
	}
	if a.broken != nil {
		return nil, a.broken
	}
	c := a.cfg
	E := c.EmbeddingDim
	if len(ids) == 0 {
		return nil, fmt.Errorf("cuda: an empty batch")
	}
	// Reject impossible batch counts in O(1) before allocating the lengths slice.
	if rowLimit := a.batchRowLimit(); len(ids) > rowLimit {
		return nil, fmt.Errorf("cuda: %w: batch has %d inputs, over the launch limit of %d total token rows", embeddinggemma2.ErrUnsupportedBatchShape, len(ids), rowLimit)
	}
	// Every check runs before the gather buffer is allocated, so a refused batch costs no more than its lengths.
	lens := make([]int, len(ids))
	for b, in := range ids {
		lens[b] = len(in)
	}
	if _, _, err := a.validateBatchLengths(lens); err != nil {
		return nil, err
	}
	for b, in := range ids {
		for _, id := range in {
			if id < 0 || id >= c.VocabSize {
				return nil, fmt.Errorf("cuda: batch input %d: token id %d out of range [0, %d)", b, id, c.VocabSize)
			}
		}
	}
	x, err := a.model.EmbedTokens(slices.Concat(ids...))
	if err != nil {
		return nil, err
	}
	out, _, err := a.forwardRows(x, lens, false)
	if err != nil {
		return nil, err
	}
	res := make([][]float32, len(ids))
	off := 0
	for b, n := range lens {
		res[b] = out[off*E : (off+n)*E : (off+n)*E]
		off += n
	}
	return res, nil
}

type eg2RopeKey struct {
	hd    int
	theta float64
}

// batchRope builds, for inputs of the given lengths laid end to end, one cos and one sin table per (head dim, theta)
// the layers use: input b's rows hold the cached table's first len(b) rows, its own positions 0..len(b)-1. Every cached
// table is grown before the first copy is queued, so no source a copy reads is replaced under it; the tables are
// per-call scratch (towerOps.af), freed when the forward's scratch is. Copy errors latch as launch errors do. The
// caller is on the executor.
func (a *eg2Accel) batchRope(lens []int, pending *[]Buffer, releaseErr *error) (map[eg2RopeKey][2]Buffer, error) {
	o := a.ops
	T, maxT := 0, 0
	for _, n := range lens {
		T, maxT = T+n, max(maxT, n)
	}
	keys := a.distinctRopeKeys()
	src := make(map[eg2RopeKey][2]Buffer, len(keys))
	for _, k := range keys {
		cs, sn, err := a.rope(maxT, k.hd, k.theta, pending, releaseErr)
		if err != nil {
			return nil, err
		}
		src[k] = [2]Buffer{cs, sn}
	}
	out := map[eg2RopeKey][2]Buffer{}
	for _, k := range keys {
		dst := [2]Buffer{o.af(T * k.hd), o.af(T * k.hd)}
		off := 0
		for _, n := range lens {
			w := n * k.hd * 4
			for j := range dst {
				if o.err == nil {
					o.err = o.q.Copy2DAsync(dst[j].At(off*k.hd*4), w, src[k][j], w, w, 1)
				}
			}
			off += n
		}
		out[k] = dst
	}
	return out, nil
}

// forwardRows is the forward over inputs of the given lengths whose prepared rows x0 holds end to end. One input takes
// the cached RoPE tables as they are; several take a table per (head dim, theta) gathered on the device from each
// input's own leading rows. The caller holds mu and has checked the lengths.
func (a *eg2Accel) forwardRows(x0 []float32, lens []int, keepLayers bool) ([]float32, [][]float32, error) {
	c := a.cfg
	H, I, P, L, E, nH := c.Hidden, c.Intermediate, c.PLEDim, c.Layers, c.EmbeddingDim, c.Heads
	T, maxT := 0, 0
	for _, n := range lens {
		T, maxT = T+n, max(maxT, n)
	}
	W := c.SlidingWindow
	if a.defect.windowWide {
		W++
	}
	if a.defect.windowNarrow {
		W--
	}
	slLo, slHi := eg2BatchBounds(lens, W)
	fuLo, fuHi := eg2BatchBounds(lens, -1)
	plan, err := a.scratchPlan(T)
	if err != nil {
		return nil, nil, err
	}
	var out []float32
	var layers [][]float32
	if keepLayers {
		layers = append(layers, append([]float32(nil), x0...))
	}
	var (
		latchedErr     error
		drainErr       error
		releaseErr     error
		ropeReleaseErr error
	)
	err = a.ops.do(func() error {
		o := a.ops
		done := false
		var pendingRope []Buffer
		defer func() {
			// A failed or panicking forward may leave copies and launches queued on its scratch: capture any latched
			// launch/copy error before clearing o.err, then drain before freeing. If drain fails, retain all possibly
			// pending buffers (Close frees them once a drain succeeds) and poison the accelerator. If rollback release
			// fails, preserve ledger ownership of failed frees and poison the accelerator.
			if !done {
				latchedErr = o.err
				o.err = nil
				if e := a.drainQueue(); e != nil {
					drainErr = e
					o.scratch = append(o.scratch, pendingRope...)
					pendingRope = nil
					return
				}
				releaseErr = o.releaseScratchChecked()
				for _, b := range pendingRope {
					if e := o.dev.ReleaseBufChecked(b); e != nil {
						releaseErr = errors.Join(releaseErr, e)
						o.scratch = append(o.scratch, b)
					}
				}
				pendingRope = nil
				if ropeReleaseErr != nil {
					releaseErr = errors.Join(ropeReleaseErr, releaseErr)
				}
				return
			}
			if e := o.releaseScratchChecked(); e != nil {
				releaseErr = e
			}
		}()
		af := o.af
		x, xn := af(plan.x), af(plan.xn)
		q, k, v := af(plan.q), af(plan.k), af(plan.v)
		qkv, ctx := af(plan.qkv), af(plan.ctx)
		ob, g, u, f := af(plan.ob), af(plan.g), af(plan.u), af(plan.f)
		ple, pleL, gg, pr := af(plan.ple), af(plan.pleL), af(plan.gg), af(plan.pr)
		res := af(plan.res)
		slLoB, slHiB, fuLoB, fuHiB := o.ai(plan.bounds), o.ai(plan.bounds), o.ai(plan.bounds), o.ai(plan.bounds)
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
		var batchRope map[eg2RopeKey][2]Buffer
		if len(lens) > 1 {
			var e error
			batchRope, e = a.batchRope(lens, &pendingRope, &ropeReleaseErr)
			if e != nil {
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
			var cs, sn Buffer
			if r, ok := batchRope[eg2RopeKey{hd, ly.w.RopeTheta}]; ok {
				cs, sn = r[0], r[1]
			} else {
				var e error
				cs, sn, e = a.rope(T, hd, ly.w.RopeTheta, &pendingRope, &ropeReleaseErr)
				if e != nil {
					return e
				}
			}
			o.ropeHalfTo(q, cs, sn, T, nH, hd)
			o.ropeHalfTo(k, cs, sn, T, nKV, hd)
			pitch := (T + 31) &^ 31
			// Gate on the longest individual input. A pair of short inputs keeps the packed kernel.
			transposed := a.combinedAttention != (Pipeline{}) && maxT >= 128 && hd <= 512 && kvd*pitch <= qkv.Len()
			if transposed {
				o.launch(a.combinedTranspose, LaunchConfig{
					GridX: uint32((kvd + 31) / 32), GridY: uint32((T + 31) / 32), GridZ: 1,
					BlockX: 32, BlockY: 8, BlockZ: 1,
				}, Arg(k), Arg(qkv), i32(int32(T)), i32(int32(kvd)), i32(int32(pitch)))
			} else if e := a.pack(qkv, q, k, v, T, nH, hd, group); e != nil {
				return e
			}
			lo, hi, maxSeg := slLoB, slHiB, min(maxT, 2*W+1)
			if ly.w.Full {
				lo, hi, maxSeg = fuLoB, fuHiB, maxT
			}
			o.cls(clsAttn)
			grid := gpu.SegAttentionGrid(T, nH, maxSeg)
			if transposed {
				o.launch(a.combinedAttention, grid, Arg(q), Arg(qkv), Arg(v), Arg(ctx), Arg(lo), Arg(hi),
					i32(int32(T)), i32(int32(nH)), i32(int32(nKV)), i32(int32(hd)), i32(int32(pitch)), f32v(1))
			} else {
				o.launch(o.vit.AttentionSeg, grid, Arg(qkv), Arg(ctx), Arg(lo), Arg(hi),
					i32(int32(T)), i32(int32(nH)), i32(int32(hd)), f32v(1))
			}
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
			if a.defect.failForward && o.err == nil {
				if a.defect.failForwardErr != nil {
					o.err = a.defect.failForwardErr
				} else {
					o.err = errors.New("cuda: planted forward failure")
				}
			}
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
		if e := gpu.Download(res, out); e != nil {
			return e
		}
		done = true
		return nil
	})
	if a.broken == nil {
		var primary error
		var fatal *cudaresult.Error
		if fe := firstCUDAFatalExec(latchedErr); fe != nil {
			fatal, primary = fe, latchedErr
		} else if fe := firstCUDAFatalExec(err); fe != nil {
			fatal, primary = fe, err
		}
		if primary != nil {
			a.broken = fmt.Errorf("cuda: fatal EmbeddingGemma 2 execution error: %w",
				errors.Join(fatalFirst(fatal, primary), drainErr, releaseErr))
		} else if drainErr != nil {
			a.broken = fmt.Errorf("cuda: a failed EmbeddingGemma 2 forward could not be drained: %w", drainErr)
		} else if releaseErr != nil {
			a.broken = fmt.Errorf("cuda: EmbeddingGemma 2 rollback release failed: %w", releaseErr)
		}
	}
	if classified := classifyEG2ForwardErr(err, latchedErr, drainErr, releaseErr, a.broken, a.closed); classified != nil {
		return nil, nil, classified
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
