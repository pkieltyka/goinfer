//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"

	gpu "github.com/pkieltyka/goinfer/backend/metal/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

const (
	eg2BatchMaxDocs   = 4
	eg2BatchMaxRows   = 2748
	eg2RopeCacheLimit = 16
)

// eg2BatchAccel adds native prepared-ID batches to the explicit selector.
type eg2BatchAccel struct {
	*eg2Accel
}

func eg2BatchGeometry(staged bool, w embeddinggemma2.Weights) bool {
	if !staged || w.Config.PLEDim != 512 || w.Config.EmbeddingDim != 768 {
		return false
	}
	var keys [eg2RopeCacheLimit]eg2RopeKey
	n := 0
	for _, ly := range w.Layers {
		if ly.RopeTheta <= 0 || math.IsNaN(ly.RopeTheta) || math.IsInf(ly.RopeTheta, 0) {
			return false
		}
		key := eg2RopeKey{hd: ly.HeadDim, theta: ly.RopeTheta}
		i := 0
		for i < n && keys[i] != key {
			i++
		}
		if i == n {
			if n == len(keys) {
				return false
			}
			keys[n] = key
			n++
		}
	}
	return true
}

func (a *eg2BatchAccel) AttachedDevice() embeddinggemma2.DeviceInfo {
	info := a.eg2Accel.AttachedDevice()
	info.TextBatching = true
	return info
}

type eg2BatchRows struct {
	start, rows int
}

type eg2BatchRun struct {
	eg2BatchRows
	staged bool
}

type eg2BatchPlan struct {
	docs                     [eg2BatchMaxDocs]eg2BatchRows
	runs                     [eg2BatchMaxDocs]eg2BatchRun
	docCount, runCount, rows int

	ropeKeys  [eg2RopeCacheLimit]eg2RopeKey
	newRope   [eg2RopeCacheLimit]bool
	layerRope [24]int
	ropeCount int
	evictRope bool

	arenaBytes      int
	scratchWords    [20]int
	growScratch     bool
	additionalBytes int64
}

// eg2BatchCharge rounds each standalone buffer up to a host page. The estimate
// excludes driver-private memory.
func eg2BatchCharge(bytes int) int64 {
	page := os.Getpagesize()
	return int64((bytes + page - 1) / page * page)
}

// planBatch reads the retained state under mu. The geometry and row bounds
// make every word count, byte offset and shader scalar fit comfortably in int
// and every shader count fit in uint32.
func (a *eg2BatchAccel) planBatch(lengths []int) (eg2BatchPlan, error) {
	var p eg2BatchPlan
	if len(lengths) == 0 {
		return p, fmt.Errorf("metal: an empty text batch")
	}
	if len(lengths) > eg2BatchMaxDocs {
		return p, fmt.Errorf("%w: Metal accepts at most %d documents", embeddinggemma2.ErrUnsupportedBatchShape, eg2BatchMaxDocs)
	}
	p.docCount = len(lengths)
	longest := 0
	for i, length := range lengths {
		if length <= 0 {
			return p, fmt.Errorf("metal: invalid length %d for document %d", length, i)
		}
		if err := a.checkLength(length); err != nil {
			return p, err
		}
		if p.docCount > 1 && length > eg2BatchMaxRows-p.rows {
			return p, fmt.Errorf("%w: Metal accepts at most %d rows in a multi-document batch", embeddinggemma2.ErrUnsupportedBatchShape, eg2BatchMaxRows)
		}
		doc := eg2BatchRows{start: p.rows, rows: length}
		p.docs[i] = doc
		staged := eg2StagedFFNRows(length)
		last := p.runCount - 1
		if last >= 0 && p.runs[last].staged == staged &&
			(!staged || eg2StagedFFNRows(p.runs[last].rows+length)) {
			p.runs[last].rows += length
		} else {
			p.runs[p.runCount] = eg2BatchRun{eg2BatchRows: doc, staged: staged}
			p.runCount++
		}
		p.rows += length
		longest = max(longest, length)
	}

	slots := eg2ArenaOpeningSlots + eg2ArenaClosingSlots +
		a.cfg.Layers*(eg2ArenaLayerSlots+16*p.docCount+10*p.runCount)
	if p.docCount == 1 {
		var err error
		slots, err = eg2ArenaSlotsFor(a.cfg, a.kvHeads, p.rows, false)
		if err != nil {
			return p, err
		}
	}
	if need := slots * 256; need > a.arena.Len() {
		p.arenaBytes = need
		p.additionalBytes += eg2BatchCharge(need)
	}
	if p.rows > a.cap {
		p.growScratch = true
		p.scratchWords = a.scratchWords(p.rows)
		for _, words := range p.scratchWords {
			p.additionalBytes += eg2BatchCharge(4 * words)
		}
	}
	missing := 0
	for li, ly := range a.layers {
		key := eg2RopeKey{T: longest, hd: ly.w.HeadDim, theta: ly.w.RopeTheta}
		idx := 0
		for idx < p.ropeCount && p.ropeKeys[idx] != key {
			idx++
		}
		if idx == p.ropeCount {
			if idx == len(p.ropeKeys) {
				return p, fmt.Errorf("%w: Metal batch needs more than %d RoPE tables", embeddinggemma2.ErrUnsupportedBatchShape, eg2RopeCacheLimit)
			}
			p.ropeKeys[idx] = key
			p.ropeCount++
			if _, ok := a.ropes[key]; !ok {
				p.newRope[idx] = true
				missing++
			}
		}
		p.layerRope[li] = idx
	}
	p.evictRope = len(a.ropes)+missing > eg2RopeCacheLimit
	for i, key := range p.ropeKeys[:p.ropeCount] {
		if p.evictRope || p.newRope[i] {
			p.newRope[i] = true
			p.additionalBytes += 2 * eg2BatchCharge(4*key.T*(key.hd/2))
		}
	}
	return p, nil
}

// BatchMemory prices explicit device buffers relative to the retained state.
// Growth keeps old buffers alive until every replacement is ready, so the
// estimate includes each full, page-rounded new allocation.
func (a *eg2BatchAccel) BatchMemory(lengths []int) (embeddinggemma2.BatchMemoryEstimate, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.open(); err != nil {
		return embeddinggemma2.BatchMemoryEstimate{}, err
	}
	p, err := a.planBatch(lengths)
	if err != nil {
		return embeddinggemma2.BatchMemoryEstimate{}, err
	}
	return embeddinggemma2.BatchMemoryEstimate{
		AdditionalPeakBytes: p.additionalBytes,
		ResidentWeightBytes: a.weightBytes,
		Quality:             embeddinggemma2.BatchMemoryApproximate,
	}, nil
}

type eg2BatchBufferKind uint8

const (
	eg2BatchScratch eg2BatchBufferKind = iota
	eg2BatchArena
	eg2BatchRoPE
)

// The allocator boundary lets native tests inject nil at individual Metal
// allocation sites and inspect the same cleanup used for an actual refusal.
type eg2BatchAlloc func(kind eg2BatchBufferKind, size int, data []float32) (Buffer, bool)

func (a *eg2BatchAccel) allocateBatchBuffer(kind eg2BatchBufferKind, size int, data []float32) (Buffer, bool) {
	switch kind {
	case eg2BatchScratch:
		return a.d.TryNewBufferLenOn(mem.LineCompanion, size)
	case eg2BatchArena:
		return a.d.TryNewBufferBytesOn(mem.LineCompanion, size)
	default:
		return gpu.TryNewBufferOfOn(a.d, mem.LineCompanion, data)
	}
}

// prepareBatch commits all replacements together before any dispatch.
func (a *eg2BatchAccel) prepareBatch(p *eg2BatchPlan, alloc eg2BatchAlloc) (_ [eg2RopeCacheLimit][2]Buffer, err error) {
	var arena Buffer
	var scratch eg2Scratch
	var added [eg2RopeCacheLimit][2]Buffer
	var tables [eg2RopeCacheLimit][2]Buffer
	defer func() {
		if err != nil {
			a.release(&arena)
			a.releaseScratch(&scratch)
			for i := range added[:p.ropeCount] {
				a.release(&added[i][0], &added[i][1])
			}
		}
	}()
	if p.arenaBytes != 0 {
		var ok bool
		arena, ok = alloc(eg2BatchArena, p.arenaBytes, nil)
		if !ok {
			return tables, fmt.Errorf("%w: Metal text scalar arena of %d bytes", embeddinggemma2.ErrBatchOutOfMemory, p.arenaBytes)
		}
	}
	if p.growScratch {
		scratch, err = a.newScratch(p.scratchWords, func(words int) (Buffer, bool) {
			return alloc(eg2BatchScratch, words, nil)
		})
		if err != nil {
			return tables, fmt.Errorf("%w: %w", embeddinggemma2.ErrBatchOutOfMemory, err)
		}
	}
	for i, key := range p.ropeKeys[:p.ropeCount] {
		if !p.newRope[i] {
			tables[i] = a.ropes[key]
			continue
		}
		cos, sin := embeddinggemma2.RopeTables(key.T, key.hd, key.theta)
		for j, data := range [2][]float32{cos, sin} {
			var ok bool
			added[i][j], ok = alloc(eg2BatchRoPE, len(data), data)
			if !ok {
				return tables, fmt.Errorf("%w: Metal text RoPE table for %d rows", embeddinggemma2.ErrBatchOutOfMemory, key.T)
			}
		}
		tables[i] = added[i]
	}
	if p.arenaBytes != 0 {
		a.release(&a.arena)
		a.arena = arena
	}
	if p.growScratch {
		a.releaseScratch(&a.scr)
		a.scr, a.cap = scratch, p.rows
	}
	if p.evictRope {
		for key, pair := range a.ropes {
			a.release(&pair[0], &pair[1])
			delete(a.ropes, key)
		}
	}
	for i, key := range p.ropeKeys[:p.ropeCount] {
		a.ropes[key] = tables[i]
	}
	return tables, nil
}

// Forward uses the same planned resources as BatchMemory for a singleton.
func (a *eg2BatchAccel) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	if keepLayers {
		return a.eg2Accel.Forward(ids, true)
	}
	out, err := a.ForwardBatch([][]int{ids})
	if err != nil {
		return nil, nil, err
	}
	return out[0], nil, nil
}

// ForwardBatch keeps packed rows resident through all layers. The output is
// copied once, then split into independent document views for encoder pooling.
func (a *eg2BatchAccel) ForwardBatch(ids [][]int) ([][]float32, error) {
	return a.forwardBatch(ids, a.allocateBatchBuffer)
}

func (a *eg2BatchAccel) forwardBatch(ids [][]int, alloc eg2BatchAlloc) ([][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.open(); err != nil {
		return nil, err
	}
	if len(ids) > eg2BatchMaxDocs {
		return nil, fmt.Errorf("%w: Metal accepts at most %d documents", embeddinggemma2.ErrUnsupportedBatchShape, eg2BatchMaxDocs)
	}
	var lengths [eg2BatchMaxDocs]int
	for i, in := range ids {
		lengths[i] = len(in)
	}
	p, err := a.planBatch(lengths[:len(ids)])
	if err != nil {
		return nil, err
	}
	for i, in := range ids {
		for _, id := range in {
			if id < 0 || id >= a.cfg.VocabSize {
				return nil, fmt.Errorf("metal: document %d has out-of-range token %d", i, id)
			}
		}
	}
	tables, err := a.prepareBatch(&p, alloc)
	if err != nil {
		return nil, err
	}
	h, d := a.cfg.Hidden, a.cfg.EmbeddingDim
	x := a.scr.x.Floats()[:p.rows*h]
	for i, doc := range p.docs[:p.docCount] {
		rows, err := a.model.EmbedTokens(ids[i])
		if err != nil {
			return nil, err
		}
		copy(x[doc.start*h:(doc.start+doc.rows)*h], rows)
	}
	if _, err := a.encodeTextRows(p.rows, false, &p, &tables); err != nil {
		return nil, err
	}
	projected := append([]float32(nil), a.scr.out.Floats()[:p.rows*d]...)
	out := make([][]float32, p.docCount)
	for i, doc := range p.docs[:p.docCount] {
		lo, hi := doc.start*d, (doc.start+doc.rows)*d
		out[i] = projected[lo:hi:hi]
	}
	return out, nil
}
