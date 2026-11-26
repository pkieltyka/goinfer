//go:build darwin

package metal

import (
	"errors"
	"fmt"
	"math"
	"runtime"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

type eg2AttentionRequest uint8

const (
	eg2AttentionDefault eg2AttentionRequest = iota
	eg2AttentionStock
	eg2AttentionMatrix
	eg2AttentionBatch
)

const (
	eg2StockProfile  = "metal-eg2-stock-v1"
	eg2MatrixProfile = "metal-eg2-combined-v1"
	eg2MatrixHeads   = 4
	eg2MatrixWindow  = 512
	eg2MatrixThreads = 128
)

type eg2MatrixPipes struct{ full, sliding Pipeline }

func eg2MatrixGeometry(w embeddinggemma2.Weights) bool {
	c := w.Config
	if c.Heads != eg2MatrixHeads || c.SlidingWindow != eg2MatrixWindow || len(w.Layers) != 24 || c.Layers != len(w.Layers) {
		return false
	}
	for i, layer := range w.Layers {
		full := i%6 == 5
		hd, kv := 256, 2
		if full {
			hd, kv = 512, 1
		}
		if layer.Full != full || layer.HeadDim != hd || layer.KVHeads != kv {
			return false
		}
	}
	return true
}

func resolveEG2Attention(request eg2AttentionRequest, eligible bool, setup func() error) (string, error) {
	if request == eg2AttentionStock {
		return eg2StockProfile, nil
	}
	err := fmt.Errorf("%w: Metal matrix attention requires the full and sliding EG2 layer geometries", embeddinggemma2.ErrUnsupportedSelector)
	if eligible {
		err = setup()
	}
	switch {
	case err == nil:
		return eg2MatrixProfile, nil
	case request == eg2AttentionDefault && errors.Is(err, embeddinggemma2.ErrUnsupportedSelector):
		return eg2StockProfile, nil
	default:
		return "", err
	}
}

func (o *eg2Ops) newEG2MatrixPipes() (eg2MatrixPipes, error) {
	d := o.d
	if runtime.GOARCH != "arm64" || !d.HasUnifiedMemory() || !d.SupportsApple7() {
		return eg2MatrixPipes{}, fmt.Errorf("%w: Metal matrix attention requires Apple7 features and unified memory", embeddinggemma2.ErrUnsupportedSelector)
	}
	// Compile the matrix entries separately so a capability decline leaves the stock
	// library unchanged. A compiler error is not evidence of unsupported hardware.
	lib, err := d.CompileLibraryPrecise("#include <metal_stdlib>\nusing namespace metal;\n"+eg2MatrixAttnExtraMSL+eg2MatrixSlidingExtraMSL, MSL3_1)
	if err != nil {
		return eg2MatrixPipes{}, fmt.Errorf("metal: compile EG2 matrix attention: %w", err)
	}
	defer d.ReleaseObject(lib)
	var pipes eg2MatrixPipes
	ok := false
	defer func() {
		if !ok {
			d.ReleasePipeline(pipes.sliding)
			d.ReleasePipeline(pipes.full)
		}
	}()
	for _, kernel := range []struct {
		name string
		pipe *Pipeline
	}{
		{"eg2_matrix_attn_hd512", &pipes.full},
		{"eg2_matrix_sliding_hd256", &pipes.sliding},
	} {
		*kernel.pipe, err = d.NewComputePipeline(lib, kernel.name)
		if err != nil {
			return eg2MatrixPipes{}, err
		}
		p := *kernel.pipe
		bytes := p.StaticThreadgroupMemoryLength()
		if p.ThreadExecutionWidth() != 32 || p.MaxTotalThreadsPerThreadgroup() < eg2MatrixThreads ||
			bytes <= 0 || bytes > d.MaxThreadgroupMemoryLength() {
			return eg2MatrixPipes{}, fmt.Errorf("%w: Metal pipeline %s cannot run the EG2 matrix launch geometry", embeddinggemma2.ErrUnsupportedSelector, kernel.name)
		}
	}
	if err := o.checkEG2Matrix(pipes); err != nil {
		return eg2MatrixPipes{}, err
	}
	ok = true
	return pipes, nil
}

// dispatchEG2Matrix takes the same row-major buffers for the probe and the
// real forward. Attachment has checked every layer's geometry.
func (o *eg2Ops) dispatchEG2Matrix(e *Encoder, p eg2MatrixPipes, q, k, v, out Buffer, rows int, full bool) {
	if full {
		e.Dispatch2D(p.full, (rows+7)/8, eg2MatrixHeads, eg2MatrixThreads, 1, q, k, v, out,
			o.u32(uint32(rows)), o.u32(eg2MatrixHeads), o.u32(1))
		return
	}
	e.Dispatch2D(p.sliding, (rows+15)/16, eg2MatrixHeads, eg2MatrixThreads, 1, q, k, v, out,
		o.u32(uint32(rows)), o.u32(eg2MatrixHeads), o.u32(2), o.u32(eg2MatrixWindow))
}

const (
	eg2ProbeRows = 17
	eg2ProbeTail = 16
	eg2ProbeMark = float32(-16384)
)

// The probe crosses both query and key tile boundaries. Each (row, head)
// selects two known logits in a distinct channel, and V varies along all axes.
func eg2ProbeInputs(q, k, v []float32, hd, kv int) {
	qStride, kvStride := eg2MatrixHeads*hd, kv*hd
	clear(q)
	clear(k)
	for row := range eg2ProbeRows {
		for head := range eg2MatrixHeads {
			group := head / (eg2MatrixHeads / kv)
			channel := 128*(row%(hd/128)) + 17*head + row
			key := (5*row + 3*head) % eg2ProbeRows
			q[row*qStride+head*hd+channel] = 3
			k[key*kvStride+group*hd+channel] = 1
			k[((key+1)%eg2ProbeRows)*kvStride+group*hd+channel] = -0.5
		}
		for group := range kv {
			for channel := range hd {
				v[row*kvStride+group*hd+channel] = float32(513*group+13*row+(37*channel+11*row+3*row*channel)%521-260) / 512
			}
		}
	}
	for _, tail := range [][]float32{q[eg2ProbeRows*qStride:], k[eg2ProbeRows*kvStride:], v[eg2ProbeRows*kvStride:]} {
		for i := range tail {
			tail[i] = 1e6
		}
	}
}

// eg2ProbeOutput checks every value against the analytical FP64 softmax of
// the two tagged logits, and checks the entire output guard.
func eg2ProbeOutput(out, v []float32, hd, kv int) error {
	for i, word := range out[eg2ProbeRows*eg2MatrixHeads*hd:] {
		if word != eg2ProbeMark {
			return fmt.Errorf("metal: EG2 matrix canary HD%d wrote outside its output at word %d", hd, i)
		}
	}
	positive, negative := math.Exp(3), math.Exp(-1.5)
	sum := positive + negative + eg2ProbeRows - 2
	kvStride := kv * hd
	for row := range eg2ProbeRows {
		for head := range eg2MatrixHeads {
			group := head / (eg2MatrixHeads / kv)
			key := (5*row + 3*head) % eg2ProbeRows
			for channel := range hd {
				offset := group*hd + channel
				var want float64
				for j := range eg2ProbeRows {
					want += float64(v[j*kvStride+offset])
				}
				want += (positive-1)*float64(v[key*kvStride+offset]) + (negative-1)*float64(v[((key+1)%eg2ProbeRows)*kvStride+offset])
				want /= sum
				got := float64(out[(row*eg2MatrixHeads+head)*hd+channel])
				if !(math.Abs(got-want) <= 2e-5+2e-5*math.Abs(want)) {
					return fmt.Errorf("%w: Metal EG2 matrix canary HD%d row %d head %d channel %d: got %.9g, want %.9g", embeddinggemma2.ErrUnsupportedSelector, hd, row, head, channel, got, want)
				}
			}
		}
	}
	return nil
}

func (o *eg2Ops) checkEG2Matrix(p eg2MatrixPipes) error {
	type probe struct {
		hd, kv int
		full   bool
		buf    Buffer
	}
	probes := [2]probe{{hd: 512, kv: 1, full: true}, {hd: 256, kv: 2}}
	defer func() {
		o.slot = 0
		for _, s := range probes {
			o.d.ReleaseBuf(s.buf)
		}
	}()
	rows := eg2ProbeRows + eg2ProbeTail
	for i := range probes {
		s := &probes[i]
		qLen, kvLen := rows*eg2MatrixHeads*s.hd, rows*s.kv*s.hd
		buf, ok := o.d.TryNewBufferLenOn(mem.LineCompanion, 2*(qLen+kvLen))
		if !ok {
			return fmt.Errorf("%w: Metal EG2 matrix canary HD%d", embeddinggemma2.ErrDeviceOutOfMemory, s.hd)
		}
		s.buf = buf
		data := buf.Floats()
		eg2ProbeInputs(data[:qLen], data[qLen:qLen+kvLen], data[qLen+kvLen:qLen+2*kvLen], s.hd, s.kv)
		for j := qLen + 2*kvLen; j < len(data); j++ {
			data[j] = eg2ProbeMark
		}
	}
	e := o.q.Begin()
	for _, s := range probes {
		qLen, kvLen := rows*eg2MatrixHeads*s.hd, rows*s.kv*s.hd
		o.dispatchEG2Matrix(e, p, s.buf, s.buf.At(4*qLen), s.buf.At(4*(qLen+kvLen)), s.buf.At(4*(qLen+2*kvLen)), eg2ProbeRows, s.full)
	}
	e.End()
	if err := e.Err(); err != nil {
		return fmt.Errorf("metal: EG2 matrix canary dispatch: %w", err)
	}
	// Guard faults are hard errors, even if another output has a numerical
	// mismatch that would normally allow the default to select stock.
	var mismatch error
	for _, s := range probes {
		qLen, kvLen := rows*eg2MatrixHeads*s.hd, rows*s.kv*s.hd
		data := s.buf.Floats()
		if err := eg2ProbeOutput(data[qLen+2*kvLen:], data[qLen+kvLen:qLen+2*kvLen], s.hd, s.kv); err != nil {
			if !errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) {
				return err
			}
			mismatch = err
		}
	}
	return mismatch
}
