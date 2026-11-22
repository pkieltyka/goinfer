package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/pkieltyka/goinfer/internal/runtime/f16"
)

// Type is the GGML on-disk tensor type.
type Type uint32

const (
	F32  Type = 0
	Q8_0 Type = 8
	Q6_K Type = 14
	BF16 Type = 30
)

type blockLayout struct{ values, bytes int }

func (t Type) block() blockLayout {
	switch t {
	case F32:
		return blockLayout{1, 4}
	case BF16:
		return blockLayout{1, 2}
	case Q8_0:
		return blockLayout{32, 34}
	case Q6_K:
		return blockLayout{256, 210}
	default:
		return blockLayout{}
	}
}

// Tensor describes a validated tensor in GGUF dimension order. Shape and Type
// are snapshots. ReadFloat32 uses the original validated layout.
type Tensor struct {
	Shape  []uint64
	Type   Type
	r      io.ReaderAt
	offset int64
	record tensorRecord
}

// ReadFloat32 decodes a tensor into new, caller-owned storage.
func (t Tensor) ReadFloat32() ([]float32, error) {
	if t.r == nil {
		return nil, fmt.Errorf("gguf: invalid tensor handle")
	}
	typ := t.record.typ
	b := typ.block()
	out := make([]float32, int(t.record.elem))
	const chunkBytes = 64 << 10
	chunk := chunkBytes / b.bytes * b.bytes
	raw := make([]byte, min(uint64(chunk), t.record.size))
	for off, start := uint64(0), 0; off < t.record.size; {
		buf := raw[:min(uint64(len(raw)), t.record.size-off)]
		if n, err := t.r.ReadAt(buf, t.offset+int64(off)); n != len(buf) || err != nil && err != io.EOF {
			if err == nil || err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return nil, fmt.Errorf("gguf: read tensor at byte %d: %w", t.offset+int64(off), err)
		}
		values := len(buf) / b.bytes * b.values
		decode(typ, out[start:start+values], buf)
		off += uint64(len(buf))
		start += values
	}
	return out, nil
}

func decode(typ Type, out []float32, raw []byte) {
	switch typ {
	case F32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case BF16:
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*i:])) << 16)
		}
	case Q8_0:
		for i := 0; i < len(out); i += 32 {
			scale := f16.ToF32(binary.LittleEndian.Uint16(raw))
			for j := range 32 {
				out[i+j] = scale * float32(int8(raw[2+j]))
			}
			raw = raw[34:]
		}
	case Q6_K:
		for i := 0; i < len(out); i += 256 {
			decodeQ6K(out[i:i+256], raw[:210])
			raw = raw[210:]
		}
	}
}

func decodeQ6K(out []float32, raw []byte) {
	scale := f16.ToF32(binary.LittleEndian.Uint16(raw[208:]))
	// Each half block interleaves four groups of 32 values in the bit planes.
	for half := range 2 {
		low, high := raw[half*64:], raw[128+half*32:]
		for group := range 4 {
			ls, hs := uint(group/2*4), uint(group*2)
			groupLow := low[group%2*32:]
			for lane := range 32 {
				q := (groupLow[lane]>>ls)&15 | ((high[lane]>>hs)&3)<<4
				i := half*128 + group*32 + lane
				out[i] = scale * float32(int8(raw[192+i/16])) * float32(int(q)-32)
			}
		}
	}
}
