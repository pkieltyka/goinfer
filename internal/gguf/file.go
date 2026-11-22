// Package gguf reads little-endian GGUF v3 metadata and supported tensor formats.
package gguf

import (
	"bufio"
	"cmp"
	"fmt"
	"io"
	"slices"
	"strings"
)

const (
	defaultAlignment = 32
	maxIndexBytes    = 64 << 20
	maxEntries       = 1 << 16
	maxDimensions    = 4
	maxTensorName    = 64
	maxMetaKey       = 1<<16 - 1
	maxTensorValues  = 1 << 28 // 1 GiB of decoded float32 values
)

// File is a validated GGUF index backed by a caller-owned ReaderAt.
type File struct {
	r         io.ReaderAt
	dataStart uint64
	metadata  map[string]any
	tensors   map[string]tensorRecord
}

type tensorRecord struct {
	shape           [maxDimensions]uint64
	rank            uint32
	typ             Type
	off, size, elem uint64
}

type span struct {
	name       string
	first, end uint64
}

// Open validates the header and tensor extents within size. Keep r open and
// unchanged until all tensor reads have finished.
func Open(r io.ReaderAt, size int64) (*File, error) {
	if r == nil || size < 24 {
		return nil, fmt.Errorf("gguf: missing reader or truncated header (size %d)", size)
	}
	index := min(size, maxIndexBytes)
	c := cursor{r: bufio.NewReaderSize(io.NewSectionReader(r, 0, index), 32<<10), left: uint64(index)}
	var magic [4]byte
	c.take(magic[:])
	if c.err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", c.err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, fmt.Errorf("gguf: invalid magic %q", magic)
	}
	version, tensorCount, metaCount := c.u32(), c.u64(), c.u64()
	if c.err != nil {
		return nil, fmt.Errorf("gguf: read header: %w", c.err)
	}
	if version != 3 {
		return nil, fmt.Errorf("gguf: unsupported little-endian version %d", version)
	}
	if tensorCount > maxEntries || metaCount > maxEntries || tensorCount*33+metaCount*14 > c.left {
		return nil, fmt.Errorf("gguf: index counts exceed limits or file size (%d tensors, %d metadata)", tensorCount, metaCount)
	}

	f := &File{r: r, metadata: make(map[string]any, int(metaCount)), tensors: make(map[string]tensorRecord, int(tensorCount))}
	for i := uint64(0); i < metaCount; i++ {
		key := c.readString(maxMetaKey, true)
		if !validKey(key) {
			c.fail("invalid metadata key %q", key)
		}
		if c.err != nil {
			return nil, fmt.Errorf("gguf: metadata entry %d: %w", i, c.err)
		}
		if _, exists := f.metadata[key]; exists {
			return nil, fmt.Errorf("gguf: duplicate metadata key %q", key)
		}
		kind := valueType(c.u32())
		keep := kind != vArray || !strings.HasPrefix(key, "tokenizer.")
		f.metadata[key] = c.value(kind, 1, false, keep, 0)
		if c.err != nil {
			return nil, fmt.Errorf("gguf: metadata %q: %w", key, c.err)
		}
	}

	alignment := uint64(defaultAlignment)
	if v, ok := f.metadata["general.alignment"]; ok {
		a, ok := v.(uint32)
		if !ok || a == 0 || a%8 != 0 {
			return nil, fmt.Errorf("gguf: general.alignment must be a positive uint32 multiple of 8")
		}
		alignment = uint64(a)
	}
	if tensorCount == 0 {
		return f, nil
	}

	spans := make([]span, 0, int(tensorCount))
	for i := uint64(0); i < tensorCount; i++ {
		name, t := c.readTensor()
		if c.err != nil {
			return nil, fmt.Errorf("gguf: tensor %d (%q): %w", i, name, c.err)
		}
		if _, exists := f.tensors[name]; exists {
			return nil, fmt.Errorf("gguf: duplicate tensor %q", name)
		}
		if t.off%alignment != 0 {
			return nil, fmt.Errorf("gguf: misaligned tensor %q offset %d (alignment %d)", name, t.off, alignment)
		}
		if t.off > uint64(size) || t.size > uint64(size)-t.off {
			return nil, fmt.Errorf("gguf: tensor %q extends beyond file size", name)
		}
		f.tensors[name] = t
		spans = append(spans, span{name, t.off, t.off + t.size})
	}

	pos := uint64(index) - c.left
	f.dataStart = pos + (alignment-pos%alignment)%alignment
	if f.dataStart > uint64(size) {
		return nil, fmt.Errorf("gguf: truncated tensor data alignment padding")
	}
	slices.SortFunc(spans, func(a, b span) int { return cmp.Compare(a.first, b.first) })
	end := uint64(size) - f.dataStart
	for i, s := range spans {
		if s.end > end {
			return nil, fmt.Errorf("gguf: tensor %q extends beyond tensor data", s.name)
		}
		if i != 0 && s.first < spans[i-1].end {
			return nil, fmt.Errorf("gguf: overlapping tensors %q and %q", spans[i-1].name, s.name)
		}
	}
	return f, nil
}

// Meta returns the exact scalar type or a read-only slice. Absent keys and
// skipped tokenizer arrays return nil.
func (f *File) Meta(key string) any { return f.metadata[key] }

func (f *File) TensorCount() int { return len(f.tensors) }

func (f *File) Tensor(name string) (Tensor, bool) {
	t, ok := f.tensors[name]
	if !ok {
		return Tensor{}, false
	}
	return Tensor{
		Shape: slices.Clone(t.shape[:t.rank]), Type: t.typ,
		r: f.r, offset: int64(f.dataStart + t.off), record: t,
	}, true
}

func (c *cursor) readTensor() (string, tensorRecord) {
	name := c.readString(maxTensorName, true)
	t := tensorRecord{rank: c.u32(), elem: 1}
	if name == "" || strings.ContainsRune(name, 0) {
		c.fail("invalid tensor name")
	}
	if t.rank == 0 || t.rank > maxDimensions {
		c.fail("invalid rank %d", t.rank)
		return name, t
	}
	for i := uint32(0); i < t.rank; i++ {
		d := c.u64()
		if d == 0 || t.elem > maxTensorValues/d {
			c.fail("invalid or oversized tensor dimensions")
			return name, t
		}
		t.shape[i] = d
		t.elem *= d
	}
	t.typ, t.off = Type(c.u32()), c.u64()
	block := t.typ.block()
	if block.values == 0 {
		c.fail("unsupported tensor type %d", t.typ)
		return name, t
	}
	if t.shape[0]%uint64(block.values) != 0 {
		c.fail("row width %d is not a multiple of the block size %d", t.shape[0], block.values)
		return name, t
	}
	t.size = t.elem / uint64(block.values) * uint64(block.bytes)
	return name, t
}

func validKey(key string) bool {
	if key == "" {
		return false
	}
	empty := true
	for i := range len(key) {
		b := key[i]
		switch {
		case b == '.':
			if empty {
				return false
			}
			empty = true
		case b >= 'a' && b <= 'z' || b >= '0' && b <= '9' || b == '_' || b == '-':
			empty = false
		default:
			return false
		}
	}
	return !empty
}
