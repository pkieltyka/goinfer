package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"unicode/utf8"
)

const (
	maxStringBytes = 1 << 20
	maxArrayItems  = 1 << 20
	maxArrayDepth  = 8
)

// cursor bounds header reads and retained metadata before allocating.
type cursor struct {
	r       *bufio.Reader
	left    uint64
	used    uint64
	scratch []byte
	err     error
}

func (c *cursor) fail(format string, args ...any) {
	if c.err == nil {
		c.err = fmt.Errorf(format, args...)
	}
}

func (c *cursor) fits(n uint64) bool {
	if c.err != nil {
		return false
	}
	if n > c.left {
		c.fail("index truncated or exceeds %d bytes", maxIndexBytes)
		return false
	}
	return true
}

func (c *cursor) reserve(n uint64) bool {
	if c.err != nil {
		return false
	}
	if n > maxIndexBytes-c.used {
		c.fail("metadata exceeds %d retained bytes", maxIndexBytes)
		return false
	}
	c.used += n
	return true
}

func (c *cursor) Read(p []byte) (int, error) {
	if !c.fits(uint64(len(p))) {
		return 0, c.err
	}
	n, err := c.r.Read(p)
	c.left -= uint64(n)
	return n, err
}

func (c *cursor) take(p []byte) {
	if c.err == nil {
		_, c.err = io.ReadFull(c, p)
	}
}

func (c *cursor) skip(n uint64) {
	if c.fits(n) {
		dropped, err := c.r.Discard(int(n))
		c.left -= uint64(dropped)
		c.err = err
	}
}

func (c *cursor) u32() uint32 {
	var b [4]byte
	c.take(b[:])
	return binary.LittleEndian.Uint32(b[:])
}

func (c *cursor) u64() uint64 {
	var b [8]byte
	c.take(b[:])
	return binary.LittleEndian.Uint64(b[:])
}

func (c *cursor) readString(limit uint64, keep bool) string {
	n := c.u64()
	if n > limit {
		c.fail("string length %d exceeds limit %d", n, limit)
	}
	if !c.fits(n) || keep && !c.reserve(n) {
		return ""
	}
	if int(n) > cap(c.scratch) {
		c.scratch = make([]byte, n)
	}
	p := c.scratch[:int(n)]
	c.take(p)
	if c.err != nil {
		return ""
	}
	if !utf8.Valid(p) {
		c.fail("string is not valid UTF-8")
		return ""
	}
	if keep {
		return string(p)
	}
	return ""
}

type valueType uint32

const (
	vU8 valueType = iota
	vI8
	vU16
	vI16
	vU32
	vI32
	vF32
	vBool
	vString
	vArray
	vU64
	vI64
	vF64
)

func (k valueType) minWidth() uint64 {
	switch k {
	case vU8, vI8, vBool:
		return 1
	case vU16, vI16:
		return 2
	case vU32, vI32, vF32:
		return 4
	case vU64, vI64, vF64, vString:
		return 8
	case vArray:
		return 12
	default:
		return 0
	}
}

func (c *cursor) array(keep bool, depth int) any {
	if depth >= maxArrayDepth {
		c.fail("metadata array nesting exceeds %d", maxArrayDepth)
		return nil
	}
	kind, n := valueType(c.u32()), c.u64()
	width := kind.minWidth()
	if width == 0 {
		c.fail("unsupported metadata array element type %d", kind)
		return nil
	}
	if n > maxArrayItems || n > c.left/width {
		c.fail("metadata array length %d exceeds limits or remaining index", n)
		return nil
	}
	return c.value(kind, int(n), true, keep, depth+1)
}

func (c *cursor) value(k valueType, n int, array, keep bool, depth int) any {
	switch k {
	case vU8:
		return readNumbers[uint8](c, n, k, array, keep)
	case vI8:
		return readNumbers[int8](c, n, k, array, keep)
	case vU16:
		return readNumbers[uint16](c, n, k, array, keep)
	case vI16:
		return readNumbers[int16](c, n, k, array, keep)
	case vU32:
		return readNumbers[uint32](c, n, k, array, keep)
	case vI32:
		return readNumbers[int32](c, n, k, array, keep)
	case vF32:
		return readNumbers[float32](c, n, k, array, keep)
	case vBool:
		return readValues(c, n, 1, array, keep, c.boolean)
	case vString:
		return readValues(c, n, 16, array, keep, func() string { return c.readString(maxStringBytes, keep) })
	case vArray:
		return readValues(c, n, 16, array, keep, func() any { return c.array(keep, depth) })
	case vU64:
		return readNumbers[uint64](c, n, k, array, keep)
	case vI64:
		return readNumbers[int64](c, n, k, array, keep)
	case vF64:
		return readNumbers[float64](c, n, k, array, keep)
	default:
		c.fail("unsupported metadata type %d", k)
		return nil
	}
}

func (c *cursor) boolean() bool {
	var b [1]byte
	c.take(b[:])
	if b[0] > 1 {
		c.fail("invalid bool value %d", b[0])
	}
	return b[0] == 1
}

func readNumbers[T any](c *cursor, n int, kind valueType, array, keep bool) any {
	bytes := uint64(n) * kind.minWidth()
	if !c.fits(bytes) {
		return nil
	}
	if !keep {
		c.skip(bytes)
		return nil
	}
	if !array {
		var v T
		c.err = binary.Read(c, binary.LittleEndian, &v)
		return v
	}
	if !c.reserve(bytes) {
		return nil
	}
	v := make([]T, n)
	c.err = binary.Read(c, binary.LittleEndian, v)
	return v
}

func readValues[T any](c *cursor, n int, width uint64, array, keep bool, one func() T) any {
	if !array {
		return one()
	}
	var v []T
	if keep {
		if !c.reserve(uint64(n) * width) {
			return nil
		}
		v = make([]T, n)
	}
	for i := 0; i < n && c.err == nil; i++ {
		item := one()
		if keep {
			v[i] = item
		}
	}
	if !keep {
		return nil
	}
	return v
}
