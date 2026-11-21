//go:build linux

package gpudev

import (
	"fmt"
	"math/bits"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// Copy2DAsync enqueues a strided device-to-device copy on this queue's stream (the default stream when the queue has
// none, as Launch submits there): rows rows of width bytes, row r read from src at its bind offset plus r*srcPitch and
// written to dst at its bind offset plus r*dstPitch (cuMemcpy2DAsync). It returns without synchronizing; like a launch,
// a kernel queued after it on this queue sees the bytes. It is how a caller gathers column blocks of one row-major
// matrix into another without a kernel.
//
// The source and destination ranges must not overlap. Both are checked against their buffer's own bounds (a
// sub-allocation's [base, lim), a whole allocation's size) before the driver is called.
func (q Queue) Copy2DAsync(dst Buffer, dstPitch int, src Buffer, srcPitch int, width, rows int) error {
	if src.raw != 0 || dst.raw != 0 {
		return fmt.Errorf("cuda: Copy2DAsync with a mapped-host buffer, whose bytes live in host memory")
	}
	if dst.b == nil || src.b == nil {
		return fmt.Errorf("cuda: Copy2DAsync with a nil buffer")
	}
	if src.cx != nil && dst.cx != nil && src.cx != dst.cx {
		return fmt.Errorf("cuda: Copy2DAsync between buffers on different contexts")
	}
	if width < 0 || rows < 0 {
		return fmt.Errorf("cuda: Copy2DAsync of %d rows of %d bytes", rows, width)
	}
	if width == 0 || rows == 0 {
		return nil
	}
	if dstPitch < width || srcPitch < width {
		return fmt.Errorf("cuda: Copy2DAsync of %d-byte rows with pitches %d (dst) and %d (src)", width, dstPitch, srcPitch)
	}
	if err := copy2DRange("destination", dst, dstPitch, width, rows); err != nil {
		return err
	}
	if err := copy2DRange("source", src, srcPitch, width, rows); err != nil {
		return err
	}
	dptr := dst.b.DevicePtr() + cudasys.CUdeviceptr(dst.off)
	sptr := src.b.DevicePtr() + cudasys.CUdeviceptr(src.off)
	if TraceForTest != nil {
		TraceForTest(TraceEvent{Op: TraceCopy2D, Dst: uint64(dptr), SrcPtr: uint64(sptr), Bytes: width * rows})
	}
	desc := cudasys.Memcpy2D{
		SrcMemoryType: cudasys.MemoryTypeDevice, SrcDevice: sptr, SrcPitch: uint64(srcPitch),
		DstMemoryType: cudasys.MemoryTypeDevice, DstDevice: dptr, DstPitch: uint64(dstPitch),
		WidthInBytes: uint64(width), Height: uint64(rows),
	}
	var stream cudasys.CUstream
	if q.s != nil {
		stream = q.s.Raw()
	}
	return q.d.onThread(func(drv *cudasys.Driver) error {
		return cudaresult.Memcpy2DAsync(drv, &desc, stream)
	})
}

// copy2DRange checks that rows rows of width bytes at pitch, from b's bind offset, lie inside b: the offset at or past
// the buffer's base (At takes a negative offset) and the last byte before its end, with no overflow on the way.
func copy2DRange(what string, b Buffer, pitch, width, rows int) error {
	if b.off < b.base {
		return fmt.Errorf("cuda: Copy2DAsync %s offset %d is before its buffer's start %d", what, b.off, b.base)
	}
	hi, lo := bits.Mul64(uint64(rows-1), uint64(pitch))
	end, carry := bits.Add64(lo, uint64(width), 0)
	end, carry2 := bits.Add64(end, uint64(b.off), 0)
	if hi != 0 || carry != 0 || carry2 != 0 || end > b.bytes() {
		return fmt.Errorf("cuda: Copy2DAsync of %d rows at pitch %d from offset %d overruns a %d-byte %s", rows, pitch, b.off, b.bytes(), what)
	}
	return nil
}
