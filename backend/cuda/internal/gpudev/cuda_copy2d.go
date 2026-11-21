//go:build linux

package gpudev

import (
	"fmt"

	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// Copy2DAsync enqueues a strided device-to-device copy on this queue's stream: rows rows of width bytes, row r read from
// src at its bind offset plus r*srcPitch and written to dst at its bind offset plus r*dstPitch (cuMemcpy2DAsync). It
// returns without synchronizing; like a launch, a kernel queued after it on this queue sees the bytes. It is how a
// caller gathers column blocks of one row-major matrix into another without a kernel.
//
// The source and destination ranges must not overlap. Both are bounds-checked before the driver is called.
func (q Queue) Copy2DAsync(dst Buffer, dstPitch int, src Buffer, srcPitch int, width, rows int) error {
	if dst.b == nil || src.b == nil {
		return fmt.Errorf("cuda: Copy2DAsync with a nil or mapped-host buffer")
	}
	if q.s == nil {
		return fmt.Errorf("cuda: Copy2DAsync on a queue with no stream")
	}
	if width <= 0 || rows <= 0 {
		return nil
	}
	if dstPitch < width || srcPitch < width {
		return fmt.Errorf("cuda: Copy2DAsync of %d-byte rows with pitches %d (dst) and %d (src)", width, dstPitch, srcPitch)
	}
	span := func(pitch int) uint64 { return uint64(rows-1)*uint64(pitch) + uint64(width) }
	if got, want := dst.bytes(), uint64(dst.off)+span(dstPitch); got < want {
		return fmt.Errorf("cuda: Copy2DAsync of %d rows at pitch %d from offset %d overruns a %d-byte destination", rows, dstPitch, dst.off, got)
	}
	if got, want := src.bytes(), uint64(src.off)+span(srcPitch); got < want {
		return fmt.Errorf("cuda: Copy2DAsync of %d rows at pitch %d from offset %d overruns a %d-byte source", rows, srcPitch, src.off, got)
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
	return q.d.onThread(func(drv *cudasys.Driver) error {
		return cudaresult.Memcpy2DAsync(drv, &desc, q.s.Raw())
	})
}
