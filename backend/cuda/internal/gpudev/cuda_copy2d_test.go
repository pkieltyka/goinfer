//go:build linux

package gpudev

import "testing"

// TestCopy2DAsync: a strided copy gathers a column block of one row-major matrix into a column block of another,
// leaves every other byte of the destination alone, orders before a later Sync, runs on a queue with no stream of its
// own (the default stream), and refuses a pitch narrower than its width, a range past either buffer's end and an
// offset before a sub-allocation's start before calling the driver.
func TestCopy2DAsync(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()
	q := d.NewCommandQueue()
	const rows, srcCols, dstCols = 5, 6, 9
	src := make([]float32, rows*srcCols)
	for i := range src {
		src[i] = float32(i + 1)
	}
	dst := make([]float32, rows*dstCols)
	for i := range dst {
		dst[i] = -1
	}
	sb, db := NewBufferOf(d, src), NewBufferOf(d, dst)
	// src columns 2..4 into dst columns 5..7
	if err := q.Copy2DAsync(db.At(5*4), dstCols*4, sb.At(2*4), srcCols*4, 3*4, rows); err != nil {
		t.Fatal(err)
	}
	if err := q.Sync(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, rows*dstCols)
	if err := Download(db, got); err != nil {
		t.Fatal(err)
	}
	for r := range rows {
		for c := range dstCols {
			want := float32(-1)
			if c >= 5 && c < 8 {
				want = src[r*srcCols+2+c-5]
			}
			if got[r*dstCols+c] != want {
				t.Fatalf("dst[%d][%d] = %v, want %v", r, c, got[r*dstCols+c], want)
			}
		}
	}
	// The default stream: a queue whose stream could not be created submits there, launches and copies alike.
	null := Queue{d: d}
	if err := Upload(db, dst); err != nil {
		t.Fatal(err)
	}
	if err := null.Copy2DAsync(db.At(5*4), dstCols*4, sb.At(2*4), srcCols*4, 3*4, rows); err != nil {
		t.Fatalf("on the default stream: %v", err)
	}
	if err := null.Sync(); err != nil {
		t.Fatal(err)
	}
	again := make([]float32, rows*dstCols)
	if err := Download(db, again); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if again[i] != got[i] {
			t.Fatalf("on the default stream: dst element %d = %v, want %v", i, again[i], got[i])
		}
	}
	whole := NewBufferOf(d, make([]float32, 16))
	sub := whole.Sub(16, 16, 4) // floats 4..7
	for _, bad := range []struct {
		name string
		err  error
	}{
		{"source before a sub-allocation's start", q.Copy2DAsync(db, dstCols*4, sub.At(-4), 4, 4, 1)},
		{"destination before a sub-allocation's start", q.Copy2DAsync(sub.At(-4), 4, sb, srcCols*4, 4, 1)},
		{"past a sub-allocation's end", q.Copy2DAsync(db, dstCols*4, sub.At(4), 4, 16, 1)},
		{"pitch overflow", q.Copy2DAsync(db, dstCols*4, sb, 1<<62, 4, 8)},
		{"pitch under width", q.Copy2DAsync(db, 2*4, sb, srcCols*4, 3*4, rows)},
		{"past the destination", q.Copy2DAsync(db.At(7*4), dstCols*4, sb, srcCols*4, 3*4, rows)},
		{"past the source", q.Copy2DAsync(db, dstCols*4, sb.At(4*4), srcCols*4, 3*4, rows)},
	} {
		if bad.err == nil {
			t.Errorf("%s: accepted", bad.name)
		}
	}
	// The boundary itself: a copy that ends on a sub-allocation's last byte, from its first, is accepted.
	if err := q.Copy2DAsync(db, dstCols*4, sub, 16, 16, 1); err != nil {
		t.Errorf("an exact fit of a sub-allocation was refused: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatal(err)
	}
}
