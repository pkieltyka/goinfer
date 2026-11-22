//go:build cuda

package gpudev

import (
	"errors"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

type recordingLease struct {
	live    int64
	charges int
	frees   int
}

func (l *recordingLease) Lease(onDevice bool, line string, requested, charged int64) func() {
	l.live += charged
	l.charges++
	return func() {
		l.live -= charged
		l.frees++
	}
}

func TestReleaseBufChecked_PreservesLedgerOnFailedClose(t *testing.T) {
	lease := &recordingLease{}
	d := &Device{leaser: lease}

	raw := new(gc.Buffer[uint8])
	rel := d.lease(true, "test", 4096, 4096)
	d.allocs = append(d.allocs, devAlloc{b: raw, release: rel})
	buf := Buffer{b: raw, d: d, n: 1024}

	closeErr := errors.New("cuda: synthetic close failure")
	d.SetCloseBufHookForTest(func(func() error) error { return closeErr })

	if err := d.ReleaseBufChecked(buf); !errors.Is(err, closeErr) {
		t.Fatalf("ReleaseBufChecked on failing close = %v, want %v", err, closeErr)
	}
	if allocs, _ := d.LedgerLen(); allocs != 1 {
		t.Fatalf("LedgerLen after failed ReleaseBufChecked = %d, want 1", allocs)
	}
	if lease.live != 4096 || lease.frees != 0 {
		t.Fatalf("lease after failed ReleaseBufChecked = (live=%d, frees=%d), want (4096, 0)", lease.live, lease.frees)
	}

	// Once native Close succeeds, ledger ownership and lease charge are released.
	d.SetCloseBufHookForTest(func(func() error) error { return nil })
	if err := d.ReleaseBufChecked(buf); err != nil {
		t.Fatalf("ReleaseBufChecked retry = %v, want nil", err)
	}
	if allocs, _ := d.LedgerLen(); allocs != 0 {
		t.Fatalf("LedgerLen after successful ReleaseBufChecked = %d, want 0", allocs)
	}
	if lease.live != 0 || lease.frees != 1 {
		t.Fatalf("lease after successful ReleaseBufChecked = (live=%d, frees=%d), want (0, 1)", lease.live, lease.frees)
	}

	// Zero buffer and sub-allocation are no-ops.
	if err := d.ReleaseBufChecked(Buffer{}); err != nil {
		t.Fatalf("zero buffer ReleaseBufChecked = %v", err)
	}
	if err := d.ReleaseBufChecked(Buffer{b: raw, lim: 16}); err != nil {
		t.Fatalf("sub-allocation ReleaseBufChecked = %v", err)
	}
}

func TestReleaseAllChecked_PreservesFailedBuffersAndRetries(t *testing.T) {
	lease := &recordingLease{}
	d := &Device{leaser: lease}

	raw1 := new(gc.Buffer[uint8])
	raw2 := new(gc.Buffer[uint8])
	d.allocs = append(d.allocs,
		devAlloc{b: raw1, release: d.lease(true, "a", 2048, 2048)},
		devAlloc{b: raw2, release: d.lease(true, "b", 4096, 4096)},
	)

	closeErr := errors.New("cuda: synthetic repeat close failure")
	calls := 0
	d.SetCloseBufHookForTest(func(func() error) error {
		calls++
		if calls == 2 {
			return closeErr
		}
		return nil
	})
	if err := d.ReleaseAllChecked(); !errors.Is(err, closeErr) {
		t.Fatalf("first ReleaseAllChecked = %v, want %v", err, closeErr)
	}
	if allocs, _ := d.LedgerLen(); allocs != 1 {
		t.Fatalf("LedgerLen after partial ReleaseAllChecked = %d, want 1", allocs)
	}
	if lease.live != 4096 || lease.frees != 1 {
		t.Fatalf("lease after partial ReleaseAllChecked = (live=%d, frees=%d), want (4096, 1)", lease.live, lease.frees)
	}

	// Retry once native close succeeds frees only the remaining buffer cleanly.
	d.SetCloseBufHookForTest(func(func() error) error { return nil })
	if err := d.ReleaseAllChecked(); err != nil {
		t.Fatalf("second ReleaseAllChecked = %v, want nil", err)
	}
	if allocs, _ := d.LedgerLen(); allocs != 0 {
		t.Fatalf("LedgerLen after second ReleaseAllChecked = %d, want 0", allocs)
	}
	if lease.live != 0 || lease.frees != 2 {
		t.Fatalf("lease after second ReleaseAllChecked = (live=%d, frees=%d), want (0, 2)", lease.live, lease.frees)
	}
}

type fakeCloser struct {
	fail bool
	err  error
}

func (f *fakeCloser) Close() error {
	if f.fail {
		return f.err
	}
	return nil
}

func TestReleaseObjectsChecked_PreservesFailedObjectsAndRetries(t *testing.T) {
	lease := &recordingLease{}
	d := &Device{leaser: lease, hosts: map[closer]func(){}}

	raw := new(gc.Buffer[uint8])
	d.allocs = append(d.allocs, devAlloc{b: raw, release: d.lease(true, "buf", 4096, 4096)})

	objErr := errors.New("cuda: synthetic object close failure")
	obj := &fakeCloser{fail: true, err: objErr}
	d.objs = append(d.objs, obj)
	d.hosts[obj] = d.lease(false, "host", 1024, 1024)

	bufErr := errors.New("cuda: synthetic buffer close failure")
	d.SetCloseBufHookForTest(func(func() error) error { return bufErr })

	// First attempt fails on buffer close: both buffer and object stay owned.
	if err := d.ReleaseObjectsChecked(); !errors.Is(err, bufErr) {
		t.Fatalf("first ReleaseObjectsChecked = %v, want %v", err, bufErr)
	}
	if allocs, objs := d.LedgerLen(); allocs != 1 || objs != 1 {
		t.Fatalf("LedgerLen after buffer failure = (%d, %d), want (1, 1)", allocs, objs)
	}
	if lease.live != 5120 {
		t.Fatalf("lease.live after buffer failure = %d, want 5120", lease.live)
	}

	// Second attempt succeeds on buffer close, fails on object close: buffer released, object + host lease retained.
	d.SetCloseBufHookForTest(func(func() error) error { return nil })
	if err := d.ReleaseObjectsChecked(); !errors.Is(err, objErr) {
		t.Fatalf("second ReleaseObjectsChecked = %v, want %v", err, objErr)
	}
	if allocs, objs := d.LedgerLen(); allocs != 0 || objs != 1 {
		t.Fatalf("LedgerLen after object failure = (%d, %d), want (0, 1)", allocs, objs)
	}
	if lease.live != 1024 {
		t.Fatalf("lease.live after object failure = %d, want 1024", lease.live)
	}

	// Third attempt succeeds on object close: ledger and lease drain cleanly to zero.
	obj.fail = false
	if err := d.ReleaseObjectsChecked(); err != nil {
		t.Fatalf("third ReleaseObjectsChecked = %v, want nil", err)
	}
	if allocs, objs := d.LedgerLen(); allocs != 0 || objs != 0 {
		t.Fatalf("LedgerLen after recovery = (%d, %d), want (0, 0)", allocs, objs)
	}
	if lease.live != 0 || lease.frees != 2 {
		t.Fatalf("lease after recovery = (live=%d, frees=%d), want (0, 2)", lease.live, lease.frees)
	}
}
