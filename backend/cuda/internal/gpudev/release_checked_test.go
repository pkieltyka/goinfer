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
