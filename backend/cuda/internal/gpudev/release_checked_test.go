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
