//go:build cuda

package tower

import (
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

// Every device created in a process that links this package leases its allocations to a mem.Ledger, whichever package
// creates it.
func init() { gpu.NewLeaser = func() gpu.Leaser { return &Ledger{} } }

// Ledger is a device's mem.Ledger: device buffers on VRAM, pinned host memory on HostAnon.
type Ledger struct{ mem.Ledger }

// UnplannedForTest is called when non-nil with each lease on no plan line and the ledger it lands on, on the allocating
// goroutine, so a test can name the call site of an unplanned byte. A test seam, nil in production.
var UnplannedForTest func(l *Ledger, line string, charged int64)

// LeaseSeenForTest is told when non-nil of every lease (the ledger's own release func comes back wrapped), on the allocating goroutine:
// a test records the order of a build's allocations and frees through it. A test seam, nil in production.
var LeaseSeenForTest func(onDevice bool, line string, requested int64, release func()) func()

func (l *Ledger) Lease(onDevice bool, line string, requested, charged int64) func() {
	tier := mem.HostAnon
	if onDevice {
		tier = mem.VRAM
	}
	if UnplannedForTest != nil && !mem.LineID(line).Known() {
		UnplannedForTest(l, line, charged)
	}
	release := l.Reserve(tier, mem.LineID(line), requested, charged).Release
	if LeaseSeenForTest != nil {
		return LeaseSeenForTest(onDevice, line, requested, release)
	}
	return release
}
