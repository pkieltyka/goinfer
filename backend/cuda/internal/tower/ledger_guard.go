//go:build cuda && goinfer_testhooks

package tower

import (
	"fmt"
	"io"
	"runtime"
	"sort"
	"strings"
	"sync"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

// TowerLedger returns l so test wrappers embedding *Ledger or Ledger satisfy the same lookup.
func (l *Ledger) TowerLedger() *Ledger { return l }

type ledgerProvider interface {
	TowerLedger() *Ledger
}

// LedgerOf is the mem.Ledger attached to d, or nil when d has no tower.Ledger.
func LedgerOf(d *gpu.Device) *mem.Ledger {
	if p, ok := d.Leaser().(ledgerProvider); ok {
		if l := p.TowerLedger(); l != nil {
			return &l.Ledger
		}
	}
	return nil
}

// LedgerBreak says how a released device's ledger breaks I1 (an unplanned byte ever) or I2 (a lease open after release),
// or "". A device with no ledger breaks I1: nothing it allocated was leased.
func LedgerBreak(l *mem.Ledger) string {
	if l == nil {
		return "a device with no ledger: its allocations were not leased"
	}
	u := l.Usage()
	if u.PeakUnplannedCharged > 0 || u.Leases > 0 {
		return fmt.Sprintf("unplanned peak %d bytes; %d leases (%d bytes) open after release, by line %v",
			u.PeakUnplannedCharged, u.Leases, u.Charged, u.ByLine)
	}
	return ""
}

// LedgerGuard enforces MM3 ledger invariants (I1 nothing unplanned, I2 close to zero) and open-device tracking across
// a test binary's TestMain.
type LedgerGuard struct {
	output   io.Writer
	openMu   sync.Mutex
	openDevs map[gpu.Leaser]string

	unplannedMu    sync.Mutex
	unplannedSites map[*Ledger]map[string]int64

	releasedMu   sync.Mutex
	releasedSeen int
	releasedBad  []string
	expected     map[*mem.Ledger]func(mem.Usage, int64) bool
}

// InstallLedgerGuard routes unlined test buffers onto mem.LineTest, tracks every device opened through gpu.NewLeaser,
// and records unplanned allocations and release-time ledger checks. Call from TestMain after any process-long helper
// device (such as a VRAM sampler) is created.
func InstallLedgerGuard(output io.Writer) *LedgerGuard {
	g := &LedgerGuard{
		output:         output,
		openDevs:       map[gpu.Leaser]string{},
		unplannedSites: map[*Ledger]map[string]int64{},
		expected:       map[*mem.Ledger]func(mem.Usage, int64) bool{},
	}
	gpu.UnlinedLine = mem.LineTest
	gpu.ReleasedForTest = g.recordReleased
	gpu.NewLeaser = g.trackOpen(gpu.NewLeaser)
	UnplannedForTest = g.recordUnplanned
	return g
}

// MarkReleased drops a released device from the open set. recordReleased calls it; a test that swaps gpu.ReleasedForTest
// must call it from its own hook, or its devices read as open.
func (g *LedgerGuard) MarkReleased(d *gpu.Device) {
	g.openMu.Lock()
	delete(g.openDevs, d.Leaser())
	g.openMu.Unlock()
}

// OpenDevices names the test that created each device still open, sorted.
func (g *LedgerGuard) OpenDevices() []string {
	g.openMu.Lock()
	defer g.openMu.Unlock()
	out := make([]string, 0, len(g.openDevs))
	for _, name := range g.openDevs {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// ExpectPlantedRelease applies check once, to this device's ledger only.
func (g *LedgerGuard) ExpectPlantedRelease(d *gpu.Device, check func(mem.Usage, int64) bool) {
	g.releasedMu.Lock()
	defer g.releasedMu.Unlock()
	if l := LedgerOf(d); l != nil {
		g.expected[l] = check
	}
}

// Breaks logs the count of released devices and returns every release-time ledger invariant failure recorded so far.
func (g *LedgerGuard) Breaks() []string {
	g.releasedMu.Lock()
	defer g.releasedMu.Unlock()
	fmt.Fprintf(g.output, "ledger: %d device(s) released in this run\n", g.releasedSeen)
	return g.releasedBad
}

// Check prints any ledger invariant failures or unreleased devices to stderr and returns 1 when any occurred, else code.
func (g *LedgerGuard) Check(code int) int {
	if bad := g.Breaks(); len(bad) > 0 {
		fmt.Fprintf(g.output, "FAIL: %d released device(s) broke the ledger's invariants (I1 nothing unplanned, I2 close to zero):\n", len(bad))
		for _, b := range bad {
			fmt.Fprintln(g.output, "  "+b)
		}
		code = 1
	}
	if open := g.OpenDevices(); len(open) > 0 {
		fmt.Fprintf(g.output, "FAIL: %d device(s) never released in this run, by the test that created each: %s\n", len(open),
			strings.Join(open, ", "))
		code = 1
	}
	return code
}

func (g *LedgerGuard) trackOpen(next func() gpu.Leaser) func() gpu.Leaser {
	return func() gpu.Leaser {
		l := next()
		g.openMu.Lock()
		g.openDevs[l] = creatingTest()
		g.openMu.Unlock()
		return l
	}
}

func (g *LedgerGuard) recordUnplanned(l *Ledger, line string, charged int64) {
	site := fmt.Sprintf("line %q at %s", line, allocSite())
	g.unplannedMu.Lock()
	defer g.unplannedMu.Unlock()
	if g.unplannedSites[l] == nil {
		g.unplannedSites[l] = map[string]int64{}
	}
	g.unplannedSites[l][site] += charged
}

func (g *LedgerGuard) takeUnplannedSites(d *gpu.Device) (string, int64) {
	var l *Ledger
	if p, ok := d.Leaser().(ledgerProvider); ok {
		l = p.TowerLedger()
	}
	g.unplannedMu.Lock()
	sites := g.unplannedSites[l]
	delete(g.unplannedSites, l)
	g.unplannedMu.Unlock()
	keys := make([]string, 0, len(sites))
	for k := range sites {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return sites[keys[i]] > sites[keys[j]] })
	var b strings.Builder
	var total int64
	for _, k := range keys {
		total += sites[k]
		fmt.Fprintf(&b, "\n      %d bytes leased in all, %s", sites[k], k)
	}
	return b.String(), total
}

func (g *LedgerGuard) recordReleased(d *gpu.Device) {
	g.openMu.Lock()
	creator := g.openDevs[d.Leaser()]
	g.openMu.Unlock()
	g.MarkReleased(d)
	l := LedgerOf(d)
	b := LedgerBreak(l)
	sites, unplanned := g.takeUnplannedSites(d)
	g.releasedMu.Lock()
	defer g.releasedMu.Unlock()
	g.releasedSeen++
	expected := g.expected[l]
	delete(g.expected, l)
	if b == "" {
		return
	}
	b = fmt.Sprintf("%s (created by %s, released while in flight: %s)%s", b, creator, strings.Join(testsInFlight(), " "), sites)
	if expected != nil && expected(l.Usage(), unplanned) {
		fmt.Fprintln(g.output, "LEDGER PLANT [expected]: "+b)
		return
	}
	fmt.Fprintln(g.output, "LEDGER BREAK: "+b)
	g.releasedBad = append(g.releasedBad, b)
}

// allocSite is the first three frames above the ledger's hook, skipping gpudev, mem, Ledger and LedgerGuard.
func allocSite() string {
	pc := make([]uintptr, 32)
	frames := runtime.CallersFrames(pc[:runtime.Callers(3, pc)])
	var out []string
	for len(out) < 3 {
		f, more := frames.Next()
		fn := f.Function[strings.LastIndex(f.Function, "/")+1:]
		if !strings.HasPrefix(fn, "gpudev.") && !strings.HasPrefix(fn, "mem.") &&
			!strings.HasPrefix(fn, "tower.(*Ledger)") && !strings.HasPrefix(fn, "tower.(*LedgerGuard)") &&
			!strings.HasSuffix(fn, ").Lease") {
			out = append(out, fmt.Sprintf("%s (%s:%d)", fn, f.File[strings.LastIndex(f.File, "/")+1:], f.Line))
		}
		if !more {
			break
		}
	}
	return strings.Join(out, " < ")
}

// creatingTest is the outermost Test or Benchmark function on the caller's stack. A device created off a test goroutine (a
// resident or tower build creates its device on the executor thread) is named by the tests whose goroutines are in flight.
func creatingTest() string {
	pc := make([]uintptr, 64)
	frames := runtime.CallersFrames(pc[:runtime.Callers(2, pc)])
	name := ""
	for {
		f, more := frames.Next()
		if top := testOf(f.Function); top != "" {
			name = top
		}
		if !more {
			break
		}
	}
	if name != "" {
		return name
	}
	return "(off a test goroutine, while in flight: " + strings.Join(testsInFlight(), " ") + ")"
}

// testOf is the top-level Test or Benchmark name of a qualified function name, or "".
func testOf(fn string) string {
	fn = fn[strings.LastIndex(fn, "/")+1:]
	if _, after, ok := strings.Cut(fn, "."); ok {
		if top, _, _ := strings.Cut(after, "."); strings.HasPrefix(top, "Test") || strings.HasPrefix(top, "Benchmark") {
			return top
		}
	}
	return ""
}

// testsInFlight is every Test or Benchmark function on any goroutine's stack, sorted.
func testsInFlight() []string {
	buf := make([]byte, 1<<20)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			buf = buf[:n]
			break
		}
		buf = make([]byte, 2*len(buf))
	}
	seen := map[string]bool{}
	for _, line := range strings.Split(string(buf), "\n") {
		if line == "" || line[0] == '\t' || strings.HasPrefix(line, "goroutine ") || strings.HasPrefix(line, "created by ") {
			continue
		}
		fn, _, _ := strings.Cut(line, "(")
		if top := testOf(fn); top != "" {
			seen[top] = true
		}
	}
	out := make([]string, 0, len(seen))
	for t := range seen {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}
