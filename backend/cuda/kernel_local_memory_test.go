//go:build cuda && goinfer_testhooks

package cuda

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"unsafe"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/launchgeom"
	"os"
	"path/filepath"
)

// pipeShim mirrors aikit's `type Pipeline struct{ f *gc.Function }` so a test can reach the driver
// handle. aikit exposes no accessor and the field is unexported in another module; the layout is a
// single pointer, so the cast is well-defined for as long as that holds. If aikit ever adds a field
// this stops compiling rather than reading garbage, because the shim is a distinct type.
type pipeShim struct{ f *gc.Function }

func rawFunc(p Pipeline) *gc.Function { return (*pipeShim)(unsafe.Pointer(&p)).f }

var ptxEntry = regexp.MustCompile(`\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_$]*)`)

var ptxSM = regexp.MustCompile(`(?m)^\s*\.target\s+sm_([0-9]+)(?:[ ,\t]|$)`)

type censusPTXModule struct {
	name  string
	ptx   []byte
	minSM int
}

func validCensusPTXTarget(name string, target int) bool {
	if name == "eg2_ffn_fp16.ptx" {
		return target == 10*eg2FP16MinMajor
	}
	return target > 0 && target <= 10*ptxTargetMajor+ptxTargetMinor
}

// ptxModules derives the census inputs and their minimum SM from production embeds.
func ptxModules(t *testing.T) []censusPTXModule {
	t.Helper()
	names, err := launchgeom.EmbeddedPTXModules(".")
	if err != nil {
		t.Fatal(err)
	}
	out := make([]censusPTXModule, 0, len(names))
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		match := ptxSM.FindSubmatch(b)
		if len(match) != 2 {
			t.Fatalf("%s has no numeric PTX SM target", name)
		}
		minSM, err := strconv.Atoi(string(match[1]))
		if err != nil {
			t.Fatalf("%s PTX SM target: %v", name, err)
		}
		if !validCensusPTXTarget(name, minSM) {
			t.Fatalf("%s PTX target sm_%d exceeds or disagrees with its supported architecture", name, minSM)
		}
		out = append(out, censusPTXModule{name: name, ptx: b, minSM: minSM})
	}
	return out
}

// TestKernelLocalMemoryCensus reports CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES for every entry point in
// every embedded module, and checks the backing-store multiplier against a measured reservation.
//
// Local memory per thread is a per-kernel compile-time property, so any kernel with per-thread
// arrays carries its own deferred reservation (moe_route's first launch reserves 138,412,032 B,
// TestMoERouteFirstLaunchReservation); this is the loop over all of them. It also settles the
// multiplier: "MOE_MAX_E 256 -> 512 doubled the cost" assumes the backing store is linear in
// per-thread bytes with a constant occupancy factor, an assumption about the driver that is checked
// here against multiProcessorCount x maxThreadsPerMultiProcessor rather than asserted.
func TestKernelLocalMemoryCensus(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer dev.ReleaseObjects()
	raw := dev.Context().Device()
	major, minor, err := raw.ComputeCapability()
	if err != nil {
		t.Fatalf("ComputeCapability: %v", err)
	}
	deviceSM := 10*major + minor
	sms, err := raw.Attribute(gc.DeviceAttributeMultiprocessorCount)
	if err != nil {
		t.Fatalf("MultiprocessorCount: %v", err)
	}
	maxT, err := raw.Attribute(gc.DeviceAttributeMaxThreadsPerMultiprocessor)
	if err != nil {
		t.Fatalf("MaxThreadsPerMultiprocessor: %v", err)
	}
	t.Logf("device: %d SMs x %d max threads/SM = %d resident threads", sms, maxT, sms*maxT)

	type row struct {
		mod, fn   string
		local     int
		predicted int64
	}
	var rows []row
	var total int64
	modules := ptxModules(t)
	var measured []string
	for _, m := range modules {
		if m.minSM > deviceSM {
			t.Logf("NOT APPLICABLE: %s requires SM%d, device is SM%d", m.name, m.minSM, deviceSM)
			continue
		}
		names := map[string]bool{}
		for _, mm := range ptxEntry.FindAllStringSubmatch(string(m.ptx), -1) {
			names[mm[1]] = true
		}
		if len(names) == 0 {
			t.Errorf("%s: no .visible .entry found — the census cannot be complete if a module "+
				"contributes no entries; either the regexp is wrong or the blob is empty", m.name)
			continue
		}
		mod, e := dev.CompileLibrary(m.ptx)
		if e != nil {
			t.Errorf("%s: CompileLibrary: %v", m.name, e)
			continue
		}
		measured = append(measured, m.name)
		sorted := make([]string, 0, len(names))
		for n := range names {
			sorted = append(sorted, n)
		}
		sort.Strings(sorted)
		for _, n := range sorted {
			p, e := dev.NewComputePipeline(mod, n)
			if e != nil {
				t.Errorf("%s/%s: NewComputePipeline: %v", m.name, n, e)
				continue
			}
			local, e := rawFunc(p).Attribute(gc.FuncAttrLocalSizeBytes)
			if e != nil {
				t.Errorf("%s/%s: LOCAL_SIZE_BYTES: %v", m.name, n, e)
				continue
			}
			pred := int64(local) * int64(sms) * int64(maxT)
			rows = append(rows, row{m.name, n, local, pred})
			total += pred
		}
	}

	sort.Slice(rows, func(i, j int) bool { return rows[i].local > rows[j].local })
	t.Logf("%-22s %-28s %10s %16s", "module", "kernel", "local/thr", "SMs*maxT*local")
	nonZero := 0
	for _, r := range rows {
		if r.local == 0 {
			continue
		}
		nonZero++
		t.Logf("%-22s %-28s %10d %16d  (%.1f MiB)", r.mod, r.fn, r.local, r.predicted,
			float64(r.predicted)/(1<<20))
	}
	t.Logf("%d entry points across %d modules; %d declare per-thread local memory",
		len(rows), len(measured), nonZero)
	t.Logf("EXAMINED: %d of %d production PTX modules — %s", len(measured), len(modules), strings.Join(measured, " "))
	t.Logf("SUM of per-kernel backing stores at full occupancy: %d B (%.1f MiB) — an UPPER BOUND on "+
		"the deferred cost, not a prediction: the driver may share or reuse a backing store across "+
		"kernels, and nothing here shows that it allocates them all simultaneously",
		total, float64(total)/(1<<20))

	// ---- PINNED: the per-kernel local footprint of the shipped PTX ----
	//
	// A gate that asserts only "a reservation exists" lets a future MOE_MAX_E change double a hidden
	// cost while still passing, so the BYTE FIGURES are pinned.
	//
	// This is the enumerated form, not a sample: every entry point with non-zero local memory is
	// listed, so a regeneration that gives any kernel per-thread scratch trips the gate even if
	// moe_route is untouched. LOCAL_SIZE_BYTES is read from the compiled shipped artifact, so it
	// tracks the PTX rather than the source constant — MOE_MAX_E cannot change without moving it.
	pinned := map[string]int{
		"moe_route":       4416, // two float[MOE_MAX_E] at MOE_MAX_E=512, plus the group scratch
		"rope_kv":         32,
		"rope_kv_batched": 32,
		// The ragged step's twin of rope_kv_batched (prefill_batched.cu): the same 28 B cosf/sinf slow-path scratch, so the same
		// 1.2 MiB at full occupancy, and no larger than the largest kernel already pinned.
		"rope_kv_ragged":        32,
		"route_gptoss":          4608, // gptoss_act.ptx; larger than moe_route, so backend.go forces it too
		"rope_kv_mrope_batched": 32,   // rope_mrope_prefill.ptx
		// rope_pairwise.ptx: the GPT-J pairwise twins; the same 28 B cosf/sinf slow-path scratch
		// (argument reduction) as their NeoX counterparts above.
		"rope_kv_pw":               32,
		"rope_kv_batched_pw":       32,
		"rope_kv_mrope_batched_pw": 32,
		"mla_q_rope":               32, // mla.ptx (FeatMLA)
		"mla_latent_store":         32, // mla.ptx (FeatMLA)
		"mamba_conv":               32, // mamba.ptx (Mamba-2 conv); the same 1.2 MiB at full occupancy as the 32 B rope kernels above
		// deltanet.ptx, the batched Gated-DeltaNet prefill: the
		// scan holds its 128-float state row in registers across the rows and spills 24 floats of it at 255
		// registers (launch_bounds(128, 1) gives the same 96 B). That is 3.8 MiB at full occupancy, accepted
		// for the speed over the no-spill generic scan.
		"delta_rule_rows_128": 96,
		// Its verify twin runs the same row body over
		// the folded rows and its own, and spills the same 96 B: the backing store, sized by the largest kernel launched, does
		// not grow.
		"delta_rule_rows_128_spec": 96,
	}
	got := map[string]int{}
	for _, r := range rows {
		if r.local != 0 {
			got[r.fn] = r.local
		}
	}
	for fn, want := range pinned {
		if g, ok := got[fn]; !ok {
			t.Errorf("PINNED: %s no longer declares local memory (was %d B/thread) — its deferred "+
				"first-launch reservation has changed and the recorded figures are stale", fn, want)
		} else if g != want {
			t.Errorf("PINNED: %s local memory %d B/thread, pinned at %d. A change here moves a "+
				"deferred first-launch reservation that the expert-cache sizing does not account "+
				"for; re-measure the demand threshold before updating this number", fn, g, want)
		}
	}
	for fn, g := range got {
		if _, ok := pinned[fn]; !ok {
			t.Errorf("PINNED: %s declares %d B/thread of local memory and is not in the pinned set — "+
				"a NEW deferred reservation exists that nothing has measured", fn, g)
		}
	}

	// ---- the moe_route-forcing precondition ----
	//
	// BuildResident forces moe_route BY NAME before sizing the expert cache, which is only sound
	// because the local-memory backing store is shared and sized by the LARGEST kernel (launching the
	// whole census gives a threshold and residual identical to moe_route alone, to the byte). Forcing
	// the maximum forces the pool for everything.
	//
	// That moe_route IS the maximum is checked here rather than assumed. Without this, a new kernel
	// with deeper per-thread scratch would make the warm-up force the wrong pool, allocSlots would
	// again size against memory about to be taken, and nothing would say so — naming one member of a
	// set is the sibling-drift shape, and this is what keeps the naming honest.
	//
	// route_gptoss is the one exception, and it declares more. It is bound only on gpt-oss,
	// and backend.go forces it in the same warm-up wherever it is bound; that launch is checked below.
	// Every other kernel runs where moe_route alone is forced, so moe_route must be the maximum of
	// the rest.
	maxFn, maxLocal := "", -1
	for _, r := range rows {
		if r.fn != "route_gptoss" && r.local > maxLocal {
			maxFn, maxLocal = r.fn, r.local
		}
	}
	if maxFn != "moe_route" {
		t.Errorf("%s declares %d B/thread, more than moe_route's %d — cuda/backend.go forces "+
			"moe_route before allocSlots to pay the deferred local-memory reservation, and that is "+
			"sound only while moe_route is the maximum. Force %s there instead, or force both, and "+
			"re-measure the demand threshold", maxFn, maxLocal, pinned["moe_route"], maxFn)
	}
	if be, err := os.ReadFile("backend.go"); err != nil {
		t.Fatalf("read backend.go: %v", err)
	} else if !strings.Contains(string(be), "r.stream.Launch(r.gptOssRoute, onecfg(1, 0),") {
		t.Errorf("backend.go no longer forces route_gptoss before allocSlots, and it declares %d B/thread "+
			"against moe_route's %d: on gpt-oss the larger local-memory pool would grow after the "+
			"expert cache was sized (G-13(b))", got["route_gptoss"], pinned["moe_route"])
	}

	// ---- the multiplier, checked rather than assumed ----
	//
	// moe_route's reservation was measured at 138,412,032 B (TestMoERouteFirstLaunchReservation). If the
	// naive form local x SMs x maxThreadsPerSM reproduces it, the "256 -> 512 doubles it" derivation is
	// sound. If it does not, the multiplier carries something else and the derivation was doing more
	// work than it looked.
	const measuredMoERoute = 138412032
	for _, r := range rows {
		if r.fn != "moe_route" {
			continue
		}
		t.Logf("moe_route: local %d B/thread, naive form predicts %d B, MEASURED %d B (ratio %.4f)",
			r.local, r.predicted, int64(measuredMoERoute), float64(measuredMoERoute)/float64(r.predicted))
		if r.predicted != measuredMoERoute {
			t.Logf("  => the naive multiplier does NOT reproduce the measurement. The occupancy "+
				"factor is not SMs x maxThreadsPerSM (%d x %d); proportionality in local-bytes is "+
				"therefore UNVERIFIED, and any 'halving MOE_MAX_E halves the cost' claim must be "+
				"measured by building at 256 rather than derived from this form.", sms, maxT)
		} else {
			t.Logf("  => the naive multiplier reproduces the measurement exactly, so the backing " +
				"store is linear in per-thread bytes at fixed occupancy and the halving derivation " +
				"is sound.")
		}
	}
}
