package decoder

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestCompareLogits_judgment(t *testing.T) {
	want := []float32{1, -2, 3, 0.5, -0.25}
	if cos, rel := compareLogits(want, want); cos < 0.9999999 || rel != 0 {
		t.Errorf("identical vectors: cosine %v, relative L2 %v", cos, rel)
	}
	// A uniform scale is invisible to the cosine; the relative L2 is the half that sees it, so a missing logit scale cannot pass on the cosine alone.
	scaled := make([]float32, len(want))
	for i, v := range want {
		scaled[i] = 2 * v
	}
	cos, rel := compareLogits(scaled, want)
	if cos < probeCosineFloor || rel <= probeRelL2Ceil {
		t.Errorf("a 2x scale: cosine %v (should pass), relative L2 %v (should exceed %v)", cos, rel, probeRelL2Ceil)
	}
	// An all-zero row has no direction; it must read as a defect, not as a match.
	if cos, rel := compareLogits(make([]float32, len(want)), want); cos >= probeCosineFloor || !math.IsInf(rel, 1) {
		t.Errorf("all-zero logits: cosine %v, relative L2 %v; want a failing pair", cos, rel)
	}
	if cos, _ := compareLogits(want[:3], want); cos >= probeCosineFloor {
		t.Errorf("length mismatch passed (cosine %v)", cos)
	}
	if !nonFinite([]float32{1, float32(math.NaN())}) || !nonFinite([]float32{float32(math.Inf(-1))}) || nonFinite(want) {
		t.Error("nonFinite misjudges NaN, Inf or a finite row")
	}
}

// The embedded fixtures are copies of testdata/ ones; the copy exists only so a released binary can carry them, and must not drift from the originals the parity gates use.
func TestSelfTestFixtures_matchTestdataAndFitTheirWindow(t *testing.T) {
	for _, fx := range selfTestFixtures {
		ents, err := selfTestFS.ReadDir("selftestdata/" + fx.name)
		if err != nil || len(ents) == 0 {
			t.Fatalf("%s: no embedded files (%v)", fx.name, err)
		}
		hasWeights := false
		for _, e := range ents {
			emb, err := selfTestFS.ReadFile("selftestdata/" + fx.name + "/" + e.Name())
			if err != nil {
				t.Fatal(err)
			}
			orig, err := os.ReadFile(filepath.Join("..", "testdata", fx.name, e.Name()))
			if err != nil {
				t.Skipf("no testdata original for %s/%s: %v", fx.name, e.Name(), err)
			}
			if !bytes.Equal(emb, orig) {
				t.Errorf("%s/%s drifted from testdata/%s/%s; re-copy it", fx.name, e.Name(), fx.name, e.Name())
			}
			if e.Name() == "model.safetensors" {
				hasWeights = true
			}
			if e.Name() == "config.json" {
				var c map[string]any
				if err := json.Unmarshal(emb, &c); err != nil {
					t.Fatal(err)
				}
				if tc, ok := c["text_config"].(map[string]any); ok {
					c = tc
				}
				win, _ := c["max_position_embeddings"].(float64)
				if win > 0 && float64(fx.positions) > win {
					t.Errorf("%s: probes %d positions past its %v-position window; the resident refuses a position beyond its cap", fx.name, fx.positions, win)
				}
			}
		}
		if !hasWeights {
			t.Errorf("%s: no model.safetensors embedded (a .gitignore rule dropped it?)", fx.name)
		}
	}
}

// The calling convention under test is withResidency's: the probe is reached from a real Load, runs once per backend and quant before the model's own build, and a decline skips that
// build. A resident that answers with a one-hot row (fakeResident) cannot agree with the CPU's logits on any
// fixture.
func TestGPUProbe_declinesAWrongResidentThroughLoad(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	var warns []string
	var mu sync.Mutex
	prevWarn := selfTestWarn
	selfTestWarn = func(m string) { mu.Lock(); warns = append(warns, m); mu.Unlock() }
	t.Cleanup(func() { selfTestWarn = prevWarn })
	resetSelfTestCaches()

	var builds atomic.Int64
	const name = "fake-probe-gpu"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be := &fakeNamedResidency{name: name}
		be.Backend = cpu
		return &countingBuild{fakeNamedResidency: be, n: &builds}, nil
	})

	m, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.resident != nil {
		t.Fatal("a resident whose logits disagree with the CPU's everywhere stayed attached")
	}
	if d := m.ResidentDecline(); !strings.Contains(d, "startup self-test declined") {
		t.Errorf("decline reason %q does not name the self-test", d)
	}
	if got := int64(len(selfTestFixtures)); builds.Load() != got {
		t.Errorf("%d resident builds on first load; want one per fixture (%d) and none for the model the probe declined", builds.Load(), got)
	}
	var rec *SelfTestResult
	for _, r := range SelfTestResults() {
		if r.Backend == name {
			r := r
			rec = &r
		}
	}
	if rec == nil || rec.Status != SelfTestDeclined {
		t.Fatalf("recorded result %+v; want a declined one for %s", rec, name)
	}
	mu.Lock()
	nw := len(warns)
	mu.Unlock()
	if nw == 0 {
		t.Error("no WARN line for a declined backend")
	}

	// Cached: a second model on the same backend and quant is declined from the cache, building nothing.
	before := builds.Load()
	m2, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	if builds.Load()-before != 0 || m2.resident != nil || !strings.Contains(m2.ResidentDecline(), "startup self-test declined") {
		t.Errorf("second load: %d builds (want 0: no re-probe, no build), resident attached = %v, decline %q", builds.Load()-before, m2.resident != nil, m2.ResidentDecline())
	}

	// -no-selftest: nothing is checked, so the (wrong) resident stays, which is exactly what the flag trades away.
	t.Cleanup(func() { selfTestSkipped.Store(false) })
	SkipSelfTests()
	m3, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m3.Close()
	if m3.resident == nil {
		t.Error("with self-tests skipped the resident was still dropped")
	}
}

// TestGPUProbe_runsBeforeTheBuild is H6C-S1 and H6C-S3: on a first load
// the probe's fixture builds all come before the model's own, a probe that does not decline keeps the model's resident, a second load builds the
// model once and no fixture, and with self-tests skipped no fixture is built. The fixtures decline to go resident here, so the probe reads
// skipped, not declined, and the order is all that is under test.
func TestGPUProbe_runsBeforeTheBuild(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	const name = "fake-order-gpu"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	var mu sync.Mutex
	var seq []bool // true: a fixture's build; false: a model's own
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be := &fakeNamedResidency{name: name}
		be.Backend = cpu
		return &orderedBuild{fakeNamedResidency: be, seq: &seq, mu: &mu}, nil
	})
	load := func(q string) *Model {
		t.Helper()
		m, err := Load(tinyFixture(t), Options{Backend: name, Quant: q})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = m.Close() })
		return m
	}
	m := load("int4")
	mu.Lock()
	first := append([]bool(nil), seq...)
	mu.Unlock()
	if len(first) != len(selfTestFixtures)+1 || first[len(first)-1] {
		t.Fatalf("first load built %v (true: a fixture); want %d fixtures, then the model", first, len(selfTestFixtures))
	}
	for i, fx := range first[:len(first)-1] {
		if !fx {
			t.Errorf("build %d of the first load was a model's, before the probe finished: %v", i, first)
		}
	}
	if m.resident == nil {
		t.Errorf("a probe that did not decline dropped the resident: %s", m.ResidentDecline())
	}
	m2 := load("int4")
	mu.Lock()
	second := append([]bool(nil), seq[len(first):]...)
	mu.Unlock()
	if len(second) != 1 || second[0] || m2.resident == nil {
		t.Errorf("second load built %v (want one model build, no fixture), resident attached %v", second, m2.resident != nil)
	}
	t.Cleanup(func() { selfTestSkipped.Store(false) })
	SkipSelfTests()
	m3 := load("int8")
	mu.Lock()
	third := append([]bool(nil), seq[len(first)+len(second):]...)
	mu.Unlock()
	if len(third) != 1 || third[0] || m3.resident == nil {
		t.Errorf("with self-tests skipped the load built %v (want one model build, no fixture), resident attached %v", third, m3.resident != nil)
	}
}

// orderedBuild records each build in order (true for a self-test fixture's) and declines the fixtures, so the probe reads skipped.
type orderedBuild struct {
	*fakeNamedResidency
	seq *[]bool
	mu  *sync.Mutex
}

func (o *orderedBuild) BuildResident(m *Model) (ResidentForward, bool, error) {
	o.mu.Lock()
	*o.seq = append(*o.seq, m.opts.noSelfTest)
	o.mu.Unlock()
	if m.opts.noSelfTest {
		return nil, false, nil
	}
	return o.fakeNamedResidency.BuildResident(m)
}

type countingBuild struct {
	*fakeNamedResidency
	n *atomic.Int64
}

func (c *countingBuild) BuildResident(m *Model) (ResidentForward, bool, error) {
	c.n.Add(1)
	return c.fakeNamedResidency.BuildResident(m)
}

// With no resident on the backend (a CPU-only binary, or a name that is not compiled in) nothing was checked, and the result must SAY so rather than read as a pass.
func TestGPUProbe_aBackendWithNoResidentIsSkippedNotPassed(t *testing.T) {
	r := probeBackend("not-compiled-in", "int4")
	if r.Status != SelfTestSkipped || r.Note == "" {
		t.Errorf("status %s, note %q; want skipped with the reasons", r.Status, r.Note)
	}
	for _, fx := range selfTestFixtures {
		if !strings.Contains(r.Note, fx.name) {
			t.Errorf("note does not name fixture %s: %q", fx.name, r.Note)
		}
	}
}

// A backend whose margins nobody measured is not probed at all: a wrong resident stays attached and nothing is recorded, rather than the probe judging it against bars it was never seen to clear.
func TestGPUProbe_anUnmeasuredBackendIsNotProbed(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	var builds atomic.Int64
	const name = "fake-unmeasured-gpu"
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be := &fakeNamedResidency{name: name}
		be.Backend = cpu
		return &countingBuild{fakeNamedResidency: be, n: &builds}, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.resident == nil || builds.Load() != 1 {
		t.Errorf("resident attached = %v, %d builds; want the one resident kept and no probe builds", m.resident != nil, builds.Load())
	}
	for _, r := range SelfTestResults() {
		if r.Backend == name {
			t.Errorf("a result was recorded for an unmeasured backend: %s", r.Summary())
		}
	}
	// The report path says so instead of staying silent.
	if r, _ := probeStandalone(name); r.Status != SelfTestSkipped || !strings.Contains(r.Note, "no self-test for this backend yet") {
		t.Errorf("standalone probe of an unmeasured backend: %s", r.Summary())
	}
}

// deviceState is the mutable answer a fake standalone device check gives across calls.
type deviceState struct {
	mu  sync.Mutex
	err error
}

func (s *deviceState) set(err error) { s.mu.Lock(); s.err = err; s.mu.Unlock() }
func (s *deviceState) get() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// standaloneFake is a temporary backend whose standalone device check reads state and counts its own
// calls; countingBuild counts every fixture build, so a skipped check is visible as 0 builds.
type standaloneFake struct {
	*countingBuild
	hooks *atomic.Int64
	state *deviceState
}

func (f *standaloneFake) SelfTestDeviceError() error {
	f.hooks.Add(1)
	return f.state.get()
}

func registerStandaloneFake(name string, builds, hooks *atomic.Int64, state *deviceState) {
	RegisterBackend(name, func() (Backend, error) {
		cpu, e := NewBackend("cpu")
		if e != nil {
			return nil, e
		}
		be := &fakeNamedResidency{name: name}
		be.Backend = cpu
		return &standaloneFake{countingBuild: &countingBuild{fakeNamedResidency: be, n: builds}, hooks: hooks, state: state}, nil
	})
}

// A device the standalone check cannot open is skipped before any fixture is loaded: four fixture loads would
// only repeat the same decline. The original driver error is kept in the note, and the temporary backend is
// still created (NewBackend per standalone call) and closed.
func TestGPUProbe_standaloneSkipsFixturesWhenTheDeviceIsUnavailable(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	const name = "fake-unavailable-gpu"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	var builds, hooks atomic.Int64
	registerStandaloneFake(name, &builds, &hooks, &deviceState{err: errors.New("libcuda.so.1: cannot open shared object file: No such file or directory")})
	r, override := probeStandalone(name)
	if r.Status != SelfTestSkipped || !override || !strings.Contains(r.Note, "libcuda.so.1") {
		t.Fatalf("standalone probe of an unavailable device: %s", r.Summary())
	}
	if n := builds.Load(); n != 0 {
		t.Errorf("%d fixture builds on an unavailable device; want 0", n)
	}
	if n := hooks.Load(); n != 1 {
		t.Errorf("standalone device check ran %d times; want 1", n)
	}
	for _, got := range SelfTestResults() {
		if got.Backend == name {
			t.Errorf("an availability-only skip was stored in the health-visible results: %s", got.Summary())
		}
	}
	// The public report carries the observation for this invocation; the process snapshot does not.
	var report *SelfTestResult
	for _, got := range RunSelfTests() {
		if got.Backend == name {
			got := got
			report = &got
		}
	}
	if report == nil || report.Status != SelfTestSkipped || !strings.Contains(report.Note, "libcuda.so.1") {
		t.Errorf("RunSelfTests report %+v; want the unavailable-device skip", report)
	}
	for _, got := range SelfTestResults() {
		if got.Backend == name {
			t.Errorf("RunSelfTests stored the availability skip globally: %s", got.Summary())
		}
	}
}

// A device that opens still runs every fixture, and a genuine mismatch still declines: the availability check
// must not turn a healthy but disagreeing GPU into a skip.
func TestGPUProbe_standaloneRunsEveryFixtureAndStillDeclinesAMismatch(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	const name = "fake-healthy-but-wrong-gpu"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	var builds, hooks atomic.Int64
	registerStandaloneFake(name, &builds, &hooks, &deviceState{})
	r, override := probeStandalone(name)
	if r.Status != SelfTestDeclined || !override {
		t.Fatalf("a wrong resident on an open device: %s; want declined", r.Summary())
	}
	if n := builds.Load(); n != int64(len(selfTestFixtures)) {
		t.Errorf("%d fixture builds on a reachable device; want %d", n, len(selfTestFixtures))
	}
	if n := hooks.Load(); n != 1 {
		t.Errorf("standalone device check ran %d times; want 1", n)
	}
}

// A model load uses the in-load probe, never the standalone device check: the optional seam must not open a
// second context on that path.
func TestGPUProbe_modelLoadDoesNotUseTheStandaloneDeviceCheck(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	const name = "fake-loaded-model-gpu"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	var builds, hooks atomic.Int64
	registerStandaloneFake(name, &builds, &hooks, &deviceState{err: errors.New("must not be consulted by a load")})
	m, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if n := hooks.Load(); n != 0 {
		t.Errorf("standalone device check ran %d times during a model load; want 0", n)
	}
	if n := builds.Load(); n != int64(len(selfTestFixtures)) {
		t.Errorf("model load ran %d fixture builds; want %d", n, len(selfTestFixtures))
	}
}

// The report's self-test list keeps the CPU test when a linked GPU has no device: skipping the GPU fixtures must
// not skip the work this process can still do.
func TestRunSelfTests_keepsCPUTestedWhenAGPUIsUnavailable(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	const name = "fake-unavailable-cpu-control"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	var builds, hooks atomic.Int64
	registerStandaloneFake(name, &builds, &hooks, &deviceState{err: errors.New("no CUDA device")})
	cpuOK, gpuSkipped := false, false
	for _, r := range RunSelfTests() {
		switch r.Backend {
		case "cpu":
			cpuOK = r.Status == SelfTestPass || r.Status == SelfTestRepaired
		case name:
			gpuSkipped = r.Status == SelfTestSkipped && strings.Contains(r.Note, "no CUDA device")
		}
	}
	if !cpuOK {
		t.Error("the CPU self-test did not run or did not pass beside an unavailable GPU")
	}
	if !gpuSkipped {
		t.Error("the unavailable GPU was not recorded as skipped with its original reason")
	}
	if n := builds.Load(); n != 0 {
		t.Errorf("%d fixture builds beside an unavailable GPU; want 0", n)
	}
}

// A device that is reachable, then unavailable, then reachable again must report the cached numerical
// result each time it is reachable: the unavailable skip overwrites the display, and recovery restores it
// without re-running fixtures or announcing the result twice.
func TestRunSelfTests_restoresACachedResultWhenTheDeviceReturns(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	const name = "fake-recovered-gpu"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	var builds, hooks atomic.Int64
	state := &deviceState{}
	registerStandaloneFake(name, &builds, &hooks, state)
	result := func() SelfTestResult {
		t.Helper()
		for _, r := range RunSelfTests() {
			if r.Backend == name {
				return r
			}
		}
		t.Fatal("no result recorded for the fake backend")
		return SelfTestResult{}
	}
	health := func() SelfTestResult {
		t.Helper()
		for _, r := range SelfTestResults() {
			if r.Backend == name {
				return r
			}
		}
		return SelfTestResult{}
	}
	if r := result(); r.Status != SelfTestDeclined {
		t.Fatalf("reachable/wrong first: %s; want declined", r.Summary())
	}
	if r := health(); r.Status != SelfTestDeclined {
		t.Fatalf("first health-visible result: %s; want declined", r.Summary())
	}
	if n := builds.Load(); n != int64(len(selfTestFixtures)) {
		t.Fatalf("first reachable check built %d fixtures; want %d", n, len(selfTestFixtures))
	}
	state.set(errors.New("no CUDA device"))
	if r := result(); r.Status != SelfTestSkipped || !strings.Contains(r.Note, "no CUDA device") {
		t.Fatalf("unavailable second: %s; want skipped with the device error", r.Summary())
	}
	if r := health(); r.Status != SelfTestDeclined || len(r.Mismatches) == 0 {
		t.Fatalf("health-visible result while missing: %s; the numerical decline and its details must survive", r.Summary())
	}
	if n := builds.Load(); n != int64(len(selfTestFixtures)) {
		t.Errorf("unavailable check built %d fixtures; want no new build", n-int64(len(selfTestFixtures)))
	}
	// A real load after the skip reads the numerical cache; it must not republish the availability skip
	// or overwrite the health-visible decline.
	m, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if r := health(); r.Status != SelfTestDeclined || len(r.Mismatches) == 0 {
		t.Fatalf("health-visible result after the load: %s; want the numerical decline", r.Summary())
	}
	if n := builds.Load(); n != int64(len(selfTestFixtures)) {
		t.Errorf("the cached load built %d fixtures; want none", n-int64(len(selfTestFixtures)))
	}
	state.set(nil)
	if r := result(); r.Status != SelfTestDeclined {
		t.Fatalf("reachable again: %s; want the cached declined result restored", r.Summary())
	}
	if r := health(); r.Status != SelfTestDeclined {
		t.Fatalf("health-visible result after recovery: %s; want declined", r.Summary())
	}
	if n := builds.Load(); n != int64(len(selfTestFixtures)) {
		t.Errorf("recovery built %d fixtures; want no rerun", n-int64(len(selfTestFixtures)))
	}
	if n := hooks.Load(); n != 3 {
		t.Errorf("standalone device check ran %d times over three reports; want 3", n)
	}
}

// A newer numerical result for another quant wins in the process snapshot: an availability skip is only a
// report-local overlay and must never replace it, whether the device is missing or returns.
func TestRunSelfTests_keepsANewerQuantResultOverAvailability(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	const name = "fake-newer-quant-gpu"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	var builds, hooks atomic.Int64
	state := &deviceState{}
	registerStandaloneFake(name, &builds, &hooks, state)
	// A cached int4 pass exists, and the newest real result is an int8int8 decline.
	gpuProbeMu.Lock()
	gpuProbeCache[name+"/int4"] = SelfTestResult{Backend: name, Status: SelfTestPass}
	gpuProbeMu.Unlock()
	prevWarn := selfTestWarn
	selfTestWarn = func(string) {}
	t.Cleanup(func() { selfTestWarn = prevWarn })
	RecordSelfTest(SelfTestResult{Backend: name, Status: SelfTestDeclined, Note: "int8int8 decline", Mismatches: []string{"int8int8: observed 1, allowed 0"}})
	snapshot := func() SelfTestResult {
		t.Helper()
		for _, r := range SelfTestResults() {
			if r.Backend == name {
				return r
			}
		}
		return SelfTestResult{}
	}
	report := func() SelfTestResult {
		t.Helper()
		for _, r := range RunSelfTests() {
			if r.Backend == name {
				return r
			}
		}
		return SelfTestResult{}
	}
	if r := report(); r.Status != SelfTestPass {
		t.Fatalf("report with device present: %s; the report follows the standalone int4 check", r.Summary())
	}
	if r := snapshot(); r.Status != SelfTestDeclined || !strings.Contains(r.Note, "int8int8 decline") || len(r.Mismatches) == 0 {
		t.Fatalf("snapshot with device present: %s; want the newer int8int8 decline and its detail", r.Summary())
	}
	state.set(errors.New("no CUDA device"))
	if r := report(); r.Status != SelfTestSkipped {
		t.Fatalf("report with device missing: %s; want the report-local skip", r.Summary())
	}
	if r := snapshot(); r.Status != SelfTestDeclined || !strings.Contains(r.Note, "int8int8 decline") || len(r.Mismatches) == 0 {
		t.Fatalf("snapshot with device missing: %s; the newer quant result and detail must survive", r.Summary())
	}
	state.set(nil)
	if r := report(); r.Status != SelfTestPass {
		t.Fatalf("report after recovery: %s; the report follows the standalone int4 check", r.Summary())
	}
	if r := snapshot(); r.Status != SelfTestDeclined || !strings.Contains(r.Note, "int8int8 decline") || len(r.Mismatches) == 0 {
		t.Fatalf("snapshot after recovery: %s; the newer quant result and detail must survive", r.Summary())
	}
	if n := builds.Load(); n != 0 {
		t.Errorf("%d fixture builds; the cached int4 result and the newer result need none", n)
	}
}

// A backend that names itself "webgpu:vulkan" is probed under its registry name, and one that declares itself ineligible (a software adapter) is skipped with its reason, keeps its resident, and is never recorded as a pass.
type ineligibleBackend struct {
	*fakeNamedResidency
	why string
}

func (b *ineligibleBackend) SelfTestEligible() (bool, string) { return false, b.why }

func TestGPUProbe_normalizesTheBackendNameAndHonoursIneligibility(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	if got := probeBackendName(&fakeNamedResidency{name: "webgpu:vulkan"}); got != "webgpu" {
		t.Errorf("probeBackendName(webgpu:vulkan) = %q, want webgpu", got)
	}
	const reg = "fake-soft-gpu"
	probeMeasured[reg] = true
	t.Cleanup(func() { delete(probeMeasured, reg) })
	RegisterBackend(reg, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be := &fakeNamedResidency{name: reg + ":software"}
		be.Backend = cpu
		return &ineligibleBackend{fakeNamedResidency: be, why: "software adapter: test"}, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: reg, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.resident == nil {
		t.Error("an ineligible backend's resident was dropped; it should be left alone, unprobed")
	}
	var rec *SelfTestResult
	for _, r := range SelfTestResults() {
		if r.Backend == reg {
			r := r
			rec = &r
		}
	}
	if rec == nil || rec.Status != SelfTestSkipped || !strings.Contains(rec.Note, "software adapter") {
		t.Errorf("recorded %+v; want skipped with the adapter reason", rec)
	}
	if r, _ := probeStandalone(reg); r.Status != SelfTestSkipped || !strings.Contains(r.Note, "software adapter") {
		t.Errorf("standalone probe of an ineligible backend: %s", r.Summary())
	}
}

// A resident that computes another precision than the load quant is not judged at the load quant (H2 on Metal: Metal
// re-quantizes int8int8 DeltaNet and MoE models to int4). Through Load, a fake GPU whose resident reports int4 for an
// int8int8 load is skipped fixture by fixture, recorded as skipped, not passed, and keeps its resident; the control, the
// same wrong resident reporting the load quant, is declined. residentPrecision also reads Metal's re-quantizing labels.
func TestGPUProbe_aReQuantizingResidentIsNotJudgedAtTheLoadQuant(t *testing.T) {
	for _, c := range []struct {
		backend, quant, report, want string
	}{
		{"metal", "int8int8", "", "int4"}, {"metal", "int8", "", "int4"}, {"metal", "int4mix", "", "int4"},
		{"metal", "int8int8", "int8int8", "int8int8"}, {"metal", "int4", "", "int4"}, {"cuda", "int8int8", "", "int8int8"},
	} {
		var rf ResidentForward = &fakeResident{}
		if c.report != "" {
			rf = quantReporting{ResidentForward: rf, q: c.report}
		}
		if got := residentPrecision(c.backend, c.quant, rf); got != c.want {
			t.Errorf("residentPrecision(%s, %s, reports %q) = %q, want %q", c.backend, c.quant, c.report, got, c.want)
		}
	}
	for _, c := range []struct {
		report string
		want   string
		kept   bool
	}{{"int4", SelfTestSkipped, true}, {"int8int8", SelfTestDeclined, false}} {
		t.Run("reports "+c.report, func(t *testing.T) {
			t.Cleanup(resetSelfTestCaches)
			resetSelfTestCaches()
			prevWarn := selfTestWarn
			selfTestWarn = func(string) {}
			t.Cleanup(func() { selfTestWarn = prevWarn })
			name := "fake-requant-gpu-" + c.report
			probeMeasured[name] = true
			t.Cleanup(func() { delete(probeMeasured, name) })
			RegisterBackend(name, func() (Backend, error) {
				cpu, err := NewBackend("cpu")
				if err != nil {
					return nil, err
				}
				be := &fakeNamedResidency{name: name, report: c.report}
				be.Backend = cpu
				return be, nil
			})
			m, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int8int8"})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if (m.resident != nil) != c.kept {
				t.Errorf("resident attached = %v, want %v (%s)", m.resident != nil, c.kept, m.ResidentDecline())
			}
			var rec *SelfTestResult
			for _, r := range SelfTestResults() {
				if r.Backend == name {
					r := r
					rec = &r
				}
			}
			if rec == nil || rec.Status != c.want {
				t.Fatalf("recorded %+v; want status %s", rec, c.want)
			}
			if c.want == SelfTestSkipped {
				for _, fx := range selfTestFixtures {
					if !strings.Contains(rec.Note, fx.name+": re-quantized to int4") {
						t.Errorf("note does not say %s was re-quantized and not probed: %q", fx.name, rec.Note)
					}
				}
			}
		})
	}
}
