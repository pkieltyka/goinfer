//go:build cuda

package eg2

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
	"unsafe"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/launchgeom"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/ptxtest"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/tower"
)

func eg2BoundKernelNames(t *testing.T) []string {
	t.Helper()
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	res := []*regexp.Regexp{
		regexp.MustCompile(`NewComputePipeline\(\s*\w+\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)`),
		regexp.MustCompile(`\{\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*&[\w.\[\]]+\s*\}`),
		regexp.MustCompile(`\{\s*&[\w.\[\]]+\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\}`),
	}
	seen := map[string]bool{}
	usesPackedSpec := false
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		text := string(b)
		for _, re := range res {
			for _, m := range re.FindAllStringSubmatch(text, -1) {
				seen[m[1]] = true
			}
		}
		if strings.Contains(text, "NewComputePipeline(lib, spec.Entry)") && strings.Contains(text, "EG2PackedAttnSpecForHD") {
			usesPackedSpec = true
		}
	}
	if usesPackedSpec {
		for _, hd := range []int{16, 32, 256, 512} {
			spec, ok := launchgeom.EG2PackedAttnSpecForHD(hd)
			if !ok {
				t.Fatalf("EG2PackedAttnSpecForHD(%d) unexpectedly unsupported", hd)
			}
			seen[spec.Entry] = true
		}
	}
	out := make([]string, 0, len(seen))
	for name := range seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

type eg2CensusModule struct {
	name  string
	ptx   []byte
	minSM int
}

func validEG2CensusPTXTarget(name string, target int) bool {
	if name == "eg2_ffn_fp16.ptx" {
		return target == 10*eg2FP16MinMajor
	}
	return target > 0 && target <= 10*tower.PTXTargetMajor+tower.PTXTargetMinor
}

func eg2CensusModules(t *testing.T) []eg2CensusModule {
	t.Helper()
	names, err := launchgeom.EmbeddedPTXModules(".")
	if err != nil {
		t.Fatal(err)
	}
	out := make([]eg2CensusModule, 0, len(names))
	for _, name := range names {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		minSM, ok := ptxtest.TargetSM(b)
		if !ok {
			t.Fatalf("%s has no numeric PTX SM target", name)
		}
		if !validEG2CensusPTXTarget(name, minSM) {
			t.Fatalf("%s PTX target sm_%d exceeds or disagrees with its supported architecture", name, minSM)
		}
		out = append(out, eg2CensusModule{name: name, ptx: b, minSM: minSM})
	}
	return out
}

func TestEG2PTX_censusAndSourceMatch(t *testing.T) {
	if !validEG2CensusPTXTarget("eg2_fusions.ptx", 75) || validEG2CensusPTXTarget("eg2_fusions.ptx", 80) ||
		!validEG2CensusPTXTarget("eg2_ffn_fp16.ptx", 80) || validEG2CensusPTXTarget("eg2_ffn_fp16.ptx", 75) {
		t.Fatal("production sm_75 targets and the FP16 sm_80 exception must be checked independently")
	}
	mods := eg2CensusModules(t)
	wantMods := []string{
		"eg2_ffn_fp16.ptx",
		"eg2_fusions.ptx",
		"eg2_packed_attn.ptx",
		"eg2_transposed_experiment.ptx",
	}
	var gotNames []string
	entryMod := map[string]string{}
	checkedKernels := 0
	for _, m := range mods {
		gotNames = append(gotNames, m.name)
		if m.name == "eg2_ffn_fp16.ptx" && m.minSM != 80 {
			t.Errorf("FP16 FFN module requires SM%d, want SM80", m.minSM)
		}
		mod := strings.TrimSuffix(m.name, ".ptx")
		src, _ := launchgeom.PTXSourceName(mod)
		cu, err := os.ReadFile(filepath.Join("..", src+".cu"))
		if err != nil {
			t.Fatalf("%s is embedded but ../%s.cu is missing: %v", m.name, src, err)
		}
		globals := ptxtest.Globals(string(cu))
		entries := ptxtest.Entries(string(m.ptx))
		if len(globals) == 0 {
			t.Fatalf("no __global__ kernels found in %s.cu", src)
		}
		if len(entries) == 0 {
			t.Fatalf("%s has no .visible .entry — empty or unparseable", m.name)
		}
		for name := range entries {
			entryMod[name] = mod
		}
		for name, want := range globals {
			checkedKernels++
			got, ok := entries[name]
			if !ok {
				t.Errorf("%s.cu declares __global__ %s missing from %s", src, name, m.name)
				continue
			}
			if len(got) != len(want) {
				t.Errorf("%s: %s.cu has %d parameters, %s has %d — the PTX lags its source", name, src, len(want), m.name, len(got))
				continue
			}
			for i := range want {
				if want[i] == "?" || got[i] == "?" || want[i] != got[i] {
					t.Errorf("%s parameter %d: %s.cu kind %s, %s kind %s — the PTX lags its source", name, i, src, want[i], m.name, got[i])
				}
			}
		}
		for name := range entries {
			if _, ok := globals[name]; !ok {
				t.Errorf("%s declares stale entry %s not in %s.cu", m.name, name, src)
			}
		}
	}
	sort.Strings(gotNames)
	if strings.Join(gotNames, ",") != strings.Join(wantMods, ",") {
		t.Fatalf("embedded EG2 PTX modules = %v, want %v", gotNames, wantMods)
	}
	bound := eg2BoundKernelNames(t)
	if len(bound) < 10 {
		t.Fatalf("found only %d bound kernel names in eg2 — binding scan is broken (%v)", len(bound), bound)
	}
	for _, name := range bound {
		if _, ok := entryMod[name]; !ok {
			t.Errorf("production code binds kernel %q, which no embedded EG2 PTX defines", name)
		}
	}
	t.Logf("%d .cu/.ptx pairs, %d kernels checked by name and signature; %d bound kernel names resolved", len(mods), checkedKernels, len(bound))
}

type eg2PipeShim struct{ f *gc.Function }

func TestEG2KernelLocalMemoryCensus(t *testing.T) {
	dev, err := gpu.CreateSystemDefaultDevice()
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
	for _, m := range eg2CensusModules(t) {
		if m.minSM > deviceSM {
			t.Logf("NOT APPLICABLE: %s requires SM%d, device is SM%d", m.name, m.minSM, deviceSM)
			continue
		}
		mod, err := dev.CompileLibrary(m.ptx)
		if err != nil {
			t.Fatalf("CompileLibrary(%s): %v", m.name, err)
		}
		entries := ptxtest.Entries(string(m.ptx))
		if len(entries) == 0 {
			t.Fatalf("%s has no .visible .entry", m.name)
		}
		for name := range entries {
			p, err := dev.NewComputePipeline(mod, name)
			if err != nil {
				t.Fatalf("%s/%s: NewComputePipeline: %v", m.name, name, err)
			}
			fn := (*eg2PipeShim)(unsafe.Pointer(&p)).f
			local, err := fn.Attribute(gc.FuncAttrLocalSizeBytes)
			if err != nil {
				t.Fatalf("%s/%s LOCAL_SIZE_BYTES: %v", m.name, name, err)
			}
			if local != 0 {
				t.Errorf("%s/%s declares %d B/thread local memory, pinned at 0", m.name, name, local)
			}
		}
	}
}
