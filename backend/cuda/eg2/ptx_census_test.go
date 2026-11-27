//go:build cuda

package eg2

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"unsafe"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/launchgeom"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/tower"
)

var (
	eg2CensusPTXSM    = regexp.MustCompile(`(?m)^\s*\.target\s+sm_([0-9]+)(?:[ ,\t]|$)`)
	eg2CensusPTXEntry = regexp.MustCompile(`\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_$]*)`)
	eg2CensusCUGlobal = regexp.MustCompile(`__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	eg2CensusCUComm   = regexp.MustCompile(`(?s)/\*.*?\*/|//[^\n]*`)
)

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
		match := eg2CensusPTXSM.FindSubmatch(b)
		if len(match) != 2 {
			t.Fatalf("%s has no numeric PTX SM target", name)
		}
		minSM, err := strconv.Atoi(string(match[1]))
		if err != nil {
			t.Fatalf("%s PTX SM target: %v", name, err)
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
	for _, m := range mods {
		gotNames = append(gotNames, m.name)
		if m.name == "eg2_ffn_fp16.ptx" && m.minSM != 80 {
			t.Errorf("FP16 FFN module requires SM%d, want SM80", m.minSM)
		}
		src, _ := launchgeom.PTXSourceName(strings.TrimSuffix(m.name, ".ptx"))
		cu, err := os.ReadFile(filepath.Join("..", src+".cu"))
		if err != nil {
			t.Fatalf("%s is embedded but ../%s.cu is missing: %v", m.name, src, err)
		}
		want := map[string]bool{}
		for _, mm := range eg2CensusCUGlobal.FindAllStringSubmatch(eg2CensusCUComm.ReplaceAllString(string(cu), ""), -1) {
			want[mm[1]] = true
		}
		got := map[string]bool{}
		for _, mm := range eg2CensusPTXEntry.FindAllStringSubmatch(string(m.ptx), -1) {
			got[mm[1]] = true
		}
		if len(want) == 0 {
			t.Fatalf("no __global__ kernels found in %s.cu", src)
		}
		for k := range want {
			if !got[k] {
				t.Errorf("%s.cu declares %s missing from %s", src, k, m.name)
			}
		}
		for k := range got {
			if !want[k] {
				t.Errorf("%s declares stale entry %s not in %s.cu", m.name, k, src)
			}
		}
	}
	sort.Strings(gotNames)
	if strings.Join(gotNames, ",") != strings.Join(wantMods, ",") {
		t.Fatalf("embedded EG2 PTX modules = %v, want %v", gotNames, wantMods)
	}
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
		seen := map[string]bool{}
		for _, mm := range eg2CensusPTXEntry.FindAllStringSubmatch(string(m.ptx), -1) {
			seen[mm[1]] = true
		}
		if len(seen) == 0 {
			t.Fatalf("%s has no .visible .entry", m.name)
		}
		for name := range seen {
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
