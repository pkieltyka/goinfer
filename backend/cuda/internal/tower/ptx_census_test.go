//go:build cuda

package tower

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
)

var (
	censusPTXSM    = regexp.MustCompile(`(?m)^\s*\.target\s+sm_([0-9]+)(?:[ ,\t]|$)`)
	censusPTXEntry = regexp.MustCompile(`\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_$]*)`)
	censusCUGlobal = regexp.MustCompile(`__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	censusCUComm   = regexp.MustCompile(`(?s)/\*.*?\*/|//[^\n]*`)
)

func towerEmbeddedPTX(t *testing.T) []string {
	t.Helper()
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	var out []string
	re := regexp.MustCompile(`//go:embed testdata/([A-Za-z0-9_]+\.ptx)`)
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllStringSubmatch(string(b), -1) {
			if !seen[m[1]] {
				seen[m[1]] = true
				out = append(out, m[1])
			}
		}
	}
	sort.Strings(out)
	return out
}

func TestTowerPTX_censusAndSourceMatch(t *testing.T) {
	mods := towerEmbeddedPTX(t)
	if len(mods) != 1 || mods[0] != "tower_base.ptx" {
		t.Fatalf("embedded PTX modules = %v, want [tower_base.ptx]", mods)
	}
	ptx, err := os.ReadFile(filepath.Join("testdata", "tower_base.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	smMatch := censusPTXSM.FindSubmatch(ptx)
	if len(smMatch) != 2 {
		t.Fatal("tower_base.ptx has no numeric .target sm_XX")
	}
	target, err := strconv.Atoi(string(smMatch[1]))
	if err != nil {
		t.Fatal(err)
	}
	maxSM := 10*PTXTargetMajor + PTXTargetMinor
	if target <= 0 || target > maxSM {
		t.Fatalf("tower_base.ptx target sm_%d outside (0, sm_%d]", target, maxSM)
	}
	cu, err := os.ReadFile(filepath.Join("..", "..", "tower_base.cu"))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{}
	for _, m := range censusCUGlobal.FindAllStringSubmatch(censusCUComm.ReplaceAllString(string(cu), ""), -1) {
		want[m[1]] = true
	}
	got := map[string]bool{}
	for _, m := range censusPTXEntry.FindAllStringSubmatch(string(ptx), -1) {
		got[m[1]] = true
	}
	if len(want) == 0 {
		t.Fatal("no __global__ kernels found in tower_base.cu")
	}
	for k := range want {
		if !got[k] {
			t.Errorf("tower_base.cu declares %s missing from tower_base.ptx", k)
		}
	}
	for k := range got {
		if !want[k] {
			t.Errorf("tower_base.ptx declares stale entry %s not in tower_base.cu", k)
		}
	}
}

type towerPipeShim struct{ f *gc.Function }

func TestTowerKernelLocalMemoryCensus(t *testing.T) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer dev.ReleaseObjects()
	mod, err := dev.CompileLibrary(BasePTX)
	if err != nil {
		t.Fatalf("CompileLibrary(tower_base.ptx): %v", err)
	}
	seen := map[string]bool{}
	for _, m := range censusPTXEntry.FindAllStringSubmatch(string(BasePTX), -1) {
		seen[m[1]] = true
	}
	if len(seen) == 0 {
		t.Fatal("tower_base.ptx has no .visible .entry")
	}
	for name := range seen {
		p, err := dev.NewComputePipeline(mod, name)
		if err != nil {
			t.Fatalf("NewComputePipeline(%s): %v", name, err)
		}
		fn := (*towerPipeShim)(unsafe.Pointer(&p)).f
		local, err := fn.Attribute(gc.FuncAttrLocalSizeBytes)
		if err != nil {
			t.Fatalf("%s LOCAL_SIZE_BYTES: %v", name, err)
		}
		if local != 0 {
			t.Errorf("tower_base/%s declares %d B/thread local memory, pinned at 0", name, local)
		}
	}
}
