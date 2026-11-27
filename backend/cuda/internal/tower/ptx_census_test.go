//go:build cuda

package tower

import (
	"fmt"
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
	"github.com/pkieltyka/goinfer/backend/cuda/internal/ptxtest"
)

func towerBoundKernelNames(t *testing.T) []string {
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
	attnLoop := regexp.MustCompile(`for\s+_,\s*hd\s*:=\s*range\s+\[\]int\{([0-9,\s]+)\}`)
	seen := map[string]bool{}
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
		if strings.Contains(text, `fmt.Sprintf("tower_attn_hd%d", hd)`) {
			for _, m := range attnLoop.FindAllStringSubmatch(text, -1) {
				for part := range strings.SplitSeq(m[1], ",") {
					hd, err := strconv.Atoi(strings.TrimSpace(part))
					if err != nil {
						t.Fatalf("%s: parse tower_attn hd %q: %v", f, part, err)
					}
					seen[fmt.Sprintf("tower_attn_hd%d", hd)] = true
				}
			}
		}
	}
	out := make([]string, 0, len(seen))
	for name := range seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

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
	target, ok := ptxtest.TargetSM(ptx)
	if !ok {
		t.Fatal("tower_base.ptx has no numeric .target sm_XX")
	}
	maxSM := 10*PTXTargetMajor + PTXTargetMinor
	if target <= 0 || target > maxSM {
		t.Fatalf("tower_base.ptx target sm_%d outside (0, sm_%d]", target, maxSM)
	}
	cu, err := os.ReadFile(filepath.Join("..", "..", "tower_base.cu"))
	if err != nil {
		t.Fatal(err)
	}
	globals := ptxtest.Globals(string(cu))
	entries := ptxtest.Entries(string(ptx))
	if len(globals) == 0 {
		t.Fatal("no __global__ kernels found in tower_base.cu")
	}
	if len(entries) == 0 {
		t.Fatal("tower_base.ptx has no .visible .entry — empty or unparseable")
	}
	for name, want := range globals {
		got, ok := entries[name]
		if !ok {
			t.Errorf("tower_base.cu declares __global__ %s missing from tower_base.ptx", name)
			continue
		}
		if len(got) != len(want) {
			t.Errorf("%s: tower_base.cu has %d parameters, tower_base.ptx has %d — the PTX lags its source", name, len(want), len(got))
			continue
		}
		for i := range want {
			if want[i] == "?" || got[i] == "?" || want[i] != got[i] {
				t.Errorf("%s parameter %d: tower_base.cu kind %s, tower_base.ptx kind %s — the PTX lags its source", name, i, want[i], got[i])
			}
		}
	}
	for name := range entries {
		if _, ok := globals[name]; !ok {
			t.Errorf("tower_base.ptx declares stale entry %s not in tower_base.cu", name)
		}
	}
	bound := towerBoundKernelNames(t)
	if len(bound) < 10 {
		t.Fatalf("found only %d bound kernel names in tower — binding scan is broken (%v)", len(bound), bound)
	}
	for _, name := range bound {
		if _, ok := entries[name]; !ok {
			t.Errorf("production code binds kernel %q, which tower_base.ptx does not define", name)
		}
	}
	t.Logf("tower_base: %d kernels checked by name and signature; %d bound kernel names resolved", len(globals), len(bound))
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
	entries := ptxtest.Entries(string(BasePTX))
	if len(entries) == 0 {
		t.Fatal("tower_base.ptx has no .visible .entry")
	}
	for name := range entries {
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
