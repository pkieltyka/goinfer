//go:build cuda

package eg2

import (
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

// TestEG2Leaf_importGraph verifies that importing backend/cuda/eg2 under the
// fullest EG2 tag set pulls neither backend/cuda nor any decoder/generation
// runtime package.
func TestEG2Leaf_importGraph(t *testing.T) {
	cmd := exec.Command("go", "list", "-deps", "-tags", "cuda,eg2_combined,eg2_fp16", ".")
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go list -deps: %v\n%s", err, out)
	}
	deps := map[string]bool{}
	for line := range strings.Lines(string(out)) {
		if p := strings.TrimSpace(line); p != "" {
			deps[p] = true
		}
	}
	forbidden := []string{
		"github.com/pkieltyka/goinfer/backend/cuda",
		"github.com/pkieltyka/goinfer/decoder",
		"github.com/pkieltyka/goinfer/constrain",
		"github.com/pkieltyka/goinfer/decoder/config",
		"github.com/pkieltyka/goinfer/decoder/fit",
		"github.com/pkieltyka/goinfer/decoder/resident",
		"github.com/pkieltyka/goinfer/internal/cost",
		"github.com/pkieltyka/goinfer/internal/runtime/adapter",
		"github.com/pkieltyka/goinfer/internal/runtime/diskfree",
		"github.com/pkieltyka/goinfer/internal/runtime/giw",
		"github.com/pkieltyka/goinfer/internal/runtime/hybridattn",
		"github.com/pkieltyka/goinfer/internal/runtime/kvmgr",
		"github.com/pkieltyka/goinfer/internal/runtime/logs",
		"github.com/pkieltyka/goinfer/internal/runtime/mamba2",
		"github.com/pkieltyka/goinfer/internal/runtime/program",
		"github.com/pkieltyka/goinfer/internal/runtime/sampling",
	}
	for _, pkg := range forbidden {
		if deps[pkg] {
			t.Errorf("backend/cuda/eg2 transitively imports forbidden package %s", pkg)
		}
	}
}

// TestEG2Leaf_exportsNothing verifies that backend/cuda/eg2 is a side-effect-only
// package with no exported identifiers.
func TestEG2Leaf_exportsNothing(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		af, err := parser.ParseFile(fset, f, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range af.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if d.Recv == nil && ast.IsExported(d.Name.Name) {
					t.Errorf("%s exports func %s", f, d.Name.Name)
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(s.Name.Name) {
							t.Errorf("%s exports type %s", f, s.Name.Name)
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if ast.IsExported(name.Name) {
								t.Errorf("%s exports value %s", f, name.Name)
							}
						}
					}
				}
			}
		}
	}
}

var (
	wantEG2CombinedSelector bool
	wantEG2FP16Selector     bool
	initEG2Accelerators     []string
)

func init() {
	initEG2Accelerators = embeddinggemma2.Accelerators()
}

// TestEG2Leaf_registryMatchesTags verifies that importing backend/cuda/eg2 registers
// the exact expected CUDA selectors for the active build tags.
func TestEG2Leaf_registryMatchesTags(t *testing.T) {
	want := []string{"cuda"}
	if wantEG2CombinedSelector {
		want = append(want, "cuda-eg2-combined")
	}
	if wantEG2FP16Selector {
		want = append(want, "cuda-eg2-fp16")
	}
	slices.Sort(want)
	if !slices.Equal(initEG2Accelerators, want) {
		t.Fatalf("Accelerators() = %v, want %v", initEG2Accelerators, want)
	}
}

// TestEG2Leaf_movedAssetDigests pins the exact pre-move SHA-256 digests of the
// relocated PTX and sm_120 cubin artifacts.
func TestEG2Leaf_movedAssetDigests(t *testing.T) {
	want := map[string]string{
		"../internal/tower/testdata/tower_base.ptx": "7ee3fdb069e37cd3a1f00364ac1ae8ade5f346de1af92f91ce51d7e51904c4c6",
		"testdata/eg2_ffn_fp16.ptx":                 "4211568d2b58019d7375bbec6a6e4f2560f2640a4d4808929bbfde352ccdd358",
		"testdata/eg2_fusions.ptx":                  "446e34045eeafac5e229894d90f429379d0d4c8c1768c56515e1ae6210198c62",
		"testdata/eg2_packed_attn.ptx":              "d0ea4dac4c894d51c7df5e5632c69a7303a1b99dfb41e63d2d24ac3412ec01fa",
		"testdata/eg2_transposed_experiment.ptx":    "b965898a6eed3233e7cef13298e8d29517d9efbd6e50d2b7faffdda6cfa181a4",
		"testdata/eg2-sm120/vit.sm120.cubin":        "615d6a7738f404f83f9480ebbc19023b030989f0e3aa6f2b99c3c3228f37449b",
		"testdata/eg2-sm120/base.sm120.cubin":       "55f647fb589b54f3f917a1894158475331c2d21efeecc15f8439aa5673d4a9b1",
		"testdata/eg2-sm120/transposed.sm120.cubin": "1b61c884935704fdcdc19075134f6f0232e400af0355b1881fd37851f6b6814f",
		"testdata/eg2-sm120/fusion.sm120.cubin":     "93693f491ba1483c6969d5bab67845e7645aa03627e7604c2b1a541fac55c714",
		"testdata/eg2-sm120/ffn.sm120.cubin":        "91dc9d7b53111a8c0ad9575c3067f29225b782fce24a40e591c2cedf8c0948a0",
		"testdata/eg2-sm120/packed.sm120.cubin":     "26f456727fdcda465cc31f3f3846e33ae281eb29913fab881acd4186e363d7d4",
	}
	for path, expected := range want {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(b)); got != expected {
			t.Errorf("%s: SHA-256 %s, want %s", path, got, expected)
		}
	}
}

// TestBackendCUDA_noDuplicateEG2Registration ensures EmbeddingGemma 2 registration
// happens exclusively in backend/cuda/eg2 (which backend/cuda blank-imports).
func TestBackendCUDA_noDuplicateEG2Registration(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("..", "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) < 10 {
		t.Fatalf("expected backend/cuda non-test files, found %d", len(files))
	}
	forbidden := []string{
		"embeddinggemma2.RegisterAccelerator",
		"embeddinggemma2.RegisterDeviceAcceleratorWithSupport",
		"embeddinggemma2.RegisterDeviceProber",
	}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		text := string(b)
		for _, fn := range forbidden {
			if strings.Contains(text, fn) {
				t.Errorf("%s calls %s directly; EG2 registration must live only in backend/cuda/eg2", f, fn)
			}
		}
	}
}
