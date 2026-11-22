package decoder

import (
	"os"
	"regexp"
	"strconv"
	"testing"
)

// TestResidentMoECap_pinsKernelConstant gates M-17: the feature-matrix MoE caps in residentBackendMoECap MUST equal
// the router kernel's actual fixed-array bounds, or the hardware-matrix generator declines archs BuildResident admits
// (or vice versa). It extracts each constant from the kernel source (e.g. backend/cuda/moe.cu's MOE_MAX_G) and pins the cap
// to it; a kernel bump that forgets the map fails here.
func TestResidentMoECap_pinsKernelConstant(t *testing.T) {
	grep := func(path, pat string) int {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("kernel source %s unavailable: %v", path, err)
		}
		m := regexp.MustCompile(pat).FindSubmatch(b)
		if m == nil {
			t.Fatalf("%s: pattern %q not found (kernel refactored? update this pin)", path, pat)
		}
		n, _ := strconv.Atoi(string(m[1]))
		return n
	}
	// cuda: #define MOE_MAX_E 256 / MOE_MAX_G 64
	cudaE := grep("../backend/cuda/moe.cu", `#define MOE_MAX_E (\d+)`)
	cudaG := grep("../backend/cuda/moe.cu", `#define MOE_MAX_G (\d+)`)
	if got := residentBackendMoECap["cuda"]; got.experts != cudaE || got.groups != cudaG {
		t.Errorf("cuda cap %+v != kernel (experts %d, groups %d) — one-source-of-truth broken (M-17)", got, cudaE, cudaG)
	}
}

// The declared webgpu cap must actually admit the models the published matrices call resident.
// Without this, raising the kernel and the map together while some third place still says 256
// would go unnoticed — which is the shape of M-31 itself.
func TestResidentMoECap_admitsThePublishedModels(t *testing.T) {
	e, g, ok := ResidentBackendMoECap("webgpu")
	if !ok {
		t.Fatal("webgpu has no declared MoE cap")
	}
	// Kimi-K2 and DeepSeek-V4-Pro: 384 experts, both listed "✅ resident" for webgpu.
	if e < 384 {
		t.Errorf("webgpu expert cap %d < 384: docs/hardware-matrix.md and "+
			"docs/capability-matrix.md publish Kimi-K2 / DeepSeek-V4-Pro as resident", e)
	}
	if g < 32 {
		t.Errorf("webgpu group cap %d < 32", g)
	}
}
