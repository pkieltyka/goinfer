package decoder

import (
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/pkieltyka/goinfer/internal/modelcore/linalg"
)

// Hardware-coverage H1.3: CI builds this package with modelcore_noavx512 / modelcore_noavx2 / modelcore_nopopcnt / modelcore_nodotprod tags, which make the
// dispatchers take the narrower kernels on whatever CPU the runner has. forcedFallbacks reports which the test binary was built with (nil normally).
func forcedFallbacks() []string { return linalg.ForcedFallbacks() }

// TestForcedFallbacks_expected is the guard against a job that passes -tags and forces nothing: a tag that matches no file compiles and runs green. The job sets GOINFER_EXPECT_FORCED_CPU to the fallbacks
// it asked for ("noavx2", "none", ...) and this fails unless linalg.ForcedFallbacks says the same. Unset it checks nothing (an ordinary run) and says so, and a SKIP is not a pass: the job's own log
// must show --- PASS for this test.
func TestForcedFallbacks_expected(t *testing.T) {
	want, set := os.LookupEnv("GOINFER_EXPECT_FORCED_CPU")
	if !set {
		t.Skip("GOINFER_EXPECT_FORCED_CPU unset: not asserting which CPU fallbacks this build forces")
	}
	var wantList []string
	if want != "none" && want != "" {
		wantList = strings.Split(want, ",")
	}
	got := forcedFallbacks()
	slices.Sort(wantList)
	slices.Sort(got)
	if !slices.Equal(got, wantList) {
		t.Fatalf("linalg.ForcedFallbacks() = %v, the job expected %v: a -tags value that matched no file forces nothing and would pass silently", got, wantList)
	}
}

// Forced-fallback floors. Under a forced narrower kernel the numerics are a different realization of the same
// arithmetic, so the goldens that pin ONE path's token sequence or sample values (recorded on the default path; the
// int4 goldens are arm64-baked) cannot be compared exactly. These replace them under the tags with a closeness bound,
// chosen between the measured healthy value and what a real
// defect does, each shown red by a mutation.
const (
	// forcedInt4CosFloor: centered cosine of an int4 fixture's logit samples against its recorded golden. The bar sits
	// between the healthy forced-noavx2 values (lowest on gemma4-dense-scaled, the fixture built to amplify numeric
	// differences; argmax equal) and what a group-size mutation does (int4_golden_test.go).
	forcedInt4CosFloor = 0.99
	// forcedLogitCloseness: 1 - cosine between two paths' logits for the same ids, where the pair differs by design: int4
	// against int8int8, and the f32 fast-attention prefill against the exact path. The bound sits above the deficit of
	// both, pure Go and AVX2.
	forcedLogitCloseness = 3e-2
)

func cosineDistance(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return 1 - d/(math.Sqrt(na)*math.Sqrt(nb))
}
