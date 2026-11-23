//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// TestPTXModules_coverEveryEmbed checks the census list against the main and opt-in production embeds, so
// the census's "moe_route declares the maximum local memory" precondition is checked against every module (a
// hand-written list had left gptoss_act.ptx, the gpt-oss expert-cache path, out). TestKernelFMALint_coversEmbeddedPTX
// closes the same gap for the FMA lint.
func TestPTXModules_coverEveryEmbed(t *testing.T) {
	if !validCensusPTXTarget("argmax.ptx", 75) || validCensusPTXTarget("argmax.ptx", 80) ||
		!validCensusPTXTarget("eg2_ffn_fp16.ptx", 80) || validCensusPTXTarget("eg2_ffn_fp16.ptx", 75) {
		t.Fatal("production sm_75 targets and the FP16 sm_80 exception must be checked independently")
	}
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	embedded := map[string]bool{}
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		kb, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range regexp.MustCompile(`//go:embed testdata/([A-Za-z0-9_]+\.ptx)`).FindAllStringSubmatch(string(kb), -1) {
			embedded[m[1]] = true
		}
	}
	if len(embedded) < 10 || !embedded["eg2_ffn_fp16.ptx"] {
		t.Fatalf("found %d embedded PTX in production embed files", len(embedded))
	}
	listed := map[string]bool{}
	for _, m := range ptxModules(t) {
		listed[m.name] = true
		if m.name == "eg2_ffn_fp16.ptx" && m.minSM != 80 {
			t.Errorf("FP16 FFN module requires SM%d, want SM80", m.minSM)
		}
	}
	var missing, stale []string
	for f := range embedded {
		if !listed[f] {
			missing = append(missing, f)
		}
	}
	for f := range listed {
		if !embedded[f] {
			stale = append(stale, f)
		}
	}
	sort.Strings(missing)
	sort.Strings(stale)
	if len(missing) > 0 {
		t.Errorf("ptxModules() omits %d embedded module(s): %s — the local-memory census cannot see them",
			len(missing), strings.Join(missing, " "))
	}
	if len(stale) > 0 {
		t.Errorf("ptxModules() lists %d module(s) not found in the expected embed files: %s", len(stale), strings.Join(stale, " "))
	}
	t.Logf("%d embedded modules, %d in ptxModules()", len(embedded), len(listed))
}
