package chat

import (
	"os"
	"strings"
	"testing"
)

// Check that shipped backend modules can build in module mode.
func TestReleasing_standaloneBuildGateExists(t *testing.T) {
	wf, err := os.ReadFile("../.github/workflows/standalone-build.yml")
	if err != nil {
		t.Fatalf("RELEASING.md relies on a standalone-build workflow that is not here: %v", err)
	}
	src := string(wf)
	for _, want := range []string{
		`GOWORK: "off"`, // the point of the gate
		"tags:",         // triggered by tag pushes, not every push
	} {
		if !strings.Contains(src, want) {
			t.Errorf("standalone-build.yml lacks %q", want)
		}
	}
	// Every shipped submodule must be covered, or the gate proves less than the doc claims.
	for _, mod := range []string{"backend/cuda", "backend/metal", "backend/vulkan"} {
		if !strings.Contains(src, "module: "+mod) {
			t.Errorf("standalone-build.yml does not cover the %s module", mod)
		}
	}
	// It must NOT create a workspace — that would silently defeat the whole check. Comment
	// lines are skipped: the workflow EXPLAINS why it does not run `go work init`, and matching
	// that explanation is the "a check that matches its own comment" trap.
	for ln := range strings.SplitSeq(src, "\n") {
		if strings.HasPrefix(strings.TrimSpace(ln), "#") {
			continue
		}
		if strings.Contains(ln, "go work init") {
			t.Errorf("standalone-build.yml runs `go work init` (%s): it would then prove the "+
				"opposite of what RELEASING.md claims, since a workspace is exactly what the "+
				"shipped tag must not need", strings.TrimSpace(ln))
		}
	}

}
