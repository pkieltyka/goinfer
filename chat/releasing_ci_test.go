package chat

import (
	"os"
	"regexp"
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
		`GOFLAGS: "-mod=readonly"`,
		`GOPROXY: "https://proxy.golang.org"`,
		`GOSUMDB: "sum.golang.org"`,
	} {
		if !strings.Contains(src, want) {
			t.Errorf("standalone-build.yml lacks %q", want)
		}
	}
	for _, route := range []struct{ module, tag string }{
		{"root", "v*"},
		{"backend/cuda", "backend/cuda/v*"},
		{"backend/metal", "backend/metal/v*"},
		{"backend/vulkan", "backend/vulkan/v*"},
	} {
		if !strings.Contains(src, "- '"+route.tag+"'") {
			t.Errorf("standalone-build.yml has no trigger for %s", route.tag)
		}
		arm := regexp.MustCompile(`(?m)^[ \t]*` + regexp.QuoteMeta("refs/tags/"+route.tag+")") +
			`[ \t]+` + regexp.QuoteMeta("modules=("+route.module+")") + `[ \t]*;;[ \t]*$`)
		if !arm.MatchString(src) {
			t.Errorf("standalone-build.yml must route %s to %s only", route.tag, route.module)
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
