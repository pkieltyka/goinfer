package serveapp

import (
	"flag"
	"fmt"
	"runtime"
	"strings"

	"github.com/pkieltyka/goinfer/decoder"
	"github.com/pkieltyka/goinfer/internal/app/cliutil"
)

// injectedVersion is set via `-ldflags -X` on a release-built binary, because the release workflow's own path stamps
// a dirty pseudo-version: its `go mod edit -replace` on the ephemeral submodule checkout is an uncommitted go.mod
// edit, which alone makes the VCS stamp read "modified". A binary built the ordinary way (`go install
// .../cmd/serve@vX.Y.Z`, no replace, no ephemeral checkout) is unaffected and reports its real tag from
// cliutil.BuildIdent with no injection at all.
var injectedVersion string

// versionReport is what `serve --version` prints. Its load-bearing line is `backends:`, the list of backends compiled
// into this binary, which is not the list --backend accepts: a binary that links no backend runs `--backend metal` on the
// CPU, with only a warning line that scrolls past before the banner. The line can be read without loading a model, and
// the release workflow greps it to prove each asset carries the backend for its platform.
func versionReport(prog string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s %s\n", cliutil.BinaryName(prog), buildVersion())
	// Space-separated, one line, lowercase: greppable from a shell without jq.
	fmt.Fprintf(&b, "backends: %s\n", strings.Join(decoder.CompiledBackends(), " "))
	fmt.Fprintf(&b, "go: %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	return b.String()
}

func buildVersion() string { return cliutil.VersionString(injectedVersion) }

// isVersionArg matches the forms a user actually types. `version` (no dashes) is included
// because `serve check` and `serve pull` are subcommands, so a bare word is the shape this
// binary has taught people to expect.
func isVersionArg(a string) bool {
	switch a {
	case "--version", "-version", "version":
		return true
	}
	return false
}

// countFlags reports how many flags are registered, so the help header's "all N flags" line cannot drift from reality
// the way a hand-typed count would.
func countFlags() int {
	n := 0
	flag.VisitAll(func(*flag.Flag) { n++ })
	return n
}
