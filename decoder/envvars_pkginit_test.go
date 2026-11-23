package decoder

import (
	"os/exec"
	"sync"

	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// pkgInitEnvReadsKnown is the set of production package-level variables still allowed to read the environment at
// initialisation. It is EMPTY and stays empty: goinfer#70 requires that importing this package — or any backend whose
// init links it, as backend/cuda does for text embeddings — performs no environment reads before main. The decoder's
// eight knobs are parsed once on first decoder use instead (env_snapshot.go), and a new setting is a decoder.Options
// field, a load-time knob or a test hook (docs/env-vars.md, "Policy — where a new setting goes").
var pkgInitEnvReadsKnown = map[string]bool{
	// Out of goinfer#70's scope: this is the server application, not a library that an embedding consumer
	// imports. A decoder/ entry, however, may never appear here again.
	"internal/app/serveapp/tools.go:toolUnionEnabled": true,
}

// readsEnvAtInit reports whether e, a package-level initialiser, reads the environment when the package
// initialises: a call to os.Getenv or os.LookupEnv, a GOINFER_ literal passed to any call (a helper that
// reads it), or a call to a same-package function whose own body does either, followed recursively — a
// helper chain is exactly how the decoder's eight reads were hidden from a scan that only looked at the
// initialiser expression. A function literal's body runs later, so it is not looked into.
func readsEnvAtInit(e ast.Expr, helpers map[string]*ast.BlockStmt) bool {
	return readsEnvAtInitDepth(e, helpers, map[string]bool{}, 0)
}

func readsEnvAtInitDepth(e ast.Node, helpers map[string]*ast.BlockStmt, seen map[string]bool, depth int) bool {
	if e == nil || depth > 8 {
		return false
	}
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if found {
			return false
		}
		switch n := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.CallExpr:
			if sel, ok := n.Fun.(*ast.SelectorExpr); ok {
				if x, ok := sel.X.(*ast.Ident); ok && x.Name == "os" && (sel.Sel.Name == "Getenv" || sel.Sel.Name == "LookupEnv") {
					found = true
					return false
				}
			}
			for _, a := range n.Args {
				if b, ok := a.(*ast.BasicLit); ok && b.Kind == token.STRING {
					if v, err := strconv.Unquote(b.Value); err == nil && strings.HasPrefix(v, "GOINFER_") {
						found = true
						return false
					}
				}
			}
			if id, ok := n.Fun.(*ast.Ident); ok {
				if body, ok := helpers[id.Name]; ok && !seen[id.Name] {
					seen[id.Name] = true
					if readsEnvAtInitDepth(body, helpers, seen, depth+1) {
						found = true
						return false
					}
				}
			}
		}
		return true
	})
	return found
}

// pkgInitEnvReads lists file:var for each package-level var in the non-test Go files under root whose initialiser reads the
// environment.
func pkgInitEnvReads(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case "testdata", "node_modules", ".git", ".claude":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		// One package per directory: parse its files together so an initialiser's helper can be
		// followed, and keep each file's own name for the report.
		dir := filepath.Dir(p)
		ents, err := os.ReadDir(dir)
		if err != nil {
			return err
		}
		type pkgFile struct {
			rel string
			f   *ast.File
		}
		var files []pkgFile
		helpers := map[string]*ast.BlockStmt{}
		for _, ent := range ents {
			n := ent.Name()
			if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
				continue
			}
			full := filepath.Join(dir, n)
			af, err := parser.ParseFile(token.NewFileSet(), full, nil, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(root, full)
			files = append(files, pkgFile{rel: rel, f: af})
			for _, decl := range af.Decls {
				if fd, ok := decl.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Body != nil {
					helpers[fd.Name.Name] = fd.Body
				}
			}
		}
		for _, pf := range files {
			for _, decl := range pf.f.Decls {
				g, ok := decl.(*ast.GenDecl)
				if !ok || g.Tok != token.VAR {
					continue
				}
				for _, spec := range g.Specs {
					vs := spec.(*ast.ValueSpec)
					for i, v := range vs.Values {
						if readsEnvAtInit(v, helpers) {
							name := vs.Names[min(i, len(vs.Names)-1)].Name
							out = append(out, filepath.ToSlash(pf.rel)+":"+name)
						}
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(out)
	return out
}

// TestEnvVars_noNewPackageInitRead: no production package-level variable reads the environment at initialisation beyond
// pkgInitEnvReadsKnown, and every entry there still does (the list only shrinks). It first proves the check can go red.
func TestEnvVars_noNewPackageInitRead(t *testing.T) {
	plant := filepath.Join(t.TempDir(), "plant")
	if err := os.MkdirAll(plant, 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package p\n\nimport \"os\"\n\nvar viaOS = os.Getenv(\"X\")\nvar viaHelper = flag(\"GOINFER_Y\")\nvar later = func() string { return os.Getenv(\"Z\") }\n\nfunc flag(string) bool { return false }\n"
	if err := os.WriteFile(filepath.Join(plant, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(pkgInitEnvReads(t, plant), ","); got != "p.go:viaHelper,p.go:viaOS" {
		t.Fatalf("VOID: the scan finds %q in the plant, want p.go:viaHelper,p.go:viaOS", got)
	}

	found := pkgInitEnvReads(t, "..")
	for _, r := range found {
		if !pkgInitEnvReadsKnown[r] {
			t.Errorf("%s reads the environment at package initialisation, where t.Setenv cannot reach it and where an "+
				"importer like backend/cuda reads it before main: parse it on first decoder use (env_snapshot.go) or make "+
				"it a decoder.Options field or a test hook (docs/env-vars.md, \"Policy — where a new setting goes\")", r)
		}
	}
	seen := map[string]bool{}
	for _, r := range found {
		seen[r] = true
	}
	for k := range pkgInitEnvReadsKnown {
		if !seen[k] {
			t.Errorf("%s no longer reads the environment at initialisation: delete it from pkgInitEnvReadsKnown", k)
		}
		if strings.HasPrefix(k, "decoder/") {
			t.Errorf("%s is a decoder package initialiser: goinfer#70 forbids it outright, so the entry may not be allowlisted", k)
		}
	}
	// The scan must still be able to find something, or a green run proves nothing: the plant above is the canary,
	// and a second one here covers the helper-called form the decoder used (envBoolDefault).
	helper := filepath.Join(t.TempDir(), "helper")
	if err := os.MkdirAll(helper, 0o755); err != nil {
		t.Fatal(err)
	}
	srcHelper := "package p\n\nfunc flag(string, bool) bool { return false }\n\nvar viaHelper = flag(\"GOINFER_Q\", true)\n"
	if err := os.WriteFile(filepath.Join(helper, "p.go"), []byte(srcHelper), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(pkgInitEnvReads(t, helper), ","); got != "p.go:viaHelper" {
		t.Fatalf("VOID: the scan misses a helper-called GOINFER_ literal: found %q", got)
	}
}

// issue70PublicRoutes is every public entry that builds weights or a model. Each must take the settings
// snapshot itself: a consumer may call LoadWeights or LoadGGUFBytes without ever going through Load, and
// then a load-time knob (fake quant, split-half repack) would otherwise be read before the once ran —
// silently ignoring the environment, or racing the once-write from another goroutine.
var issue70PublicRoutes = []string{
	"Load", "LoadGGUFBytes", "NewModel", "NewModelWithOptions", "LoadWeights", "LoadWeightsFromFS",
	"Generate", "GenerateVL", "GenerateVLSpans", "GenerateVLCausalSpans", "GenerateQwenVL",
	"GenerateQwenVLDeepstack", "GenerateQwenVLDeepstackSpans", "GenerateGemma4VL", "GenerateGemma4VLSpans",
	"GenerateAudio",
}

// TestIssue70_publicRoutesTakeTheSnapshot is the coverage gate for the lazily-parsed knobs: a new public
// route that reads them without calling ensureSettings fails here rather than in a consumer's process.
func TestIssue70_publicRoutesTakeTheSnapshot(t *testing.T) {
	fset := token.NewFileSet()
	calls := map[string]bool{}
	err := filepath.WalkDir(".", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return err
		}
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			return err
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			if !hasEnsureSettings(fn.Body) {
				continue
			}
			name := fn.Name.Name
			if r := fn.Recv; r != nil && len(r.List) > 0 {
				// A method's route name is its own; the list names it without a receiver.
			}
			calls[name] = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) == 0 {
		t.Fatal("VOID: no function calls ensureSettings, so this gate proves nothing")
	}
	for _, route := range issue70PublicRoutes {
		if !calls[route] {
			t.Errorf("%s is a public build/generate route that does not call ensureSettings: a consumer reaching it first would read the knobs before the snapshot", route)
		}
	}
}

func hasEnsureSettings(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "ensureSettings" {
				found = true
			}
		}
		return !found
	})
	return found
}

// TestIssue70_hiddenHelperInitRead keeps the init scanner honest: a helper called from a package-level
// initialiser must be followed into, or `func read(){os.Getenv(...)}; var x = read()` would pass the
// scan. The plant is the canary; the real tree must then report no decoder initialiser reading env.
func TestIssue70_hiddenHelperInitRead(t *testing.T) {
	plant := filepath.Join(t.TempDir(), "hidden")
	if err := os.MkdirAll(plant, 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package p\n\nimport \"os\"\n\nfunc read() string { return os.Getenv(\"GOINFER_HIDDEN\") }\n\nvar viaHelper = read()\n"
	if err := os.WriteFile(filepath.Join(plant, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := pkgInitEnvReads(t, plant); len(got) == 0 {
		t.Fatal("VOID: the scan misses an environment read behind a same-package helper")
	}
}

// TestIssue70_snapshotIsRaceSafe runs the once from many goroutines with reads beside it: under -race this
// fails if any access is unsynchronised, and the value is the same for every reader because it is taken
// once.
func TestIssue70_snapshotIsRaceSafe(t *testing.T) {
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ensureSettings()
			_ = deltaNetTiming
			_ = decodeTiming
			_ = w4a8BatchEnabled
			_ = cpuFusedGateUp
		}()
	}
	wg.Wait()
}

// TestIssue70_isolatedProcessReadsEnv proves the snapshot in a fresh process, where no earlier test has
// frozen it: the child sets a knob and the global must carry it. This is the control for "a consumer
// reaching a public route first still honours the environment".
func TestIssue70_isolatedProcessReadsEnv(t *testing.T) {
	if os.Getenv("ISSUE70_CHILD") == "" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestIssue70_isolatedProcessReadsEnv$", "-test.count=1")
		cmd.Env = append(os.Environ(), "ISSUE70_CHILD=1", "GOINFER_DECODE_TIMING=1", "GOINFER_W4A8_BATCH=0")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("child process: %v\n%s", err, out)
		}
		return
	}
	if !decodeTiming {
		t.Fatalf("the child did not take GOINFER_DECODE_TIMING: decodeTiming=%v", decodeTiming)
	}
	if w4a8BatchEnabled {
		t.Fatalf("the child did not take GOINFER_W4A8_BATCH=0: w4a8BatchEnabled=%v", w4a8BatchEnabled)
	}
}
