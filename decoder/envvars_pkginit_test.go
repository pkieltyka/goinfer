package decoder

import (
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

// readsEnvAtInit reports whether e, a package-level initialiser, reads the environment when the package initialises: a
// call to os.Getenv or os.LookupEnv, or a GOINFER_ literal passed to any call (a helper that reads it). A function
// literal's body runs later, so it is not looked into.
func readsEnvAtInit(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.CallExpr:
			if sel, ok := n.Fun.(*ast.SelectorExpr); ok {
				if x, ok := sel.X.(*ast.Ident); ok && x.Name == "os" && (sel.Sel.Name == "Getenv" || sel.Sel.Name == "LookupEnv") {
					found = true
				}
			}
			for _, a := range n.Args {
				if b, ok := a.(*ast.BasicLit); ok && b.Kind == token.STRING {
					if s, err := strconv.Unquote(b.Value); err == nil && strings.HasPrefix(s, "GOINFER_") {
						found = true
					}
				}
			}
		}
		return !found
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
		f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		for _, decl := range f.Decls {
			g, ok := decl.(*ast.GenDecl)
			if !ok || g.Tok != token.VAR {
				continue
			}
			for _, spec := range g.Specs {
				vs := spec.(*ast.ValueSpec)
				for i, v := range vs.Values {
					if readsEnvAtInit(v) {
						name := vs.Names[min(i, len(vs.Names)-1)].Name
						out = append(out, filepath.ToSlash(rel)+":"+name)
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
