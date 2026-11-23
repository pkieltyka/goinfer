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
func readsEnvAtInit(e ast.Expr, scope *pkgScope) bool {
	return readsEnvAtInitDepth(e, scope, map[string]bool{}, 0)
}

// pkgScope is what an initialiser can reach inside its own package: free functions by name, methods by
// name (a receiver type is hard to resolve statically, so a name match is followed conservatively), and
// the package names its files import, which must not be mistaken for local methods.
type pkgScope struct {
	funcs   map[string]*ast.BlockStmt
	methods map[string]map[string]*ast.BlockStmt // receiver type name -> method name -> body
	imports map[string]bool
}

func readsEnvAtInitDepth(e ast.Node, scope *pkgScope, seen map[string]bool, depth int) bool {
	if e == nil {
		return false
	}
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if found {
			return false
		}
		switch n := n.(type) {
		case *ast.FuncLit:
			// A literal that is merely stored runs later, so its body is not init work. An immediately
			// invoked one is handled from its CallExpr below, where the body does run synchronously.
			return false
		case *ast.CallExpr:
			if lit, ok := n.Fun.(*ast.FuncLit); ok {
				// `func(){ os.Getenv(...) }()` runs during the initialiser, so follow its body.
				if readsEnvAtInitDepth(lit.Body, scope, seen, depth+1) {
					found = true
					return false
				}
			}
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
				if body, ok := scope.funcs[id.Name]; ok && !seen[id.Name] {
					seen[id.Name] = true
					if readsEnvAtInitDepth(body, scope, seen, depth+1) {
						found = true
						return false
					}
				}
			}
			// A method call on an obvious concrete receiver: `(&settings{}).read()` or `settings{}.read()`.
			// Only that shape is resolved — a stored method value, an interface call, or a selector on an
			// imported package is not followed, so the scan cannot invent a call edge that does not exist.
			if sel, ok := n.Fun.(*ast.SelectorExpr); ok {
				if recv := compositeReceiverType(sel.X); recv != "" {
					if body, ok := scope.methods[recv][sel.Sel.Name]; ok {
						key := "method:" + recv + "." + sel.Sel.Name
						if !seen[key] {
							seen[key] = true
							if readsEnvAtInitDepth(body, scope, seen, depth+1) {
								found = true
								return false
							}
						}
					}
				}
			}
		}
		return true
	})
	return found
}

// recvTypeName is a method's receiver type name, `T` or `*T` alike; "" when it is not a local identifier.
func recvTypeName(fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return ""
	}
	t := fd.Recv.List[0].Type
	if star, ok := t.(*ast.StarExpr); ok {
		t = star.X
	}
	if id, ok := t.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

// compositeReceiverType is the type name of a receiver written as a composite literal, `T{}` or `(&T{})`,
// and "" for anything else: a variable, an interface, or an imported package's identifier.
func compositeReceiverType(x ast.Expr) string {
	switch e := x.(type) {
	case *ast.CompositeLit:
		if id, ok := e.Type.(*ast.Ident); ok {
			return id.Name
		}
	case *ast.ParenExpr:
		return compositeReceiverType(e.X)
	case *ast.UnaryExpr:
		if e.Op == token.AND {
			return compositeReceiverType(e.X)
		}
	}
	return ""
}

// pkgInitEnvReads lists file:name for every production package initialiser that reads the environment: a
// package-level variable whose initialiser does, or a package `init` function whose body does. It walks
// directories, not files, and parses each directory once — the decoder package alone holds hundreds of
// files, and re-parsing the whole directory per file made this test quadratic.
func pkgInitEnvReads(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			return nil
		}
		switch d.Name() {
		case "testdata", "node_modules", ".git", ".claude":
			return filepath.SkipDir
		}
		ents, err := os.ReadDir(p)
		if err != nil {
			return err
		}
		type pkgFile struct {
			rel string
			f   *ast.File
		}
		type pkgInit struct {
			rel  string
			body *ast.BlockStmt
		}
		var files []pkgFile
		var inits []pkgInit
		scope := &pkgScope{funcs: map[string]*ast.BlockStmt{}, methods: map[string]map[string]*ast.BlockStmt{}, imports: map[string]bool{}}
		for _, ent := range ents {
			n := ent.Name()
			if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
				continue
			}
			full := filepath.Join(p, n)
			af, err := parser.ParseFile(token.NewFileSet(), full, nil, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(root, full)
			files = append(files, pkgFile{rel: rel, f: af})
			for _, imp := range af.Imports {
				if path, err := strconv.Unquote(imp.Path.Value); err == nil {
					name := path[strings.LastIndex(path, "/")+1:]
					if imp.Name != nil {
						name = imp.Name.Name
					}
					scope.imports[name] = true
				}
			}
			for _, decl := range af.Decls {
				fd, ok := decl.(*ast.FuncDecl)
				if !ok || fd.Body == nil {
					continue
				}
				if fd.Recv != nil {
					if recv := recvTypeName(fd); recv != "" {
						if scope.methods[recv] == nil {
							scope.methods[recv] = map[string]*ast.BlockStmt{}
						}
						scope.methods[recv][fd.Name.Name] = fd.Body
					}
					continue
				}
				if fd.Name.Name == "init" {
					inits = append(inits, pkgInit{rel: rel, body: fd.Body})
					continue
				}
				scope.funcs[fd.Name.Name] = fd.Body
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
						if readsEnvAtInit(v, scope) {
							name := vs.Names[min(i, len(vs.Names)-1)].Name
							out = append(out, filepath.ToSlash(pf.rel)+":"+name)
						}
					}
				}
			}
		}
		for _, in := range inits {
			if readsEnvAtInitDepth(in.body, scope, map[string]bool{}, 0) {
				out = append(out, filepath.ToSlash(in.rel)+":init")
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
	// An init FUNCTION body is a root too: `func init(){ read() }` runs before main exactly as a variable
	// initialiser does, and the linked gate only sees a direct Getenv inside decoder.init, not a call.
	initPlant := filepath.Join(t.TempDir(), "initfn")
	if err := os.MkdirAll(initPlant, 0o755); err != nil {
		t.Fatal(err)
	}
	srcInit := "package p\n\nimport \"os\"\n\nfunc read() string { return os.Getenv(\"GOINFER_HIDDEN\") }\n\nfunc init() { _ = read() }\n"
	if err := os.WriteFile(filepath.Join(initPlant, "p.go"), []byte(srcInit), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(pkgInitEnvReads(t, initPlant), ","); got != "p.go:init" {
		t.Fatalf("VOID: the scan misses an environment read in a package init function: found %q", got)
	}
	// An immediately invoked function literal runs inside the initialiser; a stored one does not.
	iife := filepath.Join(t.TempDir(), "iife")
	if err := os.MkdirAll(iife, 0o755); err != nil {
		t.Fatal(err)
	}
	srcIIFE := "package p\n\nimport \"os\"\n\nvar viaIIFE = func() string { return os.Getenv(\"GOINFER_IIFE\") }()\n"
	if err := os.WriteFile(filepath.Join(iife, "p.go"), []byte(srcIIFE), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(pkgInitEnvReads(t, iife), ","); got != "p.go:viaIIFE" {
		t.Fatalf("VOID: the scan misses an environment read inside an immediately invoked literal: found %q", got)
	}
	stored := filepath.Join(t.TempDir(), "stored")
	if err := os.MkdirAll(stored, 0o755); err != nil {
		t.Fatal(err)
	}
	srcStored := "package p\n\nimport \"os\"\n\nvar later = func() string { return os.Getenv(\"GOINFER_LATER\") }\n"
	if err := os.WriteFile(filepath.Join(stored, "p.go"), []byte(srcStored), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := pkgInitEnvReads(t, stored); len(got) != 0 {
		t.Fatalf("a stored literal runs later, not at init, but the scan reported %q", got)
	}
	// A receiver method reached from an initialiser: `var snapshot = (&settings{}).read()` has no direct
	// env call in its init expression, and the method body is only reachable through the call edge.
	method := filepath.Join(t.TempDir(), "method")
	if err := os.MkdirAll(method, 0o755); err != nil {
		t.Fatal(err)
	}
	srcMethod := "package p\n\nimport \"os\"\n\ntype settings struct{}\n\nfunc (*settings) read() string { return os.Getenv(\"GOINFER_METHOD\") }\n\nvar snapshot = (&settings{}).read()\n"
	if err := os.WriteFile(filepath.Join(method, "p.go"), []byte(srcMethod), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(pkgInitEnvReads(t, method), ","); got != "p.go:snapshot" {
		t.Fatalf("VOID: the scan misses an environment read behind a receiver method: found %q", got)
	}
	// A call through an interface, or a stored method value, is not a concrete receiver: the scan must not
	// follow it just because a method of that name exists in the package.
	iface := filepath.Join(t.TempDir(), "iface")
	if err := os.MkdirAll(iface, 0o755); err != nil {
		t.Fatal(err)
	}
	srcIface := "package p\n\nimport \"os\"\n\ntype settings struct{}\n\nfunc (*settings) read() string { return os.Getenv(\"GOINFER_IFACE\") }\n\ntype reader interface{ read() string }\n\nfunc later(r reader) string { return r.read() }\n\nvar viaIface = later(&settings{})\n"
	if err := os.WriteFile(filepath.Join(iface, "p.go"), []byte(srcIface), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := pkgInitEnvReads(t, iface); len(got) != 0 {
		t.Fatalf("an interface call is not an obvious concrete receiver, but the scan followed it: %q", got)
	}
	// A method named like an imported package's function, `strings.Cut`, is not a local method.
	pkgmethod := filepath.Join(t.TempDir(), "pkgmethod")
	if err := os.MkdirAll(pkgmethod, 0o755); err != nil {
		t.Fatal(err)
	}
	srcPkgMethod := "package p\n\nimport \"strings\"\n\nvar viaPkg = strings.Cut(\"a\", \"b\")\n"
	if err := os.WriteFile(filepath.Join(pkgmethod, "p.go"), []byte(srcPkgMethod), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := pkgInitEnvReads(t, pkgmethod); len(got) != 0 {
		t.Fatalf("an imported package's selector must not be followed as a local method: found %q", got)
	}
	// And a long helper chain: the cycle guard bounds recursion, so no depth cap may hide the ninth hop.
	chain := filepath.Join(t.TempDir(), "chain")
	if err := os.MkdirAll(chain, 0o755); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	b.WriteString("package p\n\nimport \"os\"\n\n")
	for i := 1; i <= 9; i++ {
		if i == 9 {
			b.WriteString("func h9() string { return os.Getenv(\"GOINFER_CHAIN\") }\n")
			continue
		}
		b.WriteString("func h" + strconv.Itoa(i) + "() string { return h" + strconv.Itoa(i+1) + "() }\n")
	}
	b.WriteString("\nvar viaChain = h1()\n")
	if err := os.WriteFile(filepath.Join(chain, "p.go"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(pkgInitEnvReads(t, chain), ","); got != "p.go:viaChain" {
		t.Fatalf("VOID: the scan stops before the ninth helper: found %q", got)
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
	// A real public first-use route, with an invalid path so no fixture is needed: LoadWeights snapshots
	// before it touches the filesystem, which is the boundary the constructor fixes implement.
	if _, err := LoadWeights(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Fatal("LoadWeights on an absent directory returned no error")
	}
	if !decodeTiming {
		t.Fatalf("the child did not take GOINFER_DECODE_TIMING through LoadWeights: decodeTiming=%v", decodeTiming)
	}
	if w4a8BatchEnabled {
		t.Fatalf("the child did not take GOINFER_W4A8_BATCH=0: w4a8BatchEnabled=%v", w4a8BatchEnabled)
	}
}
