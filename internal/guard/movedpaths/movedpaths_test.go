// Package movedpaths guards against reintroducing retired module paths.
package movedpaths

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

var retiredModules = []string{"github.com/pkieltyka/aikit", "github.com/townsendmerino/aikit"}

func retired(path string) bool {
	for _, name := range retiredModules {
		if path == name || strings.HasPrefix(path, name+"/") {
			return true
		}
	}
	return false
}

func retiredMetadataLine(line string) bool {
	line, _, _ = strings.Cut(line, "//")
	for _, field := range strings.Fields(line) {
		if retired(strings.Trim(field, "\"`")) {
			return true
		}
	}
	return false
}

func TestRetiredModuleMatchesPaths(t *testing.T) {
	for _, path := range []string{retiredModules[0], retiredModules[1] + "/gpu", retiredModules[0] + "/linalg"} {
		if !retired(path) {
			t.Errorf("missed import %s", path)
		}
	}
	for _, line := range []string{
		"require " + retiredModules[0] + " v1.2.3 // indirect",
		"replace old.example/pkg => " + retiredModules[1] + "/gpu v1.2.3",
		"replace \"" + retiredModules[0] + "\" => ./local",
		retiredModules[0] + " v1.2.3/go.mod h1:abc",
	} {
		if !retiredMetadataLine(line) {
			t.Errorf("missed metadata %q", line)
		}
	}
	for _, line := range []string{"// require " + retiredModules[0], "require " + retiredModules[0] + "-other v1.2.3"} {
		if retiredMetadataLine(line) {
			t.Errorf("unrelated metadata matched %q", line)
		}
	}
}

// TestNoRetiredModule scans imports in test files and under inactive build tags.
// It also checks require, replace, and checksum paths in every module.
func TestNoRetiredModule(t *testing.T) {
	var goFiles, modFiles int
	err := filepath.WalkDir("../../..", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", ".claude", "docs", "testdata", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		switch name := d.Name(); {
		case strings.HasSuffix(name, ".go"):
			goFiles++
			f, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
			if err != nil {
				return err
			}
			for _, im := range f.Imports {
				if p, _ := strconv.Unquote(im.Path.Value); retired(p) {
					t.Errorf("%s imports retired module path %s", path, p)
				}
			}
		case name == "go.mod" || name == "go.sum" || name == "go.work.example":
			modFiles++
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			for i, line := range strings.Split(string(b), "\n") {
				if retiredMetadataLine(line) {
					t.Errorf("%s:%d contains a retired module path: %s", path, i+1, strings.TrimSpace(line))
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if goFiles < 1000 || modFiles < 10 {
		t.Fatalf("scanned %d Go files and %d module files", goFiles, modFiles)
	}
	t.Logf("scanned %d Go files and %d module files", goFiles, modFiles)
}
