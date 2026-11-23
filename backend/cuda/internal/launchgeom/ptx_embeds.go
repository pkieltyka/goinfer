package launchgeom

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// EmbeddedPTXModules returns the PTX basenames declared by production Go sources.
// It includes opt-in files whose build tags may be off in the checking binary.
func EmbeddedPTXModules(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]bool)
	var modules []string
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		for line := range strings.Lines(string(data)) {
			fields := strings.Fields(line)
			if len(fields) < 2 || fields[0] != "//go:embed" {
				continue
			}
			for _, path := range fields[1:] {
				name, ok := strings.CutPrefix(path, "testdata/")
				if ok && strings.HasSuffix(name, ".ptx") && !seen[name] {
					seen[name] = true
					modules = append(modules, name)
				}
			}
		}
	}
	slices.Sort(modules)
	return modules, nil
}
