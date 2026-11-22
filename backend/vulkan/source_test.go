package vulkan

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestVulkanSourceDoesNotReadEnvironmentOrPrint(t *testing.T) {
	set := token.NewFileSet()
	err := filepath.WalkDir(".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return err
		}
		file, err := parser.ParseFile(set, name, nil, 0)
		if err != nil {
			return err
		}
		imports := make(map[string]string)
		for _, item := range file.Imports {
			p, err := strconv.Unquote(item.Path.Value)
			if err != nil {
				return err
			}
			switch p {
			case "os", "fmt", "log", "log/slog":
				local := path.Base(p)
				if item.Name != nil {
					local = item.Name.Name
				}
				if local == "." {
					t.Errorf("%s: cannot check dot-imported process I/O package %s", name, p)
				}
				imports[local] = p
			}
		}
		ast.Inspect(file, func(node ast.Node) bool {
			sel, ok := node.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			pkg, ok := sel.X.(*ast.Ident)
			if !ok {
				return true
			}
			p, member := imports[pkg.Name], sel.Sel.Name
			if p == "log" || p == "log/slog" ||
				p == "fmt" && strings.HasPrefix(member, "Print") ||
				p == "os" && (member == "Getenv" || member == "LookupEnv" || member == "Stdout" || member == "Stderr") {
				t.Errorf("%s: production Vulkan uses %s.%s", name, p, member)
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
