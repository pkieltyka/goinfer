package embeddinggemma2

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

const markdownReplayDir = "../testdata/embeddinggemma2-markdown-replay/replay-v1"

var markdownReplayStations = [...]string{
	"Amber Landing", "Aspen Walk", "Cedar Arch", "Clay Point", "Copper Island", "East Weir",
	"Hazel Dock", "Moss Causeway", "North Reach", "Orchard Lock", "Reed Crossing", "Silt Bend",
	"Stone Narrows", "Willow Bluff",
}

type markdownReplayRecipe struct {
	Station  int   `json:"station"`
	Sections []int `json:"sections"`
}

func markdownReplaySectionBank(raw []byte) ([]string, error) {
	if !utf8.Valid(raw) || bytes.ContainsRune(raw, '\r') || !bytes.HasSuffix(raw, []byte("\n")) {
		return nil, fmt.Errorf("section bank must be UTF-8 with LF endings")
	}
	sections := strings.Split(strings.TrimSuffix(string(raw), "\n"), "\n\n---\n")
	for i, section := range sections {
		if !strings.HasPrefix(section, "### ") || !strings.Contains(section, "\n\n") {
			return nil, fmt.Errorf("invalid heading or content for section %d", i)
		}
	}
	return sections, nil
}

func markdownReplayRender(i int, recipe markdownReplayRecipe, sections []string) (string, string, error) {
	if recipe.Station < 0 || recipe.Station >= len(markdownReplayStations) || len(recipe.Sections) == 0 {
		return "", "", fmt.Errorf("invalid recipe for watch %d", i)
	}
	id := fmt.Sprintf("watch-%03d", i+1)
	station := markdownReplayStations[recipe.Station]
	var text strings.Builder
	fmt.Fprintf(&text, "# %s\n\nWatch %03d\n", station, i+1)
	seen := make(map[int]bool, len(recipe.Sections))
	for _, n := range recipe.Sections {
		if n < 0 || n >= len(sections) || seen[n] {
			return "", "", fmt.Errorf("%s has an out-of-range or repeated section %d", id, n)
		}
		seen[n] = true
		fmt.Fprintf(&text, "\n%s\n", strings.ReplaceAll(sections[n], "{station}", station))
	}
	return id, text.String(), nil
}

func TestMarkdownReplayFixture(t *testing.T) {
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(markdownReplayDir, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	sections, err := markdownReplaySectionBank(read("sections.md"))
	if err != nil {
		t.Fatal(err)
	}
	rawRecipes := read("recipes.json")
	var recipes [][]markdownReplayRecipe
	if err := json.Unmarshal(rawRecipes, &recipes); err != nil {
		t.Fatal(err)
	}
	raw := read("corpus.md")
	var manifest struct {
		Version    int        `json:"version"`
		CorpusSHA  string     `json:"corpus_sha256"`
		BankSHA    string     `json:"section_bank_sha256"`
		RecipesSHA string     `json:"recipes_sha256"`
		Groups     [][]string `json:"groups"`
	}
	if err := json.Unmarshal(read("manifest.json"), &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Version != 1 || manifest.CorpusSHA != fmt.Sprintf("%x", sha256.Sum256(raw)) ||
		manifest.BankSHA != fmt.Sprintf("%x", sha256.Sum256(read("sections.md"))) ||
		manifest.RecipesSHA != fmt.Sprintf("%x", sha256.Sum256(rawRecipes)) {
		t.Fatal("replay version or source digest mismatch")
	}
	if len(recipes) != 55 || len(manifest.Groups) != len(recipes) {
		t.Fatal("expected 55 replay groups")
	}
	var rendered strings.Builder
	seen := make(map[[32]byte]bool)
	pos := 0
	for g, group := range recipes {
		n := 4
		if g == len(recipes)-1 {
			n = 3
		}
		if len(group) != n || len(manifest.Groups[g]) != n {
			t.Fatalf("expected group %d to contain %d passages", g, n)
		}
		for j, recipe := range group {
			id, text, err := markdownReplayRender(pos, recipe, sections)
			if err != nil {
				t.Fatal(err)
			}
			if pos == 0 {
				const sample = "# Amber Landing\n\nWatch 001\n\n### Spare probe\n\nThe spare conductivity probe is in the green case. Return the cap after use.\n"
				if text != sample {
					t.Fatalf("rendered sample differs: %q", text)
				}
			}
			if id != manifest.Groups[g][j] {
				t.Fatalf("group %d passage %d = %s, want %s", g, j, id, manifest.Groups[g][j])
			}
			digest := sha256.Sum256([]byte(text))
			if seen[digest] {
				t.Fatalf("repeated Markdown for %s", id)
			}
			seen[digest] = true
			fmt.Fprintf(&rendered, "<!-- passage:%s -->\n%s", id, text)
			pos++
		}
	}
	if pos != 219 || rendered.String() != string(raw) {
		t.Fatal("authored section bank and recipes do not reproduce the ordered corpus")
	}
}
