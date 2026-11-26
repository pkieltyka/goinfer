package markdowneval

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
)

func TestFixture(t *testing.T) {
	f, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	for kind, model := range f.Checkpoints {
		var short, nearWindow, long bool
		pos := 0
		for _, group := range f.Groups {
			rows := 0
			for _, pin := range model.Documents[pos : pos+len(group)] {
				rows += pin.Tokens
				short = short || pin.Tokens < 31
				nearWindow = nearWindow || pin.Tokens >= 512 && pin.Tokens < 1024
				long = long || pin.Tokens >= 1024
			}
			if rows > 2748 {
				t.Errorf("%s group %v has %d tokens, over the Metal batch cap", kind, group, rows)
			}
			pos += len(group)
		}
		if !short || !nearWindow || !long {
			t.Errorf("%s lacks short, near-window or long passage coverage", kind)
		}
	}
}

func TestFixturePreservesMarkdownBody(t *testing.T) {
	const body = "  indented \n\n```text\n## content\n```\n"
	source := "<!-- passage:a -->\n" + body + "<!-- passage:b -->\n  body\n"
	docs, err := parsePassages(source)
	if err != nil || !slices.Equal(docs, []Passage{{"a", body}, {"b", "  body\n"}}) {
		t.Fatalf("parsePassages = %#v, %v", docs, err)
	}
	for _, bad := range []string{"\xff" + source, strings.ReplaceAll(source, "\n", "\r\n"), strings.TrimSuffix(source, "\n"),
		"<!-- passage:a -->\n \n" + source, strings.Replace(source, "passage:a", "passage:a?", 1)} {
		if _, err := parsePassages(bad); err == nil {
			t.Errorf("accepted invalid corpus %q", bad)
		}
	}
}

func TestFixtureRejectsInconsistentMetadata(t *testing.T) {
	corpus, err := files.ReadFile("testdata/v1/corpus.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name   string
		change func(*Fixture)
	}{
		{"corpus hash", func(f *Fixture) { f.CorpusSHA = strings.Repeat("0", 64) }},
		{"group order", func(f *Fixture) { f.Groups[0][0], f.Groups[0][1] = f.Groups[0][1], f.Groups[0][0] }},
		{"unknown label", func(f *Fixture) { f.Queries[0].Relevant = []string{"missing"} }},
		{"conflicting labels", func(f *Fixture) { f.Queries[0].Distractors = f.Queries[0].Relevant }},
		{"duplicate query", func(f *Fixture) { f.Queries[1].ID = f.Queries[0].ID }},
		{"wrong prepared role", func(f *Fixture) {
			pin := f.Checkpoints["mlx8"]
			pin.Queries[0] = pin.Documents[0]
		}},
		{"missing weight pin", func(f *Fixture) {
			pin := f.Checkpoints["mlx8"]
			pin.Files = slices.Delete(pin.Files, 2, 3)
			f.Checkpoints["mlx8"] = pin
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f, err := Load()
			if err != nil {
				t.Fatal(err)
			}
			tt.change(&f)
			meta, err := json.Marshal(f)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := readFixture(corpus, meta); err == nil {
				t.Fatal("accepted inconsistent metadata")
			}
		})
	}
}

func TestPreparedIDDigest(t *testing.T) {
	want := fmt.Sprintf("%x", sha256.Sum256([]byte{0, 0, 0, 0, 0x67, 0x45, 0x23, 0x01}))
	if got := IDsDigest([]int{0, 0x01234567}); got != want {
		t.Fatalf("prepared digest %s, want %s", got, want)
	}
}

func TestLabeledRanking(t *testing.T) {
	f := Fixture{
		Passages: []Passage{{ID: "b"}, {ID: "a"}, {ID: "c"}},
		Queries:  []Query{{ID: "q", Relevant: []string{"b"}, Distractors: []string{"a"}}},
	}
	docs := [][]float32{{1, 0}, {1, 0}, {0, 1}}
	queries := [][]float32{{1, 0}}
	ranks, err := f.Rank(docs, queries, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []Hit{{ID: "a", Score: 1, Label: "distractor"}, {ID: "b", Score: 1, Label: "relevant"}, {ID: "c", Label: "unjudged"}}
	got := ranks[0]
	if got.FirstRelevantRank != 2 || got.RecallAt5 != 1 || !slices.Equal(got.Hits, want) {
		t.Fatalf("ranking %+v, want hits %+v, rank 2, recall 1", got, want)
	}
	for _, bad := range [][]float32{{}, {1}, {0, 0}, {float32(math.NaN()), 0}, {0, float32(math.Inf(1))}} {
		if _, err := f.Rank(docs, [][]float32{bad}, 2); err == nil {
			t.Errorf("ranked invalid vector %v", bad)
		}
	}
}
