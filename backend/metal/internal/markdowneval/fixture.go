// Package markdowneval reads the versioned Markdown fixture used by Metal tests.
package markdowneval

import (
	"crypto/sha256"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"
)

const (
	CorpusVersion = 1
	PassageCount  = 15
	QueryCount    = 8
)

//go:embed testdata/v1/corpus.md testdata/v1/manifest.json
var files embed.FS

type Passage struct {
	ID, Text string
}

type Query struct {
	ID          string   `json:"id"`
	Text        string   `json:"text"`
	Relevant    []string `json:"relevant"`
	Distractors []string `json:"distractors"`
}

type InputPin struct {
	ID     string `json:"id"`
	Tokens int    `json:"tokens"`
	SHA256 string `json:"sha256"`
}

type FilePin struct {
	Name   string `json:"name"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

type Checkpoint struct {
	Repo      string     `json:"repo"`
	Revision  string     `json:"revision"`
	Files     []FilePin  `json:"files"`
	Documents []InputPin `json:"documents"`
	Queries   []InputPin `json:"queries"`
}

type Fixture struct {
	Version     int                   `json:"version"`
	CorpusSHA   string                `json:"corpus_sha256"`
	Groups      [][]string            `json:"groups"`
	Queries     []Query               `json:"queries"`
	Checkpoints map[string]Checkpoint `json:"checkpoints"`
	Passages    []Passage             `json:"-"`
}

func Load() (Fixture, error) {
	manifest, err := files.ReadFile("testdata/v1/manifest.json")
	if err != nil {
		return Fixture{}, err
	}
	corpus, err := files.ReadFile("testdata/v1/corpus.md")
	if err != nil {
		return Fixture{}, err
	}
	return readFixture(corpus, manifest)
}

func readFixture(corpus, manifest []byte) (Fixture, error) {
	var f Fixture
	if err := json.Unmarshal(manifest, &f); err != nil {
		return f, err
	}
	if f.Version != CorpusVersion || f.CorpusSHA != fmt.Sprintf("%x", sha256.Sum256(corpus)) {
		return f, fmt.Errorf("fixture version or corpus digest mismatch")
	}
	var err error
	f.Passages, err = parsePassages(string(corpus))
	if err != nil {
		return f, err
	}
	docs := make([]string, 0, PassageCount)
	for _, p := range f.Passages {
		docs = append(docs, p.ID)
	}
	if err := checkGroups(f.Groups, docs); err != nil {
		return f, err
	}
	queries, err := checkQueries(f.Queries, docs)
	if err != nil {
		return f, err
	}
	if len(f.Checkpoints) != 2 {
		return f, fmt.Errorf("fixture requires two checkpoint pins")
	}
	for kind, names := range map[string][]string{
		"q8_0": {"embeddinggemma-2-Q8_0.gguf"},
		"mlx8": {"config.json", "config_sentence_transformers.json", "model.safetensors", "tokenizer.json"},
	} {
		pin := f.Checkpoints[kind]
		if err := pin.validate(names, docs, queries); err != nil {
			return f, fmt.Errorf("%s: %w", kind, err)
		}
	}
	return f, nil
}

const delimiter = "<!-- passage:"

func parsePassages(corpus string) ([]Passage, error) {
	if !utf8.ValidString(corpus) || strings.ContainsRune(corpus, '\r') ||
		!strings.HasPrefix(corpus, delimiter) || !strings.HasSuffix(corpus, "\n") {
		return nil, fmt.Errorf("corpus requires UTF-8, LF and an opening passage delimiter")
	}
	var docs []Passage
	parts := strings.Split(strings.TrimPrefix(corpus, delimiter), "\n"+delimiter)
	for i, part := range parts {
		id, body, ok := strings.Cut(part, " -->\n")
		if !ok || !validID(id) || strings.TrimSpace(body) == "" {
			return nil, fmt.Errorf("invalid passage delimiter or body for %q", id)
		}
		// Split consumes the LF immediately before the next delimiter. Restore it.
		if i+1 < len(parts) {
			body += "\n"
		}
		docs = append(docs, Passage{ID: id, Text: body})
	}
	return docs, nil
}

func validID(id string) bool {
	return id != "" && strings.IndexFunc(id, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-')
	}) == -1
}

func checkGroups(groups [][]string, docs []string) error {
	if len(docs) != PassageCount || len(groups) != 4 {
		return fmt.Errorf("fixture requires %d passages in four groups", PassageCount)
	}
	var order []string
	for i, group := range groups {
		if len(group) != [...]int{4, 4, 4, 3}[i] {
			return fmt.Errorf("group %d has unexpected size %d", i, len(group))
		}
		order = append(order, group...)
	}
	if !slices.Equal(order, docs) {
		return fmt.Errorf("group order differs from corpus order")
	}
	seen := make(map[string]bool)
	for _, id := range docs {
		if seen[id] {
			return fmt.Errorf("duplicate passage %q", id)
		}
		seen[id] = true
	}
	return nil
}

func checkQueries(queries []Query, docs []string) ([]string, error) {
	if len(queries) != QueryCount {
		return nil, fmt.Errorf("fixture requires %d queries", QueryCount)
	}
	var ids []string
	for _, q := range queries {
		if !validID(q.ID) || slices.Contains(ids, q.ID) || strings.TrimSpace(q.Text) == "" ||
			!utf8.ValidString(q.Text) || len(q.Relevant) == 0 || len(q.Distractors) == 0 {
			return nil, fmt.Errorf("invalid query %q", q.ID)
		}
		ids = append(ids, q.ID)
		labels := make(map[string]bool)
		for _, group := range [][]string{q.Relevant, q.Distractors} {
			for _, id := range group {
				if labels[id] || !slices.Contains(docs, id) {
					return nil, fmt.Errorf("query %q has a duplicate or unknown label %q", q.ID, id)
				}
				labels[id] = true
			}
		}
	}
	return ids, nil
}

func (c Checkpoint) validate(names, docs, queries []string) error {
	if c.Repo == "" || !validHex(c.Revision, 20) || len(c.Files) != len(names) {
		return fmt.Errorf("incomplete checkpoint identity")
	}
	for i, f := range c.Files {
		if f.Name != names[i] || f.Size <= 0 || !validHex(f.SHA256, sha256.Size) {
			return fmt.Errorf("invalid file pin for %q", f.Name)
		}
	}
	if err := checkInputs(c.Documents, docs); err != nil {
		return err
	}
	return checkInputs(c.Queries, queries)
}

func checkInputs(pins []InputPin, ids []string) error {
	if len(pins) != len(ids) {
		return fmt.Errorf("prepared pin count %d, want %d", len(pins), len(ids))
	}
	for i, p := range pins {
		if p.ID != ids[i] || p.Tokens < 2 || p.Tokens > 8192 || !validHex(p.SHA256, sha256.Size) {
			return fmt.Errorf("invalid prepared pin for %q", p.ID)
		}
	}
	return nil
}

func validHex(s string, n int) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == n && s == strings.ToLower(s)
}

// IDsDigest hashes prepared IDs as unsigned little-endian 32-bit words.
func IDsDigest(ids []int) string {
	h := sha256.New()
	var word [4]byte
	for _, id := range ids {
		binary.LittleEndian.PutUint32(word[:], uint32(id))
		h.Write(word[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}
