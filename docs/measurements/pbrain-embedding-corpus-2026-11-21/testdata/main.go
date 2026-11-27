// Command corpus evaluates the pinned pbrain shared Markdown with the native text encoder.
package main

import (
	"cmp"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"time"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/tokenizer"
)

type filePin struct {
	Path, SHA256 string
}

type manifest struct {
	Source, Revision string
	FixtureClock     string `json:"fixture_clock"`
	LicenseSHA       string `json:"license_sha256"`
	Questions        filePin
	Documents        []struct {
		filePin
		ID      string
		Current bool
	}
	Model         string
	ModelRevision string            `json:"model_revision"`
	ModelFiles    map[string]string `json:"model_files"`
	Cutoff        float64
}

type question struct {
	ID, Kind, Style, Half string
	Text                  string `json:"q"`
	Expect                []string
	Regression            bool
}

type input struct{ id, role, text string }

type tokenRow struct {
	ID    string `json:"id"`
	Role  string `json:"role"`
	Count int    `json:"tokens"`
	SHA   string `json:"ids_sha256"`
}

type observation struct {
	tokenRow
	Nanos     int64  `json:"wall_ns"`
	Allocated uint64 `json:"allocated_bytes"`
	VectorSHA string `json:"vector_sha256"`
}

type match struct {
	ID    string  `json:"id"`
	Score float64 `json:"cosine"`
}

type ranking struct {
	ID            string   `json:"id"`
	Kind          string   `json:"kind"`
	Style         string   `json:"style"`
	Half          string   `json:"half,omitempty"`
	Regression    bool     `json:"regression,omitempty"`
	Expected      []string `json:"expected,omitempty"`
	FirstExpected int      `json:"first_expected_rank,omitempty"`
	Recall1       float64  `json:"recall_at_1"`
	Recall5       float64  `json:"recall_at_5"`
	Abstained     bool     `json:"abstained"`
	Top           []match  `json:"top_5"`
}

type workload struct {
	pins   manifest
	pinSHA string
	docs   []input
	qs     []question
}

func loadWorkload(manifestPath, fixture string) (workload, error) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return workload{}, err
	}
	w := workload{pinSHA: fmt.Sprintf("%x", sha256.Sum256(raw))}
	if err := json.Unmarshal(raw, &w.pins); err != nil {
		return workload{}, err
	}
	if _, err := readPinned(fixture, filePin{"LICENSE", w.pins.LicenseSHA}); err != nil {
		return workload{}, err
	}
	raw, err = readPinned(fixture, w.pins.Questions)
	if err != nil {
		return workload{}, err
	}
	if err := json.Unmarshal(raw, &w.qs); err != nil {
		return workload{}, err
	}
	ids := make(map[string]bool)
	for _, doc := range w.pins.Documents {
		text, err := readPinned(fixture, doc.filePin)
		if err != nil {
			return workload{}, err
		}
		if doc.Current {
			if ids[doc.ID] || doc.ID == "" {
				return workload{}, fmt.Errorf("invalid or duplicate document ID %q", doc.ID)
			}
			w.docs = append(w.docs, input{doc.ID, "document", string(text)})
			ids[doc.ID] = true
		}
	}
	if len(w.docs) == 0 || len(w.qs) == 0 {
		return workload{}, fmt.Errorf("workload needs documents and questions")
	}
	seen := make(map[string]bool)
	for _, q := range w.qs {
		if q.ID == "" || q.Text == "" || seen[q.ID] {
			return workload{}, fmt.Errorf("invalid or duplicate question %q", q.ID)
		}
		seen[q.ID] = true
		switch q.Kind {
		case "answerable":
			if len(q.Expect) == 0 {
				return workload{}, fmt.Errorf("%s: no expected IDs", q.ID)
			}
		case "unanswerable":
			if len(q.Expect) != 0 || (q.Half != "calibration" && q.Half != "held-out") {
				return workload{}, fmt.Errorf("%s: invalid unanswerable label", q.ID)
			}
		default:
			return workload{}, fmt.Errorf("%s: unknown label %q", q.ID, q.Kind)
		}
		for _, id := range q.Expect {
			if !ids[id] {
				return workload{}, fmt.Errorf("%s: expected ID %q is absent from the current corpus", q.ID, id)
			}
		}
	}
	return w, nil
}

func readPinned(root string, pin filePin) ([]byte, error) {
	data, err := os.ReadFile(filepath.Join(root, pin.Path))
	if err != nil {
		return nil, err
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != pin.SHA256 {
		return nil, fmt.Errorf("%s: SHA-256 %s, want %s", pin.Path, got, pin.SHA256)
	}
	return data, nil
}

func verifyModel(dir string, files map[string]string) error {
	for name, want := range files {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		hash := sha256.New()
		_, err = io.Copy(hash, f)
		closeErr := f.Close()
		if err != nil {
			return err
		}
		if closeErr != nil {
			return closeErr
		}
		if got := fmt.Sprintf("%x", hash.Sum(nil)); got != want {
			return fmt.Errorf("%s: SHA-256 %s, want %s", name, got, want)
		}
	}
	return nil
}

func tokenize(enc *embeddinggemma2.Encoder, in input) (tokenRow, error) {
	ids, err := enc.Tokenize(in.text, in.role)
	if err != nil {
		return tokenRow{}, fmt.Errorf("%s: %w", in.id, err)
	}
	buf := make([]byte, 0, 4*len(ids))
	for _, id := range ids {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(id))
	}
	return tokenRow{in.id, in.role, len(ids), fmt.Sprintf("%x", sha256.Sum256(buf))}, nil
}

func queryInput(q question) input { return input{q.ID, "query", q.Text} }

type preparation struct {
	ManifestSHA string            `json:"manifest_sha256"`
	Prompts     map[string]string `json:"prompts"`
	Tokens      []tokenRow        `json:"inputs"`
}

func prepare(w workload, model string) (preparation, error) {
	if err := verifyModel(model, w.pins.ModelFiles); err != nil {
		return preparation{}, err
	}
	if _, err := embeddinggemma2.LoadConfig(model); err != nil {
		return preparation{}, err
	}
	tok, err := tokenizer.Load(model)
	if err != nil {
		return preparation{}, err
	}
	raw, err := os.ReadFile(filepath.Join(model, "config_sentence_transformers.json"))
	if err != nil {
		return preparation{}, err
	}
	var cfg struct{ Prompts map[string]string }
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return preparation{}, err
	}
	// Tokenize only needs the tokenizer and prompt table.
	enc, err := embeddinggemma2.NewEncoder(new(embeddinggemma2.Model), tok, cfg.Prompts)
	if err != nil {
		return preparation{}, err
	}
	out := preparation{ManifestSHA: w.pinSHA, Prompts: make(map[string]string)}
	for _, role := range []string{"document", "query"} {
		out.Prompts[role], err = enc.PromptText(role)
		if err != nil {
			return preparation{}, err
		}
	}
	inputs := append([]input{queryInput(w.qs[0])}, w.docs...)
	for _, q := range w.qs[1:] {
		inputs = append(inputs, queryInput(q))
	}
	for _, in := range inputs {
		row, err := tokenize(enc, in)
		if err != nil {
			return preparation{}, err
		}
		out.Tokens = append(out.Tokens, row)
	}
	return out, nil
}

func measure(f func() error) (int64, uint64, error) {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	err := f()
	delta := time.Since(start)
	runtime.ReadMemStats(&after)
	return delta.Nanoseconds(), after.TotalAlloc - before.TotalAlloc, err
}

func vectorDigest(vector []float32, width int) (string, error) {
	if len(vector) != width {
		return "", fmt.Errorf("vector width %d, want %d", len(vector), width)
	}
	var norm float64
	buf := make([]byte, 0, 4*width)
	for _, v := range vector {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		norm += float64(v) * float64(v)
	}
	if math.IsNaN(norm) || math.Abs(norm-1) > 1e-4 {
		return "", fmt.Errorf("vector squared norm %.9g", norm)
	}
	return fmt.Sprintf("%x", sha256.Sum256(buf)), nil
}

func rankQuestion(q question, docs []input, vectors [][]float32, query []float32, cutoff float64) ranking {
	all := make([]match, 0, len(vectors))
	for i, vector := range vectors {
		var score float64
		for j, v := range vector {
			score += float64(v) * float64(query[j])
		}
		all = append(all, match{docs[i].id, score})
	}
	slices.SortFunc(all, func(a, b match) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 {
			return c
		}
		return cmp.Compare(a.ID, b.ID)
	})
	r := ranking{
		ID: q.ID, Kind: q.Kind, Style: q.Style, Half: q.Half, Regression: q.Regression,
		Expected: q.Expect, Top: all[:min(5, len(all))], Abstained: all[0].Score < cutoff,
	}
	for i, m := range all {
		if !slices.Contains(q.Expect, m.ID) {
			continue
		}
		if r.FirstExpected == 0 {
			r.FirstExpected = i + 1
		}
		fraction := 1 / float64(len(q.Expect))
		if i == 0 {
			r.Recall1 += fraction
		}
		if i < 5 {
			r.Recall5 += fraction
		}
	}
	return r
}

type report struct {
	Source       string        `json:"source"`
	Revision     string        `json:"revision"`
	FixtureClock string        `json:"fixture_clock"`
	ManifestSHA  string        `json:"manifest_sha256"`
	Model        string        `json:"model"`
	ModelRev     string        `json:"model_revision"`
	Threads      int           `json:"gomaxprocs"`
	Backend      string        `json:"backend"`
	Cutoff       float64       `json:"abstention_cutoff"`
	LoadNanos    int64         `json:"load_wall_ns"`
	LoadBytes    uint64        `json:"load_allocated_bytes"`
	Samples      []observation `json:"samples"`
	Rankings     []ranking     `json:"queries"`
}

func run(w workload, model string, p preparation) (report, error) {
	if p.ManifestSHA != w.pinSHA || len(p.Tokens) != len(w.docs)+len(w.qs) {
		return report{}, fmt.Errorf("preflight and corpus manifest differ")
	}
	r := report{
		Source: w.pins.Source, Revision: w.pins.Revision, FixtureClock: w.pins.FixtureClock, ManifestSHA: w.pinSHA,
		Model: w.pins.Model, ModelRev: w.pins.ModelRevision, Threads: runtime.GOMAXPROCS(0), Backend: "cpu", Cutoff: w.pins.Cutoff,
	}
	var enc *embeddinggemma2.Encoder
	ns, bytes, err := measure(func() error {
		var err error
		enc, err = embeddinggemma2.LoadEncoder(model)
		return err
	})
	if err != nil {
		return report{}, err
	}
	r.LoadNanos, r.LoadBytes = ns, bytes
	embed := func(in input) ([]float32, error) {
		pin := p.Tokens[len(r.Samples)]
		if in.id != pin.ID || in.role != pin.Role {
			return nil, fmt.Errorf("preflight input order differs at %s", in.id)
		}
		var count int
		var vector []float32
		ns, bytes, err := measure(func() error {
			var err error
			vector, count, err = enc.EmbedText(in.text, in.role)
			return err
		})
		if err != nil {
			return nil, fmt.Errorf("%s: %w", in.id, err)
		}
		token, err := tokenize(enc, in)
		if err != nil {
			return nil, err
		}
		if token.Count != count || token != pin {
			return nil, fmt.Errorf("%s: run tokenization differs from preflight", in.id)
		}
		digest, err := vectorDigest(vector, enc.HiddenDim())
		if err != nil {
			return nil, fmt.Errorf("%s: %w", in.id, err)
		}
		r.Samples = append(r.Samples, observation{token, ns, bytes, digest})
		return vector, nil
	}
	first, err := embed(queryInput(w.qs[0]))
	if err != nil {
		return report{}, err
	}
	vectors := make([][]float32, 0, len(w.docs))
	for _, in := range w.docs {
		vector, err := embed(in)
		if err != nil {
			return report{}, err
		}
		vectors = append(vectors, vector)
	}
	for i, q := range w.qs {
		query := first
		if i != 0 {
			query, err = embed(queryInput(q))
			if err != nil {
				return report{}, err
			}
		}
		r.Rankings = append(r.Rankings, rankQuestion(q, w.docs, vectors, query, w.pins.Cutoff))
	}
	return r, nil
}

func execute(manifestPath, fixture, model, preflight string) error {
	w, err := loadWorkload(manifestPath, fixture)
	if err != nil {
		return err
	}
	out := json.NewEncoder(os.Stdout)
	out.SetIndent("", "  ")
	if preflight == "" {
		p, err := prepare(w, model)
		if err != nil {
			return err
		}
		return out.Encode(p)
	}
	raw, err := os.ReadFile(preflight)
	if err != nil {
		return err
	}
	var p preparation
	if err := json.Unmarshal(raw, &p); err != nil {
		return err
	}
	r, err := run(w, model, p)
	if err != nil {
		return err
	}
	return out.Encode(r)
}

func main() {
	pins := flag.String("manifest", "", "corpus and model manifest")
	fixture := flag.String("fixture", "", "directory containing the extracted pbrain files")
	model := flag.String("model", "", "local EmbeddingGemma 2 checkpoint directory")
	preflight := flag.String("preflight", "", "preparation JSON from this command, or omit to verify and prepare only")
	flag.Parse()
	if *pins == "" || *fixture == "" || *model == "" || flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}
	if err := execute(*pins, *fixture, *model, *preflight); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
