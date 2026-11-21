//go:build cuda

// Command corpus-batch times the pinned pbrain shared Markdown on the CUDA EmbeddingGemma 2 accelerator two ways in one
// binary: -arm single runs each document as its own forward, -arm pairs runs the documents in fixed adjacent pairs of
// the manifest's current order through one ForwardBatch each. Queries run one at a time in both arms. Both pool the
// projected rows exactly as the encoder does (embeddinggemma2's mean pool and L2 normalisation, float64 accumulation).
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

	_ "github.com/pkieltyka/goinfer/backend/cuda"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
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

func tokenize(enc *embeddinggemma2.Encoder, in input, export map[string][]int) (tokenRow, error) {
	ids, err := enc.Tokenize(in.text, in.role)
	if err != nil {
		return tokenRow{}, fmt.Errorf("%s: %w", in.id, err)
	}
	buf := make([]byte, 0, 4*len(ids))
	for _, id := range ids {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(id))
	}
	if export != nil {
		export[in.id] = ids
	}
	return tokenRow{in.id, in.role, len(ids), fmt.Sprintf("%x", sha256.Sum256(buf))}, nil
}

func queryInput(q question) input { return input{q.ID, "query", q.Text} }

type preparation struct {
	ManifestSHA string            `json:"manifest_sha256"`
	Prompts     map[string]string `json:"prompts"`
	Tokens      []tokenRow        `json:"inputs"`
	TokenIDs    map[string][]int  `json:"token_ids,omitempty"`
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

type group struct {
	IDs       []string `json:"ids"`
	Tokens    int      `json:"tokens"`
	Nanos     int64    `json:"wall_ns"`
	Allocated uint64   `json:"allocated_bytes"`
}

type report struct {
	Source          string        `json:"source"`
	Revision        string        `json:"revision"`
	FixtureClock    string        `json:"fixture_clock"`
	ManifestSHA     string        `json:"manifest_sha256"`
	Model           string        `json:"model"`
	ModelRev        string        `json:"model_revision"`
	Threads         int           `json:"gomaxprocs"`
	Backend         string        `json:"backend"`
	Arm             string        `json:"arm"`
	Cutoff          float64       `json:"abstention_cutoff"`
	LoadNanos       int64         `json:"load_wall_ns"`
	LoadBytes       uint64        `json:"load_allocated_bytes"`
	DeviceLoadNanos int64         `json:"device_load_wall_ns"`
	DeviceLoadBytes uint64        `json:"device_load_allocated_bytes"`
	Samples         []observation `json:"samples"`
	Groups          []group       `json:"document_groups"`
	Rankings        []ranking     `json:"queries"`
}

// batcher is the CUDA accelerator's surface this driver uses: the singleton forward and the batch.
type batcher interface {
	Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error)
	ForwardBatch(ids [][]int) ([][]float32, error)
}

// poolNormalize is embeddinggemma2's pooling, kept identical: the float64 mean over T rows of width d, then division by
// max(norm, 1e-12), each value rounded to float32 once.
func poolNormalize(h []float32, T, d int) []float32 {
	mean := make([]float64, d)
	for t := range T {
		for j, v := range h[t*d : (t+1)*d] {
			mean[j] += float64(v)
		}
	}
	var ss float64
	for j := range mean {
		mean[j] /= float64(T)
		ss += mean[j] * mean[j]
	}
	n := math.Max(math.Sqrt(ss), 1e-12)
	out := make([]float32, d)
	for j := range mean {
		out[j] = float32(mean[j] / n)
	}
	return out
}

func run(w workload, model string, p preparation, arm string, fit bool) (report, error) {
	if p.ManifestSHA != w.pinSHA || len(p.Tokens) != len(w.docs)+len(w.qs) {
		return report{}, fmt.Errorf("preflight and corpus manifest differ")
	}
	r := report{
		Source: w.pins.Source, Revision: w.pins.Revision, FixtureClock: w.pins.FixtureClock, ManifestSHA: w.pinSHA,
		Model: w.pins.Model, ModelRev: w.pins.ModelRevision, Threads: runtime.GOMAXPROCS(0), Backend: "cuda", Arm: arm,
		Cutoff: w.pins.Cutoff,
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
	var device embeddinggemma2.Accelerator
	ns, bytes, err = measure(func() error {
		var err error
		device, err = enc.UseAccelerator("cuda")
		return err
	})
	if err != nil {
		return report{}, fmt.Errorf("CUDA required: %w", err)
	}
	defer device.Close()
	b, ok := device.(batcher)
	if device.Name() != "cuda" || !ok {
		return report{}, fmt.Errorf("unexpected device %q (batch %v)", device.Name(), ok)
	}
	r.DeviceLoadNanos, r.DeviceLoadBytes = ns, bytes
	if fit {
		return r, nil
	}
	E := enc.HiddenDim()
	// check verifies an input against the preflight after its timed call, untimed: order, role, framed IDs and vector.
	check := func(in input, ids []int, vector []float32, ns int64, bytes uint64) error {
		pin := p.Tokens[len(r.Samples)]
		if in.id != pin.ID || in.role != pin.Role {
			return fmt.Errorf("preflight input order differs at %s", in.id)
		}
		token, err := tokenize(enc, in, nil)
		if err != nil {
			return err
		}
		if token.Count != len(ids) || token != pin {
			return fmt.Errorf("%s: run tokenization differs from preflight", in.id)
		}
		digest, err := vectorDigest(vector, E)
		if err != nil {
			return fmt.Errorf("%s: %w", in.id, err)
		}
		r.Samples = append(r.Samples, observation{token, ns, bytes, digest})
		return nil
	}
	one := func(in input) ([]float32, error) {
		var ids []int
		var vector []float32
		ns, bytes, err := measure(func() error {
			var err error
			if ids, err = enc.Tokenize(in.text, in.role); err != nil {
				return err
			}
			h, _, err := b.Forward(ids, false)
			if err != nil {
				return err
			}
			vector = poolNormalize(h, len(ids), E)
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("%s: %w", in.id, err)
		}
		return vector, check(in, ids, vector, ns, bytes)
	}
	first, err := one(queryInput(w.qs[0]))
	if err != nil {
		return report{}, err
	}
	vectors := make([][]float32, 0, len(w.docs))
	switch arm {
	case "single":
		for _, in := range w.docs {
			vector, err := one(in)
			if err != nil {
				return report{}, err
			}
			vectors = append(vectors, vector)
			n := r.Samples[len(r.Samples)-1]
			r.Groups = append(r.Groups, group{[]string{in.id}, n.Count, n.Nanos, n.Allocated})
		}
	case "pairs":
		if len(w.docs)%2 != 0 {
			return report{}, fmt.Errorf("%d documents do not form pairs", len(w.docs))
		}
		for i := 0; i < len(w.docs); i += 2 {
			pair := w.docs[i : i+2]
			var ids [][]int
			var vs [][]float32
			ns, bytes, err := measure(func() error {
				ids = make([][]int, len(pair))
				for j, in := range pair {
					var err error
					if ids[j], err = enc.Tokenize(in.text, in.role); err != nil {
						return err
					}
				}
				hs, err := b.ForwardBatch(ids)
				if err != nil {
					return err
				}
				vs = make([][]float32, len(pair))
				for j := range pair {
					vs[j] = poolNormalize(hs[j], len(ids[j]), E)
				}
				return nil
			})
			if err != nil {
				return report{}, fmt.Errorf("%s+%s: %w", pair[0].id, pair[1].id, err)
			}
			g := group{Nanos: ns, Allocated: bytes}
			for j, in := range pair {
				if err := check(in, ids[j], vs[j], 0, 0); err != nil {
					return report{}, err
				}
				g.IDs, g.Tokens = append(g.IDs, in.id), g.Tokens+len(ids[j])
				vectors = append(vectors, vs[j])
			}
			r.Groups = append(r.Groups, g)
		}
	default:
		return report{}, fmt.Errorf("unknown arm %q", arm)
	}
	for i, q := range w.qs {
		query := first
		if i != 0 {
			query, err = one(queryInput(q))
			if err != nil {
				return report{}, err
			}
		}
		r.Rankings = append(r.Rankings, rankQuestion(q, w.docs, vectors, query, w.pins.Cutoff))
	}
	return r, nil
}

func execute(manifestPath, fixture, model, preflight, arm string, fit bool) error {
	w, err := loadWorkload(manifestPath, fixture)
	if err != nil {
		return err
	}
	if err := verifyModel(model, w.pins.ModelFiles); err != nil {
		return err
	}
	raw, err := os.ReadFile(preflight)
	if err != nil {
		return err
	}
	var p preparation
	if err := json.Unmarshal(raw, &p); err != nil {
		return err
	}
	r, err := run(w, model, p, arm, fit)
	if err != nil {
		return err
	}
	out := json.NewEncoder(os.Stdout)
	out.SetIndent("", "  ")
	return out.Encode(r)
}

func main() {
	pins := flag.String("manifest", "", "corpus and model manifest")
	fixture := flag.String("fixture", "", "directory containing the extracted pbrain files")
	model := flag.String("model", "", "local EmbeddingGemma 2 checkpoint directory")
	preflight := flag.String("preflight", "", "preparation JSON from the corpus command")
	arm := flag.String("arm", "", "single (one forward per document) or pairs (fixed adjacent document pairs per forward)")
	fit := flag.Bool("fit", false, "load the model and attach CUDA without running the corpus")
	flag.Parse()
	if *pins == "" || *fixture == "" || *model == "" || *preflight == "" || flag.NArg() != 0 ||
		(*arm != "single" && *arm != "pairs") {
		flag.Usage()
		os.Exit(2)
	}
	if err := execute(*pins, *fixture, *model, *preflight, *arm, *fit); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
