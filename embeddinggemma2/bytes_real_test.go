//go:build realckpt

package embeddinggemma2

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/pkieltyka/goinfer/tokenizer"
)

// TestReal_bytes is TestReal_parity for the loader from bytes: the real google/embeddinggemma-2 checkpoint
// (GOINFER_EG2_DIR, default ~/models/embeddinggemma-2) is read into memory once and split into the text-only form
// NewModel takes, the token table as stored and a safetensors file of the other language_model.* tensors. The model
// built from those bytes, with a tokenizer loaded from its pack and NewEncoder, must
//   - produce the reference's input ids for each of the golden's 48 texts,
//   - embed them to cosine >= 0.9999 with the sentence-transformers reference, the bar of TestReal_parity, and
//   - equal, bit for bit, the embeddings of the encoder LoadEncoder builds from the directory.
//
// It prints how long each way of loading took. Nothing is read from the directory after the first reads.
func TestReal_bytes(t *testing.T) {
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", dir)
	}
	golden := os.Getenv("GOINFER_EG2_GOLDEN")
	if golden == "" {
		golden = "../testdata/embeddinggemma2-real/golden.json"
	}
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("no reference golden (scripts/pin_embeddinggemma2_real.py writes it): %v", err)
	}
	var g struct {
		Items []struct {
			Prompt    string    `json:"prompt"`
			Text      string    `json:"text"`
			IDs       []int     `json:"ids"`
			Embedding []float64 `json:"embedding"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	read := func(name string) []byte {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	logf := func(format string, a ...any) { fmt.Fprintf(os.Stderr, "[eg2 bytes] "+format+"\n", a...) }

	cfg, err := LoadConfigBytes(read("config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tokens, layers, n := splitText(t, readST(t, read("model.safetensors")))
	tokens = bytes.Clone(tokens) // let the 1.5 GB file go; only the text-only form stays
	if len(tokens) != 2*cfg.VocabSize*cfg.Hidden {
		t.Fatalf("the token table is %d bytes, not %d x %d bfloat16 values", len(tokens), cfg.VocabSize, cfg.Hidden)
	}
	logf("text-only form: token table %d bytes, %d other tensors in %d bytes", len(tokens), n, len(layers))

	t0 := time.Now()
	m, err := NewModel(cfg, tokens, layers)
	if err != nil {
		t.Fatal(err)
	}
	dModel := time.Since(t0)
	pack, err := tokenizer.BuildPack(read("tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	t0 = time.Now()
	tok, err := tokenizer.LoadPack(pack)
	if err != nil {
		t.Fatal(err)
	}
	dTok := time.Since(t0)
	var st struct {
		Prompts map[string]string `json:"prompts"`
	}
	if err := json.Unmarshal(read("config_sentence_transformers.json"), &st); err != nil {
		t.Fatal(err)
	}
	e, err := NewEncoder(m, tok, st.Prompts)
	if err != nil {
		t.Fatal(err)
	}
	logf("from bytes: NewModel %v, tokenizer.LoadPack %v", dModel.Round(time.Millisecond), dTok.Round(time.Millisecond))

	t0 = time.Now()
	ref, err := LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	logf("from the directory: LoadEncoder %v", time.Since(t0).Round(time.Millisecond))

	worst, compared := 1.0, 0
	for i, it := range g.Items {
		ids, err := e.Tokenize(it.Text, it.Prompt)
		if err != nil {
			t.Fatalf("item %d: %v", i, err)
		}
		if fmt.Sprint(ids) != fmt.Sprint(it.IDs) {
			t.Errorf("item %d (prompt %q): ids differ from the reference's", i, it.Prompt)
			continue
		}
		v, err := m.Embed(ids)
		if err != nil {
			t.Fatal(err)
		}
		var dot, nr float64
		for j, w := range it.Embedding {
			dot += float64(v[j]) * w
			nr += w * w
		}
		cos := dot / math.Sqrt(nr)
		worst = math.Min(worst, cos)
		if cos < 0.9999 {
			t.Errorf("item %d (prompt %q, %d tokens): cosine %.9f, under 0.9999", i, it.Prompt, len(ids), cos)
		}
		// The directory's encoder is the slow half of this test; the short texts are enough to show the two agree.
		if len(ids) <= 256 {
			w, err := ref.Model().Embed(ids)
			if err != nil {
				t.Fatal(err)
			}
			if !sameBits(v, w) {
				t.Errorf("item %d (%d tokens): the model from bytes and the model from the directory differ", i, len(ids))
			}
			compared++
		}
	}
	logf("RESULT %d texts, worst cosine %.9f to the reference; %d compared bit for bit with LoadEncoder", len(g.Items), worst, compared)
	if compared == 0 {
		t.Fatal("no text was short enough to compare the two loaders")
	}
}
