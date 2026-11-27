package embeddinggemma2_test

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/tokenizer"
)

var embeddingPrepAssets = flag.String("embedding-prep-assets", "", "directory with pinned EmbeddingGemma 2 tokenizer assets (see tokenizer/testdata/embedding-prep/README.md)")

type embeddingPrepCase struct {
	Row         int    `json:"row"`
	Role        string `json:"role"`
	Text        string `json:"text"`
	IDs         []int  `json:"ids"`
	PreparedIDs []int  `json:"prepared_ids"`
}

// TestEmbeddingPrepReference checks exact token IDs for the five pinned
// EmbeddingGemma 2 rows in tokenizer/testdata/embedding-prep/reference.json
// through embeddinggemma2.Encoder.Tokenize and tokenizer.Tokenizer.EncodeSegments
// (literal <eos> handling on row 11). The four bge-small-en-v1.5 rows stay
// archived in reference.json for external callers; goinfer does not import or
// ship an external BGE encoder.
func TestEmbeddingPrepReference(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "tokenizer", "testdata", "embedding-prep", "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases map[string][]embeddingPrepCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases["embeddinggemma-2"]) != 5 {
		t.Fatalf("embeddinggemma-2 reference rows = %d, want 5", len(cases["embeddinggemma-2"]))
	}
	if len(cases["bge-small-en-v1.5"]) != 4 {
		t.Fatalf("archived bge-small-en-v1.5 reference rows = %d, want 4", len(cases["bge-small-en-v1.5"]))
	}
	if *embeddingPrepAssets == "" {
		t.Skip("set -embedding-prep-assets to the pinned EmbeddingGemma 2 files described in tokenizer/testdata/embedding-prep/README.md")
	}

	const model = "embeddinggemma-2"
	tok, err := tokenizer.LoadJSONBytes(readEmbeddingPrepAsset(t, model, "tokenizer.json",
		"4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"))
	if err != nil {
		t.Fatal(err)
	}
	var config struct {
		Prompts map[string]string `json:"prompts"`
	}
	err = json.Unmarshal(readEmbeddingPrepAsset(t, model, "config_sentence_transformers.json",
		"031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5"), &config)
	if err != nil {
		t.Fatal(err)
	}
	// Tokenize uses the tokenizer and prompt table, so the model needs no weights here.
	enc, err := embeddinggemma2.NewEncoder(new(embeddinggemma2.Model), tok, config.Prompts)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases[model] {
		t.Run(model+"/"+c.Role+"/"+strconv.Itoa(c.Row), func(t *testing.T) {
			got, err := enc.Tokenize(c.Text, c.Role)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, c.IDs) {
				t.Errorf("Tokenize(%q, %q) = %v, want %v", c.Text, c.Role, got, c.IDs)
			}
			if len(c.PreparedIDs) > 0 {
				t.Run("literal-content", func(t *testing.T) {
					prompt, err := enc.PromptText(c.Role)
					if err != nil {
						t.Fatal(err)
					}
					got, err := tok.EncodeSegments([]tokenizer.Segment{
						{Text: prompt + c.Text},
						{Text: "<eos>", Special: true},
					}, true)
					if err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(got, c.PreparedIDs) {
						t.Errorf("literal token IDs for %q = %v, want %v", c.Text, got, c.PreparedIDs)
					}
				})
			}
		})
	}
}

func readEmbeddingPrepAsset(t *testing.T, model, name, wantSHA string) []byte {
	t.Helper()
	path := filepath.Join(*embeddingPrepAssets, model, name)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != wantSHA {
		t.Fatalf("%s: SHA-256 %s, want %s", path, got, wantSHA)
	}
	return raw
}
