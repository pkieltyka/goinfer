// Package bert loads sentence-transformers BERT models and their tokenizers.
package bert

import (
	"io/fs"

	"github.com/pkieltyka/goinfer/internal/modelcore/embed"
	"github.com/pkieltyka/goinfer/internal/modelcore/encoder"
)

// BERT embeds prepared token IDs with a loaded BERT model.
// Calls to Embed can run concurrently. Close after the last call finishes.
type BERT struct {
	model     *encoder.BERT
	tokenizer *Tokenizer
}

// LoadBERTFromFS loads the model files under dir in fsys, reading each once.
// The model may retain the bytes returned for model.safetensors. Callers must
// keep those bytes unchanged until Close.
func LoadBERTFromFS(fsys fs.FS, dir string) (*BERT, error) {
	model, err := encoder.LoadBERTFromFS(fsys, dir)
	if err != nil {
		return nil, err
	}
	b := &BERT{model: model}
	if tok := model.Tokenizer(); tok != nil {
		b.tokenizer = &Tokenizer{tokenizer: tok}
	}
	return b, nil
}

// Embed returns the normalized embedding of ids. Supply any special tokens
// the checkpoint requires as part of ids.
func (b *BERT) Embed(ids []int32) []float32 { return b.model.Embed(ids) }

// Close releases the model's weights. Closing a zero BERT succeeds.
func (b *BERT) Close() error {
	if b.model == nil {
		return nil
	}
	return b.model.Close()
}

// Tokenizer returns the tokenizer loaded with the model, or nil if unavailable.
func (b *BERT) Tokenizer() *Tokenizer { return b.tokenizer }

// Tokenizer converts text and special tokens to IDs for a checkpoint.
type Tokenizer struct {
	tokenizer *embed.Tokenizer
}

// LoadTokenizerFromFS loads a tokenizer JSON file named by name in fsys.
func LoadTokenizerFromFS(fsys fs.FS, name string) (*Tokenizer, error) {
	tok, err := embed.LoadTokenizerFromFS(fsys, name)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{tokenizer: tok}, nil
}

// Encode converts text to token IDs without inserting special tokens.
func (t *Tokenizer) Encode(text string) []int32 { return t.tokenizer.Encode(text) }

// SpecialID looks up literal in the tokenizer's vocabulary of special tokens.
func (t *Tokenizer) SpecialID(literal string) (int32, bool) { return t.tokenizer.SpecialID(literal) }
