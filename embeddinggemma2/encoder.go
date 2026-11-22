package embeddinggemma2

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/pkieltyka/goinfer/tokenizer"
)

// MaxTokens is the context the model card gives for every modality (8,192). An input longer than this is refused,
// not cut: a silently truncated input would embed a different text than the caller sent.
const MaxTokens = 8192

// MatryoshkaWidths are the embedding widths a server offers for OpenAI's `dimensions`. The model card lists 768, 512,
// 256 and 128; Gate 2's pre-registered truncation reading kept 768 and 512 and
// dropped the other two: on 341 sections of this repo's docs, heading-to-section recall@10 fell to 0.888x of 768's at
// 256 and 0.684x at 128, under the rule's 0.90x.
var MatryoshkaWidths = []int{768, 512}

// Encoder is an EmbeddingGemma 2 model with its tokenizer and its named task prompts
// (config_sentence_transformers.json's `prompts`). It implements aikit's encoder.Encoder (Encode, EncodeBatch,
// HiddenDim), where isQuery selects the checkpoint's own "query" prompt and false its "document" prompt, and offers
// EncodeTasks for a caller that names the prompt, or none. It is safe for concurrent use.
type Encoder struct {
	m       *Model
	tok     *tokenizer.Tokenizer
	prompts map[string]string
	bos     int
	eos     int
	accel   Accelerator                                    // nil: the CPU forward (UseAccelerator)
	dir     string                                         // the checkpoint directory (EnableVision reads the tower from it)
	vis     *visionTower                                   // nil until EnableVision
	aud     *audioTower                                    // nil until EnableAudio
	resize  ImageResize                                    // "" = the default (ImageResizeMode)
	onTower func(tower, device string, took time.Duration) // OnTowerLoad
}

// LoadEncoder loads the model, tokenizer.json and the prompt table from an HF checkpoint directory.
func LoadEncoder(dir string) (*Encoder, error) {
	m, err := Load(dir)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(dir)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: tokenizer: %w", err)
	}
	prompts := map[string]string{}
	if raw, err := os.ReadFile(filepath.Join(dir, "config_sentence_transformers.json")); err == nil {
		var st struct {
			Prompts map[string]string `json:"prompts"`
		}
		if err := json.Unmarshal(raw, &st); err != nil {
			return nil, fmt.Errorf("embeddinggemma2: config_sentence_transformers.json: %w", err)
		}
		prompts = st.Prompts
	}
	e, err := NewEncoder(m, tok, prompts)
	if err != nil {
		return nil, err
	}
	e.dir = dir
	return e, nil
}

// NewEncoder builds an Encoder from parts that are already in memory: a model (Load or NewModel), its tokenizer
// (tokenizer.Load, tokenizer.LoadJSONBytes or tokenizer.LoadPack) and the prompt table of
// config_sentence_transformers.json, which may be nil. The table is copied. LoadEncoder reads all three from a
// checkpoint directory; this is for a caller that already holds them and must read nothing by path.
//
// Such an encoder has no checkpoint directory, so EnableVision and EnableAudio refuse it. The text path is complete.
func NewEncoder(m *Model, tok *tokenizer.Tokenizer, prompts map[string]string) (*Encoder, error) {
	if m == nil || tok == nil {
		return nil, fmt.Errorf("embeddinggemma2: NewEncoder needs a model and a tokenizer")
	}
	bos, okB := tok.TokenID("<bos>")
	eos, okE := tok.TokenID("<eos>")
	if !okB || !okE {
		return nil, fmt.Errorf("embeddinggemma2: tokenizer has no <bos>/<eos>")
	}
	p := make(map[string]string, len(prompts))
	maps.Copy(p, prompts)
	return &Encoder{m: m, tok: tok, prompts: p, bos: bos, eos: eos}, nil
}

// Model returns the underlying text encoder.
func (e *Encoder) Model() *Model { return e.m }

// Tokenizer returns the tokenizer used by the encoder. Treat it as read-only.
func (e *Encoder) Tokenizer() *tokenizer.Tokenizer { return e.tok }

// HiddenDim is the embedding width (768).
func (e *Encoder) HiddenDim() int { return e.m.Dim() }

// PromptNames lists the checkpoint's named prompts, sorted.
func (e *Encoder) PromptNames() []string {
	names := make([]string, 0, len(e.prompts))
	for n := range e.prompts {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// PromptText returns the text a named prompt prepends. The empty name is no prompt.
func (e *Encoder) PromptText(name string) (string, error) {
	if name == "" {
		return "", nil
	}
	if p, ok := e.prompts[name]; ok {
		return p, nil
	}
	return "", fmt.Errorf("unknown task %q; this model's tasks are %s, or none", name, strings.Join(e.PromptNames(), ", "))
}

// Tokenize returns the ids the model sees for text under the named prompt: <bos>, the prompt and the text as one
// string, <eos>, as the checkpoint's tokenizer post-processor writes them. sentence-transformers mean-pools over all of
// them, the prompt and both special tokens included.
func (e *Encoder) Tokenize(text, prompt string) ([]int, error) {
	p, err := e.PromptText(prompt)
	if err != nil {
		return nil, err
	}
	ids, err := e.tok.Encode(p+text, false)
	if err != nil {
		return nil, err
	}
	out := make([]int, 0, len(ids)+2)
	out = append(out, e.bos)
	out = append(out, ids...)
	out = append(out, e.eos)
	if len(out) > MaxTokens {
		return nil, fmt.Errorf("input is %d tokens, over the model's %d-token context", len(out), MaxTokens)
	}
	return out, nil
}

// EmbedIDs returns the unit-length sentence embedding for pre-tokenized ids on whichever path the encoder runs: the
// selected accelerator when one is active (UseAccelerator), otherwise the CPU model. It does not re-tokenize or add
// prompts or special tokens, does not modify ids, and returns any accelerator error without falling back to CPU.
func (e *Encoder) EmbedIDs(ids []int) ([]float32, error) {
	return e.embedIDs(ids)
}

// batchForwarder runs prepared inputs in a single logical forward and returns
// each input's projected last hidden rows in request order.
type batchForwarder interface {
	ForwardBatch(ids [][]int) ([][]float32, error)
}

// EmbedIDsBatch returns the unit-length sentence embedding of each pre-tokenized input, in input order, as EmbedIDs
// would give each alone. For a fixed model pack, build, accelerator selector,
// physical device and driver, the same input produces identical float32 bits
// regardless of its position or neighbors in an admitted batch. The CPU runs
// each input in turn. An accelerator must implement batching, and each result
// is pooled as EmbedIDs pools it.
//
// The call is all or nothing: an empty batch, an empty input, or any error from the model or the accelerator returns
// nil and that error, with no partial result. The caller groups inputs to fit
// the accelerator's launch and memory limits, with at most MaxTokens ids per
// input. ids is unchanged and the returned vectors belong to the caller.
func (e *Encoder) EmbedIDsBatch(ids [][]int) ([][]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("embeddinggemma2: an empty batch")
	}
	for i, in := range ids {
		if len(in) == 0 {
			return nil, fmt.Errorf("embeddinggemma2: batch input %d has no ids", i)
		}
	}
	out := make([][]float32, len(ids))
	if e.accel == nil {
		for i, in := range ids {
			v, err := e.embedIDs(in)
			if err != nil {
				return nil, fmt.Errorf("embeddinggemma2: batch input %d: %w", i, err)
			}
			out[i] = v
		}
		return out, nil
	}
	b, ok := e.accel.(batchForwarder)
	if !ok {
		return nil, fmt.Errorf("embeddinggemma2: the %s accelerator cannot run a batch", e.accel.Name())
	}
	hs, err := b.ForwardBatch(ids)
	if err != nil {
		return nil, err
	}
	d := e.m.cfg.EmbeddingDim
	if len(hs) != len(ids) {
		return nil, fmt.Errorf("embeddinggemma2: %s returned %d batch outputs for %d inputs", e.accel.Name(), len(hs), len(ids))
	}
	for i, h := range hs {
		if want := len(ids[i]) * d; len(h) != want {
			return nil, fmt.Errorf("embeddinggemma2: %s returned %d hidden values for batch input %d, want %d", e.accel.Name(), len(h), i, want)
		}
	}
	for i, h := range hs {
		out[i] = poolNormalize(h, len(ids[i]), d)
	}
	return out, nil
}

// EmbedText is the sentence embedding of text under the named prompt ("" for none), unit length.
func (e *Encoder) EmbedText(text, prompt string) ([]float32, int, error) {
	ids, err := e.Tokenize(text, prompt)
	if err != nil {
		return nil, 0, err
	}
	v, err := e.embedIDs(ids)
	return v, len(ids), err
}

// EncodeTasks embeds every text under one named prompt ("" for none) and reports each input's token count.
func (e *Encoder) EncodeTasks(texts []string, prompt string) ([][]float32, []int, error) {
	vecs := make([][]float32, len(texts))
	counts := make([]int, len(texts))
	for i, t := range texts {
		v, n, err := e.EmbedText(t, prompt)
		if err != nil {
			return nil, nil, fmt.Errorf("input %d: %w", i, err)
		}
		vecs[i], counts[i] = v, n
	}
	return vecs, counts, nil
}

// queryDoc maps aikit's isQuery flag onto the checkpoint's own query and document prompts.
func queryDoc(isQuery bool) string {
	if isQuery {
		return "query"
	}
	return "document"
}

// Encode implements aikit's encoder.Encoder: the "query" prompt when isQuery, else the "document" prompt.
func (e *Encoder) Encode(text string, isQuery bool) ([]float32, error) {
	v, _, err := e.EmbedText(text, queryDoc(isQuery))
	return v, err
}

// EncodeBatch implements aikit's encoder.Encoder. concurrency is ignored: the matmuls parallelise internally.
func (e *Encoder) EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error) {
	vecs, _, err := e.EncodeBatchCounted(texts, isQueries, concurrency)
	return vecs, err
}

// EncodeBatchCounted is EncodeBatch with each input's token count.
func (e *Encoder) EncodeBatchCounted(texts []string, isQueries []bool, _ int) ([][]float32, []int, error) {
	vecs := make([][]float32, len(texts))
	counts := make([]int, len(texts))
	for i, t := range texts {
		q := i < len(isQueries) && isQueries[i]
		v, n, err := e.EmbedText(t, queryDoc(q))
		if err != nil {
			return nil, nil, fmt.Errorf("input %d: %w", i, err)
		}
		vecs[i], counts[i] = v, n
	}
	return vecs, counts, nil
}

// CountTokens is the token count Encode would see.
func (e *Encoder) CountTokens(text string, isQuery bool) int {
	ids, err := e.Tokenize(text, queryDoc(isQuery))
	if err != nil {
		return 0
	}
	return len(ids)
}
