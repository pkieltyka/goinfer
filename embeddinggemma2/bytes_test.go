package embeddinggemma2

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/pkieltyka/goinfer/tokenizer"
)

// stEntry is one tensor of a safetensors header.
type stEntry struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

// stTensor is one tensor's header entry with its bytes.
type stTensor struct {
	stEntry
	raw []byte
}

// readST parses a safetensors file into its tensors, each with the bytes the file stores for it.
func readST(t testing.TB, file []byte) map[string]stTensor {
	t.Helper()
	n := int(binary.LittleEndian.Uint64(file))
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(file[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	payload := file[8+n:]
	out := map[string]stTensor{}
	for name, raw := range hdr {
		if name == "__metadata__" {
			continue
		}
		var e stEntry
		if err := json.Unmarshal(raw, &e); err != nil {
			t.Fatal(err)
		}
		out[name] = stTensor{e, payload[e.Offsets[0]:e.Offsets[1]]}
	}
	return out
}

// writeST writes tensors as a safetensors file: names sorted, bytes unchanged and in that order, the header padded
// with spaces so the payload starts on a multiple of 8.
func writeST(tensors map[string]stTensor) []byte {
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	var hdr, payload bytes.Buffer
	hdr.WriteByte('{')
	for i, n := range names {
		tn := tensors[n]
		e := stEntry{tn.DType, tn.Shape, [2]int{payload.Len(), payload.Len() + len(tn.raw)}}
		payload.Write(tn.raw)
		k, _ := json.Marshal(n)
		v, _ := json.Marshal(e)
		if i > 0 {
			hdr.WriteByte(',')
		}
		hdr.Write(k)
		hdr.WriteByte(':')
		hdr.Write(v)
	}
	hdr.WriteByte('}')
	for (8+hdr.Len())%8 != 0 {
		hdr.WriteByte(' ')
	}
	out := make([]byte, 8, 8+hdr.Len()+payload.Len())
	binary.LittleEndian.PutUint64(out, uint64(hdr.Len()))
	out = append(out, hdr.Bytes()...)
	return append(out, payload.Bytes()...)
}

const tokensName = "language_model.embed_tokens.weight"

// splitText is the text-only form NewModel takes: the token table's bytes as stored, and a safetensors file of every
// other language_model.* tensor, names and bytes unchanged.
func splitText(t testing.TB, tensors map[string]stTensor) (tokens, layers []byte, count int) {
	t.Helper()
	rest := map[string]stTensor{}
	for n, tn := range tensors {
		if strings.HasPrefix(n, "language_model.") && n != tokensName {
			rest[n] = tn
		}
	}
	tok, ok := tensors[tokensName]
	if !ok {
		t.Fatalf("no %s", tokensName)
	}
	return tok.raw, writeST(rest), len(rest)
}

// asBF16 returns float32 tensors as bfloat16 (the low 16 bits dropped) and, as their twin, the same values widened
// back to float32: two files that hold exactly the same numbers in the two dtypes.
func asBF16(t testing.TB, tensors map[string]stTensor) (bf, f32 map[string]stTensor) {
	t.Helper()
	bf, f32 = map[string]stTensor{}, map[string]stTensor{}
	for n, tn := range tensors {
		if tn.DType != "F32" {
			t.Fatalf("%s is %s, the tiny fixture should be F32", n, tn.DType)
		}
		b := make([]byte, len(tn.raw)/2)
		w := make([]byte, len(tn.raw))
		for i := 0; i < len(tn.raw)/4; i++ {
			h := uint16(binary.LittleEndian.Uint32(tn.raw[4*i:]) >> 16)
			binary.LittleEndian.PutUint16(b[2*i:], h)
			binary.LittleEndian.PutUint32(w[4*i:], uint32(h)<<16)
		}
		bf[n] = stTensor{stEntry{DType: "BF16", Shape: tn.Shape}, b}
		f32[n] = stTensor{stEntry{DType: "F32", Shape: tn.Shape}, w}
	}
	return bf, f32
}

func tinyParts(t testing.TB) (Config, map[string]stTensor) {
	t.Helper()
	cfg, err := LoadConfig(tinyDir)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.ReadFile(tinyDir + "/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	return cfg, readST(t, file)
}

func sameBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// sameOutputs fails unless the two models give bit-identical hidden states and embeddings on every golden case.
func sameOutputs(t *testing.T, what string, a, b *Model, g tinyGolden) {
	t.Helper()
	for ci, c := range g.Cases {
		ha, err := a.Hidden(c.IDs)
		if err != nil {
			t.Fatal(err)
		}
		hb, err := b.Hidden(c.IDs)
		if err != nil {
			t.Fatal(err)
		}
		ea, _ := a.Embed(c.IDs)
		eb, _ := b.Embed(c.IDs)
		if !sameBits(ha, hb) || !sameBits(ea, eb) {
			t.Errorf("%s, case %d (%d tokens): the two models differ", what, ci, len(c.IDs))
		}
	}
}

// TestNewModel_matchesLoad: a model built from the tiny checkpoint's bytes, split into the token table and the other
// tensors, gives the same config and, bit for bit, the same hidden states and embeddings as Load on every golden case.
func TestNewModel_matchesLoad(t *testing.T) {
	want, g := loadTiny(t)
	cfg, tensors := tinyParts(t)
	tokens, layers, n := splitText(t, tensors)
	if n != len(tensors)-1 {
		t.Fatalf("%d tensors beside the token table, the file has %d", n, len(tensors))
	}
	got, err := NewModel(cfg, tokens, layers)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dim() != want.Dim() || fmt.Sprint(got.Config()) != fmt.Sprint(want.Config()) {
		t.Fatal("config differs")
	}
	sameOutputs(t, "float32", got, want, g)
}

func TestTransformerWeights_keepsPackedTokens(t *testing.T) {
	cfg, tensors := tinyParts(t)
	tokens, layers, _ := splitText(t, tensors)
	model, err := NewModel(cfg, tokens, layers)
	if err != nil {
		t.Fatal(err)
	}
	weights := model.TransformerWeights()
	if weights.Embed != nil || model.embed != nil {
		t.Fatal("TransformerWeights materialized the token table")
	}
	full := model.Weights()
	if len(full.Embed) != cfg.VocabSize*cfg.Hidden {
		t.Fatalf("Weights returned %d token values", len(full.Embed))
	}
	full.Embed = nil
	if !reflect.DeepEqual(weights, full) {
		t.Fatal("transformer weights differ from Weights")
	}
}

// TestNewModel_bfloat16: the same numbers stored as bfloat16 (token table and layers) and as float32 give
// bit-identical outputs, so the widening of a row per token and of each layer tensor is exact; and the rows the
// encoder starts from are the stored values widened and scaled.
func TestNewModel_bfloat16(t *testing.T) {
	_, g := loadTiny(t)
	cfg, tensors := tinyParts(t)
	bf, f32 := asBF16(t, tensors)
	tb, lb, _ := splitText(t, bf)
	tf, lf, _ := splitText(t, f32)
	mb, err := NewModel(cfg, tb, lb)
	if err != nil {
		t.Fatal(err)
	}
	mf, err := NewModel(cfg, tf, lf)
	if err != nil {
		t.Fatal(err)
	}
	sameOutputs(t, "bfloat16 against its float32 twin", mb, mf, g)
	ids := []int{0, 1, cfg.VocabSize - 1}
	x, err := mb.EmbedTokens(ids)
	if err != nil {
		t.Fatal(err)
	}
	scale := float32(math.Sqrt(float64(cfg.Hidden)))
	for r, id := range ids {
		for j := range cfg.Hidden {
			h := binary.LittleEndian.Uint16(tb[(id*cfg.Hidden+j)*2:])
			if want := math.Float32frombits(uint32(h)<<16) * scale; math.Float32bits(x[r*cfg.Hidden+j]) != math.Float32bits(want) {
				t.Fatalf("token %d value %d: %v, want %v", id, j, x[r*cfg.Hidden+j], want)
			}
		}
	}
	// Mixed dtypes are fine too: a bfloat16 table over float32 layers.
	mm, err := NewModel(cfg, tb, lf)
	if err != nil {
		t.Fatal(err)
	}
	sameOutputs(t, "bfloat16 table over float32 layers", mm, mf, g)
}

// TestNewModel_ownsItsLayers: NewModel does not keep the layers bytes (overwriting them afterwards changes nothing,
// for float32 and for bfloat16 files), does not modify the token table, and does read the table it was given.
func TestNewModel_ownsItsLayers(t *testing.T) {
	_, g := loadTiny(t)
	cfg, tensors := tinyParts(t)
	bf, _ := asBF16(t, tensors)
	for name, set := range map[string]map[string]stTensor{"float32": tensors, "bfloat16": bf} {
		tokens, layers, _ := splitText(t, set)
		tokens = bytes.Clone(tokens)
		before := bytes.Clone(tokens)
		ref, err := NewModel(cfg, tokens, bytes.Clone(layers))
		if err != nil {
			t.Fatal(err)
		}
		m, err := NewModel(cfg, tokens, layers)
		if err != nil {
			t.Fatal(err)
		}
		for i := range layers {
			layers[i] = 0xA5
		}
		sameOutputs(t, name+" after the layers bytes were overwritten", m, ref, g)
		if !bytes.Equal(tokens, before) {
			t.Fatalf("%s: NewModel or Embed changed the token table", name)
		}
		// The table is used where it lies: change one stored row and that token's input row changes.
		id := g.Cases[len(g.Cases)-1].IDs[0]
		x0, _ := m.EmbedTokens([]int{id})
		for j := range cfg.Hidden * len(tokens) / (cfg.VocabSize * cfg.Hidden) {
			tokens[id*cfg.Hidden*len(tokens)/(cfg.VocabSize*cfg.Hidden)+j] ^= 0x40
		}
		x1, _ := m.EmbedTokens([]int{id})
		if sameBits(x0, x1) {
			t.Fatalf("%s: the model does not read the caller's table", name)
		}
	}
}

// TestNewModel_weightsForAnAccelerator: Weights hands an accelerator the whole table as float32, equal to Load's, and
// the same slice on a second call.
func TestNewModel_weightsForAnAccelerator(t *testing.T) {
	want, _ := loadTiny(t)
	cfg, tensors := tinyParts(t)
	tokens, layers, _ := splitText(t, tensors)
	m, err := NewModel(cfg, tokens, layers)
	if err != nil {
		t.Fatal(err)
	}
	w := m.Weights()
	if !sameBits(w.Embed, want.Weights().Embed) {
		t.Fatal("the widened table differs from Load's")
	}
	if again := m.Weights(); &again.Embed[0] != &w.Embed[0] {
		t.Fatal("the table was widened twice")
	}
	if !sameBits(w.Layers[0].Q, want.Weights().Layers[0].Q) || w.Layers[1].LayerScalar != want.Weights().Layers[1].LayerScalar {
		t.Fatal("layer weights differ from Load's")
	}
}

// TestNewModel_refuses: a token table of the wrong length, bytes that are not a safetensors file, a missing tensor, a
// tensor of the wrong shape or of a dtype that is not a float, and an incomplete config are errors, not a model.
func TestNewModel_refuses(t *testing.T) {
	cfg, tensors := tinyParts(t)
	tokens, layers, _ := splitText(t, tensors)
	edit := func(f func(map[string]stTensor)) []byte {
		c := map[string]stTensor{}
		for n, tn := range tensors {
			c[n] = tn
		}
		f(c)
		_, l, _ := splitText(t, c)
		return l
	}
	const q = "language_model.layers.0.self_attn.q_proj.weight"
	bad := cfg
	bad.LayerTypes = bad.LayerTypes[:1]
	for name, c := range map[string]struct {
		cfg            Config
		tokens, layers []byte
		want           string
	}{
		"a row short":          {cfg, tokens[:len(tokens)-4*cfg.Hidden], layers, "token table"},
		"an odd length":        {cfg, tokens[:len(tokens)-1], layers, "token table"},
		"no table":             {cfg, nil, layers, "token table"},
		"not safetensors":      {cfg, tokens, []byte("not a safetensors file"), "layers"},
		"empty layers":         {cfg, tokens, nil, "layers"},
		"a missing tensor":     {cfg, tokens, edit(func(m map[string]stTensor) { delete(m, q) }), "self_attn.q_proj.weight"},
		"a wrong shape":        {cfg, tokens, edit(func(m map[string]stTensor) { x := m[q]; x.Shape = []int{x.Shape[0] * x.Shape[1]}; m[q] = x }), "shape"},
		"an integer tensor":    {cfg, tokens, edit(func(m map[string]stTensor) { x := m[q]; x.DType = "I32"; m[q] = x }), "dtype"},
		"an incomplete config": {bad, tokens, layers, "text_config incomplete"},
	} {
		m, err := NewModel(c.cfg, c.tokens, c.layers)
		if err == nil || m != nil {
			t.Errorf("%s: built a model", name)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not name %q", name, err, c.want)
		}
	}
}

// TestLoadConfigBytes: the bytes of a config.json give the config LoadConfig reads from the directory, and another
// model type is refused.
func TestLoadConfigBytes(t *testing.T) {
	raw, err := os.ReadFile(tinyDir + "/config.json")
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadConfigBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	want, err := LoadConfig(tinyDir)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatal("LoadConfigBytes and LoadConfig differ")
	}
	if _, err := LoadConfigBytes([]byte(`{"model_type":"gemma3","text_config":{}}`)); err == nil {
		t.Fatal("a gemma3 config loaded as embedding_gemma2")
	}
}

// byteTokenizer is a tokenizer.json with the specials, the 256 byte pieces and one merge over three pieces the tests'
// text never uses, so every id of that text is a byte piece, below the tiny model's vocabulary of 256.
func byteTokenizer(t testing.TB, specials ...string) []byte {
	t.Helper()
	vocab := map[string]int{}
	var added []map[string]any
	for i, s := range specials {
		vocab[s] = i
		added = append(added, map[string]any{"id": i, "content": s})
	}
	for b := range 256 {
		vocab[fmt.Sprintf("<0x%02X>", b)] = len(specials) + b
	}
	for i, w := range []string{"x", "y", "xy"} {
		vocab[w] = len(specials) + 256 + i
	}
	raw, err := json.Marshal(map[string]any{
		"added_tokens": added,
		"model":        map[string]any{"type": "BPE", "byte_fallback": true, "unk_token": "<unk>", "vocab": vocab, "merges": [][2]string{{"x", "y"}}},
		"decoder":      map[string]any{"type": "Sequence", "decoders": []any{map[string]any{"type": "ByteFallback"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// TestNewEncoder: an encoder built from a model made of bytes, a tokenizer loaded from its pack and a prompt table
// tokenizes as <bos>, prompt and text, <eos>, embeds what the model embeds for those ids, copies the prompt table, has
// no towers, and is refused without a model or a tokenizer. (A tokenizer without <bos> or <eos> cannot be loaded at
// all, so NewEncoder's check for them, the one LoadEncoder had, has no fixture.)
func TestNewEncoder(t *testing.T) {
	cfg, tensors := tinyParts(t)
	tokens, layers, _ := splitText(t, tensors)
	m, err := NewModel(cfg, tokens, layers)
	if err != nil {
		t.Fatal(err)
	}
	pack, err := tokenizer.BuildPack(byteTokenizer(t, "<pad>", "<eos>", "<bos>", "<unk>"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadPack(pack)
	if err != nil {
		t.Fatal(err)
	}
	prompts := map[string]string{"query": "q: ", "document": "d: "}
	e, err := NewEncoder(m, tok, prompts)
	if err != nil {
		t.Fatal(err)
	}
	prompts["query"] = "changed after the call"
	if p, _ := e.PromptText("query"); p != "q: " {
		t.Fatalf("the prompt table was not copied: %q", p)
	}
	if e.Model() != m || e.HiddenDim() != m.Dim() {
		t.Fatal("the encoder does not hold the model it was given")
	}
	ids, err := e.Tokenize("hi", "query")
	if err != nil {
		t.Fatal(err)
	}
	body, err := tok.Encode("q: hi", false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != len(body)+2 || ids[0] != 2 || ids[len(ids)-1] != 1 || fmt.Sprint(ids[1:len(ids)-1]) != fmt.Sprint(body) {
		t.Fatalf("ids %v, want <bos> %v <eos>", ids, body)
	}
	v, n, err := e.EmbedText("hi", "query")
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(ids) || !sameBits(v, want) {
		t.Fatal("EmbedText is not the model's embedding of the tokenized text")
	}
	if e.EnableVision() == nil || e.EnableAudio() == nil {
		t.Fatal("an encoder with no checkpoint directory enabled a tower")
	}
	if e2, err := NewEncoder(m, tok, nil); err != nil || len(e2.PromptNames()) != 0 {
		t.Fatalf("a nil prompt table: %v", err)
	}
	if _, err := NewEncoder(nil, tok, nil); err == nil {
		t.Fatal("no model")
	}
	if _, err := NewEncoder(m, nil, nil); err == nil {
		t.Fatal("no tokenizer")
	}
}
