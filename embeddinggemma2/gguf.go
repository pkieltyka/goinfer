package embeddinggemma2

import (
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/pkieltyka/goinfer/internal/gguf"
	"github.com/pkieltyka/goinfer/tokenizer"
)

const ggufArchitecture = "gemma-embedding2"

// LoadGGUFEncoder loads a text EmbeddingGemma 2 GGUF with embedded tokenizer metadata.
// It supports the 24-layer Q8_0 and UD-Q6_K_XL weight profiles. Their tensors
// are expanded to FP32 once, including for an accelerator attached after loading.
// Prompts are the caller's named text prefixes. A nil table permits unprompted text
// and prepared input IDs. The returned encoder does not hold the file open.
func LoadGGUFEncoder(path string, prompts map[string]string) (*Encoder, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: GGUF: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: GGUF: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("embeddinggemma2: GGUF requires a regular file")
	}
	encoder, err := LoadGGUFEncoderReaderAt(file, info.Size(), prompts)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: GGUF %q: %w", path, err)
	}
	return encoder, nil
}

// LoadGGUFEncoderReaderAt loads the same text checkpoint formats as LoadGGUFEncoder
// from the size bytes in r. Keep those bytes unchanged until the call returns.
// The encoder and its tokenizer retain neither r nor its backing data.
func LoadGGUFEncoderReaderAt(r io.ReaderAt, size int64, prompts map[string]string) (*Encoder, error) {
	f, err := gguf.Open(r, size)
	if err != nil {
		return nil, err
	}
	cfg, err := ggufConfig(f)
	if err != nil {
		return nil, err
	}
	quant, err := ggufQuant(f)
	if err != nil {
		return nil, err
	}
	model := &Model{cfg: cfg, layers: make([]layer, cfg.Layers)}
	refs, scalars, err := model.plan()
	if err != nil {
		return nil, err
	}
	refs = append(refs, weightRef{&model.embed, "embed_tokens.weight", []int{cfg.VocabSize, cfg.Hidden}})
	if got := f.TensorCount(); got != len(refs) {
		return nil, fmt.Errorf("tensor count %d, want %d for the text encoder", got, len(refs))
	}
	tensors := make([]gguf.Tensor, 0, len(refs))
	for _, ref := range refs {
		name, err := ggufWeightName(ref.name)
		if err != nil {
			return nil, err
		}
		t, ok := f.Tensor(name)
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		if err := validateGGUFTensor(t, name, ref.shape, quant); err != nil {
			return nil, err
		}
		tensors = append(tensors, t)
	}
	tok, err := tokenizer.LoadGGUFReaderAt(r, size)
	if err != nil {
		return nil, err
	}
	if sp := tok.Special(); tok.Size() != cfg.VocabSize || sp.BOS != 2 || sp.EOS != 1 {
		return nil, fmt.Errorf("tokenizer vocabulary or special IDs disagree with the EmbeddingGemma 2 profile")
	}
	if _, err := tok.Encode("a", false); err != nil {
		return nil, fmt.Errorf("tokenizer cannot encode: %w", err)
	}
	encoder, err := NewEncoder(model, tok, prompts)
	if err != nil {
		return nil, err
	}
	if encoder.bos != 2 || encoder.eos != 1 {
		return nil, fmt.Errorf("tokenizer <bos>/<eos> pieces do not match their declared IDs")
	}
	for i, t := range tensors {
		*refs[i].dst, err = t.ReadFloat32()
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", refs[i].name, err)
		}
	}
	model.setScalars(scalars)
	return encoder, nil
}

// ggufConfig admits the text architecture that the encoder and device graphs implement.
func ggufConfig(f *gguf.File) (Config, error) {
	required := []struct {
		key  string
		want any
	}{
		{"general.architecture", ggufArchitecture},
		{"general.type", "model"},
		{"gemma-embedding2.block_count", uint32(24)},
		{"gemma-embedding2.context_length", uint32(262144)},
		{"gemma-embedding2.embedding_length", uint32(512)},
		{"gemma-embedding2.embedding_length_out", uint32(768)},
		{"gemma-embedding2.embedding_length_per_layer_input", uint32(512)},
		{"gemma-embedding2.feed_forward_length", uint32(2048)},
		{"gemma-embedding2.attention.head_count", uint32(4)},
		{"gemma-embedding2.attention.key_length", uint32(512)},
		{"gemma-embedding2.attention.value_length", uint32(512)},
		{"gemma-embedding2.attention.key_length_swa", uint32(256)},
		{"gemma-embedding2.attention.value_length_swa", uint32(256)},
		{"gemma-embedding2.attention.sliding_window", uint32(1024)},
		{"gemma-embedding2.attention.shared_kv_layers", uint32(0)},
		{"gemma-embedding2.attention.causal", false},
		{"gemma-embedding2.pooling_type", uint32(1)},
		{"gemma-embedding2.rope.dimension_count", uint32(512)},
		{"gemma-embedding2.rope.dimension_count_swa", uint32(256)},
		{"gemma-embedding2.rope.freq_base", float32(1e6)},
		{"gemma-embedding2.rope.freq_base_swa", float32(1e4)},
		{"tokenizer.ggml.model", "gemma4"},
		{"tokenizer.ggml.add_bos_token", true},
		{"tokenizer.ggml.add_eos_token", true},
		{"tokenizer.ggml.add_space_prefix", false},
	}
	for _, m := range required {
		if got := f.Meta(m.key); got != m.want {
			return Config{}, fmt.Errorf("metadata %q is %v, want %v", m.key, got, m.want)
		}
	}
	for _, key := range []string{"type", "factor", "attn_factor", "original_context_length", "finetuned"} {
		key = ggufArchitecture + ".rope.scaling." + key
		if f.Meta(key) != nil {
			return Config{}, fmt.Errorf("unsupported metadata %q", key)
		}
	}
	const layers = 24
	window, _ := f.Meta("gemma-embedding2.attention.sliding_window_pattern").([]bool)
	kvHeads, _ := f.Meta("gemma-embedding2.attention.head_count_kv").([]int32)
	if len(window) != layers || len(kvHeads) != layers {
		return Config{}, fmt.Errorf("metadata sliding-window pattern and KV head counts must each have %d entries", layers)
	}
	eps, ok := f.Meta("gemma-embedding2.attention.layer_norm_rms_epsilon").(float32)
	if !ok || eps <= 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return Config{}, fmt.Errorf("metadata layer_norm_rms_epsilon must be positive and finite")
	}
	cfg := Config{
		VocabSize: 262144, Hidden: 512, Intermediate: 2048, Layers: layers, Heads: 4,
		KVHeads: 2, HeadDim: 256, SlidingWindow: 512, PLEDim: 512, EmbeddingDim: 768,
		// Expose the effective input cap, which is lower than the file's position count.
		MaxPositions: MaxTokens, HiddenAct: "gelu_pytorch_tanh", RMSNormEps: float64(eps),
		LayerTypes: make([]string, layers),
		PerLayerConfig: make(map[string]struct {
			HeadDim int `json:"head_dim"`
			KVHeads int `json:"num_key_value_heads"`
		}),
		RopeParameters: map[string]struct {
			Theta float64 `json:"rope_theta"`
			Type  string  `json:"rope_type"`
		}{
			"full_attention":    {Theta: 1e6, Type: "default"},
			"sliding_attention": {Theta: 1e4, Type: "default"},
		},
	}
	for i, sliding := range window {
		if sliding && kvHeads[i] == 2 {
			cfg.LayerTypes[i] = "sliding_attention"
			continue
		}
		if sliding || kvHeads[i] != 1 {
			return Config{}, fmt.Errorf("layer %d has sliding=%t and unsupported KV head count %d", i, sliding, kvHeads[i])
		}
		cfg.LayerTypes[i] = "full_attention"
		override := cfg.PerLayerConfig[strconv.Itoa(i)]
		override.HeadDim, override.KVHeads = 512, 1
		cfg.PerLayerConfig[strconv.Itoa(i)] = override
	}
	return cfg, nil
}

func ggufQuant(f *gguf.File) (gguf.Type, error) {
	if got := f.Meta("general.quantization_version"); got != uint32(2) {
		return 0, fmt.Errorf("GGUF quantization version %v, want 2", got)
	}
	switch got := f.Meta("general.file_type"); got {
	case uint32(7):
		return gguf.Q8_0, nil
	case uint32(18):
		return gguf.Q6_K, nil
	default:
		return 0, fmt.Errorf("GGUF file_type %v, want Q8_0 (7) or Q6_K (18)", got)
	}
}

func validateGGUFTensor(t gguf.Tensor, name string, shape []int, quant gguf.Type) error {
	if len(t.Shape) != len(shape) {
		return fmt.Errorf("tensor %q rank %d, want %d", name, len(t.Shape), len(shape))
	}
	for i, dim := range shape {
		if t.Shape[len(t.Shape)-1-i] != uint64(dim) {
			return fmt.Errorf("tensor %q shape %v, want reverse of %v", name, t.Shape, shape)
		}
	}
	var want gguf.Type
	switch {
	case len(shape) == 1:
		want = gguf.F32
	case name == "per_layer_model_proj.weight":
		want = gguf.BF16
	case strings.HasSuffix(name, ".inp_gate.weight"), strings.HasSuffix(name, ".proj.weight"):
		want = gguf.Q8_0
	default:
		want = quant
	}
	if t.Type != want {
		return fmt.Errorf("tensor %q GGML type %d, want %d", name, t.Type, want)
	}
	return nil
}

var ggufGlobalWeights = map[string]string{
	"embed_tokens.weight":                   "token_embd.weight",
	"ple.per_layer_model_projection.weight": "per_layer_model_proj.weight",
	"ple.per_layer_projection_norm.weight":  "per_layer_proj_norm.weight",
	"norm.weight":                           "output_norm.weight",
	"embedding_projection.weight":           "output.weight",
}

var ggufLayerWeights = map[string]string{
	"input_layernorm.weight":                     "attn_norm.weight",
	"self_attn.q_proj.weight":                    "attn_q.weight",
	"self_attn.k_proj.weight":                    "attn_k.weight",
	"self_attn.v_proj.weight":                    "attn_v.weight",
	"self_attn.o_proj.weight":                    "attn_output.weight",
	"self_attn.q_norm.weight":                    "attn_q_norm.weight",
	"self_attn.k_norm.weight":                    "attn_k_norm.weight",
	"post_attention_layernorm.weight":            "post_attention_norm.weight",
	"pre_feedforward_layernorm.weight":           "ffn_norm.weight",
	"mlp.gate_proj.weight":                       "ffn_gate.weight",
	"mlp.up_proj.weight":                         "ffn_up.weight",
	"mlp.down_proj.weight":                       "ffn_down.weight",
	"post_feedforward_layernorm.weight":          "post_ffw_norm.weight",
	"ple_block.per_layer_input_gate.weight":      "inp_gate.weight",
	"ple_block.per_layer_projection.weight":      "proj.weight",
	"ple_block.post_per_layer_input_norm.weight": "post_norm.weight",
	"layer_scalar":                               "layer_output_scale.weight",
}

func ggufWeightName(name string) (string, error) {
	if gguf := ggufGlobalWeights[name]; gguf != "" {
		return gguf, nil
	}
	if layer, role, ok := strings.Cut(strings.TrimPrefix(name, "layers."), "."); ok {
		if gguf := ggufLayerWeights[role]; gguf != "" {
			return "blk." + layer + "." + gguf, nil
		}
	}
	return "", fmt.Errorf("no GGUF tensor mapping for %q", name)
}
