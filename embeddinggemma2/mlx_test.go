package embeddinggemma2

import (
	"encoding/binary"
	"encoding/json"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

const mlx8TestMetadata = `{"bits":8,"group_size":64,"mode":"affine"}`

func mlx8Fixture(t *testing.T) ([]byte, map[string]stTensor, map[string]stTensor) {
	t.Helper()
	cfg := Config{
		VocabSize: 272, Hidden: 128, Intermediate: 128, Layers: 1, Heads: 2, KVHeads: 1,
		HeadDim: 64, SlidingWindow: 3, PLEDim: 64, EmbeddingDim: 16,
		RMSNormEps: 1e-6, HiddenAct: "gelu_pytorch_tanh", LayerTypes: []string{"sliding_attention"},
	}
	if err := json.Unmarshal([]byte(`{"sliding_attention":{"rope_theta":10000,"rope_type":"default"}}`), &cfg.RopeParameters); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(composite{ModelType: "embedding_gemma2", TextConfig: cfg})
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{cfg: cfg, layers: make([]layer, cfg.Layers)}
	refs, _, err := m.plan()
	if err != nil {
		t.Fatal(err)
	}
	refs = append(refs, weightRef{name: "embed_tokens.weight", shape: []int{cfg.VocabSize, cfg.Hidden}})
	packed, dense := map[string]stTensor{}, map[string]stTensor{}
	entry := func(dtype string, shape []int, data []byte) stTensor {
		return stTensor{stEntry: stEntry{DType: dtype, Shape: shape}, raw: data}
	}
	for i, ref := range refs {
		name := "language_model." + ref.name
		out := ref.shape[0]
		if len(ref.shape) == 1 {
			data := make([]byte, 2*out)
			for r := range out {
				binary.LittleEndian.PutUint16(data[2*r:], 0x3f80)
			}
			packed[name] = entry("BF16", ref.shape, data)
			dense[name] = packed[name]
			continue
		}
		in := ref.shape[1]
		words, values := make([]byte, out*in), make([]byte, 4*out*in)
		groups := in / 64
		scales, biases := make([]byte, 2*out*groups), make([]byte, 2*out*groups)
		for r := range out {
			for g := range groups {
				offset := 2 * (r*groups + g)
				scale, bias := float32(g+1)/4096, -float32(g+1)/64
				binary.LittleEndian.PutUint16(scales[offset:], uint16(math.Float32bits(scale)>>16))
				binary.LittleEndian.PutUint16(biases[offset:], uint16(math.Float32bits(bias)>>16))
				for c := range 64 {
					at := r*in + g*64 + c
					q := byte(i*29 + r*17 + c*13)
					words[at] = q
					binary.LittleEndian.PutUint32(values[4*at:], math.Float32bits(float32(q)*scale+bias))
				}
			}
		}
		prefix := strings.TrimSuffix(name, ".weight")
		packed[name] = entry("U32", []int{out, in / 4}, words)
		packed[prefix+".scales"] = entry("BF16", []int{out, groups}, scales)
		packed[prefix+".biases"] = entry("BF16", []int{out, groups}, biases)
		dense[name] = entry("F32", ref.shape, values)
	}
	// The checkpoint also carries audio and vision tensors. Text loading must ignore them.
	packed["embed_audio.embedding_projection.weight"] = entry("U32", []int{1, 1}, make([]byte, 4))
	return raw, packed, dense
}

func writeMLX8Fixture(t *testing.T, config []byte, quantization, quantizationConfig string, tensors map[string]stTensor) string {
	t.Helper()
	dir := t.TempDir()
	if quantization != "" {
		config = []byte(strings.TrimSuffix(string(config), "}") + `,"quantization":` + quantization + "}")
	}
	if quantizationConfig != "" {
		config = []byte(strings.TrimSuffix(string(config), "}") + `,"quantization_config":` + quantizationConfig + "}")
	}
	for name, data := range map[string][]byte{
		"config.json": config, "model.safetensors": writeST(tensors),
		"tokenizer.json":                    byteTokenizer(t, "<pad>", "<eos>", "<bos>", "<unk>"),
		"config_sentence_transformers.json": []byte(`{"prompts":{"query":"q: "}}`),
	} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestLoadMLX8(t *testing.T) {
	cfg, packed, dense := mlx8Fixture(t)
	mlx, err := LoadEncoder(writeMLX8Fixture(t, cfg, mlx8TestMetadata, mlx8TestMetadata, packed))
	if err != nil {
		t.Fatal(err)
	}
	stock, err := LoadEncoder(writeMLX8Fixture(t, cfg, "", "", dense))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mlx.Model().Weights(), stock.Model().Weights()) {
		t.Fatal("reconstructed weights differ from independently written FP32 tensors")
	}
	ids, err := mlx.Tokenize("some text", "query")
	if err != nil {
		t.Fatal(err)
	}
	want, err := stock.EmbedIDs(ids)
	if err != nil {
		t.Fatal(err)
	}
	got, err := mlx.EmbedIDs(ids)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("packed and dense forwards differ: got %v, want %v, error %v", got, want, err)
	}
	for _, enable := range []func() error{mlx.EnableAudio, mlx.EnableVision} {
		if err := enable(); err == nil || !strings.Contains(err.Error(), "text only") {
			t.Fatalf("a media tower did not refuse an MLX affine checkpoint: %v", err)
		}
	}
}

func TestLoadMLX8Malformed(t *testing.T) {
	cfg, packed, _ := mlx8Fixture(t)
	const scales = "language_model.embed_tokens.scales"
	const biases = "language_model.embed_tokens.biases"
	const norm = "language_model.norm.weight"
	for _, tc := range []struct {
		name, quant, quantConfig, want string
		mutate                         func(map[string]stTensor)
	}{
		{"missing metadata", mlx8TestMetadata, "", "quantization_config", nil},
		{"conflicting metadata", mlx8TestMetadata, strings.Replace(mlx8TestMetadata, "64", "32", 1), "quantization_config", nil},
		{"wrong bits", strings.Replace(mlx8TestMetadata, ":8", ":4", 1), mlx8TestMetadata, "quantization", nil},
		{"unknown override", strings.TrimSuffix(mlx8TestMetadata, "}") + `,"language_model.embed_tokens":{"bits":4}}`, mlx8TestMetadata, "unknown field", nil},
		{"missing bias", mlx8TestMetadata, mlx8TestMetadata, biases, func(ts map[string]stTensor) { delete(ts, biases) }},
		{"mixed matrix", mlx8TestMetadata, mlx8TestMetadata, "expected U32", func(ts map[string]stTensor) {
			w := ts[tokensName]
			w.DType = "F32"
			ts[tokensName] = w
		}},
		{"wrong scales dtype", mlx8TestMetadata, mlx8TestMetadata, "expected BF16", func(ts map[string]stTensor) {
			scale := ts[scales]
			scale.DType = "F16"
			ts[scales] = scale
		}},
		{"wrong packed shape", mlx8TestMetadata, mlx8TestMetadata, tokensName, func(ts map[string]stTensor) {
			w := ts[tokensName]
			w.Shape = []int{w.Shape[0] * 2, w.Shape[1] / 2}
			ts[tokensName] = w
		}},
		{"wrong bias shape", mlx8TestMetadata, mlx8TestMetadata, biases, func(ts map[string]stTensor) {
			bias := ts[biases]
			bias.Shape = []int{bias.Shape[0] * 2, bias.Shape[1] / 2}
			ts[biases] = bias
		}},
		{"wrong norm dtype", mlx8TestMetadata, mlx8TestMetadata, norm, func(ts map[string]stTensor) {
			v := ts[norm]
			v.DType = "F16"
			ts[norm] = v
		}},
		{"no declared format", "", "", tokensName + "\" shape", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tensors := maps.Clone(packed)
			if tc.mutate != nil {
				tc.mutate(tensors)
			}
			dir := writeMLX8Fixture(t, cfg, tc.quant, tc.quantConfig, tensors)
			if m, err := Load(dir); m != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected a refusal containing %q, got model=%t error=%v", tc.want, m != nil, err)
			}
		})
	}
}
