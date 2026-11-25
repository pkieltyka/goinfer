package embeddinggemma2

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/pkieltyka/goinfer/decoder/quant"
	"github.com/pkieltyka/goinfer/internal/modelcore/embed"
)

const (
	mlxAffineBits      = 8
	mlxAffineGroupSize = 64
)

// mlxAffine8Config admits the uniform affine layout used by the MLX Community checkpoint.
// Both declarations must describe the same layout. Per-module overrides are unsupported.
func mlxAffine8Config(raw []byte) (bool, error) {
	var cfg struct {
		Quantization       json.RawMessage `json:"quantization"`
		QuantizationConfig json.RawMessage `json:"quantization_config"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return false, fmt.Errorf("embeddinggemma2: config.json: %w", err)
	}
	absent := func(msg json.RawMessage) bool { return len(msg) == 0 || bytes.Equal(msg, []byte("null")) }
	if absent(cfg.Quantization) && absent(cfg.QuantizationConfig) {
		return false, nil
	}
	type layout struct {
		Bits      int    `json:"bits"`
		GroupSize int    `json:"group_size"`
		Mode      string `json:"mode"`
	}
	want := layout{Bits: mlxAffineBits, GroupSize: mlxAffineGroupSize, Mode: "affine"}
	for _, field := range []struct {
		name string
		raw  json.RawMessage
	}{{"quantization", cfg.Quantization}, {"quantization_config", cfg.QuantizationConfig}} {
		var got layout
		if absent(field.raw) {
			return false, fmt.Errorf("embeddinggemma2: config.json: missing MLX %s", field.name)
		}
		dec := json.NewDecoder(bytes.NewReader(field.raw))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&got); err != nil {
			return false, fmt.Errorf("embeddinggemma2: config.json: %s: %w", field.name, err)
		}
		if got != want {
			return false, fmt.Errorf("embeddinggemma2: config.json: %s has bits=%d group_size=%d mode=%q, want 8-bit affine with group_size 64",
				field.name, got.Bits, got.GroupSize, got.Mode)
		}
	}
	return true, nil
}

func readMLXAffine8(st *embed.SafetensorsFile, name string, shape []int) ([]float32, error) {
	if len(shape) == 1 {
		return mlxBF16(st, name, shape...)
	}
	out, in := shape[0], shape[1]
	if out <= 0 || in <= 0 || in%mlxAffineGroupSize != 0 || out > math.MaxInt/4/in {
		return nil, fmt.Errorf("embeddinggemma2: MLX tensor %q has unsupported logical shape %v", name, shape)
	}
	packedCols, groups := in/4, in/mlxAffineGroupSize
	words, err := quant.TensorU32(st, name, out, packedCols)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: MLX: %w", err)
	}
	base := strings.TrimSuffix(name, ".weight")
	scales, err := mlxBF16(st, base+".scales", out, groups)
	if err != nil {
		return nil, err
	}
	biases, err := mlxBF16(st, base+".biases", out, groups)
	if err != nil {
		return nil, err
	}
	values := make([]float32, out*in)
	for row := range out {
		err := quant.MLXDequantAffineRow(
			words[row*packedCols:(row+1)*packedCols],
			scales[row*groups:(row+1)*groups],
			biases[row*groups:(row+1)*groups],
			in, mlxAffineBits, mlxAffineGroupSize, values[row*in:(row+1)*in],
		)
		if err != nil {
			return nil, fmt.Errorf("embeddinggemma2: MLX tensor %q row %d: %w", name, row, err)
		}
	}
	return values, nil
}

// mlxAffineTable is the token table kept in its packed affine-8 form. It owns
// the codes and the BF16-widened FP32 scale/bias companions; the safetensors
// mapping is not retained.
type mlxAffineTable struct {
	words  []uint32  // [vocab, hidden/4], four unsigned codes per word
	scales []float32 // [vocab, hidden/64]
	biases []float32 // [vocab, hidden/64]
	cols   int       // hidden
	groups int       // hidden/64
}

func readMLXAffine8Table(st *embed.SafetensorsFile, name string, shape []int) (*mlxAffineTable, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("embeddinggemma2: MLX token table shape %v", shape)
	}
	out, in := shape[0], shape[1]
	if out <= 0 || in <= 0 || in%mlxAffineGroupSize != 0 || out > math.MaxInt/4/in {
		return nil, fmt.Errorf("embeddinggemma2: MLX token table unsupported shape %v", shape)
	}
	packedCols, groups := in/4, in/mlxAffineGroupSize
	words, err := quant.TensorU32(st, name, out, packedCols)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: MLX token table: %w", err)
	}
	base := strings.TrimSuffix(name, ".weight")
	scales, err := mlxBF16(st, base+".scales", out, groups)
	if err != nil {
		return nil, err
	}
	biases, err := mlxBF16(st, base+".biases", out, groups)
	if err != nil {
		return nil, err
	}
	return &mlxAffineTable{
		words:  append([]uint32(nil), words...),
		scales: scales,
		biases: biases,
		cols:   in,
		groups: groups,
	}, nil
}

// row decodes one token row, without the sqrt(hidden) EmbedTokens scale.
func (t *mlxAffineTable) row(id int, dst []float32) error {
	if t == nil || id < 0 || id >= len(t.words)/(t.cols/4) || len(dst) < t.cols {
		return fmt.Errorf("embeddinggemma2: MLX token table row %d out of range", id)
	}
	packed := t.cols / 4
	return quant.MLXDequantAffineRow(
		t.words[id*packed:(id+1)*packed],
		t.scales[id*t.groups:(id+1)*t.groups],
		t.biases[id*t.groups:(id+1)*t.groups],
		t.cols, mlxAffineBits, mlxAffineGroupSize, dst[:t.cols],
	)
}

func mlxBF16(st *embed.SafetensorsFile, name string, shape ...int) ([]float32, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: MLX: %w", err)
	}
	if t.DType != "BF16" {
		return nil, fmt.Errorf("embeddinggemma2: MLX tensor %q: expected BF16, got %s", name, t.DType)
	}
	// TensorF32 checks the shape and widens BF16 into Go-owned memory.
	v, err := st.TensorF32(name, shape...)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: MLX: %w", err)
	}
	return v, nil
}
