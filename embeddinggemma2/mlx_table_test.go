package embeddinggemma2

import (
	"math"
	"reflect"
	"testing"
)

func issue94Pack4(q []byte) uint32 {
	return uint32(q[0]) | uint32(q[1])<<8 | uint32(q[2])<<16 | uint32(q[3])<<24
}

func issue94Table() (*mlxAffineTable, []float32) {
	const vocab, cols = 4, 64
	t := &mlxAffineTable{cols: cols, groups: 1}
	t.scales = []float32{0.5, 0.75, 1.25, 2.0}
	t.biases = []float32{-0.25, 0.125, -1.0, 0.5}
	raw := make([]float32, vocab*cols)
	for id := 0; id < vocab; id++ {
		for k := 0; k < cols; k += 4 {
			q := []byte{byte(id*17 + k), byte(id*17 + k + 1), byte(id*17 + k + 2), byte(id*17 + k + 3)}
			t.words = append(t.words, issue94Pack4(q))
			for j := range 4 {
				raw[id*cols+k+j] = float32(q[j])*t.scales[id] + t.biases[id]
			}
		}
	}
	return t, raw
}

func TestMLXTokenTablePackedEmbedTokens(t *testing.T) {
	table, raw := issue94Table()
	const H = 64
	m := &Model{cfg: Config{VocabSize: 4, Hidden: H}, embedAffine: table}
	if m.embed != nil {
		t.Fatal("a fresh packed model already has a dense table")
	}
	got, err := m.EmbedTokens([]int{0, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	scale := float32(math.Sqrt(H))
	for i, id := range []int{0, 2, 2} {
		for k := 0; k < H; k++ {
			want := raw[id*H+k] * scale
			if math.Float32bits(got[i*H+k]) != math.Float32bits(want) {
				t.Fatalf("EmbedTokens row %d col %d: %v != %v", i, k, got[i*H+k], want)
			}
		}
	}
	if m.embed != nil {
		t.Fatal("EmbedTokens materialized the dense table")
	}
	if _, err := m.EmbedTokens([]int{4}); err == nil {
		t.Fatal("out-of-range id accepted")
	}
	if _, err := m.EmbedTokens(nil); err == nil {
		t.Fatal("empty ids accepted")
	}
	repeat, err := m.EmbedTokens([]int{2, 2})
	if err != nil || !reflect.DeepEqual(repeat[:H], repeat[H:]) {
		t.Fatalf("repeated id differs: %v", err)
	}
}

func TestMLXTokenTableLazyWeights(t *testing.T) {
	table, raw := issue94Table()
	const H = 64
	m := &Model{cfg: Config{VocabSize: 4, Hidden: H}, embedAffine: table}
	tw := m.TransformerWeights()
	if tw.Embed != nil {
		t.Fatal("TransformerWeights materialized the table")
	}
	if m.embed != nil {
		t.Fatal("TransformerWeights materialized the dense table")
	}
	weights := make(chan Weights, 1)
	go func() { weights <- m.Weights() }()
	if rows, err := m.EmbedTokens([]int{1, 2}); err != nil || len(rows) != 2*H {
		t.Fatalf("EmbedTokens while widening weights: %v", err)
	}
	w := <-weights
	if m.embed == nil || len(w.Embed) != len(raw) {
		t.Fatalf("Weights did not materialize %d values (got %d)", len(raw), len(w.Embed))
	}
	for i := range raw {
		if math.Float32bits(w.Embed[i]) != math.Float32bits(raw[i]) {
			t.Fatalf("dense word %d: %v != %v", i, w.Embed[i], raw[i])
		}
	}
	w2 := m.Weights()
	if reflect.ValueOf(w.Embed).Pointer() != reflect.ValueOf(w2.Embed).Pointer() {
		t.Fatal("Weights widened the table more than once")
	}
	if got, err := m.EmbedTokens([]int{1}); err != nil || len(got) != H {
		t.Fatalf("EmbedTokens after Weights: %v", err)
	}
}

func TestMLXTokenTableLoaderKeepsPacked(t *testing.T) {
	cfg, packed, dense := mlx8Fixture(t)
	mlx, err := LoadEncoder(writeMLX8Fixture(t, cfg, mlx8TestMetadata, mlx8TestMetadata, packed))
	if err != nil {
		t.Fatal(err)
	}
	stock, err := LoadEncoder(writeMLX8Fixture(t, cfg, "", "", dense))
	if err != nil {
		t.Fatal(err)
	}
	m := mlx.Model()
	if m.embed != nil || m.embedAffine == nil {
		t.Fatal("MLX loader did not keep the token table packed")
	}
	ids, err := mlx.Tokenize("some text", "query")
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.EmbedTokens(ids)
	if err != nil {
		t.Fatal(err)
	}
	want, err := stock.Model().EmbedTokens(ids)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("packed EmbedTokens differs from the dense table")
	}
	if m.embed != nil {
		t.Fatal("EmbedTokens materialized the dense table")
	}
	_ = m.Weights()
	if m.embed == nil {
		t.Fatal("Weights did not materialize the dense table")
	}
	if !reflect.DeepEqual(m.Weights(), stock.Model().Weights()) {
		t.Fatal("lazy dense table differs from the independently written FP32 table")
	}
}
