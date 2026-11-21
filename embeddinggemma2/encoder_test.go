package embeddinggemma2

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/pkieltyka/goinfer/tokenizer"
)

type fakeEmbedAccel struct {
	name          string
	rows          []float32
	err           error
	calls         int
	gotIDs        []int
	gotKeepLayers bool
}

func (f *fakeEmbedAccel) Name() string { return f.name }

func (f *fakeEmbedAccel) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	f.calls++
	f.gotIDs = append([]int(nil), ids...)
	f.gotKeepLayers = keepLayers
	if f.err != nil {
		return nil, nil, f.err
	}
	return f.rows, nil, nil
}

func (f *fakeEmbedAccel) ForwardEmbeds([]float32, int, bool) ([]float32, [][]float32, error) {
	return nil, nil, errors.New("unexpected ForwardEmbeds call")
}

func (f *fakeEmbedAccel) Close() error { return nil }

func newTinyEncoderForTest(t *testing.T) *Encoder {
	t.Helper()
	m, _ := loadTiny(t)
	pack, err := tokenizer.BuildPack(byteTokenizer(t, "<pad>", "<eos>", "<bos>", "<unk>"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadPack(pack)
	if err != nil {
		t.Fatal(err)
	}
	e, err := NewEncoder(m, tok, map[string]string{"query": "q: ", "document": "d: "})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func registerTempAccelerator(t *testing.T, name string, accel Accelerator) {
	t.Helper()
	RegisterAccelerator(name, func(*Model) (Accelerator, error) { return accel, nil })
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, name)
		accelMu.Unlock()
	})
}

func expectedPoolNormalize(rows []float32, T, d int) []float32 {
	mean := make([]float64, d)
	for row := range T {
		for j := range d {
			mean[j] += float64(rows[row*d+j])
		}
	}
	var ss float64
	for j := range mean {
		mean[j] /= float64(T)
		ss += mean[j] * mean[j]
	}
	norm := math.Max(math.Sqrt(ss), 1e-12)
	out := make([]float32, d)
	for j := range mean {
		out[j] = float32(mean[j] / norm)
	}
	return out
}

func distinguishableRows(T, d int) []float32 {
	rows := make([]float32, T*d)
	for row := range T {
		for j := range d {
			rows[row*d+j] = float32((row+1)*(j+2)) * 0.25
			if (row+j)%3 == 0 {
				rows[row*d+j] = -rows[row*d+j]
			}
		}
	}
	return rows
}

// TestEncoder_EmbedIDs verifies the prepared-ID seam on CPU and through a selected accelerator: exact IDs forwarded
// without re-tokenization or mutation, mean pooling and L2 normalization applied to the accelerator's projected rows,
// EmbedText parity preserved, and accelerator errors surfaced without CPU fallback.
func TestEncoder_EmbedIDs(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()

	ids := []int{5, 17, 42}
	origIDs := append([]int(nil), ids...)

	// CPU path (no accelerator selected).
	cpuVec, err := e.EmbedIDs(ids)
	if err != nil {
		t.Fatalf("CPU EmbedIDs: %v", err)
	}
	if !slices.Equal(ids, origIDs) {
		t.Fatalf("CPU EmbedIDs mutated caller ids: got %v, want %v", ids, origIDs)
	}
	wantCPU, err := e.Model().Embed(origIDs)
	if err != nil {
		t.Fatalf("Model.Embed: %v", err)
	}
	if !sameBits(cpuVec, wantCPU) {
		t.Fatal("CPU EmbedIDs does not match Model.Embed")
	}

	tokIDs, err := e.Tokenize("hi", "query")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	textCPU, textN, err := e.EmbedText("hi", "query")
	if err != nil {
		t.Fatalf("CPU EmbedText: %v", err)
	}
	fromTokCPU, err := e.EmbedIDs(tokIDs)
	if err != nil {
		t.Fatalf("CPU EmbedIDs(Tokenize): %v", err)
	}
	if textN != len(tokIDs) || !sameBits(textCPU, fromTokCPU) {
		t.Fatal("CPU EmbedText disagrees with EmbedIDs(Tokenize)")
	}
	if _, err := e.EmbedIDs(nil); err == nil {
		t.Fatal("CPU EmbedIDs(nil) succeeded, want error")
	}

	// Accelerated path: distinguishable projected rows pooled and L2-normalized.
	rows := distinguishableRows(len(ids), dim)
	origRows := append([]float32(nil), rows...)
	fake := &fakeEmbedAccel{name: "fake-embed-ids", rows: rows}
	registerTempAccelerator(t, fake.name, fake)
	if _, err := e.UseAccelerator(fake.name); err != nil {
		t.Fatalf("UseAccelerator: %v", err)
	}

	gotAccel, err := e.EmbedIDs(ids)
	if err != nil {
		t.Fatalf("accelerated EmbedIDs: %v", err)
	}
	if fake.calls != 1 {
		t.Fatalf("accelerator calls = %d, want 1", fake.calls)
	}
	if !slices.Equal(fake.gotIDs, origIDs) {
		t.Fatalf("accelerator received ids %v, want %v", fake.gotIDs, origIDs)
	}
	if fake.gotKeepLayers {
		t.Fatal("accelerator received keepLayers=true, want false")
	}
	if !slices.Equal(ids, origIDs) {
		t.Fatalf("accelerated EmbedIDs mutated caller ids: got %v, want %v", ids, origIDs)
	}
	if !sameBits(rows, origRows) {
		t.Fatal("EmbedIDs mutated accelerator projected rows")
	}
	wantAccel := expectedPoolNormalize(origRows, len(origIDs), dim)
	if !sameBits(gotAccel, wantAccel) {
		t.Fatal("accelerated EmbedIDs did not apply expected mean pooling and L2 normalization")
	}
	var normSq float64
	for _, v := range gotAccel {
		normSq += float64(v) * float64(v)
	}
	if math.Abs(math.Sqrt(normSq)-1.0) > 1e-6 {
		t.Fatalf("accelerated EmbedIDs L2 norm = %.8f, want 1", math.Sqrt(normSq))
	}
	if sameBits(gotAccel, cpuVec) {
		t.Fatal("accelerated EmbedIDs unexpectedly matched CPU output")
	}

	// EmbedText uses the same accelerator dispatch with tokenized IDs.
	textRows := distinguishableRows(len(tokIDs), dim)
	fake.rows = textRows
	gotTextAccel, n, err := e.EmbedText("hi", "query")
	if err != nil {
		t.Fatalf("accelerated EmbedText: %v", err)
	}
	if n != len(tokIDs) {
		t.Fatalf("accelerated EmbedText token count = %d, want %d", n, len(tokIDs))
	}
	if !slices.Equal(fake.gotIDs, tokIDs) {
		t.Fatalf("accelerated EmbedText forwarded ids %v, want %v", fake.gotIDs, tokIDs)
	}
	if !sameBits(gotTextAccel, expectedPoolNormalize(textRows, len(tokIDs), dim)) {
		t.Fatal("accelerated EmbedText did not pool and normalize accelerator rows")
	}

	// Empty ids rejected before calling the accelerator.
	callsBeforeEmpty := fake.calls
	if _, err := e.EmbedIDs([]int{}); err == nil {
		t.Fatal("accelerated EmbedIDs(empty) succeeded, want error")
	}
	if fake.calls != callsBeforeEmpty {
		t.Fatalf("accelerated EmbedIDs(empty) invoked Forward (%d -> %d)", callsBeforeEmpty, fake.calls)
	}

	// Malformed hidden length from accelerator returns an error rather than panicking or falling back to CPU.
	fake.rows = rows[:len(rows)-1]
	if got, err := e.EmbedIDs(ids); err == nil || got != nil {
		t.Fatalf("short accelerator hidden: got (%v, %v), want (nil, error)", got, err)
	}

	// Accelerator error stays visible: no CPU fallback, no input or encoder mutation.
	wantErr := errors.New("fake accelerator failure")
	fake.rows = nil
	fake.err = wantErr
	callsBeforeErr := fake.calls
	gotErrVec, err := e.EmbedIDs(ids)
	if !errors.Is(err, wantErr) {
		t.Fatalf("accelerator error = %v, want %v", err, wantErr)
	}
	if gotErrVec != nil {
		t.Fatalf("accelerator error returned non-nil vector (len %d)", len(gotErrVec))
	}
	if fake.calls != callsBeforeErr+1 {
		t.Fatalf("accelerator error calls = %d, want %d", fake.calls, callsBeforeErr+1)
	}
	if !slices.Equal(fake.gotIDs, origIDs) {
		t.Fatalf("accelerator error received ids %v, want %v", fake.gotIDs, origIDs)
	}
	if !slices.Equal(ids, origIDs) {
		t.Fatalf("accelerator error mutated caller ids: got %v, want %v", ids, origIDs)
	}
	if e.Accelerator() != fake {
		t.Fatal("accelerator error cleared or changed encoder accelerator")
	}
	if _, _, err := e.EmbedText("hi", "query"); !errors.Is(err, wantErr) {
		t.Fatalf("EmbedText accelerator error = %v, want %v", err, wantErr)
	}
}
