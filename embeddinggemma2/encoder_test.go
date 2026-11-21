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

type fakeBatchAccel struct {
	fakeEmbedAccel
	batchRows  [][]float32
	batchErr   error
	batchCalls int
	gotBatch   [][]int
}

func (f *fakeBatchAccel) ForwardBatch(ids [][]int) ([][]float32, error) {
	f.batchCalls++
	f.gotBatch = nil
	for _, in := range ids {
		f.gotBatch = append(f.gotBatch, slices.Clone(in))
	}
	if f.batchErr != nil {
		return nil, f.batchErr
	}
	return f.batchRows, nil
}

// TestEncoder_EmbedIDsBatch: on the CPU, EmbedIDsBatch gives each input what EmbedIDs gives it; through an
// accelerator that batches, it makes one ForwardBatch call with the inputs as given (no Forward) and pools each
// input's rows as EmbedIDs does, into vectors the caller owns; an accelerator that cannot batch is an error with no
// forward at all; and an empty batch or input, an accelerator error, a wrong output count or width, or a bad id on the
// CPU returns nil and an error.
func TestEncoder_EmbedIDsBatch(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	in := [][]int{{5, 17, 42}, {3}, {9, 9, 9, 9}}
	keep := [][]int{slices.Clone(in[0]), slices.Clone(in[1]), slices.Clone(in[2])}
	same := func(a, b [][]int) bool { return slices.EqualFunc(a, b, slices.Equal[[]int]) }

	cpu, err := e.EmbedIDsBatch(in)
	if err != nil || len(cpu) != len(in) {
		t.Fatalf("CPU EmbedIDsBatch: %d vectors, %v", len(cpu), err)
	}
	for i, ids := range in {
		want, err := e.EmbedIDs(ids)
		if err != nil || !sameBits(cpu[i], want) {
			t.Fatalf("CPU input %d: differs from EmbedIDs (error %v)", i, err)
		}
	}
	if !same(in, keep) {
		t.Fatal("CPU EmbedIDsBatch modified its ids")
	}
	for _, bad := range []struct {
		name string
		ids  [][]int
	}{{"empty batch", nil}, {"empty last input", [][]int{in[0], {}}}, {"out-of-vocabulary last input", [][]int{in[0], {1 << 30}}}} {
		if got, err := e.EmbedIDsBatch(bad.ids); err == nil || got != nil {
			t.Errorf("CPU %s: (%v, %v), want (nil, error)", bad.name, got, err)
		}
	}

	rows := [][]float32{distinguishableRows(3, dim), distinguishableRows(1, dim), distinguishableRows(4, dim)}
	rows[1][0], rows[2][dim] = 7, -3 // so no two inputs pool alike
	keepRows := [][]float32{slices.Clone(rows[0]), slices.Clone(rows[1]), slices.Clone(rows[2])}
	fake := &fakeBatchAccel{fakeEmbedAccel: fakeEmbedAccel{name: "fake-embed-batch"}, batchRows: rows}
	registerTempAccelerator(t, fake.name, fake)
	if _, err := e.UseAccelerator(fake.name); err != nil {
		t.Fatal(err)
	}
	got, err := e.EmbedIDsBatch(in)
	if err != nil {
		t.Fatal(err)
	}
	if fake.batchCalls != 1 || fake.calls != 0 || !same(fake.gotBatch, keep) || !same(in, keep) {
		t.Fatalf("ForwardBatch calls %d, Forward calls %d, ids received %v (sent %v)", fake.batchCalls, fake.calls, fake.gotBatch, keep)
	}
	for i := range in {
		if !sameBits(got[i], expectedPoolNormalize(keepRows[i], len(keep[i]), dim)) {
			t.Fatalf("input %d: not the mean pool and L2 normalisation of its own rows", i)
		}
	}
	clear(got[0])
	again, err := e.EmbedIDsBatch(in)
	if err != nil || !sameBits(again[0], expectedPoolNormalize(keepRows[0], len(keep[0]), dim)) {
		t.Fatalf("after the caller cleared a returned vector, the next batch differs (error %v)", err)
	}
	for i := range rows {
		if !sameBits(rows[i], keepRows[i]) {
			t.Fatalf("EmbedIDsBatch wrote the accelerator's rows for input %d", i)
		}
	}

	calls := fake.batchCalls
	wantErr := errors.New("fake batch failure")
	for _, c := range []struct {
		name string
		ids  [][]int
		rows [][]float32
		err  error
		call bool
	}{
		{"empty batch", nil, rows, nil, false},
		{"empty last input", [][]int{in[0], {}}, rows, nil, false},
		{"accelerator error", in, nil, wantErr, true},
		{"too few outputs", in, rows[:2], nil, true},
		{"short last output", in, [][]float32{rows[0], rows[1], rows[2][:len(rows[2])-1]}, nil, true},
	} {
		fake.batchRows, fake.batchErr = c.rows, c.err
		got, err := e.EmbedIDsBatch(c.ids)
		if err == nil || got != nil || (c.err != nil && !errors.Is(err, c.err)) {
			t.Errorf("%s: (%v, %v), want (nil, error)", c.name, got, err)
		}
		if c.call {
			calls++
		}
		if fake.batchCalls != calls || fake.calls != 0 {
			t.Errorf("%s: %d ForwardBatch and %d Forward calls, want %d and 0", c.name, fake.batchCalls, fake.calls, calls)
		}
	}

	single := &fakeEmbedAccel{name: "fake-embed-nobatch", rows: distinguishableRows(3, dim)}
	registerTempAccelerator(t, single.name, single)
	if _, err := e.UseAccelerator(single.name); err != nil {
		t.Fatal(err)
	}
	if got, err := e.EmbedIDsBatch(in); err == nil || got != nil || single.calls != 0 {
		t.Fatalf("an accelerator that cannot batch: (%v, %v) after %d Forward calls; want an error and none", got, err, single.calls)
	}
}
