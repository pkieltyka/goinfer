package vulkan

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func requiredDevice(t *testing.T) uint32 {
	t.Helper()
	id := os.Getenv("GOINFER_TEST_VULKAN_DEVICE")
	if id == "" {
		t.Skip("set GOINFER_TEST_VULKAN_DEVICE to the hexadecimal AMD PCI device ID")
	}
	n, err := strconv.ParseUint(id, 16, 32)
	if err != nil || n == 0 {
		t.Fatalf("invalid hexadecimal PCI device ID %q", id)
	}
	return uint32(n)
}
func checkDevice(t *testing.T, a *encoder, id uint32) {
	t.Helper()
	info := a.device.Info()
	bound, ok := AttachedDevice(a)
	if !ok || bound != (DeviceInfo{Name: info.Name, VendorID: info.VendorID, DeviceID: info.DeviceID,
		APIVersion: info.APIVersion, DriverVersion: info.DriverVersion, Type: "discrete-gpu"}) {
		t.Fatalf("attached identity %+v differs from physical device %+v", bound, info)
	}
	if info.VendorID != 0x1002 || info.DeviceID != id {
		t.Fatalf("unexpected device %+v", info)
	}
	t.Logf("Vulkan %s %04x:%04x API %#x driver %#x", info.Name, info.VendorID, info.DeviceID, info.APIVersion, info.DriverVersion)
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Error(err)
		}
	})
}

// Compare every component as well as each row's cosine. One wrong token
// must be visible even if hundreds of other rows match.
func checkRows(t *testing.T, label string, got, want []float32, width int) {
	t.Helper()
	if width <= 0 || len(want) == 0 || len(want)%width != 0 || len(got) != len(want) {
		t.Fatalf("%s: incompatible row width %d and lengths %d/%d", label, width, len(got), len(want))
	}
	minCos, maxAbs, maxFraction, maxRowRelative := 1.0, 0.0, 0.0, 0.0
	for row := 0; row < len(want)/width; row++ {
		var dot, aa, bb, rowMax, rowErr float64
		for j := range width {
			i := row*width + j
			x, y := float64(got[i]), float64(want[i])
			delta := math.Abs(x - y)
			limit := 1e-4 * max(1, math.Abs(y))
			if !(delta <= limit) {
				t.Fatalf("%s row %d col %d: got %g want %g absolute error %g exceeds %g", label, row, j, x, y, delta, limit)
			}
			maxFraction = max(maxFraction, delta/limit)
			rowMax = max(rowMax, math.Abs(y))
			rowErr = max(rowErr, delta)
			dot += x * y
			aa += x * x
			bb += y * y
		}
		if rowMax > 0 {
			maxRowRelative = max(maxRowRelative, rowErr/rowMax)
		}
		maxAbs = max(maxAbs, rowErr)
		if aa == 0 && bb == 0 {
			continue
		}
		cosine := dot / math.Sqrt(aa*bb)
		if !(cosine >= 0.9999) {
			t.Fatalf("%s row %d cosine %.12f", label, row, cosine)
		}
		minCos = min(minCos, cosine)
	}
	t.Logf("%s min row cosine %.12f max abs %.7g max row-relative %.7g max fraction of limit %.7g", label, minCos, maxAbs, maxRowRelative, maxFraction)
}
func checkedActivity(t *testing.T, a *encoder, prev vk.Activity, rows int, keep bool) {
	t.Helper()
	after := a.device.Activity()
	ds := after.Dispatches - prev.Dispatches
	submissions := after.Submissions - prev.Submissions
	waits := after.Waits - prev.Waits
	copies := after.Copies - prev.Copies
	// 24 row-wise dispatches per layer plus a score/softmax/value triplet for each block.
	want := uint64(5 + a.cfg.Layers*(24+3*ceilDiv(rows, 128)))
	wantCopies := uint64(2)
	if keep {
		wantCopies += uint64(a.cfg.Layers + 1)
	}
	if ds != want || submissions != 1 || waits != 1 || copies != wantCopies {
		t.Fatalf("%d rows: dispatch/submit/wait/copy=%d/%d/%d/%d, want %d/1/1/%d", rows, ds, submissions, waits, copies, want, wantCopies)
	}
	t.Logf("%d rows: %d dispatches, one submission and fence, %d copies", rows, ds, copies)
}
func TestVulkanEncoderTiny(t *testing.T) {
	id := requiredDevice(t)
	dir := "../../testdata/embeddinggemma2-tiny"
	m, err := embeddinggemma2.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	a, err := newEncoder(m, vk.Selector{DeviceID: id})
	if err != nil {
		t.Fatal(err)
	}
	checkDevice(t, a, id)
	data, err := os.ReadFile(filepath.Join(dir, "golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			IDs    []int
			Layers [][][]float32
			Last   [][]float32 `json:"last_hidden_state"`
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 4 {
		t.Fatalf("tiny fixture has %d cases", len(fixture.Cases))
	}
	for _, item := range fixture.Cases {
		prev := a.device.Activity()
		got, states, err := a.Forward(item.IDs, true)
		if err != nil {
			t.Fatal(err)
		}
		checkedActivity(t, a, prev, len(item.IDs), true)
		_, cpu, err := m.ForwardCPU(item.IDs, true)
		if err != nil {
			t.Fatal(err)
		}
		if len(states) != len(cpu) || len(states) != len(item.Layers) {
			t.Fatal("layer count mismatch")
		}
		if !slices.Equal(states[0], cpu[0]) {
			t.Fatal("initial GPU snapshot differs from the host input")
		}
		for i, s := range states {
			checkRows(t, fmt.Sprintf("layer %d CPU", i), s, cpu[i], a.cfg.Hidden)
			// Transformers records the projection in its final hidden-state slot.
			if i < len(states)-1 {
				checkRows(t, fmt.Sprintf("layer %d Transformers", i), s, slices.Concat(item.Layers[i]...), a.cfg.Hidden)
			}
		}
		checkRows(t, "projected rows Transformers", got, slices.Concat(item.Last...), m.Dim())
	}
	fixed := fixture.Cases[3].IDs
	full, fullStates, err := a.Forward(fixed, true)
	if err != nil {
		t.Fatal(err)
	}
	beforeSplit := a.device.Activity()
	budget := a.dispatchBudget
	a.dispatchBudget = 5 + 2*(24+3) // one score block, exactly two layers per submission
	split, splitStates, err := a.Forward(fixed, true)
	a.dispatchBudget = budget
	if err != nil {
		t.Fatal(err)
	}
	afterSplit := a.device.Activity()
	if afterSplit.Submissions-beforeSplit.Submissions != 3 || afterSplit.Waits-beforeSplit.Waits != 3 {
		t.Fatalf("split forward used %d submits and %d waits", afterSplit.Submissions-beforeSplit.Submissions, afterSplit.Waits-beforeSplit.Waits)
	}
	if !slices.Equal(full, split) || !slices.EqualFunc(fullStates, splitStates, slices.Equal[[]float32]) {
		t.Fatal("split changed output or a layer snapshot")
	}
	t.Log("forced three submissions: all layer snapshots and projected rows match one submission exactly")
	left, right := make([]int, 115), []int{2, 7, 1}
	for i := range left {
		left[i] = i % 213
	}
	batch := [][]int{left, fixed, right}
	prev := a.device.Activity()
	packed, err := a.ForwardBatch(batch)
	if err != nil {
		t.Fatal(err)
	}
	checkedActivity(t, a, prev, 135, false)
	for i, ids := range batch {
		single, _, err := a.Forward(ids, false)
		if err != nil {
			t.Fatal(err)
		}
		checkRows(t, fmt.Sprintf("packed member %d vs singleton", i), packed[i], single, m.Dim())
	}
	for i := range left {
		left[i] = (left[i] + 35) % 213
	}
	other, err := a.ForwardBatch(batch)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(packed[1], other[1]) || !slices.Equal(packed[2], other[2]) {
		t.Fatal("changing preceding input changed another sequence")
	}
	if slices.Equal(packed[0], other[0]) {
		t.Fatal("changing first input did not change its output")
	}
	x, err := m.EmbedTokens(fixed)
	if err != nil {
		t.Fatal(err)
	}
	x[0] += 1
	original := slices.Clone(x)
	got, _, err := a.ForwardEmbeds(x, len(fixed), false)
	if err != nil {
		t.Fatal(err)
	}
	want, _, err := m.ForwardEmbedsCPU(x, len(fixed), false)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(x, original) {
		t.Fatal("forward changed prepared embeddings")
	}
	checkRows(t, "prepared embeddings", got, want, m.Dim())
	prev = a.device.Activity()
	for _, bad := range [][][]int{nil, {{1}, {}}, {{1}, {m.Config().VocabSize}}, {make([]int, embeddinggemma2.MaxTokens), {1}}} {
		if output, err := a.ForwardBatch(bad); err == nil || output != nil {
			t.Fatal("accepted invalid batch or returned partial results")
		}
	}
	if a.device.Activity() != prev {
		t.Fatal("failed validation issued device work")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := a.Forward([]int{1}, false); err == nil {
		t.Fatal("forward after Close succeeded")
	}
}

// The hashes pin google/embeddinggemma-2 f9972ac22e696a105c482202646979e33e4b5a1e.
func TestVulkanEncoderRealPreparedIDs(t *testing.T) {
	id := requiredDevice(t)
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		t.Skip("set GOINFER_EG2_DIR to the pinned local checkpoint")
	}
	pins := map[string]string{
		"config.json":                       "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
		"tokenizer.json":                    "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
		"config_sentence_transformers.json": "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5",
		"model.safetensors":                 "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
	}
	for name, pin := range pins {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.New()
		_, err = io.Copy(hash, f)
		closeErr := f.Close()
		if err != nil || closeErr != nil || fmt.Sprintf("%x", hash.Sum(nil)) != pin {
			t.Fatalf("cannot verify %s: read=%v close=%v", name, err, closeErr)
		}
	}
	e, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	query := []int{2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 1217, 1440, 740, 2536, 577, 1680, 901, 974, 3463, 1135, 1}
	doc := []int{2, 3250, 236787, 7293, 1109, 1816, 236787, 87215, 1}
	for _, tc := range []struct {
		text, role string
		ids        []int
	}{
		{"how long can results be before they get cut off", "query", query},
		{"Redis", "document", doc},
	} {
		ids, err := e.Tokenize(tc.text, tc.role)
		if err != nil || !slices.Equal(tc.ids, ids) {
			t.Fatalf("%s prepared IDs: got %v, want %v (error %v)", tc.role, ids, tc.ids, err)
		}
	}
	edge := make([]int, e.Model().Config().SlidingWindow+2)
	for i := range edge {
		edge[i] = query[1+i%(len(query)-2)]
	}
	edge[0], edge[len(edge)-1] = 2, 1
	ids := [][]int{query, doc, edge}
	want, err := e.EmbedIDsBatch(ids)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := e.UseAccelerator("vulkan")
	if err != nil {
		t.Fatal(err)
	}
	m := e.Model()
	a := acc.(*encoder)
	checkDevice(t, a, id)
	prev := a.device.Activity()
	got, err := e.EmbedIDsBatch(ids)
	if err != nil {
		t.Fatal(err)
	}
	checkedActivity(t, a, prev, len(query)+len(doc)+len(edge), false)
	for i, v := range got {
		checkRows(t, fmt.Sprintf("pooled %d-token input", len(ids[i])), v, want[i], m.Dim())
		var norm float64
		for _, x := range v {
			norm += float64(x) * float64(x)
		}
		if !(math.Abs(norm-1) < 1e-6) {
			t.Fatalf("pooled squared norm %g", norm)
		}
	}
	ref, refLayers, err := m.ForwardCPU(edge, true)
	if err != nil {
		t.Fatal(err)
	}
	before := a.device.Activity()
	last, layers, err := a.Forward(edge, true)
	if err != nil {
		t.Fatal(err)
	}
	checkedActivity(t, a, before, len(edge), true)
	if len(layers) != len(refLayers) || !slices.Equal(layers[0], refLayers[0]) {
		t.Fatal("real layer count or initial snapshot differs")
	}
	for i, state := range layers {
		checkRows(t, fmt.Sprintf("real window edge layer %d", i), state, refLayers[i], m.Config().Hidden)
	}
	checkRows(t, "real window edge projected", last, ref, m.Dim())
}
func TestBufferLimits(t *testing.T) {
	for _, d := range [][]int{{0}, {-1, 1}, {129, 2}, {math.MaxInt, 2}} {
		if _, err := byteSize(1024, d...); err == nil {
			t.Fatalf("accepted %v", d)
		}
	}
	if n, err := byteSize(1024, 64, 4); err != nil || n != 1024 {
		t.Fatalf("exact limit: %d %v", n, err)
	}
}
