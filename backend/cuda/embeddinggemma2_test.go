//go:build cuda

package cuda

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/tokenizer"
)

func TestResolveCUDADevice(t *testing.T) {
	const first = "GPU-abcd1234-1111-2222-3333-444444444444"
	const second = "MIG-abcd1234-1111-2222-3333-555555555555"
	const old = "GPU-abcd1234-1111-2222-3333-666666666666"
	adapters := []gpu.Adapter{
		{Ordinal: 0, UUID: first, Major: 7, Minor: 5},
		{Ordinal: 1, UUID: old, Major: 7},
		{Ordinal: 2, UUID: second, Major: 12},
		{Ordinal: 3, Major: 8},
	}
	for _, tt := range []struct {
		token   string
		ordinal int
		want    error
	}{
		{first, 0, nil}, {strings.ToUpper(second), 2, nil}, {"2", 2, nil},
		{old, 0, embeddinggemma2.ErrNoDevice}, {"1", 0, embeddinggemma2.ErrNoDevice},
		{"02", 0, embeddinggemma2.ErrNoDevice}, {"3", 0, embeddinggemma2.ErrNoDevice},
		{"", 0, embeddinggemma2.ErrDeviceUnavailable}, {"zz", 0, embeddinggemma2.ErrDeviceUnavailable},
		{"GPU-xyz", 0, embeddinggemma2.ErrDeviceUnavailable},
		{"foo1", 0, embeddinggemma2.ErrDeviceUnavailable},
		{"GPU-abcd1234-1111-2222-3333-55555555555z", 0, embeddinggemma2.ErrDeviceUnavailable},
		{"MIG-abcd1234-11112222-3333-555555555555", 0, embeddinggemma2.ErrDeviceUnavailable},
	} {
		got, err := resolveCUDADevice(tt.token, adapters)
		if !errors.Is(err, tt.want) || err == nil && got.Ordinal != tt.ordinal ||
			tt.want == embeddinggemma2.ErrDeviceUnavailable && errors.Is(err, embeddinggemma2.ErrNoDevice) {
			t.Errorf("resolve %q: got %+v, %v; want ordinal %d, %v", tt.token, got, err, tt.ordinal, tt.want)
		}
	}
	adapters[3].UUID = second
	for _, token := range []string{second, "2", "3"} {
		if _, err := resolveCUDADevice(token, adapters); !errors.Is(err, embeddinggemma2.ErrAmbiguousDevice) {
			t.Errorf("duplicate UUID, select %q: %v", token, err)
		}
	}
}

func TestCUDAAttachErrorCategories(t *testing.T) {
	invalidAlloc := gpu.AllocFailedPanic("weight", 4096, gc.ErrInvalidValue)
	textOOM := errors.New("OUT_OF_MEMORY without a native result")
	for _, tt := range []struct {
		native error
		want   error
	}{
		{fmt.Errorf("%w: %w", gpu.ErrDriverInit, gc.ErrOutOfMemory), embeddinggemma2.ErrDeviceOutOfMemory},
		{fmt.Errorf("%w: %w", gpu.ErrDriverInit, gc.ErrNoDevice), embeddinggemma2.ErrNoDevice},
		{fmt.Errorf("%w: %w", gpu.ErrDriverInit, gc.ErrStubLibrary), embeddinggemma2.ErrBackendUnavailable},
		{fmt.Errorf("%w: %w", gpu.ErrAdapterChanged, gc.ErrInvalidOrdinal), embeddinggemma2.ErrNoDevice},
		{fmt.Errorf("%w: %w", gpu.ErrAdapterChanged, gc.ErrInvalidDevice), embeddinggemma2.ErrNoDevice},
		{fmt.Errorf("%w: %w", gpu.ErrAdapterChanged, gc.ErrNoDevice), embeddinggemma2.ErrNoDevice},
		{gpu.AllocFailedPanic("weight", 4096, gc.ErrOutOfMemory), embeddinggemma2.ErrDeviceOutOfMemory},
		{invalidAlloc, invalidAlloc},
		{textOOM, textOOM},
		{gc.ErrInvalidDevice, gc.ErrInvalidDevice},
		{gc.ErrInvalidPTX, gc.ErrInvalidPTX},
	} {
		got := classifyCUDAAttach(tt.native)
		if !errors.Is(got, tt.want) || !errors.Is(got, tt.native) ||
			tt.want == embeddinggemma2.ErrDeviceOutOfMemory && errors.Is(got, embeddinggemma2.ErrDeviceUnavailable) {
			t.Errorf("classify %v: got %v, want %v preserving native cause", tt.native, got, tt.want)
		}
		if tt.native == tt.want && got != tt.native {
			t.Errorf("unclassified %v unexpectedly wrapped: %v", tt.native, got)
		}
	}
}

func TestCUDAEG2OpenerFailure(t *testing.T) {
	m := eg2TinyModel(t)
	for _, combined := range []bool{false, true} {
		calls := 0
		got, err := newEG2AccelOn(m, combined, func() (*Device, error) {
			calls++
			return nil, gc.ErrOutOfMemory
		})
		if got != nil || calls != 1 || !errors.Is(err, gc.ErrOutOfMemory) ||
			!errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) {
			t.Fatalf("combined %t: got %v, %v, %d open calls", combined, got, err, calls)
		}
	}
}

func TestCUDAEG2SelectedDevice(t *testing.T) {
	uuid := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if uuid == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID to the target CUDA device's GPU UUID")
	}
	adapters, err := gpu.Adapters()
	if err != nil {
		t.Fatal(err)
	}
	target, err := resolveCUDADevice(uuid, adapters)
	if err != nil {
		t.Fatal(err)
	}
	m := eg2TinyModel(t)
	ids := eg2TinyInputs(t, m.Config().VocabSize)[2]
	want, _, err := m.ForwardCPU(ids, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, selector := range embeddinggemma2.Accelerators() {
		if selector != "cuda" && selector != "cuda-eg2-combined" {
			continue
		}
		for _, token := range []string{uuid, strconv.Itoa(target.Ordinal)} {
			if !t.Run(selector+"/"+token, func(t *testing.T) {
				acc, err := m.NewAccelerator(selector + ":" + token)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := acc.Close(); err != nil {
						t.Error(err)
					}
				})
				a := acc.(*eg2Accel)
				native := a.ops.dev.Context().Device()
				gotUUID, err := native.UUID()
				if err != nil || !strings.EqualFold(gotUUID, uuid) || native.Ordinal() != target.Ordinal {
					t.Fatalf("attached UUID %q ordinal %d, error %v; want %q ordinal %d", gotUUID, native.Ordinal(), err, uuid, target.Ordinal)
				}
				result := make(chan error, 1)
				go func() {
					got, _, err := acc.Forward(ids, false)
					if err == nil && (len(got) != len(want) || !eg2Finite(got) || eg2Cos(got, want) < 0.9999) {
						err = fmt.Errorf("CUDA/CPU projected rows diverged")
					}
					result <- err
				}()
				reading, readErr := a.DeviceMemory()
				if err := <-result; err != nil {
					t.Fatal(err)
				}
				if readErr != nil || reading.Basis != embeddinggemma2.MemoryLive || reading.AvailableBytes < 0 ||
					reading.AvailableBytes > reading.TotalBytes || reading.TotalBytes <= 0 {
					t.Fatalf("memory %+v: %v", reading, readErr)
				}
				launches, _ := a.ops.dev.Activity()
				if launches == 0 {
					t.Fatal("no native CUDA launches")
				}
				t.Logf("%s PCI %s ordinal %d UUID %s: %d launches, memory %+v", target.Name, target.PCIBusID, target.Ordinal, gotUUID, launches, reading)
				if err := acc.Close(); err != nil {
					t.Fatal(err)
				}
				if _, err := a.DeviceMemory(); err == nil {
					t.Fatal("memory read succeeded after close")
				}
			}) {
				return
			}
		}
	}
	if _, err := m.NewAccelerator("cuda:GPU-00000000-0000-0000-0000-000000000000"); !errors.Is(err, embeddinggemma2.ErrNoDevice) {
		t.Fatalf("absent UUID: %v", err)
	}
}

func TestCUDAEG2InventoryAttach(t *testing.T) {
	uuid := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if uuid == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID to the target CUDA device's GPU UUID")
	}
	linked := linkedEG2CUDASelectors(t)
	adapters, err := gpu.Adapters()
	if err != nil {
		t.Fatal(err)
	}
	target, err := resolveCUDADevice(uuid, adapters)
	if err != nil {
		t.Fatal(err)
	}
	rows, failures := embeddinggemma2.Devices(embeddinggemma2.ProbeOptions{})
	if len(failures) != 0 {
		t.Fatalf("device inventory failed: %v", failures)
	}
	var row embeddinggemma2.DeviceInfo
	for _, device := range rows {
		if strings.EqualFold(device.ID, "cuda:"+uuid) {
			row = device
			break
		}
	}
	wantCC := embeddinggemma2.CUDAComputeCapability{Major: target.Major, Minor: target.Minor}
	if row.ID == "" || row.ComputeCapability != wantCC || row.Alias != "cuda:"+strconv.Itoa(target.Ordinal) ||
		row.Memory.Basis != embeddinggemma2.MemoryUnknown {
		t.Fatalf("cheap inventory row %+v disagrees with pinned adapter %+v", row, target)
	}
	want := linked
	rejectFP16 := target.Major < 8 && slices.Contains(linked, eg2FP16TestSelector)
	if rejectFP16 {
		want = want[:len(want)-1]
	}
	selectors := row.RunnableSelectors()
	if !slices.Equal(selectors, want) {
		t.Fatalf("pinned device runnable selectors %v, want %v", selectors, want)
	}

	tok, err := tokenizer.LoadJSONBytes([]byte(`{
		"added_tokens": [{"id":0,"content":"<pad>"},{"id":1,"content":"<eos>"},{"id":2,"content":"<bos>"},{"id":3,"content":"<unk>"}],
		"model": {"type":"BPE","unk_token":"<unk>","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3},"merges":[]}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	m := eg2TinyModel(t)
	for _, selector := range selectors {
		t.Run(selector, func(t *testing.T) {
			enc, err := embeddinggemma2.NewEncoder(m, tok, nil)
			if err != nil {
				t.Fatal(err)
			}
			acc, err := enc.UseDevice(row.ID, selector)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := acc.Close(); err != nil {
					t.Error(err)
				}
			})
			a := acc.(*eg2Accel)
			native := a.ops.dev.Context().Device()
			gotUUID, uuidErr := native.UUID()
			major, minor, ccErr := native.ComputeCapability()
			if uuidErr != nil || ccErr != nil || !strings.EqualFold(gotUUID, uuid) || native.Ordinal() != target.Ordinal ||
				major != wantCC.Major || minor != wantCC.Minor {
				t.Fatalf("retained native UUID %q ordinal %d CC %d.%d, errors %v, %v", gotUUID, native.Ordinal(), major, minor, uuidErr, ccErr)
			}
			reported := acc.(embeddinggemma2.AttachedDeviceReporter).AttachedDevice()
			if reported.ID != row.ID || reported.Alias != row.Alias || reported.ComputeCapability != wantCC ||
				!slices.Equal(reported.RunnableSelectors(), selectors) {
				t.Fatalf("retained row %+v disagrees with inventory %+v", reported, row)
			}
			if wantFP16 := selector == eg2FP16TestSelector; (a.ffnFP16 != (Pipeline{})) != wantFP16 || a.packedAttnMode != wantFP16 {
				t.Fatalf("%s attached the wrong precision path", selector)
			}
			if rejectFP16 && selector == "cuda" {
				failed, err := enc.UseDevice(row.ID, eg2FP16TestSelector)
				if failed != nil || !errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) ||
					errors.Is(err, embeddinggemma2.ErrNoDevice) || enc.Accelerator() != acc {
					t.Fatalf("FP16 rejection returned %v, %v; accelerator %v", failed, err, enc.Accelerator())
				}
			}
			t.Logf("%s, CUDA %d.%d, ordinal %d, UUID %s, selectors %v", target.Name, major, minor, target.Ordinal, gotUUID, selectors)
		})
	}
}

const eg2TinyDir = "../../testdata/embeddinggemma2-tiny"

// eg2TinyModel is the tiny fixture (sliding radius 3; head dims 16 with 2 KV heads and 32 with 1) with every norm weight
// redrawn in [0.5, 1.5) through Weights' aliasing slices, so a norm given the wrong weight cannot hide behind the
// fixture's own.
func eg2TinyModel(t *testing.T) *embeddinggemma2.Model {
	t.Helper()
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatalf("load the tiny fixture: %v", err)
	}
	w := m.Weights()
	r := rand.New(rand.NewPCG(11, 12))
	redraw := func(vs ...[]float32) {
		for _, v := range vs {
			for i := range v {
				v[i] = 0.5 + r.Float32()
			}
		}
	}
	redraw(w.PLENorm, w.FinalNorm)
	for _, l := range w.Layers {
		redraw(l.InNorm, l.QNorm, l.KNorm, l.PostAttnNorm, l.PreFFNorm, l.PostFFNorm, l.PLEPostNorm)
	}
	return m
}

func newEG2ForTest(t *testing.T, m *embeddinggemma2.Model) *eg2Accel {
	t.Helper()
	acc, err := m.NewAccelerator("cuda")
	if err != nil {
		t.Skipf("no CUDA device for EmbeddingGemma 2: %v", err)
	}
	t.Cleanup(func() { _ = acc.Close() })
	return acc.(*eg2Accel)
}

// eg2TinyInputs are the golden's four cases (1, 2, 9 and 17 tokens) and random inputs whose lengths put rows at both
// window edges and in the middle: 7 = 2W+1 (one row reaches both edges), 8 and 64, and 300.
func eg2TinyInputs(t *testing.T, vocab int) [][]int {
	t.Helper()
	raw, err := os.ReadFile(eg2TinyDir + "/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Cases []struct {
			IDs []int `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	var in [][]int
	for _, c := range g.Cases {
		in = append(in, c.IDs)
	}
	for _, n := range []int{7, 8, 64, 300} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7919 + 13 + n) % vocab
		}
		in = append(in, ids)
	}
	return in
}

func eg2Cos(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func eg2Finite(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

// eg2PoolNorm is sentence-transformers' mean pool then L2 normalisation over the first d of each row's E values.
func eg2PoolNorm(h []float32, T, E, d int) []float32 {
	mean := make([]float64, d)
	for t := range T {
		for j := range d {
			mean[j] += float64(h[t*E+j])
		}
	}
	var ss float64
	for j := range mean {
		mean[j] /= float64(T)
		ss += mean[j] * mean[j]
	}
	out := make([]float32, d)
	for j := range mean {
		out[j] = float32(mean[j] / math.Max(math.Sqrt(ss), 1e-12))
	}
	return out
}

// eg2Worst is the lowest cosine between the accelerator's and the CPU's states over every input: each layer's input and
// the last layer's output, the projected last hidden state and the pooled embedding. It fails the test on a non-finite
// value or a shape that differs.
func eg2Worst(t *testing.T, m *embeddinggemma2.Model, a *eg2Accel, inputs [][]int) float64 {
	t.Helper()
	worst := 1.0
	E := m.Dim()
	for ci, ids := range inputs {
		cpuLast, cpuLayers, err := m.ForwardCPU(ids, true)
		if err != nil {
			t.Fatal(err)
		}
		last, layers, err := a.Forward(ids, true)
		if err != nil {
			t.Fatalf("case %d (%d tokens): cuda forward: %v", ci, len(ids), err)
		}
		if len(layers) != len(cpuLayers) || len(last) != len(cpuLast) {
			t.Fatalf("case %d: %d layer states and %d values, CPU %d and %d", ci, len(layers), len(last), len(cpuLayers), len(cpuLast))
		}
		lw, lwAt := 1.0, -1
		for li := range cpuLayers {
			if !eg2Finite(layers[li]) {
				t.Fatalf("case %d: layer state %d is not finite", ci, li)
			}
			if c := eg2Cos(layers[li], cpuLayers[li]); c < lw {
				lw, lwAt = c, li
			}
		}
		if !eg2Finite(last) {
			t.Fatalf("case %d: the last hidden state is not finite", ci)
		}
		lc := eg2Cos(last, cpuLast)
		ec := eg2Cos(eg2PoolNorm(last, len(ids), E, E), eg2PoolNorm(cpuLast, len(ids), E, E))
		t.Logf("case %d (%d tokens): worst layer state %.9f (index %d), last hidden %.9f, embedding %.9f", ci, len(ids), lw, lwAt, lc, ec)
		worst = min(worst, lw, lc, ec)
	}
	return worst
}

// TestEG2CUDA_matchesCPU: on the tiny fixture, the CUDA accelerator's forward matches the CPU forward to cosine 0.9999
// on every layer's input, the last layer's output, the projected last hidden state and the pooled embedding, at both
// head geometries and both attention kinds.
func TestEG2CUDA_matchesCPU(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	if w := eg2Worst(t, m, a, eg2TinyInputs(t, m.Config().VocabSize)); w < 0.9999 {
		t.Fatalf("worst cosine %.9f, under 0.9999", w)
	}
}

// TestEG2CUDA_defectsNoticed: each planted defect alone (a sliding window one wider or one narrower than the radius,
// v normed with a weight) takes some state of some tiny input under the bar TestEG2CUDA_matchesCPU holds.
func TestEG2CUDA_defectsNoticed(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	inputs := eg2TinyInputs(t, m.Config().VocabSize)
	for _, d := range []struct {
		name string
		d    eg2Defect
	}{{"window W+1", eg2Defect{windowWide: true}}, {"window W-1", eg2Defect{windowNarrow: true}}, {"weighted v norm", eg2Defect{vNormWeight: true}}} {
		a.defect = d.d
		w := eg2Worst(t, m, a, inputs)
		t.Logf("%s: worst cosine %.9f", d.name, w)
		if w >= 0.9999 {
			t.Errorf("%s: worst cosine %.9f; the defect went unnoticed", d.name, w)
		}
	}
	a.defect = eg2Defect{}
}

// TestEG2CUDA_runsOnDevice: a forward is device work. Each layer launches attention_seg once on the device the
// accelerator opened, at its own key ranges, and the strided copies gather q, k and v; the test logs the device.
func TestEG2CUDA_runsOnDevice(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	c := m.Config()
	var mu sync.Mutex
	var launches, attn, copies int
	attnPtr := a.ops.vit.AttentionSeg.Ptr()
	gpu.TraceForTest = func(e gpu.TraceEvent) {
		mu.Lock()
		defer mu.Unlock()
		switch e.Op {
		case gpu.TraceLaunch:
			launches++
			if e.Pipe == attnPtr {
				attn++
			}
		case gpu.TraceCopy2D:
			copies++
		}
	}
	defer func() { gpu.TraceForTest = nil }()
	if _, _, err := a.Forward([]int{2, 17, 33, 41, 5, 99, 120, 7, 1}, false); err != nil {
		t.Fatal(err)
	}
	gpu.TraceForTest = nil
	var name string
	if err := a.ops.do(func() error { name = a.ops.dev.Name(); return nil }); err != nil {
		t.Fatal(err)
	}
	t.Logf("device %q: %d launches, %d attention_seg, %d strided copies over %d layers", name, launches, attn, copies, c.Layers)
	if attn != c.Layers {
		t.Errorf("%d attention_seg launches, want one per layer (%d)", attn, c.Layers)
	}
	if want := c.Layers * (2 + 2*c.Heads); copies != want {
		t.Errorf("%d strided copies, want %d (q, each head's k and v, and the layer's per-layer input)", copies, want)
	}
	if launches < 20*c.Layers {
		t.Errorf("%d launches for %d layers", launches, c.Layers)
	}
}

// eg2Counts is what the device saw while f ran: launches (attention_seg among them), strided copies and allocations.
type eg2Counts struct{ launches, attn, copies, allocs int }

func eg2Count(t *testing.T, a *eg2Accel, f func()) eg2Counts {
	t.Helper()
	var mu sync.Mutex
	var n eg2Counts
	attnPtr := a.ops.vit.AttentionSeg.Ptr()
	gpu.TraceForTest = func(e gpu.TraceEvent) {
		mu.Lock()
		defer mu.Unlock()
		switch e.Op {
		case gpu.TraceLaunch:
			n.launches++
			if e.Pipe == attnPtr {
				n.attn++
			}
		case gpu.TraceCopy2D:
			n.copies++
		case gpu.TraceAlloc:
			n.allocs++
		}
	}
	defer func() { gpu.TraceForTest = nil }()
	f()
	return n
}

// eg2FirstDiff is the first value at which got and want differ bit for bit, as row and column of width w, or "".
func eg2FirstDiff(got, want []float32, w int) string {
	if len(got) != len(want) {
		return fmt.Sprintf("%d values, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			return fmt.Sprintf("row %d column %d: %#08x (%v), want %#08x (%v)", i/w, i%w, math.Float32bits(got[i]), got[i],
				math.Float32bits(want[i]), want[i])
		}
	}
	return ""
}

func eg2SeqIDs(n, seed, vocab int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*2654435761 + seed*40503 + 17) % vocab
		if ids[i] < 0 {
			ids[i] = -ids[i]
		}
	}
	return ids
}

// TestEG2CUDA_batchMatchesSingles: a batch of inputs of mixed lengths (1 to 300 tokens; both sides of the radius-3
// window, both head geometries) gives each input's projected rows bit for bit what Forward gives that input alone, in
// either order, and each is within the CPU bar.
func TestEG2CUDA_batchMatchesSingles(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	V, E := m.Config().VocabSize, m.Dim()
	var inputs [][]int
	for i, n := range []int{1, 2, 7, 9, 64, 300} {
		inputs = append(inputs, eg2SeqIDs(n, i+1, V))
	}
	singles := make([][]float32, len(inputs))
	for i, ids := range inputs {
		h, _, err := a.Forward(ids, false)
		if err != nil {
			t.Fatal(err)
		}
		singles[i] = h
		cpu, _, err := m.ForwardCPU(ids, false)
		if err != nil {
			t.Fatal(err)
		}
		if c := eg2Cos(h, cpu); !(c >= 0.9999) {
			t.Fatalf("input %d (%d tokens) alone: cosine %.9f with the CPU", i, len(ids), c)
		}
	}
	for _, order := range [][]int{{0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}, {3, 0, 5, 1, 4, 2}} {
		batch := make([][]int, len(order))
		for j, i := range order {
			batch[j] = inputs[i]
		}
		got, err := a.ForwardBatch(batch)
		if err != nil {
			t.Fatal(err)
		}
		for j, i := range order {
			if d := eg2FirstDiff(got[j], singles[i], E); d != "" {
				t.Fatalf("order %v, input %d (%d tokens): batch differs from the input alone at %s", order, i, len(inputs[i]), d)
			}
		}
	}
}

// TestEG2CUDA_batchNoLeakage: what sits beside an input in a batch — other tokens, another length, before or after it,
// one or two neighbours — never changes that input's rows by a bit.
func TestEG2CUDA_batchNoLeakage(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	V, E := m.Config().VocabSize, m.Dim()
	b := eg2SeqIDs(9, 99, V)
	want, _, err := a.Forward(b, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{1, 3, 4, 8, 64} {
		for seed := range 2 {
			other := eg2SeqIDs(n, 1000+seed, V)
			for _, c := range []struct {
				name  string
				batch [][]int
				at    int
			}{{"after", [][]int{other, b}, 1}, {"before", [][]int{b, other}, 0}, {"between", [][]int{other, b, other}, 1}} {
				got, err := a.ForwardBatch(c.batch)
				if err != nil {
					t.Fatal(err)
				}
				if d := eg2FirstDiff(got[c.at], want, E); d != "" {
					t.Fatalf("neighbour of %d tokens (seed %d) %s: the 9-token input differs at %s", n, seed, c.name, d)
				}
			}
		}
	}
}

// TestEG2CUDA_batchRopeSegments: the batch's RoPE tables hold, for every input, the cached table's leading rows bit for
// bit: each input starts again at position 0, also after a nonempty one, for both (head dim, theta) pairs.
func TestEG2CUDA_batchRopeSegments(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	lens := []int{5, 9, 3}
	if err := a.ops.do(func() error {
		defer a.ops.releaseScratch()
		var pending []Buffer
		var relErr error
		tabs, err := a.batchRope(lens, &pending, &relErr)
		if err != nil {
			return err
		}
		if err := a.ops.finish(); err != nil {
			return err
		}
		if len(tabs) != 2 {
			return fmt.Errorf("%d (head dim, theta) pairs, want the fixture's 2", len(tabs))
		}
		for k, tb := range tabs {
			cs, sn, err := a.rope(9, k.hd, k.theta, &pending, &relErr)
			if err != nil {
				return err
			}
			for j, src := range []Buffer{cs, sn} {
				ref := make([]float32, 9*k.hd)
				got := make([]float32, 17*k.hd)
				if err := errors.Join(gpu.Download(src, ref), gpu.Download(tb[j], got)); err != nil {
					return err
				}
				off := 0
				for b, n := range lens {
					if d := eg2FirstDiff(got[off*k.hd:(off+n)*k.hd], ref[:n*k.hd], k.hd); d != "" {
						return fmt.Errorf("hd %d theta %g table %d, input %d: %s", k.hd, k.theta, j, b, d)
					}
					off += n
				}
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// TestEG2CUDA_batchContract: a batch's outputs are the caller's (appending to one leaves the next alone; a later
// forward changes neither); a batch totaling the single-input cap runs; an empty batch, an empty, out-of-vocabulary
// or oversized member (in the last slot) are refused before any device allocation, copy or launch; a
// batch of one issues what Forward issues; a batch of N issues one forward's launches (one attention_seg per layer)
// and Forward's copies plus two per input for each RoPE pair; and after Close a batch is an error.
func TestEG2CUDA_batchContract(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	c := m.Config()
	V, E := c.VocabSize, m.Dim()
	in := [][]int{eg2SeqIDs(7, 1, V), eg2SeqIDs(5, 2, V), eg2SeqIDs(12, 3, V), eg2SeqIDs(2, 4, V)}
	got, err := a.ForwardBatch(in)
	if err != nil {
		t.Fatal(err)
	}
	keep := slices.Clone(got[1])
	_ = append(got[0], 1, 2, 3)
	if _, err := a.ForwardBatch(in[:2]); err != nil {
		t.Fatal(err)
	}
	if d := eg2FirstDiff(got[1], keep, E); d != "" {
		t.Fatalf("an append to one output or a later forward changed the next output at %s", d)
	}
	if _, err := a.ForwardBatch([][]int{eg2SeqIDs(eg2MaxTokens-5, 5, V), eg2SeqIDs(5, 6, V)}); err != nil {
		t.Fatalf("a batch of exactly %d positions: %v", eg2MaxTokens, err)
	}
	for _, bad := range []struct {
		name  string
		batch [][]int
	}{
		{"empty batch", nil},
		{"empty last member", [][]int{in[0], {}}},
		{"out-of-vocabulary last member", [][]int{in[0], {1, V}}},
		{"oversized member", [][]int{in[0], eg2SeqIDs(eg2MaxTokens+1, 6, V)}},
	} {
		var err error
		n := eg2Count(t, a, func() { _, err = a.ForwardBatch(bad.batch) })
		if err == nil || n != (eg2Counts{}) {
			t.Errorf("%s: error %v, device work %+v; want a refusal and none", bad.name, err, n)
		}
	}
	// Refused before the gather too: an out-of-vocabulary id at the end of a batch at the cap allocates far less host
	// memory than the gather buffer would take.
	capBatch := [][]int{eg2SeqIDs(eg2MaxTokens-5, 5, V), append(eg2SeqIDs(4, 6, V), V)}
	var ms0, ms1 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	_, cerr := a.ForwardBatch(capBatch)
	runtime.ReadMemStats(&ms1)
	if gather := uint64(eg2MaxTokens * c.Hidden * 4); cerr == nil || ms1.TotalAlloc-ms0.TotalAlloc >= gather/2 {
		t.Errorf("an out-of-vocabulary id at the cap: error %v, %d host bytes allocated (the gather buffer is %d)", cerr,
			ms1.TotalAlloc-ms0.TotalAlloc, gather)
	}
	single := eg2Count(t, a, func() {
		if _, _, err := a.Forward(in[2], false); err != nil {
			t.Fatal(err)
		}
	})
	one := eg2Count(t, a, func() {
		if _, err := a.ForwardBatch(in[2:3]); err != nil {
			t.Fatal(err)
		}
	})
	if one != single {
		t.Errorf("a batch of one: %+v, Forward: %+v", one, single)
	}
	var cat []int
	for _, ids := range in {
		cat = append(cat, ids...)
	}
	flat := eg2Count(t, a, func() {
		if _, _, err := a.Forward(cat, false); err != nil {
			t.Fatal(err)
		}
	})
	batch := eg2Count(t, a, func() {
		if _, err := a.ForwardBatch(in); err != nil {
			t.Fatal(err)
		}
	})
	t.Logf("%d tokens as one input: %+v; as %d inputs: %+v", len(cat), flat, len(in), batch)
	if batch.launches != flat.launches || batch.attn != c.Layers || batch.copies != flat.copies+2*2*len(in) {
		t.Errorf("a batch of %d: %+v; want %d launches with %d attention_seg and %d copies", len(in), batch, flat.launches,
			c.Layers, flat.copies+2*2*len(in))
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ForwardBatch(in); !errors.Is(err, errEG2Closed) {
		t.Fatalf("a batch after Close: %v, want errEG2Closed", err)
	}
}

// TestEG2CUDA_failedDrain: a failed forward is drained before its scratch is freed; when that drain fails the
// accelerator refuses forwards and batches and keeps its device memory, and Close frees nothing until a drain
// succeeds, then empties the ledger; a second Close does nothing.
func TestEG2CUDA_failedDrain(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	V := m.Config().VocabSize
	in := [][]int{eg2SeqIDs(7, 1, V), eg2SeqIDs(5, 2, V)}
	if _, err := a.ForwardBatch(in); err != nil {
		t.Fatal(err)
	}
	dev := a.ops.dev
	held, _ := dev.LedgerLen()
	a.defect.failForward = true
	if _, err := a.ForwardBatch(in); err == nil {
		t.Fatal("the planted failure was not reported")
	}
	if n, _ := dev.LedgerLen(); n != held || a.broken != nil {
		t.Fatalf("after a drained failure: %d allocations (want %d), broken %v", n, held, a.broken)
	}
	a.defect.drainFails = 1
	if _, err := a.ForwardBatch(in); err == nil {
		t.Fatal("the planted failure was not reported")
	}
	a.defect.failForward = false
	kept, _ := dev.LedgerLen()
	if a.broken == nil || kept <= held {
		t.Fatalf("after an undrained failure: broken %v, %d allocations (held %d before)", a.broken, kept, held)
	}
	if _, err := a.ForwardBatch(in); !errors.Is(err, a.broken) {
		t.Fatalf("a batch on a broken accelerator: %v", err)
	}
	if _, _, err := a.Forward(in[0], false); !errors.Is(err, a.broken) {
		t.Fatalf("a forward on a broken accelerator: %v", err)
	}
	a.defect.drainFails = 1
	if err := a.Close(); err == nil {
		t.Fatal("Close returned nil though its drain failed")
	}
	if n, _ := dev.LedgerLen(); n != kept || a.closed {
		t.Fatalf("after a failed Close: %d allocations (want %d), closed %v", n, kept, a.closed)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close once the drain recovers: %v", err)
	}
	if allocs, objs := dev.LedgerLen(); allocs != 0 || objs != 0 {
		t.Fatalf("after Close the ledger holds %d allocations and %d objects", allocs, objs)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("a second Close: %v", err)
	}
}

// TestEG2CUDA_realBatchParity: on the real checkpoint (GOINFER_HEAVY_TESTS=1, GOINFER_EG2_DIR), ForwardBatch gives every
// input bit for bit what Forward gives it alone: nine inputs of 1 to 2,048 tokens around the 512-radius window in one
// batch (6,673 rows); a 1,025-token input beside neighbours of 1, 512, 513 and 1,500 tokens; and the golden's 48 texts
// in batches of up to 8,192 rows, each pooled embedding at cosine >= 0.9999 with the sentence-transformers reference.
// Two inputs alone are also checked against the CPU forward.
func TestEG2CUDA_realBatchParity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" || os.Getenv("GOINFER_EG2_DIR") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and GOINFER_EG2_DIR (loads the real checkpoint)")
	}
	dir := os.Getenv("GOINFER_EG2_DIR")
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not local disk (CLAUDE.md)", dir)
	}
	m, err := embeddinggemma2.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("cuda")
	if err != nil {
		t.Fatalf("the cuda accelerator: %v", err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	var name string
	if err := a.ops.do(func() error { name = a.ops.dev.Name(); return nil }); err != nil {
		t.Fatal(err)
	}
	c := m.Config()
	V, E := c.VocabSize, m.Dim()
	t.Logf("device %q; %s; window radius %d", name, dir, c.SlidingWindow)
	single := map[string][]float32{}
	alone := func(key string, ids []int) []float32 {
		if h, ok := single[key]; ok {
			return h
		}
		h, _, err := a.Forward(ids, false)
		if err != nil {
			t.Fatalf("%s alone: %v", key, err)
		}
		single[key] = h
		return h
	}
	check := func(what string, batch [][]int, keys []string) {
		t.Helper()
		got, err := a.ForwardBatch(batch)
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		for i, ids := range batch {
			if d := eg2FirstDiff(got[i], alone(keys[i], ids), E); d != "" {
				t.Fatalf("%s, input %d (%d tokens): differs from the input alone at %s", what, i, len(ids), d)
			}
		}
	}
	lens := []int{1, 13, 511, 512, 513, 1024, 1025, 1026, 2048}
	var mixed [][]int
	var keys []string
	for i, n := range lens {
		mixed = append(mixed, eg2SeqIDs(n, 7+i, V))
		keys = append(keys, fmt.Sprintf("len%d", n))
	}
	check("nine lengths around the window", mixed, keys)
	target := mixed[6] // 1,025 tokens
	for _, n := range []int{1, 512, 513, 1500} {
		other := eg2SeqIDs(n, 500+n, V)
		on := fmt.Sprintf("nb%d", n)
		check(fmt.Sprintf("1,025 tokens after %d", n), [][]int{other, target}, []string{on, "len1025"})
		check(fmt.Sprintf("1,025 tokens before %d", n), [][]int{target, other}, []string{"len1025", on})
	}
	for _, i := range []int{4, 6} { // 513 and 1,025 tokens alone, against the CPU
		cpu, _, err := m.ForwardCPU(mixed[i], false)
		if err != nil {
			t.Fatal(err)
		}
		if cs := eg2Cos(single[keys[i]], cpu); !(cs >= 0.9999) {
			t.Fatalf("%d tokens alone: cosine %.9f with the CPU", lens[i], cs)
		}
	}
	raw, err := os.ReadFile("../../testdata/embeddinggemma2-real/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Items []eg2GoldenItem `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	worst, groups := 1.0, 0
	for i := 0; i < len(g.Items); {
		var batch [][]int
		var bk []string
		total := 0
		start := i
		for ; i < len(g.Items) && total+len(g.Items[i].IDs) <= eg2MaxTokens; i++ {
			batch, bk, total = append(batch, g.Items[i].IDs), append(bk, fmt.Sprintf("golden%d", i)), total+len(g.Items[i].IDs)
		}
		got, err := a.ForwardBatch(batch)
		if err != nil {
			t.Fatal(err)
		}
		groups++
		for j, ids := range batch {
			if d := eg2FirstDiff(got[j], alone(bk[j], ids), E); d != "" {
				t.Fatalf("golden item %d in a batch: differs from the item alone at %s", start+j, d)
			}
			cs := eg2Cos(eg2PoolNorm(got[j], len(ids), E, E), g.Items[start+j].Embedding)
			worst = min(worst, cs)
			if !(cs >= 0.9999) {
				t.Errorf("golden item %d: cosine %.9f with the reference", start+j, cs)
			}
		}
	}
	t.Logf("RESULT bit-identical to singletons in every batch; golden: %d texts in %d batches, worst cosine %.12f", len(g.Items), groups, worst)
}

// TestEG2CUDA_encoderBatch: Encoder.EmbedIDsBatch on the CUDA accelerator runs the batch, one forward with one
// attention_seg per layer for the whole batch, and gives every input bit for bit what EmbedIDs gives it alone.
func TestEG2CUDA_encoderBatch(t *testing.T) {
	m := eg2TinyModel(t)
	vocab := map[string]int{"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
	raw, err := json.Marshal(map[string]any{
		"added_tokens": []map[string]any{{"id": 0, "content": "<pad>"}, {"id": 1, "content": "<eos>"}, {"id": 2, "content": "<bos>"},
			{"id": 3, "content": "<unk>"}},
		"model": map[string]any{"type": "BPE", "unk_token": "<unk>", "vocab": vocab, "merges": [][2]string{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadJSONBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := embeddinggemma2.NewEncoder(m, tok, nil)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := enc.UseAccelerator("cuda")
	if err != nil {
		t.Skipf("no CUDA device for EmbeddingGemma 2: %v", err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	V := m.Config().VocabSize
	in := [][]int{eg2SeqIDs(9, 1, V), eg2SeqIDs(64, 2, V), eg2SeqIDs(2, 3, V)}
	var got [][]float32
	n := eg2Count(t, a, func() {
		if got, err = enc.EmbedIDsBatch(in); err != nil {
			t.Fatal(err)
		}
	})
	if n.attn != m.Config().Layers {
		t.Fatalf("EmbedIDsBatch: %d attention_seg launches, want one forward's %d", n.attn, m.Config().Layers)
	}
	for i, ids := range in {
		want, err := enc.EmbedIDs(ids)
		if err != nil {
			t.Fatal(err)
		}
		if d := eg2FirstDiff(got[i], want, len(want)); d != "" {
			t.Fatalf("input %d: EmbedIDsBatch differs from EmbedIDs at %s", i, d)
		}
	}
}

// TestEG2CUDA_bounds: a sliding row's key range is every key within the radius, the radius itself included, clipped to
// the input; a full row's is the whole input.
func TestEG2CUDA_bounds(t *testing.T) {
	lo, hi := eg2Bounds(10, 3)
	for i := range 10 {
		for j := range 10 {
			in := int32(j) >= lo[i] && int32(j) < hi[i]
			if want := j-i <= 3 && i-j <= 3; in != want {
				t.Errorf("row %d key %d: in range %v, want %v", i, j, in, want)
			}
		}
	}
	lo, hi = eg2Bounds(5, -1)
	for i := range 5 {
		if lo[i] != 0 || hi[i] != 5 {
			t.Errorf("full row %d: [%d, %d), want [0, 5)", i, lo[i], hi[i])
		}
	}
}

// TestEG2CUDA_contract: ForwardEmbeds leaves its input alone; a repeated forward gives the same values bit for bit,
// also when several goroutines share the accelerator; over-long, empty and out-of-vocabulary inputs are refused; and
// after Close (twice) a forward is an error.
func TestEG2CUDA_contract(t *testing.T) {
	m := eg2TinyModel(t)
	a := newEG2ForTest(t, m)
	ids := []int{2, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 1}
	x, err := m.EmbedTokens(ids)
	if err != nil {
		t.Fatal(err)
	}
	keep := slices.Clone(x)
	ref, _, err := a.ForwardEmbeds(x, len(ids), false)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(x, keep) {
		t.Fatal("ForwardEmbeds wrote its input")
	}
	var wg sync.WaitGroup
	errs := make([]error, 4)
	for g := range errs {
		wg.Go(func() {
			for range 3 {
				got, _, err := a.Forward(ids, false)
				if err == nil && !slices.Equal(got, ref) {
					err = errors.New("a repeated forward differs")
				}
				if err != nil {
					errs[g] = err
					return
				}
			}
		})
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		t.Fatal(err)
	}
	if _, _, err := a.Forward(nil, false); err == nil {
		t.Error("an empty input was accepted")
	}
	if _, _, err := a.Forward([]int{m.Config().VocabSize}, false); err == nil {
		t.Error("an out-of-vocabulary id was accepted")
	}
	H := m.Config().Hidden
	if _, _, err := a.ForwardEmbeds(make([]float32, (eg2MaxTokens+1)*H), eg2MaxTokens+1, false); err == nil {
		t.Error("an input over the accelerator's length was accepted")
	}
	if err := errors.Join(a.Close(), a.Close()); err != nil {
		t.Fatal(err)
	}
	if _, _, err := a.Forward(ids, false); !errors.Is(err, errEG2Closed) {
		t.Fatalf("forward after Close: %v, want errEG2Closed", err)
	}
}

// eg2GoldenItem is one text of the real golden: its prompt name, the reference's ids and its 768-wide embedding.
type eg2GoldenItem struct {
	Prompt    string    `json:"prompt"`
	IDs       []int     `json:"ids"`
	Embedding []float32 `json:"embedding"`
}

// TestEG2CUDA_realParity: the real google/embeddinggemma-2 (GOINFER_EG2_DIR, a local-disk checkpoint) on the CUDA
// accelerator, through Accelerator.Forward on the committed sentence-transformers golden's own ids
// (testdata/embeddinggemma2-real/golden.json: 48 texts, 10 to 1,771 tokens, so the longest reach past both edges of the
// 512-radius window). Every pooled, normalised embedding is finite and at cosine >= 0.9999 with the reference, at 768
// and at its 512-wide Matryoshka prefix. Three inputs (the shortest, the longest under the window's width and the
// longest) are also run on the CPU forward, and every layer state and the projected hidden state must match it at
// 0.9999. Wall time is logged for reading, not as a speed result.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_EG2_DIR=<checkpoint> go test -tags cuda -count=1 -run '^TestEG2CUDA_realParity$' -v ./backend/cuda/
func TestEG2CUDA_realParity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and GOINFER_EG2_DIR (loads the real checkpoint)")
	}
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not local disk (CLAUDE.md)", dir)
	}
	raw, err := os.ReadFile("../../testdata/embeddinggemma2-real/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Revision string          `json:"model_revision"`
		Items    []eg2GoldenItem `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	m, err := embeddinggemma2.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	tLoad := time.Since(t0)
	acc, err := m.NewAccelerator("cuda")
	if err != nil {
		t.Fatalf("the cuda accelerator: %v", err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	tAttach := time.Since(t0) - tLoad
	var name string
	if err := a.ops.do(func() error { name = a.ops.dev.Name(); return nil }); err != nil {
		t.Fatal(err)
	}
	c := m.Config()
	t.Logf("device %q; checkpoint %s (golden revision %s); %d layers, window radius %d; host load %.1f s, device upload %.1f s",
		name, dir, g.Revision, c.Layers, c.SlidingWindow, tLoad.Seconds(), tAttach.Seconds())
	E := m.Dim()
	const prefix = 512
	worst, worstP := 1.0, 1.0
	var total time.Duration
	for i, it := range g.Items {
		s := time.Now()
		h, _, err := acc.Forward(it.IDs, false)
		if err != nil {
			t.Fatalf("item %d (%d tokens): %v", i, len(it.IDs), err)
		}
		d := time.Since(s)
		total += d
		v := eg2PoolNorm(h, len(it.IDs), E, E)
		if !eg2Finite(v) {
			t.Fatalf("item %d: the embedding is not finite", i)
		}
		cs := eg2Cos(v, it.Embedding)
		cp := eg2Cos(eg2PoolNorm(h, len(it.IDs), E, prefix), it.Embedding[:prefix])
		worst, worstP = min(worst, cs), min(worstP, cp)
		t.Logf("item %2d %-15q %5d tokens: 1-cosine %.2e, %d-prefix %.2e; %.1f ms", i, it.Prompt, len(it.IDs), 1-cs, prefix, 1-cp, d.Seconds()*1e3)
		if !(cs >= 0.9999) || !(cp >= 0.9999) {
			t.Errorf("item %d (%d tokens): cosine %.9f, %d-prefix %.9f, under 0.9999", i, len(it.IDs), cs, prefix, cp)
		}
	}
	t.Logf("RESULT %d texts: worst cosine %.12f, worst %d-prefix %.12f; device forwards %.2f s in all", len(g.Items), worst, prefix, worstP, total.Seconds())
	order := slices.Clone(g.Items)
	slices.SortFunc(order, func(x, y eg2GoldenItem) int { return len(x.IDs) - len(y.IDs) })
	var underWindow []int
	for _, it := range order {
		if len(it.IDs) <= 2*c.SlidingWindow+1 {
			underWindow = it.IDs
		}
	}
	longest := order[len(order)-1].IDs
	var stock, transposes, transposed int
	gpu.TraceForTest = func(e gpu.TraceEvent) {
		if e.Op == gpu.TraceLaunch {
			switch e.Pipe {
			case a.ops.vit.AttentionSeg.Ptr():
				stock++
			case a.combinedTranspose.Ptr():
				transposes++
			case a.combinedAttention.Ptr():
				transposed++
			}
		}
	}
	_, _, err = acc.Forward(longest, false)
	gpu.TraceForTest = nil
	if err != nil {
		t.Fatal(err)
	}
	wantStock, wantTransposed := c.Layers, 0
	if len(longest) >= 128 && a.combinedAttention != (Pipeline{}) {
		wantStock, wantTransposed = 0, c.Layers
	}
	t.Logf("the %d-token forward: %d stock, %d transpose, %d transposed-attention launches", len(longest), stock, transposes, transposed)
	if stock != wantStock || transposes != wantTransposed || transposed != wantTransposed {
		t.Errorf("want %d stock and %d each transpose and transposed-attention launches", wantStock, wantTransposed)
	}
	for _, ids := range [][]int{order[0].IDs, underWindow, order[len(order)-1].IDs} {
		cpuLast, cpuLayers, err := m.ForwardCPU(ids, true)
		if err != nil {
			t.Fatal(err)
		}
		last, layers, err := acc.Forward(ids, true)
		if err != nil {
			t.Fatal(err)
		}
		lw, lwAt := 1.0, -1
		for li := range cpuLayers {
			if cc := eg2Cos(layers[li], cpuLayers[li]); cc < lw {
				lw, lwAt = cc, li
			}
		}
		lc := eg2Cos(last, cpuLast)
		t.Logf("against the CPU forward, %d tokens: worst layer state 1-cosine %.2e (index %d), last hidden %.2e", len(ids), 1-lw, lwAt, 1-lc)
		if !(lw >= 0.9999) || !(lc >= 0.9999) {
			t.Errorf("%d tokens: worst layer state %.9f at %d, last hidden %.9f, under 0.9999", len(ids), lw, lwAt, lc)
		}
	}
}

func TestEG2CombinedTiny(t *testing.T) {
	if !slices.Contains(embeddinggemma2.Accelerators(), "cuda-eg2-combined") {
		t.Skip("requires eg2_combined build tag")
	}
	checkEG2Transposed(t, "cuda-eg2-combined", true, eg2TinyModel(t), [][]int{{36}, {37}, {12, 24}, {12, 25}, {127, 10}, {64, 64}, {5, 128}, {129, 1}})
}

func TestEG2CombinedReal(t *testing.T) {
	if !slices.Contains(embeddinggemma2.Accelerators(), "cuda-eg2-combined") {
		t.Skip("requires eg2_combined build tag")
	}
	dir := os.Getenv("GOINFER_EG2_DIR")
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" || dir == "" {
		t.Skip("requires opt-in real checkpoint")
	}
	model, err := embeddinggemma2.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	checkEG2Transposed(t, "cuda-eg2-combined", true, model, [][]int{{36}, {37}, {7, 127}, {128, 9}, {511, 13}, {31, 513}})
}

func TestEG2TransposedJITFallback(t *testing.T) {
	for _, tt := range []struct {
		err  error
		want bool
	}{
		{gc.ErrNoBinaryForGPU, true},
		{gc.ErrUnsupportedPTXVersion, true},
		{gc.ErrInvalidPTX, false},
		{gc.ErrOutOfMemory, false},
		{errors.New("missing kernel entry"), false},
	} {
		if got := eg2TransposedJITUnsupported(fmt.Errorf("load module: %w", tt.err)); got != tt.want {
			t.Errorf("optional JIT fallback for %v = %t, want %t", tt.err, got, tt.want)
		}
	}
}

func TestEG2CUDA_defaultTransposedTiny(t *testing.T) {
	uuid := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if uuid == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID to the target CUDA device's GPU UUID")
	}
	checkEG2Transposed(t, "cuda:"+uuid, false, eg2TinyModel(t), [][]int{
		{36}, {64, 64}, {127, 96}, {96, 127}, {128}, {128, 96}, {96, 128}, {96},
	})
}

func checkEG2Transposed(t *testing.T, selector string, shortTiled bool, model *embeddinggemma2.Model, shapes [][]int) {
	t.Helper()
	accel, err := model.NewAccelerator(selector)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := accel.Close(); err != nil {
			t.Error(err)
		}
	})
	a := accel.(*eg2Accel)
	if strings.HasPrefix(selector, "cuda:") && !strings.EqualFold(a.AttachedDevice().ID, selector) {
		t.Fatalf("default CUDA attached to %q", a.AttachedDevice().ID)
	}
	transpose, attention := a.combinedTranspose, a.combinedAttention
	if transpose == (Pipeline{}) || attention == (Pipeline{}) || a.ops.shortTiled != shortTiled {
		t.Fatalf("%s transposed attention or short GEMM policy is unavailable", selector)
	}
	for _, shape := range shapes {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			a.combinedTranspose, a.combinedAttention, a.ops.shortTiled = Pipeline{}, Pipeline{}, false
			ids := make([][]int, len(shape))
			want := make([][]float32, len(shape))
			total, longest := 0, 0
			for i, n := range shape {
				ids[i] = eg2SeqIDs(n, 13+i, model.Config().VocabSize)
				total, longest = total+n, max(longest, n)
				out, _, err := a.Forward(ids[i], false)
				if err != nil {
					t.Fatal(err)
				}
				want[i] = out
			}
			a.combinedTranspose, a.combinedAttention, a.ops.shortTiled = transpose, attention, shortTiled
			var tiled, transposes, attn, stock int
			gpu.TraceForTest = func(event gpu.TraceEvent) {
				if event.Op != gpu.TraceLaunch {
					return
				}
				switch event.Pipe {
				case a.ops.vit.GEMMF32Tiled.Ptr():
					tiled++
				case transpose.Ptr():
					transposes++
				case attention.Ptr():
					attn++
				case a.ops.vit.AttentionSeg.Ptr():
					stock++
				}
			}
			defer func() { gpu.TraceForTest = nil }()
			out, err := a.ForwardBatch(ids)
			if err != nil {
				t.Fatal(err)
			}
			for i, got := range out {
				if diff := eg2FirstDiff(got, want[i], model.Dim()); diff != "" {
					t.Fatalf("input %d differs at %s", i, diff)
				}
			}
			if (tiled > 0) != (shortTiled && total <= 36) {
				t.Errorf("total rows %d produced %d short tiled launches", total, tiled)
			}
			wantAttn := 0
			if longest >= 128 {
				wantAttn = model.Config().Layers
			}
			if attn != wantAttn || transposes != wantAttn || stock != model.Config().Layers-wantAttn {
				t.Errorf("longest input %d: %d transpose, %d transposed attention and %d stock, want %d, %d and %d",
					longest, transposes, attn, stock, wantAttn, wantAttn, model.Config().Layers-wantAttn)
			}
		})
	}
}

func TestEG2CUDA_largeAggregateBatch(t *testing.T) {
	m := eg2TinyModel(t)
	ids := make([][]int, 6)
	for i, length := range []int{2048, 2048, 2048, 2048, 2048, 2047} {
		ids[i] = eg2SeqIDs(length, i+1, m.Config().VocabSize)
	}
	for _, selector := range []string{"cuda", "cuda-eg2-combined"} {
		t.Run(selector, func(t *testing.T) {
			a := newEG2ForTest(t, m)
			if selector == "cuda-eg2-combined" {
				if err := a.enableCombined(); err != nil {
					t.Fatal(err)
				}
			}
			batch, err := a.ForwardBatch(ids)
			if err != nil {
				t.Fatal(err)
			}
			for i, input := range ids {
				single, _, err := a.Forward(input, false)
				if err != nil {
					t.Fatal(err)
				}
				if diff := eg2FirstDiff(batch[i], single, m.Dim()); diff != "" {
					t.Fatalf("input %d differs at %s", i, diff)
				}
			}
		})
	}
}

func TestEG2CUDA_aggregateIndexLimit(t *testing.T) {
	a := &eg2Accel{cfg: embeddinggemma2.Config{Hidden: 65536, VocabSize: 1, MaxPositions: eg2MaxTokens}}
	input := make([]int, eg2MaxTokens)
	// This crosses signed element indexing even though every member is valid.
	if _, err := a.ForwardBatch([][]int{input, input, input, input}); err == nil {
		t.Fatal("CUDA accepted an aggregate that would overflow its indexing")
	}
}

func TestEG2CUDA_refusesOversizedIDsBeforeGather(t *testing.T) {
	const width = 512
	cfg := embeddinggemma2.Config{Hidden: width, VocabSize: 1, MaxPositions: eg2MaxTokens + 10}
	a := eg2Accel{cfg: cfg}
	ids := make([]int, eg2MaxTokens+1)
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	_, _, err := a.Forward(ids, false)
	runtime.ReadMemStats(&after)
	if err == nil || !strings.Contains(err.Error(), "over the accelerator") {
		t.Fatalf("oversized input: %v", err)
	}
	gatherBytes := uint64(len(ids) * width * 4)
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated >= gatherBytes {
		t.Fatalf("refused input allocated %d bytes, enough for the %d-byte gather", allocated, gatherBytes)
	}
}

func TestEG2CUDA_closedRejectsIDsBeforeGather(t *testing.T) {
	a := eg2Accel{closed: true}
	if _, _, err := a.Forward([]int{0}, false); err != errEG2Closed {
		t.Fatalf("closed accelerator returned %v, want %v", err, errEG2Closed)
	}
}

func TestEG2CUDA_configuredPositions(t *testing.T) {
	a := newEG2ForTest(t, eg2TinyModel(t))
	a.cfg.MaxPositions = 4
	ids := make([]int, 5)
	_, _, singleErr := a.Forward(ids, false)
	_, _, embedsErr := a.ForwardEmbeds(make([]float32, len(ids)*a.cfg.Hidden), len(ids), false)
	_, batchErr := a.ForwardBatch([][]int{ids[:4], ids})
	for name, err := range map[string]error{"IDs": singleErr, "rows": embedsErr, "batch": batchErr} {
		if err == nil || !strings.Contains(err.Error(), "over max_position_embeddings 4") {
			t.Errorf("%s: expected model-position limit error, got %v", name, err)
		}
	}
	if got, err := a.ForwardBatch([][]int{ids[:4], ids[:4]}); err != nil || len(got) != 2 {
		t.Fatalf("two valid sequences whose total exceeds the per-input cap: %d results, %v", len(got), err)
	}
}

var eg2GGUFDir = flag.String("eg2-gguf-dir", "", "directory containing the pinned EmbeddingGemma 2 Q8_0 and UD-Q6_K_XL text GGUF files")

func TestEG2GGUFCUDA(t *testing.T) {
	if *eg2GGUFDir == "" {
		t.Skip("set -eg2-gguf-dir to run the GGUF CUDA device test")
	}
	files := []struct{ suffix, digest string }{
		{"Q8_0", "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"},
		{"UD-Q6_K_XL", "dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3"},
	}
	for _, model := range files {
		t.Run(model.suffix, func(t *testing.T) {
			path := filepath.Join(*eg2GGUFDir, "embeddinggemma-2-"+model.suffix+".gguf")
			verifyEG2GGUFHash(t, path, model.digest)
			enc, err := embeddinggemma2.LoadGGUFEncoder(path, map[string]string{
				"query": "task: search result | query: ",
			})
			if err != nil {
				t.Fatal(err)
			}
			query, err := enc.Tokenize("Where is the index stored?", "query")
			if err != nil {
				t.Fatal(err)
			}
			inputs := [][]int{query, eg2GGUFIDs(511), eg2GGUFIDs(514)}
			cpu, err := enc.EmbedIDsBatch(inputs)
			if err != nil {
				t.Fatal(err)
			}
			tooLong := eg2GGUFIDs(embeddinggemma2.MaxTokens + 1)
			if vec, err := enc.EmbedIDs(tooLong); vec != nil || err == nil {
				t.Fatalf("CPU accepted %d positions", len(tooLong))
			}

			accel, err := enc.UseAccelerator("cuda")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := accel.Close(); err != nil {
					t.Error(err)
				}
			})
			if accel.Name() != "cuda" {
				t.Fatalf("attached accelerator %q, want cuda", accel.Name())
			}
			single, err := enc.EmbedIDs(query)
			if err != nil {
				t.Fatal(err)
			}
			batch, err := enc.EmbedIDsBatch(inputs)
			if err != nil {
				t.Fatal(err)
			}
			if len(batch) != len(inputs) || len(cpu) != len(inputs) {
				t.Fatalf("batch sizes CPU=%d CUDA=%d, want %d", len(cpu), len(batch), len(inputs))
			}
			for i, got := range batch {
				t.Logf("batch input with %d prepared IDs", len(inputs[i]))
				checkEG2GGUFVector(t, got, cpu[i])
			}
			checkEG2GGUFVector(t, single, cpu[0])
		})
	}
}

func verifyEG2GGUFHash(t *testing.T, path, want string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != want {
		t.Fatalf("%s SHA-256 %s, want %s", path, got, want)
	}
}

func eg2GGUFIDs(n int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (7919*i+41)%200000 + 5
	}
	ids[0], ids[n-1] = 2, 1
	return ids
}

func checkEG2GGUFVector(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != 768 || len(want) != 768 || !eg2Finite(got) || !eg2Finite(want) {
		t.Fatal("expected finite 768-dimensional CPU and CUDA vectors")
	}
	var maxError, gotNorm, wantNorm float64
	for i, x := range got {
		a, b := float64(x), float64(want[i])
		maxError = max(maxError, math.Abs(a-b))
		gotNorm += a * a
		wantNorm += b * b
	}
	if math.Abs(math.Sqrt(gotNorm)-1) > 1e-5 || math.Abs(math.Sqrt(wantNorm)-1) > 1e-5 {
		t.Fatal("expected unit-length CPU and CUDA vectors")
	}
	cosine := eg2Cos(got, want)
	t.Logf("CUDA vs CPU cosine %.12f, max coordinate error %.3g", cosine, maxError)
	if !(cosine >= 0.99999) || maxError > 3e-5 {
		t.Fatal("CUDA vector differs from the same-artifact CPU result")
	}
}

func BenchmarkEG2CUDALongBatch(b *testing.B) {
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		b.Skip("set GOINFER_EG2_DIR to the EmbeddingGemma 2 checkpoint")
	}
	e, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		b.Fatal(err)
	}
	for _, selector := range []string{"cuda", "cuda-eg2-combined"} {
		if !slices.Contains(embeddinggemma2.Accelerators(), selector) {
			continue
		}
		b.Run(selector, func(b *testing.B) {
			a, err := e.UseAccelerator(selector)
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(func() {
				if err := a.Close(); err != nil {
					b.Error(err)
				}
			})
			for _, shape := range [][2]int{{64, 64}, {8, 512}, {2, 2048}, {4, 2048}} {
				count, length := shape[0], shape[1]
				b.Run(fmt.Sprintf("%dx%d", count, length), func(b *testing.B) {
					benchmarkEG2Batch(b, e, eg2FixedBatch(e, count, length))
				})
			}
			b.Run("6x2048-ragged", func(b *testing.B) {
				ids := eg2FixedBatch(e, 6, 2048)
				ids[5] = ids[5][:2047]
				benchmarkEG2Batch(b, e, ids)
			})
		})
	}
}

func benchmarkEG2Batch(b *testing.B, e *embeddinggemma2.Encoder, batch [][]int) {
	var tokens int
	for _, input := range batch {
		tokens += len(input)
	}
	if _, err := e.EmbedIDsBatch(batch); err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := e.EmbedIDsBatch(batch); err != nil {
			b.Fatal(err)
		}
	}
	passes := float64(b.N) / b.Elapsed().Seconds()
	b.ReportMetric(float64(len(batch))*passes, "passages/s")
	b.ReportMetric(float64(tokens)*passes, "tokens/s")
}

// TestEG2CUDALargeRealBatch verifies pooled output across singleton, packed,
// split and reordered inputs on the official checkpoint.
func TestEG2CUDALargeRealBatch(t *testing.T) {
	dir := os.Getenv("GOINFER_EG2_DIR")
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" || dir == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 and GOINFER_EG2_DIR")
	}
	e, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids := eg2FixedBatch(e, 6, 2048)
	ids[5] = ids[5][:2047]
	for _, selector := range []string{"cuda", "cuda-eg2-combined"} {
		t.Run(selector, func(t *testing.T) {
			if !slices.Contains(embeddinggemma2.Accelerators(), selector) {
				t.Skip("requires eg2_combined build tag")
			}
			acc, err := e.UseAccelerator(selector)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := acc.Close(); err != nil {
					t.Error(err)
				}
			})
			packed, err := e.EmbedIDsBatch(ids)
			if err != nil {
				t.Fatal(err)
			}
			reversed := slices.Clone(ids)
			slices.Reverse(reversed)
			reordered, err := e.EmbedIDsBatch(reversed)
			if err != nil {
				t.Fatal(err)
			}
			slices.Reverse(reordered)
			for i, input := range ids {
				single, err := e.EmbedIDs(input)
				if err != nil {
					t.Fatal(err)
				}
				if diff := eg2FirstDiff(packed[i], single, e.HiddenDim()); diff != "" {
					t.Errorf("input %d differs from singleton at %s", i, diff)
				}
			}
			var split [][]float32
			for _, inputs := range [][][]int{ids[:2], ids[2:]} {
				part, err := e.EmbedIDsBatch(inputs)
				if err != nil {
					t.Fatal(err)
				}
				split = append(split, part...)
			}
			for _, variant := range []struct {
				name string
				vecs [][]float32
			}{{"split", split}, {"reordered", reordered}} {
				for i, vec := range variant.vecs {
					if diff := eg2FirstDiff(packed[i], vec, e.HiddenDim()); diff != "" {
						t.Errorf("input %d differs from %s at %s", i, variant.name, diff)
					}
				}
			}
		})
	}
}

func eg2FixedBatch(e *embeddinggemma2.Encoder, count, length int) [][]int {
	batch := make([][]int, count)
	for i := range batch {
		batch[i] = eg2SeqIDs(length, i+1, e.Model().Config().VocabSize)
	}
	return batch
}

// checkPostEnqueueRecoveryOrder verifies that an ordered native event trace for
// a post-enqueue allocation failure contains at least one Copy2DAsync before
// the injected allocation failure, a successful recovery Queue.Sync after the
// failure, and checked frees only after that recovery Sync.
func checkPostEnqueueRecoveryOrder(events []string) error {
	failIdx := -1
	for i, ev := range events {
		if ev == "alloc-fail" {
			failIdx = i
			break
		}
	}
	if failIdx < 0 {
		return errors.New("trace missing alloc-fail event")
	}
	sawCopyBefore := false
	for _, ev := range events[:failIdx] {
		if ev == "copy2d" {
			sawCopyBefore = true
			break
		}
	}
	if !sawCopyBefore {
		return fmt.Errorf("expected copy2d before alloc-fail in %v", events)
	}
	syncIdx := -1
	for i := failIdx + 1; i < len(events); i++ {
		if events[i] == "free" && syncIdx < 0 {
			return fmt.Errorf("free occurred at %d before recovery sync in %v", i, events)
		}
		if events[i] == "sync-ok" && syncIdx < 0 {
			syncIdx = i
		}
	}
	if syncIdx < 0 {
		return fmt.Errorf("missing recovery sync-ok after alloc-fail in %v", events)
	}
	sawFreeAfter := false
	for _, ev := range events[syncIdx+1:] {
		if ev == "free" {
			sawFreeAfter = true
			break
		}
	}
	if !sawFreeAfter {
		return fmt.Errorf("missing checked free after recovery sync-ok in %v", events)
	}
	return nil
}

func TestEG2CUDA_RetryOOM_ClassificationContract(t *testing.T) {
	oomPanic := fmt.Errorf("cuda: executor job panicked: %w", gpu.AllocFailedPanic("typed-len", 4096, gc.ErrOutOfMemory))
	nonOOMAlloc := fmt.Errorf("cuda: executor job panicked: %w", gpu.AllocFailedPanic("typed-len", 4096, gc.ErrInvalidValue))
	assertErr := &cudaresult.Error{Code: cudasys.CUDA_ERROR_ASSERT, Op: "cuLaunchKernel"}
	destroyedErr := &cudaresult.Error{Code: cudasys.CUDA_ERROR_CONTEXT_IS_DESTROYED, Op: "cuCtxSynchronize"}
	latched := errors.New("cuda: latched launch error")
	drainErr := errors.New("cuda: drain failed")
	relErr := errors.New("cuda: close failed")
	broken := fmt.Errorf("cuda: broken: %w", drainErr)

	// Clean recovered single-chain CUDA allocation OOM authorizes retry.
	got := classifyEG2ForwardErr(oomPanic, nil, nil, nil, nil, false)
	if !errors.Is(got, embeddinggemma2.ErrBatchOutOfMemory) ||
		!errors.Is(got, embeddinggemma2.ErrDeviceOutOfMemory) ||
		!errors.Is(got, gc.ErrOutOfMemory) {
		t.Fatalf("clean OOM = %v, want ErrBatchOutOfMemory + ErrDeviceOutOfMemory + gc.ErrOutOfMemory", got)
	}
	var af *gpu.AllocFailure
	if !errors.As(got, &af) {
		t.Fatalf("clean OOM lost *gpu.AllocFailure: %v", got)
	}

	// Joined sibling error inside AllocFailure or outside AllocFailure (either order)
	// MUST disqualify retry while still surfacing common OOM and sibling/fatal cause.
	innerJoined := fmt.Errorf("cuda: executor job panicked: %w", gpu.AllocFailedPanic("scratch", 4096, errors.Join(gc.ErrOutOfMemory, assertErr)))
	if !isCUDAFatalExec(innerJoined) || !isCUDAFatalExec(destroyedErr) {
		t.Fatalf("fatal CUDA codes not recognized: assert=%v destroyed=%v", isCUDAFatalExec(innerJoined), isCUDAFatalExec(destroyedErr))
	}
	for _, tc := range []struct {
		name      string
		err       error
		wantOther error
		wantCode  cudasys.CUresult
	}{
		{"joined-inside-alloc-oom-first", innerJoined, assertErr, cudasys.CUDA_ERROR_ASSERT},
		{"joined-inside-alloc-fatal-first", gpu.AllocFailedPanic("scratch", 4096, errors.Join(assertErr, gc.ErrOutOfMemory)), assertErr, cudasys.CUDA_ERROR_ASSERT},
		{"joined-outside-oom-first", errors.Join(oomPanic, assertErr), assertErr, cudasys.CUDA_ERROR_ASSERT},
		{"joined-outside-fatal-first", errors.Join(assertErr, oomPanic), assertErr, cudasys.CUDA_ERROR_ASSERT},
		{"joined-outside-ordinary", errors.Join(oomPanic, errors.New("sibling")), nil, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res := classifyEG2ForwardErr(tc.err, nil, nil, nil, nil, false)
			if errors.Is(res, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("%s must not certify ErrBatchOutOfMemory: %v", tc.name, res)
			}
			if !errors.Is(res, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(res, gc.ErrOutOfMemory) {
				t.Fatalf("%s lost common/native OOM diagnostic: %v", tc.name, res)
			}
			if tc.wantOther != nil && !errors.Is(res, tc.wantOther) {
				t.Fatalf("%s lost sibling cause %v: %v", tc.name, tc.wantOther, res)
			}
			if tc.wantCode != 0 {
				var ce *cudaresult.Error
				if !errors.As(res, &ce) || ce.Code != tc.wantCode {
					t.Fatalf("%s first *cudaresult.Error = %v, want code %v", tc.name, ce, tc.wantCode)
				}
				// Sticky broken constructed via fatalFirst also surfaces the fatal code first.
				sticky := fmt.Errorf("cuda: fatal: %w", fatalFirst(firstCUDAFatalExec(tc.err), tc.err))
				var ceSticky *cudaresult.Error
				if !errors.As(sticky, &ceSticky) || ceSticky.Code != tc.wantCode {
					t.Fatalf("%s sticky first *cudaresult.Error = %v, want code %v", tc.name, ceSticky, tc.wantCode)
				}
			}
		})
	}
	// Latched fatal launch + later allocation OOM surfaces the fatal code first via errors.As.
	latchedFatalBroken := fmt.Errorf("cuda: fatal EmbeddingGemma 2 execution error: %w", assertErr)
	gotLatchedFatal := classifyEG2ForwardErr(oomPanic, assertErr, nil, nil, latchedFatalBroken, false)
	var ceLatched *cudaresult.Error
	if errors.Is(gotLatchedFatal, embeddinggemma2.ErrBatchOutOfMemory) ||
		!errors.Is(gotLatchedFatal, embeddinggemma2.ErrDeviceOutOfMemory) ||
		!errors.Is(gotLatchedFatal, gc.ErrOutOfMemory) ||
		!errors.As(gotLatchedFatal, &ceLatched) || ceLatched.Code != cudasys.CUDA_ERROR_ASSERT {
		t.Fatalf("latched fatal + OOM = %v (first native=%v), want CUDA_ERROR_ASSERT first + ErrDeviceOutOfMemory without retry", gotLatchedFatal, ceLatched)
	}

	// Raw native OOM from a non-allocation operation gets common OOM diagnostic only, never retry.
	rawForwardOOM := classifyEG2ForwardErr(fmt.Errorf("launch: %w", gc.ErrOutOfMemory), nil, nil, nil, nil, false)
	if errors.Is(rawForwardOOM, embeddinggemma2.ErrBatchOutOfMemory) || !errors.Is(rawForwardOOM, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(rawForwardOOM, gc.ErrOutOfMemory) {
		t.Fatalf("raw non-alloc OOM = %v, want ErrDeviceOutOfMemory without ErrBatchOutOfMemory", rawForwardOOM)
	}

	// Text-only OUT_OF_MEMORY matches neither OOM sentinel.
	textOnly := classifyEG2ForwardErr(errors.New("driver text: OUT_OF_MEMORY"), nil, nil, nil, nil, false)
	if errors.Is(textOnly, embeddinggemma2.ErrBatchOutOfMemory) || errors.Is(textOnly, embeddinggemma2.ErrDeviceOutOfMemory) {
		t.Fatalf("text-only OOM matched sentinel: %v", textOnly)
	}

	// Non-OOM AllocFailure matches neither retry nor common OOM.
	gotNonOOM := classifyEG2ForwardErr(nonOOMAlloc, nil, nil, nil, nil, false)
	if errors.Is(gotNonOOM, embeddinggemma2.ErrBatchOutOfMemory) || errors.Is(gotNonOOM, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(gotNonOOM, gc.ErrInvalidValue) {
		t.Fatalf("non-OOM alloc = %v", gotNonOOM)
	}

	// Any latched, drain, release, broken, or closed condition suppresses ErrBatchOutOfMemory
	// while keeping common OOM, native OOM cause, and the fatal cause visible.
	for _, tc := range []struct {
		name                        string
		latched, drain, rel, broken error
		closed                      bool
		wantFatal                   error
	}{
		{"latched", latched, nil, nil, nil, false, latched},
		{"drain", nil, drainErr, nil, broken, false, drainErr},
		{"release", nil, nil, relErr, fmt.Errorf("rollback: %w", relErr), false, relErr},
		{"broken-no-drain", nil, nil, nil, broken, false, broken},
		{"closed", nil, nil, nil, nil, true, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := classifyEG2ForwardErr(oomPanic, tc.latched, tc.drain, tc.rel, tc.broken, tc.closed)
			if errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("%s must not match ErrBatchOutOfMemory: %v", tc.name, err)
			}
			if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) {
				t.Fatalf("%s lost common/native OOM: %v", tc.name, err)
			}
			if tc.wantFatal != nil && !errors.Is(err, tc.wantFatal) {
				t.Fatalf("%s lost fatal cause %v: %v", tc.name, tc.wantFatal, err)
			}
		})
	}

	// Ordered recovery trace positive + negative controls (omitting recovery sync must fail).
	if err := checkPostEnqueueRecoveryOrder([]string{"copy2d", "alloc-fail", "sync-ok", "free"}); err != nil {
		t.Fatalf("valid post-enqueue recovery trace rejected: %v", err)
	}
	if err := checkPostEnqueueRecoveryOrder([]string{"copy2d", "alloc-fail", "free"}); err == nil {
		t.Fatal("negative control: trace omitting recovery sync-ok was accepted")
	}
	if err := checkPostEnqueueRecoveryOrder([]string{"copy2d", "alloc-fail", "free", "sync-ok"}); err == nil {
		t.Fatal("negative control: trace freeing before recovery sync-ok was accepted")
	}

	// Attach classification via newEG2AccelOn maps raw or wrapped native OOM to common
	// ErrDeviceOutOfMemory only, rejects nil opener, and leaves non-OOM/text errors distinct.
	m := eg2TinyModel(t)
	if _, err := newEG2AccelOn(m, false, nil); err == nil || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) {
		t.Fatalf("nil opener = %v, want non-OOM error", err)
	}
	if _, err := newEG2AccelOn(m, false, func() (*Device, error) {
		return nil, fmt.Errorf("primary retain: %w", gc.ErrOutOfMemory)
	}); !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("raw opener attach OOM = %v, want ErrDeviceOutOfMemory + gc.ErrOutOfMemory without ErrBatchOutOfMemory", err)
	}
	if _, err := newEG2AccelOn(m, false, func() (*Device, error) {
		return nil, gpu.AllocFailedPanic("attach", 1024, gc.ErrOutOfMemory)
	}); !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("alloc-panic attach OOM = %v, want ErrDeviceOutOfMemory + gc.ErrOutOfMemory without ErrBatchOutOfMemory", err)
	}
	if _, err := newEG2AccelOn(m, false, func() (*Device, error) {
		return nil, errors.New("plain text OUT_OF_MEMORY")
	}); err == nil || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("text-only attach error matched OOM: %v", err)
	}
	unrelated := errors.New("cuda: synthetic unrelated attach failure")
	if _, err := newEG2AccelOn(m, false, func() (*Device, error) {
		return nil, unrelated
	}); !errors.Is(err, unrelated) || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("unrelated attach = %v", err)
	}

	// Serialized single-slot EG2 attach-cleanup gate + literal-nil bare factory contract
	// (pure CPU): while a retained failed-attach cleanup keeps failing, repeated and
	// simultaneous attaches (including public Model.NewAccelerator for both bare selectors)
	// return literal nil Accelerator with zero opener calls; once cleanup succeeds, the
	// slot clears and the next attach invokes the requested opener.
	t.Run("serialized-single-slot-cleanup-gate", func(t *testing.T) {
		var failMu sync.Mutex
		failClose := true
		stuckErr := errors.New("cuda: synthetic stuck attach cleanup")
		obj := &testCloser{fn: func() error {
			failMu.Lock()
			f := failClose
			failMu.Unlock()
			if f {
				return stuckErr
			}
			return nil
		}}
		stuckDev := &Device{}
		stuckDev.TrackObj(obj)
		stuckOps := &towerOps{dev: stuckDev, reqCh: make(chan func() error), ackCh: make(chan error)}
		go func() {
			for j := range stuckOps.reqCh {
				stuckOps.ackCh <- runJob(j)
			}
		}()
		eg2AttachCleanup.mu.Lock()
		prevPending := eg2AttachCleanup.pending
		eg2AttachCleanup.pending = stuckOps
		eg2AttachCleanup.mu.Unlock()
		defer func() {
			failMu.Lock()
			failClose = false
			failMu.Unlock()
			_ = drainPendingEG2Cleanups()
			eg2AttachCleanup.mu.Lock()
			eg2AttachCleanup.pending = prevPending
			eg2AttachCleanup.mu.Unlock()
		}()

		// Public bare factories ("cuda" and combined bareEG2Factory(true)) must return
		// literal nil interface (not typed-nil *eg2Accel) through Model.NewAccelerator on error.
		combName := fmt.Sprintf("cuda-eg2-combined-test-%d", time.Now().UnixNano())
		embeddinggemma2.RegisterAccelerator(combName, bareEG2Factory(true))
		bareNames := []string{"cuda", combName}
		if slices.Contains(embeddinggemma2.Accelerators(), "cuda-eg2-combined") {
			bareNames = append(bareNames, "cuda-eg2-combined")
		}
		for _, sel := range bareNames {
			acc, err := m.NewAccelerator(sel)
			if acc != nil {
				t.Fatalf("Model.NewAccelerator(%q) returned non-nil interface (%T) on error %v", sel, acc, err)
			}
			if !errors.Is(err, stuckErr) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("Model.NewAccelerator(%q) err = %v, want %v without retry", sel, err, stuckErr)
			}
		}

		var openMu sync.Mutex
		openCalls := 0
		var wg sync.WaitGroup
		errs := make([]error, 6)
		for i := range errs {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				if idx%2 == 0 {
					acc, err := m.NewAccelerator("cuda")
					if acc != nil {
						t.Errorf("goroutine %d Model.NewAccelerator returned non-nil interface (%T)", idx, acc)
					}
					errs[idx] = err
					return
				}
				acc, err := newEG2AccelOn(m, false, func() (*Device, error) {
					openMu.Lock()
					openCalls++
					openMu.Unlock()
					return nil, unrelated
				})
				if acc != nil {
					t.Errorf("goroutine %d returned non-nil accel", idx)
				}
				errs[idx] = err
			}(i)
		}
		wg.Wait()
		for i, err := range errs {
			if !errors.Is(err, stuckErr) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("concurrent blocked attach %d = %v, want %v without retry", i, err, stuckErr)
			}
		}
		if openCalls != 0 {
			t.Fatalf("blocked concurrent attaches invoked opener %d times, want 0", openCalls)
		}
		if n := pendingEG2CleanupsLen(); n != 1 {
			t.Fatalf("pending cleanup count = %d, want 1", n)
		}

		failMu.Lock()
		failClose = false
		failMu.Unlock()
		_, err := newEG2AccelOn(m, false, func() (*Device, error) {
			openMu.Lock()
			openCalls++
			openMu.Unlock()
			return nil, unrelated
		})
		if !errors.Is(err, unrelated) {
			t.Fatalf("post-recovery attach = %v, want %v from opener", err, unrelated)
		}
		if openCalls != 1 {
			t.Fatalf("post-recovery opener calls = %d, want 1", openCalls)
		}
		if n := pendingEG2CleanupsLen(); n != 0 {
			t.Fatalf("pending cleanup count after recovery = %d, want 0", n)
		}
	})

	// Close teardown progress (pure CPU): once Close enters closeChecked and fails
	// on a non-buffer object/context close, a later Close retry must not call
	// drainQueue / q.Sync on an already-destroyed stream.
	t.Run("close-teardown-progress-skips-drain-on-retry", func(t *testing.T) {
		failObj := true
		objErr := errors.New("cuda: synthetic post-stream module close failure")
		dev := &Device{}
		dev.TrackObj(&testCloser{fn: func() error {
			if failObj {
				return objErr
			}
			return nil
		}})
		ops := &towerOps{dev: dev, reqCh: make(chan func() error), ackCh: make(chan error)}
		go func() {
			for j := range ops.reqCh {
				ops.ackCh <- runJob(j)
			}
		}()
		a := &eg2Accel{ops: ops}
		if err := a.Close(); !errors.Is(err, objErr) {
			t.Fatalf("first Close = %v, want %v", err, objErr)
		}
		if a.closed || !a.teardownStarted || a.broken == nil {
			t.Fatalf("after failed teardown: closed=%v teardownStarted=%v broken=%v", a.closed, a.teardownStarted, a.broken)
		}
		// Plant a failing drain: if retry wrongly calls drainQueue(), Close fails here.
		a.defect.drainFails = 1
		failObj = false
		if err := a.Close(); err != nil {
			t.Fatalf("retry Close after post-stream failure = %v, want nil", err)
		}
		if !a.closed {
			t.Fatal("retry Close did not mark accelerator closed")
		}
		if a.defect.drainFails != 1 {
			t.Fatalf("retry Close called drainQueue after teardown started (drainFails=%d)", a.defect.drainFails)
		}
		if allocs, objs := dev.LedgerLen(); allocs != 0 || objs != 0 {
			t.Fatalf("retry Close left ledger (%d, %d)", allocs, objs)
		}
	})
}

type testCloser struct{ fn func() error }

func (c *testCloser) Close() error { return c.fn() }

func TestEG2CUDA_RetryOOM_DeviceAcceptance(t *testing.T) {
	probe, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	probe.ReleaseObjects()

	for _, combined := range []bool{false, true} {
		arm := "packed"
		if combined {
			arm = "combined"
		}
		t.Run(arm, func(t *testing.T) {
			newTinyEnc := func(t *testing.T) *embeddinggemma2.Encoder {
				t.Helper()
				mod := eg2TinyModel(t)
				raw, err := json.Marshal(map[string]any{
					"added_tokens": []map[string]any{
						{"id": 0, "content": "<pad>"},
						{"id": 1, "content": "<eos>"},
						{"id": 2, "content": "<bos>"},
						{"id": 3, "content": "<unk>"},
					},
					"model": map[string]any{
						"type":      "BPE",
						"unk_token": "<unk>",
						"vocab":     map[string]int{"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3},
						"merges":    [][2]string{},
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				tok, err := tokenizer.LoadJSONBytes(raw)
				if err != nil {
					t.Fatal(err)
				}
				e, err := embeddinggemma2.NewEncoder(mod, tok, nil)
				if err != nil {
					t.Fatal(err)
				}
				return e
			}
			m := eg2TinyModel(t)
			V := m.Config().VocabSize
			orig := [][]int{
				eg2SeqIDs(6, 1, V),
				eg2SeqIDs(8, 2, V),
				eg2SeqIDs(5, 3, V),
				eg2SeqIDs(7, 4, V),
			}

			// Helper that attaches a fresh eg2Accel on enc with the chosen selector.
			attachFresh := func(t *testing.T) (*embeddinggemma2.Encoder, *eg2Accel) {
				t.Helper()
				e := newTinyEnc(t)
				name := fmt.Sprintf("cuda-issue25-test-%s-%d", arm, time.Now().UnixNano())
				embeddinggemma2.RegisterAccelerator(name, func(mod *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
					return newEG2Accel(mod, combined)
				})
				acc, err := e.UseAccelerator(name)
				if err != nil {
					t.Fatalf("attach %s: %v", arm, err)
				}
				return e, acc.(*eg2Accel)
			}

			// Baseline outputs on a healthy encoder for exact bit comparison.
			eBase, aBase := attachFresh(t)
			baseBatch, err := eBase.EmbedIDsBatch(orig)
			if err != nil {
				t.Fatalf("baseline EmbedIDsBatch: %v", err)
			}
			baseSingle, err := eBase.EmbedIDs(orig[0])
			if err != nil {
				t.Fatalf("baseline EmbedIDs: %v", err)
			}
			if err := aBase.Close(); err != nil {
				t.Fatalf("baseline Close: %v", err)
			}

			// (1) Pristine multi-input real allocations with injected OOM before submit, then split retry on the same encoder.
			t.Run("pristine-batch-oom-and-split-retry", func(t *testing.T) {
				e, a := attachFresh(t)
				defer func() {
					if err := a.Close(); err != nil {
						t.Errorf("Close: %v", err)
					}
				}()
				dev := a.ops.dev
				heldBefore, _ := dev.LedgerLen()
				var traceMu sync.Mutex
				launches, copies2D := 0, 0
				gpu.TraceForTest = func(ev gpu.TraceEvent) {
					traceMu.Lock()
					defer traceMu.Unlock()
					switch ev.Op {
					case gpu.TraceLaunch:
						launches++
					case gpu.TraceCopy2D:
						copies2D++
					}
				}
				allocs := 0
				dev.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 5 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				in := make([][]int, len(orig))
				for i := range orig {
					in[i] = slices.Clone(orig[i])
				}
				got, err := e.EmbedIDsBatch(in)
				dev.SetAllocHookForTest(nil)
				gpu.TraceForTest = nil
				if got != nil {
					t.Fatalf("pristine OOM returned non-nil outputs")
				}
				if launches != 0 || copies2D != 0 {
					t.Fatalf("pristine OOM issued device work: launches=%d copy2d=%d", launches, copies2D)
				}
				if !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) ||
					!errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) ||
					!errors.Is(err, gc.ErrOutOfMemory) {
					t.Fatalf("pristine OOM err = %v", err)
				}
				for i := range orig {
					if !slices.Equal(in[i], orig[i]) {
						t.Fatalf("input %d mutated", i)
					}
				}
				if heldAfter, _ := dev.LedgerLen(); heldAfter != heldBefore {
					t.Fatalf("scratch leaked on pristine OOM: ledger %d, want %d", heldAfter, heldBefore)
				}
				// Halve between complete inputs on the same encoder and check bit-exact match.
				left, err := e.EmbedIDsBatch(in[:2])
				if err != nil {
					t.Fatalf("left split retry: %v", err)
				}
				right, err := e.EmbedIDsBatch(in[2:])
				if err != nil {
					t.Fatalf("right split retry: %v", err)
				}
				combinedVecs := append(left, right...)
				for i := range orig {
					if diff := eg2FirstDiff(combinedVecs[i], baseBatch[i], m.Dim()); diff != "" {
						t.Fatalf("split output %d differed from baseline at %s", i, diff)
					}
				}
			})

			// (2a) Post-enqueue OOM (fail 2nd batchRope scratch pair after 1st key's Copy2DAsync queued).
			// (2b) Repeatedly fail 2nd RoPE allocation (sin) to prove zero partial-pair leak.
			t.Run("post-enqueue-oom-and-rope-second-alloc-leak", func(t *testing.T) {
				e, a := attachFresh(t)
				defer func() {
					if err := a.Close(); err != nil {
						t.Errorf("Close: %v", err)
					}
				}()
				dev := a.ops.dev
				// First prove repeated 2nd RoPE allocation failure leaks 0 buffers.
				// Singleton allocates 20 base scratch buffers, then rope() allocates cos (#21) and sin (#22).
				heldCold, _ := dev.LedgerLen()
				for attempt := range 5 {
					allocs := 0
					dev.SetAllocHookForTest(func(what string, nBytes int) error {
						allocs++
						if allocs == 22 { // 20 scratch + cos (#21), fail sin (#22)
							return gc.ErrOutOfMemory
						}
						return nil
					})
					_, err := e.EmbedIDs(orig[0])
					dev.SetAllocHookForTest(nil)
					if !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
						t.Fatalf("attempt %d: 2nd RoPE alloc OOM = %v, want ErrBatchOutOfMemory", attempt, err)
					}
					if n, _ := dev.LedgerLen(); n != heldCold {
						t.Fatalf("attempt %d: partial RoPE pair leaked: ledger %d, want %d", attempt, n, heldCold)
					}
					if len(a.ropes) != 0 {
						t.Fatalf("attempt %d: partial RoPE pair published in a.ropes: %d", attempt, len(a.ropes))
					}
				}
				// Warm up caches on orig[0] (length 6) so subsequent singleton at length 10 grows RoPE
				// AFTER layer 0 has queued work on the sliding head_dim and hits the full layer's RoPE growth.
				v0, err := e.EmbedIDs(orig[0])
				if err != nil {
					t.Fatalf("post-recovery singleton: %v", err)
				}
				if diff := eg2FirstDiff(v0, baseSingle, m.Dim()); diff != "" {
					t.Fatalf("post-recovery singleton differed at %s", diff)
				}
				// Now run a multi-input batch whose caches are already warm at >=8:
				// Warm caches up to 8 first.
				if _, err := e.EmbedIDsBatch(orig[:2]); err != nil {
					t.Fatalf("warm batch: %v", err)
				}
				heldWarm, _ := dev.LedgerLen()
				// On warm batch: 20 base scratch + 2 batchRope buffers for key 0 (#21,#22) + Copy2DAsync queued,
				// then fail key 1's batchRope buffer (#23) AFTER copies were enqueued. Record ordered native
				// events (copy2d -> alloc-fail -> sync-ok -> free) to prove recovery drain precedes frees.
				var evMu sync.Mutex
				var events []string
				record := func(ev string) {
					evMu.Lock()
					events = append(events, ev)
					evMu.Unlock()
				}
				gpu.TraceForTest = func(ev gpu.TraceEvent) {
					switch ev.Op {
					case gpu.TraceCopy2D:
						record("copy2d")
					case gpu.TraceFree:
						record("free")
					}
				}
				dev.SetSyncHookForTest(func(err error) {
					if err == nil {
						record("sync-ok")
					} else {
						record("sync-err")
					}
				})
				allocs := 0
				dev.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 23 {
						record("alloc-fail")
						return gc.ErrOutOfMemory
					}
					return nil
				})
				got, err := e.EmbedIDsBatch(orig)
				dev.SetAllocHookForTest(nil)
				dev.SetSyncHookForTest(nil)
				gpu.TraceForTest = nil
				if got != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("post-enqueue OOM = (%v, %v), want nil + ErrBatchOutOfMemory", got, err)
				}
				if orderErr := checkPostEnqueueRecoveryOrder(events); orderErr != nil {
					t.Fatalf("post-enqueue recovery order invalid: %v", orderErr)
				}
				// Negative control: removing sync-ok from the recorded live trace must fail the assertion.
				withoutSync := make([]string, 0, len(events))
				for _, ev := range events {
					if ev != "sync-ok" {
						withoutSync = append(withoutSync, ev)
					}
				}
				if err := checkPostEnqueueRecoveryOrder(withoutSync); err == nil {
					t.Fatalf("drain negative control failed: trace without sync-ok was accepted (%v)", withoutSync)
				}
				if n, _ := dev.LedgerLen(); n != heldWarm {
					t.Fatalf("post-enqueue OOM leaked buffers: ledger %d, want %d", n, heldWarm)
				}
				retryOut, err := e.EmbedIDsBatch(orig)
				if err != nil {
					t.Fatalf("retry after post-enqueue OOM: %v", err)
				}
				for i := range orig {
					if diff := eg2FirstDiff(retryOut[i], baseBatch[i], m.Dim()); diff != "" {
						t.Fatalf("post-enqueue retry output %d differed at %s", i, diff)
					}
				}
			})

			// (3) Fatal combinations: OOM + latched launch error, OOM + failed drain, OOM + failed free.
			t.Run("fatal-combinations-never-retry", func(t *testing.T) {
				// 3a: Fatal launch error at layer 0 + later allocation OOM on layer 1's RoPE alloc.
				// eg2TinyModel has layer 0 sliding (hd=16) and a full layer (hd=32), so singleton hits layer 0 first
				// (latching CUDA_ERROR_ASSERT) and hits OOM on alloc #23. Both the initial failing call and subsequent
				// sticky calls must return CUDA_ERROR_ASSERT first via errors.As (not OOM) with no retry marker, while
				// the initial call also retains ErrDeviceOutOfMemory and gc.ErrOutOfMemory.
				eLatch, aLatch := attachFresh(t)
				devLatch := aLatch.ops.dev
				assertLatch := &cudaresult.Error{Code: cudasys.CUDA_ERROR_ASSERT, Op: "cuLaunchKernel"}
				aLatch.defect.failForward = true
				aLatch.defect.failForwardErr = assertLatch
				allocs := 0
				devLatch.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 23 { // 20 scratch + layer-0 cos(#21), sin(#22); fail full-layer cos(#23)
						return gc.ErrOutOfMemory
					}
					return nil
				})
				_, err := eLatch.EmbedIDs(orig[0])
				devLatch.SetAllocHookForTest(nil)
				aLatch.defect.failForward = false
				aLatch.defect.failForwardErr = nil
				if err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("OOM+latched-fatal must not match ErrBatchOutOfMemory: %v", err)
				}
				var ceLatch *cudaresult.Error
				if !errors.As(err, &ceLatch) || ceLatch.Code != cudasys.CUDA_ERROR_ASSERT {
					t.Fatalf("OOM+latched-fatal first *cudaresult.Error = %v, want CUDA_ERROR_ASSERT", ceLatch)
				}
				if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || !errors.Is(err, assertLatch) {
					t.Fatalf("OOM+latched-fatal missing OOM or fatal cause: %v", err)
				}
				lLatch0, wLatch0 := devLatch.Activity()
				_, errSticky := eLatch.EmbedIDs(orig[0])
				var ceLatchSticky *cudaresult.Error
				if errSticky == nil || errors.Is(errSticky, embeddinggemma2.ErrBatchOutOfMemory) ||
					!errors.As(errSticky, &ceLatchSticky) || ceLatchSticky.Code != cudasys.CUDA_ERROR_ASSERT {
					t.Fatalf("sticky call after OOM+latched-fatal = %v (first native=%v), want CUDA_ERROR_ASSERT without retry", errSticky, ceLatchSticky)
				}
				if lLatch1, wLatch1 := devLatch.Activity(); lLatch1 != lLatch0 || wLatch1 != wLatch0 {
					t.Fatalf("sticky poisoned accelerator ran device work: launches %d->%d waits %d->%d", lLatch0, lLatch1, wLatch0, wLatch1)
				}
				if err := aLatch.Close(); err != nil {
					t.Fatalf("aLatch.Close: %v", err)
				}

				// 3b: OOM + failed drain -> no retry, sticky broken, no premature free, Close obeys drain contract.
				eDrain, aDrain := attachFresh(t)
				devDrain := aDrain.ops.dev
				heldBefore, _ := devDrain.LedgerLen()
				aDrain.defect.drainFails = 1
				allocs = 0
				devDrain.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 5 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				_, err = eDrain.EmbedIDsBatch(orig)
				devDrain.SetAllocHookForTest(nil)
				if err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("OOM+failed-drain must not match ErrBatchOutOfMemory: %v", err)
				}
				if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) {
					t.Fatalf("OOM+failed-drain lost OOM cause: %v", err)
				}
				kept, _ := devDrain.LedgerLen()
				if aDrain.broken == nil || kept <= heldBefore {
					t.Fatalf("OOM+failed-drain did not retain pending buffers: broken=%v ledger=%d before=%d", aDrain.broken, kept, heldBefore)
				}
				if _, err := eDrain.EmbedIDsBatch(orig[:1]); !errors.Is(err, aDrain.broken) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("post-broken batch = %v, want aDrain.broken without retry", err)
				}
				aDrain.defect.drainFails = 1
				if err := aDrain.Close(); err == nil {
					t.Fatal("Close succeeded despite planted drain failure")
				}
				if n, _ := devDrain.LedgerLen(); n != kept {
					t.Fatalf("failed Close altered ledger: %d vs %d", n, kept)
				}
				if err := aDrain.Close(); err != nil {
					t.Fatalf("Close retry after drain recovery: %v", err)
				}
				if allocsLeft, objsLeft := devDrain.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
					t.Fatalf("Close left ledger (%d, %d)", allocsLeft, objsLeft)
				}

				// 3c: OOM + failed rollback free -> no retry, sticky broken, ledger and lease retain failed
				// buffer across both rollback AND a failing first Close, then retryable second Close succeeds.
				recFree := &eg2RecordingLeaser{}
				origLeaser := gpu.NewLeaser
				gpu.NewLeaser = func() gpu.Leaser { return recFree }
				eFree, aFree := attachFresh(t)
				gpu.NewLeaser = origLeaser
				devFree := aFree.ops.dev
				heldBeforeFree, _ := devFree.LedgerLen()
				freeErr := errors.New("cuda: planted buffer close failure")
				failFirst := true
				closeCalls := 0
				devFree.SetCloseBufHookForTest(func(closeReal func() error) error {
					closeCalls++
					if failFirst && closeCalls == 1 {
						return freeErr // leave real buffer open so later Close retry frees it cleanly
					}
					return closeReal()
				})
				allocs = 0
				devFree.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 5 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				_, err = eFree.EmbedIDsBatch(orig)
				devFree.SetAllocHookForTest(nil)
				if err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("OOM+failed-free must not match ErrBatchOutOfMemory: %v", err)
				}
				if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || !errors.Is(err, freeErr) {
					t.Fatalf("OOM+failed-free missing cause: %v", err)
				}
				if aFree.broken == nil {
					t.Fatal("OOM+failed-free did not poison accelerator")
				}
				if n, _ := devFree.LedgerLen(); n != heldBeforeFree+1 {
					t.Fatalf("failed rollback free dropped ledger ownership: got %d, want %d", n, heldBeforeFree+1)
				}
				if _, err := eFree.EmbedIDsBatch(orig[:1]); !errors.Is(err, aFree.broken) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("subsequent call after failed free = %v", err)
				}
				// Keep close failure active across the first Close: Close must fail, keep aFree unclosed,
				// and preserve the remaining live ledger entry and lease charge.
				closeCalls = 0
				if err := aFree.Close(); !errors.Is(err, freeErr) {
					t.Fatalf("first Close with active close failure = %v, want %v", err, freeErr)
				}
				if aFree.closed {
					t.Fatal("first failing Close marked accelerator closed")
				}
				recFree.mu.Lock()
				liveAfterFailClose := recFree.devLive
				recFree.mu.Unlock()
				if n, _ := devFree.LedgerLen(); n == 0 || liveAfterFailClose <= 0 {
					t.Fatalf("failing first Close lost ownership/lease: ledger=%d liveBytes=%d", n, liveAfterFailClose)
				}
				// Clear close hook and retry Close: remaining buffer and lease release cleanly to zero.
				devFree.SetCloseBufHookForTest(nil)
				if err := aFree.Close(); err != nil {
					t.Fatalf("retry aFree.Close: %v", err)
				}
				recFree.mu.Lock()
				liveAfterClose := recFree.devLive
				recFree.mu.Unlock()
				if allocsLeft, objsLeft := devFree.LedgerLen(); allocsLeft != 0 || objsLeft != 0 || liveAfterClose != 0 {
					t.Fatalf("retry aFree.Close left ledger (%d, %d) liveBytes=%d", allocsLeft, objsLeft, liveAfterClose)
				}

				// 3d: Primary fatal CUDA launch error outranks subsequent drain failure, and subsequent
				// valid forwards preserve the original fatal cause while performing zero device work.
				eFatal, aFatal := attachFresh(t)
				devFatal := aFatal.ops.dev
				assertErr := &cudaresult.Error{Code: cudasys.CUDA_ERROR_ASSERT, Op: "cuLaunchKernel"}
				aFatal.defect.failForward = true
				aFatal.defect.failForwardErr = assertErr
				aFatal.defect.drainFails = 1
				_, err = eFatal.EmbedIDsBatch(orig[:1])
				aFatal.defect.failForward = false
				aFatal.defect.failForwardErr = nil
				if err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || !errors.Is(err, assertErr) {
					t.Fatalf("fatal launch + failed drain = %v, want primary %v without retry", err, assertErr)
				}
				l0, w0 := devFatal.Activity()
				if _, err2 := eFatal.EmbedIDsBatch(orig[:1]); !errors.Is(err2, assertErr) || errors.Is(err2, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("subsequent call after primary fatal = %v, want preserved %v", err2, assertErr)
				}
				l1, w1 := devFatal.Activity()
				if l1 != l0 || w1 != w0 {
					t.Fatalf("poisoned accelerator performed device work: launches %d->%d waits %d->%d", l0, l1, w0, w1)
				}
				if err := aFatal.Close(); err != nil {
					t.Fatalf("aFatal.Close: %v", err)
				}

				// 3e: Real-device Close failure AFTER native stream destruction (module/object close
				// fails in ReleaseObjectsChecked after real command stream q is closed): retry Close
				// must skip q.Sync on the destroyed stream and finish tearing down cleanly.
				eTear, aTear := attachFresh(t)
				devTear := aTear.ops.dev
				failTearObj := true
				tearErr := errors.New("cuda: planted post-stream close failure")
				devTear.TrackObj(&testCloser{fn: func() error {
					if failTearObj {
						return tearErr
					}
					return nil
				}})
				if err := aTear.Close(); !errors.Is(err, tearErr) {
					t.Fatalf("first Close with post-stream failure = %v, want %v", err, tearErr)
				}
				if aTear.closed || !aTear.teardownStarted || aTear.broken == nil {
					t.Fatalf("post-stream Close state: closed=%v teardownStarted=%v broken=%v", aTear.closed, aTear.teardownStarted, aTear.broken)
				}
				if allocsLeft, objsLeft := devTear.LedgerLen(); allocsLeft != 0 || objsLeft != 1 {
					t.Fatalf("post-stream failed Close ledger = (%d, %d), want (0, 1)", allocsLeft, objsLeft)
				}
				if _, err := eTear.EmbedIDs(orig[0]); err == nil || !errors.Is(err, aTear.broken) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("forward after post-stream Close failure = %v, want aTear.broken without retry", err)
				}
				failTearObj = false
				if err := aTear.Close(); err != nil {
					t.Fatalf("retry Close after real stream already destroyed = %v, want nil", err)
				}
				if !aTear.closed {
					t.Fatal("retry Close did not mark accelerator closed")
				}
				if allocsLeft, objsLeft := devTear.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
					t.Fatalf("retry Close left ledger (%d, %d)", allocsLeft, objsLeft)
				}
			})

			// (4) Singleton EmbedIDs OOM + attach partial-weight-upload OOM + unrelated attach error.
			t.Run("singleton-and-attach-oom", func(t *testing.T) {
				e, a := attachFresh(t)
				defer func() {
					if err := a.Close(); err != nil {
						t.Errorf("Close: %v", err)
					}
				}()
				dev := a.ops.dev
				allocs := 0
				dev.SetAllocHookForTest(func(what string, nBytes int) error {
					allocs++
					if allocs == 4 {
						return gc.ErrOutOfMemory
					}
					return nil
				})
				in := slices.Clone(orig[0])
				v, err := e.EmbedIDs(in)
				dev.SetAllocHookForTest(nil)
				if v != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || !slices.Equal(in, orig[0]) {
					t.Fatalf("singleton OOM = (%v, %v)", v, err)
				}
				retryV, err := e.EmbedIDs(in)
				if err != nil {
					t.Fatalf("singleton retry: %v", err)
				}
				if diff := eg2FirstDiff(retryV, baseSingle, m.Dim()); diff != "" {
					t.Fatalf("singleton retry differed at %s", diff)
				}

				// Attach OOM after several real weight uploads -> ErrDeviceOutOfMemory only, 0 leaks.
				var attachDev *Device
				attachAllocs := 0
				gotAcc, err := newEG2AccelOn(m, combined, func() (*Device, error) {
					d, err := CreateSystemDefaultDevice()
					if err != nil {
						return nil, err
					}
					attachDev = d
					d.SetAllocHookForTest(func(what string, nBytes int) error {
						attachAllocs++
						if attachAllocs == 8 {
							return gc.ErrOutOfMemory
						}
						return nil
					})
					return d, nil
				})
				if gotAcc != nil || !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, gc.ErrOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("attach partial-upload OOM = (%v, %v)", gotAcc, err)
				}
				if allocsLeft, objsLeft := attachDev.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
					t.Fatalf("attach OOM leaked ledger (%d, %d)", allocsLeft, objsLeft)
				}

				// Partial attach OOM + failed rollback free -> returns nil with ErrDeviceOutOfMemory,
				// gc.ErrOutOfMemory, and the cleanup cause (no ErrBatchOutOfMemory), retains the single
				// unfinished cleanup owner + ledger + lease, refuses subsequent attaches with zero new
				// device opens while cleanup keeps failing, then drains cleanly and recovers attach.
				recAttach := &eg2RecordingLeaser{}
				origAttachLeaser := gpu.NewLeaser
				gpu.NewLeaser = func() gpu.Leaser { return recAttach }
				var stuckDev *Device
				attachFreeErr := errors.New("cuda: planted partial-attach rollback close failure")
				attachAllocs = 0
				var closeMu sync.Mutex
				closeAttempts := 0
				failRemaining := false
				gotAcc, err = newEG2AccelOn(m, combined, func() (*Device, error) {
					d, err := CreateSystemDefaultDevice()
					if err != nil {
						return nil, err
					}
					stuckDev = d
					d.SetAllocHookForTest(func(what string, nBytes int) error {
						attachAllocs++
						if attachAllocs == 8 {
							return gc.ErrOutOfMemory
						}
						return nil
					})
					d.SetCloseBufHookForTest(func(closeReal func() error) error {
						closeMu.Lock()
						closeAttempts++
						shouldFail := closeAttempts == 1 || failRemaining
						closeMu.Unlock()
						if shouldFail {
							return attachFreeErr // leave real buffer open so cleanup retry frees it
						}
						return closeReal()
					})
					return d, nil
				})
				gpu.NewLeaser = origAttachLeaser
				defer func() {
					if stuckDev != nil {
						stuckDev.SetCloseBufHookForTest(nil)
					}
					_ = drainPendingEG2Cleanups()
				}()
				if gotAcc != nil || !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) ||
					!errors.Is(err, gc.ErrOutOfMemory) || !errors.Is(err, attachFreeErr) ||
					errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("partial attach OOM + failed close = (%v, %v)", gotAcc, err)
				}
				closeMu.Lock()
				failRemaining = true
				closeMu.Unlock()
				recAttach.mu.Lock()
				liveAfterFailAttach := recAttach.devLive
				recAttach.mu.Unlock()
				if nPending := pendingEG2CleanupsLen(); nPending != 1 {
					t.Fatalf("pending attach cleanups = %d, want 1", nPending)
				}
				if allocsLeft, objsLeft := stuckDev.LedgerLen(); allocsLeft == 0 || objsLeft == 0 || liveAfterFailAttach <= 0 {
					t.Fatalf("failed attach rollback dropped ownership: ledger=(%d, %d) liveBytes=%d", allocsLeft, objsLeft, liveAfterFailAttach)
				}
				// Attempt repeated and simultaneous attaches (including public Model.NewAccelerator)
				// while cleanup keeps failing: every attempt must fail closed with attachFreeErr (no
				// ErrBatchOutOfMemory), return literal nil interface, perform ZERO new device opens,
				// and keep pending count bounded at 1 with ledger and lease intact.
				var blockedMu sync.Mutex
				blockedOpens := 0
				for attempt := range 2 {
					accBlocked, errBlocked := newEG2AccelOn(m, combined, func() (*Device, error) {
						blockedMu.Lock()
						blockedOpens++
						blockedMu.Unlock()
						return CreateSystemDefaultDevice()
					})
					if accBlocked != nil || !errors.Is(errBlocked, attachFreeErr) || errors.Is(errBlocked, embeddinggemma2.ErrBatchOutOfMemory) {
						t.Fatalf("blocked attach attempt %d = (%v, %v), want %v without retry", attempt, accBlocked, errBlocked, attachFreeErr)
					}
					if blockedOpens != 0 {
						t.Fatalf("blocked attach attempt %d opened %d new devices while cleanup failed", attempt, blockedOpens)
					}
					if nPending := pendingEG2CleanupsLen(); nPending != 1 {
						t.Fatalf("blocked attach attempt %d pending count = %d, want 1", attempt, nPending)
					}
				}
				// Simultaneous public + internal attach attempts while cleanup keeps failing.
				var simWG sync.WaitGroup
				simErrs := make([]error, 4)
				for i := range simErrs {
					simWG.Add(1)
					go func(idx int) {
						defer simWG.Done()
						if idx%2 == 0 {
							accPub, errPub := m.NewAccelerator("cuda")
							if accPub != nil {
								t.Errorf("simultaneous Model.NewAccelerator(%d) returned non-nil interface (%T)", idx, accPub)
							}
							simErrs[idx] = errPub
							return
						}
						accInt, errInt := newEG2AccelOn(m, combined, func() (*Device, error) {
							blockedMu.Lock()
							blockedOpens++
							blockedMu.Unlock()
							return CreateSystemDefaultDevice()
						})
						if accInt != nil {
							t.Errorf("simultaneous newEG2AccelOn(%d) returned non-nil accel", idx)
						}
						simErrs[idx] = errInt
					}(i)
				}
				simWG.Wait()
				for i, errSim := range simErrs {
					if !errors.Is(errSim, attachFreeErr) || errors.Is(errSim, embeddinggemma2.ErrBatchOutOfMemory) {
						t.Fatalf("simultaneous blocked attach %d = %v, want %v without retry", i, errSim, attachFreeErr)
					}
				}
				if blockedOpens != 0 {
					t.Fatalf("simultaneous blocked attaches invoked opener %d times, want 0", blockedOpens)
				}
				// Explicit-selection token validation still rejects invalid selectors without mutating pending cleanup.
				if badSel, errSel := selectedEG2Factory(combined)(m, "invalid-token!"); badSel != nil || !errors.Is(errSel, embeddinggemma2.ErrDeviceUnavailable) {
					t.Fatalf("invalid explicit selector while cleanup pending = (%v, %v), want ErrDeviceUnavailable", badSel, errSel)
				}
				// Explicit drain while close failure is still active keeps failing and preserves ownership.
				if err := drainPendingEG2Cleanups(); !errors.Is(err, attachFreeErr) {
					t.Fatalf("repeat failing attach cleanup drain = %v, want %v", err, attachFreeErr)
				}
				recAttach.mu.Lock()
				liveAfterRepeat := recAttach.devLive
				recAttach.mu.Unlock()
				if allocsLeft, objsLeft := stuckDev.LedgerLen(); allocsLeft == 0 || objsLeft == 0 || liveAfterRepeat <= 0 {
					t.Fatalf("repeat failing attach cleanup lost ownership: ledger=(%d, %d) liveBytes=%d", allocsLeft, objsLeft, liveAfterRepeat)
				}
				// Clear close hook: the next attach drains the retained cleanup to zero and succeeds cleanly.
				stuckDev.SetCloseBufHookForTest(nil)
				recoveredOpens := 0
				recovAcc, errRecov := newEG2AccelOn(m, combined, func() (*Device, error) {
					recoveredOpens++
					return CreateSystemDefaultDevice()
				})
				if errRecov != nil || recovAcc == nil {
					t.Fatalf("recovered attach after cleanup cleared = (%v, %v)", recovAcc, errRecov)
				}
				defer func() {
					if err := recovAcc.Close(); err != nil {
						t.Errorf("recovAcc.Close: %v", err)
					}
				}()
				if recoveredOpens != 1 {
					t.Fatalf("recovered attach opens = %d, want 1", recoveredOpens)
				}
				recAttach.mu.Lock()
				liveAfterDrain := recAttach.devLive
				recAttach.mu.Unlock()
				if nPending := pendingEG2CleanupsLen(); nPending != 0 {
					t.Fatalf("pending attach cleanups after recovery = %d, want 0", nPending)
				}
				if allocsLeft, objsLeft := stuckDev.LedgerLen(); allocsLeft != 0 || objsLeft != 0 || liveAfterDrain != 0 {
					t.Fatalf("attach cleanup recovery left ledger=(%d, %d) liveBytes=%d", allocsLeft, objsLeft, liveAfterDrain)
				}
				if outRecov, _, err := recovAcc.Forward(orig[0], false); err != nil || len(outRecov) == 0 {
					t.Fatalf("recovered accel Forward = (%d, %v)", len(outRecov), err)
				}

				// Unrelated attach allocation error (gc.ErrInvalidValue) -> neither OOM sentinel, 0 leaks.
				attachAllocs = 0
				gotAcc, err = newEG2AccelOn(m, combined, func() (*Device, error) {
					d, err := CreateSystemDefaultDevice()
					if err != nil {
						return nil, err
					}
					attachDev = d
					d.SetAllocHookForTest(func(what string, nBytes int) error {
						attachAllocs++
						if attachAllocs == 8 {
							return gc.ErrInvalidValue
						}
						return nil
					})
					return d, nil
				})
				if gotAcc != nil || err == nil || !errors.Is(err, gc.ErrInvalidValue) ||
					errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("unrelated attach failure = (%v, %v)", gotAcc, err)
				}
				if allocsLeft, objsLeft := attachDev.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
					t.Fatalf("unrelated attach failure leaked ledger (%d, %d)", allocsLeft, objsLeft)
				}
			})

			// (5) Non-OOM errors stay distinguishable from retry-safe OOM.
			t.Run("non-oom-distinctions", func(t *testing.T) {
				e, a := attachFresh(t)
				checkNotOOM := func(label string, err error) {
					t.Helper()
					if err == nil {
						t.Fatalf("%s: expected error", label)
					}
					if errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) {
						t.Fatalf("%s must not match OOM sentinels: %v", label, err)
					}
				}
				_, err := e.EmbedIDsBatch(nil)
				checkNotOOM("empty batch", err)
				_, err = e.EmbedIDsBatch([][]int{{1}, {}})
				checkNotOOM("empty member", err)
				_, err = e.EmbedIDsBatch([][]int{{1, 2}, {3, V}})
				checkNotOOM("invalid last ID", err)
				overMaxPos := make([]int, m.Config().MaxPositions+1)
				_, err = e.EmbedIDsBatch([][]int{overMaxPos})
				checkNotOOM("over MaxPositions", err)
				overAgg := make([][]int, a.batchRowLimit()+1)
				for i := range overAgg {
					overAgg[i] = []int{1}
				}
				_, err = e.EmbedIDsBatch(overAgg)
				checkNotOOM("over batchRowLimit", err)

				a.defect.failForward = true
				_, err = e.EmbedIDsBatch(orig[:1])
				a.defect.failForward = false
				checkNotOOM("ordinary execution failure", err)

				if err := a.Close(); err != nil {
					t.Fatalf("Close: %v", err)
				}
				_, err = e.EmbedIDsBatch(orig[:1])
				checkNotOOM("closed encoder", err)
			})
		})
	}
}

// eg2RecordingLeaser tracks live and peak charged device allocations separately
// from host allocations as an independent oracle for BatchMemory.
type eg2RecordingLeaser struct {
	mu           sync.Mutex
	devLive      int64
	devPeak      int64
	hostLive     int64
	devAllocs    int
	hostAllocs   int
	devReleases  int
	hostReleases int
}

func (l *eg2RecordingLeaser) Lease(onDevice bool, _ string, _, charged int64) func() {
	l.mu.Lock()
	if onDevice {
		l.devAllocs++
		l.devLive += charged
		if l.devLive > l.devPeak {
			l.devPeak = l.devLive
		}
	} else {
		l.hostAllocs++
		l.hostLive += charged
	}
	l.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			l.mu.Lock()
			if onDevice {
				l.devReleases++
				l.devLive -= charged
			} else {
				l.hostReleases++
				l.hostLive -= charged
			}
			l.mu.Unlock()
		})
	}
}

func (l *eg2RecordingLeaser) resetWindow() (baseLive int64, allocs int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.devPeak = l.devLive
	return l.devLive, l.devAllocs
}

func (l *eg2RecordingLeaser) windowDelta(baseLive int64, baseAllocs int) (peakDelta, endDelta int64, allocs int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.devPeak - baseLive, l.devLive - baseLive, l.devAllocs - baseAllocs
}

func syntheticEG2Accel() *eg2Accel {
	cfg := embeddinggemma2.Config{
		Hidden:       768,
		Intermediate: 3072,
		PLEDim:       256,
		Layers:       24,
		Heads:        8,
		EmbeddingDim: 768,
		MaxPositions: 2048,
	}
	layers := make([]eg2Layer, cfg.Layers)
	for i := range layers {
		if i%6 == 5 {
			layers[i].w = embeddinggemma2.LayerWeights{Full: true, HeadDim: 256, KVHeads: 2, RopeTheta: 1_000_000}
		} else {
			layers[i].w = embeddinggemma2.LayerWeights{Full: false, HeadDim: 128, KVHeads: 2, RopeTheta: 10_000}
		}
	}
	return &eg2Accel{
		cfg:                 cfg,
		layers:              layers,
		residentWeightBytes: 123 << 20,
	}
}

func TestEG2CUDA_BatchMemory_UnitContract(t *testing.T) {
	a := syntheticEG2Accel()

	// Invalid shapes return an ordinary error distinct from ErrUnsupportedBatchShape.
	for _, tc := range []struct {
		name string
		lens []int
	}{
		{"nil", nil},
		{"empty", []int{}},
		{"zero", []int{0}},
		{"negative", []int{16, -2}},
		{"overflow", []int{math.MaxInt, 1}},
	} {
		_, err := a.BatchMemory(tc.lens)
		if err == nil {
			t.Errorf("%s: got nil error, want validation error", tc.name)
		}
		if errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
			t.Errorf("%s: invalid shape must not match ErrUnsupportedBatchShape: %v", tc.name, err)
		}
	}

	// Per-sequence limit and geometry-derived batchRowLimit boundary -> ErrUnsupportedBatchShape.
	if _, err := a.BatchMemory([]int{a.cfg.MaxPositions + 1}); !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
		t.Fatalf("over MaxPositions: %v", err)
	}
	limit := a.batchRowLimit()
	atLimit := make([]int, 0, limit/a.cfg.MaxPositions+1)
	for rem := limit; rem > 0; {
		n := min(rem, a.cfg.MaxPositions)
		atLimit = append(atLimit, n)
		rem -= n
	}
	estLimit, err := a.BatchMemory(atLimit)
	if err != nil {
		t.Fatalf("batch at exact batchRowLimit (%d): %v", limit, err)
	}
	if estLimit.Quality != embeddinggemma2.BatchMemoryUpperBound || estLimit.AdditionalPeakBytes <= 0 {
		t.Fatalf("batchRowLimit estimate = %+v", estLimit)
	}
	overLimit := append(append([]int(nil), atLimit...), 1)
	if _, err := a.BatchMemory(overLimit); !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
		t.Fatalf("batch over batchRowLimit (%d+1): got %v, want ErrUnsupportedBatchShape", limit, err)
	}
	tooManyItems := make([]int, limit+1)
	for i := range tooManyItems {
		tooManyItems[i] = 1
	}
	if _, err := a.BatchMemory(tooManyItems); !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
		t.Fatalf("len(lens) > batchRowLimit: got %v, want ErrUnsupportedBatchShape", err)
	}
	// ForwardBatch also rejects len(ids) > batchRowLimit in O(1) before allocating lens.
	if _, err := a.ForwardBatch(make([][]int, limit+1)); !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
		t.Fatalf("ForwardBatch len(ids) > batchRowLimit: got %v, want ErrUnsupportedBatchShape", err)
	}

	// Singleton BatchMemory prices EmbedIDs (Forward) and uses singleton
	// checkPositions admission rather than ForwardBatch's aggregate batchRowLimit.
	wide := &eg2Accel{
		cfg: embeddinggemma2.Config{
			Hidden:       768,
			Intermediate: 262144, // maximal stride -> batchRowLimit = (MaxInt32-255)/262144 - 31 = 8160
			PLEDim:       256,
			Layers:       1,
			Heads:        8,
			EmbeddingDim: 768,
			MaxPositions: 8192,
		},
		layers: []eg2Layer{{w: embeddinggemma2.LayerWeights{HeadDim: 128, KVHeads: 2, RopeTheta: 10_000}}},
	}
	if got := wide.batchRowLimit(); got != 8160 {
		t.Fatalf("wide.batchRowLimit() = %d, want 8160", got)
	}
	estSingleWide, err := wide.BatchMemory([]int{8161})
	if err != nil {
		t.Fatalf("singleton length 8161 (<= MaxPositions=8192, > batchRowLimit=8160) rejected: %v", err)
	}
	if estSingleWide.Quality != embeddinggemma2.BatchMemoryUpperBound || estSingleWide.AdditionalPeakBytes <= 0 {
		t.Fatalf("singleton wide estimate = %+v", estSingleWide)
	}
	if _, err := wide.BatchMemory([]int{8160, 1}); !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
		t.Fatalf("multi-input [8160, 1] over batchRowLimit=8160: got %v, want ErrUnsupportedBatchShape", err)
	}
	if _, err := wide.BatchMemory([]int{8193}); !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
		t.Fatalf("singleton 8193 over MaxPositions=8192: got %v, want ErrUnsupportedBatchShape", err)
	}

	// Same aggregate T=128 with different max member lengths: [64,64] vs [1,127].
	cold64, err := a.BatchMemory([]int{64, 64})
	if err != nil {
		t.Fatal(err)
	}
	cold127, err := a.BatchMemory([]int{1, 127})
	if err != nil {
		t.Fatal(err)
	}
	if cold127.AdditionalPeakBytes <= cold64.AdditionalPeakBytes {
		t.Fatalf("cold [1,127] (%d) should exceed cold [64,64] (%d) due to maxT RoPE cache growth", cold127.AdditionalPeakBytes, cold64.AdditionalPeakBytes)
	}
	if cold64.ResidentWeightBytes != a.residentWeightBytes || cold127.ResidentWeightBytes != a.residentWeightBytes {
		t.Fatalf("resident weight bytes mismatch: %+v / %+v", cold64, cold127)
	}

	// Warm cache at rows=128 covers both [64,64] and [1,127] identically (same aggregate T=128).
	a.ropes = []eg2Rope{
		{hd: 128, theta: 10_000, rows: 128},
		{hd: 256, theta: 1_000_000, rows: 128},
	}
	warm64, err := a.BatchMemory([]int{64, 64})
	if err != nil {
		t.Fatal(err)
	}
	warm127, err := a.BatchMemory([]int{1, 127})
	if err != nil {
		t.Fatal(err)
	}
	if warm64.AdditionalPeakBytes != warm127.AdditionalPeakBytes {
		t.Fatalf("warm [64,64] (%d) != warm [1,127] (%d) with same aggregate T=128", warm64.AdditionalPeakBytes, warm127.AdditionalPeakBytes)
	}
	if warm64.AdditionalPeakBytes >= cold64.AdditionalPeakBytes {
		t.Fatalf("warm [64,64] (%d) should be strictly less than cold [64,64] (%d)", warm64.AdditionalPeakBytes, cold64.AdditionalPeakBytes)
	}

	// Byte-product overflow guard before rounding or int conversion.
	huge := &eg2Accel{
		cfg: embeddinggemma2.Config{
			Hidden:       math.MaxInt / 2,
			Intermediate: 16,
			PLEDim:       16,
			Layers:       1,
			Heads:        1,
			EmbeddingDim: 16,
			MaxPositions: 16,
		},
		layers: []eg2Layer{{w: embeddinggemma2.LayerWeights{HeadDim: 16, KVHeads: 1, RopeTheta: 10_000}}},
	}
	if _, err := huge.scratchPlan(4); err == nil || errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
		t.Fatalf("scratchPlan overflow: got %v, want ordinary validation error", err)
	}

	// Broken and closed accelerators preserve their original errors.
	brokenErr := errors.New("cuda: synthetic broken queue")
	a.broken = brokenErr
	if _, err := a.BatchMemory([]int{16}); !errors.Is(err, brokenErr) {
		t.Fatalf("broken accel: got %v, want %v", err, brokenErr)
	}
	a.broken = nil
	a.closed = true
	if _, err := a.BatchMemory([]int{16}); !errors.Is(err, errEG2Closed) {
		t.Fatalf("closed accel: got %v, want %v", err, errEG2Closed)
	}
}

func TestEG2CUDA_BatchMemory_InstrumentedOracle(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	dev.ReleaseObjects()

	m := eg2TinyModel(t)
	V := m.Config().VocabSize

	for _, combined := range []bool{false, true} {
		name := "packed"
		if combined {
			name = "combined"
		}
		t.Run(name, func(t *testing.T) {
			rec := &eg2RecordingLeaser{}
			origNewLeaser := gpu.NewLeaser
			gpu.NewLeaser = func() gpu.Leaser { return rec }
			a, err := newEG2Accel(m, combined)
			gpu.NewLeaser = origNewLeaser
			if err != nil {
				t.Fatalf("newEG2Accel(combined=%v): %v", combined, err)
			}
			defer a.Close()

			// Distinguish persistent weights from synthetic helpers (ones, zeroBias) and host allocations.
			est0, err := a.BatchMemory([]int{16})
			if err != nil {
				t.Fatalf("BatchMemory: %v", err)
			}
			if est0.Quality != embeddinggemma2.BatchMemoryUpperBound {
				t.Fatalf("Quality = %v, want upper_bound", est0.Quality)
			}
			maxHD := 0
			c := m.Config()
			maxN := max(c.Layers*c.PLEDim, c.Intermediate, c.Hidden, c.EmbeddingDim)
			for _, ly := range a.layers {
				maxHD = max(maxHD, ly.w.HeadDim)
				maxN = max(maxN, c.Heads*ly.w.HeadDim)
			}
			helperBytes := gpu.ChargedBytes(int64(maxHD)*4) + gpu.ChargedBytes(int64(max(maxN, 1))*4)
			rec.mu.Lock()
			attachDevLive := rec.devLive
			attachHostAllocs := rec.hostAllocs
			rec.mu.Unlock()
			if est0.ResidentWeightBytes != attachDevLive-helperBytes {
				t.Fatalf("ResidentWeightBytes=%d, want attachDevLive(%d)-helpers(%d)=%d (hostAllocs=%d)",
					est0.ResidentWeightBytes, attachDevLive, helperBytes, attachDevLive-helperBytes, attachHostAllocs)
			}

			// Verify BatchMemory performs zero allocations and zero kernel launches/waits.
			baseLive, baseAllocs := rec.resetWindow()
			l0, w0 := a.ops.dev.Activity()
			if _, err := a.BatchMemory([]int{64, 64}); err != nil {
				t.Fatal(err)
			}
			l1, w1 := a.ops.dev.Activity()
			peakD, endD, newAllocs := rec.windowDelta(baseLive, baseAllocs)
			if newAllocs != 0 || peakD != 0 || endD != 0 || l1 != l0 || w1 != w0 {
				t.Fatalf("BatchMemory mutated device: allocs=%d peak=%d end=%d launches=%d waits=%d",
					newAllocs, peakD, endD, l1-l0, w1-w0)
			}

			// Exercise cold, warm-reuse, and warm-to-long growth across singleton, [64,64], [1,127], and long batches.
			cases := []struct {
				label string
				lens  []int
			}{
				{"singleton-cold-32", []int{32}},
				{"singleton-warm-16", []int{16}},
				{"batch-64-64-grow", []int{64, 64}},
				{"batch-64-64-warm", []int{64, 64}},
				{"batch-1-127-grow", []int{1, 127}},
				{"batch-1-127-warm", []int{1, 127}},
				{"batch-long-514-514-grow", []int{514, 514}},
				{"batch-long-514-514-warm", []int{514, 514}},
			}
			for _, tc := range cases {
				maxT := 0
				for _, n := range tc.lens {
					maxT = max(maxT, n)
				}
				var releasedOldRope int64
				for _, k := range a.distinctRopeKeys() {
					if oldRows := a.ropeCachedRows(k); oldRows > 0 && oldRows < maxT {
						releasedOldRope += 2 * gpu.ChargedBytes(int64(oldRows*k.hd)*4)
					}
				}

				est, err := a.BatchMemory(tc.lens)
				if err != nil {
					t.Fatalf("%s: BatchMemory: %v", tc.label, err)
				}
				baseLive, baseAllocs := rec.resetWindow()
				var freeBefore, freeAfter uint64
				_ = a.ops.do(func() error {
					freeBefore, _, _ = a.ops.dev.Context().MemInfo()
					return nil
				})
				if len(tc.lens) == 1 {
					if _, _, err := a.Forward(eg2SeqIDs(tc.lens[0], 1, V), false); err != nil {
						t.Fatalf("%s: Forward: %v", tc.label, err)
					}
				} else {
					batch := make([][]int, len(tc.lens))
					for i, n := range tc.lens {
						batch[i] = eg2SeqIDs(n, i+1, V)
					}
					if _, err := a.ForwardBatch(batch); err != nil {
						t.Fatalf("%s: ForwardBatch: %v", tc.label, err)
					}
				}
				_ = a.ops.do(func() error {
					freeAfter, _, _ = a.ops.dev.Context().MemInfo()
					return nil
				})
				peakDelta, endDelta, _ := rec.windowDelta(baseLive, baseAllocs)
				if peakDelta > est.AdditionalPeakBytes {
					t.Errorf("%s: measured ledger peak delta %d > estimated AdditionalPeakBytes %d", tc.label, peakDelta, est.AdditionalPeakBytes)
				}
				if gotMargin := est.AdditionalPeakBytes - peakDelta; gotMargin != releasedOldRope {
					t.Errorf("%s: margin (est %d - peak %d) = %d, want releasedOldRope %d", tc.label, est.AdditionalPeakBytes, peakDelta, gotMargin, releasedOldRope)
				}
				t.Logf("[%s/%s] est=%d ledger_peak=%d margin=%d end_delta=%d phys_free_delta=%d",
					name, tc.label, est.AdditionalPeakBytes, peakDelta, est.AdditionalPeakBytes-peakDelta, endDelta, int64(freeBefore)-int64(freeAfter))
			}
		})
	}
}

func TestCUDAInventoryRows(t *testing.T) {
	adapters := []gpu.Adapter{
		{Ordinal: 0, UUID: "Old", Major: 7},
		{Ordinal: 1, UUID: "First", Name: "GPU", Major: 7, Minor: 5, TotalBytes: 2048, IntegratedKnown: true},
		{Ordinal: 2, Major: 8, TotalBytes: 1024},
		{Ordinal: 3, UUID: "Third", Name: "GPU", Major: 12, TotalBytes: 4096, Integrated: true, IntegratedKnown: true},
		{Ordinal: 4, UUID: "Duplicate", Major: 8},
		{Ordinal: 5, UUID: "duplicate", Major: 8},
	}
	var sampled []int
	failure := errors.New("memory unavailable")
	read := func(a gpu.Adapter) (embeddinggemma2.DeviceMemory, error) {
		sampled = append(sampled, a.Ordinal)
		if a.Ordinal == 3 {
			return embeddinggemma2.DeviceMemory{}, failure
		}
		return embeddinggemma2.DeviceMemory{TotalBytes: int64(a.TotalBytes), AvailableBytes: 512, Basis: embeddinggemma2.MemoryLive}, nil
	}
	cheap, err := describeCUDADevices(adapters, nil, embeddinggemma2.ProbeOptions{}, read)
	if err != nil || len(sampled) != 0 || len(cheap) != 5 || cheap[0].Alias != "cuda:1" || cheap[2].Alias != "cuda:3" ||
		cheap[0].ID != "cuda:First" || cheap[0].Backend != "cuda" || cheap[0].MemoryKind != embeddinggemma2.MemoryDedicated ||
		cheap[0].VendorID != 0x10de || !cheap[0].TextBatching || cheap[0].Memory.TotalBytes != 2048 ||
		cheap[0].Memory.Basis != embeddinggemma2.MemoryUnknown {
		t.Fatalf("cheap inventory %+v, queries %v, error %v", cheap, sampled, err)
	}
	for _, i := range []int{1, 3, 4} {
		if cheap[i].ID != "" || cheap[i].Alias != "" {
			t.Fatalf("unverified identifier at %d: %+v", i, cheap[i])
		}
	}
	if cheap[1].MemoryKind != embeddinggemma2.MemoryKindUnknown || cheap[2].MemoryKind != embeddinggemma2.MemoryShared {
		t.Fatalf("memory architecture: %+v", cheap)
	}
	live, err := describeCUDADevices(adapters, nil, embeddinggemma2.ProbeOptions{IncludeMemory: true}, read)
	if !errors.Is(err, failure) || !slices.Equal(sampled, []int{1, 3}) || len(live) != len(cheap) ||
		live[0].Memory.Basis != embeddinggemma2.MemoryLive || live[0].Memory.AvailableBytes != 512 || live[2] != cheap[2] {
		t.Fatalf("live inventory %+v, queries %v, error %v", live, sampled, err)
	}
}

func linkedEG2CUDASelectors(t *testing.T) []string {
	t.Helper()
	want := []string{"cuda"}
	if eg2TaggedBinary(t, "eg2_combined") {
		want = append(want, "cuda-eg2-combined")
		if eg2TaggedBinary(t, "eg2_fp16") {
			want = append(want, eg2FP16TestSelector)
		}
	}
	if linked := embeddinggemma2.DeviceSelectors("cuda"); !slices.Equal(linked, want) {
		t.Fatalf("linked CUDA selectors = %v, want %v for build tags", linked, want)
	}
	return want
}

func TestCUDAInventoryRunnableSelectors(t *testing.T) {
	linked := linkedEG2CUDASelectors(t)
	base := linked
	if slices.Contains(linked, eg2FP16TestSelector) {
		base = base[:len(base)-1]
	}
	adapters := []gpu.Adapter{
		{Ordinal: 0, UUID: "GPU-abcd1234-1111-2222-3333-444444444440", Major: 7},
		{Ordinal: 1, UUID: "GPU-abcd1234-1111-2222-3333-444444444441", Major: 7, Minor: 5},
		{Ordinal: 2, UUID: "GPU-abcd1234-1111-2222-3333-444444444442", Major: 7, Minor: 9},
		{Ordinal: 3, UUID: "GPU-abcd1234-1111-2222-3333-444444444443", Major: 8},
		{Ordinal: 4, UUID: "MIG-abcd1234-1111-2222-3333-444444444444", Major: 12},
		{Ordinal: 5, Major: 12},
		{Ordinal: 6, UUID: "GPU-abcd1234-1111-2222-3333-444444444446", Major: 8},
		{Ordinal: 7, UUID: "gpu-ABCD1234-1111-2222-3333-444444444446", Major: 8},
	}
	rows, err := describeCUDADevices(adapters, nil, embeddinggemma2.ProbeOptions{}, func(gpu.Adapter) (embeddinggemma2.DeviceMemory, error) {
		t.Fatal("cheap CUDA probe acquired a memory context")
		return embeddinggemma2.DeviceMemory{}, nil
	})
	if err != nil || len(rows) != len(adapters)-1 {
		t.Fatalf("inventory has %d rows, err %v", len(rows), err)
	}
	wantSelectors := [][]string{base, base, linked, linked, nil, nil, nil}
	for i, row := range rows {
		adapter := adapters[i+1]
		want := embeddinggemma2.CUDAComputeCapability{Major: adapter.Major, Minor: adapter.Minor}
		if row.ComputeCapability != want || !slices.Equal(row.RunnableSelectors(), wantSelectors[i]) {
			t.Errorf("ordinal %d: capability %+v and selectors %v, want %+v and %v",
				adapter.Ordinal, row.ComputeCapability, row.RunnableSelectors(), want, wantSelectors[i])
		}
		if i < 4 {
			if row.ID != "cuda:"+adapter.UUID || row.Alias != "cuda:"+strconv.Itoa(adapter.Ordinal) {
				t.Errorf("ordinal %d lost identity: %+v", adapter.Ordinal, row)
			}
		} else if row.ID != "" || row.Alias != "" || row.RunnableSelectors() != nil {
			t.Errorf("ordinal %d advertised unverified identity: %+v", adapter.Ordinal, row)
		}
	}
	unknown := rows[2]
	unknown.ComputeCapability = embeddinggemma2.CUDAComputeCapability{}
	if got := unknown.RunnableSelectors(); got != nil {
		t.Errorf("unknown capability advertised %v", got)
	}
	for _, id := range []string{"cuda:0", "cuda:GPU-invalid"} {
		invalid := rows[2]
		invalid.ID = id
		if got := invalid.RunnableSelectors(); got != nil {
			t.Errorf("invalid persistent ID %q advertised %v", id, got)
		}
	}
}

func TestCUDAInventoryPartialIdentity(t *testing.T) {
	adapter := gpu.Adapter{Ordinal: 1, UUID: "GPU-abcd1234-1111-2222-3333-444444444444",
		Name: "healthy", Major: 8, TotalBytes: 2048}
	capability := embeddinggemma2.CUDAComputeCapability{Major: 8}
	enumerateErr := fmt.Errorf("ordinal 0: %w: %w", gpu.ErrAdapterChanged, gc.ErrInvalidDevice)
	opts := embeddinggemma2.ProbeOptions{IncludeMemory: true}
	adapters := []gpu.Adapter{adapter}
	for _, includeMemory := range []bool{false, true} {
		rows, err := describeCUDADevices(adapters, enumerateErr, embeddinggemma2.ProbeOptions{IncludeMemory: includeMemory},
			func(gpu.Adapter) (embeddinggemma2.DeviceMemory, error) {
				t.Error("partial inventory tried to open an unverified UUID")
				return embeddinggemma2.DeviceMemory{}, nil
			})
		if len(rows) != 1 || !errors.Is(err, embeddinggemma2.ErrNoDevice) || !errors.Is(err, gc.ErrInvalidDevice) {
			t.Fatalf("partial inventory: rows %+v, err %v", rows, err)
		}
		row := rows[0]
		if row.Name != adapter.Name || row.ComputeCapability != capability || row.Memory.TotalBytes != 2048 ||
			row.ID != "" || row.Alias != "" || row.RunnableSelectors() != nil {
			t.Errorf("partial inventory advertised an unverified identity or hid healthy device metadata: %+v", row)
		}
	}

	memoryErr := errors.New("optional memory read failed")
	reads := 0
	rows, err := describeCUDADevices(adapters, nil, opts, func(got gpu.Adapter) (embeddinggemma2.DeviceMemory, error) {
		reads++
		if got != adapter {
			t.Errorf("memory queried wrong adapter: %+v", got)
		}
		return embeddinggemma2.DeviceMemory{}, memoryErr
	})
	if len(rows) != 1 || !errors.Is(err, memoryErr) || reads != 1 {
		t.Fatalf("optional memory failure: rows %+v, err %v, reads %d", rows, err, reads)
	}
	row := rows[0]
	if row.ID != "cuda:"+adapter.UUID || row.Alias != "cuda:1" || row.ComputeCapability != capability ||
		row.Memory.Basis != embeddinggemma2.MemoryUnknown || row.Memory.TotalBytes != 2048 ||
		!slices.Equal(row.RunnableSelectors(), linkedEG2CUDASelectors(t)) {
		t.Errorf("optional memory failure hid verified identity or capability: %+v", row)
	}
}

func TestCUDAEG2FP16RetainedCapability(t *testing.T) {
	for _, tt := range []struct {
		name         string
		major, minor int
		query, want  error
	}{
		{name: "below-default-floor", major: 7, want: embeddinggemma2.ErrNoDevice},
		{name: "SM75", major: 7, minor: 5, want: embeddinggemma2.ErrUnsupportedSelector},
		{name: "SM79", major: 7, minor: 9, want: embeddinggemma2.ErrUnsupportedSelector},
		{name: "SM80", major: 8},
		{name: "SM120", major: 12},
		{name: "native-read-failure", major: 7, minor: 5, query: gc.ErrInvalidDevice, want: gc.ErrInvalidDevice},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cc := embeddinggemma2.CUDAComputeCapability{Major: tt.major, Minor: tt.minor}
			if tt.want == nil {
				if err := requireEG2FP16Capability(cc, tt.query); err != nil {
					t.Fatal(err)
				}
				return
			}
			// A rejection must precede any JIT work on the device.
			pipeline, err := eg2FP16Pipeline(nil, nil, cc, tt.query)
			if pipeline != (Pipeline{}) || !errors.Is(err, tt.want) {
				t.Fatalf("capability %+v: pipeline %v, error %v; want %v", cc, pipeline, err, tt.want)
			}
			if errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) != (tt.want == embeddinggemma2.ErrUnsupportedSelector) ||
				errors.Is(err, embeddinggemma2.ErrNoDevice) != (tt.want == embeddinggemma2.ErrNoDevice) ||
				errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) != (tt.query == nil) {
				t.Fatalf("capability %+v misclassified: %v", cc, err)
			}
			if tt.want == embeddinggemma2.ErrUnsupportedSelector &&
				(!strings.Contains(err.Error(), eg2FP16TestSelector) || !strings.Contains(err.Error(), "8.0") ||
					!strings.Contains(err.Error(), fmt.Sprintf("%d.%d", tt.major, tt.minor))) {
				t.Fatalf("missing selector and observed/required capability: %v", err)
			}
		})
	}
}

type fakeCUDAMemoryContext struct {
	closeErr      error
	reads, closes int
}

func (c *fakeCUDAMemoryContext) MemInfo() (uint64, uint64, error) {
	c.reads++
	return 1024, 2048, nil
}

func (c *fakeCUDAMemoryContext) Close() error {
	c.closes++
	return c.closeErr
}

func TestCUDAProbeRetainsFailedCloseOwnership(t *testing.T) {
	first, second := gpu.Adapter{Ordinal: 0, UUID: "First", Major: 8, TotalBytes: 2048}, gpu.Adapter{Ordinal: 1, UUID: "Second", Major: 8, TotalBytes: 2048}
	wantClose := errors.New("native release failed")
	failed := &fakeCUDAMemoryContext{closeErr: wantClose}
	var probe cudaMemoryProbe
	var firstOpens, secondOpens int
	var fresh *fakeCUDAMemoryContext
	open := func(a gpu.Adapter) (cudaMemoryContext, error) {
		if a == first {
			firstOpens++
			if firstOpens == 1 {
				return failed, nil
			}
			fresh = &fakeCUDAMemoryContext{}
			return fresh, nil
		}
		secondOpens++
		return &fakeCUDAMemoryContext{}, nil
	}
	list := func() ([]embeddinggemma2.DeviceInfo, error) {
		return describeCUDADevices([]gpu.Adapter{first, second}, nil, embeddinggemma2.ProbeOptions{IncludeMemory: true},
			func(a gpu.Adapter) (embeddinggemma2.DeviceMemory, error) { return probe.read(a, open) })
	}
	for attempt := 1; attempt <= 2; attempt++ {
		rows, err := list()
		if !errors.Is(err, wantClose) || len(rows) != 2 || rows[0].ID != "cuda:First" ||
			rows[0].Memory.Basis != embeddinggemma2.MemoryUnknown || rows[1].Memory.Basis != embeddinggemma2.MemoryLive ||
			firstOpens != 1 || secondOpens != attempt || failed.reads != 1 || failed.closes != attempt {
			t.Fatalf("attempt %d: rows %+v/error %v; first/second opens %d/%d, failed context %+v",
				attempt, rows, err, firstOpens, secondOpens, failed)
		}
	}
	failed.closeErr = nil
	rows, err := list()
	if err != nil || len(rows) != 2 || rows[0].Memory.Basis != embeddinggemma2.MemoryLive ||
		firstOpens != 2 || failed.closes != 3 || fresh == nil || fresh.closes != 1 || len(probe.pending) != 0 {
		t.Fatalf("recovery: rows %+v/error %v; first opens %d, old %+v, new %+v, retained %d",
			rows, err, firstOpens, failed, fresh, len(probe.pending))
	}
}

const eg2FP16TestSelector = "cuda-eg2-fp16"

func eg2TaggedBinary(t *testing.T, required ...string) bool {
	t.Helper()
	info, ok := debug.ReadBuildInfo()
	if !ok {
		t.Fatal("test binary has no Go build information")
	}
	for _, setting := range info.Settings {
		if setting.Key == "-tags" {
			tags := strings.Fields(strings.ReplaceAll(setting.Value, ",", " "))
			for _, tag := range required {
				if !slices.Contains(tags, tag) {
					return false
				}
			}
			return true
		}
	}
	return false
}

func TestEG2FP16Geometry(t *testing.T) {
	tagsLinked := eg2TaggedBinary(t, "eg2_combined", "eg2_fp16")
	selectors := embeddinggemma2.DeviceSelectors("cuda")
	registered := slices.Contains(selectors, eg2FP16TestSelector)
	if registered != tagsLinked {
		t.Fatalf("FP16 selector registration = %t for combined and FP16 build tags = %t: %v", registered, tagsLinked, selectors)
	}
	t.Logf("combined and FP16 build tags = %t, selector registered = %t", tagsLinked, registered)
	for _, shape := range [][2]int{{64, 96}, {512, 2048}, {16, 16}} {
		if !eg2FFNGeometry(shape[0], shape[1]) {
			t.Errorf("refused %v", shape)
		}
	}
	for _, shape := range [][2]int{{0, 16}, {16, -16}, {32, 15}, {1 << 20, 1 << 20}} {
		if eg2FFNGeometry(shape[0], shape[1]) {
			t.Errorf("admitted %v", shape)
		}
	}
	acc, err := selectedEG2FactoryOn(eg2CUDAAttach(true, nil))(nil, "invalid")
	if acc != nil || !errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) {
		t.Fatalf("invalid explicit device = %v, %v", acc, err)
	}
	// Verify the synthetic 256/512 safetensors fixture builds and runs on CPU without a GPU.
	m256 := eg2Synthetic256_512Model(t)
	w256 := m256.TransformerWeights()
	if len(w256.Layers) != 2 || w256.Layers[0].HeadDim != 256 || w256.Layers[0].KVHeads != 2 ||
		w256.Layers[1].HeadDim != 512 || w256.Layers[1].KVHeads != 1 {
		t.Fatalf("eg2Synthetic256_512Model layers = %+v", w256.Layers)
	}
	out256, _, err := m256.ForwardCPU([]int{0, 1, 2}, false)
	if err != nil || len(out256) != 3*m256.Dim() || !eg2Finite(out256) {
		t.Fatalf("eg2Synthetic256_512Model ForwardCPU = (%d, %v)", len(out256), err)
	}
}

func TestEG2FP16Weights(t *testing.T) {
	values := []float32{65504, 65519, -65504, -65519, 0, math.Float32frombits(1 << 31), 0x1p-24, 0x1p-25, 1 + 0x1p-11}
	want := []uint16{0x7bff, 0x7bff, 0xfbff, 0xfbff, 0, 0x8000, 1, 0, 0x3c00}
	if got, err := eg2FP16Weights(values); err != nil || !slices.Equal(got, want) {
		t.Fatalf("weights = %#x, %v, want %#x", got, err, want)
	}
	for _, bad := range []float32{65520, -65520, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if packed, err := eg2FP16Weights([]float32{1, bad}); packed != nil || err == nil {
			t.Errorf("invalid weight %g returned %v, %v", bad, packed, err)
		}
	}
}

func eg2FP16TestAdapter(t *testing.T) gpu.Adapter {
	t.Helper()
	if !eg2TaggedBinary(t, "eg2_combined", "eg2_fp16") {
		t.Skip("build with cuda,eg2_combined,eg2_fp16 to run FP16 encoder tests")
	}
	if !slices.Contains(embeddinggemma2.DeviceSelectors("cuda"), eg2FP16TestSelector) {
		t.Fatalf("tagged FP16 build is missing the %q device selector", eg2FP16TestSelector)
	}
	id := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if id == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID to run FP16 FFN device tests")
	}
	devices, err := gpu.Adapters()
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := resolveCUDADevice(id, devices)
	if err != nil {
		t.Fatal(err)
	}
	if adapter.Major < 8 {
		t.Fatalf("FP16 FFN device test requires SM80 or newer, got %d.%d", adapter.Major, adapter.Minor)
	}
	return adapter
}

func TestEG2FP16Encoder(t *testing.T) {
	device := eg2FP16TestAdapter(t)
	m := eg2TinyModel(t)
	rec := &eg2RecordingLeaser{}
	origHook := leaseSeenForTest
	t.Cleanup(func() { leaseSeenForTest = origHook })
	leaseSeenForTest = func(onDevice bool, line string, requested int64, release func()) func() {
		if origHook != nil {
			release = origHook(onDevice, line, requested, release)
		}
		recordRelease := rec.Lease(onDevice, line, requested, gpu.ChargedBytes(requested))
		return func() {
			release()
			recordRelease()
		}
	}
	acc, err := m.NewAccelerator(eg2FP16TestSelector + ":" + device.UUID)
	if err != nil {
		t.Fatal(err)
	}
	a := acc.(*eg2Accel)
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Error(err)
		}
	})
	if a.ffnFP16 == (Pipeline{}) || a.AttachedDevice().ID != "cuda:"+device.UUID {
		t.Fatalf("FP16 pipeline/device not selected: %s", a.AttachedDevice().ID)
	}
	maxHD, maxN := 0, max(a.cfg.Layers*a.cfg.PLEDim, a.cfg.Intermediate, a.cfg.Hidden, a.cfg.EmbeddingDim)
	for _, ly := range a.layers {
		maxHD = max(maxHD, ly.w.HeadDim)
		maxN = max(maxN, a.cfg.Heads*ly.w.HeadDim)
	}
	est, err := a.BatchMemory([]int{17, 31})
	if err != nil {
		t.Fatal(err)
	}
	if want := rec.devLive - gpu.ChargedBytes(int64(maxHD)*4) - gpu.ChargedBytes(int64(maxN)*4); est.ResidentWeightBytes != want {
		t.Fatalf("resident weight estimate %d, want %d charged bytes", est.ResidentWeightBytes, want)
	}
	cfg := a.cfg
	cfg.EmbeddingDim = 64
	plain := &eg2Accel{cfg: cfg, layers: a.layers}
	withStatus := &eg2Accel{cfg: cfg, layers: a.layers, ffnFP16: a.ffnFP16}
	refPlan, err := plain.scratchPlan(8192)
	if err != nil {
		t.Fatal(err)
	}
	fp16Plan, err := withStatus.scratchPlan(8192)
	if err != nil {
		t.Fatal(err)
	}
	if fp16Plan.res != refPlan.res+1 || fp16Plan.chargedBytes-refPlan.chargedBytes != gpu.AllocQuantumBytes {
		t.Fatalf("status crosses a charge boundary: elements %d to %d, charge %d to %d",
			refPlan.res, fp16Plan.res, refPlan.chargedBytes, fp16Plan.chargedBytes)
	}
	old := gpu.TraceForTest
	t.Cleanup(func() { gpu.TraceForTest = old })
	launches := 0
	gpu.TraceForTest = func(event gpu.TraceEvent) {
		if event.Op == gpu.TraceLaunch && event.Pipe == a.ffnFP16.Ptr() {
			launches++
		}
	}
	inputs := make([][]int, 0, 8)
	for _, n := range []int{1, 15, 16, 17, 31, 32, 33, 65} {
		inputs = append(inputs, eg2SeqIDs(n, n+1, m.Config().VocabSize))
	}
	singles := make([][]float32, len(inputs))
	minCosine := 1.0
	for i, ids := range inputs {
		h, _, err := a.Forward(ids, false)
		if err != nil {
			t.Fatal(err)
		}
		cp, err := m.Embed(ids)
		if err != nil {
			t.Fatal(err)
		}
		v := eg2PoolNorm(h, len(ids), m.Dim(), m.Dim())
		if c := eg2Cos(v, cp); !eg2Finite(v) || !(c >= .999) {
			t.Fatalf("input %d length %d CPU cosine %g", i, len(ids), c)
		} else {
			minCosine = min(minCosine, c)
		}
		singles[i] = h
	}
	for _, order := range [][]int{{0, 1, 2, 3, 4, 5, 6, 7}, {7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3}, {4, 5, 6, 7}, {7, 2}, {0, 2}} {
		group := make([][]int, len(order))
		for i, pos := range order {
			group[i] = inputs[pos]
		}
		before := launches
		out, err := a.ForwardBatch(group)
		if err != nil {
			t.Fatal(err)
		}
		if launches-before != 3*m.Config().Layers {
			t.Fatalf("order %v: FP16 launches %d", order, launches-before)
		}
		for i, pos := range order {
			if d := eg2FirstDiff(out[i], singles[pos], m.Dim()); d != "" {
				t.Fatalf("order %v input %d: %s", order, pos, d)
			}
		}
	}
	t.Logf("raw rows match singleton bits; minimum pooled CPU cosine %.12f", minCosine)

	// A nonfinite value in the later prepared member invalidates the entire packed forward.
	x, err := m.EmbedTokens(append(slices.Clone(inputs[0]), inputs[1]...))
	if err != nil {
		t.Fatal(err)
	}
	x[len(x)-1] = float32(math.Inf(1))
	h, layers, err := a.forwardRows(x, []int{len(inputs[0]), len(inputs[1])}, true)
	if h != nil || layers != nil || !errors.Is(err, errEG2FP16InputRange) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("invalid later input returned %d rows, %d layers, %v", len(h), len(layers), err)
	}
	base, allocs := rec.resetWindow()
	est, err = a.BatchMemory([]int{len(inputs[0]), len(inputs[1])})
	if err != nil {
		t.Fatal(err)
	}
	out, err := a.ForwardBatch(inputs[:2])
	if err != nil {
		t.Fatal(err)
	}
	peak, end, _ := rec.windowDelta(base, allocs)
	if peak > est.AdditionalPeakBytes || end != 0 {
		t.Fatalf("recovery scratch peak=%d, estimate=%d, end=%d", peak, est.AdditionalPeakBytes, end)
	}
	for i := range out {
		if d := eg2FirstDiff(out[i], singles[i], m.Dim()); d != "" {
			t.Fatalf("after range failure input %d: %s", i, d)
		}
	}

	oom := gc.ErrOutOfMemory
	a.ops.dev.SetAllocHookForTest(func(string, int) error { return oom })
	out, err = a.ForwardBatch(inputs[:2])
	a.ops.dev.SetAllocHookForTest(nil)
	if out != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || !errors.Is(err, oom) {
		t.Fatalf("forward allocation failure returned %d outputs, %v", len(out), err)
	}
	recovered, err := a.ForwardBatch(inputs[:2])
	if err != nil {
		t.Fatal(err)
	}
	for i := range recovered {
		if d := eg2FirstDiff(recovered[i], singles[i], m.Dim()); d != "" {
			t.Fatalf("after allocation retry input %d: %s", i, d)
		}
	}

	fatal := &cudaresult.Error{Code: cudasys.CUDA_ERROR_ASSERT, Op: "injected FFN execution failure"}
	a.defect.failForward, a.defect.failForwardErr = true, fatal
	beforeFatal := launches
	out, err = a.ForwardBatch(inputs[:2])
	if out != nil || !errors.Is(err, fatal) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || launches <= beforeFatal {
		t.Fatalf("post-FFN execution failure returned %d outputs, %v, launched %d FP16 kernels", len(out), err, launches-beforeFatal)
	}
	beforeFatal = launches
	out, err = a.ForwardBatch(inputs[:2])
	if out != nil || !errors.Is(err, fatal) || launches != beforeFatal {
		t.Fatalf("poisoned forward returned %d outputs, %v, launched %d FP16 kernels", len(out), err, launches-beforeFatal)
	}
}

func TestEG2FP16FailedAttach(t *testing.T) {
	adapter := eg2FP16TestAdapter(t)
	m := eg2TinyModel(t)
	ptx, err := os.ReadFile("testdata/eg2_ffn_fp16.ptx")
	if err != nil {
		t.Fatal(err)
	}
	bytes := m.Config().Hidden * m.Config().Intermediate * 2
	var dev *Device
	var count int
	acc, err := newEG2AccelWithFFN(m, true, ptx, func() (*Device, error) {
		d, err := gpu.CreateSelectedDevice(adapter)
		if err != nil {
			return nil, err
		}
		dev = d
		d.SetAllocHookForTest(func(_ string, n int) error {
			if n == bytes {
				count++
				if count == 2 {
					return gc.ErrOutOfMemory
				}
			}
			return nil
		})
		return d, nil
	})
	if acc != nil || !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) ||
		!errors.Is(err, gc.ErrOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("failed attach = %v, %v", acc, err)
	}
	if dev == nil || count != 2 {
		t.Fatalf("device %v allocated %d FP16-size matrices before failure", dev != nil, count)
	}
	if allocs, objects := dev.LedgerLen(); allocs != 0 || objects != 0 {
		t.Fatalf("failed attach retains %d allocations and %d objects", allocs, objects)
	}
	m.Weights().Layers[0].Up[0] = 65520
	acc, err = newEG2AccelWithFFN(m, true, ptx, func() (*Device, error) {
		d, err := gpu.CreateSelectedDevice(adapter)
		dev = d
		return d, err
	})
	if acc != nil || err == nil || !strings.Contains(err.Error(), "outside the finite FP16 range") ||
		errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("attach with invalid weight = %v, %v", acc, err)
	}
	if dev == nil {
		t.Fatal("device did not open for the invalid-weight test")
	}
	if allocs, objects := dev.LedgerLen(); allocs != 0 || objects != 0 {
		t.Fatalf("invalid-weight attach retains %d allocations and %d objects", allocs, objects)
	}
}

// eg2Synthetic256_512Model builds a minimal 2-layer EmbeddingGemma 2 model
// with real head dimensions (layer 0: HD=256 nKV=2 sliding BM=32; layer 1:
// HD=512 nKV=1 full BM=16) so tests exercise both distinct BM descriptor plans
// and actual tracked device allocations.
func eg2Synthetic256_512Model(t *testing.T) *embeddinggemma2.Model {
	t.Helper()
	const (
		vocab = 8
		H     = 32
		I     = 32
		P     = 16
		E     = 16
		L     = 2
	)
	cfg := embeddinggemma2.Config{
		VocabSize:     vocab,
		Hidden:        H,
		Intermediate:  I,
		Layers:        L,
		Heads:         4,
		KVHeads:       2,
		HeadDim:       256,
		SlidingWindow: 512,
		PLEDim:        P,
		EmbeddingDim:  E,
		RMSNormEps:    1e-6,
		HiddenAct:     "gelu_pytorch_tanh",
		LayerTypes:    []string{"sliding_attention", "full_attention"},
		MaxPositions:  2048,
		PerLayerConfig: map[string]struct {
			HeadDim int `json:"head_dim"`
			KVHeads int `json:"num_key_value_heads"`
		}{
			"1": {HeadDim: 512, KVHeads: 1},
		},
		RopeParameters: map[string]struct {
			Theta float64 `json:"rope_theta"`
			Type  string  `json:"rope_type"`
		}{
			"sliding_attention": {Theta: 10000, Type: "default"},
			"full_attention":    {Theta: 1000000, Type: "default"},
		},
	}
	type spec struct {
		name  string
		shape []int
		fill  float32
	}
	specs := []spec{
		{"ple.per_layer_model_projection.weight", []int{L * P, H}, 0.02},
		{"ple.per_layer_projection_norm.weight", []int{P}, 1.0},
		{"norm.weight", []int{H}, 1.0},
		{"embedding_projection.weight", []int{E, H}, 0.05},
	}
	for li, geom := range [][2]int{{256, 2}, {512, 1}} {
		hd, kvh := geom[0], geom[1]
		qd, kvd := 4*hd, kvh*hd
		p := fmt.Sprintf("layers.%d.", li)
		specs = append(specs,
			spec{p + "input_layernorm.weight", []int{H}, 1.0},
			spec{p + "self_attn.q_proj.weight", []int{qd, H}, 0.01},
			spec{p + "self_attn.k_proj.weight", []int{kvd, H}, 0.01},
			spec{p + "self_attn.v_proj.weight", []int{kvd, H}, 0.01},
			spec{p + "self_attn.o_proj.weight", []int{H, qd}, 0.01},
			spec{p + "self_attn.q_norm.weight", []int{hd}, 1.0},
			spec{p + "self_attn.k_norm.weight", []int{hd}, 1.0},
			spec{p + "post_attention_layernorm.weight", []int{H}, 1.0},
			spec{p + "pre_feedforward_layernorm.weight", []int{H}, 1.0},
			spec{p + "mlp.gate_proj.weight", []int{I, H}, 0.02},
			spec{p + "mlp.up_proj.weight", []int{I, H}, 0.02},
			spec{p + "mlp.down_proj.weight", []int{H, I}, 0.02},
			spec{p + "post_feedforward_layernorm.weight", []int{H}, 1.0},
			spec{p + "ple_block.per_layer_input_gate.weight", []int{P, H}, 0.02},
			spec{p + "ple_block.per_layer_projection.weight", []int{H, P}, 0.02},
			spec{p + "ple_block.post_per_layer_input_norm.weight", []int{H}, 1.0},
			spec{p + "layer_scalar", []int{1}, 1.0},
		)
	}
	header := make(map[string]map[string]any, len(specs))
	var rawData []byte
	for idx, sp := range specs {
		n := 1
		for _, d := range sp.shape {
			n *= d
		}
		start := len(rawData)
		buf := make([]byte, n*4)
		for i := range n {
			val := sp.fill
			if sp.fill < 0.5 && (i+idx)%2 == 1 {
				val = -sp.fill
			}
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(val))
		}
		rawData = append(rawData, buf...)
		header["language_model."+sp.name] = map[string]any{
			"dtype":        "F32",
			"shape":        sp.shape,
			"data_offsets": []int{start, len(rawData)},
		}
	}
	hdrJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	st := make([]byte, 8+len(hdrJSON)+len(rawData))
	binary.LittleEndian.PutUint64(st[:8], uint64(len(hdrJSON)))
	copy(st[8:], hdrJSON)
	copy(st[8+len(hdrJSON):], rawData)
	tokens := make([]byte, vocab*H*4)
	for i := range vocab * H {
		val := float32(0.1) * float32(1+(i%7))
		if i%2 == 1 {
			val = -val
		}
		binary.LittleEndian.PutUint32(tokens[i*4:], math.Float32bits(val))
	}
	m, err := embeddinggemma2.NewModel(cfg, tokens, st)
	if err != nil {
		t.Fatalf("NewModel(256/512): %v", err)
	}
	return m
}

// TestEG2FP16PackedAttnContract verifies that cuda-eg2-fp16 enables FP16 FFN,
// packed split-f16 attention, and both GELU/PLE fusions on the shared
// per-forward status word, matches BatchMemory vs actual tracked lease peaks
// across ragged lengths and both BM=32 and BM=16, preserves raw FP32 bits
// across the M=36/37 shortTiled boundary, clamps sliding_window >= 2^32 before
// int32 narrowing, and keeps native cleanup/execution errors strictly ahead of
// errEG2FP16InputRange.
func TestEG2FP16PackedAttnContract(t *testing.T) {
	adapter := eg2FP16TestAdapter(t)
	m := eg2TinyModel(t)
	rec := &eg2RecordingLeaser{}
	origHook := leaseSeenForTest
	t.Cleanup(func() { leaseSeenForTest = origHook })
	leaseSeenForTest = func(onDevice bool, line string, requested int64, release func()) func() {
		if origHook != nil {
			release = origHook(onDevice, line, requested, release)
		}
		recordRelease := rec.Lease(onDevice, line, requested, gpu.ChargedBytes(requested))
		return func() {
			release()
			recordRelease()
		}
	}

	acc, err := m.NewAccelerator(eg2FP16TestSelector + ":" + adapter.UUID)
	if err != nil {
		t.Fatal(err)
	}
	a := acc.(*eg2Accel)
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Error(err)
		}
	})
	if !a.packedAttnMode || len(a.packedAttn) == 0 || a.ffnFP16 == (Pipeline{}) ||
		a.fusions == nil || a.fusions.geluMul == (Pipeline{}) || a.fusions.geluPLE == (Pipeline{}) {
		t.Fatalf("cuda-eg2-fp16 did not enable packedAttn (%v, %d pipes), ffnFP16 (%v), and fusions (%v)",
			a.packedAttnMode, len(a.packedAttn), a.ffnFP16 != (Pipeline{}), a.fusions != nil)
	}

	expectedColdRope := func(acc *eg2Accel, maxT int) int64 {
		var total int64
		for _, k := range acc.distinctRopeKeys() {
			total += 2 * gpu.ChargedBytes(int64(maxT*k.hd)*4)
		}
		return total
	}

	// 1. Actual tracked lease peak vs BatchMemory on ragged many-short-segment
	// lengths (BM=32), plus strict savings vs cuda-eg2-combined and exact
	// cold/warm no-eviction peaks and RoPE cache retention.
	comb, err := m.NewAccelerator("cuda-eg2-combined:" + adapter.UUID)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := comb.Close(); err != nil {
			t.Error(err)
		}
	})
	raggedLens := []int{1, 1, 1, 1, 1, 1, 1, 1, 3, 17, 33} // total 61, 11 ragged segments, maxT=33
	pEst, err := a.BatchMemory(raggedLens)
	if err != nil {
		t.Fatal(err)
	}
	cEst, err := comb.(*eg2Accel).BatchMemory(raggedLens)
	if err != nil {
		t.Fatal(err)
	}
	if pEst.AdditionalPeakBytes >= cEst.AdditionalPeakBytes {
		t.Fatalf("cuda-eg2-fp16 AdditionalPeakBytes=%d >= combined %d (expected qkv/bounds/pleL savings)",
			pEst.AdditionalPeakBytes, cEst.AdditionalPeakBytes)
	}
	raggedBatch := make([][]int, len(raggedLens))
	for i, n := range raggedLens {
		row := make([]int, n)
		for j := range n {
			row[j] = (i*7 + j) % m.Config().VocabSize
		}
		raggedBatch[i] = row
	}
	wantTinyColdEnd := expectedColdRope(a, 33)
	base, allocs := rec.resetWindow()
	if _, err := a.ForwardBatch(raggedBatch); err != nil {
		t.Fatalf("ragged ForwardBatch (cold): %v", err)
	}
	peak, end, _ := rec.windowDelta(base, allocs)
	if peak != pEst.AdditionalPeakBytes || end != wantTinyColdEnd {
		t.Fatalf("tiny BM=32 ragged cold peak=%d (want %d), end=%d (want %d)",
			peak, pEst.AdditionalPeakBytes, end, wantTinyColdEnd)
	}
	pEstWarm, err := a.BatchMemory(raggedLens)
	if err != nil {
		t.Fatal(err)
	}
	base, allocs = rec.resetWindow()
	if _, err := a.ForwardBatch(raggedBatch); err != nil {
		t.Fatalf("ragged ForwardBatch (warm): %v", err)
	}
	peak, end, _ = rec.windowDelta(base, allocs)
	if peak != pEstWarm.AdditionalPeakBytes || end != 0 {
		t.Fatalf("tiny BM=32 ragged warm peak=%d (want %d), end=%d (want 0)",
			peak, pEstWarm.AdditionalPeakBytes, end)
	}

	// 2. Both BM=32 (HD=256) and BM=16 (HD=512) actual lease peak vs BatchMemory
	// on ragged many-short-segment lengths.
	mRealGeom := eg2Synthetic256_512Model(t)
	accReal, err := mRealGeom.NewAccelerator(eg2FP16TestSelector + ":" + adapter.UUID)
	if err != nil {
		t.Fatalf("256/512 attach: %v", err)
	}
	aReal := accReal.(*eg2Accel)
	t.Cleanup(func() {
		if err := aReal.Close(); err != nil {
			t.Error(err)
		}
	})
	if bms := aReal.distinctPackedBMs(); !slices.Equal(bms, []int{32, 16}) {
		t.Fatalf("distinctPackedBMs = %v, want [32 16]", bms)
	}
	estReal, err := aReal.BatchMemory(raggedLens)
	if err != nil {
		t.Fatal(err)
	}
	realRagged := make([][]int, len(raggedLens))
	for i, n := range raggedLens {
		row := make([]int, n)
		for j := range n {
			row[j] = (i + j) % mRealGeom.Config().VocabSize
		}
		realRagged[i] = row
	}
	wantRealColdEnd := expectedColdRope(aReal, 33)
	base, allocs = rec.resetWindow()
	if _, err := aReal.ForwardBatch(realRagged); err != nil {
		t.Fatalf("256/512 ragged ForwardBatch (cold): %v", err)
	}
	peak, end, _ = rec.windowDelta(base, allocs)
	if peak != estReal.AdditionalPeakBytes || end != wantRealColdEnd {
		t.Fatalf("BM=32+16 ragged cold peak=%d (want %d), end=%d (want %d)",
			peak, estReal.AdditionalPeakBytes, end, wantRealColdEnd)
	}
	estRealWarm, err := aReal.BatchMemory(raggedLens)
	if err != nil {
		t.Fatal(err)
	}
	base, allocs = rec.resetWindow()
	if _, err := aReal.ForwardBatch(realRagged); err != nil {
		t.Fatalf("256/512 ragged ForwardBatch (warm): %v", err)
	}
	peak, end, _ = rec.windowDelta(base, allocs)
	if peak != estRealWarm.AdditionalPeakBytes || end != 0 {
		t.Fatalf("BM=32+16 ragged warm peak=%d (want %d), end=%d (want 0)",
			peak, estRealWarm.AdditionalPeakBytes, end)
	}

	// 3. M=36/37 shortTiled crossing: compare every raw projected FP32 bit of a
	// 36-token member alone vs paired with a 1-token member in both orders.
	id36 := make([]int, 36)
	for i := range id36 {
		id36[i] = (i*5 + 1) % m.Config().VocabSize
	}
	id1 := []int{2 % m.Config().VocabSize}
	single36, _, err := a.Forward(id36, false)
	if err != nil {
		t.Fatalf("Forward(36): %v", err)
	}
	for _, tc := range []struct {
		label string
		batch [][]int
		idx   int
	}{
		{"[36,1]", [][]int{id36, id1}, 0},
		{"[1,36]", [][]int{id1, id36}, 1},
	} {
		got, err := a.ForwardBatch(tc.batch)
		if err != nil {
			t.Fatalf("%s ForwardBatch: %v", tc.label, err)
		}
		if d := eg2FirstDiff(got[tc.idx], single36, m.Dim()); d != "" {
			t.Fatalf("M=36/37 crossing %s raw bit mismatch: %s", tc.label, d)
		}
	}

	// 4. Sliding window >= 2^32 clamp before int32 narrowing on >= 2 distinct rows:
	// radius 4294967296 must preserve full-segment visibility rather than wrapping to 0.
	cfgBytes, err := os.ReadFile(filepath.Join(eg2TinyDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	wideDir := t.TempDir()
	wideCfg := strings.Replace(string(cfgBytes), `"sliding_window": 3`, `"sliding_window": 4294967296`, 1)
	if !strings.Contains(wideCfg, `"sliding_window": 4294967296`) {
		t.Fatal("failed to patch sliding_window in tiny config.json")
	}
	if err := os.WriteFile(filepath.Join(wideDir, "config.json"), []byte(wideCfg), 0o600); err != nil {
		t.Fatal(err)
	}
	stAbs, err := filepath.Abs(filepath.Join(eg2TinyDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(stAbs, filepath.Join(wideDir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	wideModel, err := embeddinggemma2.Load(wideDir)
	if err != nil {
		t.Fatal(err)
	}
	wideAcc, err := wideModel.NewAccelerator(eg2FP16TestSelector + ":" + adapter.UUID)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := wideAcc.Close(); err != nil {
			t.Error(err)
		}
	})
	wideIDs := eg2TinyInputs(t, wideModel.Config().VocabSize)[2]
	wantWide, _, err := wideModel.ForwardCPU(wideIDs, false)
	if err != nil {
		t.Fatal(err)
	}
	gotWide, _, err := wideAcc.Forward(wideIDs, false)
	if err != nil || len(gotWide) != len(wantWide) || !eg2Finite(gotWide) || eg2Cos(gotWide, wantWide) < 0.999 {
		t.Fatalf("sliding_window=2^32 parity: cos=%.8f err=%v", eg2Cos(gotWide, wantWide), err)
	}

	// 5. Simultaneously armed FP16 status + planted failing ReleaseBufChecked
	// in composed (FFN+packed+fusions), packed-attention-only, and FFN-only modes:
	// cleanup error must win without joining errEG2FP16InputRange or
	// ErrBatchOutOfMemory, next call must be sticky with zero GPU work, and
	// Close must release ownership.
	inputs := eg2TinyInputs(t, m.Config().VocabSize)
	packedPTX, err := os.ReadFile("testdata/eg2_packed_attn.ptx")
	if err != nil {
		t.Fatal(err)
	}
	ffnPTX, err := os.ReadFile("testdata/eg2_ffn_fp16.ptx")
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []struct {
		name      string
		ffnPTX    []byte
		packedPTX []byte
		fusionPTX []byte
		ffnOnly   bool
	}{
		{"fp16-ffn+packed+fusions", ffnPTX, packedPTX, eg2FusionsPTX, false},
		{"packed-attn-only", nil, packedPTX, nil, false},
		{"fp16-ffn-only", ffnPTX, nil, nil, true},
	} {
		t.Run(mode.name, func(t *testing.T) {
			ovModel := eg2TinyModel(t)
			if !mode.ffnOnly {
				for i := range ovModel.Weights().Layers[0].KNorm {
					ovModel.Weights().Layers[0].KNorm[i] = 1e5
				}
			}
			ovA, err := newEG2AccelConfigured(ovModel, true, mode.ffnPTX, mode.packedPTX, mode.fusionPTX, func() (*Device, error) {
				return gpu.CreateSelectedDevice(adapter)
			})
			if err != nil {
				t.Fatalf("attach: %v", err)
			}
			runOverflow := func(keepLayers bool) ([]float32, [][]float32, error) {
				if mode.ffnOnly {
					x, err := ovModel.EmbedTokens(inputs[0])
					if err != nil {
						t.Fatal(err)
					}
					x[len(x)-1] = float32(math.Inf(1))
					return ovA.forwardRows(x, []int{len(inputs[0])}, keepLayers)
				}
				return ovA.Forward(inputs[0], keepLayers)
			}
			// Healthy numeric rejection first: nil outputs + errEG2FP16InputRange + valid reuse.
			ovOut, ovLays, err := runOverflow(true)
			if !errors.Is(err, errEG2FP16InputRange) || ovOut != nil || ovLays != nil || ovA.broken != nil {
				t.Fatalf("healthy range reject: len(out)=%d len(layers)=%d broken=%v err=%v", len(ovOut), len(ovLays), ovA.broken, err)
			}
			zeros := make([]float32, 4*ovModel.Config().Hidden)
			reuseOut, _, err := ovA.ForwardEmbeds(zeros, 4, false)
			if err != nil || len(reuseOut) != 4*ovModel.Config().EmbeddingDim || !eg2Finite(reuseOut) {
				t.Fatalf("clean reuse after range reject failed: len=%d err=%v", len(reuseOut), err)
			}

			// Now arm BOTH overflow status AND failing ReleaseBufChecked: cleanup
			// error must take precedence with nil kept layers, no range/retry sentinel,
			// sticky next call with zero GPU launches, and clean Close recovery.
			plantedFree := errors.New("cuda: planted scratch free failure")
			ovA.ops.dev.SetCloseBufHookForTest(func(func() error) error { return plantedFree })
			ovOut, ovLays, err = runOverflow(true)
			if ovOut != nil || ovLays != nil || !errors.Is(err, plantedFree) ||
				errors.Is(err, errEG2FP16InputRange) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("armed status + failed free: len(out)=%d len(layers)=%d err=%v", len(ovOut), len(ovLays), err)
			}
			if ovA.broken == nil {
				t.Fatal("failed free did not poison accelerator")
			}
			launches := 0
			origTrace := gpu.TraceForTest
			gpu.TraceForTest = func(ev gpu.TraceEvent) {
				if origTrace != nil {
					origTrace(ev)
				}
				if ev.Op == gpu.TraceLaunch {
					launches++
				}
			}
			_, _, stickyErr := ovA.Forward(inputs[0], false)
			gpu.TraceForTest = origTrace
			if !errors.Is(stickyErr, plantedFree) || errors.Is(stickyErr, errEG2FP16InputRange) || launches != 0 {
				t.Fatalf("sticky follow-up: err=%v launches=%d, want plantedFree and 0 launches", stickyErr, launches)
			}
			// Clear free hook and prove Close releases all remaining ownership cleanly.
			ovA.ops.dev.SetCloseBufHookForTest(nil)
			dev := ovA.ops.dev
			if err := ovA.Close(); err != nil {
				t.Fatalf("Close retry after cleared free hook failed: %v", err)
			}
			if allocsLeft, objsLeft := dev.LedgerLen(); allocsLeft != 0 || objsLeft != 0 {
				t.Fatalf("Close left %d allocs and %d objects", allocsLeft, objsLeft)
			}
		})
	}
}

var eg2FusionPack = flag.String("eg2-fusion-pack", "", "EmbeddingGemma 2 pack for the fusion test")
var eg2FusionIDs = flag.String("eg2-fusion-ids", "", "pinned prepared IDs for the fusion test")

func TestEG2FusionsLinked(t *testing.T) {
	tagged := eg2TaggedBinary(t, "eg2_fusions") || eg2TaggedBinary(t, "eg2_combined", "eg2_fp16")
	if linked := len(eg2FusionsPTX) != 0; linked != tagged {
		t.Fatalf("fusion PTX linked = %t, fusion build tags = %t", linked, tagged)
	}
}

func TestEG2FusionsAdmission(t *testing.T) {
	for _, tc := range []struct {
		name string
		i, p int
	}{
		{"FFN", 262144, 256},
		{"PLE", 2048, 262144},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := &eg2Accel{
				cfg: embeddinggemma2.Config{
					Hidden: 768, Intermediate: tc.i, PLEDim: tc.p,
					Layers: 1, Heads: 8, EmbeddingDim: 768, MaxPositions: 8192,
				},
			}
			if _, err := a.BatchMemory([]int{8192}); err != nil {
				t.Fatalf("baseline singleton estimate rejected: %v", err)
			}
			a.fusions = &eg2Fusions{}
			if _, err := a.BatchMemory([]int{8191}); err != nil {
				t.Fatalf("in-range fused singleton estimate rejected: %v", err)
			}
			if _, err := a.BatchMemory([]int{8192}); !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
				t.Fatalf("out-of-range fused singleton estimate: %v", err)
			}
			rows, layers, err := a.forwardRows(nil, []int{8192}, true)
			if rows != nil || layers != nil || !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
				t.Fatalf("out-of-range fused forward returned rows = %v, layers = %v, err = %v", rows, layers, err)
			}
		})
	}
}

func TestEG2FusionsTiny(t *testing.T) {
	m := eg2TinyModel(t)
	inputs := make([][]int, 3)
	for i, n := range []int{17, 129, 36} {
		inputs[i] = eg2SeqIDs(n, i+1, m.Config().VocabSize)
	}
	checkEG2Fusions(t, m, inputs, true)
}

func TestEG2FusionsReal(t *testing.T) {
	if *eg2FusionPack == "" || *eg2FusionIDs == "" {
		t.Skip("set eg2-fusion-pack and eg2-fusion-ids for the fixed real batch")
	}
	data, err := os.ReadFile(*eg2FusionIDs)
	if err != nil {
		t.Fatal(err)
	}
	if digest := fmt.Sprintf("%x", sha256.Sum256(data)); digest != "97c426da8c155b04bf0b34e270dcfa5e8cf18bfc1343e926f63e638fea6bcd22" {
		t.Fatalf("unexpected prepared-ID digest: %s", digest)
	}
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	scanner.Buffer(nil, 1<<17)
	if !scanner.Scan() || scanner.Text() != "batch\tids" {
		t.Fatal("missing prepared-ID TSV header")
	}
	var inputs [][]int
	positions := 0
	for scanner.Scan() {
		f := strings.Fields(scanner.Text())
		if len(f) < 2 {
			t.Fatal("empty input row")
		}
		if f[0] != "1" {
			break
		}
		ids := make([]int, len(f)-1)
		for i, field := range f[1:] {
			id, err := strconv.Atoi(field)
			if err != nil || id < 0 {
				t.Fatalf("bad token %q", field)
			}
			ids[i] = id
		}
		inputs = append(inputs, ids)
		positions += len(ids)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(inputs) != 17 || positions != 7924 {
		t.Fatalf("wanted the first frozen group, got %d inputs and %d positions", len(inputs), positions)
	}

	read := func(name string) []byte {
		t.Helper()
		buf, err := os.ReadFile(filepath.Join(*eg2FusionPack, name))
		if err != nil {
			t.Fatal(err)
		}
		return buf
	}
	var packed struct {
		Config json.RawMessage
	}
	if err := json.Unmarshal(read("config.json"), &packed); err != nil {
		t.Fatal(err)
	}
	cfg, err := embeddinggemma2.LoadConfigBytes(packed.Config)
	if err != nil {
		t.Fatal(err)
	}
	m, err := embeddinggemma2.NewModel(cfg, read("tokens.bf16"), read("layers.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	checkEG2Fusions(t, m, inputs, false)
}

func checkEG2Fusions(t *testing.T, m *embeddinggemma2.Model, inputs [][]int, keepLayers bool) {
	t.Helper()
	hasFP16 := eg2TaggedBinary(t, "eg2_combined", "eg2_fp16")
	if !eg2TaggedBinary(t, "eg2_fusions") && !hasFP16 {
		t.Skip("build with cuda,eg2_fusions or cuda,eg2_combined,eg2_fp16 for the fused encoder check")
	}
	if len(eg2FusionsPTX) == 0 {
		t.Fatal("tagged build is missing the fusion PTX")
	}
	id := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if id == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID for the fusion encoder check")
	}
	all, err := gpu.Adapters()
	if err != nil {
		t.Fatal(err)
	}
	device, err := resolveCUDADevice(id, all)
	if err != nil {
		t.Fatal(err)
	}
	lengths := make([]int, len(inputs))
	rows := 0
	for i, in := range inputs {
		lengths[i] = len(in)
		rows += len(in)
	}

	type fusionArm struct {
		name              string
		fp16PTX           []byte
		packedPTX         []byte
		checkPublicSelect bool
	}
	arms := []fusionArm{{name: "combined-fp32"}}
	if hasFP16 && device.Major >= 8 {
		ffnPTX, err := os.ReadFile("testdata/eg2_ffn_fp16.ptx")
		if err != nil {
			t.Fatal(err)
		}
		packedPTX, err := os.ReadFile("testdata/eg2_packed_attn.ptx")
		if err != nil {
			t.Fatal(err)
		}
		arms = append(arms, fusionArm{
			name:              "composed-fp16-packed",
			fp16PTX:           ffnPTX,
			packedPTX:         packedPTX,
			checkPublicSelect: true,
		})
	}

	for _, arm := range arms {
		t.Run(arm.name, func(t *testing.T) {
			type evidence struct {
				output, snapshots        [][]float32
				estimate, peak           int64
				launches, copies, allocs int
			}
			run := func(ptx []byte) evidence {
				t.Helper()
				rec := &eg2RecordingLeaser{}
				old := leaseSeenForTest
				leaseSeenForTest = func(onDevice bool, line string, requested int64, release func()) func() {
					if old != nil {
						release = old(onDevice, line, requested, release)
					}
					recordRelease := rec.Lease(onDevice, line, requested, gpu.ChargedBytes(requested))
					return func() { release(); recordRelease() }
				}
				defer func() { leaseSeenForTest = old }()
				a, err := newEG2AccelConfigured(m, true, arm.fp16PTX, arm.packedPTX, ptx, func() (*Device, error) {
					return gpu.CreateSelectedDevice(device)
				})
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := a.Close(); err != nil {
						t.Error(err)
					}
				}()
				if a.AttachedDevice().ID != "cuda:"+device.UUID || (a.fusions != nil) != (len(ptx) != 0) {
					t.Fatal("wrong device or fusion mode")
				}
				fit, err := a.BatchMemory(lengths)
				if err != nil || fit.Quality != embeddinggemma2.BatchMemoryUpperBound {
					t.Fatalf("batch fit = %v, %v", fit, err)
				}
				mem, err := a.DeviceMemory()
				const GiB = 1 << 30
				if err != nil || mem.AvailableBytes < fit.AdditionalPeakBytes+GiB ||
					mem.TotalBytes-mem.AvailableBytes+fit.AdditionalPeakBytes+GiB > 6*GiB {
					t.Fatalf("device fit refused: memory=%v estimate=%v error=%v", mem, fit, err)
				}

				stats := evidence{estimate: fit.AdditionalPeakBytes}
				base, beforeAllocs := rec.resetWindow()
				lo, _ := a.ops.dev.Activity()
				oldTrace := gpu.TraceForTest
				func() {
					defer func() { gpu.TraceForTest = oldTrace }()
					gpu.TraceForTest = func(event gpu.TraceEvent) {
						if oldTrace != nil {
							oldTrace(event)
						}
						if event.Op == gpu.TraceCopy2D {
							stats.copies++
						}
					}
					stats.output, err = a.ForwardBatch(inputs)
				}()
				if err != nil {
					t.Fatal(err)
				}
				hi, _ := a.ops.dev.Activity()
				stats.launches = int(hi - lo)
				stats.peak, _, stats.allocs = rec.windowDelta(base, beforeAllocs)
				if stats.peak > stats.estimate {
					t.Fatalf("observed peak %d exceeds fit estimate %d", stats.peak, stats.estimate)
				}

				if keepLayers {
					single, snapshots, err := a.Forward(inputs[0], true)
					if err != nil {
						t.Fatal(err)
					}
					if len(snapshots) != m.Config().Layers+1 {
						t.Fatalf("got %d retained layer snapshots", len(snapshots))
					}
					if diff := eg2FirstDiff(single, stats.output[0], m.Dim()); diff != "" {
						t.Fatalf("kept singleton differs from packed: %s", diff)
					}
					stats.snapshots = snapshots
				}
				if len(ptx) == 0 {
					return stats
				}
				for i, ids := range inputs {
					single, _, err := a.Forward(ids, false)
					if err != nil {
						t.Fatal(err)
					}
					if diff := eg2FirstDiff(single, stats.output[i], m.Dim()); diff != "" {
						t.Fatalf("singleton input %d: %s", i, diff)
					}
				}
				mid := len(inputs) / 2
				for _, span := range [][2]int{{0, mid}, {mid, len(inputs)}} {
					part, err := a.ForwardBatch(inputs[span[0]:span[1]])
					if err != nil {
						t.Fatal(err)
					}
					for i, got := range part {
						if diff := eg2FirstDiff(got, stats.output[span[0]+i], m.Dim()); diff != "" {
							t.Fatalf("split input %d: %s", span[0]+i, diff)
						}
					}
				}
				reverse := slices.Clone(inputs)
				slices.Reverse(reverse)
				got, err := a.ForwardBatch(reverse)
				if err != nil {
					t.Fatal(err)
				}
				for i, out := range got {
					if diff := eg2FirstDiff(out, stats.output[len(inputs)-1-i], m.Dim()); diff != "" {
						t.Fatalf("reversed input %d: %s", i, diff)
					}
				}
				return stats
			}
			ref, fused := run(nil), run(eg2FusionsPTX)
			for i, out := range fused.output {
				if diff := eg2FirstDiff(out, ref.output[i], m.Dim()); diff != "" {
					t.Fatalf("packed input %d: %s", i, diff)
				}
				if !eg2Finite(out) {
					t.Fatalf("input %d produced nonfinite output", i)
				}
				// Pooling is shared by the two paths and receives identical rows.
				got := eg2PoolNorm(out, lengths[i], m.Dim(), m.Dim())
				want := eg2PoolNorm(ref.output[i], lengths[i], m.Dim(), m.Dim())
				if diff := eg2FirstDiff(got, want, m.Dim()); diff != "" {
					t.Fatalf("pooled input %d: %s", i, diff)
				}
			}
			for i := range fused.snapshots {
				if diff := eg2FirstDiff(fused.snapshots[i], ref.snapshots[i], m.Config().Hidden); diff != "" {
					t.Fatalf("layer state %d: %s", i, diff)
				}
			}
			wantBytes := gpu.ChargedBytes(int64(rows) * int64(m.Config().PLEDim) * 4)
			layers := m.Config().Layers
			if ref.launches-fused.launches != 2*layers || ref.copies-fused.copies != layers ||
				ref.allocs-fused.allocs != 1 || ref.estimate-fused.estimate != wantBytes || ref.peak-fused.peak != wantBytes {
				t.Fatalf("unexpected fusion savings: launches %d, copies %d, allocations %d, estimated bytes %d, peak bytes %d",
					ref.launches-fused.launches, ref.copies-fused.copies, ref.allocs-fused.allocs,
					ref.estimate-fused.estimate, ref.peak-fused.peak)
			}
			if arm.checkPublicSelect {
				pub, err := m.NewAccelerator(eg2FP16TestSelector + ":" + device.UUID)
				if err != nil {
					t.Fatalf("public %s attach: %v", eg2FP16TestSelector, err)
				}
				defer func() {
					if err := pub.Close(); err != nil {
						t.Error(err)
					}
				}()
				pubA := pub.(*eg2Accel)
				if pubA.fusions == nil || !pubA.packedAttnMode || pubA.ffnFP16 == (Pipeline{}) {
					t.Fatalf("public %s missing composed pipelines: ffn=%v packed=%v fusions=%v",
						eg2FP16TestSelector, pubA.ffnFP16 != (Pipeline{}), pubA.packedAttnMode, pubA.fusions != nil)
				}
				pubOut, err := pubA.ForwardBatch(inputs)
				if err != nil {
					t.Fatalf("public %s ForwardBatch: %v", eg2FP16TestSelector, err)
				}
				for i, got := range pubOut {
					if diff := eg2FirstDiff(got, fused.output[i], m.Dim()); diff != "" {
						t.Fatalf("public %s input %d differs from composed fused arm: %s", eg2FP16TestSelector, i, diff)
					}
				}
			}
			t.Logf("[%s] %d inputs, %d rows: raw and pooled bits match, kernel launches %d -> %d, D2D copies %d -> %d, allocations %d -> %d, tracked peak bytes %d -> %d",
				arm.name, len(inputs), rows, ref.launches, fused.launches, ref.copies, fused.copies, ref.allocs, fused.allocs, ref.peak, fused.peak)
		})
	}
}
