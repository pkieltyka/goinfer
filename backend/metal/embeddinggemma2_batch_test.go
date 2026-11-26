//go:build darwin

package metal

import (
	"errors"
	"flag"
	"fmt"
	"maps"
	"math"
	"slices"
	"testing"

	gpu "github.com/pkieltyka/goinfer/backend/metal/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/internal/modelcore/embed"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
)

func eg2BatchTestState() *eg2BatchAccel {
	a := &eg2Accel{
		eg2Ops: eg2Ops{d: &Device{}},
		cfg: embeddinggemma2.Config{Hidden: 512, Intermediate: 2048, PLEDim: 512, EmbeddingDim: 768,
			Layers: 24, Heads: 4, VocabSize: 262144, MaxPositions: 8192},
		profile: eg2MatrixProfile,
		kvHeads: eg2GoogleKVHeads(),
		ropes:   make(map[eg2RopeKey][2]Buffer),
	}
	for _, n := range a.kvHeads {
		hd, theta := 256, float64(10000)
		if n == 1 {
			hd, theta = 512, 1000000
		}
		a.layers = append(a.layers, eg2Layer{w: embeddinggemma2.LayerWeights{HeadDim: hd, KVHeads: n, RopeTheta: theta}})
	}
	return &eg2BatchAccel{eg2Accel: a}
}

func TestEG2BatchPlanner(t *testing.T) {
	a := eg2BatchTestState()
	a.weightBytes = 123456
	lengths := []int{30, 31, 1690, 997}
	p, err := a.planBatch(lengths)
	if err != nil {
		t.Fatal(err)
	}
	if p.rows != 2748 || p.runCount != 3 || p.ropeCount != 2 || !p.growScratch || p.arenaBytes == 0 {
		t.Fatalf("cold plan: %+v", p)
	}
	wantRuns := []eg2BatchRun{{eg2BatchRows{0, 30}, false}, {eg2BatchRows{30, 1721}, true}, {eg2BatchRows{1751, 997}, true}}
	if !slices.Equal(p.runs[:p.runCount], wantRuns) {
		t.Fatalf("runs: %v, want %v", p.runs[:p.runCount], wantRuns)
	}
	var logical int
	for _, n := range p.scratchWords {
		logical += 4 * n
	}
	if logical != 321314816 {
		t.Fatalf("workspace for 2748 rows: %d", logical)
	}
	cold, err := a.BatchMemory(lengths)
	if err != nil || cold.Quality != embeddinggemma2.BatchMemoryApproximate ||
		cold.ResidentWeightBytes != 123456 || cold.AdditionalPeakBytes != p.additionalBytes ||
		cold.AdditionalPeakBytes < int64(logical+p.arenaBytes) {
		t.Fatalf("cold estimate: %+v, %v", cold, err)
	}
	if n := testing.AllocsPerRun(20, func() { _, _ = a.BatchMemory(lengths) }); n != 0 {
		t.Fatalf("BatchMemory allocated %g times", n)
	}
	for _, lengths := range [][]int{{1, 2748}, {1, 1, 1, 1, 1}, {8193}} {
		ids := make([][]int, len(lengths))
		for i, n := range lengths {
			ids[i] = make([]int, n)
		}
		_, estimateErr := a.BatchMemory(lengths)
		_, execErr := a.ForwardBatch(ids)
		if !errors.Is(estimateErr, embeddinggemma2.ErrUnsupportedBatchShape) ||
			!errors.Is(execErr, embeddinggemma2.ErrUnsupportedBatchShape) {
			t.Fatalf("%v: estimate %v, execute %v", lengths, estimateErr, execErr)
		}
	}
	if _, err := a.ForwardBatch([][]int{{1}, {-1}}); err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
		t.Fatalf("invalid ID: %v", err)
	}
	if _, err := a.BatchMemory([]int{8192}); err != nil {
		t.Fatalf("singleton model limit: %v", err)
	}
	if a.eg2Accel.AttachedDevice().TextBatching || !a.AttachedDevice().TextBatching {
		t.Fatal("batch capability leaked to legacy identity")
	}
	if _, ok := any(a.eg2Accel).(interface {
		ForwardBatch([][]int) ([][]float32, error)
	}); ok {
		t.Fatal("legacy attachment implements batch forwarding")
	}
	a.d = nil
	if _, err := a.BatchMemory(lengths); !errors.Is(err, errTowerClosed) {
		t.Fatalf("closed estimate: %v", err)
	}
	if _, err := a.ForwardBatch([][]int{{1}, {1}}); !errors.Is(err, errTowerClosed) {
		t.Fatalf("closed forward: %v", err)
	}
}

func eg2BatchDevice(t *testing.T) *Device {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal: %v", err)
	}
	t.Cleanup(func() {
		d.ReleaseAll()
		d.ReleaseObjects()
	})
	if d.Name() != "Apple M4" || d.ArchitectureName() != "applegpu_g16g" {
		t.Skipf("base Apple M4 required, got %s (%s)", d.Name(), d.ArchitectureName())
	}
	return d
}

func eg2AllocatedBytes(d *Device) int64 {
	_, charges := d.BufferSizes()
	var total int64
	for _, b := range charges {
		total += b
	}
	return total
}

func TestEG2BatchMetalMemory(t *testing.T) {
	a := eg2BatchTestState()
	a.d = eg2BatchDevice(t)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	p := func(lengths ...int) eg2BatchPlan {
		t.Helper()
		plan, err := a.planBatch(lengths)
		if err != nil {
			t.Fatal(err)
		}
		return plan
	}
	preflight := func(lengths []int, wantGrowth, wantEviction bool) {
		t.Helper()
		plan := p(lengths...)
		if plan.growScratch != wantGrowth || plan.evictRope != wantEviction {
			t.Fatalf("%v: unexpected growth/eviction %+v", lengths, plan)
		}
		before := eg2AllocatedBytes(a.d)
		peak := before
		var allocs, frees, submits int
		gpu.TraceForTest = func(e gpu.TraceEvent) {
			if e.Op == gpu.TraceCommit {
				submits++
			}
		}
		gpu.AllocTraceForTest = func(d *Device, event gpu.AllocTrace) {
			allocs++
			peak = max(peak, eg2AllocatedBytes(d)+event.Allocated)
		}
		gpu.FreeTraceForTest = func(*Device, uintptr, int64) { frees++ }
		defer func() {
			gpu.TraceForTest, gpu.AllocTraceForTest, gpu.FreeTraceForTest = nil, nil, nil
		}()
		est, err := a.BatchMemory(lengths)
		if err != nil || allocs != 0 || frees != 0 || submits != 0 || eg2AllocatedBytes(a.d) != before {
			t.Fatalf("estimate mutated device or failed: %+v, %v", est, err)
		}
		if _, err := a.prepareBatch(&plan, a.allocateBatchBuffer); err != nil {
			t.Fatal(err)
		}
		if peak-before > est.AdditionalPeakBytes || (allocs == 0) != (est.AdditionalPeakBytes == 0) {
			t.Fatalf("%v: estimate=%d measured increase=%d allocs=%d frees=%d", lengths, est.AdditionalPeakBytes, peak-before, allocs, frees)
		}
		t.Logf("lengths=%v additional=%d tracked_peak_increase=%d allocs=%d frees=%d", lengths, est.AdditionalPeakBytes, peak-before, allocs, frees)
	}
	preflight([]int{31, 36}, true, false)
	preflight([]int{31, 36}, false, false)
	preflight([]int{30, 36}, false, false)
	preflight([]int{31, 129}, true, false)
	// Grow neither scratch nor the arena on these new table lengths.
	for l := 20; l < 26; l++ {
		preflight([]int{1, l}, false, false)
	}
	preflight([]int{1, 26}, false, true)
	preflight([]int{2749}, true, false)
	preflight([]int{31, 129}, false, false)
}

func TestEG2BatchMetalAllocationRollback(t *testing.T) {
	a := eg2BatchTestState()
	a.d = eg2BatchDevice(t)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	ids := [][]int{make([]int, 31), make([]int, 64)}
	seed, err := a.planBatch([]int{30, 31})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.prepareBatch(&seed, a.allocateBatchBuffer); err != nil {
		t.Fatal(err)
	}
	a.release(&a.arena)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	arena, scr, capBefore := a.arena, a.scr, a.cap
	ropes := maps.Clone(a.ropes)
	bytes := eg2AllocatedBytes(a.d)
	n, _ := a.d.LedgerLen()
	// Initial arena, middle scratch, and each half of the last cos/sin
	// pair. No failure may publish a new buffer or start a command.
	for _, fail := range []int{1, 5, 22, 23, 24, 25} {
		attempt := 0
		alloc := func(kind eg2BatchBufferKind, size int, data []float32) (Buffer, bool) {
			attempt++
			if attempt == fail {
				return Buffer{}, false
			}
			return a.allocateBatchBuffer(kind, size, data)
		}
		got, err := a.forwardBatch(ids, alloc)
		if got != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || attempt != fail {
			t.Fatalf("allocation %d: output=%v error=%v attempts=%d", fail, got, err, attempt)
		}
		dispatches, submits, waits := a.d.Activity()
		now, _ := a.d.LedgerLen()
		if now != n || a.arena != arena || a.scr != scr || a.cap != capBefore || !maps.Equal(a.ropes, ropes) ||
			eg2AllocatedBytes(a.d) != bytes || dispatches != 0 || submits != 0 || waits != 0 {
			t.Fatalf("allocation %d: partial growth or command escaped", fail)
		}
	}
	retry, err := a.planBatch([]int{31})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.prepareBatch(&retry, a.allocateBatchBuffer); err != nil {
		t.Fatalf("smaller retry cannot allocate: %v", err)
	}
}

var eg2BatchGGUFTest = flag.String("eg2-batch-test-gguf", "", "local EmbeddingGemma 2 Q8_0 GGUF for Metal batch parity")
var eg2BatchMLXTest = flag.String("eg2-batch-test-mlx", "", "local EmbeddingGemma 2 MLX8 checkpoint for Metal batch parity")

func TestEG2BatchMetalPublic(t *testing.T) {
	for _, tc := range []struct {
		name, path string
		load       func(string) (*embeddinggemma2.Encoder, error)
	}{
		{"q8", *eg2BatchGGUFTest, func(path string) (*embeddinggemma2.Encoder, error) { return embeddinggemma2.LoadGGUFEncoder(path, nil) }},
		{"mlx", *eg2BatchMLXTest, embeddinggemma2.LoadEncoder},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.path == "" {
				t.Skip("set the checkpoint flag")
			}
			enc, err := tc.load(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			eg2CheckBatchPublic(t, enc)
		})
	}
}

func eg2CheckBatchPublic(t *testing.T, enc *embeddinggemma2.Encoder) {
	t.Helper()
	// Ragged tails, W32/staged boundaries, repeated and same-length
	// different neighbors, and a singleton above the multi-input cap.
	lengths := []int{30, 31, 31, 524, 870, 879, 2042, 2043, 2749}
	in := make([][]int, len(lengths))
	refs, rawRefs := make([][]float32, len(lengths)), make([][]float32, len(lengths))
	vocab := enc.Model().Config().VocabSize
	for i, n := range lengths {
		in[i] = make([]int, n)
		for j := range n {
			in[i][j] = (7919*j + 104729*i + 13) % vocab
		}
	}
	check := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length %d want %d", label, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s first differing word %d", label, i)
			}
		}
	}
	legacy, err := enc.UseAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = legacy.Close() })
	if legacy.(embeddinggemma2.AttachedDeviceReporter).AttachedDevice().TextBatching {
		t.Fatal("legacy selector advertises batching")
	}
	for i, ids := range in {
		refs[i], err = enc.EmbedIDs(ids)
		if err != nil {
			t.Fatal(err)
		}
		rawRefs[i], _, err = legacy.Forward(ids, false)
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}
	batch, err := enc.UseAccelerator("metal-eg2-batch")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = batch.Close() })
	a := batch.(*eg2BatchAccel)
	if !a.AttachedDevice().TextBatching || a.AttachedProfile() != legacy.(embeddinggemma2.AttachedProfileReporter).AttachedProfile() {
		t.Fatalf("unexpected capability or profile: %+v %s", a.AttachedDevice(), a.AttachedProfile())
	}
	expectWeight := eg2AllocatedBytes(a.d) - a.ones.AllocatedBytes() - a.arena.AllocatedBytes()
	if got, err := enc.BatchMemory([]int{31, 36}); err != nil || got.ResidentWeightBytes != expectWeight || expectWeight <= 0 {
		t.Fatalf("weight estimate %+v, want %d, error %v", got, expectWeight, err)
	}
	for i, ids := range in {
		got, err := enc.EmbedIDs(ids)
		if err != nil {
			t.Fatal(err)
		}
		check(fmt.Sprintf("explicit singleton %d", i), got, refs[i])
		one, err := enc.EmbedIDsBatch([][]int{ids})
		if err != nil {
			t.Fatal(err)
		}
		check("public one-document batch", one[0], refs[i])
	}
	for _, order := range [][]int{
		{3, 4, 5, 1}, {1, 5, 4, 3}, {3, 4, 5, 2}, {3, 4}, {5, 1}, {3, 3, 1},
		{0, 1}, {1, 0}, {1, 6}, {6, 1}, {1, 7}, {7, 1},
	} {
		group := make([][]int, len(order))
		for i, idx := range order {
			group[i] = in[idx]
		}
		before, submits0, waits0 := a.d.Activity()
		got, err := enc.EmbedIDsBatch(group)
		if err != nil {
			t.Fatal(err)
		}
		after, submits1, waits1 := a.d.Activity()
		if submits1-submits0 != 1 || waits1-waits0 != 1 || after-before <= 604 {
			t.Fatalf("packed call %v activity: %d dispatches, %d submits, %d waits", order, after-before, submits1-submits0, waits1-waits0)
		}
		raw, err := a.ForwardBatch(group)
		if err != nil {
			t.Fatal(err)
		}
		for i, idx := range order {
			check("projected rows", raw[i], rawRefs[idx])
			check("pooled 768", got[i], refs[idx])
			check("normalized 512", embed.Truncate(got[i], 512), embed.Truncate(refs[idx], 512))
		}
	}
	for _, group := range [][][]int{{in[0], in[8]}, {in[0], in[1], in[2], in[0], in[1]}} {
		if out, err := enc.EmbedIDsBatch(group); out != nil || !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
			t.Fatalf("oversize batch produced %d outputs, error %v", len(out), err)
		}
	}
}
