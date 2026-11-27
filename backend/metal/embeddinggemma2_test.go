//go:build darwin

package metal

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	gpu "github.com/pkieltyka/goinfer/backend/metal/internal/gpudev"
	"github.com/pkieltyka/goinfer/backend/metal/internal/markdowneval"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/internal/runtime/mem"
	"github.com/pkieltyka/goinfer/tokenizer"
)

const eg2TinyDir = "../../testdata/embeddinggemma2-tiny"

func eg2DigestF32(rows [][]float32, normalized bool) string {
	h := sha256.New()
	var b [4]byte
	for _, v := range rows {
		if normalized {
			v = eg2Normalized512(v)
		}
		for _, f := range v {
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(f))
			h.Write(b[:])
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil))
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

func eg2Pool(h []float32, T, d int) []float32 {
	out := make([]float32, d)
	for t := range T {
		for j := range d {
			out[j] += h[t*d+j] / float32(T)
		}
	}
	return out
}

// TestEG2Metal_matchesCPU is Phase M's Gate 1: on the tiny fixture, the Metal
// accelerator's forward matches the CPU forward (itself gated against transformers, Gate 1 of the CPU path) to cosine
// 0.9999 on every layer's input, on the projected last hidden state and on the pooled embedding. Cases: the golden's
// four (1, 2, 9 and 17 tokens; sliding radius 3 and full layers of different head dims and KV-head counts) and two
// random inputs of 300 and 700 tokens, each at attention blocks of 256 (the default), 4 and 5 query rows, so blocks
// cut through the radius-3 window and leave partial last blocks. The tiny fixture's head dims are 16 and 32; the real
// checkpoint's 256 and 512 are Gate 2's.
func TestEG2Metal_matchesCPU(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatalf("load the tiny fixture: %v", err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	defer acc.Close()
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
	var inputs [][]int
	for _, c := range g.Cases {
		inputs = append(inputs, c.IDs)
	}
	for _, n := range []int{300, 700} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*7919 + 13) % m.Config().VocabSize
		}
		inputs = append(inputs, ids)
	}
	E := m.Dim()
	defer func(b int) { eg2Block = b }(eg2Block)
	for _, blk := range []int{256, 4, 5} {
		eg2Block = blk
		for ci, ids := range inputs {
			cpuLast, cpuLayers, err := m.ForwardCPU(ids, true)
			if err != nil {
				t.Fatal(err)
			}
			gpuLast, gpuLayers, err := acc.Forward(ids, true)
			if err != nil {
				t.Fatalf("case %d: metal forward: %v", ci, err)
			}
			if len(gpuLayers) != len(cpuLayers) {
				t.Fatalf("case %d: %d layer states, CPU %d", ci, len(gpuLayers), len(cpuLayers))
			}
			worst, worstAt := 1.0, -1
			for li := range cpuLayers {
				if c := eg2Cos(gpuLayers[li], cpuLayers[li]); c < worst {
					worst, worstAt = c, li
				}
			}
			lc := eg2Cos(gpuLast, cpuLast)
			ec := eg2Cos(eg2Pool(gpuLast, len(ids), E), eg2Pool(cpuLast, len(ids), E))
			t.Logf("block %d, case %d (%d tokens): worst layer-input cosine %.9f (index %d), last hidden %.9f, embedding %.9f", blk, ci, len(ids), worst, worstAt, lc, ec)
			if worst < 0.9999 || lc < 0.9999 || ec < 0.9999 {
				t.Errorf("block %d, case %d (%d tokens): worst layer %.9f at index %d, last hidden %.9f, embedding %.9f (bar 0.9999)", blk, ci, len(ids), worst, worstAt, lc, ec)
			}
		}
	}
}

func TestMetalInventoryMemory(t *testing.T) {
	oldRAM, oldCeiling := physicalRAM, metalMemoryCeilingForTest
	t.Cleanup(func() { physicalRAM, metalMemoryCeilingForTest = oldRAM, oldCeiling })
	physicalRAM = func() (uint64, error) { return 4096, nil }
	metalMemoryCeilingForTest = 1024
	for _, tc := range []struct {
		unified, include bool
		kind             embeddinggemma2.MemoryKind
		memory           embeddinggemma2.DeviceMemory
	}{
		{true, false, embeddinggemma2.MemoryShared, embeddinggemma2.DeviceMemory{TotalBytes: 4096}},
		{true, true, embeddinggemma2.MemoryShared, embeddinggemma2.DeviceMemory{TotalBytes: 4096, AvailableBytes: 1024, Basis: embeddinggemma2.MemoryEstimated}},
		{false, true, embeddinggemma2.MemoryDedicated, embeddinggemma2.DeviceMemory{}},
	} {
		got := describeMetalDevice("GPU", tc.unified, tc.include)
		if got.Backend != "metal" || got.Name != "GPU" || got.ID != "" || got.Alias != "" ||
			got.TextBatching || got.MemoryKind != tc.kind || got.Memory != tc.memory {
			t.Errorf("unified %t include %t: %+v", tc.unified, tc.include, got)
		}
	}
	physicalRAM = func() (uint64, error) { return 0, errors.New("unreadable") }
	if got := describeMetalDevice("GPU", true, true).Memory; got != (embeddinggemma2.DeviceMemory{}) {
		t.Fatalf("unavailable physical RAM should be unknown: %+v", got)
	}
}

func TestMetalDeviceRowsIdentityAndToken(t *testing.T) {
	facts := []gpu.DeviceFact{
		{Name: "Apple M4", Unified: true, RegistryID: 11},
		{Name: "Apple M4", Unified: true, RegistryID: 22}, // duplicate display name, unique ID
		{Name: "Duplicate", Unified: true, RegistryID: 22},
		{Name: "Unidentified", Unified: true, RegistryID: 0},
	}
	rows := metalDeviceRows(facts, "boot-A", false)
	if rows[0].ID != "metal:11@boot-A" || rows[0].Alias != "metal:0" || rows[1].ID != "" || rows[1].Alias != "" ||
		rows[2].ID != "" || rows[2].Alias != "" || rows[3].ID != "" || rows[3].Alias != "" {
		t.Fatalf("rows = %+v", rows)
	}
	reordered := metalDeviceRows([]gpu.DeviceFact{facts[1], facts[0]}, "boot-A", false)
	if reordered[0].ID != "metal:22@boot-A" || reordered[0].Alias != "metal:0" ||
		reordered[1].ID != "metal:11@boot-A" || reordered[1].Alias != "metal:1" {
		t.Fatalf("reordered rows = %+v", reordered)
	}
	for _, tok := range []string{"0", "3", "11@boot-A"} {
		if _, err := parseMetalToken(tok); err != nil {
			t.Errorf("parseMetalToken(%q): %v", tok, err)
		}
	}
	for _, tok := range []string{"", "x", "@boot-A", "11@", "0@boot-A"} {
		if _, err := parseMetalToken(tok); err == nil {
			t.Errorf("parseMetalToken(%q) accepted a malformed token", tok)
		}
	}
	selectToken := func(tok string, boot string) (int, error) {
		parsed, err := parseMetalToken(tok)
		if err != nil {
			t.Fatal(err)
		}
		return selectMetalIndex(facts, boot, parsed)
	}
	if i, err := selectToken("0", "boot-A"); err != nil || i != 0 {
		t.Fatalf("unique alias: %d, %v", i, err)
	}
	if _, err := selectToken("1", "boot-A"); !errors.Is(err, embeddinggemma2.ErrAmbiguousDevice) {
		t.Fatalf("duplicate alias error = %v", err)
	}
	if _, err := selectToken("3", "boot-A"); !errors.Is(err, embeddinggemma2.ErrNoDevice) {
		t.Fatalf("zero-ID alias error = %v", err)
	}
	if _, err := selectToken("0", ""); !errors.Is(err, embeddinggemma2.ErrNoDevice) {
		t.Fatalf("alias without boot marker error = %v", err)
	}
	if i, err := selectToken("11@boot-A", "boot-A"); err != nil || i != 0 {
		t.Fatalf("exact ID: %d, %v", i, err)
	}
	if _, err := selectToken("11@boot-A", "boot-B"); !errors.Is(err, embeddinggemma2.ErrNoDevice) {
		t.Fatalf("stale exact ID error = %v", err)
	}
}

// TestMetalAttachedDeviceMemory is the native memory-reporter qualification
// for the system-default Metal attachment. It uses the real probe/reporter
// (not a synthetic describe call), asserts only host-derived bounds, and keeps
// the legacy blank device ID contract. #86's exact 24-GiB figures stay in the
// qualification record rather than being enforced across M4s.
func TestMetalAttachedDeviceMemory(t *testing.T) {
	legacyRows, legacyFailures := embeddinggemma2.Devices(embeddinggemma2.ProbeOptions{IncludeMemory: true})
	if len(legacyFailures) != 0 {
		t.Fatalf("legacy Metal discovery: %+v", legacyFailures)
	}
	var legacyMetal []embeddinggemma2.DeviceInfo
	for _, row := range legacyRows {
		if row.Backend == "metal" {
			legacyMetal = append(legacyMetal, row)
		}
	}
	if len(legacyMetal) != 1 {
		t.Fatalf("legacy Metal discovery returned %d rows, want one system-default row", len(legacyMetal))
	}
	legacyRow := legacyMetal[0]
	if legacyRow.ID != "" || legacyRow.Alias != "" {
		t.Fatalf("legacy Metal row has identity: %+v", legacyRow)
	}
	rows, failures := embeddinggemma2.AllDevices(embeddinggemma2.ProbeOptions{IncludeMemory: true})
	if len(failures) != 0 {
		t.Fatalf("native Metal discovery: %+v", failures)
	}
	var usable *embeddinggemma2.DeviceInfo
	for i := range rows {
		if rows[i].Backend == "metal" && rows[i].ID != "" && rows[i].Alias != "" {
			usable = &rows[i]
			break
		}
	}
	if usable == nil {
		t.Fatalf("native Metal inventory exposes no addressable row: %+v", rows)
	}
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadJSONBytes([]byte(`{"model":{"type":"BPE","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3},"merges":[]}}`))
	if err != nil {
		t.Fatal(err)
	}
	enc, err := embeddinggemma2.NewEncoder(m, tok, nil)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{1, 2, 3}
	if m.Config().VocabSize <= 3 {
		t.Fatalf("tiny fixture vocabulary %d cannot hold the forward input", m.Config().VocabSize)
	}
	cpuVector, err := enc.EmbedIDs(ids)
	if err != nil {
		t.Fatalf("CPU EmbedIDs: %v", err)
	}
	acc, err := enc.UseAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = acc.Close() })
	report, ok := acc.(embeddinggemma2.AttachedDeviceReporter)
	if !ok {
		t.Fatal("legacy attachment cannot report its device")
	}
	attached := report.AttachedDevice()
	if attached.ID != "" || attached.Alias != "" || attached.Backend != "metal" {
		t.Fatalf("legacy attachment identity = %+v; want blank ID/alias", attached)
	}
	if attached.MemoryKind != legacyRow.MemoryKind || attached.Memory.TotalBytes != legacyRow.Memory.TotalBytes {
		t.Fatalf("legacy attachment memory %+v, legacy inventory %+v", attached.Memory, legacyRow.Memory)
	}
	ram, ramErr := physicalRAM()
	if ramErr == nil && ram > 0 {
		if legacyRow.MemoryKind == embeddinggemma2.MemoryShared && legacyRow.Memory.TotalBytes != int64(ram) {
			t.Errorf("legacy inventory total memory %d, want host total %d", legacyRow.Memory.TotalBytes, ram)
		}
		if legacyRow.MemoryKind == embeddinggemma2.MemoryDedicated && legacyRow.Memory.TotalBytes != 0 {
			t.Errorf("dedicated legacy row reported total %d without a verified device probe", legacyRow.Memory.TotalBytes)
		}
	}
	reporter, ok := acc.(embeddinggemma2.DeviceMemoryReporter)
	if !ok {
		t.Fatal("legacy attachment cannot report memory")
	}
	reading, err := reporter.DeviceMemory()
	if err != nil {
		t.Fatal(err)
	}
	if reading.TotalBytes != legacyRow.Memory.TotalBytes || reading.Basis != legacyRow.Memory.Basis ||
		reading.AvailableBytes < 0 || reading.AvailableBytes > reading.TotalBytes {
		t.Fatalf("legacy attached memory %+v, legacy inventory %+v", reading, legacyRow.Memory)
	}
	if ramErr != nil {
		if reading.Basis != embeddinggemma2.MemoryUnknown || reading.AvailableBytes != 0 {
			t.Fatalf("unreadable host RAM should be unknown memory: %+v", reading)
		}
	} else if legacyRow.MemoryKind == embeddinggemma2.MemoryShared {
		if reading.Basis != embeddinggemma2.MemoryEstimated || reading.TotalBytes <= 0 ||
			reading.AvailableBytes < 0 || reading.AvailableBytes > reading.TotalBytes {
			t.Fatalf("shared legacy attached memory = %+v", reading)
		}
	} else if reading.Basis != embeddinggemma2.MemoryUnknown || reading.AvailableBytes != 0 {
		t.Fatalf("dedicated/unknown legacy attached memory = %+v", reading)
	}
	prev := enc.Accelerator()
	if _, err := enc.UseDevice("metal:11@stale-boot", "metal"); !errors.Is(err, embeddinggemma2.ErrNoDevice) {
		t.Fatalf("stale exact-ID error = %v", err)
	}
	if enc.Accelerator() != prev {
		t.Fatal("failed explicit selection changed the working attachment")
	}
	// A refusal leaves the previous attachment usable, not merely equal.
	if _, err := enc.EmbedIDs(ids); err != nil {
		t.Fatalf("forward after refused selection: %v", err)
	}
	// A selection reason survives the public UseDevice call and is not folded
	// into the sibling runtime-unavailable category.
	const ambiguousSel, runtimeSel = "metal-test-ambiguous", "metal-test-runtime"
	embeddinggemma2.RegisterDeviceAccelerator(ambiguousSel, "metal", func(*embeddinggemma2.Model, string) (embeddinggemma2.Accelerator, error) {
		_, err := acquireMetalDevice(func() (*Device, error) {
			return nil, fmt.Errorf("%w: test duplicate registry ID", embeddinggemma2.ErrAmbiguousDevice)
		})
		return nil, err
	})
	embeddinggemma2.RegisterDeviceAccelerator(runtimeSel, "metal", func(*embeddinggemma2.Model, string) (embeddinggemma2.Accelerator, error) {
		_, err := acquireMetalDevice(func() (*Device, error) { return nil, errors.New("framework missing") })
		return nil, err
	})
	if _, err := enc.UseDevice("metal:0", ambiguousSel); !errors.Is(err, embeddinggemma2.ErrAmbiguousDevice) ||
		!errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) || errors.Is(err, embeddinggemma2.ErrBackendUnavailable) {
		t.Fatalf("ambiguous selection error = %v", err)
	}
	if _, err := enc.UseDevice("metal:0", runtimeSel); !errors.Is(err, embeddinggemma2.ErrBackendUnavailable) {
		t.Fatalf("runtime selection error = %v", err)
	}
	if enc.Accelerator() != prev {
		t.Fatal("typed selection failures changed the working attachment")
	}
	if _, err := enc.EmbedIDs(ids); err != nil {
		t.Fatalf("forward after typed refusals: %v", err)
	}
	if err := acc.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := reporter.DeviceMemory(); err == nil {
		t.Fatal("DeviceMemory after Close returned no error")
	}

	selectedRegistry, boot, ok := strings.Cut(strings.TrimPrefix(usable.ID, "metal:"), "@")
	if !ok || boot == "" {
		t.Fatalf("selected ID %q has no boot marker", usable.ID)
	}
	wantRegistry, err := strconv.ParseUint(selectedRegistry, 10, 64)
	if err != nil || wantRegistry == 0 {
		t.Fatalf("selected ID %q has no registry ID: %v", usable.ID, err)
	}
	// The published exact ID and its current-order alias must both attach the
	// same physical device, report the selected inventory row, and run a real
	// forward on that retained handle.
	for _, token := range []string{usable.ID, usable.Alias} {
		explicit, err := enc.UseDevice(token, "metal")
		if err != nil {
			t.Fatalf("explicit %q: %v", token, err)
		}
		t.Cleanup(func() { _ = explicit.Close() })
		explicitReport, ok := explicit.(embeddinggemma2.AttachedDeviceReporter)
		if !ok {
			t.Fatalf("%q attachment cannot report its device", token)
		}
		identity := explicitReport.AttachedDevice()
		if identity.ID != usable.ID || identity.Alias != usable.Alias || identity.Backend != "metal" ||
			identity.MemoryKind != usable.MemoryKind || identity.Memory.TotalBytes != usable.Memory.TotalBytes {
			t.Fatalf("%q attached %+v, inventory %+v", token, identity, *usable)
		}
		a, ok := explicit.(*eg2Accel)
		if !ok || a.d == nil {
			t.Fatalf("%q attachment is not the selected Metal handle", token)
		}
		if a.d.RegistryID() != wantRegistry {
			t.Fatalf("%q attached registry ID %d, inventory %q", token, a.d.RegistryID(), usable.ID)
		}
		explicitMemory, ok := explicit.(embeddinggemma2.DeviceMemoryReporter)
		if !ok {
			t.Fatalf("%q attachment cannot report memory", token)
		}
		memory, err := explicitMemory.DeviceMemory()
		if err != nil {
			t.Fatalf("%q memory: %v", token, err)
		}
		if memory.TotalBytes != usable.Memory.TotalBytes || memory.Basis != usable.Memory.Basis ||
			memory.AvailableBytes < 0 || memory.AvailableBytes > memory.TotalBytes {
			t.Fatalf("%q memory %+v, inventory %+v", token, memory, usable.Memory)
		}
		d0, s0, w0 := a.d.Activity()
		vector, err := enc.EmbedIDs(ids)
		if err != nil {
			t.Fatalf("%q EmbedIDs: %v", token, err)
		}
		d1, s1, w1 := a.d.Activity()
		if d1 <= d0 || s1 <= s0 || w1 <= w0 {
			t.Fatalf("%q activity dispatches %d->%d, submits %d->%d, waits %d->%d; want all to increase",
				token, d0, d1, s0, s1, w0, w1)
		}
		var norm float64
		for _, v := range vector {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%q EmbedIDs returned a non-finite value", token)
			}
			norm += float64(v) * float64(v)
		}
		if math.Abs(norm-1) > 1e-4 {
			t.Fatalf("%q EmbedIDs norm %v, want unit length", token, norm)
		}
		if cosine := eg2Cos(vector, cpuVector); cosine < 0.9999 {
			t.Fatalf("%q EmbedIDs matches CPU at cosine %.9f", token, cosine)
		}
		if err := explicit.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := explicitMemory.DeviceMemory(); err == nil {
			t.Fatalf("%q DeviceMemory after Close returned no error", token)
		}
	}
}

func TestMetalInventoryAndAttachAcquisitionErrors(t *testing.T) {
	for _, tt := range []struct{ native, category error }{
		{gpu.ErrNoDevice, embeddinggemma2.ErrNoDevice},
		{errors.New("framework missing"), embeddinggemma2.ErrBackendUnavailable},
	} {
		open := func() (*Device, error) { return nil, tt.native }
		ops, err := newEG2OpsOn(open, "", nil)
		if ops.d != nil || !errors.Is(err, tt.category) || !errors.Is(err, tt.native) {
			t.Fatalf("attach %v: %+v, %v", tt.native, ops.d, err)
		}
		list := func() ([]gpu.DeviceFact, error) { return nil, tt.native }
		rows, err := probeMetalDevicesOn(list, embeddinggemma2.ProbeOptions{})
		if len(rows) != 0 ||
			tt.category == embeddinggemma2.ErrNoDevice && err != nil ||
			tt.category == embeddinggemma2.ErrBackendUnavailable && (!errors.Is(err, tt.category) || !errors.Is(err, tt.native)) {
			t.Fatalf("inventory %v: rows %+v, error %v", tt.native, rows, err)
		}
	}
}

// #84 scalar-arena sizing and growth regressions.

func eg2GoogleKVHeads() []int {
	// Google geometry: every sixth layer is full attention with one KV head,
	// the other twenty are sliding with two.
	out := make([]int, 24)
	for i := range out {
		out[i] = 2
		if (i+1)%6 == 0 {
			out[i] = 1
		}
	}
	return out
}

func TestEG2ArenaSlotsGoogle(t *testing.T) {
	cfg := embeddinggemma2.Config{Heads: 4, Layers: 24}
	kv := eg2GoogleKVHeads()
	for _, tc := range []struct{ tokens, want int }{
		{1536, 14512}, {1537, 16556}, {2048, 18600}, {8192, 67656},
	} {
		got, err := eg2ArenaSlotsFor(cfg, kv, tc.tokens, false)
		if err != nil {
			t.Fatalf("%d tokens: %v", tc.tokens, err)
		}
		if got != tc.want {
			t.Fatalf("%d tokens: %d slots, want %d", tc.tokens, got, tc.want)
		}
	}
	// The initial arena must not already cover the observed 1,537 boundary, so
	// the growth path is the one exercised there.
	slots, err := eg2ArenaSlotsFor(cfg, kv, 1537, false)
	if err != nil {
		t.Fatal(err)
	}
	if slots <= eg2ArenaSlots {
		t.Fatalf("1,537 needs %d slots, initial arena already holds %d", slots, eg2ArenaSlots)
	}
}

func TestEG2ArenaSlotsKeepLayers(t *testing.T) {
	cfg := embeddinggemma2.Config{Heads: 4, Layers: 24}
	kv := eg2GoogleKVHeads()
	got, err := eg2ArenaSlotsFor(cfg, kv, 1536, true)
	if err != nil {
		t.Fatal(err)
	}
	// Largest interval is the opening plus the first (sliding, two-KV-head)
	// layer: 6 blocks of (19*4 + 5*2) plus 8 + 93.
	if want := 8 + 93 + 6*(19*4+5*2); got != want {
		t.Fatalf("keepLayers interval %d slots, want %d", got, want)
	}
	if got >= 14512 {
		t.Fatalf("keepLayers interval %d should be far below the whole forward", got)
	}
}

func TestEG2ArenaSlotsInvalid(t *testing.T) {
	cfg := embeddinggemma2.Config{Heads: 4, Layers: 24}
	kv := eg2GoogleKVHeads()
	if _, err := eg2ArenaSlotsFor(cfg, kv, 0, false); err == nil {
		t.Fatal("zero rows accepted")
	}
	if _, err := eg2ArenaSlotsFor(cfg, kv, -1, false); err == nil {
		t.Fatal("negative rows accepted")
	}
	if _, err := eg2ArenaSlotsFor(cfg, kv[:23], 128, false); err == nil {
		t.Fatal("short KV slice accepted")
	}
	bad := append([]int(nil), kv...)
	bad[3] = 0
	if _, err := eg2ArenaSlotsFor(cfg, bad, 128, false); err == nil {
		t.Fatal("zero KV heads accepted")
	}
	if _, err := eg2ArenaSlotsFor(cfg, kv, math.MaxInt/2, false); err == nil {
		t.Fatal("addressable-range overflow accepted")
	}
	old := eg2Block
	eg2Block = 0
	defer func() { eg2Block = old }()
	if _, err := eg2ArenaSlotsFor(cfg, kv, 128, false); err == nil {
		t.Fatal("zero block accepted")
	}
}

// TestEG2Metal_growsScalarArena is the red-first caller regression: with a
// one-slot arena the old fixed-capacity guard panicked; the sized path grows
// before encoding and reproduces the normal run's bytes.
func TestEG2Metal_growsScalarArena(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	ids := make([]int, 300)
	for i := range ids {
		ids[i] = (i*7919 + 13) % m.Config().VocabSize
	}
	want, _, err := a.Forward(ids, false)
	if err != nil {
		t.Fatal(err)
	}
	short := make([]int, 17)
	for i := range short {
		short[i] = (i*104729 + 7) % m.Config().VocabSize
	}
	shortWant, _, err := a.Forward(short, false)
	if err != nil {
		t.Fatal(err)
	}
	a.d.ReleaseBuf(a.arena)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	got, _, err := a.Forward(ids, false)
	if err != nil {
		t.Fatalf("grown forward: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("%d values, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("value %d changed after growing the arena", i)
		}
	}
	if a.arena.Len() <= 256 {
		t.Fatalf("arena did not grow: %d bytes", a.arena.Len())
	}
	shortGot, _, err := a.Forward(short, false)
	if err != nil {
		t.Fatalf("short forward after growth: %v", err)
	}
	for i := range shortWant {
		if math.Float32bits(shortGot[i]) != math.Float32bits(shortWant[i]) {
			t.Fatalf("short output %d changed after growth", i)
		}
	}
}

// TestEG2Metal_arenaGrowRefusalKeepsArena proves a refused growth returns an
// error before encoding, keeps the old arena, and a later retry succeeds with
// the ledger back to its starting length.
// TestEG2Metal_longForwardAfterClose is the caller regression for the closed
// path: a long ForwardEmbeds whose arena requirement exceeds the retained one
// must return errTowerClosed, not dereference a released device.
func TestEG2Metal_longForwardAfterClose(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	a := acc.(*eg2Accel)
	// Shrink the retained arena so the long requirement exceeds it after Close.
	a.d.ReleaseBuf(a.arena)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	if err := acc.Close(); err != nil {
		t.Fatal(err)
	}
	H := m.Config().Hidden
	x := make([]float32, 300*H)
	for i := range x {
		x[i] = float32(i%17) / 17
	}
	if _, _, err := a.ForwardEmbeds(x, 300, false); !errors.Is(err, errTowerClosed) {
		t.Fatalf("long ForwardEmbeds after Close: %v, want errTowerClosed", err)
	}
}

func TestEG2Metal_arenaGrowRefusalKeepsArena(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("metal")
	if err != nil {
		t.Fatal(err)
	}
	defer acc.Close()
	a := acc.(*eg2Accel)
	ids := make([]int, 300)
	for i := range ids {
		ids[i] = (i*7919 + 13) % m.Config().VocabSize
	}
	if _, _, err := a.Forward(ids, false); err != nil { // warm rope and scratch
		t.Fatal(err)
	}
	a.d.ReleaseBuf(a.arena)
	a.arena = a.d.NewBufferBytesOn(mem.LineCompanion, 256)
	old := a.arena
	allocs, _ := a.d.LedgerLen()
	eg2ArenaAllocForTest = func(int) (Buffer, bool) { return Buffer{}, false }
	_, _, err = a.Forward(ids, false)
	eg2ArenaAllocForTest = nil
	if err == nil {
		t.Fatal("refused growth did not fail the forward")
	}
	if a.arena.Len() != old.Len() {
		t.Fatalf("refusal replaced the arena: %d bytes, was %d", a.arena.Len(), old.Len())
	}
	if _, _, err := a.Forward(ids, false); err != nil {
		t.Fatalf("retry after a refused growth: %v", err)
	}
	if a.arena.Len() <= old.Len() {
		t.Fatal("retry did not grow the arena")
	}
	if got, _ := a.d.LedgerLen(); got != allocs {
		t.Fatalf("ledger length %d after grow, want %d", got, allocs)
	}
}

func TestEG2AttentionResolution(t *testing.T) {
	hard := errors.New("command failed")
	for _, tc := range []struct {
		name    string
		request eg2AttentionRequest
		shape   bool
		setup   error
		calls   int
		profile string
		err     error
	}{
		{"default eligible", eg2AttentionDefault, true, nil, 1, eg2MatrixProfile, nil},
		{"default geometry decline", eg2AttentionDefault, false, nil, 0, eg2StockProfile, nil},
		{"default capability decline", eg2AttentionDefault, true, embeddinggemma2.ErrUnsupportedSelector, 1, eg2StockProfile, nil},
		{"default native failure", eg2AttentionDefault, true, hard, 1, "", hard},
		{"default OOM", eg2AttentionDefault, true, embeddinggemma2.ErrDeviceOutOfMemory, 1, "", embeddinggemma2.ErrDeviceOutOfMemory},
		{"forced matrix decline", eg2AttentionMatrix, true, embeddinggemma2.ErrUnsupportedSelector, 1, "", embeddinggemma2.ErrUnsupportedSelector},
		{"forced matrix geometry decline", eg2AttentionMatrix, false, nil, 0, "", embeddinggemma2.ErrUnsupportedSelector},
		{"forced matrix eligible", eg2AttentionMatrix, true, nil, 1, eg2MatrixProfile, nil},
		{"forced stock bypasses setup", eg2AttentionStock, true, hard, 0, eg2StockProfile, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			got, err := resolveEG2Attention(tc.request, tc.shape, func() error {
				calls++
				return tc.setup
			})
			if got != tc.profile || !errors.Is(err, tc.err) || calls != tc.calls {
				t.Fatalf("profile=%q err=%v setup calls=%d", got, err, calls)
			}
		})
	}
}

func TestEG2MatrixGeometry(t *testing.T) {
	w := embeddinggemma2.Weights{
		Config: embeddinggemma2.Config{Heads: 4, Layers: 24, SlidingWindow: 512},
		Layers: make([]embeddinggemma2.LayerWeights, 24),
	}
	for i := range w.Layers {
		w.Layers[i] = embeddinggemma2.LayerWeights{HeadDim: 256, KVHeads: 2}
		if i%6 == 5 {
			w.Layers[i] = embeddinggemma2.LayerWeights{Full: true, HeadDim: 512, KVHeads: 1}
		}
	}
	if !eg2MatrixGeometry(w) {
		t.Fatal("eligible graph declined")
	}
	for _, change := range []func(*embeddinggemma2.Weights){
		func(w *embeddinggemma2.Weights) { w.Config.Heads = 2 },
		func(w *embeddinggemma2.Weights) { w.Config.SlidingWindow = 511 },
		func(w *embeddinggemma2.Weights) { w.Layers = w.Layers[:23] },
		func(w *embeddinggemma2.Weights) { w.Layers[5].Full = false },
		func(w *embeddinggemma2.Weights) { w.Layers[11].HeadDim = 256 },
		func(w *embeddinggemma2.Weights) { w.Layers[0].KVHeads = 1 },
	} {
		copy := w
		copy.Layers = append([]embeddinggemma2.LayerWeights(nil), w.Layers...)
		change(&copy)
		if eg2MatrixGeometry(copy) {
			t.Fatal("unsupported graph accepted")
		}
	}
}

func TestEG2MatrixLengthAdmission(t *testing.T) {
	for _, rawCap := range []int{0, 4096, 262144} {
		limit := embeddinggemma2.MaxTokens
		if rawCap > 0 {
			limit = min(limit, rawCap)
		}
		// Only the rejection paths below use Forward. A zero Device lets
		// the guard run while making any unintended device work fail.
		a := &eg2Accel{
			eg2Ops:  eg2Ops{d: &Device{}},
			cfg:     embeddinggemma2.Config{Hidden: 1, MaxPositions: rawCap},
			profile: eg2MatrixProfile,
		}
		est, err := a.BatchMemory([]int{limit})
		if err != nil || est.Quality != embeddinggemma2.BatchMemoryUnknown {
			t.Fatalf("max %d memory admission: %+v, %v", limit, est, err)
		}
		_, err = a.BatchMemory([]int{limit + 1})
		_, _, idErr := a.Forward(make([]int, limit+1), false)
		_, _, rowErr := a.ForwardEmbeds(make([]float32, limit+1), limit+1, false)
		for _, err := range []error{err, idErr, rowErr} {
			if !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
				t.Fatalf("max+1 at %d: %v", limit, err)
			}
		}
		if _, _, err := a.ForwardEmbeds(nil, limit+1, false); err == nil || errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
			t.Fatalf("malformed prepared input: %v", err)
		}
	}
}

func TestEG2MatrixCanaryOracle(t *testing.T) {
	for _, dims := range [][2]int{{512, 1}, {256, 2}} {
		hd, kv := dims[0], dims[1]
		rows := eg2ProbeRows + eg2ProbeTail
		q, k, v := make([]float32, rows*4*hd), make([]float32, rows*kv*hd), make([]float32, rows*kv*hd)
		eg2ProbeInputs(q, k, v, hd, kv)
		out := make([]float32, len(q))
		for i := eg2ProbeRows * 4 * hd; i < len(out); i++ {
			out[i] = eg2ProbeMark
		}
		// Direct FP64 Q·K, softmax and P·V, independent of the tagged-logit shortcut.
		for row := range eg2ProbeRows {
			for h := range 4 {
				group := h / (4 / kv)
				var sum float64
				var p [eg2ProbeRows]float64
				for key := range eg2ProbeRows {
					var dot float64
					for c := range hd {
						dot += float64(q[(row*4+h)*hd+c]) * float64(k[(key*kv+group)*hd+c])
					}
					p[key] = math.Exp(dot)
					sum += p[key]
				}
				for c := range hd {
					var x float64
					for key := range eg2ProbeRows {
						x += p[key] * float64(v[(key*kv+group)*hd+c])
					}
					out[(row*4+h)*hd+c] = float32(x / sum)
				}
			}
		}
		if err := eg2ProbeOutput(out, v, hd, kv); err != nil {
			t.Fatal(err)
		}
		out[0], out[1] = out[1], out[0]
		if err := eg2ProbeOutput(out, v, hd, kv); !errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) {
			t.Fatalf("permuted words: %v", err)
		}
		out[len(out)-1] = 0
		if err := eg2ProbeOutput(out, v, hd, kv); err == nil || errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) {
			t.Fatalf("guard write with a simultaneous numerical mismatch must fail hard: %v", err)
		}
	}
}

func TestEG2MetalMatrixNativeCanary(t *testing.T) {
	ops, err := newEG2Ops("", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.closeDevice()
	if !ops.d.HasUnifiedMemory() || !ops.d.SupportsApple7() {
		t.Skip("Apple7+ with unified memory required")
	}
	beforeBuffers, beforeObjects := ops.d.LedgerLen()
	_, err = ops.newEG2MatrixPipes()
	if err != nil {
		t.Fatal(err)
	}
	if buffers, objects := ops.d.LedgerLen(); buffers != beforeBuffers || objects != beforeObjects+2 || ops.slot != 0 {
		t.Fatalf("temporary canary resources retained: buffers %d/%d objects %d/%d scalar slot %d", buffers, beforeBuffers, objects, beforeObjects+2, ops.slot)
	}
}

func TestEG2MetalStockSelector(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, selector := range []string{"metal", "metal-eg2-stock"} {
		a, err := m.NewAccelerator(selector)
		if err != nil {
			t.Fatal(err)
		}
		defer a.Close()
		if reporter, ok := a.(embeddinggemma2.AttachedProfileReporter); !ok || reporter.AttachedProfile() != eg2StockProfile || a.Name() != "metal" {
			t.Fatalf("%s: %s has no resolved stock profile", selector, a.Name())
		}
	}
	if a, err := m.NewAccelerator("metal-eg2-combined"); !errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) {
		if a != nil {
			a.Close()
		}
		t.Fatalf("strict matrix selector accepted tiny geometry: %v", err)
	}
}

func TestEG2StagedFFNEligibility(t *testing.T) {
	c := embeddinggemma2.Config{Hidden: 512, Intermediate: 2048, Layers: 24}
	if !eg2StagedFFNEligible("Apple M4", "applegpu_g16g", eg2MatrixProfile, c) {
		t.Fatal("measured device, graph and profile rejected")
	}
	for _, test := range []struct {
		name, device, arch, profile string
		cfg                         embeddinggemma2.Config
	}{
		{"unknown arch", "Apple M4", "", eg2MatrixProfile, c},
		{"other arch", "Apple M4", "applegpu_g15g", eg2MatrixProfile, c},
		{"other model", "Apple M4 Pro", "applegpu_g16g", eg2MatrixProfile, c},
		{"stock", "Apple M4", "applegpu_g16g", eg2StockProfile, c},
		{"other graph", "Apple M4", "applegpu_g16g", eg2MatrixProfile, embeddinggemma2.Config{Hidden: 512, Intermediate: 4096, Layers: 24}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if eg2StagedFFNEligible(test.device, test.arch, test.profile, test.cfg) {
				t.Fatal("unqualified graph or device admitted")
			}
		})
	}
	for _, tt := range []struct {
		rows int
		want bool
	}{{1, false}, {30, false}, {31, true}, {36, true}, {879, true}, {2042, true}, {2043, false}, {8192, false}} {
		if got := eg2StagedFFNRows(tt.rows); got != tt.want {
			t.Errorf("rows %d: staged=%t, want %t", tt.rows, got, tt.want)
		}
	}
}

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
var eg2BatchRMSFixture = flag.String("eg2-batch-test-fixture", "", "pinned prepared-219-ids.jsonl for the RMS replacement trace")

// eg2ReadRMSFixture reads the frozen prepared-ID corpus for the replacement trace.
func eg2ReadRMSFixture(t *testing.T) [][]int {
	t.Helper()
	f, err := os.Open(*eg2BatchRMSFixture)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var ids [][]int
	positions := 0
	dec := json.NewDecoder(f)
	for {
		var row struct {
			IDs []int `json:"ids"`
		}
		if err := dec.Decode(&row); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		positions += len(row.IDs)
		ids = append(ids, row.IDs)
	}
	if len(ids) != 219 || positions != 110157 {
		t.Fatalf("fixture: %d documents, %d positions", len(ids), positions)
	}
	return ids
}

// TestEG2BatchMetalRMSReplacementTrace is the off-clock replacement witness for the
// fused residual sites: one 219-document B4 replay on the explicit batch attachment
// must show 2,640 plain + 1,320 scaled fused RMS calls, 1,968 GeGLU up, 1,320 PLE
// activations, 55 commands and 41,044 total dispatches (the released baseline is
// 45,004), with the published vectors.
func TestEG2BatchMetalRMSReplacementTrace(t *testing.T) {
	if *eg2BatchRMSFixture == "" {
		t.Skip("set -eg2-batch-test-fixture")
	}
	ids := eg2ReadRMSFixture(t)
	groups := make([][][]int, 0, 55)
	for i := 0; i < len(ids); i += 4 {
		end := min(i+4, len(ids))
		groups = append(groups, ids[i:end])
	}
	if len(groups) != 55 || len(groups[54]) != 3 {
		t.Fatalf("wrong groups: %d", len(groups))
	}
	for _, tc := range []struct {
		name, path string
		load       func(string) (*embeddinggemma2.Encoder, error)
		raw, norm  string
	}{
		{"q8", *eg2BatchGGUFTest, func(path string) (*embeddinggemma2.Encoder, error) {
			return embeddinggemma2.LoadGGUFEncoder(path, nil)
		}, "754952f8d9e8e54d5754b9b5764bfcb2d6fd3db0f743b0c9f51c9ec86c1c45d7",
			"ed74cd5b2f46d3add8f337ab23a0b074a4acec766a0e2a8391bb70df5d8bfd9e"},
		{"mlx", *eg2BatchMLXTest, embeddinggemma2.LoadEncoder,
			"fab76bbfcdb18da676a47a8c25e50918588d6dc53d2a7f8dd6bfb71ee17e8e6d",
			"5fb18b80ed39e504963e9e10392dba644089442f6e6b9db8a0a8fe5fdf34bc2b"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.path == "" {
				t.Skip("set the checkpoint flag")
			}
			enc, err := tc.load(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			acc, err := enc.UseAccelerator("metal-eg2-batch")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = acc.Close() })
			a := acc.(*eg2BatchAccel)
			if a.p.rmsAdd == (Pipeline{}) || a.p.rmsAddScale == (Pipeline{}) {
				t.Fatal("fused RMS pipelines missing")
			}
			var out [][]float32
			c := eg2RouteCounts(a.eg2Accel, func() {
				for _, g := range groups {
					v, err := enc.EmbedIDsBatch(g)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, v...)
				}
			})
			if c.Total != 41044 || c.RMSAdd != 2640 || c.RMSAddScale != 1320 ||
				c.GateUp != 1968 || c.Fused != 1968 || c.GeluMul != 0 || c.PLE != 1320 ||
				c.Add != 0 || c.AddScale != 0 || c.Commands != 55 {
				t.Fatalf("replacement census %+v", c)
			}
			if len(out) != 219 {
				t.Fatalf("%d documents", len(out))
			}
			if got := eg2DigestF32(out, false); got != tc.raw {
				t.Fatalf("768 digest %s, want %s", got, tc.raw)
			}
			if got := eg2DigestF32(out, true); got != tc.norm {
				t.Fatalf("512 digest %s, want %s", got, tc.norm)
			}
		})
	}
}

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

// eg2RouteCensus is one call's dispatch census across the product pipelines.
type eg2RouteCensus struct {
	GateUp, Fused, GeluMul, PLE, RMSAdd, RMSAddScale, Add, AddScale, Total, Commands int
}

// eg2RouteCounts traces dispatches on one Metal attachment's pipelines.
func eg2RouteCounts(ops *eg2Accel, call func()) eg2RouteCensus {
	var c eg2RouteCensus
	gpu.TraceForTest = func(ev gpu.TraceEvent) {
		switch ev.Op {
		case gpu.TraceDispatch:
			c.Total++
			switch ev.Pipe {
			case ops.stagedFFN:
				c.GateUp++
			case ops.stagedFFNGeGLU:
				c.Fused++
			case ops.p.geluMul:
				c.GeluMul++
			case ops.p.geluMulPLE:
				c.PLE++
			case ops.p.rmsAdd:
				c.RMSAdd++
			case ops.p.rmsAddScale:
				c.RMSAddScale++
			case ops.p.add:
				c.Add++
			case ops.p.addScale:
				c.AddScale++
			}
		case gpu.TraceCommit:
			c.Commands++
		}
	}
	defer func() { gpu.TraceForTest = nil }()
	call()
	return c
}

// eg2BatchGeGLURoutes checks the product route on the explicit batch attachment:
// a fully staged plan fuses (24 gate + 24 fused up, no standalone activation), a
// mixed plan keeps the released gate/up + 24 activations, and both explicit
// singleton APIs take the one-document plan while keepLayers stays on the legacy
// no-plan route.
func eg2BatchGeGLURoutes(t *testing.T, enc *embeddinggemma2.Encoder, a *eg2BatchAccel, in [][]int, refs [][]float32) {
	t.Helper()
	if a.stagedFFN == (Pipeline{}) || a.stagedFFNGeGLU == (Pipeline{}) || a.p.rmsAdd == (Pipeline{}) || a.p.rmsAddScale == (Pipeline{}) {
		t.Fatalf("batch attachment lacks staged/RMS pipelines: gate/up=%v fused=%v rms=%v/%v",
			a.stagedFFN != (Pipeline{}), a.stagedFFNGeGLU != (Pipeline{}),
			a.p.rmsAdd != (Pipeline{}), a.p.rmsAddScale != (Pipeline{}))
	}
	if a.stagedFFNGeGLU.ThreadExecutionWidth() != 32 ||
		a.stagedFFNGeGLU.MaxTotalThreadsPerThreadgroup() < eg2StagedThreads ||
		a.stagedFFNGeGLU.StaticThreadgroupMemoryLength() != eg2StagedTGBytes {
		t.Fatalf("fused pipeline limits: width=%d threads=%d tg=%d",
			a.stagedFFNGeGLU.ThreadExecutionWidth(), a.stagedFFNGeGLU.MaxTotalThreadsPerThreadgroup(),
			a.stagedFFNGeGLU.StaticThreadgroupMemoryLength())
	}
	stockTG := a.vit.RMSNorm.StaticThreadgroupMemoryLength()
	if stockTG <= 0 || a.p.rmsAdd.StaticThreadgroupMemoryLength() <= 0 || a.p.rmsAddScale.StaticThreadgroupMemoryLength() <= 0 ||
		a.p.rmsAdd.StaticThreadgroupMemoryLength() > stockTG || a.p.rmsAddScale.StaticThreadgroupMemoryLength() > stockTG {
		t.Fatalf("RMS static TG plain/scaled %d/%d exceeds stock %d", a.p.rmsAdd.StaticThreadgroupMemoryLength(),
			a.p.rmsAddScale.StaticThreadgroupMemoryLength(), stockTG)
	}
	check := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d words, want %d", label, len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: first differing word %d", label, i)
			}
		}
	}
	runPlan := func(label string, docs [][]int, refIdx []int, wantGateUp, wantFused, wantGelu int) {
		t.Helper()
		var got [][]float32
		c := eg2RouteCounts(a.eg2Accel, func() {
			var err error
			got, err = enc.EmbedIDsBatch(docs)
			if err != nil {
				t.Fatal(err)
			}
		})
		if c.GateUp != wantGateUp || c.Fused != wantFused || c.GeluMul != wantGelu || c.PLE != 24 ||
			c.RMSAdd != 48 || c.RMSAddScale != 24 || c.Add != 0 || c.AddScale != 0 || c.Commands != 1 {
			t.Fatalf("%s: census %+v, want gate/up=%d fused=%d gelu=%d rms=48/24 add=0/0 commands=1",
				label, c, wantGateUp, wantFused, wantGelu)
		}
		for i := range refIdx {
			check(fmt.Sprintf("%s doc %d", label, i), got[i], refs[refIdx[i]])
		}
	}
	// in: [0]=30, [1]=31, [3]=524, [6]=2042, [7]=2043.
	runPlan("fully staged [31 524]", [][]int{in[1], in[3]}, []int{1, 3}, 24, 24, 0)
	runPlan("mixed [30 31]", [][]int{in[0], in[1]}, []int{0, 1}, 48, 0, 24)
	runPlan("staged [31]", [][]int{in[1]}, []int{1}, 24, 24, 0)
	runPlan("staged [2042]", [][]int{in[6]}, []int{6}, 24, 24, 0)
	runPlan("unstaged [2043]", [][]int{in[7]}, []int{7}, 0, 0, 24)
	runPlan("unstaged [30]", [][]int{in[0]}, []int{0}, 0, 0, 24)
	// Explicit singleton EmbedIDs takes the same one-document plan on this selector.
	var one []float32
	c := eg2RouteCounts(a.eg2Accel, func() {
		v, err := enc.EmbedIDs(in[1])
		if err != nil {
			t.Fatal(err)
		}
		one = v
	})
	if c.GateUp != 24 || c.Fused != 24 || c.GeluMul != 0 || c.PLE != 24 ||
		c.RMSAdd != 48 || c.RMSAddScale != 24 || c.Add != 0 || c.AddScale != 0 || c.Commands != 1 {
		t.Fatalf("explicit singleton: %+v", c)
	}
	check("explicit singleton", one, refs[1])
	// keepLayers enters with batch=nil and must keep the released no-plan route.
	kc := eg2RouteCounts(a.eg2Accel, func() {
		if _, _, err := a.Forward(in[1], true); err != nil {
			t.Fatal(err)
		}
	})
	if kc.Fused != 0 || kc.GeluMul != 24 || kc.PLE != 24 || kc.RMSAdd != 0 || kc.RMSAddScale != 0 ||
		kc.Add != 48 || kc.AddScale != 24 {
		t.Fatalf("keepLayers route: %+v", kc)
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
	legacyOps, ok := legacy.(*eg2Accel)
	if !ok {
		t.Fatalf("legacy selector returned %T", legacy)
	}
	lc := eg2RouteCounts(legacyOps, func() {
		if _, err := enc.EmbedIDs(in[1]); err != nil {
			t.Fatal(err)
		}
	})
	if lc.Fused != 0 || lc.GeluMul != 24 || lc.PLE != 24 || lc.RMSAdd != 0 || lc.RMSAddScale != 0 ||
		lc.Add != 48 || lc.AddScale != 24 || lc.Commands != 1 || legacyOps.p.rmsAdd != (Pipeline{}) {
		t.Fatalf("legacy fallback: %+v rms=%v", lc, legacyOps.p.rmsAdd != (Pipeline{}))
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
	// Refuse the last new RoPE buffer after the rest of the workspace
	// exists, then retry via the same live encoder and its real weights.
	beforeRetry := eg2AllocatedBytes(a.d)
	ropeAllocations := 0
	failRoPE := func(kind eg2BatchBufferKind, size int, data []float32) (Buffer, bool) {
		if kind == eg2BatchRoPE {
			ropeAllocations++
			if ropeAllocations == 4 {
				return Buffer{}, false
			}
		}
		return a.allocateBatchBuffer(kind, size, data)
	}
	out, err := a.forwardBatch([][]int{in[1], in[3]}, failRoPE)
	if out != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) ||
		ropeAllocations != 4 || eg2AllocatedBytes(a.d) != beforeRetry {
		t.Fatalf("pre-submit RoPE refusal: outputs %d, allocated %d, error %v", len(out), eg2AllocatedBytes(a.d), err)
	}
	retried, err := enc.EmbedIDsBatch([][]int{in[1]})
	if err != nil {
		t.Fatalf("public retry after rollback: %v", err)
	}
	check("public retry after rollback", retried[0], refs[1])
	eg2BatchGeGLURoutes(t, enc, a, in, refs)
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
	// One packed call takes 48 plain + 24 scaled fused residual dispatches and
	// 24 PLE activations; the released norm+add pairs are gone. The exact total
	// is 4 + 24*(16 + 3*documents + 2*FFN_runs + standalone_GeGLU_per_layer):
	// adjacent staged documents merge into one run while the combined rows stay
	// in [31,2042], and any unstaged document adds a run and a standalone GeGLU.
	for _, tc := range []struct {
		order      []int
		dispatches int
	}{
		{[]int{3, 4, 5, 1}, 772}, // 2 staged runs
		{[]int{1, 5, 4, 3}, 772}, // 2 staged runs
		{[]int{3, 4, 5, 2}, 772}, // 2 staged runs
		{[]int{3, 4}, 580},       // 1 staged run
		{[]int{5, 1}, 580},       // 1 staged run
		{[]int{3, 3, 1}, 652},    // 1 staged run, 3 documents
		{[]int{0, 1}, 652},       // mixed: 1 staged + 1 unstaged run, standalone GeGLU
		{[]int{1, 0}, 652},       // mixed: 1 staged + 1 unstaged run, standalone GeGLU
		{[]int{1, 6}, 628},       // 2 staged runs (2042 + 31 exceeds 2042 rows)
		{[]int{6, 1}, 628},       // 2 staged runs
		{[]int{1, 7}, 652},       // mixed: staged + unstaged run, standalone GeGLU
		{[]int{7, 1}, 652},       // mixed: staged + unstaged run, standalone GeGLU
	} {
		order := tc.order
		group := make([][]int, len(order))
		for i, idx := range order {
			group[i] = in[idx]
		}
		before, submits0, waits0 := a.d.Activity()
		var got [][]float32
		c := eg2RouteCounts(a.eg2Accel, func() {
			var err error
			got, err = enc.EmbedIDsBatch(group)
			if err != nil {
				t.Fatal(err)
			}
		})
		after, submits1, waits1 := a.d.Activity()
		if c.RMSAdd != 48 || c.RMSAddScale != 24 || c.PLE != 24 ||
			c.Add != 0 || c.AddScale != 0 || c.Commands != 1 {
			t.Fatalf("packed call %v census %+v", order, c)
		}
		if submits1-submits0 != 1 || waits1-waits0 != 1 || after-before != int64(tc.dispatches) {
			t.Fatalf("packed call %v activity: %d dispatches, want %d, %d submits, %d waits, census %+v",
				order, after-before, tc.dispatches, submits1-submits0, waits1-waits0, c)
		}
		raw, err := a.ForwardBatch(group)
		if err != nil {
			t.Fatal(err)
		}
		for i, idx := range order {
			check("projected rows", raw[i], rawRefs[idx])
			check("pooled 768", got[i], refs[idx])
			check("normalized 512", eg2Normalized512(got[i]), eg2Normalized512(refs[idx]))
		}
	}
	for _, group := range [][][]int{{in[0], in[8]}, {in[0], in[1], in[2], in[0], in[1]}} {
		if out, err := enc.EmbedIDsBatch(group); out != nil || !errors.Is(err, embeddinggemma2.ErrUnsupportedBatchShape) {
			t.Fatalf("oversize batch produced %d outputs, error %v", len(out), err)
		}
	}
}

func eg2Normalized512(v []float32) []float32 {
	v = v[:512]
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out
}

func TestEG2BatchSelectorRejectsTiny(t *testing.T) {
	m, err := embeddinggemma2.Load(eg2TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	acc, err := m.NewAccelerator("metal-eg2-batch")
	if acc != nil {
		_ = acc.Close()
	}
	if acc != nil || !errors.Is(err, embeddinggemma2.ErrUnsupportedSelector) {
		t.Fatalf("batch selector on the tiny graph: %v, %v", acc, err)
	}
}

var eg2MarkdownEval = flag.Bool("eg2-markdown-eval", false, "run the labeled Markdown fixture on pinned local checkpoints")

func TestEG2MetalMarkdownEval(t *testing.T) {
	if !*eg2MarkdownEval {
		t.Skip("enable -eg2-markdown-eval with both local checkpoint flags")
	}
	if *eg2BatchGGUFTest == "" || *eg2BatchMLXTest == "" {
		t.Fatal("set both -eg2-batch-test-gguf and -eg2-batch-test-mlx")
	}
	fixture, err := markdowneval.Load()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("fixture v%d corpus_sha256=%s", fixture.Version, fixture.CorpusSHA)
	prompts := map[string]string{
		"query":    "task: search result | query: ",
		"document": "title: none | text: ",
	}
	for _, model := range []struct {
		kind, path string
		directory  bool
		load       func(string) (*embeddinggemma2.Encoder, error)
	}{
		{"q8_0", *eg2BatchGGUFTest, false, func(path string) (*embeddinggemma2.Encoder, error) {
			return embeddinggemma2.LoadGGUFEncoder(path, prompts)
		}},
		{"mlx8", *eg2BatchMLXTest, true, embeddinggemma2.LoadEncoder},
	} {
		t.Run(model.kind, func(t *testing.T) {
			pin := fixture.Checkpoints[model.kind]
			verifyMarkdownModel(t, model.path, model.directory, pin)
			enc, load := markdownMeasure(t, func() (*embeddinggemma2.Encoder, error) { return model.load(model.path) })
			for _, role := range []string{"document", "query"} {
				if got, err := enc.PromptText(role); err != nil || got != prompts[role] {
					t.Fatalf("%s prompt %q, want %q: %v", role, got, prompts[role], err)
				}
			}
			queries := make([]markdowneval.Passage, len(fixture.Queries))
			for i, q := range fixture.Queries {
				queries[i] = markdowneval.Passage{ID: q.ID, Text: q.Text}
			}
			docIDs, docPrep := markdownPrepare(t, enc, fixture.Passages, pin.Documents, "document")
			queryIDs, queryPrep := markdownPrepare(t, enc, queries, pin.Queries, "query")
			const selector = "metal-eg2-batch"
			acc, attach := markdownMeasure(t, func() (embeddinggemma2.Accelerator, error) {
				return enc.UseAccelerator(selector)
			})
			t.Cleanup(func() {
				if err := acc.Close(); err != nil {
					t.Error(err)
				}
			})
			a, ok := acc.(*eg2BatchAccel)
			if !ok || acc.Name() != "metal" || !a.AttachedDevice().TextBatching || a.AttachedProfile() != eg2MatrixProfile {
				t.Fatalf("%s did not attach the combined Metal batch route: %T (%s)", selector, acc, acc.Name())
			}
			dev := a.AttachedDevice()
			t.Logf("selector=%s backend=%s device=%q architecture=%q profile=%s text_batching=%t",
				selector, dev.Backend, dev.Name, a.d.ArchitectureName(), a.AttachedProfile(), dev.TextBatching)

			one, singleton := markdownSingletons(t, enc, docIDs)
			query, queryTime := markdownSingletons(t, enc, queryIDs)
			order := make([]int, len(docIDs))
			for i := range order {
				order[i] = i
			}
			packed, batch := markdownBatches(t, enc, a, docIDs, fixture.Groups, order, "packed")
			rotated := append(slices.Clone(order[1:]), order[0])
			reordered, _ := markdownBatches(t, enc, a, docIDs, fixture.Groups, rotated, "reordered")
			repeated, _ := markdownBatches(t, enc, a, docIDs, fixture.Groups, order, "repeat")
			for i, passage := range fixture.Passages {
				for _, run := range []struct {
					name string
					rows [][]float32
				}{{"packed", packed}, {"reordered", reordered}, {"repeat", repeated}} {
					markdownExact(t, passage.ID, run.name, one[i], run.rows[i])
				}
			}
			t.Logf("equal document vectors at both widths across singleton, packed, reorder and repeat sha768=%s sha512=%s",
				eg2DigestF32(one, false), eg2DigestF32(one, true))
			for _, phase := range []struct {
				name string
				data markdownObservation
			}{
				{"load", load}, {"attach", attach}, {"prepare-documents", docPrep}, {"prepare-queries", queryPrep},
				{"embed-singletons", singleton}, {"embed-batches", batch}, {"embed-queries", queryTime},
			} {
				t.Logf("phase=%s wall_ns=%d go_total_alloc_delta_bytes=%d", phase.name, phase.data.wall.Nanoseconds(), phase.data.alloc)
			}
			markdownRank(t, fixture, one, query)
		})
	}
}

func verifyMarkdownModel(t *testing.T, path string, directory bool, pin markdowneval.Checkpoint) {
	t.Helper()
	t.Logf("model=%s revision=%s", pin.Repo, pin.Revision)
	for _, blob := range pin.Files {
		filename := path
		if directory {
			filename = filepath.Join(path, blob.Name)
		}
		got, err := markdownFileDigest(filename, blob.Size)
		if err != nil || got != blob.SHA256 {
			t.Fatalf("%s: SHA-256 %s, want %s: %v", blob.Name, got, blob.SHA256, err)
		}
		t.Logf("checkpoint_file=%s bytes=%d sha256=%s", blob.Name, blob.Size, got)
	}
}

func markdownFileDigest(path string, size int64) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return "", err
	}
	if !fi.Mode().IsRegular() || fi.Size() != size {
		return "", fmt.Errorf("expected a regular file of %d bytes", size)
	}
	h := sha256.New()
	if n, err := io.Copy(h, f); err != nil || n != size {
		return "", fmt.Errorf("hashed %d bytes, want %d: %v", n, size, err)
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

type markdownObservation struct {
	wall  time.Duration
	alloc uint64
}

func markdownMeasure[T any](t *testing.T, f func() (T, error)) (T, markdownObservation) {
	t.Helper()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	v, err := f()
	wall := time.Since(start)
	runtime.ReadMemStats(&after)
	if err != nil {
		t.Fatal(err)
	}
	return v, markdownObservation{wall: wall, alloc: after.TotalAlloc - before.TotalAlloc}
}

func markdownPrepare(t *testing.T, enc *embeddinggemma2.Encoder, texts []markdowneval.Passage, pins []markdowneval.InputPin, role string) ([][]int, markdownObservation) {
	t.Helper()
	rows, observed := markdownMeasure(t, func() ([][]int, error) {
		rows := make([][]int, len(texts))
		for i, text := range texts {
			var err error
			rows[i], err = enc.Tokenize(text.Text, role)
			if err != nil {
				return nil, fmt.Errorf("%s %s: %w", role, text.ID, err)
			}
		}
		return rows, nil
	})
	for i, row := range rows {
		digest := markdowneval.IDsDigest(row)
		p := pins[i]
		if p.ID != texts[i].ID || p.Tokens != len(row) || p.SHA256 != digest {
			t.Fatalf("%s %s: prepared %d IDs with SHA-256 %s, want %+v", role, texts[i].ID, len(row), digest, p)
		}
		t.Logf("prepared role=%s id=%s tokens=%d ids_sha256=%s", role, p.ID, p.Tokens, digest)
	}
	return rows, observed
}

func markdownSingletons(t *testing.T, enc *embeddinggemma2.Encoder, ids [][]int) ([][]float32, markdownObservation) {
	t.Helper()
	return markdownMeasure(t, func() ([][]float32, error) {
		out := make([][]float32, len(ids))
		for i, row := range ids {
			var err error
			out[i], err = enc.EmbedIDs(row)
			if err != nil {
				return nil, fmt.Errorf("singleton %d: %w", i, err)
			}
		}
		return out, nil
	})
}

func markdownBatches(t *testing.T, enc *embeddinggemma2.Encoder, a *eg2BatchAccel, ids [][]int, groups [][]string, order []int, layout string) ([][]float32, markdownObservation) {
	t.Helper()
	out := make([][]float32, len(ids))
	var observed markdownObservation
	start := 0
	for group, members := range groups {
		indexes := order[start : start+len(members)]
		batch, lengths := make([][]int, len(indexes)), make([]int, len(indexes))
		rows := 0
		for i, index := range indexes {
			batch[i] = ids[index]
			lengths[i] = len(batch[i])
			rows += lengths[i]
		}
		estimate, err := enc.BatchMemory(lengths)
		if err != nil || estimate.Quality == embeddinggemma2.BatchMemoryUnknown {
			t.Fatalf("%s group %d: BatchMemory(%v): %+v %v", layout, group, lengths, estimate, err)
		}
		_, s0, w0 := a.d.Activity()
		got, m := markdownMeasure(t, func() ([][]float32, error) { return enc.EmbedIDsBatch(batch) })
		_, s1, w1 := a.d.Activity()
		if len(got) != len(indexes) || s1-s0 != 1 || w1-w0 != 1 {
			t.Fatalf("%s group %d: outputs=%d submits=%d waits=%d", layout, group, len(got), s1-s0, w1-w0)
		}
		for i, index := range indexes {
			out[index] = got[i]
		}
		t.Logf("layout=%s group=%d indexes=%v lengths=%v tokens=%d submits=1 waits=1 estimate=%s additional_peak_bytes=%d resident_weight_bytes=%d",
			layout, group, indexes, lengths, rows, estimate.Quality, estimate.AdditionalPeakBytes, estimate.ResidentWeightBytes)
		start += len(members)
		observed.wall += m.wall
		observed.alloc += m.alloc
	}
	return out, observed
}

func markdownExact(t *testing.T, id, layout string, want, got []float32) {
	t.Helper()
	for _, v := range [][]float32{want, got} {
		if err := markdowneval.CheckVector(v, 768); err != nil {
			t.Fatalf("%s %s: %v", id, layout, err)
		}
	}
	for _, pair := range [][2][]float32{{want, got}, {eg2Normalized512(want), eg2Normalized512(got)}} {
		width := len(pair[0])
		for _, v := range pair {
			if err := markdowneval.CheckVector(v, width); err != nil {
				t.Fatalf("%s %s width %d: %v", id, layout, width, err)
			}
		}
		for j, f := range pair[0] {
			if math.Float32bits(f) != math.Float32bits(pair[1][j]) {
				t.Fatalf("%s %s width %d word %d: got %08x, want %08x", id, layout, width, j, math.Float32bits(pair[1][j]), math.Float32bits(f))
			}
		}
	}
}

func markdownRank(t *testing.T, fixture markdowneval.Fixture, docs, queries [][]float32) {
	t.Helper()
	for _, width := range []int{768, 512} {
		ranks, err := fixture.Rank(docs, queries, width)
		if err != nil {
			t.Fatal(err)
		}
		for _, rank := range ranks {
			data, err := json.Marshal(rank)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("labeled_ranking width=%d %s", width, data)
		}
		if width == 768 {
			normalize := func(vectors [][]float32) [][]float32 {
				result := make([][]float32, len(vectors))
				for i, row := range vectors {
					result[i] = eg2Normalized512(row)
				}
				return result
			}
			docs, queries = normalize(docs), normalize(queries)
		}
	}
}
