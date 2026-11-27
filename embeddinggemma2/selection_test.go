package embeddinggemma2

import (
	"errors"
	"fmt"
	"slices"
	"testing"
)

// selectionFake is a single-input accelerator with configurable reporting.
type selectionFake struct {
	name        string
	device      DeviceInfo
	profile     string
	memory      DeviceMemory
	memoryErr   error
	memoryN     func() (DeviceMemory, error)
	estimate    BatchMemoryEstimate
	estimateErr error
	estimateN   func([]int) (BatchMemoryEstimate, error)
	forwardErr  error
	forwardN    func([][]int) error
	closeErr    error
	dim         int
	forwards    int
	estimates   int
	closes      int
	got         [][]int
}

func (f *selectionFake) Name() string { return f.name }

func (f *selectionFake) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	f.forwards++
	f.got = append(f.got, slices.Clone(ids))
	if f.forwardErr != nil {
		return nil, nil, f.forwardErr
	}
	if f.forwardN != nil {
		if err := f.forwardN([][]int{slices.Clone(ids)}); err != nil {
			return nil, nil, err
		}
	}
	return selectionRows([][]int{ids}, f.dim)[0], nil, nil
}

func (f *selectionFake) ForwardEmbeds(x []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	return nil, nil, errors.New("unexpected ForwardEmbeds")
}

func (f *selectionFake) Close() error {
	f.closes++
	return f.closeErr
}

func (f *selectionFake) AttachedDevice() DeviceInfo { return f.device }
func (f *selectionFake) AttachedProfile() string    { return f.profile }

func (f *selectionFake) DeviceMemory() (DeviceMemory, error) {
	if f.memoryN != nil {
		return f.memoryN()
	}
	return f.memory, f.memoryErr
}

func (f *selectionFake) BatchMemory(lengths []int) (BatchMemoryEstimate, error) {
	f.estimates++
	if f.estimateN != nil {
		return f.estimateN(lengths)
	}
	return f.estimate, f.estimateErr
}

type selectionBatchFake struct {
	*selectionFake
	batchForwards int
	batchErr      error
	batchN        func([][]int) error
}

func (f *selectionBatchFake) ForwardBatch(ids [][]int) ([][]float32, error) {
	f.batchForwards++
	if f.batchErr != nil {
		return nil, f.batchErr
	}
	if f.batchN != nil {
		if err := f.batchN(ids); err != nil {
			return nil, err
		}
	}
	return selectionRows(ids, f.dim), nil
}

// noProfile hides every optional reporter, leaving only the Accelerator methods.
type noProfile struct{ Accelerator }

func selectionRows(ids [][]int, dim int) [][]float32 {
	out := make([][]float32, len(ids))
	for i, in := range ids {
		row := make([]float32, len(in)*dim)
		for j := range row {
			row[j] = float32(i + 1)
		}
		out[i] = row
	}
	return out
}

func registerSelectionAccel(t *testing.T, name string, accel Accelerator) {
	t.Helper()
	RegisterAccelerator(name, func(*Model) (Accelerator, error) { return accel, nil })
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, name)
		delete(selectedAccels, name)
		accelMu.Unlock()
	})
}

func registerSelectionDevice(t *testing.T, backend, selector string, rows []DeviceInfo, probeErr error, open func(*Model, string) (Accelerator, error)) {
	t.Helper()
	RegisterDeviceProber(backend, func(ProbeOptions) ([]DeviceInfo, error) { return nil, nil })
	RegisterDeviceInventoryProber(backend, func(ProbeOptions) ([]DeviceInfo, error) { return rows, probeErr })
	RegisterDeviceAccelerator(selector, backend, open)
	t.Cleanup(func() {
		accelMu.Lock()
		delete(deviceProbes, backend)
		delete(deviceInventoryProbes, backend)
		delete(selectedAccels, selector)
		accelMu.Unlock()
	})
}

func TestSelectTextOrdersTargetsProfilesAndLeavesReceiver(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	receiver := &selectionFake{name: "receiver", dim: dim}
	e.accel = receiver

	first := [][]int{{1, 2}, {3}}
	mismatch := &selectionFake{name: "mismatch", dim: dim, profile: "profile-b", device: DeviceInfo{Backend: "fake", Name: "A"}}
	winner := &selectionBatchFake{selectionFake: &selectionFake{
		name: "winner", dim: dim, profile: "profile-a",
		device: DeviceInfo{ID: "fake:dev-1", Alias: "fake:0", Backend: "fake", Name: "B"},
		memory: DeviceMemory{AvailableBytes: 1000, TotalBytes: 2000, Basis: MemoryLive},
	}}
	RegisterAccelerator("sel-mismatch", func(*Model) (Accelerator, error) { return mismatch, nil })
	RegisterAccelerator("sel-winner", func(*Model) (Accelerator, error) { return winner, nil })
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, "sel-mismatch")
		delete(accels, "sel-winner")
		accelMu.Unlock()
	})

	session, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{
		{Selector: "sel-mismatch", RequiredProfile: "profile-a"},
		{Selector: "sel-winner", RequiredProfile: "profile-a"},
	}}, first)
	if err != nil {
		t.Fatal(err)
	}
	if report.Winner != 1 || len(report.Attempts) != 2 {
		t.Fatalf("winner %d, attempts %+v", report.Winner, report.Attempts)
	}
	if !errors.Is(report.Attempts[0].Err, ErrUnsupportedSelector) || mismatch.closes != 1 {
		t.Fatalf("mismatch attempt %+v, closes %d", report.Attempts[0], mismatch.closes)
	}
	second := report.Attempts[1]
	if second.Profile != "profile-a" || !second.ProfileKnown || !second.Batched || second.Device.ID != "fake:dev-1" ||
		second.Memory.Basis != MemoryLive || second.EstimateErr != nil {
		t.Fatalf("winner attempt %+v", second)
	}
	if len(session.First) != 2 || len(session.First[0]) != dim {
		t.Fatalf("first vectors %d x %d", len(session.First), len(session.First[0]))
	}
	if e.Accelerator() != receiver || receiver.forwards != 0 {
		t.Fatal("selection touched the receiver encoder's attachment")
	}
	if session.enc == e || session.enc.Accelerator() != winner {
		t.Fatal("session did not own a private encoder and the selected accelerator")
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if winner.closes != 1 {
		t.Fatalf("winner closes %d", winner.closes)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := session.EmbedIDsBatch(first); !errors.Is(err, ErrDeviceUnavailable) {
		t.Fatalf("closed session error = %v", err)
	}
}

func TestSelectTextResolvesAliasAndVerifiesIdentity(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	const backend, selector = "fake-select-backend", "fake-select-selector"
	row := DeviceInfo{ID: backend + ":dev-1", Alias: backend + ":0", Backend: backend, Name: "Fake"}
	fake := &selectionFake{name: selector, dim: dim, device: row}
	registerSelectionDevice(t, backend, selector, []DeviceInfo{row}, nil,
		func(_ *Model, token string) (Accelerator, error) {
			if token != "dev-1" {
				return nil, fmt.Errorf("token = %q", token)
			}
			return fake, nil
		})

	session, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{
		{DeviceID: backend + ":0", Selector: selector},
	}}, [][]int{{4, 5}})
	if err != nil {
		t.Fatal(err)
	}
	if report.Winner != 0 || report.Attempts[0].Device.ID != row.ID || report.Attempts[0].Device.Alias != row.Alias {
		t.Fatalf("report %+v", report)
	}
	if fake.closes != 0 {
		t.Fatal("selected accelerator closed early")
	}
	if err := session.Close(); err != nil || fake.closes != 1 {
		t.Fatalf("close: %v, closes %d", err, fake.closes)
	}

	stale := &selectionFake{name: selector, dim: dim}
	registerSelectionDevice(t, backend, selector, []DeviceInfo{row}, nil,
		func(*Model, string) (Accelerator, error) { return stale, nil })
	_, report, err = e.SelectText(TextSelectionOptions{Targets: []TextTarget{
		{DeviceID: backend + ":missing", Selector: selector},
	}}, [][]int{{1}})
	if !errors.Is(err, ErrNoDevice) || report.Winner != -1 || len(report.Attempts) != 1 ||
		!errors.Is(report.Attempts[0].Err, ErrNoDevice) || stale.closes != 0 {
		t.Fatalf("stale alias: %v, report %+v, closes %d", err, report, stale.closes)
	}
}

func TestSelectTextPartialInventoryAndMissingIdentity(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	const backend, selector = "fake-partial-backend", "fake-partial-selector"
	row := DeviceInfo{ID: backend + ":dev-1", Alias: backend + ":0", Backend: backend, Name: "Partial"}
	partial := errors.New("one probe failed")
	fake := &selectionFake{name: selector, dim: dim, device: row}
	registerSelectionDevice(t, backend, selector, []DeviceInfo{row}, partial,
		func(*Model, string) (Accelerator, error) { return fake, nil })
	session, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{
		{DeviceID: backend + ":0", Selector: selector},
	}}, [][]int{{1}})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.InventoryFailures) != 1 || report.InventoryFailures[0].Backend != backend ||
		!errors.Is(report.InventoryFailures[0].Err, partial) {
		t.Fatalf("inventory failures %+v", report.InventoryFailures)
	}
	session.Close()

	twin := row
	twin.ID = backend + ":dev-2"
	ambiguous := &selectionFake{name: selector, dim: dim, device: row}
	registerSelectionDevice(t, backend, selector, []DeviceInfo{row, twin}, nil,
		func(*Model, string) (Accelerator, error) { return ambiguous, nil })
	fallback := &selectionFake{name: "ambiguous-fallback", dim: dim}
	registerSelectionAccel(t, "sel-ambiguous-fallback", fallback)
	_, report, err = e.SelectText(TextSelectionOptions{Targets: []TextTarget{
		{DeviceID: row.Alias, Selector: selector},
		{Selector: "sel-ambiguous-fallback"},
	}}, [][]int{{1}})
	if !errors.Is(err, ErrAmbiguousDevice) || len(report.Attempts) != 1 || ambiguous.closes != 0 || fallback.forwards != 0 {
		t.Fatalf("ambiguous alias: %v, report %+v, closes %d, fallback %d", err, report, ambiguous.closes, fallback.forwards)
	}

	hidden := &selectionFake{name: selector, dim: dim, device: row}
	registerSelectionDevice(t, backend, selector, []DeviceInfo{row}, nil,
		func(*Model, string) (Accelerator, error) { return noProfile{Accelerator: hidden}, nil })
	_, report, err = e.SelectText(TextSelectionOptions{Targets: []TextTarget{
		{DeviceID: row.ID, Selector: selector},
	}}, [][]int{{1}})
	if !errors.Is(err, ErrDeviceUnavailable) || report.Winner != -1 || hidden.closes != 1 {
		t.Fatalf("missing identity: %v, report %+v, closes %d", err, report, hidden.closes)
	}
}

func TestSelectTextBudgetFitAndSplit(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	zero := int64(0)
	one := int64(1)

	t.Run("zero cap is real", func(t *testing.T) {
		fake := &selectionFake{name: "cap", dim: dim, estimate: BatchMemoryEstimate{AdditionalPeakBytes: 1, Quality: BatchMemoryUpperBound}}
		registerSelectionAccel(t, "sel-cap", fake)
		_, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-cap"}}, MaxAdditionalBytes: &zero}, [][]int{{1}})
		if !errors.Is(err, ErrSelectionFit) || report.Attempts[0].EstimateErr == nil || fake.forwards != 0 {
			t.Fatalf("zero cap: %v, report %+v, forwards %d", err, report, fake.forwards)
		}
		zeroEstimate := &selectionFake{name: "zero-est", dim: dim, estimate: BatchMemoryEstimate{Quality: BatchMemoryUpperBound}}
		registerSelectionAccel(t, "sel-zero-est", zeroEstimate)
		session, _, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-zero-est"}}, MaxAdditionalBytes: &zero}, [][]int{{1}})
		if err != nil {
			t.Fatal(err)
		}
		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("approximate needs permission", func(t *testing.T) {
		fake := &selectionFake{name: "approx", dim: dim, estimate: BatchMemoryEstimate{AdditionalPeakBytes: 5, Quality: BatchMemoryApproximate}}
		registerSelectionAccel(t, "sel-approx", fake)
		capTen := int64(10)
		_, _, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-approx"}}, MaxAdditionalBytes: &capTen}, [][]int{{1}})
		if !errors.Is(err, ErrSelectionFit) {
			t.Fatalf("without permission: %v", err)
		}
		session, _, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-approx"}}, MaxAdditionalBytes: &capTen, AllowEstimates: true}, [][]int{{1}})
		if err != nil {
			t.Fatal(err)
		}
		session.Close()
	})

	t.Run("estimated headroom needs permission", func(t *testing.T) {
		fake := &selectionFake{name: "estimated", dim: dim,
			memory:   DeviceMemory{AvailableBytes: 100, Basis: MemoryEstimated},
			estimate: BatchMemoryEstimate{AdditionalPeakBytes: 200, Quality: BatchMemoryUpperBound}}
		registerSelectionAccel(t, "sel-estimated", fake)
		_, _, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-estimated"}}, AllowEstimates: true}, [][]int{{1}})
		if !errors.Is(err, ErrSelectionFit) {
			t.Fatalf("estimated headroom with permission: %v", err)
		}
		session, _, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-estimated"}}}, [][]int{{1}})
		if err != nil {
			t.Fatal(err)
		}
		session.Close()
	})

	t.Run("unknown estimate cannot certify a cap", func(t *testing.T) {
		fake := &selectionFake{name: "unknown", dim: dim}
		registerSelectionAccel(t, "sel-unknown", fake)
		_, _, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-unknown"}}, MaxAdditionalBytes: &one}, [][]int{{1}})
		if !errors.Is(err, ErrSelectionFit) || errors.Is(err, ErrDeviceOutOfMemory) {
			t.Fatalf("unknown estimate: %v", err)
		}
		session, _, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-unknown"}}}, [][]int{{1}})
		if err != nil {
			t.Fatal(err)
		}
		session.Close()
	})

	t.Run("preflight shape splits in order", func(t *testing.T) {
		fake := &selectionBatchFake{selectionFake: &selectionFake{name: "split", dim: dim}}
		fake.estimateN = func(lengths []int) (BatchMemoryEstimate, error) {
			if len(lengths) > 2 {
				return BatchMemoryEstimate{}, fmt.Errorf("%w: test limit", ErrUnsupportedBatchShape)
			}
			return BatchMemoryEstimate{}, nil
		}
		registerSelectionAccel(t, "sel-split", fake)
		session, report, err := e.SelectText(TextSelectionOptions{
			Targets: []TextTarget{{Selector: "sel-split"}}, AllowSplit: true,
		}, [][]int{{1}, {2, 3}, {4}, {5, 6}, {7}})
		if err != nil {
			t.Fatal(err)
		}
		if len(session.First) != 5 || !slices.EqualFunc(report.Attempts[0].FittedLengths, [][]int{{1, 2}, {1}, {2, 1}}, slices.Equal[[]int]) {
			t.Fatalf("first %d, fitted %v", len(session.First), report.Attempts[0].FittedLengths)
		}
		session.Close()
	})

	t.Run("certified OOM splits, ordinary OOM stops", func(t *testing.T) {
		split := &selectionBatchFake{selectionFake: &selectionFake{name: "oom-split", dim: dim}}
		split.batchN = func(ids [][]int) error {
			if len(ids) > 1 {
				return fmt.Errorf("native: %w", ErrBatchOutOfMemory)
			}
			return nil
		}
		registerSelectionAccel(t, "sel-oom-split", split)
		session, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-oom-split"}}, AllowSplit: true}, [][]int{{1}, {2}, {3}})
		if err != nil {
			t.Fatal(err)
		}
		if len(session.First) != 3 || !slices.EqualFunc(report.Attempts[0].FittedLengths, [][]int{{1}, {1}, {1}}, slices.Equal[[]int]) {
			t.Fatalf("OOM split first %d, fitted %v", len(session.First), report.Attempts[0].FittedLengths)
		}
		session.Close()

		ordinary := &selectionBatchFake{selectionFake: &selectionFake{name: "oom-stop", dim: dim}}
		ordinary.batchErr = fmt.Errorf("native: %w", ErrDeviceOutOfMemory)
		registerSelectionAccel(t, "sel-oom-stop", ordinary)
		fallback := &selectionFake{name: "fallback", dim: dim}
		registerSelectionAccel(t, "sel-oom-fallback", fallback)
		_, report, err = e.SelectText(TextSelectionOptions{Targets: []TextTarget{
			{Selector: "sel-oom-stop"}, {Selector: "sel-oom-fallback"},
		}, AllowSplit: true}, [][]int{{1}, {2}})
		if !errors.Is(err, ErrDeviceOutOfMemory) || errors.Is(err, ErrBatchOutOfMemory) ||
			len(report.Attempts) != 1 || fallback.forwards != 0 {
			t.Fatalf("ordinary OOM: %v, report %+v, fallback forwards %d", err, report, fallback.forwards)
		}

		lost := &selectionBatchFake{selectionFake: &selectionFake{name: "lost", dim: dim}}
		lost.batchErr = fmt.Errorf("native: %w", ErrDeviceLost)
		registerSelectionAccel(t, "sel-lost", lost)
		lostFallback := &selectionFake{name: "lost-fallback", dim: dim}
		registerSelectionAccel(t, "sel-lost-fallback", lostFallback)
		_, report, err = e.SelectText(TextSelectionOptions{Targets: []TextTarget{
			{Selector: "sel-lost"}, {Selector: "sel-lost-fallback"},
		}, AllowSplit: true}, [][]int{{1}, {2}})
		if !errors.Is(err, ErrDeviceLost) || len(report.Attempts) != 1 || lostFallback.forwards != 0 {
			t.Fatalf("device lost: %v, report %+v, fallback forwards %d", err, report, lostFallback.forwards)
		}
	})
}

func TestSelectTextSessionKeepsRouteAndNoPartialBatch(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	first := &selectionBatchFake{selectionFake: &selectionFake{name: "first", dim: dim}}
	registerSelectionAccel(t, "sel-first", first)
	second := &selectionBatchFake{selectionFake: &selectionFake{name: "second", dim: dim}}
	registerSelectionAccel(t, "sel-second", second)
	first.memory = DeviceMemory{AvailableBytes: 10, Basis: MemoryLive}
	first.estimate = BatchMemoryEstimate{AdditionalPeakBytes: 5, Quality: BatchMemoryUpperBound}
	session, _, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-first"}}}, [][]int{{1}})
	if err != nil {
		t.Fatal(err)
	}
	if first.forwards != 1 {
		t.Fatalf("first group ran %d times, want once", first.forwards)
	}
	RegisterAccelerator("sel-route-swap", func(*Model) (Accelerator, error) { return second, nil })
	got, err := session.EmbedIDsBatch([][]int{{2}, {3}})
	if err != nil || len(got) != 2 || first.batchForwards != 1 || second.batchForwards != 0 {
		t.Fatalf("later batch %d, %v; first %d second %d", len(got), err, first.batchForwards, second.batchForwards)
	}

	first.memory = DeviceMemory{AvailableBytes: 1, Basis: MemoryLive}
	if got, err := session.EmbedIDsBatch([][]int{{6}}); !errors.Is(err, ErrSelectionFit) || got != nil || first.forwards != 1 {
		t.Fatalf("changed headroom batch = %d, %v, forwards %d", len(got), err, first.forwards)
	}
	first.memory = DeviceMemory{AvailableBytes: 10, Basis: MemoryLive}
	first.batchN = func(ids [][]int) error {
		if len(ids) > 1 {
			return errors.New("second half failed")
		}
		return nil
	}
	session.allowSplit = true
	if got, err := session.EmbedIDsBatch([][]int{{4}, {5}}); err == nil || got != nil {
		t.Fatalf("partial batch = %d vectors, %v", len(got), err)
	}
	beforeForwards := first.forwards
	if got, err := session.EmbedIDsBatch([][]int{{e.m.cfg.VocabSize}}); err == nil || got != nil || first.forwards != beforeForwards {
		t.Fatalf("out-of-range session input = %d, %v, forwards %d", len(got), err, first.forwards)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	accelMu.Lock()
	delete(accels, "sel-route-swap")
	accelMu.Unlock()
}

func TestSelectTextRefreshesHeadroomPerSubgroup(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	fake := &selectionBatchFake{selectionFake: &selectionFake{name: "headroom", dim: dim}}
	fake.estimate = BatchMemoryEstimate{AdditionalPeakBytes: 5, Quality: BatchMemoryUpperBound}
	reads := 0
	fake.memoryN = func() (DeviceMemory, error) {
		reads++
		if reads%2 == 0 {
			return DeviceMemory{AvailableBytes: 1, Basis: MemoryLive}, nil
		}
		return DeviceMemory{AvailableBytes: 10, Basis: MemoryLive}, nil
	}
	fake.batchN = func(ids [][]int) error {
		if len(ids) > 1 {
			return fmt.Errorf("native: %w", ErrBatchOutOfMemory)
		}
		return nil
	}
	registerSelectionAccel(t, "sel-headroom", fake)
	session, report, err := e.SelectText(TextSelectionOptions{
		Targets: []TextTarget{{Selector: "sel-headroom"}}, AllowSplit: true,
	}, [][]int{{1}, {2}})
	if session != nil || !errors.Is(err, ErrSelectionFit) || report.Winner != -1 {
		t.Fatalf("per-subgroup headroom: session %v, %v, report %+v", session, err, report)
	}
	if fake.forwards != 1 || fake.batchForwards != 0 {
		t.Fatalf("failed split ran forwards %d batch %d; want only the admitted left singleton", fake.forwards, fake.batchForwards)
	}
}

func TestSelectTextRejectedAttachKeepsCloseFailure(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	const backend, selector = "fake-close-backend", "fake-close-selector"
	row := DeviceInfo{ID: backend + ":dev-1", Backend: backend, Name: "Close"}
	closeErr := errors.New("close failed")
	hidden := &selectionFake{name: selector, dim: dim, device: row, closeErr: closeErr}
	registerSelectionDevice(t, backend, selector, []DeviceInfo{row}, nil,
		func(*Model, string) (Accelerator, error) { return noProfile{Accelerator: hidden}, nil })
	fallback := &selectionFake{name: "close-fallback", dim: dim}
	registerSelectionAccel(t, "sel-close-fallback", fallback)
	_, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{
		{DeviceID: row.ID, Selector: selector}, {Selector: "sel-close-fallback"},
	}}, [][]int{{1}})
	if !errors.Is(err, errSelectionCleanup) || !errors.Is(err, closeErr) ||
		len(report.Attempts) != 1 || hidden.closes != 1 || fallback.forwards != 0 {
		t.Fatalf("rejected attach close: %v, report %+v, closes %d, fallback %d", err, report, hidden.closes, fallback.forwards)
	}
}

func TestSelectTextMemoryFailureVersusKnownZero(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	readErr := errors.New("memory query failed")
	failed := &selectionFake{name: "memory-failed", dim: dim, memoryErr: readErr}
	registerSelectionAccel(t, "sel-memory-failed", failed)
	session, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-memory-failed"}}}, [][]int{{1}})
	if err != nil {
		t.Fatal(err)
	}
	if !errors.Is(report.Attempts[0].MemoryErr, readErr) || report.Attempts[0].Memory.Basis != MemoryUnknown {
		t.Fatalf("failed memory attempt %+v", report.Attempts[0])
	}
	session.Close()

	zero := &selectionFake{name: "memory-zero", dim: dim, memory: DeviceMemory{AvailableBytes: 0, Basis: MemoryLive},
		estimate: BatchMemoryEstimate{Quality: BatchMemoryUpperBound}}
	registerSelectionAccel(t, "sel-memory-zero", zero)
	session, report, err = e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-memory-zero"}}}, [][]int{{1}})
	if err != nil {
		t.Fatal(err)
	}
	if report.Attempts[0].MemoryErr != nil || report.Attempts[0].Memory.Basis != MemoryLive ||
		report.Attempts[0].Memory.AvailableBytes != 0 {
		t.Fatalf("known-zero memory attempt %+v", report.Attempts[0])
	}
	session.Close()
}

func TestSelectTextCPUValidationAndReporterGaps(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	first := [][]int{{1, 2}, {3}}
	cpuFirst, err := e.EmbedIDsBatch(first)
	if err != nil {
		t.Fatal(err)
	}
	session, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "cpu"}}}, first)
	if err != nil {
		t.Fatal(err)
	}
	if report.Attempts[0].Device.Backend != "cpu" || report.Attempts[0].Batched ||
		report.Attempts[0].Memory != (DeviceMemory{}) || !sameBits(session.First[0], cpuFirst[0]) {
		t.Fatalf("CPU attempt %+v, first match %v", report.Attempts[0], sameBits(session.First[0], cpuFirst[0]))
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	fake := &selectionFake{name: "no-profile", dim: dim, profile: "visible"}
	RegisterAccelerator("sel-no-profile", func(*Model) (Accelerator, error) { return noProfile{Accelerator: fake}, nil })
	if _, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{{Selector: "sel-no-profile", RequiredProfile: "visible"}}}, [][]int{{1}}); !errors.Is(err, ErrUnsupportedSelector) || report.Attempts[0].ProfileKnown {
		t.Fatalf("missing profile reporter: %v, report %+v", err, report)
	}
	accelMu.Lock()
	delete(accels, "sel-no-profile")
	accelMu.Unlock()

	for _, tc := range []struct {
		name  string
		opts  TextSelectionOptions
		first [][]int
	}{
		{"no targets", TextSelectionOptions{}, [][]int{{1}}},
		{"cpu with device", TextSelectionOptions{Targets: []TextTarget{{Selector: "cpu", DeviceID: "x:y"}}}, [][]int{{1}}},
		{"empty first", TextSelectionOptions{Targets: []TextTarget{{Selector: "cpu"}}}, nil},
		{"empty row", TextSelectionOptions{Targets: []TextTarget{{Selector: "cpu"}}}, [][]int{{}}},
		{"out of range id", TextSelectionOptions{Targets: []TextTarget{{Selector: "cpu"}}}, [][]int{{e.m.cfg.VocabSize}}},
		{"negative cap", TextSelectionOptions{Targets: []TextTarget{{Selector: "cpu"}}, MaxAdditionalBytes: func() *int64 { v := int64(-1); return &v }()}, [][]int{{1}}},
	} {
		if session, _, err := e.SelectText(tc.opts, tc.first); err == nil || session != nil {
			t.Errorf("%s: (%v, %v), want error", tc.name, session, err)
		}
	}
}

func TestSelectTextCleanupFailureStops(t *testing.T) {
	e := newTinyEncoderForTest(t)
	dim := e.HiddenDim()
	failing := &selectionFake{name: "close-fail", dim: dim, closeErr: errors.New("close failed"), forwardErr: errors.New("forward failed")}
	next := &selectionFake{name: "next", dim: dim}
	registerSelectionAccel(t, "sel-close-fail", failing)
	registerSelectionAccel(t, "sel-close-next", next)
	_, report, err := e.SelectText(TextSelectionOptions{Targets: []TextTarget{
		{Selector: "sel-close-fail"}, {Selector: "sel-close-next"},
	}}, [][]int{{1}})
	if !errors.Is(err, errSelectionCleanup) || len(report.Attempts) != 1 || next.forwards != 0 {
		t.Fatalf("cleanup failure: %v, report %+v, next forwards %d", err, report, next.forwards)
	}
	if e.Accelerator() != nil {
		t.Fatal("failed selection changed the receiver encoder")
	}
}
