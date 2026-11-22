package embeddinggemma2

import (
	"errors"
	"testing"
)

func TestUseAutoAccelerator_selectsWorkingBackend(t *testing.T) {
	const unavailable, selected, unused = "test-unavailable", "test-selected", "test-unused"
	e := newTinyEncoderForTest(t)
	refusal := errors.New("adapter unavailable")
	RegisterAccelerator(unavailable, func(*Model) (Accelerator, error) { return nil, refusal })
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, unavailable)
		accelMu.Unlock()
	})
	ids := []int{5, 17, 42}
	fake := &fakeEmbedAccel{name: selected, rows: distinguishableRows(len(ids), e.HiddenDim())}
	registerTempAccelerator(t, selected, fake)
	RegisterAccelerator(unused, func(*Model) (Accelerator, error) {
		t.Error("selector tried a backend after the selected one")
		return nil, errors.New("unexpected backend")
	})
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, unused)
		accelMu.Unlock()
	})

	choice := e.UseAutoAccelerator(unavailable, selected, unused)
	if choice.Backend != selected || e.Accelerator() != fake ||
		len(choice.Declined) != 1 || choice.Declined[0].Backend != unavailable ||
		!errors.Is(choice.Declined[0].Err, refusal) {
		t.Fatalf("unexpected selection: %+v", choice)
	}
	if again := e.UseAutoAccelerator(unused); again.Backend != selected {
		t.Fatalf("second selection replaced an attached accelerator: %+v", again)
	}

	if _, err := e.EmbedIDs(ids); err != nil || fake.calls != 1 {
		t.Fatalf("selected accelerator was not used: %v, calls=%d", err, fake.calls)
	}
	fake.err = errors.New("device lost")
	if got, err := e.EmbedIDs(ids); got != nil || !errors.Is(err, fake.err) {
		t.Fatalf("forward should surface the selected backend's failure: %v, %v", got, err)
	}
}

func TestUseAutoAccelerator_reportsCPUFallback(t *testing.T) {
	e := newTinyEncoderForTest(t)
	const missing = "test-unlinked"
	choice := e.UseAutoAccelerator(missing, "cpu", "test-after-cpu")
	if choice.Backend != "cpu" || e.Accelerator() != nil ||
		len(choice.Declined) != 1 || choice.Declined[0].Backend != missing ||
		choice.Declined[0].Err == nil {
		t.Fatalf("unexpected fallback: %+v", choice)
	}
	ids := []int{4, 5}
	want, err := e.Model().Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := e.EmbedIDs(ids); err != nil || !sameBits(got, want) {
		t.Fatalf("fallback did not use CPU: %v", err)
	}
}

func TestUseAutoAccelerator_defaultCPU(t *testing.T) {
	e := newTinyEncoderForTest(t)
	choice := e.UseAutoAccelerator()
	if choice.Backend != "cpu" || e.Accelerator() != nil || len(choice.Declined) != 0 {
		t.Fatalf("CPU-only binary selected %+v", choice)
	}
}
