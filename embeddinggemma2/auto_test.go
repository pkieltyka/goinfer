package embeddinggemma2

import (
	"errors"
	"testing"
)

func TestUseAutoAccelerator_selectsWorkingBackend(t *testing.T) {
	const unavailable, selected, unused = "test-unavailable", "test-selected", "test-unused"
	const implementation = "test-implementation"
	e := newTinyEncoderForTest(t)
	refusal := errors.New("adapter unavailable")
	RegisterAccelerator(unavailable, func(*Model) (Accelerator, error) { return nil, refusal })
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, unavailable)
		accelMu.Unlock()
	})
	ids := []int{5, 17, 42}
	fake := &fakeEmbedAccel{name: implementation, rows: distinguishableRows(len(ids), e.HiddenDim())}
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
	if choice.Backend != implementation || choice.Selector != selected || e.Accelerator() != fake ||
		len(choice.Declined) != 1 || choice.Declined[0].Selector != unavailable ||
		!errors.Is(choice.Declined[0].Err, refusal) {
		t.Fatalf("unexpected selection: %+v", choice)
	}
	if again := e.UseAutoAccelerator("cpu", unused); again.Backend != implementation || again.Selector != "" {
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
		len(choice.Declined) != 1 || choice.Declined[0].Selector != missing ||
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

func TestUseAutoAccelerator_defaultOrder(t *testing.T) {
	refusal := errors.New("CUDA unavailable")
	cudaAvailable := true
	cuda := &fakeEmbedAccel{name: "cuda"}
	vulkan := &fakeEmbedAccel{name: "vulkan"}
	RegisterAccelerator("cuda", func(*Model) (Accelerator, error) {
		if !cudaAvailable {
			return nil, refusal
		}
		return cuda, nil
	})
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, "cuda")
		accelMu.Unlock()
	})
	registerTempAccelerator(t, "vulkan", vulkan)
	RegisterAccelerator("test-experimental", func(*Model) (Accelerator, error) {
		t.Error("auto attempted a backend outside its default policy")
		return nil, refusal
	})
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, "test-experimental")
		accelMu.Unlock()
	})

	both := newTinyEncoderForTest(t)
	if got := both.UseAutoAccelerator(); got.Backend != "cuda" || both.Accelerator() != cuda {
		t.Fatalf("CUDA did not take precedence: %+v", got)
	}
	cudaAvailable = false
	amd := newTinyEncoderForTest(t)
	got := amd.UseAutoAccelerator()
	if got.Backend != "vulkan" || amd.Accelerator() != vulkan ||
		len(got.Declined) != 1 || got.Declined[0].Selector != "cuda" ||
		!errors.Is(got.Declined[0].Err, refusal) {
		t.Fatalf("auto did not try Vulkan after a CUDA decline: %+v", got)
	}
}

func TestUseAutoAccelerator_emptyFactory(t *testing.T) {
	const nilFactory, nilResult = "test-nil-factory", "test-nil-result"
	RegisterAccelerator(nilFactory, nil)
	RegisterAccelerator(nilResult, func(*Model) (Accelerator, error) { return nil, nil })
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, nilFactory)
		delete(accels, nilResult)
		accelMu.Unlock()
	})
	e := newTinyEncoderForTest(t)
	for _, name := range []string{nilFactory, nilResult} {
		if a, err := e.UseAccelerator(name); a != nil || err == nil || e.Accelerator() != nil {
			t.Fatalf("UseAccelerator(%q) accepted an empty factory result", name)
		}
	}
	got := e.UseAutoAccelerator(nilFactory, nilResult)
	if got.Backend != "cpu" || len(got.Declined) != 2 || e.Accelerator() != nil {
		t.Fatalf("auto accepted an empty factory result: %+v", got)
	}
}
