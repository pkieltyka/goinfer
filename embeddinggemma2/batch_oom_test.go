package embeddinggemma2

import (
	"errors"
	"fmt"
	"slices"
	"testing"
)

type stubRetryOOMAccel struct {
	name     string
	maxTotal int
	failErr  error
	failOnce bool
	calls    int
	embedDim int
}

func (s *stubRetryOOMAccel) Name() string { return s.name }
func (s *stubRetryOOMAccel) Close() error { return nil }

func (s *stubRetryOOMAccel) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	s.calls++
	if s.failErr != nil && (s.failOnce || len(ids) > s.maxTotal) {
		err := s.failErr
		if s.failOnce {
			s.failErr = nil
		}
		return nil, nil, err
	}
	return syntheticHidden(ids, s.embedDim), nil, nil
}

func (s *stubRetryOOMAccel) ForwardEmbeds(x []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	return nil, nil, errors.New("unexpected ForwardEmbeds")
}

func (s *stubRetryOOMAccel) ForwardBatch(ids [][]int) ([][]float32, error) {
	s.calls++
	total := 0
	for _, in := range ids {
		total += len(in)
	}
	if s.failErr != nil && (s.failOnce || total > s.maxTotal) {
		err := s.failErr
		if s.failOnce {
			s.failErr = nil
		}
		return nil, err
	}
	out := make([][]float32, len(ids))
	for i, in := range ids {
		out[i] = syntheticHidden(in, s.embedDim)
	}
	return out, nil
}

func syntheticHidden(ids []int, dim int) []float32 {
	out := make([]float32, len(ids)*dim)
	for row, id := range ids {
		for d := range dim {
			out[row*dim+d] = float32(id+1) + 0.1*float32(d+1)
		}
	}
	return out
}

func TestErrBatchOutOfMemory_Hierarchy(t *testing.T) {
	if !errors.Is(ErrBatchOutOfMemory, ErrDeviceOutOfMemory) {
		t.Fatal("ErrBatchOutOfMemory must wrap ErrDeviceOutOfMemory")
	}
	if errors.Is(ErrDeviceOutOfMemory, ErrBatchOutOfMemory) {
		t.Fatal("ErrDeviceOutOfMemory alone must not match ErrBatchOutOfMemory")
	}
	if errors.Is(ErrBatchOutOfMemory, ErrDeviceUnavailable) || errors.Is(ErrBatchOutOfMemory, ErrDeviceLost) {
		t.Fatal("ErrBatchOutOfMemory must stay distinct from unavailable and device-lost errors")
	}
}

func TestEncoderEmbedIDsBatch_RetrySafeOOMAndSplitRetry(t *testing.T) {
	nativeOOM := errors.New("native: out of device memory")
	accel := &stubRetryOOMAccel{
		name:     "stub-gpu",
		maxTotal: 6,
		failErr:  fmt.Errorf("backend forward: %w: %w", ErrBatchOutOfMemory, nativeOOM),
		embedDim: 4,
	}
	enc := &Encoder{
		m:     &Model{cfg: Config{Hidden: 4, EmbeddingDim: 4, MaxPositions: 64}},
		accel: accel,
	}

	orig := [][]int{{1, 2, 3}, {4, 5}, {6, 7}, {8, 9}}
	in := make([][]int, len(orig))
	for i := range orig {
		in[i] = slices.Clone(orig[i])
	}

	got, err := enc.EmbedIDsBatch(in)
	if got != nil {
		t.Fatalf("failed EmbedIDsBatch returned non-nil outputs: %v", got)
	}
	if !errors.Is(err, ErrBatchOutOfMemory) || !errors.Is(err, ErrDeviceOutOfMemory) || !errors.Is(err, nativeOOM) {
		t.Fatalf("batch OOM error = %v, want ErrBatchOutOfMemory + ErrDeviceOutOfMemory + native cause", err)
	}
	for i := range orig {
		if !slices.Equal(in[i], orig[i]) {
			t.Fatalf("input %d mutated on OOM: got %v, want %v", i, in[i], orig[i])
		}
	}

	// Explicit caller retry halving between complete inputs on the same encoder.
	left, err := enc.EmbedIDsBatch(in[:2])
	if err != nil {
		t.Fatalf("left half retry failed: %v", err)
	}
	right, err := enc.EmbedIDsBatch(in[2:])
	if err != nil {
		t.Fatalf("right half retry failed: %v", err)
	}
	combined := append(left, right...)
	if len(combined) != len(orig) {
		t.Fatalf("combined len = %d, want %d", len(combined), len(orig))
	}
	for i, ids := range orig {
		want, err := enc.EmbedIDs(ids)
		if err != nil {
			t.Fatalf("singleton %d: %v", i, err)
		}
		if !slices.Equal(combined[i], want) {
			t.Fatalf("split output %d mismatch: got %v, want %v", i, combined[i], want)
		}
	}
}

func TestEncoderEmbedIDs_SingletonRetrySafeAndNonRetryErrors(t *testing.T) {
	nativeOOM := errors.New("native: out of device memory")
	accel := &stubRetryOOMAccel{
		name:     "stub-gpu",
		failOnce: true,
		failErr:  fmt.Errorf("backend forward: %w: %w", ErrBatchOutOfMemory, nativeOOM),
		embedDim: 4,
	}
	enc := &Encoder{
		m:     &Model{cfg: Config{Hidden: 4, EmbeddingDim: 4, MaxPositions: 64}},
		accel: accel,
	}

	orig := []int{3, 5, 7}
	in := slices.Clone(orig)
	got, err := enc.EmbedIDs(in)
	if got != nil {
		t.Fatalf("failed EmbedIDs returned non-nil output: %v", got)
	}
	if !errors.Is(err, ErrBatchOutOfMemory) || !errors.Is(err, ErrDeviceOutOfMemory) || !errors.Is(err, nativeOOM) {
		t.Fatalf("singleton OOM = %v, want ErrBatchOutOfMemory + ErrDeviceOutOfMemory + native cause", err)
	}
	if !slices.Equal(in, orig) {
		t.Fatalf("singleton input mutated: got %v, want %v", in, orig)
	}
	retryVec, err := enc.EmbedIDs(in)
	if err != nil || len(retryVec) != 4 {
		t.Fatalf("singleton retry = (%v, %v), want 4-dim vector", retryVec, err)
	}

	// Common ErrDeviceOutOfMemory without ErrBatchOutOfMemory must not authorize retry.
	accel.failOnce = false
	accel.maxTotal = 0
	accel.failErr = fmt.Errorf("backend forward: %w: %w", ErrDeviceOutOfMemory, nativeOOM)
	if v, err := enc.EmbedIDsBatch([][]int{{1, 2}}); v != nil || !errors.Is(err, ErrDeviceOutOfMemory) || errors.Is(err, ErrBatchOutOfMemory) {
		t.Fatalf("non-retry OOM = (%v, %v), want ErrDeviceOutOfMemory without ErrBatchOutOfMemory", v, err)
	}

	// Empty batch and empty member never match retry or device OOM.
	if _, err := enc.EmbedIDsBatch(nil); err == nil || errors.Is(err, ErrBatchOutOfMemory) || errors.Is(err, ErrDeviceOutOfMemory) {
		t.Fatalf("empty batch = %v", err)
	}
	if _, err := enc.EmbedIDsBatch([][]int{{1}, {}}); err == nil || errors.Is(err, ErrBatchOutOfMemory) || errors.Is(err, ErrDeviceOutOfMemory) {
		t.Fatalf("empty member = %v", err)
	}
}

func TestNewAccelerator_AttachOOMIsCommonOnly(t *testing.T) {
	nativeOOM := errors.New("native: weight upload OOM")
	nativeOther := errors.New("native: pipeline compile failed")

	oomName := "test-issue25-attach-oom"
	otherName := "test-issue25-attach-other"
	RegisterAccelerator(oomName, func(*Model) (Accelerator, error) {
		return nil, fmt.Errorf("attach: %w: %w", ErrDeviceOutOfMemory, nativeOOM)
	})
	RegisterAccelerator(otherName, func(*Model) (Accelerator, error) {
		return nil, fmt.Errorf("attach: %w", nativeOther)
	})
	t.Cleanup(func() {
		accelMu.Lock()
		delete(accels, oomName)
		delete(accels, otherName)
		accelMu.Unlock()
	})

	m := &Model{cfg: Config{Hidden: 4, EmbeddingDim: 4, MaxPositions: 64}}
	enc := &Encoder{m: m}

	if a, err := enc.UseAccelerator(oomName); a != nil || !errors.Is(err, ErrDeviceOutOfMemory) || !errors.Is(err, nativeOOM) || errors.Is(err, ErrBatchOutOfMemory) {
		t.Fatalf("attach OOM = (%v, %v), want nil + ErrDeviceOutOfMemory + native cause without ErrBatchOutOfMemory", a, err)
	}
	if enc.accel != nil {
		t.Fatal("failed attach OOM left a non-nil accelerator")
	}

	if a, err := enc.UseAccelerator(otherName); a != nil || err == nil || !errors.Is(err, nativeOther) || errors.Is(err, ErrDeviceOutOfMemory) || errors.Is(err, ErrBatchOutOfMemory) {
		t.Fatalf("unrelated attach failure = (%v, %v), want nil + native cause without OOM sentinels", a, err)
	}
	if enc.accel != nil {
		t.Fatal("unrelated failed attach left a non-nil accelerator")
	}
}
