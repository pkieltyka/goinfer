package embeddinggemma2

import (
	"errors"
	"math"
	"testing"
)

type stubSingletonAccel struct {
	name string
}

func (s *stubSingletonAccel) Name() string { return s.name }
func (s *stubSingletonAccel) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	return nil, nil, errors.New("unexpected forward")
}
func (s *stubSingletonAccel) ForwardEmbeds(x []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	return nil, nil, errors.New("unexpected forward")
}
func (s *stubSingletonAccel) Close() error { return nil }

type stubBatchAccel struct {
	stubSingletonAccel
}

func (s *stubBatchAccel) ForwardBatch(ids [][]int) ([][]float32, error) {
	return nil, errors.New("unexpected batch forward")
}

type stubEstimatorAccel struct {
	stubBatchAccel
	got []int
	est BatchMemoryEstimate
	err error
}

func (s *stubEstimatorAccel) BatchMemory(preparedLengths []int) (BatchMemoryEstimate, error) {
	s.got = append([]int(nil), preparedLengths...)
	return s.est, s.err
}

func newTestEncoder(maxPos int, accel Accelerator) *Encoder {
	m := &Model{cfg: Config{MaxPositions: maxPos, Hidden: 16, EmbeddingDim: 16}}
	return &Encoder{m: m, accel: accel}
}

func TestBatchMemoryQuality_String(t *testing.T) {
	if got := BatchMemoryUnknown.String(); got != "unknown" {
		t.Fatalf("unknown = %q", got)
	}
	if got := BatchMemoryUpperBound.String(); got != "upper_bound" {
		t.Fatalf("upper_bound = %q", got)
	}
	if got := BatchMemoryApproximate.String(); got != "approximate" {
		t.Fatalf("approximate = %q", got)
	}
	if got := BatchMemoryQuality(99).String(); got != "unknown" {
		t.Fatalf("out-of-range = %q", got)
	}
}

func TestEncoderBatchMemory_InvalidAndUnsupportedShapes(t *testing.T) {
	enc := newTestEncoder(512, nil)

	for _, tc := range []struct {
		name string
		lens []int
	}{
		{"nil", nil},
		{"empty", []int{}},
		{"zero", []int{0}},
		{"negative", []int{4, -1}},
		{"overflow", []int{math.MaxInt, 1}},
	} {
		_, err := enc.BatchMemory(tc.lens)
		if err == nil {
			t.Errorf("%s: got nil error, want validation error", tc.name)
		}
		if errors.Is(err, ErrUnsupportedBatchShape) {
			t.Errorf("%s: invalid shape must not match ErrUnsupportedBatchShape: %v", tc.name, err)
		}
	}

	// Exceeds model MaxPositions -> ErrUnsupportedBatchShape.
	if _, err := enc.BatchMemory([]int{16, 513}); !errors.Is(err, ErrUnsupportedBatchShape) {
		t.Fatalf("over MaxPositions: got %v, want ErrUnsupportedBatchShape", err)
	}
	// Exceeds global MaxTokens even when MaxPositions is 0.
	encNoMaxPos := newTestEncoder(0, nil)
	if _, err := encNoMaxPos.BatchMemory([]int{MaxTokens + 1}); !errors.Is(err, ErrUnsupportedBatchShape) {
		t.Fatalf("over MaxTokens: got %v, want ErrUnsupportedBatchShape", err)
	}

	// Legal shapes on CPU report BatchMemoryUnknown with zero bytes.
	est, err := enc.BatchMemory([]int{16, 32})
	if err != nil {
		t.Fatalf("CPU legal shape: %v", err)
	}
	if est.Quality != BatchMemoryUnknown || est.AdditionalPeakBytes != 0 || est.ResidentWeightBytes != 0 {
		t.Fatalf("CPU estimate = %+v, want zero Unknown", est)
	}
}

func TestEncoderBatchMemory_MissingEstimatorRespectsBatchCapability(t *testing.T) {
	// Singleton-only accelerator without BatchMemoryEstimator:
	// 1 input -> Unknown (prices EmbedIDs); >1 inputs -> ErrUnsupportedBatchShape (lacks batchForwarder).
	singletonEnc := newTestEncoder(512, &stubSingletonAccel{name: "singleton-only"})
	est, err := singletonEnc.BatchMemory([]int{32})
	if err != nil || est.Quality != BatchMemoryUnknown || est.AdditionalPeakBytes != 0 {
		t.Fatalf("singleton legal shape = (%+v, %v), want Unknown nil", est, err)
	}
	if _, err := singletonEnc.BatchMemory([]int{32, 16}); !errors.Is(err, ErrUnsupportedBatchShape) {
		t.Fatalf("singleton multi-input = %v, want ErrUnsupportedBatchShape", err)
	}

	// Batch-capable accelerator without BatchMemoryEstimator: legal multi-input -> Unknown.
	batchEnc := newTestEncoder(512, &stubBatchAccel{stubSingletonAccel{name: "batch-no-est"}})
	est, err = batchEnc.BatchMemory([]int{32, 16})
	if err != nil || est.Quality != BatchMemoryUnknown || est.AdditionalPeakBytes != 0 {
		t.Fatalf("batch without estimator = (%+v, %v), want Unknown nil", est, err)
	}
}

func TestEncoderBatchMemory_DelegatesToEstimator(t *testing.T) {
	want := BatchMemoryEstimate{
		AdditionalPeakBytes: 12 << 20,
		ResidentWeightBytes: 300 << 20,
		Quality:             BatchMemoryUpperBound,
	}
	stub := &stubEstimatorAccel{
		stubBatchAccel: stubBatchAccel{stubSingletonAccel{name: "cuda"}},
		est:            want,
	}
	enc := newTestEncoder(512, stub)
	got, err := enc.BatchMemory([]int{64, 128})
	if err != nil {
		t.Fatalf("delegated BatchMemory: %v", err)
	}
	if got != want {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	if len(stub.got) != 2 || stub.got[0] != 64 || stub.got[1] != 128 {
		t.Fatalf("estimator saw %v, want [64 128]", stub.got)
	}

	// Backend closed/broken error is returned untouched.
	closedErr := errors.New("backend closed")
	stub.err = closedErr
	if _, err := enc.BatchMemory([]int{64}); !errors.Is(err, closedErr) {
		t.Fatalf("backend error = %v, want %v", err, closedErr)
	}
}
