package embeddinggemma2

import (
	"errors"
	"fmt"
	"math"
)

// ErrUnsupportedBatchShape reports a well-formed batch shape that the active
// backend or encoder cannot run (for example exceeding per-input or aggregate
// token limits, or passing multiple inputs to a singleton-only accelerator).
var ErrUnsupportedBatchShape = errors.New("embeddinggemma2: unsupported batch shape")

// BatchMemoryQuality classifies how a backend obtained a BatchMemoryEstimate.
type BatchMemoryQuality uint8

const (
	// BatchMemoryUnknown means the backend provides no usable estimate for an
	// otherwise admissible batch shape. AdditionalPeakBytes is zero.
	BatchMemoryUnknown BatchMemoryQuality = iota
	// BatchMemoryUpperBound means AdditionalPeakBytes is a conservative upper
	// bound over explicitly tracked backend device buffer allocations for the
	// next forward. It does not guarantee physical device availability against
	// driver-private overhead or concurrent allocations by other processes.
	BatchMemoryUpperBound
	// BatchMemoryApproximate means AdditionalPeakBytes is an estimate with a
	// backend-documented margin rather than a strict bound.
	BatchMemoryApproximate
)

func (q BatchMemoryQuality) String() string {
	switch q {
	case BatchMemoryUpperBound:
		return "upper_bound"
	case BatchMemoryApproximate:
		return "approximate"
	default:
		return "unknown"
	}
}

// BatchMemoryEstimate describes the device memory a prepared batch needs on the
// attached accelerator. It is advisory only, performs no forward, makes no
// device allocation, and reserves no memory against concurrent processes.
//
// Callers comparing against post-attach free bytes compare AdditionalPeakBytes;
// enforcing a total process cap also requires observed current usage including
// resident weights, caches, helper buffers, and driver context overhead.
type BatchMemoryEstimate struct {
	// AdditionalPeakBytes is the high-water increase in live explicitly
	// allocated backend device buffers relative to immediately before the call.
	// With a known quality, zero means existing resident buffers are reused
	// without new device buffer allocations. With BatchMemoryUnknown it is zero.
	AdditionalPeakBytes int64
	// ResidentWeightBytes is the persistent device buffer memory held for
	// uploaded model weights (excluding synthetic helpers, scratch, and driver
	// context overhead). Zero when unknown or when no accelerator is attached.
	ResidentWeightBytes int64
	Quality             BatchMemoryQuality
}

// BatchMemoryEstimator is an optional interface on an attached Accelerator.
// One length prices EmbedIDs (Forward with keepLayers=false); multiple lengths
// price EmbedIDsBatch (ForwardBatch with keepLayers=false). Implementations
// perform no forward, make no device allocation, reject invalid or overflowing
// lengths with an ordinary validation error, reject unsupported shapes with
// ErrUnsupportedBatchShape, and preserve backend closed/broken errors.
type BatchMemoryEstimator interface {
	BatchMemory(preparedLengths []int) (BatchMemoryEstimate, error)
}

func validatePreparedLengths(preparedLengths []int) (total, maxLen int, err error) {
	if len(preparedLengths) == 0 {
		return 0, 0, errors.New("embeddinggemma2: empty batch")
	}
	for i, n := range preparedLengths {
		if n <= 0 {
			return 0, 0, fmt.Errorf("embeddinggemma2: input %d has invalid length %d", i, n)
		}
		if n > math.MaxInt-total {
			return 0, 0, errors.New("embeddinggemma2: total batch length overflows int")
		}
		total += n
		if n > maxLen {
			maxLen = n
		}
	}
	return total, maxLen, nil
}

func (m *Model) checkPreparedLengthCap(preparedLengths []int) error {
	limit := MaxTokens
	if m != nil && m.cfg.MaxPositions > 0 && m.cfg.MaxPositions < limit {
		limit = m.cfg.MaxPositions
	}
	for i, n := range preparedLengths {
		if n > limit {
			return fmt.Errorf("%w: input %d has %d tokens, over limit %d", ErrUnsupportedBatchShape, i, n, limit)
		}
	}
	return nil
}

// BatchMemory estimates the device working memory for a prepared batch on the
// encoder's current path without running a forward or allocating device memory.
// One length prices EmbedIDs; multiple lengths price EmbedIDsBatch. When no
// accelerator is attached or the attached accelerator does not implement
// BatchMemoryEstimator, admissible shapes return BatchMemoryUnknown.
func (e *Encoder) BatchMemory(preparedLengths []int) (BatchMemoryEstimate, error) {
	if e == nil || e.m == nil {
		return BatchMemoryEstimate{}, errors.New("embeddinggemma2: nil encoder")
	}
	if _, _, err := validatePreparedLengths(preparedLengths); err != nil {
		return BatchMemoryEstimate{}, err
	}
	if e.accel != nil {
		if est, ok := e.accel.(BatchMemoryEstimator); ok {
			return est.BatchMemory(preparedLengths)
		}
		if err := e.m.checkPreparedLengthCap(preparedLengths); err != nil {
			return BatchMemoryEstimate{}, err
		}
		if len(preparedLengths) > 1 {
			if _, canBatch := e.accel.(batchForwarder); !canBatch {
				return BatchMemoryEstimate{}, fmt.Errorf("%w: the %s accelerator cannot run a batch", ErrUnsupportedBatchShape, e.accel.Name())
			}
		}
		return BatchMemoryEstimate{Quality: BatchMemoryUnknown}, nil
	}
	if err := e.m.checkPreparedLengthCap(preparedLengths); err != nil {
		return BatchMemoryEstimate{}, err
	}
	return BatchMemoryEstimate{Quality: BatchMemoryUnknown}, nil
}
