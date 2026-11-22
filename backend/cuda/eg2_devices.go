//go:build cuda

package cuda

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func selectedEG2Factory(combined bool) func(*embeddinggemma2.Model, string) (embeddinggemma2.Accelerator, error) {
	return func(m *embeddinggemma2.Model, token string) (embeddinggemma2.Accelerator, error) {
		if !validCUDADeviceToken(token) {
			return nil, fmt.Errorf("%w: invalid CUDA device identifier %q", embeddinggemma2.ErrDeviceUnavailable, token)
		}
		adapters, err := gpu.Adapters()
		if err != nil {
			return nil, classifyCUDAAttach(err)
		}
		selected, err := resolveCUDADevice(token, adapters)
		if err != nil {
			return nil, err
		}
		acc, err := newEG2AccelOn(m, combined, func() (*Device, error) {
			return gpu.CreateSelectedDevice(selected)
		})
		if err != nil {
			return nil, err
		}
		return acc, nil
	}
}

func classifyCUDAAttach(err error) error {
	switch {
	case errors.Is(err, gc.ErrOutOfMemory):
		return fmt.Errorf("%w: %w", embeddinggemma2.ErrDeviceOutOfMemory, err)
	case errors.Is(err, gpu.ErrAdapterChanged), errors.Is(err, gpu.ErrDriverInit) && errors.Is(err, gc.ErrNoDevice):
		return fmt.Errorf("%w: %w", embeddinggemma2.ErrNoDevice, err)
	case errors.Is(err, gpu.ErrDriverInit):
		return fmt.Errorf("%w: %w", embeddinggemma2.ErrBackendUnavailable, err)
	default:
		return err
	}
}

func cudaEligible(a gpu.Adapter) bool {
	return a.Major > ptxTargetMajor || a.Major == ptxTargetMajor && a.Minor >= ptxTargetMinor
}

func validCUDADeviceToken(token string) bool {
	if token == "" {
		return false
	}
	prefix, uuid, ok := strings.Cut(token, "-")
	if ok {
		if !strings.EqualFold(prefix, "GPU") && !strings.EqualFold(prefix, "MIG") {
			return false
		}
		if len(uuid) != 36 || uuid[8] != '-' || uuid[13] != '-' || uuid[18] != '-' || uuid[23] != '-' {
			return false
		}
		hexUUID := strings.ReplaceAll(uuid, "-", "")
		_, err := hex.DecodeString(hexUUID)
		return len(hexUUID) == 32 && err == nil
	}
	return strings.IndexFunc(token, func(r rune) bool { return r < '0' || r > '9' }) < 0
}

func resolveCUDADevice(token string, adapters []gpu.Adapter) (gpu.Adapter, error) {
	if !validCUDADeviceToken(token) {
		return gpu.Adapter{}, fmt.Errorf("%w: invalid CUDA device identifier %q", embeddinggemma2.ErrDeviceUnavailable, token)
	}
	for _, a := range adapters {
		if !strings.EqualFold(a.UUID, token) && strconv.Itoa(a.Ordinal) != token {
			continue
		}
		if a.UUID == "" {
			break
		}
		if !cudaEligible(a) {
			return gpu.Adapter{}, fmt.Errorf("%w: CUDA ordinal %d has compute capability %d.%d, need %d.%d",
				embeddinggemma2.ErrNoDevice, a.Ordinal, a.Major, a.Minor, ptxTargetMajor, ptxTargetMinor)
		}
		for _, other := range adapters {
			if a.Ordinal != other.Ordinal && strings.EqualFold(a.UUID, other.UUID) {
				return gpu.Adapter{}, fmt.Errorf("%w: CUDA ordinals %d and %d report the same UUID %s",
					embeddinggemma2.ErrAmbiguousDevice, a.Ordinal, other.Ordinal, a.UUID)
			}
		}
		return a, nil
	}
	return gpu.Adapter{}, fmt.Errorf("%w: no CUDA match for %q", embeddinggemma2.ErrNoDevice, token)
}

var _ embeddinggemma2.DeviceMemoryReporter = (*eg2Accel)(nil)
var _ embeddinggemma2.AttachedDeviceReporter = (*eg2Accel)(nil)

// AttachedDevice returns the identity captured from the opened CUDA handle.
func (a *eg2Accel) AttachedDevice() embeddinggemma2.DeviceInfo { return a.identity }

// DeviceMemory returns the driver-reported headroom of the attached card.
func (a *eg2Accel) DeviceMemory() (reading embeddinggemma2.DeviceMemory, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return reading, fmt.Errorf("cuda: EmbeddingGemma 2 accelerator is closed")
	}
	err = a.ops.do(func() error {
		var readErr error
		reading, readErr = cudaMemory(a.ops.dev.Context())
		return readErr
	})
	return reading, err
}

func cudaMemory(dev interface {
	MemInfo() (uint64, uint64, error)
}) (embeddinggemma2.DeviceMemory, error) {
	free, total, err := dev.MemInfo()
	if err != nil {
		return embeddinggemma2.DeviceMemory{}, err
	}
	if free > total || total > math.MaxInt64 {
		return embeddinggemma2.DeviceMemory{}, fmt.Errorf("cuda: invalid memory reading: %d free of %d bytes", free, total)
	}
	return embeddinggemma2.DeviceMemory{
		AvailableBytes: int64(free), TotalBytes: int64(total), Basis: embeddinggemma2.MemoryLive,
	}, nil
}
