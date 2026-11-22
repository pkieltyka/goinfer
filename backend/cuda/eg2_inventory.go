//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func probeCUDADevices(opts embeddinggemma2.ProbeOptions) ([]embeddinggemma2.DeviceInfo, error) {
	adapters, err := gpu.Adapters()
	rows, memoryErr := describeCUDADevices(adapters, opts, probeCUDADeviceMemory)
	return rows, errors.Join(classifyCUDAAttach(err), memoryErr)
}

func describeCUDADevices(adapters []gpu.Adapter, opts embeddinggemma2.ProbeOptions,
	readMemory func(gpu.Adapter) (embeddinggemma2.DeviceMemory, error)) ([]embeddinggemma2.DeviceInfo, error) {
	counts := make(map[string]int)
	for _, adapter := range adapters {
		counts[strings.ToLower(adapter.UUID)]++
	}
	var rows []embeddinggemma2.DeviceInfo
	var failures []error
	for _, adapter := range adapters {
		if !cudaEligible(adapter) {
			continue
		}
		if counts[strings.ToLower(adapter.UUID)] > 1 {
			adapter.UUID = ""
		}
		row := cudaDeviceInfo(adapter)
		if opts.IncludeMemory && row.ID != "" {
			reading, err := readMemory(adapter)
			if err != nil {
				failures = append(failures, fmt.Errorf("%s memory: %w", row.ID, err))
			} else {
				row.Memory = reading
			}
		}
		rows = append(rows, row)
	}
	return rows, errors.Join(failures...)
}

func cudaDeviceInfo(adapter gpu.Adapter) embeddinggemma2.DeviceInfo {
	row := embeddinggemma2.DeviceInfo{
		Backend: "cuda", Name: adapter.Name, VendorID: 0x10de,
		Memory:       embeddinggemma2.DeviceMemory{TotalBytes: int64(min(adapter.TotalBytes, math.MaxInt64))},
		TextBatching: true,
	}
	if adapter.IntegratedKnown {
		row.MemoryKind = embeddinggemma2.MemoryDedicated
		if adapter.Integrated {
			row.MemoryKind = embeddinggemma2.MemoryShared
		}
	}
	if adapter.UUID != "" {
		row.ID = "cuda:" + adapter.UUID
		row.Alias = "cuda:" + strconv.Itoa(adapter.Ordinal)
	}
	return row
}

type cudaMemoryContext interface {
	cudaMemoryReader
	Close() error
}

type cudaMemoryProbe struct {
	mu      sync.Mutex
	pending map[string]cudaMemoryContext // failed closes retain ownership
}

var cudaProbe cudaMemoryProbe

func probeCUDADeviceMemory(adapter gpu.Adapter) (embeddinggemma2.DeviceMemory, error) {
	return cudaProbe.read(adapter, func(adapter gpu.Adapter) (cudaMemoryContext, error) {
		dev, err := gpu.CreateSelectedDevice(adapter)
		if err != nil {
			return nil, classifyCUDAAttach(err)
		}
		return dev.Context(), nil
	})
}

func (p *cudaMemoryProbe) read(adapter gpu.Adapter, open func(gpu.Adapter) (cudaMemoryContext, error)) (embeddinggemma2.DeviceMemory, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	key := strings.ToLower(adapter.UUID)
	if pending := p.pending[key]; pending != nil {
		if err := pending.Close(); err != nil {
			return embeddinggemma2.DeviceMemory{}, fmt.Errorf("retry temporary context close: %w", err)
		}
		delete(p.pending, key)
	}
	context, err := open(adapter)
	if err != nil {
		return embeddinggemma2.DeviceMemory{}, err
	}
	reading, err := cudaMemory(context)
	if closeErr := context.Close(); closeErr != nil {
		if p.pending == nil {
			p.pending = make(map[string]cudaMemoryContext)
		}
		p.pending[key] = context
		return embeddinggemma2.DeviceMemory{}, errors.Join(err, fmt.Errorf("close temporary context: %w", closeErr))
	}
	return reading, err
}
