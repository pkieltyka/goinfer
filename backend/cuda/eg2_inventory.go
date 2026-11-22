//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

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
		Backend: "cuda", Name: adapter.Name, VendorID: 0x10de, MemoryKind: embeddinggemma2.MemoryDedicated,
		Memory:       embeddinggemma2.DeviceMemory{TotalBytes: int64(min(adapter.TotalBytes, math.MaxInt64))},
		TextBatching: true,
	}
	if adapter.UUID != "" {
		row.ID = "cuda:" + adapter.UUID
		row.Alias = "cuda:" + strconv.Itoa(adapter.Ordinal)
	}
	return row
}

func probeCUDADeviceMemory(adapter gpu.Adapter) (reading embeddinggemma2.DeviceMemory, err error) {
	dev, err := gpu.CreateSelectedDevice(adapter)
	if err != nil {
		return reading, classifyCUDAAttach(err)
	}
	defer func() { err = errors.Join(err, dev.Context().Close()) }()
	return cudaMemory(dev)
}
