//go:build !linux || !amd64

package vulkan

import (
	"fmt"
	"runtime"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func openAccelerator(*embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
	return nil, unsupportedDevice()
}

// Devices is unsupported outside Linux/amd64.
func Devices() ([]DeviceInfo, error) { return Probe(embeddinggemma2.ProbeOptions{}) }

// Probe is unsupported outside Linux/amd64.
func Probe(embeddinggemma2.ProbeOptions) ([]DeviceInfo, error) { return nil, unsupportedDevice() }

func openSelectedAccelerator(*embeddinggemma2.Model, string) (embeddinggemma2.Accelerator, error) {
	return nil, unsupportedDevice()
}

func unsupportedDevice() error {
	return fmt.Errorf("%w: vulkan requires linux/amd64 (running on %s/%s)", embeddinggemma2.ErrBackendUnavailable, runtime.GOOS, runtime.GOARCH)
}
