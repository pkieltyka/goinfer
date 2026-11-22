//go:build !linux || !amd64

package vulkan

import (
	"fmt"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"runtime"
)

func openAccelerator(*embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
	return nil, fmt.Errorf("vulkan: unsupported platform %s/%s (requires linux/amd64)", runtime.GOOS, runtime.GOARCH)
}
