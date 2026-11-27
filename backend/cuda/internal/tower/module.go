//go:build cuda

package tower

import gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"

// Module identifies a compiled kernel module unit for loader dispatch.
type Module string

const (
	ModuleViT  Module = "vit"
	ModuleBase Module = "base"
)

// Compile loads a PTX or cubin image onto a CUDA device.
type Compile func([]byte) (gpu.Library, error)

// ModuleLoader compiles or selects a compiled library for a named kernel module.
type ModuleLoader func(Module, []byte) (gpu.Library, error)

// PTXLoader returns a ModuleLoader that always compiles the PTX directly.
func PTXLoader(compile Compile) ModuleLoader {
	return func(_ Module, ptx []byte) (gpu.Library, error) { return compile(ptx) }
}
