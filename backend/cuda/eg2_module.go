//go:build cuda

package cuda

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/tower"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

type eg2Module = tower.Module

const (
	eg2ModuleViT                  = tower.ModuleViT
	eg2ModuleBase                 = tower.ModuleBase
	eg2ModuleTransposed eg2Module = "transposed"
	eg2ModuleFusion     eg2Module = "fusion"
	eg2ModuleFFN        eg2Module = "ffn"
	eg2ModulePacked     eg2Module = "packed"
)

type cudaModuleCompile = tower.Compile
type towerModuleLoader = tower.ModuleLoader

var towerPTXLoader = tower.PTXLoader

//go:embed testdata/eg2-sm120/vit.sm120.cubin
var eg2ViTSM120 []byte

//go:embed testdata/eg2-sm120/base.sm120.cubin
var eg2BaseSM120 []byte

//go:embed testdata/eg2-sm120/transposed.sm120.cubin
var eg2TransposedSM120 []byte

// Optional images are linked alongside their PTX by the selector build tags.
var eg2FusionSM120, eg2FFNSM120, eg2PackedSM120 []byte
var eg2FFNFP16PTX, eg2PackedAttnPTX []byte

func eg2ModuleImage(unit eg2Module) (cubin []byte, ptxSHA string) {
	switch unit {
	case eg2ModuleViT:
		return eg2ViTSM120, "6b2f57e7360a574530e90e5c137975af6bf918e921fbbdfbb659430415f43b8d"
	case eg2ModuleBase:
		return eg2BaseSM120, "7ee3fdb069e37cd3a1f00364ac1ae8ade5f346de1af92f91ce51d7e51904c4c6"
	case eg2ModuleTransposed:
		return eg2TransposedSM120, "b965898a6eed3233e7cef13298e8d29517d9efbd6e50d2b7faffdda6cfa181a4"
	case eg2ModuleFusion:
		return eg2FusionSM120, "446e34045eeafac5e229894d90f429379d0d4c8c1768c56515e1ae6210198c62"
	case eg2ModuleFFN:
		return eg2FFNSM120, "4211568d2b58019d7375bbec6a6e4f2560f2640a4d4808929bbfde352ccdd358"
	case eg2ModulePacked:
		return eg2PackedSM120, "d0ea4dac4c894d51c7df5e5632c69a7303a1b99dfb41e63d2d24ac3412ec01fa"
	default:
		return nil, ""
	}
}

// eg2SM120Loader only substitutes a cubin for the PTX it was built from, on
// the exact attached compute capability. Each image rejection gets one PTX
// attempt, preserving both driver errors if that fails too.
func eg2SM120Loader(compile cudaModuleCompile, capability embeddinggemma2.CUDAComputeCapability) towerModuleLoader {
	if capability.Major != 12 || capability.Minor != 0 {
		return towerPTXLoader(compile)
	}
	return func(unit eg2Module, ptx []byte) (gpu.Library, error) {
		image, ptxSHA := eg2ModuleImage(unit)
		if len(image) == 0 {
			return compile(ptx)
		}
		sum := sha256.Sum256(ptx)
		if hex.EncodeToString(sum[:]) != ptxSHA {
			return compile(ptx)
		}
		lib, err := compile(image)
		if !errors.Is(err, gc.ErrNoBinaryForGPU) && !errors.Is(err, gc.ErrInvalidImage) {
			return lib, err
		}
		lib, ptxErr := compile(ptx)
		if ptxErr != nil {
			return gpu.Library{}, &eg2ImageFallbackError{image: err, ptx: ptxErr}
		}
		return lib, nil
	}
}

type eg2ImageFallbackError struct {
	image, ptx error
}

func (e *eg2ImageFallbackError) Error() string {
	return fmt.Sprintf("sm_120 image rejected (%v); PTX load failed: %v", e.image, e.ptx)
}

func (e *eg2ImageFallbackError) Unwrap() []error { return []error{e.image, e.ptx} }
