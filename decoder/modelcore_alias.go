package decoder

import (
	"github.com/pkieltyka/goinfer/decoder/resident"
	"github.com/pkieltyka/goinfer/internal/modelcore/embed"
	"github.com/pkieltyka/goinfer/internal/modelcore/linalg"
)

// WeightMat is the weight matrix shared with resident backends.
type WeightMat = resident.WeightMat

// W8A8Op describes one projection in a shared-activation W8A8 batch.
type W8A8Op = linalg.W8A8Op

// W4A8Op describes one projection in a shared-activation W4A8 batch.
type W4A8Op = linalg.W4A8Op

// GGUFFile is a GGUF checkpoint opened by OpenGGUFMmap.
type GGUFFile = embed.GGUFFile

// WrapF32 wraps row-major weights without copying the backing slice.
func WrapF32(w []float32, rows, cols int) WeightMat { return linalg.WrapF32(w, rows, cols) }
