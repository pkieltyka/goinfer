package quant

import "github.com/pkieltyka/goinfer/internal/modelcore/embed"

// SafetensorsFile is a safetensors checkpoint used by quantization readers.
type SafetensorsFile = embed.SafetensorsFile

// OpenSafetensorsMmap opens a checkpoint with read-only mapped weights. Close
// the file after the last reader of any tensor backed by the mapping finishes.
func OpenSafetensorsMmap(path string) (*SafetensorsFile, error) {
	return embed.OpenSafetensorsMmap(path)
}
