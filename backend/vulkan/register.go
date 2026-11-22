// Package vulkan registers the direct Vulkan EmbeddingGemma 2 text accelerator.
// Importing the package registers a factory. The system loader and AMD device are
// opened only when the caller selects "vulkan".
package vulkan

import "github.com/pkieltyka/goinfer/embeddinggemma2"

//go:generate sh shaders/generate.sh

func init() { embeddinggemma2.RegisterAccelerator("vulkan", openAccelerator) }
