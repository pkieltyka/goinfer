//go:build gpu

// Command chat is the WebGPU-accelerated build of goinfer's demo REPL (internal/chatapp).
// Blank-importing ./gpu registers the "webgpu" decoder backend; living in the ./gpu submodule
// keeps cgo/webgpu out of the pure-Go root module graph. Run with `--backend webgpu`.
package main

import (
	_ "github.com/pkieltyka/goinfer/backend/webgpu"
	"github.com/pkieltyka/goinfer/internal/app/chatapp"
)

func main() { chatapp.Main() }
