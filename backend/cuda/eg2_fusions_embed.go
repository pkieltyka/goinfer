//go:build cuda && eg2_fusions

package cuda

import _ "embed"

//go:embed testdata/eg2_fusions.ptx
var embeddedEG2FusionsPTX []byte

func init() { eg2FusionsPTX = embeddedEG2FusionsPTX }
