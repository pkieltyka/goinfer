//go:build cuda && (eg2_fusions || (eg2_combined && eg2_fp16))

package cuda

import _ "embed"

//go:embed testdata/eg2_fusions.ptx
var embeddedEG2FusionsPTX []byte

func init() { eg2FusionsPTX = embeddedEG2FusionsPTX }
