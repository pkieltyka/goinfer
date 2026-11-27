//go:build cuda && eg2_combined && eg2_fp16

package eg2

func init() { wantEG2FP16Selector = true }
