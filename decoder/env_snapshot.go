package decoder

import (
	"os"
	"sync"
)

// The decoder's eight environment knobs are process-wide settings, and they are parsed **once, on first
// use** rather than at package init. That distinction is the whole of goinfer#70: importing this
// package — or any backend whose init links it, such as backend/cuda when an application only wants
// text embeddings — must perform no environment reads before main runs.
//
// Semantics are unchanged. Each knob still resolves to one value per process: the first decoder
// construction or generation freezes the snapshot and every read site keeps reading the same global,
// so nothing refreshes mid-run and a Generate loop cannot observe a half-changed setting. The defaults
// and parsing are the same expressions the package-level initialisers used.
var settingsOnce sync.Once

// ensureSettings parses the knobs on first use. Every read site is downstream of it: a decoder is
// constructed by Load or NewModel before any weight conversion, repack or dispatch reads a knob, and
// the Generate entry points call it too so a Model built some other way is still covered.
func ensureSettings() {
	settingsOnce.Do(readSettings)
}

func readSettings() {
	deltaNetTiming = os.Getenv("GOINFER_DELTANET_TIMING") != ""
	decodeTiming = os.Getenv("GOINFER_DECODE_TIMING") != ""
	fakeQuantScheme = os.Getenv("GOINFER_FAKEQUANT")
	fakeQuantF32Act = os.Getenv("GOINFER_FAKEQUANT_ACT") == "f32"
	fakeQuantExpertsOnly = os.Getenv("GOINFER_FAKEQUANT_EXPERTS") == "1"
	w4a8SplitHalfRepackEnabled = os.Getenv("GOINFER_W4A8_SPLITHALF") != ""
	w4a8BatchEnabled = envBoolDefault("GOINFER_W4A8_BATCH", w4a8BatchDefault)
	cpuFusedGateUp = envBoolDefault("GOINFER_CPU_FUSED_GATEUP", fusedGateUpDefault)
}
