//go:build amd64 && modelcore_nopopcnt

package linalg

// modelcore_nopopcnt: the Hamming search takes its portable path (see force.go).
func init() {
	hasPOPCNT = false
	forcedFallbacks = append(forcedFallbacks, "nopopcnt")
}
