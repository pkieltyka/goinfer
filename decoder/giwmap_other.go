//go:build !darwin && !windows

package decoder

import "github.com/pkieltyka/goinfer/internal/modelcore/mmap"

// mapGIW maps a .giw with the model core's MAP_PRIVATE read-only mapping. darwin maps it MAP_SHARED instead, for
// Metal's no-copy buffers (giwmap_darwin.go), and Windows has its own (giwmap_windows.go); nothing here wires the mapping.
func mapGIW(path string) ([]byte, error) { return mmap.MapReadOnly(path) }

// unmapGIW releases what mapGIW returned.
func unmapGIW(data []byte) error { return mmap.Unmap(data) }
