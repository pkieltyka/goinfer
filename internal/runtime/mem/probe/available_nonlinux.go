//go:build !linux

package probe

// LinuxAvailableRAM returns zero when Linux meminfo is unavailable.
func LinuxAvailableRAM() int64 { return 0 }
