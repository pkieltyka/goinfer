//go:build !linux

package mmap

import "testing"

func TestAutoBudget_withoutLinuxMeminfo(t *testing.T) {
	if got := AvailableRAM(); got != 0 {
		t.Errorf("AvailableRAM = %d, want 0", got)
	}
	if got := AutoBudget(); got != 8<<30 {
		t.Errorf("AutoBudget = %d, want 8 GiB", got)
	}
}
