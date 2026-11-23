//go:build linux

package probe

import "testing"

// Available exists because Physical (total physical RAM) is what the fit guard budgeted against, and a
// fixed fraction of TOTAL RAM assumes nothing else on the machine ever needs more than the rest, which is false on a
// real machine with a browser or IDE open (docs/measurements/cold-user-2026-09-07-macbook-arm64.md, R13). This tests
// the shared meminfo-field parser both readers use.
const sampleMeminfo = `MemTotal:       16311288 kB
MemFree:         1234567 kB
MemAvailable:    5766224 kB
Buffers:          234567 kB
Cached:          3456789 kB
`

func TestMeminfoField_realShape(t *testing.T) {
	if got, want := meminfoField(sampleMeminfo, "MemTotal:"), int64(16311288)*1024; got != want {
		t.Errorf("MemTotal: got %d, want %d", got, want)
	}
	if got, want := meminfoField(sampleMeminfo, "MemAvailable:"), int64(5766224)*1024; got != want {
		t.Errorf("MemAvailable: got %d, want %d", got, want)
	}
}

func TestMeminfoField_missingKeyIsUnknown(t *testing.T) {
	if got := meminfoField(sampleMeminfo, "MemAvailable:"); got == 0 {
		t.Fatal("test fixture should have MemAvailable — this would make the next assertion vacuous")
	}
	if got := meminfoField(sampleMeminfo, "NoSuchKey:"); got != 0 {
		t.Errorf("missing key: got %d, want 0 (unknown)", got)
	}
}

func TestMeminfoField_unexpectedUnitIsUnknown(t *testing.T) {
	// A missing or unexpected unit means "we do not know", never a guess off by 1024.
	const badUnit = "MemTotal:       16311288 MB\n"
	if got := meminfoField(badUnit, "MemTotal:"); got != 0 {
		t.Errorf("unexpected unit: got %d, want 0 (unknown, not off by 1024)", got)
	}
}

// Available must actually read live /proc/meminfo on this real Linux box — not
// cached like Physical, since availability is the whole point of the field existing.
func TestAvailable_readsRealMeminfo(t *testing.T) {
	got := Available()
	if got <= 0 {
		t.Fatalf("Available() = %d on a real Linux box with a real /proc/meminfo", got)
	}
	total := Physical()
	if total <= 0 {
		t.Fatal("Physical() = 0 on a real Linux box — test environment problem, not this code's")
	}
	if got > total {
		t.Errorf("available (%d) > total (%d) — impossible", got, total)
	}
}

func TestLinuxAvailableRAM_parser(t *testing.T) {
	tests := []struct {
		name    string
		meminfo string
		want    int64
	}{
		{"real", sampleMeminfo, 5766224 * 1024},
		{"unreadable", "", 0},
		{"missing", "MemTotal: 42 kB\n", 0},
		{"malformed", "MemAvailable: nope kB\n", 0},
		{"after malformed", "MemAvailable: nope kB\nMemAvailable: 3 kB\n", 3 * 1024},
		{"first parseable", "MemAvailable: 4 kB\nMemAvailable: 5 kB\n", 4 * 1024},
		{"unit agnostic", "MemAvailable: 4\n", 4 * 1024},
		{"first parsed zero", "MemAvailable: 0 kB\nMemAvailable: 2 kB\n", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := linuxAvailableRAM(tt.meminfo); got != tt.want {
				t.Errorf("got %d bytes, want %d", got, tt.want)
			}
		})
	}
}
