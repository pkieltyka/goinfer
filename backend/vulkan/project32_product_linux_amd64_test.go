//go:build linux && amd64

package vulkan

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"testing"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
)

// TestProject32Eligibility pins the product gate: only the measured RX 9060 XT
// PCI product qualifies; every other product stays on the stock route.
func TestProject32Eligibility(t *testing.T) {
	for _, tc := range []struct {
		vendor, product uint32
		want            bool
	}{
		{0x1002, 0x7590, true},
		{0x1002, 0x7591, false},
		{0x1002, 0x164e, false},
		{0x10de, 0x7590, false},
		{0, 0, false},
	} {
		if got := project32Device(tc.vendor, tc.product); got != tc.want {
			t.Fatalf("%04x:%04x: %v, want %v", tc.vendor, tc.product, got, tc.want)
		}
	}
}

// TestProject32OptionalDecline classifies only missing driver capability as a
// benign optional decline. Allocation, loss and unknown creation failures must
// stay on their existing error paths.
func TestProject32OptionalDecline(t *testing.T) {
	for _, tc := range []struct {
		code int32
		want bool
	}{
		{-7, true},  // VK_ERROR_EXTENSION_NOT_PRESENT
		{-8, true},  // VK_ERROR_FEATURE_NOT_PRESENT
		{-9, true},  // VK_ERROR_INCOMPATIBLE_DRIVER
		{-1, false}, // OUT_OF_HOST_MEMORY
		{-2, false}, // OUT_OF_DEVICE_MEMORY
		{-4, false}, // DEVICE_LOST
		{-13, false},
	} {
		err := fmt.Errorf("create compute pipeline: %w", &vk.ResultError{Op: "create compute pipeline", Code: tc.code})
		if got := project32OptionalDecline(err); got != tc.want {
			t.Fatalf("code %d: %v, want %v", tc.code, got, tc.want)
		}
	}
	if project32OptionalDecline(errors.New("plain")) {
		t.Fatal("a plain error must not be a benign decline")
	}
}

// TestProject32ShippedSPIRV pins both shipped artifacts. The stock pipeline is
// unchanged; the optional pipeline is the exact measured artifact.
func TestProject32ShippedSPIRV(t *testing.T) {
	for _, tc := range []struct{ name, want string }{
		{"project", "aee8d9454fa3af610726ecdd54b0d4074a1809bc8eb47867ca956c2ec80b92dc"},
		{"project32", "6d3972611cf681813d0d008780fd5dfa53cead07d54f6fa8ccaab4d7b356002e"},
	} {
		code, err := shaders.ReadFile("shaders/" + tc.name + ".spv")
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(code)); got != tc.want {
			t.Fatalf("%s SPIR-V %s, want %s", tc.name, got, tc.want)
		}
	}
}

func TestProjectChunksCrossover(t *testing.T) {
	in, out := 512, 1024
	if got := project32Cap(in, out); got != 1536 {
		t.Fatalf("cap32 %d, want 1536", got)
	}
	for _, tc := range []struct {
		rows   int
		wide   []bool
		groups []int
	}{
		{rows: 11, wide: []bool{false}, groups: []int{2}},
		{rows: 31, wide: []bool{false}, groups: []int{4}},
		{rows: 32, wide: []bool{true}, groups: []int{1}},
		{rows: 33, wide: []bool{true}, groups: []int{2}},
		{rows: 650, wide: []bool{true}, groups: []int{21}},
	} {
		chunks, err := projectChunks(tc.rows, in, out)
		if err != nil {
			t.Fatal(err)
		}
		if len(chunks) != len(tc.wide) {
			t.Fatalf("%d rows: %d chunks, want %d", tc.rows, len(chunks), len(tc.wide))
		}
		if got := chunksWide(chunks); got != tc.wide[0] {
			t.Fatalf("%d rows: chunksWide %v, want %v", tc.rows, got, tc.wide[0])
		}
		for i := range chunks {
			if chunks[i].Wide != tc.wide[i] || chunks[i].Groups != tc.groups[i] || chunks[i].First != 0 {
				t.Fatalf("%d rows chunk %d: %+v, want wide %v groups %d", tc.rows, i, chunks[i], tc.wide[i], tc.groups[i])
			}
		}
		if tc.rows >= 32 {
			want := product(tile(tc.rows, 32), tile(out, 16), tile(in, 16))
			if chunks[0].Units != want {
				t.Fatalf("%d rows units %d, want %d", tc.rows, chunks[0].Units, want)
			}
		}
	}
}

func TestProjectChunksTailAndCap(t *testing.T) {
	if got := project32Cap(2048, 3072); got != 128 {
		t.Fatalf("cap32 %d, want 128", got)
	}
	chunks, err := projectChunks(650, 2048, 3072)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 6 {
		t.Fatalf("%d chunks, want 6", len(chunks))
	}
	for i := 0; i < 5; i++ {
		if !chunks[i].Wide || chunks[i].Rows != 128 || chunks[i].First != 128*i {
			t.Fatalf("chunk %d: %+v", i, chunks[i])
		}
	}
	if last := chunks[5]; last.Wide || last.Rows != 10 || last.First != 640 {
		t.Fatalf("tail: %+v, want stock 10 rows at 640", last)
	}
	if got := project32Cap(8192, 4096); got != 0 {
		t.Fatalf("cap32 %d, want 0", got)
	}
	chunks, err = projectChunks(650, 8192, 4096)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range chunks {
		if c.Wide {
			t.Fatalf("wide chunk on a cap32=0 shape: %+v", c)
		}
	}
}

func TestProjectChunksOddShapes(t *testing.T) {
	for _, dims := range [][2]int{{513, 1023}, {1023, 513}, {17, 65}, {31, 33}} {
		in, out := dims[0], dims[1]
		for _, rows := range []int{31, 32, 33} {
			chunks, err := projectChunks(rows, in, out)
			if err != nil {
				t.Fatalf("%d/%d/%d: %v", rows, in, out, err)
			}
			total := 0
			for _, c := range chunks {
				total += c.Rows
				if c.Rows < 32 && c.Wide {
					t.Fatalf("%d/%d/%d: wide chunk below the block: %+v", rows, in, out, c)
				}
			}
			if total != rows || chunks[0].First != 0 {
				t.Fatalf("%d/%d/%d: chunks cover %d rows from %d", rows, in, out, total, chunks[0].First)
			}
		}
	}
}
