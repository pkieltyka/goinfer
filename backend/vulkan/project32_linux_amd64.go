//go:build linux && amd64

package vulkan

import (
	"errors"
	"fmt"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
)

// projectChunk is one projection dispatch: rows [First, First+Rows) of the
// current forward, written by the stock 8-row kernel or the 32-row kernel.
type projectChunk struct {
	First  int
	Rows   int
	Wide   bool   // the 32-row kernel
	Units  uint64 // padded FMA work charged to the submission plan
	Groups int    // workgroups along Y
}

// project32Device is the product gate: the measured Radeon RX 9060 XT PCI
// product. Other products keep the stock 8-row projection.
func project32Device(vendor, product uint32) bool {
	return vendor == 0x1002 && product == 0x7590
}

// project32OptionalDecline reports whether an optional-pipeline creation
// failure means the driver lacks a feature rather than a broken device or an
// allocation failure. Only the former may fall back to stock for the encoder's
// lifetime; OOM, device loss and unknown failures remain errors.
func project32OptionalDecline(err error) bool {
	var re *vk.ResultError
	if !errors.As(err, &re) {
		return false
	}
	switch re.Code {
	case -7, -8, -9: // EXTENSION_NOT_PRESENT, FEATURE_NOT_PRESENT, INCOMPATIBLE_DRIVER
		return true
	}
	return false
}

// project32Cap is the greatest row count one 32-row dispatch may charge under
// submissionWork: 32*floor(submissionWork/(32*I16*O16)). Zero means the shape
// cannot use the wide kernel at all.
func project32Cap(in, out int) int {
	i16, o16 := tile(in, 16), tile(out, 16)
	if i16 <= 0 || o16 <= 0 {
		return 0
	}
	return 32 * int(submissionWork/(32*uint64(i16)*uint64(o16)))
}

// chunksWide reports whether any planned chunk needs the 32-row kernel.
func chunksWide(chunks []projectChunk) bool {
	for _, c := range chunks {
		if c.Wide {
			return true
		}
	}
	return false
}

// projectChunks plans the dispatches of one projection. With wide=false it is
// exactly the stock 8-row budget and grid; with wide=true it walks cap32-aligned
// chunks on the 32-row kernel and falls back to the stock kernel for a tail
// shorter than one 32-row block. The returned error matches the stock
// "one projection row exceeds the submission work limit" case.
func projectChunks(rows, in, out int) ([]projectChunk, error) {
	if rows <= 0 {
		return nil, fmt.Errorf("vulkan: projection with %d rows", rows)
	}
	i16, o16 := tile(in, 16), tile(out, 16)
	perGroup8 := product(8, i16, o16)
	maxRows8 := 8 * int(submissionWork/perGroup8)
	if maxRows8 == 0 {
		return nil, fmt.Errorf("vulkan: one projection row exceeds the submission work limit")
	}
	stock := func(first, n int) projectChunk {
		return projectChunk{First: first, Rows: n, Wide: false,
			Units: product(tile(n, 8), o16, i16), Groups: ceilDiv(n, 8)}
	}
	cap32 := project32Cap(in, out)
	if cap32 == 0 {
		var chunks []projectChunk
		for start := 0; start < rows; start += maxRows8 {
			chunks = append(chunks, stock(start, min(maxRows8, rows-start)))
		}
		return chunks, nil
	}
	var chunks []projectChunk
	for start := 0; start < rows; {
		n := min(cap32, rows-start)
		if n < 32 {
			for s := start; s < rows; s += maxRows8 {
				chunks = append(chunks, stock(s, min(maxRows8, rows-s)))
			}
			return chunks, nil
		}
		chunks = append(chunks, projectChunk{First: start, Rows: n, Wide: true,
			Units: product(tile(n, 32), o16, i16), Groups: ceilDiv(n, 32)})
		start += n
	}
	return chunks, nil
}
