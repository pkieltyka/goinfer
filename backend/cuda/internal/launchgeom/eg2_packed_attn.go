package launchgeom

import (
	"fmt"
	"math"
)

// EG2PackedAttnMaxDefaultShmem is the 48 KiB default shared-memory budget per
// block on sm_75+ (requiring no cuFuncSetAttribute opt-in).
const EG2PackedAttnMaxDefaultShmem = 48 * 1024

// EG2PackedAttnSpec describes the compile-time tile parameters and shared-memory
// budget for one eg2_packed_attn.cu entry point.
type EG2PackedAttnSpec struct {
	Entry     string
	HD        int
	BM        int
	BN        int
	WarpD     int
	DKSub     int
	BlockX    uint32
	SharedMem uint32
}

// EG2PackedAttnSpecForHD returns the eg2_packed_attn.cu launch specification
// for a supported head dimension (real EG2: 256, 512; tiny EG2 fixture: 16, 32).
// Shared memory budgets account for split-f16 (hi + lo) Q, K, and V staging
// slabs overlaid with the cross-warp score reduction buffer Ssh.
func EG2PackedAttnSpecForHD(hd int) (EG2PackedAttnSpec, bool) {
	const kpad = 8
	makeSpec := func(entry string, hd, bm, bn, warpD, dkSub int) EG2PackedAttnSpec {
		warpM := bm / 16
		dkTot := warpD * dkSub
		qBytes := 2 * bm * (hd + kpad) * 2
		kBytes := 2 * bn * (dkTot + kpad) * 2
		vBytes := 2 * dkTot * (bn + kpad) * 2
		sBytes := 0
		if warpD > 1 {
			sBytes = warpM * (warpD - 1) * 16 * bn * 4
		}
		kvBytes := max(kBytes, vBytes, sBytes)
		return EG2PackedAttnSpec{
			Entry:     entry,
			HD:        hd,
			BM:        bm,
			BN:        bn,
			WarpD:     warpD,
			DKSub:     dkSub,
			BlockX:    uint32(warpM * warpD * 32),
			SharedMem: uint32(qBytes + kvBytes),
		}
	}
	switch hd {
	case 256:
		return makeSpec("eg2_packed_attn_hd256", 256, 32, 32, 2, 32), true
	case 512:
		return makeSpec("eg2_packed_attn_hd512", 512, 16, 32, 4, 16), true
	case 16:
		return makeSpec("eg2_packed_attn_hd16", 16, 32, 32, 1, 16), true
	case 32:
		return makeSpec("eg2_packed_attn_hd32", 32, 32, 32, 1, 32), true
	default:
		return EG2PackedAttnSpec{}, false
	}
}

// EG2PackedTileTable holds per-segment fixed local query tiles for packed
// variable-length attention. Each entry belongs strictly to one segment so tile
// boundaries and arithmetic order never depend on other segments in the batch.
type EG2PackedTileTable struct {
	SegStart []int32
	SegLen   []int32
	QLocal0  []int32
}

// BuildEG2PackedTiles partitions each segment in lens independently into fixed
// local query tiles of height bm (0, bm, 2*bm, ... < len).
func BuildEG2PackedTiles(lens []int, bm int) (EG2PackedTileTable, error) {
	if bm <= 0 || bm%16 != 0 {
		return EG2PackedTileTable{}, fmt.Errorf("launchgeom: invalid packed attention tile height %d", bm)
	}
	if len(lens) == 0 {
		return EG2PackedTileTable{}, fmt.Errorf("launchgeom: empty packed attention batch")
	}
	const maxGridY = 65535
	totalTiles := 0
	totalRows := 0
	for i, n := range lens {
		if n <= 0 {
			return EG2PackedTileTable{}, fmt.Errorf("launchgeom: packed attention segment %d has invalid length %d", i, n)
		}
		if n > math.MaxInt32 {
			return EG2PackedTileTable{}, fmt.Errorf("launchgeom: packed attention segment %d length %d exceeds int32 range", i, n)
		}
		if totalRows > math.MaxInt32-n {
			return EG2PackedTileTable{}, fmt.Errorf("launchgeom: packed attention total rows exceed int32 range at segment %d", i)
		}
		tiles := 1 + (n-1)/bm
		if tiles > maxGridY-totalTiles {
			return EG2PackedTileTable{}, fmt.Errorf("launchgeom: packed attention tile count exceeds GridY limit %d", maxGridY)
		}
		totalRows += n
		totalTiles += tiles
	}
	tab := EG2PackedTileTable{
		SegStart: make([]int32, 0, totalTiles),
		SegLen:   make([]int32, 0, totalTiles),
		QLocal0:  make([]int32, 0, totalTiles),
	}
	start := 0
	for _, n := range lens {
		for q0 := 0; q0 < n; q0 += bm {
			tab.SegStart = append(tab.SegStart, int32(start))
			tab.SegLen = append(tab.SegLen, int32(n))
			tab.QLocal0 = append(tab.QLocal0, int32(q0))
		}
		start += n
	}
	return tab, nil
}
