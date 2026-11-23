//go:build cuda

package kernels

import (
	"math"
	"os"
	"regexp"
	"runtime"
	"strings"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/launchgeom"
)

// TestEG2PackedAttn_TileTableAndSpecContract verifies the pure-Go tile-table
// and launch-geometry contract for eg2_packed_attn.cu without needing a GPU.
func TestEG2PackedAttn_TileTableAndSpecContract(t *testing.T) {
	for _, hd := range []int{16, 32, 256, 512} {
		spec, ok := launchgeom.EG2PackedAttnSpecForHD(hd)
		if !ok {
			t.Fatalf("missing spec for hd=%d", hd)
		}
		if spec.SharedMem == 0 || spec.SharedMem > launchgeom.EG2PackedAttnMaxDefaultShmem {
			t.Fatalf("%s sharedMem=%d exceeds default 48 KiB budget", spec.Entry, spec.SharedMem)
		}
		if spec.BlockX == 0 || spec.BlockX%32 != 0 {
			t.Fatalf("%s invalid blockX=%d", spec.Entry, spec.BlockX)
		}
		if spec.HD != hd || spec.HD%spec.WarpD != 0 || (spec.HD/spec.WarpD)%spec.DKSub != 0 {
			t.Fatalf("%s invalid D tiling: %+v", spec.Entry, spec)
		}
	}
	if _, ok := launchgeom.EG2PackedAttnSpecForHD(96); ok {
		t.Fatal("unsupported hd=96 unexpectedly matched a spec")
	}

	// Segment-local tile descriptors (SegLen, QLocal0) must be identical across
	// singleton, packed, split, and reordered batches, with non-overlapping SegStart.
	const bm = 32
	lens := []int{1, 17, 32, 33, 64, 65, 513}
	packed, err := launchgeom.BuildEG2PackedTiles(lens, bm)
	if err != nil {
		t.Fatalf("BuildEG2PackedTiles(packed): %v", err)
	}
	type localTile struct{ segLen, q0 int32 }
	wantBySeg := make([][]localTile, len(lens))
	for i, n := range lens {
		single, err := launchgeom.BuildEG2PackedTiles([]int{n}, bm)
		if err != nil {
			t.Fatalf("BuildEG2PackedTiles(single %d): %v", n, err)
		}
		for k := range single.SegStart {
			if single.SegStart[k] != 0 {
				t.Fatalf("singleton start = %d, want 0", single.SegStart[k])
			}
			wantBySeg[i] = append(wantBySeg[i], localTile{single.SegLen[k], single.QLocal0[k]})
		}
	}
	checkLayout := func(label string, order []int) {
		t.Helper()
		ordLens := make([]int, len(order))
		for i, idx := range order {
			ordLens[i] = lens[idx]
		}
		tab, err := launchgeom.BuildEG2PackedTiles(ordLens, bm)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		cursor := 0
		segStart := int32(0)
		for _, idx := range order {
			want := wantBySeg[idx]
			for _, wt := range want {
				if cursor >= len(tab.SegStart) {
					t.Fatalf("%s: too few tiles", label)
				}
				if tab.SegStart[cursor] != segStart || tab.SegLen[cursor] != wt.segLen || tab.QLocal0[cursor] != wt.q0 {
					t.Fatalf("%s tile %d = (%d,%d,%d), want (%d,%d,%d)",
						label, cursor, tab.SegStart[cursor], tab.SegLen[cursor], tab.QLocal0[cursor],
						segStart, wt.segLen, wt.q0)
				}
				cursor++
			}
			segStart += int32(lens[idx])
		}
		if cursor != len(tab.SegStart) {
			t.Fatalf("%s: got %d tiles, want %d", label, len(tab.SegStart), cursor)
		}
	}
	checkLayout("packed", []int{0, 1, 2, 3, 4, 5, 6})
	checkLayout("split-left", []int{0, 1, 2})
	checkLayout("split-right", []int{3, 4, 5, 6})
	checkLayout("reversed", []int{6, 5, 4, 3, 2, 1, 0})
	if len(packed.SegStart) == 0 {
		t.Fatal("empty packed tile table")
	}

	// Error cases.
	if _, err := launchgeom.BuildEG2PackedTiles(nil, 32); err == nil {
		t.Fatal("expected error for empty batch")
	}
	if _, err := launchgeom.BuildEG2PackedTiles([]int{4, 0}, 32); err == nil {
		t.Fatal("expected error for zero-length segment")
	}
	if _, err := launchgeom.BuildEG2PackedTiles([]int{4}, 15); err == nil {
		t.Fatal("expected error for invalid bm")
	}
}

// TestEG2PackedAttn_PTXStaticAndFMALint checks the compiled PTX for expected
// entry points and zero .local spills, and runs the FMA-lint rules against
// eg2_packed_attn.cu without needing a GPU.
func TestEG2PackedAttn_PTXStaticAndFMALint(t *testing.T) {
	ptxBytes := readPTX("eg2_packed_attn")
	ptx := string(ptxBytes)
	for _, hd := range []int{16, 32, 256, 512} {
		spec, _ := launchgeom.EG2PackedAttnSpecForHD(hd)
		if !strings.Contains(ptx, ".visible .entry "+spec.Entry+"(") {
			t.Fatalf("eg2_packed_attn.ptx missing entry point %s", spec.Entry)
		}
	}
	if strings.Contains(ptx, ".local") {
		t.Fatalf("eg2_packed_attn.ptx contains .local memory declarations (register spill)")
	}

	cuBytes, err := os.ReadFile("../eg2_packed_attn.cu")
	if err != nil {
		t.Fatalf("read eg2_packed_attn.cu: %v", err)
	}
	macAccum := regexp.MustCompile(`\b\w[\w.]*\s*\+=\s*[^;/]* \* `)
	macExpr := regexp.MustCompile(` \* [^;/]* [+\-] `)
	macExprRev := regexp.MustCompile(` [+\-] [^;/()]* \* `)
	compliant := regexp.MustCompile(`__fmaf_rn|__fmul_rn|__fadd_rn|__fma_rn|__dp4a|__shfl`)
	arrayIndex := regexp.MustCompile(`\[[^\]]*\]`)
	isDeclOrIndex := regexp.MustCompile(`^\s*(const\s+)?(unsigned\s+)?(int|long|short|char|size_t|void|float\*|double\*|__half\*|const int\*|const unsigned|constexpr)\b|` +
		`^\s*(const\s+)?[\w]+\s*\*+\s*\w+\s*=|\(long\)|\(int\)|\(unsigned`)
	forHeader := regexp.MustCompile(`\bfor\s*\([^)]*\)`)
	for i, line := range strings.Split(string(cuBytes), "\n") {
		code := line
		if k := strings.Index(code, "//"); k >= 0 {
			code = code[:k]
		}
		code = forHeader.ReplaceAllString(code, "")
		if strings.TrimSpace(code) == "" || compliant.MatchString(code) || isDeclOrIndex.MatchString(code) {
			continue
		}
		code = arrayIndex.ReplaceAllString(code, "[]")
		if macAccum.MatchString(code) || macExpr.MatchString(code) || macExprRev.MatchString(code) {
			t.Errorf("eg2_packed_attn.cu:%d bare float MAC: %s", i+1, strings.TrimSpace(line))
		}
	}
}

// TestEG2PackedAttn_DeviceParityAndBitIdentity exercises the compiled
// eg2_packed_attn.ptx kernels on a real CUDA device across real EG2 geometries
// (HD=256 sliding W=512 GQA 4/2, HD=512 full GQA 4/1), general scaling, and
// tiny fixture geometries (HD=16, HD=32). It checks:
//  1. 0 bytes CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES per entry point;
//  2. exact math.Float32bits identity across singleton, packed, split, and reordered batches;
//  3. zero cross-segment reads against NaN and +/-1e4 poisoned adjacent segments;
//  4. GPU vs FP32 CPU reference worst per-row cosine >= 0.999;
//  5. GPU vs f16-operand f64 tile reference worst per-row cosine >= 0.99999.
func TestEG2PackedAttn_DeviceParityAndBitIdentity(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	defer dev.ReleaseObjects()

	ptxBytes := readPTX("eg2_packed_attn")
	mod, err := dev.CompileLibrary(ptxBytes)
	if err != nil {
		t.Fatalf("CompileLibrary(eg2_packed_attn): %v", err)
	}
	rawMod, err := dev.Context().LoadModule(ptxBytes)
	if err != nil {
		t.Fatalf("LoadModule(eg2_packed_attn): %v", err)
	}
	defer func() { _ = rawMod.Close() }()
	pipes := map[int]Pipeline{}
	for _, hd := range []int{16, 32, 256, 512} {
		spec, _ := launchgeom.EG2PackedAttnSpecForHD(hd)
		p, err := dev.NewComputePipeline(mod, spec.Entry)
		if err != nil {
			t.Fatalf("NewComputePipeline(%s): %v", spec.Entry, err)
		}
		fn, err := rawMod.Function(spec.Entry)
		if err != nil {
			t.Fatalf("Function(%s): %v", spec.Entry, err)
		}
		localBytes, err := fn.Attribute(gc.FuncAttrLocalSizeBytes)
		if err != nil {
			t.Fatalf("FuncAttrLocalSizeBytes(%s): %v", spec.Entry, err)
		}
		numRegs, _ := fn.Attribute(gc.FuncAttrNumRegs)
		t.Logf("%s: local_bytes=%d regs=%d shared_bytes=%d blockX=%d", spec.Entry, localBytes, numRegs, spec.SharedMem, spec.BlockX)
		if localBytes != 0 {
			t.Fatalf("%s spilled %d bytes to local memory, want 0", spec.Entry, localBytes)
		}
		pipes[hd] = p
	}
	stream := dev.NewCommandQueue()

	runPacked := func(t *testing.T, hd, nH, nKV, window int, scale float32, q, k, v []float32, lens []int) []float32 {
		t.Helper()
		spec, ok := launchgeom.EG2PackedAttnSpecForHD(hd)
		if !ok {
			t.Fatalf("no spec for hd=%d", hd)
		}
		tab, err := launchgeom.BuildEG2PackedTiles(lens, spec.BM)
		if err != nil {
			t.Fatalf("BuildEG2PackedTiles: %v", err)
		}
		totalTokens := 0
		for _, n := range lens {
			totalTokens += n
		}
		qDim := nH * hd
		qb := gpu.NewBufferLenOf[float32](dev, len(q))
		kb := gpu.NewBufferLenOf[float32](dev, len(k))
		vb := gpu.NewBufferLenOf[float32](dev, len(v))
		ob := gpu.NewBufferLenOf[float32](dev, totalTokens*qDim)
		ssb := gpu.NewBufferLenOf[int32](dev, len(tab.SegStart))
		slb := gpu.NewBufferLenOf[int32](dev, len(tab.SegLen))
		q0b := gpu.NewBufferLenOf[int32](dev, len(tab.QLocal0))
		defer func() {
			dev.ReleaseBuf(qb)
			dev.ReleaseBuf(kb)
			dev.ReleaseBuf(vb)
			dev.ReleaseBuf(ob)
			dev.ReleaseBuf(ssb)
			dev.ReleaseBuf(slb)
			dev.ReleaseBuf(q0b)
		}()
		for b, src := range map[Buffer][]float32{qb: q, kb: k, vb: v} {
			if err := gpu.Upload(b, src); err != nil {
				t.Fatal(err)
			}
		}
		for b, src := range map[Buffer][]int32{ssb: tab.SegStart, slb: tab.SegLen, q0b: tab.QLocal0} {
			if err := gpu.Upload(b, src); err != nil {
				t.Fatal(err)
			}
		}
		// Preload output with NaN so any unwritten output lane fails immediately.
		out := make([]float32, totalTokens*qDim)
		for i := range out {
			out[i] = float32(math.NaN())
		}
		if err := gpu.Upload(ob, out); err != nil {
			t.Fatal(err)
		}
		cfg := LaunchConfig{
			GridX:          uint32(nH),
			GridY:          uint32(len(tab.SegStart)),
			GridZ:          1,
			BlockX:         spec.BlockX,
			BlockY:         1,
			BlockZ:         1,
			SharedMemBytes: spec.SharedMem,
		}
		if err := stream.Launch(pipes[hd], cfg,
			Arg(qb), Arg(kb), Arg(vb),
			Arg(ssb), Arg(slb), Arg(q0b),
			gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(window)),
			gpu.ArgValue(scale), gpu.ArgValue(int32(len(tab.SegStart))),
			Arg(ob)); err != nil {
			t.Fatalf("Launch(%s): %v", spec.Entry, err)
		}
		if err := stream.Sync(); err != nil {
			t.Fatalf("Sync(%s): %v", spec.Entry, err)
		}
		if err := gpu.Download(ob, out); err != nil {
			t.Fatalf("Download(%s): %v", spec.Entry, err)
		}
		return out
	}

	makeSeg := func(seed, n, nH, nKV, hd int) (q, k, v []float32) {
		q = make([]float32, n*nH*hd)
		k = make([]float32, n*nKV*hd)
		v = make([]float32, n*nKV*hd)
		// Unit-RMS-like realistic EG2 activations (post Q/K/V RMSNorm).
		for i := range q {
			q[i] = float32(math.Sin(float64(seed*997+i)*0.173)) * 0.85
		}
		for i := range k {
			k[i] = float32(math.Cos(float64(seed*613+i)*0.131)) * 0.90
		}
		for i := range v {
			v[i] = float32(math.Sin(float64(seed*409+i)*0.097)) * 0.75
		}
		return q, k, v
	}

	concatSegs := func(segs [][3][]float32, order []int) (q, k, v []float32) {
		for _, idx := range order {
			q = append(q, segs[idx][0]...)
			k = append(k, segs[idx][1]...)
			v = append(v, segs[idx][2]...)
		}
		return q, k, v
	}

	cases := []struct {
		name   string
		nH     int
		nKV    int
		hd     int
		window int
		scale  float32
		lens   []int
	}{
		{"real-eg2-sliding-hd256-scale1", 4, 2, 256, 512, 1.0, []int{1, 17, 32, 33, 65, 513}},
		{"real-eg2-full-hd512-scale1", 4, 1, 512, -1, 1.0, []int{1, 17, 32, 33, 65, 193}},
		{"real-eg2-sliding-hd256-invsqrt", 4, 2, 256, 48, float32(1.0 / math.Sqrt(256)), []int{19, 64, 129, 7}},
		{"real-eg2-full-hd512-scale025", 4, 1, 512, -1, 0.25, []int{5, 33, 70}},
		{"tiny-eg2-sliding-hd16", 4, 2, 16, 3, 1.0, []int{1, 3, 4, 17, 33}},
		{"tiny-eg2-full-hd32", 4, 1, 32, -1, 1.0, []int{1, 5, 32, 35}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			segs := make([][3][]float32, len(tc.lens))
			singleOuts := make([][]float32, len(tc.lens))
			qDim := tc.nH * tc.hd

			// 1. Run each segment as a singleton and verify FP32 CPU & f16-tile logic parity.
			worstCPU := 1.0
			worstLogic := 1.0
			for i, n := range tc.lens {
				q, k, v := makeSeg(i+1, n, tc.nH, tc.nKV, tc.hd)
				segs[i] = [3][]float32{q, k, v}
				out := runPacked(t, tc.hd, tc.nH, tc.nKV, tc.window, tc.scale, q, k, v, []int{n})
				singleOuts[i] = out

				// Check against exact FP32 CPU reference on all segments <= 193 (fast on CPU)
				// and sampled rows on longer segments (513) so CPU time stays <1s.
				rowStride := 1
				if n > 200 {
					rowStride = 17
				}
				cpuRef := eg2AttnRefFP32(q, k, v, n, tc.nH, tc.nKV, tc.hd, tc.window, tc.scale, rowStride)
				for row := 0; row < n; row += rowStride {
					for h := range tc.nH {
						off := row*qDim + h*tc.hd
						cos := eg2AttnCosineF32(out[off:off+tc.hd], cpuRef[off:off+tc.hd])
						if !(cos >= worstCPU) {
							worstCPU = cos
						}
					}
				}
				if n <= 70 {
					spec, _ := launchgeom.EG2PackedAttnSpecForHD(tc.hd)
					logicRef := eg2AttnRefF16Tiles(q, k, v, n, tc.nH, tc.nKV, tc.hd, tc.window, tc.scale, spec.BM, spec.BN)
					for row := range n {
						for h := range tc.nH {
							off := row*qDim + h*tc.hd
							cos := eg2AttnCosineF32(out[off:off+tc.hd], logicRef[off:off+tc.hd])
							if !(cos >= worstLogic) {
								worstLogic = cos
							}
						}
					}
				}
			}
			t.Logf("%s: worst per-row cosine vs FP32 CPU = %.8f, vs f16-tile f64 = %.8f", tc.name, worstCPU, worstLogic)
			if !(worstCPU >= 0.999) {
				t.Fatalf("%s: worst FP32 CPU cosine %.8f < 0.999", tc.name, worstCPU)
			}
			if !(worstLogic >= 0.99999) {
				t.Fatalf("%s: worst f16-tile logic cosine %.8f < 0.99999", tc.name, worstLogic)
			}

			// 2. Verify exact math.Float32bits identity across packed, split, and reordered layouts.
			checkOrder := func(label string, order []int) {
				t.Helper()
				ordLens := make([]int, len(order))
				for i, idx := range order {
					ordLens[i] = tc.lens[idx]
				}
				q, k, v := concatSegs(segs, order)
				got := runPacked(t, tc.hd, tc.nH, tc.nKV, tc.window, tc.scale, q, k, v, ordLens)
				off := 0
				for _, idx := range order {
					want := singleOuts[idx]
					slice := got[off : off+len(want)]
					for j := range want {
						if math.Float32bits(slice[j]) != math.Float32bits(want[j]) {
							t.Fatalf("%s seg %d (len=%d) element %d bit mismatch: got %08x (%g), want %08x (%g)",
								label, idx, tc.lens[idx], j,
								math.Float32bits(slice[j]), slice[j],
								math.Float32bits(want[j]), want[j])
						}
					}
					off += len(want)
				}
			}
			allOrder := make([]int, len(tc.lens))
			revOrder := make([]int, len(tc.lens))
			for i := range tc.lens {
				allOrder[i] = i
				revOrder[i] = len(tc.lens) - 1 - i
			}
			mid := len(tc.lens) / 2
			checkOrder("packed-all", allOrder)
			checkOrder("split-left", allOrder[:mid])
			checkOrder("split-right", allOrder[mid:])
			checkOrder("reordered-reverse", revOrder)

			// 3. No cross-segment reads: flank a target segment with NaN and +/-1e4 poison segments.
			targetIdx := min(len(tc.lens)-1, 3)
			targetLen := tc.lens[targetIdx]
			poisonFill := func(n int, val float32) [3][]float32 {
				q := make([]float32, n*tc.nH*tc.hd)
				k := make([]float32, n*tc.nKV*tc.hd)
				v := make([]float32, n*tc.nKV*tc.hd)
				for i := range q {
					q[i] = val
				}
				for i := range k {
					k[i] = val
				}
				for i := range v {
					v[i] = val
				}
				return [3][]float32{q, k, v}
			}
			poisonSegs := [][3][]float32{
				poisonFill(19, float32(math.NaN())),
				segs[targetIdx],
				poisonFill(37, 1e4),
			}
			pq, pk, pv := concatSegs(poisonSegs, []int{0, 1, 2})
			pOut := runPacked(t, tc.hd, tc.nH, tc.nKV, tc.window, tc.scale, pq, pk, pv, []int{19, targetLen, 37})
			gotTarget := pOut[19*qDim : (19+targetLen)*qDim]
			wantTarget := singleOuts[targetIdx]
			for j := range wantTarget {
				if math.Float32bits(gotTarget[j]) != math.Float32bits(wantTarget[j]) {
					t.Fatalf("poisoned neighbors leaked into target segment at elem %d: got %08x (%g), want %08x (%g)",
						j, math.Float32bits(gotTarget[j]), gotTarget[j], math.Float32bits(wantTarget[j]), wantTarget[j])
				}
			}
		})
	}
}

func eg2AttnRefFP32(q, k, v []float32, n, nH, nKV, hd, window int, scale float32, rowStride int) []float32 {
	qDim, kvDim, group := nH*hd, nKV*hd, nH/nKV
	out := make([]float32, n*qDim)
	scores := make([]float32, n)
	for row := 0; row < n; row += rowStride {
		lo, hi := 0, n
		if window >= 0 {
			lo = max(0, row-window)
			hi = min(n, row+window+1)
		}
		for h := range nH {
			kvh := h / group
			qRow := q[row*qDim+h*hd : row*qDim+(h+1)*hd]
			mx := float32(math.Inf(-1))
			for j := lo; j < hi; j++ {
				kRow := k[j*kvDim+kvh*hd : j*kvDim+(kvh+1)*hd]
				var dot float32
				for d := range hd {
					dot += qRow[d] * kRow[d]
				}
				sc := dot * scale
				scores[j] = sc
				if sc > mx {
					mx = sc
				}
			}
			var sum float32
			for j := lo; j < hi; j++ {
				e := float32(math.Exp(float64(scores[j] - mx)))
				scores[j] = e
				sum += e
			}
			inv := 1 / sum
			dst := out[row*qDim+h*hd : row*qDim+(h+1)*hd]
			for j := lo; j < hi; j++ {
				w := scores[j] * inv
				vRow := v[j*kvDim+kvh*hd : j*kvDim+(kvh+1)*hd]
				for d := range hd {
					dst[d] += w * vRow[d]
				}
			}
		}
	}
	return out
}

func eg2AttnRefF16Tiles(q, k, v []float32, n, nH, nKV, hd, window int, scale float32, bm, bn int) []float32 {
	qDim, kvDim, group := nH*hd, nKV*hd, nH/nKV
	out := make([]float32, n*qDim)
	for q0 := 0; q0 < n; q0 += bm {
		q1 := min(n, q0+bm)
		kStart, kEnd := 0, n
		if window >= 0 {
			kStart = max(0, q0-window)
			kEnd = min(n, (q1-1)+window+1)
		}
		for row := q0; row < q1; row++ {
			lo, hi := 0, n-1
			if window >= 0 {
				lo = max(0, row-window)
				hi = min(n-1, row+window)
			}
			for h := range nH {
				kvh := h / group
				mRun, lRun := math.Inf(-1), 0.0
				acc := make([]float64, hd)
				for k0 := kStart; k0 < kEnd; k0 += bn {
					nk := min(bn, kEnd-k0)
					sc := make([]float64, nk)
					valid := make([]bool, nk)
					tmax := math.Inf(-1)
					anyValid := false
					for j := range nk {
						kl := k0 + j
						if kl < lo || kl > hi {
							continue
						}
						valid[j] = true
						anyValid = true
						var dot float64
						for d := range hd {
							dot += float64(f16rne(q[row*qDim+h*hd+d])) * float64(f16rne(k[kl*kvDim+kvh*hd+d]))
						}
						sc[j] = dot * float64(scale)
						if sc[j] > tmax {
							tmax = sc[j]
						}
					}
					if !anyValid {
						continue
					}
					mNew := tmax
					alpha := 0.0
					if lRun > 0 {
						mNew = math.Max(mRun, tmax)
						alpha = math.Exp(mRun - mNew)
					}
					var sum float64
					for j := range nk {
						if !valid[j] {
							continue
						}
						e := math.Exp(sc[j] - mNew)
						sc[j] = e
						sum += e
					}
					for d := range acc {
						acc[d] *= alpha
					}
					for j := range nk {
						if !valid[j] {
							continue
						}
						pj := float64(f16rne(float32(sc[j])))
						kl := k0 + j
						for d := range hd {
							acc[d] += pj * float64(f16rne(v[kl*kvDim+kvh*hd+d]))
						}
					}
					lRun = alpha*lRun + sum
					mRun = mNew
				}
				dst := out[row*qDim+h*hd : row*qDim+(h+1)*hd]
				for d := range hd {
					dst[d] = float32(acc[d] / lRun)
				}
			}
		}
	}
	return out
}

func eg2AttnCosineF32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
			return math.NaN()
		}
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
