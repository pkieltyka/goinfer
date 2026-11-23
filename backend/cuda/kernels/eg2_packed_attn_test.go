//go:build cuda

package kernels

import (
	"fmt"
	"math"
	"os"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

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

	// Error and overflow cases.
	if _, err := launchgeom.BuildEG2PackedTiles(nil, 32); err == nil {
		t.Fatal("expected error for empty batch")
	}
	if _, err := launchgeom.BuildEG2PackedTiles([]int{4, 0}, 32); err == nil {
		t.Fatal("expected error for zero-length segment")
	}
	if _, err := launchgeom.BuildEG2PackedTiles([]int{4}, 15); err == nil {
		t.Fatal("expected error for invalid bm")
	}
	if _, err := launchgeom.BuildEG2PackedTiles([]int{math.MaxInt}, 32); err == nil {
		t.Fatal("expected error for math.MaxInt segment length")
	}
	if _, err := launchgeom.BuildEG2PackedTiles([]int{math.MaxInt32, 1}, 32); err == nil {
		t.Fatal("expected error for total rows exceeding int32")
	}
	if _, err := launchgeom.BuildEG2PackedTiles([]int{65536 * 32}, 32); err == nil {
		t.Fatal("expected error for tile count exceeding GridY 65535")
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

	// Original sinusoidal generator preserved verbatim so the three previously
	// failing high-cancellation fixtures remain active parity gates.
	makeSinSeg := func(seed, n, nH, nKV, hd int) (q, k, v []float32) {
		q = make([]float32, n*nH*hd)
		k = make([]float32, n*nKV*hd)
		v = make([]float32, n*nKV*hd)
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

	makeRandSeg := func(seed, n, nH, nKV, hd int) (q, k, v []float32) {
		q = make([]float32, n*nH*hd)
		k = make([]float32, n*nKV*hd)
		v = make([]float32, n*nKV*hd)
		state := uint64(seed)*0x9e3779b97f4a7c15 + 0xbf58476d1ce4e5b9
		nextF32 := func() float32 {
			state += 0x9e3779b97f4a7c15
			z := state
			z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
			z = (z ^ (z >> 27)) * 0x94d049bb133111eb
			z ^= z >> 31
			return float32(int64(z>>40)-int64(1<<23)) / float32(1<<23)
		}
		for i := range q {
			q[i] = nextF32() * 0.85
		}
		for i := range k {
			k[i] = nextF32() * 0.90
		}
		for i := range v {
			v[i] = nextF32() * 0.75
		}
		return q, k, v
	}

	makeHighRangeSeg := func(seed, n, nH, nKV, hd int) (q, k, v []float32) {
		q = make([]float32, n*nH*hd)
		k = make([]float32, n*nKV*hd)
		v = make([]float32, n*nKV*hd)
		// Construct high-score / high-dynamic-range logits at scale=1.0
		// independent of hd: q[d] = +/- qAmp / sqrt(hd), k[d] = +/- kAmp / sqrt(hd),
		// so dot(q, k) == qAmp * kAmp (~+50..+72 peak). The late-tile peak
		// forces a multi-tile online-softmax max jump while runner-ups at
		// peak-1.8, peak-12 (FP16 subnormal), peak-18 and peak-25 (FP16
		// underflow to 0) exercise dynamic range after subtracting tile max.
		group := nH / nKV
		invSqrt := float32(1.0 / math.Sqrt(float64(hd)))
		for pos := range n {
			for kvh := range nKV {
				kOff := pos*nKV*hd + kvh*hd
				vOff := kOff
				kAmp := float32(8.0) // ~peak - 12 across background keys (cumulative FP16 subnormal test)
				switch {
				case pos == n-1:
					kAmp = 10.0 // late-tile peak (+60..+72)
				case pos == max(0, n-2):
					kAmp = 9.7 // close runner-up (~peak - 1.8)
				case pos == max(0, n/2):
					kAmp = 8.0 // mid-tile runner-up (~peak - 12, FP16 subnormal)
				case pos == min(n-1, 17):
					kAmp = 7.0 // early-tile runner-up (~peak - 18, FP16 underflow)
				case pos == min(n-1, 3):
					kAmp = 5.8 // very-early runner-up (~peak - 25, FP16 underflow)
				case pos%3 == 0:
					kAmp = float32(math.Sin(float64(seed*311+pos*29+kvh*13))) * 4.0
				}
				for d := range hd {
					sign := float32(1.0)
					if ((d + kvh + seed) & 1) == 1 {
						sign = -1.0
					}
					k[kOff+d] = sign * invSqrt * kAmp
					v[vOff+d] = float32(math.Cos(float64(seed*521+pos*73+kvh*37+d)*0.113)) * 0.8
				}
			}
			for h := range nH {
				kvh := h / group
				qOff := pos*nH*hd + h*hd
				qAmp := float32(6.0 + 0.4*float64(h))
				for d := range hd {
					sign := float32(1.0)
					if ((d + kvh + seed) & 1) == 1 {
						sign = -1.0
					}
					q[qOff+d] = sign * invSqrt * qAmp
				}
			}
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
		gen    string // "sin", "rand", "highrange"
	}{
		{"real-eg2-sliding-hd256-scale1", 4, 2, 256, 512, 1.0, []int{1, 17, 32, 33, 65, 513, 514}, "sin"},
		{"real-eg2-full-hd512-scale1", 4, 1, 512, -1, 1.0, []int{1, 17, 32, 33, 65, 193}, "sin"},
		{"real-eg2-sliding-hd256-invsqrt", 4, 2, 256, 48, float32(1.0 / math.Sqrt(256)), []int{19, 64, 129, 7}, "sin"},
		{"real-eg2-sliding-hd256-rand", 4, 2, 256, 512, 1.0, []int{17, 65, 514, 600}, "rand"},
		{"real-eg2-sliding-hd256-highrange", 4, 2, 256, 512, 1.0, []int{7, 33, 97, 193}, "highrange"},
		{"real-eg2-full-hd512-highrange", 4, 1, 512, -1, 1.0, []int{7, 33, 97, 193}, "highrange"},
		{"real-eg2-full-hd512-scale025", 4, 1, 512, -1, 0.25, []int{5, 33, 70}, "sin"},
		{"tiny-eg2-sliding-hd16", 4, 2, 16, 3, 1.0, []int{1, 3, 4, 17, 33}, "sin"},
		{"tiny-eg2-full-hd32", 4, 1, 32, -1, 1.0, []int{1, 5, 32, 35}, "sin"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			segs := make([][3][]float32, len(tc.lens))
			singleOuts := make([][]float32, len(tc.lens))
			qDim := tc.nH * tc.hd

			// 1. Run each segment as a singleton and verify FP32 CPU & split-f16 logic parity.
			worstCPU := 1.0
			worstCPUInfo := ""
			worstLogic := 1.0
			for i, n := range tc.lens {
				var q, k, v []float32
				switch tc.gen {
				case "highrange":
					q, k, v = makeHighRangeSeg(i+1, n, tc.nH, tc.nKV, tc.hd)
				case "rand":
					q, k, v = makeRandSeg(i+1, n, tc.nH, tc.nKV, tc.hd)
				default:
					q, k, v = makeSinSeg(i+1, n, tc.nH, tc.nKV, tc.hd)
				}
				segs[i] = [3][]float32{q, k, v}
				out := runPacked(t, tc.hd, tc.nH, tc.nKV, tc.window, tc.scale, q, k, v, []int{n})
				singleOuts[i] = out

				// Check against exact FP32 CPU reference on all rows for n <= 193,
				// and on strided + explicit sliding-window boundary rows for n > 200.
				var rows []int
				if n <= 200 {
					rows = make([]int, n)
					for r := range n {
						rows[r] = r
					}
				} else {
					seen := map[int]bool{}
					addRow := func(r int) {
						if r >= 0 && r < n && !seen[r] {
							seen[r] = true
							rows = append(rows, r)
						}
					}
					for r := 0; r < n; r += 19 {
						addRow(r)
					}
					// Explicitly verify both sliding-window boundary edges around radius and length.
					for _, r := range []int{0, 1, 2, tc.window - 1, tc.window, tc.window + 1, n - tc.window - 2, n - tc.window - 1, n - tc.window, n - 2, n - 1} {
						addRow(r)
					}
				}
				cpuRef := eg2AttnRefFP32Rows(q, k, v, n, tc.nH, tc.nKV, tc.hd, tc.window, tc.scale, rows)
				for _, row := range rows {
					for h := range tc.nH {
						off := row*qDim + h*tc.hd
						cos, refNorm, maxAbs := eg2AttnAssertCosine(t, tc.name+" FP32 CPU", i, n, row, h, out[off:off+tc.hd], cpuRef[off:off+tc.hd], 0.999)
						if cos < worstCPU {
							worstCPU = cos
							worstCPUInfo = eg2AttnRowDiag(i, n, row, h, refNorm, maxAbs)
						}
					}
				}
				if n <= 70 {
					spec, _ := launchgeom.EG2PackedAttnSpecForHD(tc.hd)
					logicRef := eg2AttnRefSplitF16Tiles(q, k, v, n, tc.nH, tc.nKV, tc.hd, tc.window, tc.scale, spec.BM, spec.BN)
					for row := range n {
						for h := range tc.nH {
							off := row*qDim + h*tc.hd
							cos, _, _ := eg2AttnAssertCosine(t, tc.name+" split-f16 f64", i, n, row, h, out[off:off+tc.hd], logicRef[off:off+tc.hd], 0.99999)
							if cos < worstLogic {
								worstLogic = cos
							}
						}
					}
				}
			}
			t.Logf("%s: worst FP32 CPU cos = %.8f (%s), worst split-f16 f64 cos (n<=70) = %.8f", tc.name, worstCPU, worstCPUInfo, worstLogic)

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
	// 4. Masked NaN/Inf/FP16-overflow isolation within a single union tile,
	// including layout-change bit-identity on the masked finite rows.
	t.Run("masked-nonfinite-isolation", func(t *testing.T) {
		for _, hd := range []int{16, 256} {
			for _, kind := range []struct {
				name  string
				val   float32
				isKey bool
			}{
				{"V-NaN", float32(math.NaN()), false},
				{"V-Inf", float32(math.Inf(1)), false},
				{"K-NaN", float32(math.NaN()), true},
			} {
				const (
					n    = 4
					nH   = 4
					nKV  = 2
					win  = 1
					colD = 7
				)
				q := make([]float32, n*nH*hd)
				k := make([]float32, n*nKV*hd)
				v := make([]float32, n*nKV*hd)
				for i := range v {
					v[i] = 1.0
				}
				if kind.isKey {
					k[0*nKV*hd+0*hd+colD] = kind.val
				} else {
					v[0*nKV*hd+0*hd+colD] = kind.val
				}
				out := runPacked(t, hd, nH, nKV, win, 1.0, q, k, v, []int{n})
				qDim := nH * hd
				for row := range n {
					for h := range nH {
						kvh := h / (nH / nKV)
						eligible := (row <= win) && (kvh == 0)
						for d := range hd {
							got := float64(out[row*qDim+h*hd+d])
							expectNonFinite := eligible && (kind.isKey || d == colD)
							if expectNonFinite {
								if !math.IsNaN(got) && !math.IsInf(got, 0) {
									t.Fatalf("hd=%d %s row=%d h=%d d=%d: got finite %g, want nonfinite", hd, kind.name, row, h, d, got)
								}
							} else {
								if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-1.0) > 1e-5 {
									t.Fatalf("hd=%d %s row=%d h=%d d=%d: got %g, want finite 1.0", hd, kind.name, row, h, d, got)
								}
							}
						}
					}
				}
				// Verify layout-change bit-identity on the finite rows (rows 2..3) when packed after a neighbor.
				if kind.name == "V-NaN" {
					nq, nk, nv := makeSinSeg(9, 13, nH, nKV, hd)
					pq, pk, pv := concatSegs([][3][]float32{{nq, nk, nv}, {q, k, v}}, []int{0, 1})
					pOut := runPacked(t, hd, nH, nKV, win, 1.0, pq, pk, pv, []int{13, n})
					gotSeg := pOut[13*qDim : (13+n)*qDim]
					for row := 2; row < n; row++ {
						for j := row * qDim; j < (row+1)*qDim; j++ {
							if math.Float32bits(gotSeg[j]) != math.Float32bits(out[j]) {
								t.Fatalf("hd=%d V-NaN packed row=%d bit mismatch at %d: got %08x (%g), want %08x (%g)",
									hd, row, j, math.Float32bits(gotSeg[j]), gotSeg[j], math.Float32bits(out[j]), out[j])
							}
						}
					}
				}
			}

			// Non-round-exact witness (V = 1.0002f): mutating ONLY V[0, 0] to NaN,
			// +Inf, or finite FP16-overflow 65520f must leave EVERY unaffected row
			// (rows 2..3) and EVERY unaffected column (d != 0 on rows 0..1)
			// bit-identical to the finite baseline!
			const (
				n    = 4
				nH   = 4
				nKV  = 2
				win  = 1
				colD = 0
			)
			q := make([]float32, n*nH*hd)
			k := make([]float32, n*nKV*hd)
			vBase := make([]float32, n*nKV*hd)
			for i := range vBase {
				vBase[i] = 1.0002
			}
			baseOut := runPacked(t, hd, nH, nKV, win, 1.0, q, k, vBase, []int{n})
			qDim := nH * hd

			for _, mut := range []struct {
				name     string
				val      float32
				isFinite bool
			}{
				{"NaN", float32(math.NaN()), false},
				{"Inf", float32(math.Inf(1)), false},
				{"FP16Overflow65520", 65520.0, true},
			} {
				vMut := append([]float32(nil), vBase...)
				vMut[0*nKV*hd+0*hd+colD] = mut.val
				mutOut := runPacked(t, hd, nH, nKV, win, 1.0, q, k, vMut, []int{n})
				for row := range n {
					for h := range nH {
						kvh := h / (nH / nKV)
						admits := (row <= win) && (kvh == 0)
						for d := range hd {
							idx := row*qDim + h*hd + d
							if admits && d == colD {
								got := float64(mutOut[idx])
								if mut.isFinite {
									want := (65520.0 + float64(row+1)*1.0002) / float64(row+2)
									if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 1e-1 {
										t.Fatalf("hd=%d witness %s row=%d h=%d d=%d: got %g, want finite %g", hd, mut.name, row, h, d, got, want)
									}
								} else if !math.IsNaN(got) && !math.IsInf(got, 0) {
									t.Fatalf("hd=%d witness %s row=%d h=%d d=%d: got finite %g, want nonfinite", hd, mut.name, row, h, d, got)
								}
							} else if math.Float32bits(mutOut[idx]) != math.Float32bits(baseOut[idx]) {
								t.Fatalf("hd=%d witness %s leaked into unaffected row=%d h=%d d=%d: got %08x (%g), baseline %08x (%g)",
									hd, mut.name, row, h, d,
									math.Float32bits(mutOut[idx]), mutOut[idx],
									math.Float32bits(baseOut[idx]), baseOut[idx])
							}
						}
					}
				}
			}
		}

		// Real EG2 sliding boundary (n=520, W=512, HD=256): poison V[0, 17]=NaN and V[519, 200]=+Inf.
		const (
			n   = 520
			nH  = 4
			nKV = 2
			hd  = 256
			win = 512
		)
		q := make([]float32, n*nH*hd)
		k := make([]float32, n*nKV*hd)
		v := make([]float32, n*nKV*hd)
		for i := range v {
			v[i] = 1.0
		}
		v[0*nKV*hd+0*hd+17] = float32(math.NaN())
		v[(n-1)*nKV*hd+1*hd+200] = float32(math.Inf(1))
		out := runPacked(t, hd, nH, nKV, win, 1.0, q, k, v, []int{n})
		qDim := nH * hd
		for _, row := range []int{0, 6, 7, 512, 513, 519} {
			g0 := float64(out[row*qDim+0*hd+17])
			if row <= win {
				if !math.IsNaN(g0) {
					t.Fatalf("W=512 boundary row=%d h=0 d=17: got %g, want NaN", row, g0)
				}
			} else if math.IsNaN(g0) || math.IsInf(g0, 0) || math.Abs(g0-1.0) > 1e-5 {
				t.Fatalf("W=512 boundary row=%d h=0 d=17: got %g, want finite 1.0", row, g0)
			}
			g2 := float64(out[row*qDim+2*hd+200])
			if row >= (n-1)-win {
				if !math.IsNaN(g2) && !math.IsInf(g2, 0) {
					t.Fatalf("W=512 boundary row=%d h=2 d=200: got %g, want nonfinite", row, g2)
				}
			} else if math.IsNaN(g2) || math.IsInf(g2, 0) || math.Abs(g2-1.0) > 1e-5 {
				t.Fatalf("W=512 boundary row=%d h=2 d=200: got %g, want finite 1.0", row, g2)
			}
		}
	})

	// 5. Clipped sliding-window endpoint test (n=514, W=512): plant a strong key
	// at k=513 with distinct V so query 0 (|0-513|=513 > 512) excludes it while
	// query 1 (|1-513|=512 <= 512) attends it strongly.
	t.Run("clipped-endpoint-514-w512", func(t *testing.T) {
		const (
			n   = 514
			nH  = 4
			nKV = 2
			hd  = 256
			win = 512
		)
		q := make([]float32, n*nH*hd)
		k := make([]float32, n*nKV*hd)
		v := make([]float32, n*nKV*hd)
		for pos := range n {
			for kvh := range nKV {
				for d := range hd {
					v[pos*nKV*hd+kvh*hd+d] = 1.0
				}
			}
		}
		for h := range nH {
			q[0*nH*hd+h*hd+0] = 10.0
			q[1*nH*hd+h*hd+0] = 10.0
		}
		for kvh := range nKV {
			k[513*nKV*hd+kvh*hd+0] = 10.0 // score = +100 for q0/q1
			for d := range hd {
				v[513*nKV*hd+kvh*hd+d] = -5.0
			}
		}
		out := runPacked(t, hd, nH, nKV, win, 1.0, q, k, v, []int{n})
		qDim := nH * hd
		for h := range nH {
			// Query 0 excludes k=513 -> must equal +1.0.
			g0 := float64(out[0*qDim+h*hd+0])
			if math.IsNaN(g0) || math.Abs(g0-1.0) > 1e-4 {
				t.Fatalf("q=0 h=%d: got %g, want +1.0 (k=513 must be masked)", h, g0)
			}
			// Query 1 includes k=513 (score +100 vs 0) -> must equal -5.0.
			g1 := float64(out[1*qDim+h*hd+0])
			if math.IsNaN(g1) || math.Abs(g1-(-5.0)) > 1e-4 {
				t.Fatalf("q=1 h=%d: got %g, want -5.0 (k=513 must be attended)", h, g1)
			}
		}
	})

	// 6. General scale / window edge cases: score below -1e30f normalizes via
	// -INFINITY tile-max initializer, and window=MaxInt32 does not overflow.
	t.Run("extreme-negative-score-and-large-window", func(t *testing.T) {
		const (
			n   = 4
			nH  = 4
			nKV = 2
			hd  = 16
		)
		q := make([]float32, n*nH*hd)
		k := make([]float32, n*nKV*hd)
		v := make([]float32, n*nKV*hd)
		for pos := range n {
			for h := range nH {
				q[pos*nH*hd+h*hd+0] = 1.0
			}
			for kvh := range nKV {
				k[pos*nKV*hd+kvh*hd+0] = -1.0 // dot = -1.0 -> score = -1e35 with scale=1e35
				for d := range hd {
					v[pos*nKV*hd+kvh*hd+d] = 2.5
				}
			}
		}
		out := runPacked(t, hd, nH, nKV, math.MaxInt32, 1e35, q, k, v, []int{n})
		for i, got := range out {
			if math.IsNaN(float64(got)) || math.Abs(float64(got)-2.5) > 1e-4 {
				t.Fatalf("extreme-negative-score idx=%d: got %g, want 2.5", i, got)
			}
		}
	})
}

func eg2AttnRowDiag(segIdx, segLen, row, head int, refNorm, maxAbs float64) string {
	return fmt.Sprintf("seg=%d(len=%d) row=%d h=%d refNorm=%.4g maxAbs=%.4g", segIdx, segLen, row, head, refNorm, maxAbs)
}

func eg2AttnAssertCosine(t *testing.T, label string, segIdx, segLen, row, head int, got, want []float32, minCos float64) (cos, refNorm, maxAbs float64) {
	t.Helper()
	var sumSq float64
	for i := range want {
		sumSq += float64(want[i]) * float64(want[i])
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	refNorm = math.Sqrt(sumSq)
	cos = eg2AttnCosineF32(got, want)
	if math.IsNaN(cos) || math.IsInf(cos, 0) || cos < minCos {
		t.Fatalf("%s seg=%d(len=%d) row=%d head=%d: cosine=%.8f (min %.6f, refNorm=%.6g, maxAbs=%.6g)",
			label, segIdx, segLen, row, head, cos, minCos, refNorm, maxAbs)
	}
	return cos, refNorm, maxAbs
}

func eg2AttnRefFP32Rows(q, k, v []float32, n, nH, nKV, hd, window int, scale float32, rows []int) []float32 {
	qDim, kvDim, group := nH*hd, nKV*hd, nH/nKV
	out := make([]float32, n*qDim)
	scores := make([]float32, n)
	for _, row := range rows {
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

func splitF16(x float32) (hi, lo float64) {
	h := f16rne(x)
	hi = float64(h)
	lo = float64(f16rne(x - h))
	return hi, lo
}

func eg2AttnRefSplitF16Tiles(q, k, v []float32, n, nH, nKV, hd, window int, scale float32, bm, bn int) []float32 {
	qDim, kvDim, group := nH*hd, nKV*hd, nH/nKV
	out := make([]float32, n*qDim)
	for q0 := 0; q0 < n; q0 += bm {
		q1 := min(n, q0+bm)
		kStart, kEnd := 0, n
		if window >= 0 {
			kStart = max(0, q0-window) &^ (bn - 1)
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
							qh, ql := splitF16(q[row*qDim+h*hd+d])
							kh, klv := splitF16(k[kl*kvDim+kvh*hd+d])
							dot += qh*kh + qh*klv + ql*kh
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
						ph, pl := splitF16(float32(sc[j]))
						kl := k0 + j
						for d := range hd {
							vh, vl := splitF16(v[kl*kvDim+kvh*hd+d])
							acc[d] += ph*vh + ph*vl + pl*vh
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

// TestEG2PackedAttn_VsCombinedBench compares the standalone eg2_packed_attn
// kernels directly against the existing combined transposed-key attention
// (eg2_transpose_keys + eg2_attention_transposed) across the 24-layer EG2 mix
// (20 sliding HD=256 nH=4 nKV=2 W=512 + 4 full HD=512 nH=4 nKV=1 W=-1).
// Opt-in via GOINFER_EG2_ATTN_BENCH=1 so normal test runs stay fast.
func TestEG2PackedAttn_VsCombinedBench(t *testing.T) {
	if os.Getenv("GOINFER_EG2_ATTN_BENCH") != "1" {
		t.Skip("set GOINFER_EG2_ATTN_BENCH=1 to run the packed vs combined attention microbenchmark")
	}
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

	packedMod, err := dev.CompileLibrary(readPTX("eg2_packed_attn"))
	if err != nil {
		t.Fatalf("CompileLibrary(eg2_packed_attn): %v", err)
	}
	combMod, err := dev.CompileLibrary(readPTX("eg2_transposed_experiment"))
	if err != nil {
		t.Fatalf("CompileLibrary(eg2_transposed_experiment): %v", err)
	}
	pHD256, err := dev.NewComputePipeline(packedMod, "eg2_packed_attn_hd256")
	if err != nil {
		t.Fatalf("NewComputePipeline(eg2_packed_attn_hd256): %v", err)
	}
	pHD512, err := dev.NewComputePipeline(packedMod, "eg2_packed_attn_hd512")
	if err != nil {
		t.Fatalf("NewComputePipeline(eg2_packed_attn_hd512): %v", err)
	}
	cTrans, err := dev.NewComputePipeline(combMod, "eg2_transpose_keys")
	if err != nil {
		t.Fatalf("NewComputePipeline(eg2_transpose_keys): %v", err)
	}
	cAttn, err := dev.NewComputePipeline(combMod, "eg2_attention_transposed")
	if err != nil {
		t.Fatalf("NewComputePipeline(eg2_attention_transposed): %v", err)
	}
	stream := dev.NewCommandQueue()

	type shape struct {
		name string
		lens []int
	}
	shapes := []shape{
		{"17-input-7924tok", []int{31, 819, 549, 392, 678, 362, 661, 374, 743, 36, 773, 522, 398, 581, 530, 441, 34}},
		{"8x512", []int{512, 512, 512, 512, 512, 512, 512, 512}},
		{"4x2048", []int{2048, 2048, 2048, 2048}},
	}

	buildSegBounds := func(lens []int, window int) (lo, hi []int32, total, maxT int) {
		for _, n := range lens {
			total += n
			if n > maxT {
				maxT = n
			}
		}
		lo = make([]int32, total)
		hi = make([]int32, total)
		off := 0
		for _, n := range lens {
			for i := range n {
				s0, s1 := 0, n
				if window >= 0 {
					s0 = max(0, i-window)
					s1 = min(n, i+window+1)
				}
				lo[off+i] = int32(off + s0)
				hi[off+i] = int32(off + s1)
			}
			off += n
		}
		return lo, hi, total, maxT
	}

	const (
		nH      = 4
		nKVSl   = 2
		hdSl    = 256
		winSl   = 512
		nLayers = 20
		nKVFu   = 1
		hdFu    = 512
		winFu   = -1
		nFu     = 4
		warmup  = 3
		iters   = 10
	)
	specSl, _ := launchgeom.EG2PackedAttnSpecForHD(hdSl)
	specFu, _ := launchgeom.EG2PackedAttnSpecForHD(hdFu)

	for _, sh := range shapes {
		t.Run(sh.name, func(t *testing.T) {
			slLo, slHi, total, maxT := buildSegBounds(sh.lens, winSl)
			fuLo, fuHi, _, _ := buildSegBounds(sh.lens, winFu)
			pitch := (total + 31) &^ 31

			tabSl, err := launchgeom.BuildEG2PackedTiles(sh.lens, specSl.BM)
			if err != nil {
				t.Fatal(err)
			}
			tabFu, err := launchgeom.BuildEG2PackedTiles(sh.lens, specFu.BM)
			if err != nil {
				t.Fatal(err)
			}

			// Allocate deterministic synthetic Q/K/V for both sliding and full geometries.
			fill := func(n, seed int, amp float32) []float32 {
				xs := make([]float32, n)
				for i := range xs {
					xs[i] = float32(math.Sin(float64(seed*1009+i)*0.137)) * amp
				}
				return xs
			}
			qSlHost := fill(total*nH*hdSl, 1, 0.85)
			kSlHost := fill(total*nKVSl*hdSl, 2, 0.90)
			vSlHost := fill(total*nKVSl*hdSl, 3, 0.75)
			qFuHost := fill(total*nH*hdFu, 4, 0.85)
			kFuHost := fill(total*nKVFu*hdFu, 5, 0.90)
			vFuHost := fill(total*nKVFu*hdFu, 6, 0.75)

			allocUploadF32 := func(src []float32) Buffer {
				b := gpu.NewBufferLenOf[float32](dev, len(src))
				if err := gpu.Upload(b, src); err != nil {
					t.Fatal(err)
				}
				return b
			}
			allocUploadI32 := func(src []int32) Buffer {
				b := gpu.NewBufferLenOf[int32](dev, len(src))
				if err := gpu.Upload(b, src); err != nil {
					t.Fatal(err)
				}
				return b
			}

			qSlB := allocUploadF32(qSlHost)
			kSlB := allocUploadF32(kSlHost)
			vSlB := allocUploadF32(vSlHost)
			ktSlB := gpu.NewBufferLenOf[float32](dev, nKVSl*hdSl*pitch)
			oSlPacked := gpu.NewBufferLenOf[float32](dev, total*nH*hdSl)
			oSlComb := gpu.NewBufferLenOf[float32](dev, total*nH*hdSl)

			qFuB := allocUploadF32(qFuHost)
			kFuB := allocUploadF32(kFuHost)
			vFuB := allocUploadF32(vFuHost)
			ktFuB := gpu.NewBufferLenOf[float32](dev, nKVFu*hdFu*pitch)
			oFuPacked := gpu.NewBufferLenOf[float32](dev, total*nH*hdFu)
			oFuComb := gpu.NewBufferLenOf[float32](dev, total*nH*hdFu)

			slLoB := allocUploadI32(slLo)
			slHiB := allocUploadI32(slHi)
			fuLoB := allocUploadI32(fuLo)
			fuHiB := allocUploadI32(fuHi)

			slSS := allocUploadI32(tabSl.SegStart)
			slSL := allocUploadI32(tabSl.SegLen)
			slQ0 := allocUploadI32(tabSl.QLocal0)
			fuSS := allocUploadI32(tabFu.SegStart)
			fuSL := allocUploadI32(tabFu.SegLen)
			fuQ0 := allocUploadI32(tabFu.QLocal0)
			defer func() {
				for _, b := range []Buffer{
					qSlB, kSlB, vSlB, ktSlB, oSlPacked, oSlComb,
					qFuB, kFuB, vFuB, ktFuB, oFuPacked, oFuComb,
					slLoB, slHiB, fuLoB, fuHiB,
					slSS, slSL, slQ0, fuSS, fuSL, fuQ0,
				} {
					dev.ReleaseBuf(b)
				}
			}()

			cfgPackedSl := LaunchConfig{
				GridX: uint32(nH), GridY: uint32(len(tabSl.SegStart)), GridZ: 1,
				BlockX: specSl.BlockX, BlockY: 1, BlockZ: 1, SharedMemBytes: specSl.SharedMem,
			}
			cfgPackedFu := LaunchConfig{
				GridX: uint32(nH), GridY: uint32(len(tabFu.SegStart)), GridZ: 1,
				BlockX: specFu.BlockX, BlockY: 1, BlockZ: 1, SharedMemBytes: specFu.SharedMem,
			}
			cfgTransSl := LaunchConfig{
				GridX: uint32((nKVSl*hdSl + 31) / 32), GridY: uint32((total + 31) / 32), GridZ: 1,
				BlockX: 32, BlockY: 8, BlockZ: 1,
			}
			cfgTransFu := LaunchConfig{
				GridX: uint32((nKVFu*hdFu + 31) / 32), GridY: uint32((total + 31) / 32), GridZ: 1,
				BlockX: 32, BlockY: 8, BlockZ: 1,
			}
			cfgCombSl := gpu.SegAttentionGrid(total, nH, min(maxT, 2*winSl+1))
			cfgCombFu := gpu.SegAttentionGrid(total, nH, maxT)

			launchPackedSl := func() {
				if err := stream.Launch(pHD256, cfgPackedSl,
					Arg(qSlB), Arg(kSlB), Arg(vSlB), Arg(slSS), Arg(slSL), Arg(slQ0),
					gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKVSl)), gpu.ArgValue(int32(winSl)),
					gpu.ArgValue(float32(1)), gpu.ArgValue(int32(len(tabSl.SegStart))), Arg(oSlPacked)); err != nil {
					t.Fatal(err)
				}
			}
			launchPackedFu := func() {
				if err := stream.Launch(pHD512, cfgPackedFu,
					Arg(qFuB), Arg(kFuB), Arg(vFuB), Arg(fuSS), Arg(fuSL), Arg(fuQ0),
					gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKVFu)), gpu.ArgValue(int32(winFu)),
					gpu.ArgValue(float32(1)), gpu.ArgValue(int32(len(tabFu.SegStart))), Arg(oFuPacked)); err != nil {
					t.Fatal(err)
				}
			}
			launchCombSl := func() {
				if err := stream.Launch(cTrans, cfgTransSl,
					Arg(kSlB), Arg(ktSlB), gpu.ArgValue(int32(total)), gpu.ArgValue(int32(nKVSl*hdSl)), gpu.ArgValue(int32(pitch))); err != nil {
					t.Fatal(err)
				}
				if err := stream.Launch(cAttn, cfgCombSl,
					Arg(qSlB), Arg(ktSlB), Arg(vSlB), Arg(oSlComb), Arg(slLoB), Arg(slHiB),
					gpu.ArgValue(int32(total)), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKVSl)),
					gpu.ArgValue(int32(hdSl)), gpu.ArgValue(int32(pitch)), gpu.ArgValue(float32(1))); err != nil {
					t.Fatal(err)
				}
			}
			launchCombFu := func() {
				if err := stream.Launch(cTrans, cfgTransFu,
					Arg(kFuB), Arg(ktFuB), gpu.ArgValue(int32(total)), gpu.ArgValue(int32(nKVFu*hdFu)), gpu.ArgValue(int32(pitch))); err != nil {
					t.Fatal(err)
				}
				if err := stream.Launch(cAttn, cfgCombFu,
					Arg(qFuB), Arg(ktFuB), Arg(vFuB), Arg(oFuComb), Arg(fuLoB), Arg(fuHiB),
					gpu.ArgValue(int32(total)), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKVFu)),
					gpu.ArgValue(int32(hdFu)), gpu.ArgValue(int32(pitch)), gpu.ArgValue(float32(1))); err != nil {
					t.Fatal(err)
				}
			}

			// Parity check between packed and combined on one sliding + one full layer.
			launchPackedSl()
			launchCombSl()
			launchPackedFu()
			launchCombFu()
			if err := stream.Sync(); err != nil {
				t.Fatal(err)
			}
			gotPSl := make([]float32, total*nH*hdSl)
			gotCSl := make([]float32, total*nH*hdSl)
			gotPFu := make([]float32, total*nH*hdFu)
			gotCFu := make([]float32, total*nH*hdFu)
			for b, dst := range map[Buffer][]float32{oSlPacked: gotPSl, oSlComb: gotCSl, oFuPacked: gotPFu, oFuComb: gotCFu} {
				if err := gpu.Download(b, dst); err != nil {
					t.Fatal(err)
				}
			}
			worstCos := func(label string, a, b []float32, hd int) float64 {
				w := 1.0
				for rowIdx, off := 0, 0; off < len(a); rowIdx, off = rowIdx+1, off+hd {
					c := eg2AttnCosineF32(a[off:off+hd], b[off:off+hd])
					if math.IsNaN(c) || math.IsInf(c, 0) || c < 0.999 {
						t.Fatalf("%s %s row=%d parity vs combined failed: cos=%.8f", sh.name, label, rowIdx, c)
					}
					if c < w {
						w = c
					}
				}
				return w
			}
			cosSl := worstCos("sliding", gotPSl, gotCSl, hdSl)
			cosFu := worstCos("full", gotPFu, gotCFu, hdFu)

			measure := func(fn func(), reps int) float64 {
				for range warmup {
					for range reps {
						fn()
					}
				}
				if err := stream.Sync(); err != nil {
					t.Fatal(err)
				}
				t0 := time.Now()
				for range iters {
					for range reps {
						fn()
					}
				}
				if err := stream.Sync(); err != nil {
					t.Fatal(err)
				}
				return float64(time.Since(t0).Microseconds()) / (1000.0 * float64(iters))
			}

			combSlMs := measure(launchCombSl, nLayers)
			combFuMs := measure(launchCombFu, nFu)
			packSlMs := measure(launchPackedSl, nLayers)
			packFuMs := measure(launchPackedFu, nFu)
			combTot := combSlMs + combFuMs
			packTot := packSlMs + packFuMs
			t.Logf("%s (T=%d maxT=%d): cos(sl=%.8f fu=%.8f) | combined 20sl=%.2fms 4fu=%.2fms tot=%.2fms | packed 20sl=%.2fms 4fu=%.2fms tot=%.2fms | speedup sl=%.2fx fu=%.2fx tot=%.2fx",
				sh.name, total, maxT, cosSl, cosFu,
				combSlMs, combFuMs, combTot,
				packSlMs, packFuMs, packTot,
				combSlMs/packSlMs, combFuMs/packFuMs, combTot/packTot)
		})
	}
}
