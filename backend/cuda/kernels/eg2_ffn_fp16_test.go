//go:build cuda

package kernels

import (
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/pkieltyka/aikit/linalg"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/wpack"
)

func fp16FFNDevice(t *testing.T) (*gpu.Device, []byte) {
	t.Helper()
	id := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if id == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID to run FP16 FFN device tests")
	}
	adapters, err := gpu.Adapters()
	if err != nil {
		t.Fatal(err)
	}
	i := slices.IndexFunc(adapters, func(row gpu.Adapter) bool { return strings.EqualFold(row.UUID, id) })
	if i < 0 {
		t.Fatalf("CUDA device %q is absent", id)
	}
	if a := adapters[i]; a.Major < 8 {
		t.Fatalf("FP16 FFN requires SM80 or newer, got %d.%d", a.Major, a.Minor)
	}
	ptx, err := os.ReadFile("../testdata/eg2_ffn_fp16.ptx")
	if err != nil {
		t.Fatal(err)
	}
	dev, err := gpu.CreateSelectedDevice(adapters[i])
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.ReleaseObjectsChecked(); err != nil {
			t.Error(err)
		}
	})
	return dev, ptx
}

func TestEG2FP16Projection(t *testing.T) {
	dev, ptx := fp16FFNDevice(t)
	module, err := dev.CompileLibrary(ptx)
	if err != nil {
		t.Fatal(err)
	}
	pipe, err := dev.NewComputePipeline(module, "eg2_ffn_fp16")
	if err != nil {
		t.Fatal(err)
	}
	queue := dev.NewCommandQueue()
	t.Cleanup(func() {
		if err := queue.Sync(); err != nil {
			t.Error(err)
		}
	})
	rng := rand.New(rand.NewPCG(63, 16))
	for _, shape := range [][2]int{{67, 16}, {63, 80}, {96, 512}, {512, 2048}} {
		n, k := shape[0], shape[1]
		w := make([]uint16, n*k)
		for i := range w {
			w[i] = wpack.F32ToF16RNE(rng.Float32()*2 - 1)
		}
		x := make([]float32, 65*k)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		wb := gpu.NewBufferOf(dev, w)
		run := func(x []float32, wantInvalid bool) []float32 {
			t.Helper()
			m := len(x) / k
			out := make([]float32, m*n)
			var status [1]uint32
			xb := gpu.NewBufferOf(dev, x)
			yb := gpu.NewBufferLenOf[float32](dev, len(out))
			invalid := gpu.NewBufferLenOf[uint32](dev, 1)
			if err := queue.ZeroAsync(invalid, 4); err != nil {
				t.Fatal(err)
			}
			grid := gpu.LaunchConfig{
				GridX: uint32((n + 63) / 64), GridY: uint32((m + 31) / 32), GridZ: 1,
				BlockX: 128, BlockY: 1, BlockZ: 1,
			}
			if err := queue.Launch(pipe, grid, gpu.Arg(xb), gpu.Arg(wb), gpu.Arg(yb), gpu.Arg(invalid),
				gpu.ArgValue(int32(m)), gpu.ArgValue(int32(n)), gpu.ArgValue(int32(k))); err != nil {
				t.Fatal(err)
			}
			if err := queue.Sync(); err != nil {
				t.Fatal(err)
			}
			if err := gpu.Download(invalid, status[:]); err != nil {
				t.Fatal(err)
			}
			if err := gpu.Download(yb, out); err != nil {
				t.Fatal(err)
			}
			for _, b := range []gpu.Buffer{invalid, yb, xb} {
				if err := dev.ReleaseBufChecked(b); err != nil {
					t.Fatal(err)
				}
			}
			if (status[0] != 0) != wantInvalid {
				t.Fatalf("range status %#x, want invalid=%t", status[0], wantInvalid)
			}
			return out
		}
		got := run(x, false)
		for r := 0; r < 65; r++ {
			for c := 0; c < n; c++ {
				var sum float64
				for d := 0; d < k; d++ {
					rx := linalg.F16ToF32(wpack.F32ToF16RNE(x[r*k+d]))
					sum += float64(rx) * float64(linalg.F16ToF32(w[c*k+d]))
				}
				if delta := math.Abs(float64(got[r*n+c]) - sum); !(delta <= 3e-4) {
					t.Fatalf("N=%d K=%d row %d col %d got %g want %g (abs error %g)", n, k, r, c, got[r*n+c], sum, delta)
				}
			}
		}
		for _, r := range []int{0, 1, 14, 15, 16, 30, 31, 32, 63, 64} {
			single := run(x[r*k:(r+1)*k], false)
			for c := 0; c < n; c++ {
				if math.Float32bits(single[c]) != math.Float32bits(got[r*n+c]) {
					t.Fatalf("N=%d K=%d row %d col %d changed in singleton", n, k, r, c)
				}
			}
		}
		if n == 67 {
			boundary := []struct {
				value   float32
				invalid bool
			}{{65504, false}, {-65519, false}, {0x1p-24, false}, {65520, true},
				{-65520, true}, {float32(math.NaN()), true}, {float32(math.Inf(1)), true}}
			late := slices.Clone(x[:33*k])
			for i, tc := range boundary {
				// Alternate the low and high half of a pair in the final row.
				late[(len(late)-2)+(i%2)] = tc.value
				run(late, tc.invalid)
				late[(len(late)-2)+(i%2)] = 0
			}
		}
		if err := dev.ReleaseBufChecked(wb); err != nil {
			t.Fatal(err)
		}
	}
}

func TestEG2FP16NativeResources(t *testing.T) {
	dev, ptx := fp16FFNDevice(t)
	module, err := dev.Context().LoadModule(ptx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	})
	f, err := module.Function("eg2_ffn_fp16")
	if err != nil {
		t.Fatal(err)
	}
	local, err := f.Attribute(gc.FuncAttrLocalSizeBytes)
	if err != nil || local != 0 {
		t.Fatalf("local memory %d B/thread, error %v", local, err)
	}
	shared, err := f.Attribute(gc.FuncAttrSharedSizeBytes)
	if err != nil {
		t.Fatal(err)
	}
	registers, err := f.Attribute(gc.FuncAttrNumRegs)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.Context().Device().ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SM%d.%d local=%d B/thread static shared=%d B/CTA registers=%d/thread", major, minor, local, shared, registers)
}
