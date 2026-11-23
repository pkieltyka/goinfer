//go:build cuda

package kernels

import (
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
)

func TestEG2FusionsArithmetic(t *testing.T) {
	id := os.Getenv("GOINFER_TEST_CUDA_UUID")
	if id == "" {
		t.Skip("set GOINFER_TEST_CUDA_UUID to run the fusion GPU check")
	}
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.ReleaseObjectsChecked(); err != nil {
			t.Error(err)
		}
	})
	if got := dev.Adapter().UUID; !strings.EqualFold(got, id) {
		t.Fatalf("visible default device is %q, want %q", got, id)
	}
	queue := dev.NewCommandQueue()
	t.Cleanup(func() {
		if err := queue.Sync(); err != nil {
			t.Error(err)
		}
	})
	pipe := func(file, name string) gpu.Pipeline {
		t.Helper()
		ptx, err := os.ReadFile("../testdata/" + file)
		if err != nil {
			t.Fatal(err)
		}
		lib, err := dev.CompileLibrary(ptx)
		if err != nil {
			t.Fatal(err)
		}
		p, err := dev.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	vit, err := dev.NewViT()
	if err != nil {
		t.Fatal(err)
	}
	mul := pipe("tower_base.ptx", "tower_mul")
	fused := pipe("eg2_fusions.ptx", "eg2_gelu_mul")
	fusedPLE := pipe("eg2_fusions.ptx", "eg2_gelu_mul_ple")
	const rows, layers, channels = 33, 4, 17
	const n = rows * channels
	grid := gpu.Grid1D(n, 256)
	rng := rand.New(rand.NewPCG(63, 24))
	gate := make([]float32, n)
	up := make([]float32, n)
	for i := range gate {
		gate[i] = rng.Float32()*20 - 10
		up[i] = rng.Float32()*2 - 1
	}
	witnessGate, witnessUp := float32(1.0/17), float32(3.0/13)
	v := float64(witnessGate)
	g64 := 0.5 * v * (1 + math.Tanh(0.7978845608028654*math.FMA(0.044715*v*v, v, v)))
	rounded := float32(g64) * witnessUp
	if rounded == float32(g64*float64(witnessUp)) {
		t.Fatal("chosen witness does not distinguish one rounding from two")
	}
	gate[0], up[0] = witnessGate, witnessUp
	copy(gate[1:], []float32{0, math.Float32frombits(1 << 31), 0x1p-148,
		-0x1p-148, 0x1p-120, -0x1p-120, -100, 100, 3.8, -3.8,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())})
	for i := 1; i <= 13; i++ {
		up[i] = (-1 + 2*float32(i%2)) * 0.75
	}

	run := func(p gpu.Pipeline, args ...gpu.KernelArg) {
		t.Helper()
		if err := queue.Launch(p, grid, args...); err != nil {
			t.Fatal(err)
		}
	}
	read := func(buffer gpu.Buffer) []float32 {
		t.Helper()
		if err := queue.Sync(); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, n)
		if err := gpu.Download(buffer, out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	compare := func(got, want []float32) {
		t.Helper()
		for i := range got {
			if math.IsNaN(float64(want[i])) && math.IsNaN(float64(got[i])) {
				continue
			}
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("index %d = %08x, want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	refGate, candidateGate, upBuffer := gpu.NewBufferOf(dev, gate), gpu.NewBufferOf(dev, gate), gpu.NewBufferOf(dev, up)
	run(vit.GELUTanh, gpu.Arg(refGate), gpu.ArgValue(int32(n)))
	run(mul, gpu.Arg(refGate), gpu.Arg(upBuffer), gpu.ArgValue(int64(n)))
	run(fused, gpu.Arg(candidateGate), gpu.Arg(upBuffer), gpu.ArgValue(int32(n)))
	ref := read(refGate)
	compare(read(candidateGate), ref)
	if ref[0] != rounded {
		t.Fatalf("rounding witness = %08x, want %08x", math.Float32bits(ref[0]), math.Float32bits(rounded))
	}

	table := make([]float32, rows*layers*channels)
	for i := range table {
		table[i] = float32(i-len(table)/2) / 1024
	}
	layerTable := gpu.NewBufferOf(dev, table)
	gathered := gpu.NewBufferLenOf[float32](dev, n)
	for _, layer := range []int{0, 2, layers - 1} {
		col := layerTable.At(layer * channels * 4)
		if err := gpu.Upload(refGate, gate); err != nil {
			t.Fatal(err)
		}
		if err := gpu.Upload(candidateGate, gate); err != nil {
			t.Fatal(err)
		}
		run(vit.GELUTanh, gpu.Arg(refGate), gpu.ArgValue(int32(n)))
		if err := queue.Copy2DAsync(gathered, channels*4, col, layers*channels*4, channels*4, rows); err != nil {
			t.Fatal(err)
		}
		run(mul, gpu.Arg(refGate), gpu.Arg(gathered), gpu.ArgValue(int64(n)))
		run(fusedPLE, gpu.Arg(candidateGate), gpu.Arg(col), gpu.ArgValue(int32(channels)),
			gpu.ArgValue(int64(layers*channels)), gpu.ArgValue(int32(n)))
		compare(read(candidateGate), read(refGate))
	}
}
