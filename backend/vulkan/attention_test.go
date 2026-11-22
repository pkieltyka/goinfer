//go:build linux && amd64

package vulkan

import (
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"testing"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
)

func TestAttentionGroupedSegments(t *testing.T) {
	if os.Getenv("GOINFER_TEST_VULKAN_DEVICE") == "" {
		t.Skip("set GOINFER_TEST_VULKAN_DEVICE to an AMD Vulkan device ID in hex")
	}
	id, err := strconv.ParseUint(os.Getenv("GOINFER_TEST_VULKAN_DEVICE"), 16, 32)
	if err != nil || id == 0 {
		t.Fatal("GOINFER_TEST_VULKAN_DEVICE must name a nonzero AMD Vulkan device ID in hex")
	}
	device, err := vk.Open(vk.Selector{DeviceID: uint32(id)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Error(err)
		}
	})
	if info := device.Info(); info.VendorID != 0x1002 || info.DeviceID != uint32(id) {
		t.Fatalf("opened the wrong device: %+v", info)
	}

	// A partial query block clips its key tile inside a packed sequence.
	const tokens, heads, kvHeads, headDim = 133, 4, 2, 21
	const queryStart, blockRows, radius, sequenceStart = 65, 11, 4, 43
	const keyStart, keyCount = queryStart - radius, blockRows + 2*radius
	rng := rand.New(rand.NewPCG(27, 339))
	samples := func(n int) []float32 {
		data := make([]float32, n)
		for i := range data {
			data[i] = rng.Float32()*4 - 2
		}
		return data
	}
	queries, keys, values := samples(tokens*heads*headDim), samples(tokens*kvHeads*headDim), samples(tokens*kvHeads*headDim)
	// Keys outside every query window must not contribute even with non-finite values.
	for k := range tokens {
		if k < queryStart-radius || k >= queryStart+blockRows+radius {
			for d := range kvHeads * headDim {
				values[k*kvHeads*headDim+d] = float32(math.NaN())
			}
		}
	}
	// Only the first query admits this key. A masked in-tile NaN must not
	// reach the value accumulation in subsequent queries.
	values[keyStart*kvHeads*headDim] = float32(math.NaN())
	buffer := func(n int, memory vk.Memory) *vk.Buffer {
		b, err := device.NewBuffer(uint64(n)*4, memory)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := b.Close(); err != nil {
				t.Error(err)
			}
		})
		return b
	}
	input := func(data []float32) *vk.Buffer {
		b := buffer(len(data), vk.HostVisible)
		if err := b.WriteF32(0, data); err != nil {
			t.Fatal(err)
		}
		return b
	}
	pipeline := func(name string, bindings int, pushBytes uint32) *vk.Pipeline {
		spirv, err := os.ReadFile("shaders/" + name + ".spv")
		if err != nil {
			t.Fatal(err)
		}
		p, err := device.NewPipeline(spirv, bindings, pushBytes)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := p.Close(); err != nil {
				t.Error(err)
			}
		})
		return p
	}

	qBuf, kBuf, vBuf := input(queries), input(keys), input(values)
	probBuf := buffer(heads*blockRows*keyCount, vk.DeviceLocal)
	ctxBuf := buffer(tokens*heads*headDim, vk.DeviceLocal)
	probRead := buffer(heads*blockRows*keyCount, vk.HostVisible)
	ctxRead := buffer(blockRows*heads*headDim, vk.HostVisible)
	score, softmax, value := pipeline("scores", 3, 36), pipeline("softmax", 1, 36), pipeline("values", 3, 36)
	push := []uint32{queryStart, blockRows, heads, kvHeads, headDim, keyStart, keyCount, sequenceStart, radius}
	err = device.Run(func(c *vk.Commands) error {
		if err := c.Dispatch(score, []*vk.Buffer{qBuf, kBuf, probBuf}, push, [3]uint32{2, 2, heads}); err != nil {
			return err
		}
		if err := c.Dispatch(softmax, []*vk.Buffer{probBuf}, push, [3]uint32{blockRows, heads, 1}); err != nil {
			return err
		}
		if err := c.Dispatch(value, []*vk.Buffer{probBuf, vBuf, ctxBuf}, push, [3]uint32{2, 2, heads}); err != nil {
			return err
		}
		if err := c.Copy(probBuf, probRead, 0, 0, probRead.Size()); err != nil {
			return err
		}
		return c.Copy(ctxBuf, ctxRead, queryStart*heads*headDim*4, 0, ctxRead.Size())
	})
	if err != nil {
		t.Fatal(err)
	}
	gotProb := make([]float32, heads*blockRows*keyCount)
	gotCtx := make([]float32, blockRows*heads*headDim)
	if err := probRead.ReadF32(0, gotProb); err != nil {
		t.Fatal(err)
	}
	if err := ctxRead.ReadF32(0, gotCtx); err != nil {
		t.Fatal(err)
	}

	for row := range blockRows {
		q := queryStart + row
		first, end := q-radius, q+radius+1
		for h := range heads {
			kh := h / (heads / kvHeads)
			qBase := (q*heads + h) * headDim
			wants := make([]float64, tokens)
			peak := math.Inf(-1)
			for k := first; k < end; k++ {
				kBase := (k*kvHeads + kh) * headDim
				for d := range headDim {
					wants[k] += float64(queries[qBase+d]) * float64(keys[kBase+d])
				}
				peak = max(peak, wants[k])
			}
			var sum float64
			for k := first; k < end; k++ {
				wants[k] = math.Exp(wants[k] - peak)
				sum += wants[k]
			}
			for k := keyStart; k < keyStart+keyCount; k++ {
				got := gotProb[(h*blockRows+row)*keyCount+k-keyStart]
				if k < first || k >= end {
					if got != 0 {
						t.Fatalf("row %d head %d key %d was unmasked: %g", q, h, k, got)
					}
					continue
				}
				wants[k] /= sum
				if !(math.Abs(float64(got)-wants[k]) <= 3e-5) {
					t.Fatalf("row %d head %d key %d probability %g, want %g", q, h, k, got, wants[k])
				}
			}
			for d := range headDim {
				var want float64
				for k := first; k < end; k++ {
					want += wants[k] * float64(values[(k*kvHeads+kh)*headDim+d])
				}
				got := gotCtx[(row*heads+h)*headDim+d]
				if math.IsNaN(want) {
					if !math.IsNaN(float64(got)) {
						t.Fatalf("row %d head %d coordinate %d: got %g, want NaN", q, h, d, got)
					}
					continue
				}
				if !(math.Abs(float64(got)-want) <= 6e-5) {
					t.Fatalf("row %d head %d coordinate %d: got %g, want %g", q, h, d, got, want)
				}
			}
		}
	}
}
