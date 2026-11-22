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

	// A partial query block straddles a document boundary at a non-tile-aligned offset.
	const tokens, heads, kvHeads, headDim = 133, 4, 2, 21
	const queryStart, blockRows, radius, split = 65, 11, 4, 70
	rng := rand.New(rand.NewPCG(27, 339))
	samples := func(n int) []float32 {
		data := make([]float32, n)
		for i := range data {
			data[i] = rng.Float32()*4 - 2
		}
		return data
	}
	queries, keys, values := samples(tokens*heads*headDim), samples(tokens*kvHeads*headDim), samples(tokens*kvHeads*headDim)
	segments := make([]float32, 2*tokens)
	for row := range tokens {
		if row < split {
			segments[2*row+1] = split
		} else {
			segments[2*row], segments[2*row+1] = split, tokens
		}
	}

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

	qBuf, kBuf, vBuf, segmentBuf := input(queries), input(keys), input(values), input(segments)
	probBuf := buffer(heads*blockRows*tokens, vk.DeviceLocal)
	ctxBuf := buffer(tokens*heads*headDim, vk.DeviceLocal)
	probRead := buffer(heads*blockRows*tokens, vk.HostVisible)
	ctxRead := buffer(blockRows*heads*headDim, vk.HostVisible)
	score, softmax, value := pipeline("scores", 3, 24), pipeline("softmax", 2, 28), pipeline("values", 3, 24)
	push := []uint32{tokens, queryStart, blockRows, heads, kvHeads, headDim}
	err = device.Run(func(c *vk.Commands) error {
		if err := c.Dispatch(score, []*vk.Buffer{qBuf, kBuf, probBuf}, push, [3]uint32{9, 2, heads}); err != nil {
			return err
		}
		if err := c.Dispatch(softmax, []*vk.Buffer{probBuf, segmentBuf}, append(push, radius), [3]uint32{blockRows, heads, 1}); err != nil {
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
	gotProb := make([]float32, heads*blockRows*tokens)
	gotCtx := make([]float32, blockRows*heads*headDim)
	if err := probRead.ReadF32(0, gotProb); err != nil {
		t.Fatal(err)
	}
	if err := ctxRead.ReadF32(0, gotCtx); err != nil {
		t.Fatal(err)
	}

	for row := range blockRows {
		q := queryStart + row
		first := max(q-radius, int(segments[2*q]))
		end := min(q+radius+1, int(segments[2*q+1]))
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
			for k := range tokens {
				got := gotProb[(h*blockRows+row)*tokens+k]
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
				if !(math.Abs(float64(got)-want) <= 6e-5) {
					t.Fatalf("row %d head %d coordinate %d: got %g, want %g", q, h, d, got, want)
				}
			}
		}
	}
}
