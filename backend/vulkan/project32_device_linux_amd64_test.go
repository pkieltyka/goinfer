//go:build linux && amd64

package vulkan

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
)

// TestProject32QualifiedTiny proves the product default and exact bytes on the
// attached card: the 11-row query stays stock and creates no optional
// pipeline; 33+ rows take the wide route and reproduce the stock bytes.
func TestProject32QualifiedTiny(t *testing.T) {
	id := requiredDevice(t)
	e := loadTinyVulkanEncoder(t)
	a := attachParityDevice(t, e, id)
	info := a.device.Info()
	if got, want := a.project32, project32Device(info.VendorID, info.DeviceID); got != want {
		t.Fatalf("qualified=%v, want %v for %04x:%04x", got, want, info.VendorID, info.DeviceID)
	}
	if !a.project32 {
		t.Skip("attached product is not the measured RX 9060 XT")
	}

	counts := map[string]int{}
	old := dispatchSeenForTest
	defer func() { dispatchSeenForTest = old }()
	dispatchSeenForTest = func(name string) { counts[name]++ }
	t.Cleanup(func() { dispatchSeenForTest = nil })

	// Short input: stock only, and the optional pipeline must stay unmade.
	short := make([]int, 11)
	for i := range short {
		short[i] = 4 + i%8
	}
	if _, err := e.EmbedIDs(short); err != nil {
		t.Fatal(err)
	}
	if counts["project32"] != 0 || a.pipes["project32"] != nil {
		t.Fatalf("short input used the wide route or created its pipeline: %v", counts)
	}
	counts = map[string]int{}

	ids := make([]int, 40)
	for i := range ids {
		ids[i] = 4 + i%8
	}
	batch := [][]int{ids[:33], ids[33:40], ids[:5], ids[10:30]}

	stock := func() [][]float32 {
		a.project32 = false
		defer func() { a.project32 = true }()
		out, err := e.EmbedIDsBatch(batch)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}()
	counts = map[string]int{}
	wide, err := e.EmbedIDsBatch(batch)
	if err != nil {
		t.Fatal(err)
	}
	if counts["project32"] == 0 {
		t.Fatalf("wide batch did not use the 32-row route: %v", counts)
	}
	compareExact(t, stock, wide)
	if err := a.status(); err != nil {
		t.Fatal(err)
	}
}

// TestProject32DeclineAndRetry uses the fake device to prove the optional
// pipeline lifecycle: a missing feature retires the route and the same
// projection runs stock; allocation and loss failures stay errors and are not
// cached as declines, so a later forward retries creation.
func TestProject32DeclineAndRetry(t *testing.T) {
	ids := make([]int, 40)
	for i := range ids {
		ids[i] = 4 + i%8
	}
	batch := [][]int{ids[:33], ids[33 : 7+33]}

	t.Run("capability decline is cached and runs stock", func(t *testing.T) {
		var fail int32
		e, a := attachFakeVulkan(t, &vk.FakeHooksForTest{
			CreatePipelines: func() int32 { return fail },
		})
		defer func() { _ = a.Close() }()
		a.project32 = true
		fail = -8
		counts := map[string]int{}
		old := dispatchSeenForTest
		dispatchSeenForTest = func(name string) { counts[name]++ }
		defer func() { dispatchSeenForTest = old }()
		for attempt := 0; attempt < 2; attempt++ {
			if _, err := e.EmbedIDsBatch(batch); err != nil {
				t.Fatalf("attempt %d: %v", attempt, err)
			}
			if counts["project32"] != 0 || a.pipes["project32"] != nil || !a.project32Declined {
				t.Fatalf("attempt %d: wide=%d pipe=%v declined=%v", attempt, counts["project32"], a.pipes["project32"], a.project32Declined)
			}
			counts = map[string]int{}
		}
	})

	for _, tc := range []struct {
		name string
		code int32
		want error
	}{
		{"oom", -1, vk.ErrOutOfMemory},
		{"device lost", -4, vk.ErrDeviceLost},
	} {
		t.Run(tc.name+" is not a decline and can be retried", func(t *testing.T) {
			var fail int32
			e, a := attachFakeVulkan(t, &vk.FakeHooksForTest{
				CreatePipelines: func() int32 { return fail },
			})
			defer func() { _ = a.Close() }()
			a.project32 = true
			fail = tc.code
			_, err := e.EmbedIDsBatch(batch)
			if err == nil || !errors.Is(err, tc.want) {
				t.Fatalf("first error %v, want %v", err, tc.want)
			}
			if a.project32Declined {
				t.Fatal("a native failure was cached as a capability decline")
			}
			fail = 0
			if _, err := e.EmbedIDsBatch(batch); err != nil {
				t.Fatalf("retry after %s: %v", tc.name, err)
			}
			if a.pipes["project32"] == nil {
				t.Fatalf("retry after %s did not create the pipeline", tc.name)
			}
		})
	}
}

// TestProject32UnqualifiedDevice pins that a non-measured product keeps the
// stock route and never creates the optional pipeline.
func TestProject32UnqualifiedDevice(t *testing.T) {
	ids := make([]int, 40)
	for i := range ids {
		ids[i] = 4 + i%8
	}
	e, a := attachFakeVulkan(t, &vk.FakeHooksForTest{})
	defer func() { _ = a.Close() }()
	if a.project32 {
		t.Fatalf("fake product %04x:%04x qualified", a.device.Info().VendorID, a.device.Info().DeviceID)
	}
	counts := map[string]int{}
	old := dispatchSeenForTest
	dispatchSeenForTest = func(name string) { counts[name]++ }
	defer func() { dispatchSeenForTest = old }()
	if _, err := e.EmbedIDsBatch([][]int{ids}); err != nil {
		t.Fatal(err)
	}
	if counts["project32"] != 0 || a.pipes["project32"] != nil {
		t.Fatalf("unqualified device used the wide route: %v", counts)
	}
}

// TestProject32RawGuardRows compares both shaders directly on signed values,
// odd inner/column sizes and a nonzero firstRow, and proves neither writes
// outside [firstRow, firstRow+rows).
func TestProject32RawGuardRows(t *testing.T) {
	id := requiredDevice(t)
	e := loadTinyVulkanEncoder(t)
	a := attachParityDevice(t, e, id)
	device := a.device
	const rows, firstRow, cols, inner, trailing = 33, 5, 19, 37, 3
	totalRows := firstRow + rows + trailing
	rng := rand.New(rand.NewPCG(81, 32))
	samples := func(n int) []float32 {
		data := make([]float32, n)
		for i := range data {
			data[i] = rng.Float32()*8 - 4
		}
		return data
	}
	src, weights := samples(totalRows*inner), samples(cols*inner)
	sentinel := float32(-777.25)
	out := make([]float32, totalRows*cols)
	for i := range out {
		out[i] = sentinel
	}
	buffer := func(memory vk.Memory, size int) *vk.Buffer {
		b, err := device.NewBuffer(uint64(size)*4, memory)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = b.Close() })
		return b
	}
	upload := func(data []float32) *vk.Buffer {
		b := buffer(vk.HostVisible, len(data))
		if err := b.WriteF32(0, data); err != nil {
			t.Fatal(err)
		}
		return b
	}
	pipeline := func(name, want string) *vk.Pipeline {
		spirv, err := shaders.ReadFile("shaders/" + name + ".spv")
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(spirv)); got != want {
			t.Fatalf("%s SPIR-V %s, want %s", name, got, want)
		}
		p, err := device.NewPipeline(spirv, 3, 4*4)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = p.Close() })
		return p
	}
	srcBuf, wBuf := upload(src), upload(weights)
	push := []uint32{firstRow, rows, cols, inner}
	run := func(p *vk.Pipeline, gridY uint32) []float32 {
		dst := buffer(vk.DeviceLocal, len(out))
		seed := upload(out)
		read := buffer(vk.HostVisible, len(out))
		err := device.Run(func(c *vk.Commands) error {
			if err := c.Copy(seed, dst, 0, 0, uint64(len(out))*4); err != nil {
				return err
			}
			if err := c.Dispatch(p, []*vk.Buffer{srcBuf, wBuf, dst}, push,
				[3]uint32{uint32(ceilDiv(cols, 16)), gridY, 1}); err != nil {
				return err
			}
			return c.Copy(dst, read, 0, 0, uint64(len(out))*4)
		})
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(out))
		if err := read.ReadF32(0, got); err != nil {
			t.Fatal(err)
		}
		return got
	}
	stock := run(pipeline("project", "aee8d9454fa3af610726ecdd54b0d4074a1809bc8eb47867ca956c2ec80b92dc"), uint32(ceilDiv(rows, 8)))
	wide := run(pipeline("project32", "6d3972611cf681813d0d008780fd5dfa53cead07d54f6fa8ccaab4d7b356002e"), uint32(ceilDiv(rows, 32)))
	for i := range out {
		if !bytes.Equal(vectorBytes(stock[i:i+1]), vectorBytes(wide[i:i+1])) {
			t.Fatalf("element %d differs: stock %v wide %v", i, stock[i], wide[i])
		}
		row := i / cols
		if row < firstRow || row >= firstRow+rows {
			if stock[i] != sentinel || wide[i] != sentinel {
				t.Fatalf("guard row %d element %d was written: stock %v wide %v", row, i%cols, stock[i], wide[i])
			}
		}
	}
}

func compareExact(t *testing.T, stock, wide [][]float32) {
	t.Helper()
	if len(stock) != len(wide) {
		t.Fatalf("%d stock rows, %d wide rows", len(stock), len(wide))
	}
	for i := range stock {
		if !bytes.Equal(vectorBytes(stock[i]), vectorBytes(wide[i])) {
			t.Fatalf("row %d differs between stock and the 32-row kernel", i)
		}
	}
}

func vectorBytes(v []float32) []byte {
	out := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(f))
	}
	return out
}
