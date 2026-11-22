package vulkan

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func loadTinyVulkanEncoder(t *testing.T) *embeddinggemma2.Encoder {
	t.Helper()
	src, err := filepath.Abs(filepath.Join("..", "..", "testdata", "embeddinggemma2-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	for _, name := range []string{"config.json", "model.safetensors"} {
		if err := os.Symlink(filepath.Join(src, name), filepath.Join(dir, name)); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := json.Marshal(map[string]any{
		"added_tokens": []map[string]any{
			{"id": 0, "content": "<pad>"},
			{"id": 1, "content": "<eos>"},
			{"id": 2, "content": "<bos>"},
			{"id": 3, "content": "<unk>"},
		},
		"model": map[string]any{
			"type":      "BPE",
			"unk_token": "<unk>",
			"vocab":     map[string]int{"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3},
			"merges":    [][2]string{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	e, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func attachFakeVulkan(t *testing.T, hooks *vk.FakeHooksForTest) (*embeddinggemma2.Encoder, *encoder) {
	t.Helper()
	e := loadTinyVulkanEncoder(t)
	name := fmt.Sprintf("vulkan-issue25-fake-%d", time.Now().UnixNano())
	embeddinggemma2.RegisterAccelerator(name, func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return attachWithOpener(m, func() (*vk.Device, error) {
			return vk.NewFakeDeviceForTest(hooks), nil
		})
	})
	acc, err := e.UseAccelerator(name)
	if err != nil {
		t.Fatalf("attach fake vulkan: %v", err)
	}
	return e, acc.(*encoder)
}

// TestVulkanRetryOOM_FakeDriverAcceptance verifies the full #25 Vulkan
// error-classification, cleanup, sticky-loss, attach-OOM, and split-retry
// contract on CPU using injected fake-driver native VkResults.
func TestVulkanRetryOOM_FakeDriverAcceptance(t *testing.T) {
	orig := [][]int{{2, 3, 4}, {5, 6, 7, 8}, {9, 10}, {11, 12, 13}}

	// (1) Pristine multi-input VK_ERROR_OUT_OF_DEVICE_MEMORY (-2) on both
	// DeviceLocal and HostVisible forward allocations before any submission.
	for _, tc := range []struct {
		name       string
		targetMem  vk.Memory
		failOnCall int
	}{
		{"device-local-5th-buffer", vk.DeviceLocal, 5},
		{"host-visible-2nd-buffer", vk.HostVisible, 2},
	} {
		t.Run("pristine-"+tc.name, func(t *testing.T) {
			hooks := &vk.FakeHooksForTest{}
			e, a := attachFakeVulkan(t, hooks)
			defer a.Close()

			heldBefore := a.device.LiveBuffersForTest()
			subBefore := a.device.Activity().Submissions
			seen := 0
			a.device.SetAllocMemoryHookForTest(func(mem vk.Memory, _ uint64) int32 {
				if mem == tc.targetMem {
					seen++
					if seen == tc.failOnCall {
						return -2 // VK_ERROR_OUT_OF_DEVICE_MEMORY
					}
				}
				return 0
			})
			in := make([][]int, len(orig))
			for i := range orig {
				in[i] = slices.Clone(orig[i])
			}
			got, err := e.EmbedIDsBatch(in)
			a.device.SetAllocMemoryHookForTest(nil)
			if got != nil {
				t.Fatalf("pristine OOM returned non-nil outputs")
			}
			if !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) ||
				!errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) ||
				!errors.Is(err, vk.ErrOutOfMemory) {
				t.Fatalf("pristine OOM err = %v", err)
			}
			var re *vk.ResultError
			if !errors.As(err, &re) || re.Code != -2 || re.Op != "allocate buffer memory" {
				t.Fatalf("native cause = %v, want allocate buffer memory (-2)", re)
			}
			for i := range orig {
				if !slices.Equal(in[i], orig[i]) {
					t.Fatalf("input %d mutated", i)
				}
			}
			if a.device.Activity().Submissions != subBefore {
				t.Fatalf("submissions advanced from %d to %d", subBefore, a.device.Activity().Submissions)
			}
			if live := a.device.LiveBuffersForTest(); live != heldBefore {
				t.Fatalf("buffers leaked: live=%d, want %d", live, heldBefore)
			}
			// Smaller split retries succeed on the same encoder handle.
			if left, err := e.EmbedIDsBatch(in[:2]); err != nil || len(left) != 2 {
				t.Fatalf("left split retry = (%d, %v)", len(left), err)
			}
			if right, err := e.EmbedIDsBatch(in[2:]); err != nil || len(right) != 2 {
				t.Fatalf("right split retry = (%d, %v)", len(right), err)
			}
			if live := a.device.LiveBuffersForTest(); live != heldBefore {
				t.Fatalf("post-retry live buffers = %d, want %d", live, heldBefore)
			}
		})
	}

	// (2) Non-retryable Vulkan failures: host OOM (-1), create/bind/map errors,
	// allocation-count cap, late submission/fence errors, and sticky device loss.
	t.Run("non-retry-and-sticky-loss", func(t *testing.T) {
		for _, tc := range []struct {
			name          string
			configure     func(*vk.FakeHooksForTest, *encoder)
			wantCommonOOM bool
			wantLost      bool
			wantCode      int32
		}{
			{
				name: "host-oom-minus-1",
				configure: func(_ *vk.FakeHooksForTest, a *encoder) {
					n := 0
					a.device.SetAllocMemoryHookForTest(func(vk.Memory, uint64) int32 {
						n++
						if n == 3 {
							return -1 // VK_ERROR_OUT_OF_HOST_MEMORY
						}
						return 0
					})
				},
				wantCommonOOM: true, wantCode: -1,
			},
			{
				name: "create-buffer-minus-2",
				configure: func(h *vk.FakeHooksForTest, _ *encoder) {
					n := 0
					h.CreateBuffer = func(uint64) int32 {
						n++
						if n == 3 {
							return -2
						}
						return 0
					}
				},
				wantCommonOOM: true, wantCode: -2,
			},
			{
				name: "bind-memory-minus-2",
				configure: func(h *vk.FakeHooksForTest, _ *encoder) {
					n := 0
					h.BindMemory = func() int32 {
						n++
						if n == 3 {
							return -2
						}
						return 0
					}
				},
				wantCommonOOM: true, wantCode: -2,
			},
			{
				name: "map-memory-failed",
				configure: func(h *vk.FakeHooksForTest, _ *encoder) {
					h.MapMemory = func() int32 { return -5 }
				},
				wantCommonOOM: false, wantCode: -5,
			},
			{
				name: "late-submit-device-oom-minus-2",
				configure: func(h *vk.FakeHooksForTest, _ *encoder) {
					h.QueueSubmit = func() int32 { return -2 }
				},
				wantCommonOOM: true, wantCode: -2,
			},
			{
				name: "late-submit-device-lost-minus-4",
				configure: func(h *vk.FakeHooksForTest, _ *encoder) {
					h.QueueSubmit = func() int32 { return -4 }
				},
				wantCommonOOM: false, wantLost: true, wantCode: -4,
			},
			{
				name: "late-fence-wait-device-lost",
				configure: func(h *vk.FakeHooksForTest, _ *encoder) {
					h.WaitFences = func() int32 { return -4 }
				},
				wantCommonOOM: false, wantLost: true, wantCode: -4,
			},
			{
				name: "alloc-device-lost-minus-4",
				configure: func(_ *vk.FakeHooksForTest, a *encoder) {
					n := 0
					a.device.SetAllocMemoryHookForTest(func(vk.Memory, uint64) int32 {
						n++
						if n == 3 {
							return -4 // VK_ERROR_DEVICE_LOST
						}
						return 0
					})
				},
				wantCommonOOM: false, wantLost: true, wantCode: -4,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				hooks := &vk.FakeHooksForTest{}
				e, a := attachFakeVulkan(t, hooks)
				defer a.Close()
				heldBefore := a.device.LiveBuffersForTest()
				tc.configure(hooks, a)
				got, err := e.EmbedIDsBatch(orig)
				a.device.SetAllocMemoryHookForTest(nil)
				hooks.CreateBuffer, hooks.BindMemory, hooks.MapMemory, hooks.QueueSubmit, hooks.WaitFences = nil, nil, nil, nil, nil
				if got != nil || err == nil {
					t.Fatalf("%s: got (%v, %v), want nil + error", tc.name, got, err)
				}
				if errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("%s must not match ErrBatchOutOfMemory: %v", tc.name, err)
				}
				if errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) != tc.wantCommonOOM {
					t.Fatalf("%s: ErrDeviceOutOfMemory=%v, want %v (err=%v)", tc.name, errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory), tc.wantCommonOOM, err)
				}
				if errors.Is(err, embeddinggemma2.ErrDeviceLost) != tc.wantLost {
					t.Fatalf("%s: ErrDeviceLost=%v, want %v (err=%v)", tc.name, errors.Is(err, embeddinggemma2.ErrDeviceLost), tc.wantLost, err)
				}
				var re *vk.ResultError
				if !errors.As(err, &re) || re.Code != tc.wantCode {
					t.Fatalf("%s: ResultError=%v, want code %d", tc.name, re, tc.wantCode)
				}
				if tc.wantLost {
					var lost *embeddinggemma2.DeviceLostError
					if !errors.As(err, &lost) {
						t.Fatalf("%s: missing *DeviceLostError wrapper: %v", tc.name, err)
					}
					// Sticky loss refuses subsequent valid work and never retries.
					if _, err2 := e.EmbedIDsBatch(orig[:1]); !errors.Is(err2, embeddinggemma2.ErrDeviceLost) || errors.Is(err2, embeddinggemma2.ErrBatchOutOfMemory) {
						t.Fatalf("%s: sticky follow-up = %v, want ErrDeviceLost without retry", tc.name, err2)
					}
				}
				if live := a.device.LiveBuffersForTest(); live != heldBefore {
					t.Fatalf("%s: live buffers = %d, want %d", tc.name, live, heldBefore)
				}
			})
		}

		// Allocation-count exhaustion never matches retry or device OOM.
		eCap := loadTinyVulkanEncoder(t)
		capName := fmt.Sprintf("vulkan-issue25-cap-%d", time.Now().UnixNano())
		var capDev *vk.Device
		embeddinggemma2.RegisterAccelerator(capName, func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
			return attachWithOpener(m, func() (*vk.Device, error) {
				// Tiny model (6 layers) uploads 101 persistent buffers (peak 117 with
				// per-layer staging); cap at 118 so upload succeeds and forward hits maxAllocations.
				capDev = vk.NewFakeDeviceForTest(&vk.FakeHooksForTest{MaxAllocations: 118})
				return capDev, nil
			})
		})
		accCap, err := eCap.UseAccelerator(capName)
		if err != nil {
			t.Fatalf("attach cap encoder: %v", err)
		}
		defer accCap.Close()
		if _, err := eCap.EmbedIDsBatch(orig); err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) {
			t.Fatalf("maxAllocations exhaustion = %v, want non-OOM error", err)
		}

		// Injected cleanup coverage (pre-submit Vulkan destruction uses void native
		// frees): allocation -2 + broken non-lost cleanup latches encoder.broken so
		// no retry is authorized and subsequent forwards refuse before allocating
		// or submitting until Close reclaims the stranded child buffer.
		t.Run("alloc-oom-plus-broken-cleanup", func(t *testing.T) {
			e, a := attachFakeVulkan(t, &vk.FakeHooksForTest{})
			dev := a.device
			defer func() {
				dev.SetCloseBufferHookForTest(nil)
				_ = a.Close()
			}()
			heldBefore := dev.LiveBuffersForTest()
			subBefore := dev.Activity().Submissions
			cleanupErr := errors.New("vulkan: injected cleanup error")
			n := 0
			dev.SetAllocMemoryHookForTest(func(vk.Memory, uint64) int32 {
				n++
				if n == 5 {
					return -2
				}
				return 0
			})
			closed := 0
			dev.SetCloseBufferHookForTest(func() error {
				closed++
				if closed == 1 {
					return cleanupErr
				}
				return nil
			})
			got, err := e.EmbedIDsBatch(orig)
			dev.SetAllocMemoryHookForTest(nil)
			dev.SetCloseBufferHookForTest(nil)
			if got != nil || err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("alloc OOM + broken cleanup = (%v, %v), must not match ErrBatchOutOfMemory", got, err)
			}
			if !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || !errors.Is(err, cleanupErr) {
				t.Fatalf("alloc OOM + broken cleanup missing cause: %v", err)
			}
			stranded := dev.LiveBuffersForTest()
			if stranded != heldBefore+1 {
				t.Fatalf("stranded live buffers = %d, want %d", stranded, heldBefore+1)
			}
			if _, err2 := e.EmbedIDs(orig[0]); err2 == nil || !errors.Is(err2, cleanupErr) || errors.Is(err2, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("singleton after broken cleanup = %v, want latched cleanup error without retry", err2)
			}
			if _, err3 := e.EmbedIDsBatch(orig[:1]); err3 == nil || !errors.Is(err3, cleanupErr) || errors.Is(err3, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("batch after broken cleanup = %v, want latched cleanup error without retry", err3)
			}
			if dev.LiveBuffersForTest() != stranded {
				t.Fatalf("broken encoder allocated new buffers: live=%d, want %d", dev.LiveBuffersForTest(), stranded)
			}
			if dev.Activity().Submissions != subBefore {
				t.Fatalf("broken encoder submitted work: %d -> %d", subBefore, dev.Activity().Submissions)
			}
			if err := a.Close(); err != nil {
				t.Fatalf("Close after broken cleanup: %v", err)
			}
			if live := dev.LiveBuffersForTest(); live != 0 {
				t.Fatalf("Close did not reclaim stranded child buffer: live=%d", live)
			}
		})

		// Allocation -2 + device loss during cleanup -> sticky ErrDeviceLost (-4)
		// wins first in errors.As while preserving prior -2 in the cause chain;
		// no retry on this call or any subsequent call.
		t.Run("alloc-oom-plus-lost-cleanup", func(t *testing.T) {
			e, a := attachFakeVulkan(t, &vk.FakeHooksForTest{})
			defer func() {
				a.device.SetCloseBufferHookForTest(nil)
				_ = a.Close()
			}()
			lostCleanup := &vk.ResultError{Op: "drain submitted work", Code: -4}
			n := 0
			a.device.SetAllocMemoryHookForTest(func(vk.Memory, uint64) int32 {
				n++
				if n == 5 {
					return -2
				}
				return 0
			})
			closed := 0
			a.device.SetCloseBufferHookForTest(func() error {
				closed++
				if closed == 1 {
					return lostCleanup
				}
				return nil
			})
			got, err := e.EmbedIDsBatch(orig)
			a.device.SetAllocMemoryHookForTest(nil)
			a.device.SetCloseBufferHookForTest(nil)
			if got != nil || err == nil || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
				t.Fatalf("alloc OOM + lost cleanup = (%v, %v), must not match ErrBatchOutOfMemory", got, err)
			}
			var lost *embeddinggemma2.DeviceLostError
			if !errors.Is(err, embeddinggemma2.ErrDeviceLost) || !errors.As(err, &lost) || !errors.Is(err, lostCleanup) {
				t.Fatalf("alloc OOM + lost cleanup must surface sticky ErrDeviceLost: %v", err)
			}
			var native *vk.ResultError
			if !errors.As(err, &native) || native.Code != -4 || native.Op != "drain submitted work" {
				t.Fatalf("first native cause = %v, want drain submitted work (-4)", native)
			}
			if !errors.Is(err, vk.ErrBufferDeviceOutOfMemory) || !errors.Is(err, vk.ErrOutOfMemory) {
				t.Fatalf("alloc OOM + lost cleanup dropped prior -2 cause: %v", err)
			}
			for _, call := range []struct {
				label string
				run   func() error
			}{
				{"batch", func() error { _, err := e.EmbedIDsBatch(orig[:1]); return err }},
				{"singleton", func() error { _, err := e.EmbedIDs(orig[0]); return err }},
			} {
				err2 := call.run()
				if !errors.Is(err2, embeddinggemma2.ErrDeviceLost) || errors.Is(err2, embeddinggemma2.ErrBatchOutOfMemory) {
					t.Fatalf("subsequent %s after lost cleanup = %v, want sticky ErrDeviceLost", call.label, err2)
				}
				var native2 *vk.ResultError
				if !errors.As(err2, &native2) || native2.Code != -4 || native2.Op != "drain submitted work" {
					t.Fatalf("subsequent %s first native cause = %v, want drain submitted work (-4)", call.label, native2)
				}
			}
		})
	})

	// (3) Public singleton EmbedIDs OOM + partial-weight-upload attach OOM + unrelated attach error.
	t.Run("singleton-and-attach-oom", func(t *testing.T) {
		e, a := attachFakeVulkan(t, &vk.FakeHooksForTest{})
		defer a.Close()
		heldBefore := a.device.LiveBuffersForTest()
		n := 0
		a.device.SetAllocMemoryHookForTest(func(vk.Memory, uint64) int32 {
			n++
			if n == 4 {
				return -2
			}
			return 0
		})
		in := slices.Clone(orig[0])
		v, err := e.EmbedIDs(in)
		a.device.SetAllocMemoryHookForTest(nil)
		if v != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || !slices.Equal(in, orig[0]) {
			t.Fatalf("singleton OOM = (%v, %v)", v, err)
		}
		if a.device.LiveBuffersForTest() != heldBefore {
			t.Fatalf("singleton OOM leaked buffers")
		}
		if retryV, err := e.EmbedIDs(in); err != nil || len(retryV) == 0 {
			t.Fatalf("singleton retry = (%d, %v)", len(retryV), err)
		}

		// Attach partial weight upload OOM -> ErrDeviceOutOfMemory only, NO ErrBatchOutOfMemory, 0 leaked buffers.
		m := e.Model()
		var attachDev *vk.Device
		allocs := 0
		got, err := attachWithOpener(m, func() (*vk.Device, error) {
			attachDev = vk.NewFakeDeviceForTest(&vk.FakeHooksForTest{
				AllocateMemory: func(vk.Memory, uint64) int32 {
					allocs++
					if allocs == 8 {
						return -2
					}
					return 0
				},
			})
			return attachDev, nil
		})
		if got != nil || !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
			t.Fatalf("attach partial-upload OOM = (%v, %v)", got, err)
		}
		var re *vk.ResultError
		if !errors.As(err, &re) || re.Code != -2 {
			t.Fatalf("attach OOM native cause = %v", re)
		}
		if attachDev.LiveBuffersForTest() != 0 {
			t.Fatalf("attach OOM leaked buffers: %d", attachDev.LiveBuffersForTest())
		}

		// Unrelated attach failure -> neither OOM sentinel, 0 leaked buffers.
		allocs = 0
		got, err = attachWithOpener(m, func() (*vk.Device, error) {
			attachDev = vk.NewFakeDeviceForTest(&vk.FakeHooksForTest{
				BindMemory: func() int32 {
					allocs++
					if allocs == 8 {
						return -3 // VK_ERROR_INITIALIZATION_FAILED
					}
					return 0
				},
			})
			return attachDev, nil
		})
		if got != nil || err == nil || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) || errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) {
			t.Fatalf("unrelated attach failure = (%v, %v)", got, err)
		}
		if attachDev.LiveBuffersForTest() != 0 {
			t.Fatalf("unrelated attach failure leaked buffers: %d", attachDev.LiveBuffersForTest())
		}
	})

	// (4) Non-OOM input/limit/closed distinctions.
	t.Run("non-oom-distinctions", func(t *testing.T) {
		e, a := attachFakeVulkan(t, &vk.FakeHooksForTest{})
		V := e.Model().Config().VocabSize
		checkNotOOM := func(label string, err error) {
			t.Helper()
			if err == nil {
				t.Fatalf("%s: expected error", label)
			}
			if errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) {
				t.Fatalf("%s must not match OOM sentinels: %v", label, err)
			}
		}
		_, err := e.EmbedIDsBatch(nil)
		checkNotOOM("empty batch", err)
		_, err = e.EmbedIDsBatch([][]int{{1}, {}})
		checkNotOOM("empty member", err)
		_, err = e.EmbedIDsBatch([][]int{{1, 2}, {3, V}})
		checkNotOOM("invalid last ID", err)
		_, err = e.EmbedIDsBatch([][]int{make([]int, e.Model().Config().MaxPositions+1)})
		checkNotOOM("over MaxPositions", err)
		overAgg := [][]int{make([]int, 2048), make([]int, 2048), make([]int, 2048), make([]int, 2048), {1}}
		_, err = e.EmbedIDsBatch(overAgg)
		checkNotOOM("over Vulkan 8192 aggregate", err)

		if err := a.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		_, err = e.EmbedIDsBatch(orig[:1])
		checkNotOOM("closed encoder", err)
	})
}

// TestVulkanRetryOOM_LiveDevice runs a real tiny Vulkan encoder on the selected
// AMD device with real buffer allocations before injecting a deterministic
// VK_ERROR_OUT_OF_DEVICE_MEMORY (-2) at vkAllocateMemory, then verifies zero
// submissions, zero leaked buffers, and bit-identical split-batch retry outputs.
func TestVulkanRetryOOM_LiveDevice(t *testing.T) {
	id := requiredDevice(t)
	e := loadTinyVulkanEncoder(t)
	V := e.Model().Config().VocabSize
	orig := [][]int{{1, 2, 3, 4}, {5, 6, 7}, {8, 9, 10, 11}, {12, 13}}
	for i := range orig {
		for j := range orig[i] {
			orig[i][j] %= V
		}
	}

	// Baseline outputs on the live device.
	eBase := loadTinyVulkanEncoder(t)
	aBase, err := newEncoder(eBase.Model(), vk.Selector{DeviceID: id})
	if err != nil {
		t.Fatal(err)
	}
	defer aBase.Close()
	checkDevice(t, aBase, id)
	baseName := fmt.Sprintf("vulkan-issue25-live-base-%d", time.Now().UnixNano())
	embeddinggemma2.RegisterAccelerator(baseName, func(*embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return aBase, nil
	})
	if _, err := eBase.UseAccelerator(baseName); err != nil {
		t.Fatal(err)
	}
	wantBatch, err := eBase.EmbedIDsBatch(orig)
	if err != nil {
		t.Fatalf("baseline EmbedIDsBatch: %v", err)
	}

	// Live handle with injected VK_ERROR_OUT_OF_DEVICE_MEMORY (-2) on the 5th allocation.
	a, err := newEncoder(e.Model(), vk.Selector{DeviceID: id})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	checkDevice(t, a, id)
	liveName := fmt.Sprintf("vulkan-issue25-live-%d", time.Now().UnixNano())
	embeddinggemma2.RegisterAccelerator(liveName, func(*embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return a, nil
	})
	if _, err := e.UseAccelerator(liveName); err != nil {
		t.Fatal(err)
	}
	heldBefore := a.device.LiveBuffersForTest()
	subBefore := a.device.Activity().Submissions
	n := 0
	a.device.SetAllocMemoryHookForTest(func(vk.Memory, uint64) int32 {
		n++
		if n == 5 {
			return -2
		}
		return 0
	})
	in := make([][]int, len(orig))
	for i := range orig {
		in[i] = slices.Clone(orig[i])
	}
	got, err := e.EmbedIDsBatch(in)
	a.device.SetAllocMemoryHookForTest(nil)
	if got != nil || !errors.Is(err, embeddinggemma2.ErrBatchOutOfMemory) || !errors.Is(err, embeddinggemma2.ErrDeviceOutOfMemory) {
		t.Fatalf("live injected OOM = (%v, %v)", got, err)
	}
	for i := range orig {
		if !slices.Equal(in[i], orig[i]) {
			t.Fatalf("live injected OOM mutated input %d", i)
		}
	}
	if a.device.Activity().Submissions != subBefore {
		t.Fatalf("submissions advanced from %d to %d", subBefore, a.device.Activity().Submissions)
	}
	if live := a.device.LiveBuffersForTest(); live != heldBefore {
		t.Fatalf("live buffers after rollback = %d, want %d", live, heldBefore)
	}
	left, err := e.EmbedIDsBatch(in[:2])
	if err != nil {
		t.Fatalf("live left split retry: %v", err)
	}
	right, err := e.EmbedIDsBatch(in[2:])
	if err != nil {
		t.Fatalf("live right split retry: %v", err)
	}
	gotCombined := append(left, right...)
	for i := range orig {
		checkPooledBits(t, fmt.Sprintf("live split output %d", i), gotCombined[i], wantBatch[i])
	}
}
