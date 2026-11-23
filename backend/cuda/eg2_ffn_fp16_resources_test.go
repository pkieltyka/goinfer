//go:build cuda && eg2_combined && eg2_fp16 && goinfer_testhooks

package cuda

import (
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/pkieltyka/goinfer/backend/cuda/internal/gpudev"
)

func TestEG2FP16NativeResources(t *testing.T) {
	adapter := eg2FP16TestAdapter(t)
	dev, err := gpu.CreateSelectedDevice(adapter)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.ReleaseObjectsChecked(); err != nil {
			t.Error(err)
		}
	})
	p, err := eg2FP16Pipeline(dev, eg2FFNFP16PTX)
	if err != nil {
		t.Fatal(err)
	}
	f := rawFunc(p)
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
	t.Logf("SM%d.%d local=%d B/thread static shared=%d B/CTA registers=%d/thread", adapter.Major, adapter.Minor, local, shared, registers)
}
