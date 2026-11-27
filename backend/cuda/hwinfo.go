//go:build cuda

package cuda

import (
	"fmt"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/pkieltyka/goinfer/backend/cuda/internal/tower"
	"github.com/pkieltyka/goinfer/decoder"
)

// ptxTargetMajor and ptxTargetMinor are the minimum compute capability for
// the default CUDA kernels. A newer card uses the driver's forward JIT.
const ptxTargetMajor, ptxTargetMinor = tower.PTXTargetMajor, tower.PTXTargetMinor

func init() { decoder.RegisterHardwareInfo("cuda", cudaHardwareInfo) }

// cudaHardwareInfo is the device facts `check --hardware` prints for the CUDA backend. It opens the device only when asked, and says so plainly when there is none.
func cudaHardwareInfo() []string {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		return []string{"no usable device: " + err.Error()}
	}
	defer dev.ReleaseObjects()
	ctx := dev.Context()
	d := ctx.Device()
	out := []string{"device: " + dev.Name()}
	maj, e1 := d.Attribute(gc.DeviceAttributeComputeCapabilityMajor)
	min, e2 := d.Attribute(gc.DeviceAttributeComputeCapabilityMinor)
	if e1 == nil && e2 == nil {
		how := "default CUDA PTX targets sm_75 and is JIT-compiled forward by the driver for this card"
		switch {
		case maj == ptxTargetMajor && min == ptxTargetMinor:
			how = "default CUDA PTX targets this card's sm_75 architecture"
		case maj < ptxTargetMajor || (maj == ptxTargetMajor && min < ptxTargetMinor):
			how = "this card is older than the default CUDA PTX target sm_75"
		}
		out = append(out, fmt.Sprintf("compute capability: %d.%d (%s)", maj, min, how))
	}
	if n := dev.SMCount(); n > 0 {
		out = append(out, fmt.Sprintf("multiprocessors: %d", n))
	}
	if free, total, err := ctx.MemInfo(); err == nil {
		out = append(out, fmt.Sprintf("memory: %.1f GiB total, %.1f GiB free", float64(total)/(1<<30), float64(free)/(1<<30)))
	}
	if v, err := gc.DriverVersion(); err == nil {
		out = append(out, fmt.Sprintf("driver supports CUDA %d.%d", v/1000, (v%1000)/10))
	}
	return out
}
