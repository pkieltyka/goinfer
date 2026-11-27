package embeddinggemma2

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/pkieltyka/goinfer/internal/cudaid"
)

// MemoryKind describes how a device's memory is provisioned.
type MemoryKind uint8

const (
	MemoryKindUnknown MemoryKind = iota
	MemoryDedicated              // the device has its own memory
	MemoryShared                 // the device shares host memory
)

// CUDAComputeCapability is the native CUDA compute capability. A zero value
// means the capability is unavailable or inapplicable.
type CUDAComputeCapability struct {
	Major, Minor int
}

// DeviceInfo describes one device a linked backend can attach.
type DeviceInfo struct {
	ID, Alias           string // backend-qualified; both empty when the device has no usable identifier. ID is stable within the backend's documented scope (CUDA UUID; Metal boot session); Alias is current-order only.
	Backend             string // device namespace and default selector
	Name                string
	VendorID, ProductID uint32 // PCI vendor and product IDs, zero when unknown
	MemoryKind          MemoryKind
	Memory              DeviceMemory // total when known; available bytes when requested and supported
	TextBatching        bool         // native EmbedIDsBatch on the reported attachment or inventory default
	ComputeCapability   CUDAComputeCapability
}

// ProbeFailure records a device inventory error from one backend.
type ProbeFailure struct {
	Backend string
	Err     error
}

// AttachedDeviceReporter provides the actual device identity captured at attach.
// For a fresh available-byte snapshot, use DeviceMemoryReporter.
type AttachedDeviceReporter interface {
	AttachedDevice() DeviceInfo
}

var deviceProbes = map[string]func(ProbeOptions) ([]DeviceInfo, error){}
var deviceInventoryProbes = map[string]func(ProbeOptions) ([]DeviceInfo, error){}

type deviceFactory struct {
	backend  string
	open     func(*Model, string) (Accelerator, error)
	supports func(DeviceInfo) bool
}

var selectedAccels = map[string]deviceFactory{}

// RegisterDeviceProber makes a backend's devices visible to Devices.
func RegisterDeviceProber(backend string, probe func(ProbeOptions) ([]DeviceInfo, error)) {
	accelMu.Lock()
	defer accelMu.Unlock()
	deviceProbes[backend] = probe
}

// RegisterDeviceInventoryProber registers a backend's addressable inventory.
// AllDevices consults it; Devices keeps the registered legacy prober.
func RegisterDeviceInventoryProber(backend string, probe func(ProbeOptions) ([]DeviceInfo, error)) {
	accelMu.Lock()
	defer accelMu.Unlock()
	deviceInventoryProbes[backend] = probe
}

// Devices asks every registered prober, in backend-name order. It returns
// successful rows and any per-backend failures, including rows from a partial probe.
// A backend may keep this legacy/default inventory separate from its explicit
// addressable inventory; use AllDevices for the latter.
func Devices(opts ProbeOptions) ([]DeviceInfo, []ProbeFailure) {
	return devicesThrough(deviceProbes, opts)
}

// AllDevices asks every backend for its addressable inventory, with
// backend-qualified IDs and aliases where it supports them. A backend without
// a separate inventory prober contributes its legacy rows. This is additive:
// Devices and ProbeOptions keep their existing shape and default behavior.
func AllDevices(opts ProbeOptions) ([]DeviceInfo, []ProbeFailure) {
	accelMu.Lock()
	probes := maps.Clone(deviceProbes)
	for name, probe := range deviceInventoryProbes {
		probes[name] = probe
	}
	accelMu.Unlock()
	return devicesThrough(probes, opts)
}

func devicesThrough(probes map[string]func(ProbeOptions) ([]DeviceInfo, error), opts ProbeOptions) ([]DeviceInfo, []ProbeFailure) {
	var devices []DeviceInfo
	var failures []ProbeFailure
	for _, name := range slices.Sorted(maps.Keys(probes)) {
		probe := probes[name]
		if probe == nil {
			failures = append(failures, ProbeFailure{Backend: name, Err: fmt.Errorf("%w: nil device prober", ErrBackendUnavailable)})
			continue
		}
		rows, err := probe(opts)
		devices = append(devices, rows...)
		if err != nil {
			failures = append(failures, ProbeFailure{Backend: name, Err: err})
		}
	}
	return devices, failures
}

// RegisterDeviceAccelerator registers selector as a kernel choice for explicit
// devices of backend. The factory receives the part of the identifier after ':'.
func RegisterDeviceAccelerator(selector, backend string, factory func(*Model, string) (Accelerator, error)) {
	RegisterDeviceAcceleratorWithSupport(selector, backend, factory, nil)
}

// RegisterDeviceAcceleratorWithSupport also registers a hardware predicate for
// RunnableSelectors. The predicate uses only the supplied inventory row and
// must not probe the device or open an accelerator. A nil predicate leaves
// eligibility unknown.
func RegisterDeviceAcceleratorWithSupport(selector, backend string, factory func(*Model, string) (Accelerator, error), supports func(DeviceInfo) bool) {
	accelMu.Lock()
	defer accelMu.Unlock()
	selectedAccels[selector] = deviceFactory{backend: backend, open: factory, supports: supports}
}

// RunnableSelectors returns a sorted, caller-owned list of linked selectors
// whose hardware requirements this CUDA row satisfies for UseDevice(d.ID, name).
// It returns nil for other backends and rows without a usable persistent ID
// or verified hardware eligibility.
// Attachment still checks device identity, model compatibility and memory fit.
func (d DeviceInfo) RunnableSelectors() []string {
	if d.Backend != "cuda" {
		return nil
	}
	token, qualified := strings.CutPrefix(d.ID, "cuda:")
	if !qualified || !cudaid.ValidUUID(token) || d.ID == d.Alias {
		return nil
	}
	accelMu.Lock()
	candidates := make(map[string]func(DeviceInfo) bool)
	for name, factory := range selectedAccels {
		if name != "" && factory.backend == d.Backend && factory.open != nil && factory.supports != nil {
			candidates[name] = factory.supports
		}
	}
	accelMu.Unlock()

	var selectors []string
	for name, supports := range candidates {
		if supports(d) {
			selectors = append(selectors, name)
		}
	}
	slices.Sort(selectors)
	return selectors
}

// DeviceSelectors lists the linked kernel selectors for explicit devices of
// backend, sorted by name. A listed selector can still fail at attachment.
func DeviceSelectors(backend string) []string {
	accelMu.Lock()
	defer accelMu.Unlock()
	var selectors []string
	for name, factory := range selectedAccels {
		if factory.backend == backend && factory.open != nil {
			selectors = append(selectors, name)
		}
	}
	slices.Sort(selectors)
	return selectors
}

// UseDevice attaches a device by backend-qualified ID or alias, with the named
// kernel selector. An empty selector uses the ID's backend. On failure the
// encoder is unchanged. Vision and audio towers stay on the CPU.
func (e *Encoder) UseDevice(id, selector string) (Accelerator, error) {
	backend, token, qualified := strings.Cut(id, ":")
	if !qualified || backend == "" || token == "" {
		return nil, fmt.Errorf("%w: expected a backend-qualified device ID or alias, got %q", ErrDeviceUnavailable, id)
	}
	if selector == "" {
		selector = backend
	}
	accelMu.Lock()
	factory := selectedAccels[selector]
	accelMu.Unlock()
	if factory.open == nil {
		return nil, fmt.Errorf("%w: no %q device selector in this binary", ErrBackendUnavailable, selector)
	}
	if factory.backend != backend {
		return nil, fmt.Errorf("%w: selector %q cannot attach a %q device", ErrDeviceUnavailable, selector, backend)
	}
	a, err := factory.open(e.m, token)
	a, err = checkAccelerator(selector, a, err)
	if err != nil {
		return nil, err
	}
	e.attachAccelerator(a, true)
	return a, nil
}
