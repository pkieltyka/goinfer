//go:build darwin

package metal

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"

	gpu "github.com/pkieltyka/goinfer/backend/metal/internal/gpudev"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func probeMetalDevice(opts embeddinggemma2.ProbeOptions) ([]embeddinggemma2.DeviceInfo, error) {
	return probeMetalDeviceOn(CreateSystemDefaultDevice, opts)
}

// probeMetalDevices is the additive addressable inventory registered with
// AllDevices. probeMetalDevice stays the legacy system-default row.
func probeMetalDevices(opts embeddinggemma2.ProbeOptions) ([]embeddinggemma2.DeviceInfo, error) {
	return probeMetalDevicesOn(gpu.AllDeviceFacts, opts)
}

func init() {
	for selector, request := range map[string]eg2AttentionRequest{
		"metal":              eg2AttentionDefault,
		"metal-eg2-stock":    eg2AttentionStock,
		"metal-eg2-combined": eg2AttentionMatrix,
		"metal-eg2-batch":    eg2AttentionBatch,
	} {
		embeddinggemma2.RegisterDeviceAccelerator(selector, "metal", metalDeviceFactory(request, selector == "metal-eg2-batch"))
	}
}

func acquireMetalDevice(open func() (*Device, error)) (*Device, error) {
	device, err := open()
	if err == nil {
		return device, nil
	}
	if errors.Is(err, gpu.ErrNoDevice) {
		return nil, fmt.Errorf("%w: %w", embeddinggemma2.ErrNoDevice, err)
	}
	// An explicit selection already names its reason (no device, several
	// matches, or a malformed token). Keep it instead of collapsing it into the
	// sibling runtime-unavailable category.
	if errors.Is(err, embeddinggemma2.ErrDeviceUnavailable) {
		return nil, err
	}
	return nil, fmt.Errorf("%w: %w", embeddinggemma2.ErrBackendUnavailable, err)
}

func probeMetalDeviceOn(open func() (*Device, error), opts embeddinggemma2.ProbeOptions) ([]embeddinggemma2.DeviceInfo, error) {
	device, err := acquireMetalDevice(open)
	if errors.Is(err, embeddinggemma2.ErrNoDevice) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer device.ReleaseObjects()
	return []embeddinggemma2.DeviceInfo{describeMetalDevice(device.Name(), device.HasUnifiedMemory(), opts.IncludeMemory)}, nil
}

func probeMetalDevicesOn(list func() ([]gpu.DeviceFact, error), opts embeddinggemma2.ProbeOptions) ([]embeddinggemma2.DeviceInfo, error) {
	facts, err := list()
	if errors.Is(err, gpu.ErrNoDevice) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %w", embeddinggemma2.ErrBackendUnavailable, err)
	}
	return metalDeviceRows(facts, metalBootSession(), opts.IncludeMemory), nil
}

// metalDeviceRows assigns a boot-scoped exact ID only to a unique, nonzero
// registry ID. A zero or duplicated registry ID leaves both ID and alias blank:
// the row is visible but not addressable. Aliases follow the current inventory
// order and are for interactive use; stored selections must use the exact ID.
func metalDeviceRows(facts []gpu.DeviceFact, boot string, includeMemory bool) []embeddinggemma2.DeviceInfo {
	counts := make(map[uint64]int)
	for _, f := range facts {
		if f.RegistryID != 0 {
			counts[f.RegistryID]++
		}
	}
	rows := make([]embeddinggemma2.DeviceInfo, 0, len(facts))
	for i, f := range facts {
		id, alias := "", ""
		if boot != "" && f.RegistryID != 0 && counts[f.RegistryID] == 1 {
			id = fmt.Sprintf("metal:%d@%s", f.RegistryID, boot)
			alias = fmt.Sprintf("metal:%d", i)
		}
		rows = append(rows, describeMetalDeviceAt(f.Name, id, alias, f.Unified, includeMemory))
	}
	return rows
}

func describeMetalDevice(name string, unified, includeMemory bool) embeddinggemma2.DeviceInfo {
	return describeMetalDeviceAt(name, "", "", unified, includeMemory)
}

func describeMetalDeviceAt(name, id, alias string, unified, includeMemory bool) embeddinggemma2.DeviceInfo {
	info := embeddinggemma2.DeviceInfo{ID: id, Alias: alias, Backend: "metal", Name: name, MemoryKind: embeddinggemma2.MemoryDedicated}
	if unified {
		info.MemoryKind = embeddinggemma2.MemoryShared
		info.Memory = metalSharedMemory(includeMemory)
	}
	return info
}

func metalSharedMemory(includeMemory bool) embeddinggemma2.DeviceMemory {
	ram, err := physicalRAM()
	if err != nil || ram == 0 {
		return embeddinggemma2.DeviceMemory{}
	}
	memory := embeddinggemma2.DeviceMemory{TotalBytes: int64(ram)}
	if includeMemory {
		memory.AvailableBytes = metalMemoryCeiling(ram)
		memory.Basis = embeddinggemma2.MemoryEstimated
	}
	return memory
}

func metalBootSession() string {
	if boot, err := syscall.Sysctl("kern.bootsessionuuid"); err == nil {
		if boot = strings.TrimSpace(boot); boot != "" {
			return boot
		}
	}
	if tv, err := unix.SysctlTimeval("kern.boottime"); err == nil && tv != nil {
		return fmt.Sprintf("%d.%06d", tv.Sec, tv.Usec)
	}
	return ""
}

type metalToken struct {
	exact      bool
	registryID uint64
	boot       string
	ordinal    int
}

func parseMetalToken(token string) (metalToken, error) {
	if strings.Contains(token, "@") {
		id, boot, ok := strings.Cut(token, "@")
		if !ok || id == "" || boot == "" {
			return metalToken{}, fmt.Errorf("%w: malformed Metal device ID %q", embeddinggemma2.ErrDeviceUnavailable, token)
		}
		reg, err := strconv.ParseUint(id, 10, 64)
		if err != nil || reg == 0 {
			return metalToken{}, fmt.Errorf("%w: malformed Metal registry ID %q", embeddinggemma2.ErrDeviceUnavailable, token)
		}
		return metalToken{exact: true, registryID: reg, boot: boot}, nil
	}
	idx, err := strconv.Atoi(token)
	if err != nil || idx < 0 {
		return metalToken{}, fmt.Errorf("%w: Metal device %q is neither a boot-scoped ID nor an ordinal", embeddinggemma2.ErrDeviceUnavailable, token)
	}
	return metalToken{ordinal: idx}, nil
}

// metalAttach is the identity resolved from the same array snapshot that
// supplied the retained device. Callers must not recompute the boot marker.
type metalAttach struct {
	id    string
	alias string
}

// openMetalDevice resolves one token against one owned array snapshot and
// retains only the selected member. Numeric aliases are refused for a zero or
// duplicate registry ID and when the boot marker is missing.
func openMetalDevice(tok metalToken) (*Device, metalAttach, error) {
	boot := metalBootSession()
	selectedIndex := -1
	selected, fact, err := gpu.SelectDevice(func(facts []gpu.DeviceFact) (int, error) {
		index, err := selectMetalIndex(facts, boot, tok)
		selectedIndex = index
		return index, err
	})
	if errors.Is(err, gpu.ErrNoDevice) {
		return nil, metalAttach{}, fmt.Errorf("%w: %w", embeddinggemma2.ErrNoDevice, err)
	}
	if err != nil {
		return nil, metalAttach{}, err
	}
	if fact.RegistryID == 0 || selected.RegistryID() != fact.RegistryID {
		selected.ReleaseObjects()
		return nil, metalAttach{}, fmt.Errorf("%w: Metal device changed between inventory and retain", embeddinggemma2.ErrNoDevice)
	}
	return selected, metalAttach{
		id:    fmt.Sprintf("metal:%d@%s", fact.RegistryID, boot),
		alias: fmt.Sprintf("metal:%d", selectedIndex),
	}, nil
}

func selectMetalIndex(facts []gpu.DeviceFact, boot string, tok metalToken) (int, error) {
	counts := make(map[uint64]int)
	for _, f := range facts {
		if f.RegistryID != 0 {
			counts[f.RegistryID]++
		}
	}
	if tok.exact {
		if boot == "" || tok.boot != boot {
			return -1, fmt.Errorf("%w: Metal device token belongs to a different boot", embeddinggemma2.ErrNoDevice)
		}
		selected := -1
		for i, f := range facts {
			if f.RegistryID != tok.registryID {
				continue
			}
			if selected >= 0 {
				return -1, fmt.Errorf("%w: several Metal devices report registry ID %d", embeddinggemma2.ErrAmbiguousDevice, tok.registryID)
			}
			selected = i
		}
		if selected < 0 {
			return -1, fmt.Errorf("%w: no Metal device matches %q", embeddinggemma2.ErrNoDevice, tokenString(tok))
		}
		return selected, nil
	}
	if boot == "" {
		return -1, fmt.Errorf("%w: Metal boot session is unavailable", embeddinggemma2.ErrNoDevice)
	}
	if tok.ordinal < 0 || tok.ordinal >= len(facts) {
		return -1, fmt.Errorf("%w: no Metal device matches %q", embeddinggemma2.ErrNoDevice, tokenString(tok))
	}
	f := facts[tok.ordinal]
	if f.RegistryID == 0 {
		return -1, fmt.Errorf("%w: Metal row %d has no registry ID", embeddinggemma2.ErrNoDevice, tok.ordinal)
	}
	if counts[f.RegistryID] != 1 {
		return -1, fmt.Errorf("%w: Metal registry ID %d is duplicated", embeddinggemma2.ErrAmbiguousDevice, f.RegistryID)
	}
	return tok.ordinal, nil
}

func tokenString(tok metalToken) string {
	if tok.exact {
		return fmt.Sprintf("%d@%s", tok.registryID, tok.boot)
	}
	return strconv.Itoa(tok.ordinal)
}

func metalDeviceFactory(request eg2AttentionRequest, batch bool) func(*embeddinggemma2.Model, string) (embeddinggemma2.Accelerator, error) {
	return func(m *embeddinggemma2.Model, token string) (embeddinggemma2.Accelerator, error) {
		tok, err := parseMetalToken(token)
		if err != nil {
			return nil, err
		}
		var attached metalAttach
		open := func() (*Device, error) {
			var openErr error
			var d *Device
			d, attached, openErr = openMetalDevice(tok)
			return d, openErr
		}
		a, err := newEG2AccelWithAttentionOn(m, request, open)
		if err != nil {
			return nil, err
		}
		a.identity = describeMetalDeviceAt(a.d.Name(), attached.id, attached.alias, a.d.HasUnifiedMemory(), false)
		if batch {
			return &eg2BatchAccel{eg2Accel: a}, nil
		}
		return a, nil
	}
}

var _ embeddinggemma2.AttachedDeviceReporter = (*eg2Accel)(nil)
var _ embeddinggemma2.DeviceMemoryReporter = (*eg2Accel)(nil)

// AttachedDevice returns the device captured at attach. A bare
// UseAccelerator("metal") keeps the legacy blank ID/alias; an explicit device
// attach reports its boot-scoped ID and current ordinal alias.
func (a *eg2Accel) AttachedDevice() embeddinggemma2.DeviceInfo { return a.identity }

// DeviceMemory estimates headroom when the attached device shares host memory.
func (a *eg2Accel) DeviceMemory() (embeddinggemma2.DeviceMemory, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.open(); err != nil {
		return embeddinggemma2.DeviceMemory{}, err
	}
	if a.identity.MemoryKind == embeddinggemma2.MemoryShared {
		return metalSharedMemory(true), nil
	}
	return a.identity.Memory, nil
}
