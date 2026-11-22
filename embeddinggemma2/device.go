package embeddinggemma2

// MemoryBasis identifies how an available-byte figure was obtained.
type MemoryBasis uint8

const (
	// MemoryUnknown means there is no usable available-byte figure.
	MemoryUnknown MemoryBasis = iota
	// MemoryLive means the driver reported its current free capacity.
	MemoryLive
	// MemoryEstimated means the backend reported a headroom estimate.
	MemoryEstimated
)

func (b MemoryBasis) String() string {
	switch b {
	case MemoryLive:
		return "live"
	case MemoryEstimated:
		return "estimated"
	default:
		return "unknown"
	}
}

// DeviceMemory is an advisory snapshot that reserves no memory.
type DeviceMemory struct {
	AvailableBytes int64 // meaningful only when Basis is not MemoryUnknown
	TotalBytes     int64 // zero when capacity is unknown
	Basis          MemoryBasis
}

// DeviceMemoryReporter is an optional accelerator interface for its attached
// device. A reading may wait for a forward. An unsupported reading has
// MemoryUnknown basis. Calling after Close returns an error.
type DeviceMemoryReporter interface {
	DeviceMemory() (DeviceMemory, error)
}

// ProbeOptions controls model-free inventory. Identity and capacity are always
// returned. IncludeMemory also requests an available-byte estimate or reading.
type ProbeOptions struct {
	IncludeMemory bool
}
