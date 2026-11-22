package embeddinggemma2

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
)

// Accelerator runs the encoder's forward on another device (Phase M of : Metal).
// It returns what Model.forward returns: the projected last hidden state, [len(ids), Dim()] row-major, and, with
// keepLayers, every layer's input and the last layer's output for per-layer differencing. Pooling, normalisation and
// tokenization stay in this package, shared with the CPU path.
type Accelerator interface {
	Name() string
	Forward(ids []int, keepLayers bool) (last []float32, layers [][]float32, err error)
	// ForwardEmbeds is Forward from T prepared input rows x [T, hidden] (Model.EmbedTokens with any image rows
	// spliced in); x is not modified.
	ForwardEmbeds(x []float32, T int, keepLayers bool) (last []float32, layers [][]float32, err error)
	Close() error
}

var (
	// ErrDeviceUnavailable covers malformed selectors and unavailable device selection.
	ErrDeviceUnavailable = errors.New("embeddinggemma2: device unavailable")
	// ErrBackendUnavailable means the requested backend is missing or could not start.
	ErrBackendUnavailable = fmt.Errorf("%w: backend runtime unavailable", ErrDeviceUnavailable)
	// ErrNoDevice means the runtime started but no eligible device matched.
	ErrNoDevice = fmt.Errorf("%w: no eligible device", ErrDeviceUnavailable)
	// ErrAmbiguousDevice means more than one eligible device matched.
	ErrAmbiguousDevice = fmt.Errorf("%w: several eligible devices", ErrDeviceUnavailable)
	// ErrDeviceOutOfMemory means a native memory allocation failed. It is a
	// separate category from ErrDeviceUnavailable and carries no retry guarantee.
	ErrDeviceOutOfMemory = errors.New("embeddinggemma2: device out of memory")
	// ErrDeviceLost means the accelerator can no longer run forwards.
	// Close it. A later attempt requires a fresh accelerator.
	ErrDeviceLost = errors.New("embeddinggemma2: device lost")
)

// DeviceLostError preserves the accelerator's native failure.
type DeviceLostError struct {
	Cause error
}

func (e *DeviceLostError) Error() string {
	if e.Cause == nil {
		return ErrDeviceLost.Error()
	}
	return ErrDeviceLost.Error() + ": " + e.Cause.Error()
}

func (e *DeviceLostError) Unwrap() error        { return e.Cause }
func (e *DeviceLostError) Is(target error) bool { return target == ErrDeviceLost }

var (
	accelMu sync.Mutex
	accels  = map[string]func(*Model) (Accelerator, error){}
)

// RegisterAccelerator makes an accelerator available by name. A backend module registers itself from init (the root
// module cannot import it), as decoder.RegisterBackend does for the decoder's backends.
func RegisterAccelerator(name string, factory func(*Model) (Accelerator, error)) {
	accelMu.Lock()
	defer accelMu.Unlock()
	accels[name] = factory
}

// RegisterSelectedAccelerator registers the factory used for a backend-qualified
// device identifier passed to UseAccelerator or NewAccelerator.
func RegisterSelectedAccelerator(name string, factory func(*Model, string) (Accelerator, error)) {
	RegisterDeviceAccelerator(name, name, factory)
}

// Accelerators lists the registered accelerator names, sorted.
func Accelerators() []string {
	accelMu.Lock()
	defer accelMu.Unlock()
	var n []string
	for k := range accels {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// NewAccelerator builds the named accelerator over this model's weights.
func (m *Model) NewAccelerator(name string) (Accelerator, error) {
	a, _, err := m.newAccelerator(name)
	return a, err
}

func (m *Model) newAccelerator(name string) (Accelerator, bool, error) {
	accelMu.Lock()
	f := accels[name]
	backend, id, qualified := strings.Cut(name, ":")
	selected := selectedAccels[backend].open
	accelMu.Unlock()
	deviceSelected := f == nil && qualified && selected != nil
	if deviceSelected {
		f = func(m *Model) (Accelerator, error) { return selected(m, id) }
	}
	if f == nil && qualified {
		return nil, false, fmt.Errorf("%w: no %q device selector in this binary", ErrBackendUnavailable, backend)
	}
	if f == nil {
		return nil, false, fmt.Errorf("%w: no %q accelerator in this binary (registered: %v)", ErrBackendUnavailable, name, Accelerators())
	}
	a, err := f(m)
	a, err = checkAccelerator(name, a, err)
	return a, deviceSelected, err
}

func checkAccelerator(name string, a Accelerator, err error) (Accelerator, error) {
	if err != nil {
		return a, err
	}
	value := reflect.ValueOf(a)
	if !value.IsValid() {
		return nil, fmt.Errorf("embeddinggemma2: %q returned no accelerator", name)
	}
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Map, reflect.Pointer, reflect.Slice:
		if value.IsNil() {
			return nil, fmt.Errorf("embeddinggemma2: %q returned a nil accelerator", name)
		}
	}
	return a, nil
}

// ForwardCPU is the CPU forward an accelerator is compared with: the projected last hidden state and, with
// keepLayers, every layer's input and the last layer's output.
func (m *Model) ForwardCPU(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	return m.forward(ids, keepLayers)
}

// LayerWeights is one encoder layer's weights as an accelerator uploads them: row-major [out, in] as nn.Linear
// stores them, float32, read-only.
type LayerWeights struct {
	Full             bool // full attention; otherwise a symmetric window of radius Config.SlidingWindow
	HeadDim, KVHeads int
	RopeTheta        float64
	InNorm           []float32
	Q, K, V, O       []float32
	QNorm, KNorm     []float32
	PostAttnNorm     []float32
	PreFFNorm        []float32
	Gate, Up, Down   []float32
	PostFFNorm       []float32
	PLEGate, PLEProj []float32
	PLEPostNorm      []float32
	LayerScalar      float32
}

// Weights is the whole text encoder as an accelerator uploads it. The slices alias the model's own; do not write
// them.
type Weights struct {
	Config     Config
	Embed      []float32 // [vocab, hidden]
	PLEProj    []float32 // [layers*ple, hidden]
	PLENorm    []float32 // [ple]
	Layers     []LayerWeights
	FinalNorm  []float32
	Projection []float32 // [embedding_dim, hidden]
}

// Weights returns the model's weights for an accelerator. A packed token table is widened to float32
// the first time this is called.
func (m *Model) Weights() Weights {
	w := m.TransformerWeights()
	w.Embed = m.embedF32()
	return w
}

// TransformerWeights returns the read-only encoder weights with Embed left nil.
// Accelerators that obtain input rows through EmbedTokens can leave a packed token table packed.
func (m *Model) TransformerWeights() Weights {
	w := Weights{Config: m.cfg, PLEProj: m.pleProj, PLENorm: m.pleNorm, FinalNorm: m.finalNorm,
		Projection: m.projection, Layers: make([]LayerWeights, len(m.layers))}
	for i := range m.layers {
		l := &m.layers[i]
		w.Layers[i] = LayerWeights{Full: l.full, HeadDim: l.headDim, KVHeads: l.kvH, RopeTheta: l.theta, InNorm: l.inNorm,
			Q: l.q, K: l.k, V: l.v, O: l.o, QNorm: l.qNorm, KNorm: l.kNorm, PostAttnNorm: l.postAttnNorm,
			PreFFNorm: l.preFFNorm, Gate: l.gate, Up: l.up, Down: l.down, PostFFNorm: l.postFFNorm, PLEGate: l.pleGate,
			PLEProj: l.pleProj, PLEPostNorm: l.plePostNorm, LayerScalar: l.scalar}
	}
	return w
}

// RopeTables returns the cos and sin tables the CPU forward uses for T positions at head dim hd and base theta,
// [T, hd/2] each, so an accelerator rotates with the same float32 values.
func RopeTables(T, hd int, theta float64) (cos, sin []float32) { return ropeTables(T, hd, theta) }

// UseAccelerator makes the encoder run its forward on the named accelerator or
// a device identified by a linked backend; EmbedIDs, EmbedText and the aikit methods
// then use it. A bare selector also moves the vision and audio towers when compatible tower accelerators
// are registered. An explicit device ID or alias leaves both towers on the CPU. An error leaves the encoder unchanged.
func (e *Encoder) UseAccelerator(name string) (Accelerator, error) {
	a, deviceSelected, err := e.m.newAccelerator(name)
	if err != nil {
		return nil, err
	}
	e.attachAccelerator(a, deviceSelected)
	return a, nil
}

func (e *Encoder) attachAccelerator(a Accelerator, deviceSelected bool) {
	e.accel, e.deviceSelected = a, deviceSelected
	e.bindVisionAccel() // towers already loaded move too
	e.bindAudioAccel()
}

const selectedDeviceTowerReason = "CPU (tower acceleration requires a bare selector)"

// Accelerator is the accelerator the encoder runs on, or nil on the CPU path.
func (e *Encoder) Accelerator() Accelerator { return e.accel }

// embedIDs is the sentence embedding of ids on whichever path the encoder runs.
func (e *Encoder) embedIDs(ids []int) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("embeddinggemma2: no input ids")
	}
	if e.accel == nil {
		return e.m.Embed(ids)
	}
	h, _, err := e.accel.Forward(ids, false)
	if err != nil {
		return nil, err
	}
	want := len(ids) * e.m.cfg.EmbeddingDim
	if len(h) != want {
		return nil, fmt.Errorf("embeddinggemma2: %s returned %d hidden values, want %d", e.accel.Name(), len(h), want)
	}
	return poolNormalize(h, len(ids), e.m.cfg.EmbeddingDim), nil
}
