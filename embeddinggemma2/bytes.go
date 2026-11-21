package embeddinggemma2

import (
	"encoding/binary"
	"fmt"
	"io/fs"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/pkieltyka/aikit/embed"
)

// bytes.go builds the text encoder from weights the caller already holds in memory. Load reads a checkpoint
// directory by path; a caller that has read its files once and checked them against digests must not have them read
// a second time from that path, where they may have changed since. NewModel is the same model from those bytes.

// layersName is the name the in-memory layers file is opened under. It never reaches a file system.
const layersName = "layers.safetensors"

// NewModel builds the text encoder from weights that are already in memory.
//
// tokens is the token-embedding table, language_model.embed_tokens.weight, as the checkpoint stores it: vocab_size x
// hidden_size little-endian values, bfloat16 (2 bytes each) or float32 (4 bytes each), told apart by its length. It is
// kept as it is and one row is widened per token, so the table is never copied or widened whole. The caller must not
// modify tokens while the Model is in use.
//
// layers is a safetensors file that holds every other language_model.* tensor of the checkpoint, under its checkpoint
// name and in float32, bfloat16 or float16; tensors it holds beyond those are ignored. Each is turned into float32
// once, across GOMAXPROCS goroutines. layers is not retained: the caller may release or reuse it when NewModel returns.
//
// The vectors are those of a model Load builds from the same weights, bit for bit.
func NewModel(cfg Config, tokens, layers []byte) (*Model, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	n := cfg.VocabSize * cfg.Hidden
	var size int
	switch len(tokens) {
	case 2 * n:
		size = 2
	case 4 * n:
		size = 4
	default:
		return nil, fmt.Errorf("embeddinggemma2: token table is %d bytes, want %d (bfloat16) or %d (float32) for %d x %d",
			len(tokens), 2*n, 4*n, cfg.VocabSize, cfg.Hidden)
	}
	st, err := embed.OpenSafetensorsFromFS(oneFile(layers), layersName)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: layers: %w", err)
	}
	m := &Model{cfg: cfg, layers: make([]layer, cfg.Layers), embedRaw: tokens, embedSize: size}
	refs, scalars, err := m.plan()
	if err != nil {
		return nil, err
	}
	workers := min(runtime.GOMAXPROCS(0), len(refs))
	errs := make([]error, workers)
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() {
			for {
				i := int(next.Add(1)) - 1
				if i >= len(refs) {
					return
				}
				if err := fillRef(st, refs[i]); err != nil && errs[w] == nil {
					errs[w] = err
				}
			}
		})
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	m.setScalars(scalars)
	return m, nil
}

// fillRef reads one weight as float32 into its place. A bfloat16 or float16 tensor is widened into a slice of its
// own; a float32 tensor comes back as a view of the layers bytes, which NewModel does not keep, so it is copied.
func fillRef(st *embed.SafetensorsFile, r weightRef) error {
	name := "language_model." + r.name
	t, err := st.Tensor(name)
	if err != nil {
		return fmt.Errorf("embeddinggemma2: %s: %w", r.name, err)
	}
	v, err := st.TensorF32(name, r.shape...)
	if err != nil {
		return fmt.Errorf("embeddinggemma2: %s: %w", r.name, err)
	}
	if t.DType == "F32" {
		v = append([]float32(nil), v...)
	}
	*r.dst = v
	return nil
}

// rawRow writes token id's embedding row, times scale, into dst (hidden values), widening it from the stored table.
// The product is the one Load's widened table gives, so the two paths agree bit for bit.
func (m *Model) rawRow(id int, scale float32, dst []float32) {
	H := m.cfg.Hidden
	if m.embedSize == 2 {
		row := m.embedRaw[id*H*2 : (id+1)*H*2]
		for j := range dst {
			dst[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[2*j:]))<<16) * scale
		}
		return
	}
	row := m.embedRaw[id*H*4 : (id+1)*H*4]
	for j := range dst {
		dst[j] = math.Float32frombits(binary.LittleEndian.Uint32(row[4*j:])) * scale
	}
}

// embedF32 is the token table as one float32 slice, [vocab, hidden]. A model from Load has it already; a model from
// NewModel widens its stored table on the first call and keeps the result.
func (m *Model) embedF32() []float32 {
	if m.embedRaw == nil {
		return m.embed
	}
	m.embedOnce.Do(func() {
		H := m.cfg.Hidden
		e := make([]float32, m.cfg.VocabSize*H)
		for id := range m.cfg.VocabSize {
			m.rawRow(id, 1, e[id*H:(id+1)*H])
		}
		m.embed = e
	})
	return m.embed
}

// oneFile serves one in-memory file, under layersName, to a reader that takes an fs.FS. ReadFile hands out the bytes
// themselves, with no copy; nothing else can be opened.
type oneFile []byte

func (f oneFile) Open(name string) (fs.File, error) {
	return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
}

func (f oneFile) ReadFile(name string) ([]byte, error) {
	if name != layersName {
		return nil, &fs.PathError{Op: "read", Path: name, Err: fs.ErrNotExist}
	}
	return f, nil
}
