package gguf

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
)

type kv struct {
	key  string
	typ  valueType
	data []byte
}

type entry struct {
	name  string
	shape []uint64
	typ   Type
	off   uint64
}

func packed(v any) []byte {
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func str(s string) []byte { return append(packed(uint64(len(s))), s...) }

func arr(typ valueType, n uint64, data ...[]byte) []byte {
	return bytes.Join(append([][]byte{packed(typ), packed(n)}, data...), nil)
}

func fixture(kvs []kv, tensors []entry, alignment int, data []byte) []byte {
	b := []byte("GGUF")
	b = binary.LittleEndian.AppendUint32(b, 3)
	b = binary.LittleEndian.AppendUint64(b, uint64(len(tensors)))
	b = binary.LittleEndian.AppendUint64(b, uint64(len(kvs)))
	for _, m := range kvs {
		b = append(b, str(m.key)...)
		b = binary.LittleEndian.AppendUint32(b, uint32(m.typ))
		b = append(b, m.data...)
	}
	for _, e := range tensors {
		b = append(b, str(e.name)...)
		b = binary.LittleEndian.AppendUint32(b, uint32(len(e.shape)))
		for _, d := range e.shape {
			b = binary.LittleEndian.AppendUint64(b, d)
		}
		b = binary.LittleEndian.AppendUint32(b, uint32(e.typ))
		b = binary.LittleEndian.AppendUint64(b, e.off)
	}
	if len(tensors) != 0 {
		b = append(b, make([]byte, (alignment-len(b)%alignment)%alignment)...)
		b = append(b, data...)
	}
	return b
}

func openFixture(t *testing.T, b []byte) *File {
	t.Helper()
	f, err := Open(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestMetadataTypes(t *testing.T) {
	cases := []struct {
		key string
		typ valueType
		one any
		two any
	}{
		{"u8", vU8, uint8(200), []uint8{0, 251}},
		{"i8", vI8, int8(-100), []int8{-2, 4}},
		{"u16", vU16, uint16(50000), []uint16{50001, 60000}},
		{"i16", vI16, int16(-1200), []int16{-32001, 192}},
		{"u32", vU32, uint32(1 << 31), []uint32{1 << 31, 16}},
		{"i32", vI32, int32(-1 << 30), []int32{-1, 1}},
		{"f32", vF32, float32(-0.25), []float32{1.5, 2.75}},
		{"bool", vBool, true, []bool{false, true}},
		{"u64", vU64, uint64(1 << 63), []uint64{1 << 63, 1024}},
		{"i64", vI64, int64(-1 << 62), []int64{-20, 10}},
		{"f64", vF64, float64(1.0 / 3), []float64{1.0 / 7, 3.125}},
	}
	var kvs []kv
	for _, tc := range cases {
		kvs = append(kvs, kv{"example." + tc.key, tc.typ, packed(tc.one)})
		kvs = append(kvs, kv{"example.arr_" + tc.key, vArray, arr(tc.typ, 2, packed(tc.two))})
	}
	kvs = append(kvs,
		kv{"general.architecture", vString, str("gemma-embedding2")},
		kv{"gemma-embedding2.attention.head_count_kv", vArray, arr(vU32, 2, packed([]uint32{2, 4}))},
		kv{"general.tags", vArray, arr(vString, 2, str(""), str("é"))},
		kv{"example.empty", vArray, arr(vString, 0)},
		kv{"example.nested", vArray, arr(vArray, 2, arr(vU32, 1, packed(uint32(3))), arr(vString, 1, str("x")))},
		kv{"tokenizer.ggml.model", vString, str("gpt2")},
		kv{"tokenizer.ggml.bos_token_id", vU32, packed(uint32(2))},
		kv{"tokenizer.ggml.eos_token_id", vU32, packed(uint32(1))},
		kv{"tokenizer.ggml.tokens", vArray, arr(vString, 2, str("a"), str("b"))},
	)
	f := openFixture(t, fixture(kvs, nil, 32, nil))
	for _, tc := range cases {
		for key, want := range map[string]any{"example." + tc.key: tc.one, "example.arr_" + tc.key: tc.two} {
			if got := f.Meta(key); !reflect.DeepEqual(got, want) {
				t.Errorf("%s = %#v (%T), want %#v (%T)", key, got, got, want, want)
			}
		}
	}
	for key, want := range map[string]any{
		"general.architecture":                     "gemma-embedding2",
		"gemma-embedding2.attention.head_count_kv": []uint32{2, 4},
		"general.tags":                             []string{"", "é"},
		"example.empty":                            []string{},
		"example.nested":                           []any{[]uint32{3}, []string{"x"}},
		"tokenizer.ggml.model":                     "gpt2",
		"tokenizer.ggml.bos_token_id":              uint32(2),
		"tokenizer.ggml.eos_token_id":              uint32(1),
		"tokenizer.ggml.tokens":                    nil,
		"absent":                                   nil,
	} {
		if got := f.Meta(key); !reflect.DeepEqual(got, want) {
			t.Errorf("%s = %#v (%T), want %#v (%T)", key, got, got, want, want)
		}
	}
	if f.TensorCount() != 0 {
		t.Fatal("metadata-only GGUF has tensors")
	}
}

func TestTensorDecoders(t *testing.T) {
	q8 := make([]byte, 34)
	binary.LittleEndian.PutUint16(q8, 0xc000) // -2
	copy(q8[2:], []byte{128, 255, 1, 127})
	// Directory enumeration is independent of physical tensor order.
	tensors := []entry{
		{"q6", []uint64{256, 1}, Q6_K, 128},
		{"f32", []uint64{2, 2}, F32, 0},
		{"bf16", []uint64{4}, BF16, 32},
		{"q8", []uint64{32}, Q8_0, 64},
	}
	raw := make([]byte, 128+210)
	copy(raw[0:], packed([]uint32{0x3fc00000, 0xc0100000, 0x80000000, 0x7fc00123}))
	copy(raw[32:], packed([]uint16{0x3f80, 0xc040, 0x0001, 0x7f80}))
	copy(raw[64:], q8)
	q6, wantQ6 := q6Example()
	copy(raw[128:], q6)
	f := openFixture(t, fixture(nil, tensors, 32, raw))
	if f.TensorCount() != len(tensors) {
		t.Fatalf("TensorCount = %d", f.TensorCount())
	}
	if _, ok := f.Tensor("absent"); ok {
		t.Fatal("absent tensor reported as present")
	}

	for _, e := range tensors {
		handle, ok := f.Tensor(e.name)
		if !ok || handle.Type != e.typ || !slices.Equal(handle.Shape, e.shape) {
			t.Fatalf("tensor %q = %+v, expected %v %v", e.name, handle, e.shape, e.typ)
		}
		out, err := handle.ReadFloat32()
		if err != nil {
			t.Fatal(err)
		}
		switch e.typ {
		case F32:
			equalBits(t, out, []uint32{0x3fc00000, 0xc0100000, 0x80000000, 0x7fc00123})
		case BF16:
			equalBits(t, out, []uint32{0x3f800000, 0xc0400000, 0x00010000, 0x7f800000})
		case Q8_0:
			if !slices.Equal(out[:4], []float32{256, 2, -2, -254}) || math.Float32bits(out[31]) != 0x80000000 {
				t.Fatalf("Q8_0 output %v", out)
			}
		case Q6_K:
			if !slices.Equal(out, wantQ6) {
				for i, v := range out {
					if v != wantQ6[i] {
						t.Fatalf("Q6_K lane %d = %v, want %v", i, v, wantQ6[i])
					}
				}
			}
		}
		out[0] = 9999
		handle.Shape[0] = 9999
		handle.Type = 9999
		reloaded, err := handle.ReadFloat32()
		if err != nil || reloaded[0] == out[0] {
			t.Fatalf("tensor %q ReadFloat32 shares mutable state: %v, %v", e.name, reloaded, err)
		}
		again, _ := f.Tensor(e.name)
		if again.Type != e.typ || !slices.Equal(again.Shape, e.shape) {
			t.Fatalf("tensor %q directory shares mutable state", e.name)
		}
	}
}

func equalBits(t *testing.T, got []float32, want []uint32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("decoded %d values, want %d", len(got), len(want))
	}
	for i, x := range got {
		if bits := math.Float32bits(x); bits != want[i] {
			t.Errorf("value %d bits %#08x, want %#08x", i, bits, want[i])
		}
	}
}

func q6Example() ([]byte, []float32) {
	raw := make([]byte, 210)
	for lane := range 32 {
		raw[lane], raw[32+lane] = 0x31, 0x42
		raw[64+lane], raw[96+lane] = 0x75, 0x86
		raw[128+lane], raw[160+lane] = 0xe4, 0x1b
	}
	raw[3], raw[131] = 0x75, 0xb1
	scales := []int8{1, -2, 3, -4, 5, -6, 7, -8, -9, 10, -11, 12, -13, 14, -15, 16}
	for i, scale := range scales {
		raw[192+i] = byte(scale)
	}
	binary.LittleEndian.PutUint16(raw[208:], 0x3800) // 0.5
	groups := []float32{-31, -14, 3, 20, 21, 6, -9, -24}
	changed := []float32{-11, -30, 23, 4}
	want := make([]float32, 256)
	for i := range want {
		q := groups[i/32]
		if i < 128 && i%32 == 3 {
			q = changed[i/32]
		}
		want[i] = 0.5 * float32(scales[i/16]) * q
	}
	return raw, want
}

type trackingRead struct {
	r     *bytes.Reader
	calls [][2]int64
}

func (r *trackingRead) ReadAt(p []byte, off int64) (int, error) {
	r.calls = append(r.calls, [2]int64{off, int64(len(p))})
	return r.r.ReadAt(p, off)
}

func TestBlockAlignedReads(t *testing.T) {
	const blocks = 320
	block, expected := q6Example()
	raw := bytes.Repeat(block, blocks)
	b := fixture(nil, []entry{{"large", []uint64{256, blocks}, Q6_K, 0}}, 32, raw)
	r := &trackingRead{r: bytes.NewReader(b)}
	f, err := Open(r, int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	handle, _ := f.Tensor("large")
	r.calls = nil
	out, err := handle.ReadFloat32()
	if err != nil {
		t.Fatal(err)
	}
	if len(r.calls) < 2 || len(out) != 256*blocks {
		t.Fatalf("expected chunked reads and complete decode, calls %v, values %d", r.calls, len(out))
	}
	start := int64(len(b) - len(raw))
	for _, call := range r.calls {
		if call[0] != start || call[1] > 64<<10 || call[1]%210 != 0 {
			t.Fatalf("misaligned or excessive tensor read %v", call)
		}
		start += call[1]
	}
	for i := 0; i < len(out); i += 256 {
		if !slices.Equal(out[i:i+256], expected) {
			t.Fatalf("incorrect result for block %d", i/256)
		}
	}
	r.r = bytes.NewReader(b[:len(b)-1])
	if got, err := handle.ReadFloat32(); got != nil || !strings.Contains(errorText(err), io.ErrUnexpectedEOF.Error()) {
		t.Fatalf("short tensor read returned %d values, error %v", len(got), err)
	}
}

func TestPackedRowsOwnEncodedBlocks(t *testing.T) {
	block, want := q6Example()
	second := slices.Clone(block)
	binary.LittleEndian.PutUint16(second[208:], 0x3c00)
	raw := bytes.Join([][]byte{block, second, second, block}, nil)
	b := fixture(nil, []entry{{"matrix", []uint64{512, 2}, Q6_K, 0}}, 32, raw)
	source := bytes.NewReader(b)
	f, err := Open(source, int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	matrix, _ := f.Tensor("matrix")
	rows, err := matrix.ReadPackedRows()
	if err != nil {
		t.Fatal(err)
	}
	dense, err := matrix.ReadFloat32()
	if err != nil {
		t.Fatal(err)
	}
	clear(b)
	source.Reset(nil)
	got := make([]float32, 512)
	for _, i := range []int{1, 0, 1} {
		rows.DecodeRow(i, got)
		if !slices.Equal(got, dense[i*512:(i+1)*512]) {
			t.Fatalf("row %d differs from dense decode after the reader is cleared", i)
		}
	}
	if !slices.Equal(got[256:], want) || got[0] != 2*want[0] {
		t.Fatal("packed rows did not preserve both Q6_K blocks in order")
	}
	if rows, err := matrix.ReadPackedRows(); rows != nil || err == nil {
		t.Fatalf("truncated packed read returned %p, %v", rows, err)
	}
	vector := openFixture(t, fixture(nil, []entry{{"vector", []uint64{256}, Q6_K, 0}}, 32, block))
	handle, _ := vector.Tensor("vector")
	if rows, err := handle.ReadPackedRows(); rows != nil || err == nil {
		t.Fatalf("rank-one packed read returned %p, %v", rows, err)
	}
}

func TestRejectsMalformedGGUF(t *testing.T) {
	e := entry{"test", []uint64{16}, F32, 0}
	other := func(name string, shape []uint64, typ Type, off uint64) []byte {
		return fixture(nil, []entry{{name, shape, typ, off}}, 32, make([]byte, 256))
	}
	metadata := func(typ valueType, payload []byte) []byte {
		return fixture([]kv{{"general.a", typ, payload}}, nil, 32, nil)
	}
	valid := fixture(nil, []entry{e}, 32, make([]byte, 64))
	patch := func(off int, payload []byte) []byte {
		b := slices.Clone(valid)
		copy(b[off:], payload)
		return b
	}
	deep := arr(vI8, 0)
	for range 9 {
		deep = arr(vArray, 1, deep)
	}
	cases := []struct {
		name string
		raw  []byte
		want string
	}{
		{"magic", patch(0, []byte("FUGG")), "magic"},
		{"version", patch(4, packed(uint32(2))), "version"},
		{"big-endian version", patch(4, []byte{0, 0, 0, 3}), "version"},
		{"count overflow", patch(8, packed(^uint64(0))), "counts"},
		{"overclaimed count", patch(8, packed(uint64(30))), "counts"},
		{"wrong alignment type", fixture([]kv{{"general.alignment", vU64, packed(uint64(32))}}, nil, 32, nil), "alignment"},
		{"zero alignment", fixture([]kv{{"general.alignment", vU32, packed(uint32(0))}}, nil, 32, nil), "alignment"},
		{"unaligned alignment", fixture([]kv{{"general.alignment", vU32, packed(uint32(3))}}, nil, 32, nil), "alignment"},
		{"duplicate key", fixture([]kv{{"tokenizer.tokens", vArray, arr(vString, 0)}, {"tokenizer.tokens", vString, str("x")}}, nil, 32, nil), "duplicate metadata"},
		{"invalid key", fixture([]kv{{"broken..key", vBool, []byte{1}}}, nil, 32, nil), "metadata key"},
		{"missing value", metadata(vU64, nil), "truncated"},
		{"unknown value type", metadata(99, []byte{1}), "metadata type"},
		{"malformed bool", metadata(vBool, []byte{2}), "bool"},
		{"invalid UTF-8", metadata(vString, str("\xff")), "UTF-8"},
		{"unexpected string length", metadata(vString, packed(^uint64(0))), "string length"},
		{"unknown array type", metadata(vArray, arr(100, 0)), "array element type"},
		{"huge array count", metadata(vArray, arr(vU64, ^uint64(0))), "array length"},
		{"short array", metadata(vArray, arr(vU64, 20, packed(uint64(7)))), "array length"},
		{"deep array", metadata(vArray, deep), "nesting"},
		{"skipped array overrun", fixture([]kv{{"tokenizer.tokens", vArray, arr(vString, 1, packed(uint64(50)))}}, nil, 32, nil), "truncated"},
		{"duplicate tensor", fixture(nil, []entry{e, e}, 32, make([]byte, 128)), "duplicate tensor"},
		{"overlap", fixture(nil, []entry{e, {"other", []uint64{16}, F32, 32}}, 32, make([]byte, 128)), "overlapping"},
		{"unsupported tensor type", other("test", []uint64{1}, 1, 0), "unsupported tensor type"},
		{"misaligned tensor", other("test", []uint64{4}, F32, 4), "misaligned"},
		{"offset overflow", other("test", []uint64{16}, F32, ^uint64(31)), "file size"},
		{"zero dimension", other("test", []uint64{1, 0}, F32, 0), "dimensions"},
		{"dimension overflow", other("test", []uint64{2, 1 << 63}, F32, 0), "dimensions"},
		{"allocation limit", other("test", []uint64{maxTensorValues + 1}, BF16, 0), "oversized"},
		{"shape rank", other("test", []uint64{1, 1, 1, 1, 1}, F32, 0), "rank"},
		{"empty rank", other("test", nil, F32, 0), "rank"},
		{"long tensor name", other(strings.Repeat("x", 65), []uint64{1}, F32, 0), "string length"},
		{"Q8 bad row width", other("test", []uint64{16, 2}, Q8_0, 0), "row width"},
		{"Q6 bad row width", other("test", []uint64{128, 2}, Q6_K, 0), "row width"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f, err := Open(bytes.NewReader(tc.raw), int64(len(tc.raw)))
			if f != nil || !strings.Contains(errorText(err), tc.want) {
				t.Fatalf("malformed fixture: file=%v err=%v, want error containing %q", f, err, tc.want)
			}
		})
	}
	for n := range len(valid) {
		if f, err := Open(bytes.NewReader(valid[:n]), int64(n)); f != nil || err == nil {
			t.Fatalf("accepted truncated input of %d bytes: %v, %v", n, f, err)
		}
	}
}

func TestCustomAlignment(t *testing.T) {
	b := fixture(
		[]kv{{"general.alignment", vU32, packed(uint32(24))}},
		[]entry{{"first", []uint64{1}, F32, 0}, {"second", []uint64{1}, BF16, 24}},
		24, append(make([]byte, 24), packed(uint16(0x4040))...),
	)
	f := openFixture(t, b)
	handle, _ := f.Tensor("second")
	got, err := handle.ReadFloat32()
	if err != nil || !slices.Equal(got, []float32{3}) {
		t.Fatalf("read with custom alignment: %v, %v", got, err)
	}
}

func errorText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
