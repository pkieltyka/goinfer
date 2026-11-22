package embeddinggemma2

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pkieltyka/goinfer/internal/gguf"
)

func TestLoadGGUFEncoderRejectsOtherArchitectures(t *testing.T) {
	const other = "gemma-embedding"
	var header bytes.Buffer
	header.WriteString("GGUF")
	for _, value := range []any{uint32(3), uint64(1), uint64(1), uint64(len("general.architecture"))} {
		if err := binary.Write(&header, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	header.WriteString("general.architecture")
	for _, value := range []any{uint32(8), uint64(len(other))} {
		if err := binary.Write(&header, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	header.WriteString(other)
	if err := binary.Write(&header, binary.LittleEndian, uint64(len("weight"))); err != nil {
		t.Fatal(err)
	}
	header.WriteString("weight")
	for _, value := range []any{uint32(1), uint64(1), uint32(0), uint64(0)} {
		if err := binary.Write(&header, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	for header.Len()%32 != 0 {
		header.WriteByte(0)
	}
	header.Write([]byte{0, 0, 0, 0})
	path := filepath.Join(t.TempDir(), "other.gguf")
	if err := os.WriteFile(path, header.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	encoder, err := LoadGGUFEncoder(path, nil)
	if encoder != nil || err == nil || !strings.Contains(err.Error(), "general.architecture") {
		t.Fatalf("got encoder %v, error %v, want wrong-architecture error", encoder, err)
	}
}

func TestGGUFTensorRejectsWrongRoleType(t *testing.T) {
	const name = "blk.0.inp_gate.weight"
	tensor := gguf.Tensor{Shape: []uint64{512, 512}, Type: gguf.Q6_K}
	if err := validateGGUFTensor(tensor, name, []int{512, 512}, gguf.Q6_K); err == nil || !strings.Contains(err.Error(), "GGML type") {
		t.Fatalf("got %v, want mixed profile rejecting Q6_K for the Q8_0 gate", err)
	}
	tensor.Type = gguf.Q8_0
	if err := validateGGUFTensor(tensor, name, []int{512, 512}, gguf.Q6_K); err != nil {
		t.Fatal(err)
	}
}

func TestGGUFQuantRejectsUnknownCodecVersion(t *testing.T) {
	const key = "general.quantization_version"
	raw := []byte("GGUF")
	raw = binary.LittleEndian.AppendUint32(raw, 3)
	raw = binary.LittleEndian.AppendUint64(raw, 0)
	raw = binary.LittleEndian.AppendUint64(raw, 1)
	raw = binary.LittleEndian.AppendUint64(raw, uint64(len(key)))
	raw = append(raw, key...)
	raw = binary.LittleEndian.AppendUint32(raw, 4)
	raw = binary.LittleEndian.AppendUint32(raw, 3)
	file, err := gguf.Open(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = ggufQuant(file); err == nil || !strings.Contains(err.Error(), "quantization version 3") {
		t.Fatalf("got %v, want unknown quantization version", err)
	}
}
