package embeddinggemma2

import (
	"bytes"
	"encoding/binary"
	"flag"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/pkieltyka/goinfer/internal/gguf"
	"github.com/pkieltyka/goinfer/tokenizer"
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

func TestLoadGGUFEncoderReaderAtRejectsMalformed(t *testing.T) {
	for _, data := range [][]byte{[]byte("GGUF"), make([]byte, 24)} {
		if e, err := LoadGGUFEncoderReaderAt(bytes.NewReader(data), int64(len(data)), nil); err == nil || e != nil {
			t.Fatalf("accepted malformed input of length %d: encoder %v, error %v", len(data), e, err)
		}
	}
}

var readerGGUFDir = flag.String("eg2-reader-gguf-dir", "", "directory containing the pinned EmbeddingGemma 2 Q8_0 and UD-Q6_K_XL GGUFs")

func TestLoadGGUFEncoderReaderAtReal(t *testing.T) {
	if *readerGGUFDir == "" {
		t.Skip("set -eg2-reader-gguf-dir to the pinned GGUF snapshots")
	}
	for _, file := range []string{"embeddinggemma-2-Q8_0.gguf", "embeddinggemma-2-UD-Q6_K_XL.gguf"} {
		t.Run(file, func(t *testing.T) {
			path := filepath.Join(*readerGGUFDir, file)
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			r := bytes.NewReader(raw)
			got, err := LoadGGUFEncoderReaderAt(r, int64(len(raw)), nil)
			if err != nil {
				t.Fatal(err)
			}
			if got.Tokenizer() != got.tok {
				t.Fatal("the accessor did not return the encoder's tokenizer")
			}
			ref, err := tokenizer.LoadGGUFBytes(raw)
			if err != nil {
				t.Fatal(err)
			}
			r.Reset(nil)
			clear(raw)
			want, err := LoadGGUFEncoder(path, nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, text := range []string{"Redis", "How long can results be before they get cut off?"} {
				a, err := got.Tokenizer().Encode(text, false)
				if err != nil {
					t.Fatal(err)
				}
				b, err := ref.Encode(text, false)
				if err != nil || !slices.Equal(a, b) {
					t.Fatalf("token IDs for %q: got %v, want %v (error %v)", text, a, b, err)
				}
				v, _, err := got.EmbedText(text, "")
				if err != nil {
					t.Fatal(err)
				}
				w, _, err := want.EmbedText(text, "")
				if err != nil || !slices.Equal(v, w) {
					t.Fatalf("reader and path embeddings differ for %q (error %v)", text, err)
				}
			}
		})
	}
}
