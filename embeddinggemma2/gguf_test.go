package embeddinggemma2

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"flag"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

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

type testReaderAt func([]byte, int64) (int, error)

func (r testReaderAt) ReadAt(p []byte, off int64) (int, error) { return r(p, off) }

func TestLoadWeightsDrainsFailedReaderAt(t *testing.T) {
	previous := runtime.GOMAXPROCS(2)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	refused := errors.New("failed source read")
	synctest.Test(t, func(t *testing.T) {
		started, finish := make(chan struct{}), make(chan struct{})
		result := make(chan error, 1)
		var active atomic.Int32
		r := testReaderAt(func(p []byte, off int64) (int, error) {
			active.Add(1)
			defer active.Add(-1)
			switch off {
			case 0:
				<-started
				return 0, refused
			case 1:
				close(started)
				<-finish
				p[0] = 42
				return 1, nil
			default:
				t.Error("scheduled another read after failure")
				return 0, refused
			}
		})
		go func() {
			result <- loadWeights(3, func(i int) error {
				var p [1]byte
				_, err := r.ReadAt(p[:], int64(i))
				return err
			})
		}()
		synctest.Wait()
		select {
		case err := <-result:
			t.Errorf("loader returned %v before its in-flight ReadAt finished", err)
			close(finish)
			return
		default:
		}
		close(finish)
		if err := <-result; !errors.Is(err, refused) {
			t.Fatalf("loader returned %v, want %v", err, refused)
		}
		if n := active.Load(); n != 0 {
			t.Fatalf("loader returned with %d active ReadAt calls", n)
		}
	})
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
			if got.m.embed != nil || got.m.embedGGUF == nil {
				t.Fatal("opening widened the token table")
			}
			if got.Tokenizer() != got.tok {
				t.Fatal("the accessor did not return the encoder's tokenizer")
			}
			index, err := gguf.Open(r, int64(len(raw)))
			if err != nil {
				t.Fatal(err)
			}
			table, ok := index.Tensor("token_embd.weight")
			if !ok {
				t.Fatal("missing token table")
			}
			dense, err := table.ReadFloat32()
			if err != nil {
				t.Fatal(err)
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
			want.m.embedGGUF, want.m.embed = nil, dense
			ids := []int{1, 2, 262143, 120000, 1}
			gotRows, err := got.m.EmbedTokens(ids)
			if err != nil {
				t.Fatal(err)
			}
			wantRows, err := want.m.EmbedTokens(ids)
			if err != nil {
				t.Fatal(err)
			}
			for i, x := range gotRows {
				if math.Float32bits(x) != math.Float32bits(wantRows[i]) {
					t.Fatalf("packed row component %d changed bits", i)
				}
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
				if err != nil || !slices.EqualFunc(v, w, func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }) {
					t.Fatalf("packed and dense embeddings differ for %q (error %v)", text, err)
				}
			}
		})
	}
}

// BenchmarkGGUFVerifiedStartup measures digest verification of resident bytes,
// model construction and one CPU query. The stage metrics can overlap.
// Run one leaf benchmark per process when comparing RSS.
func BenchmarkGGUFVerifiedStartup(b *testing.B) {
	if *readerGGUFDir == "" {
		b.Skip("set -eg2-reader-gguf-dir to the pinned GGUF snapshots")
	}
	for _, artifact := range []struct {
		quant, digest string
	}{
		{"Q8_0", "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"},
		{"UD-Q6_K_XL", "dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3"},
	} {
		b.Run(artifact.quant, func(b *testing.B) {
			path := filepath.Join(*readerGGUFDir, "embeddinggemma-2-"+artifact.quant+".gguf")
			raw, err := os.ReadFile(path)
			if err != nil {
				b.Fatal(err)
			}
			index, err := gguf.Open(bytes.NewReader(raw), int64(len(raw)))
			if err != nil {
				b.Fatal(err)
			}
			token, ok := index.Tensor("token_embd.weight")
			if !ok {
				b.Fatal("missing token table")
			}
			for _, mode := range []string{"dense-serial", "packed-serial", "packed-parallel"} {
				b.Run(mode, func(b *testing.B) {
					benchmarkGGUFStartup(b, raw, artifact.digest, token, mode)
				})
			}
		})
	}
}

func benchmarkGGUFStartup(b *testing.B, raw []byte, digest string, table gguf.Tensor, mode string) {
	var verifyTotal, loadTotal, queryTotal, weightsTotal, tokenTotal, othersTotal time.Duration
	var retained, rss, peakRSS uint64
	for b.Loop() {
		b.StopTimer()
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		b.StartTimer()
		start := time.Now()
		sha := sha256.Sum256(raw)
		if hex.EncodeToString(sha[:]) != digest {
			b.Fatal("snapshot digest mismatch")
		}
		verified := time.Now()
		var dense []float32
		e, err := loadGGUFEncoderReaderAt(bytes.NewReader(raw), int64(len(raw)), nil, func(n int, job func(int) error) error {
			start := time.Now()
			var token time.Duration
			var otherEnd atomic.Int64
			timeJob := func(i int) error {
				before := time.Now()
				var err error
				if i == 0 && mode == "dense-serial" {
					dense, err = table.ReadFloat32()
				} else {
					err = job(i)
				}
				if i == 0 {
					token = time.Since(before)
				} else {
					end := int64(time.Since(start))
					for previous := otherEnd.Load(); end > previous && !otherEnd.CompareAndSwap(previous, end); previous = otherEnd.Load() {
					}
				}
				return err
			}
			if mode == "packed-parallel" {
				if err := loadWeights(n, timeJob); err != nil {
					return err
				}
			} else {
				for i := 1; i < n; i++ {
					if err := timeJob(i); err != nil {
						return err
					}
				}
				if err := timeJob(0); err != nil {
					return err
				}
			}
			weightsTotal += time.Since(start)
			tokenTotal += token
			othersTotal += time.Duration(otherEnd.Load())
			return nil
		})
		loaded := time.Now()
		if err != nil {
			b.Fatal(err)
		}
		if dense != nil {
			e.m.embed = dense
		}
		if _, _, err := e.EmbedText("How long can results be before they get cut off?", ""); err != nil {
			b.Fatal(err)
		}
		verifyTotal += verified.Sub(start)
		loadTotal += loaded.Sub(verified)
		queryTotal += time.Since(loaded)
		b.StopTimer()
		runtime.GC()
		runtime.ReadMemStats(&after)
		current, high := ggufBenchmarkRSS()
		rss += current
		peakRSS = max(peakRSS, high)
		runtime.KeepAlive(e)
		if after.HeapAlloc > before.HeapAlloc {
			retained += after.HeapAlloc - before.HeapAlloc
		}
		b.StartTimer()
	}
	for _, row := range []struct {
		label    string
		duration time.Duration
	}{
		{"ms/verify", verifyTotal}, {"ms/load", loadTotal}, {"ms/first-query", queryTotal},
		{"ms/weights", weightsTotal}, {"ms/token-table", tokenTotal}, {"ms/other-wall", othersTotal},
	} {
		b.ReportMetric(float64(row.duration)/float64(time.Millisecond)/float64(b.N), row.label)
	}
	b.ReportMetric(float64(retained)/float64(b.N)/(1<<20), "MiB/retained-heap")
	if peakRSS != 0 {
		b.ReportMetric(float64(rss)/float64(b.N)/1024, "MiB/settled-RSS")
		b.ReportMetric(float64(peakRSS)/1024, "MiB/peak-RSS")
	}
}

func ggufBenchmarkRSS() (rss, peak uint64) {
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0, 0
	}
	for _, line := range strings.Split(string(status), "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok || key != "VmRSS" && key != "VmHWM" {
			continue
		}
		number, _, _ := strings.Cut(strings.TrimSpace(value), " ")
		kib, _ := strconv.ParseUint(number, 10, 64)
		if key == "VmRSS" {
			rss = kib
		} else {
			peak = kib
		}
	}
	return rss, peak
}
