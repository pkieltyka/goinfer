package pull

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDownloadPinnedBundle(t *testing.T) {
	const revision = "0123456789abcdef"
	const contents = "the pinned bytes"
	plan := Plan{Repo: "owner/model", Files: []File{
		{Path: "model.gguf", Size: int64(len(contents)), SHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(contents)))},
	}, Bytes: int64(len(contents))}
	var requests atomic.Int64
	var corrupt atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if r.URL.Path != "/owner/model/resolve/"+revision+"/model.gguf" {
			t.Errorf("unexpected endpoint %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		if corrupt.Load() {
			_, _ = fmt.Fprint(w, strings.Repeat("x", len(contents)))
			return
		}
		_, _ = fmt.Fprint(w, contents)
	}))
	defer srv.Close()
	old := hfCDN
	hfCDN = srv.URL
	t.Cleanup(func() { hfCDN = old })

	root := t.TempDir()
	dest := filepath.Join(root, "bundle")
	corrupt.Store(true)
	if _, err := downloadPinnedBundle(t.Context(), plan, revision, dest, nil); err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("wrong bytes returned %v", err)
	}
	if entries, err := os.ReadDir(root); err != nil || len(entries) != 0 {
		t.Fatalf("failed attempt left cache entries %v, err %v", entries, err)
	}
	corrupt.Store(false)
	var callers sync.WaitGroup
	for range 2 {
		callers.Go(func() {
			got, err := downloadPinnedBundle(t.Context(), plan, revision, dest, nil)
			if err != nil || got != dest {
				t.Errorf("download returned %q, %v", got, err)
			}
		})
	}
	callers.Wait()

	got, err := os.ReadFile(filepath.Join(dest, "model.gguf"))
	if err != nil || string(got) != contents {
		t.Fatalf("published bytes = %q, %v", got, err)
	}
	before := requests.Load()
	if before < 2 || before > 3 {
		t.Fatalf("got %d requests before cache hit", before)
	}
	srv.Close()
	if cached, err := downloadPinnedBundle(t.Context(), plan, revision, dest, nil); err != nil || cached != dest {
		t.Fatalf("offline cache returned %q, %v", cached, err)
	}
	if requests.Load() != before {
		t.Fatal("cache hit used the network")
	}
}

func TestDownloadEmbeddingGemma2RejectsUnknownVariant(t *testing.T) {
	if got, err := DownloadEmbeddingGemma2(t.Context(), "../q8_0", nil); err == nil || got != "" {
		t.Fatalf("unknown variant returned %q, %v", got, err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if got, err := DownloadEmbeddingGemma2(ctx, " Q8_0 ", nil); err != context.Canceled || got != "" {
		t.Fatalf("canceled download returned %q, %v", got, err)
	}
}

func TestEmbeddingGemma2Pins(t *testing.T) {
	verify := func(f File) {
		if digest, err := hex.DecodeString(f.SHA256); err != nil || len(digest) != sha256.Size || f.Size <= 0 {
			t.Errorf("invalid pin for %s", f.Path)
		}
	}
	for _, bundle := range embeddingGemma2Bundles {
		for _, f := range bundle.files {
			verify(f)
		}
	}
}

func TestEmbeddingGemma2Models(t *testing.T) {
	got := EmbeddingGemma2Models()
	want := []string{"google", "mlx-8bit", "bf16", "f16", "q8_0", "ud-q6_k_xl", "ud-q5_k_xl", "ud-q4_k_xl"}
	if len(got) != len(want) {
		t.Fatalf("catalog has %d variants, want %d", len(got), len(want))
	}
	for i, model := range got {
		if model.Variant != want[i] {
			t.Errorf("catalog row %d is %s, want %s", i, model.Variant, want[i])
		}
		switch model.Variant {
		case "google", "mlx-8bit":
			if model.Format != "safetensors" || len(model.QualifiedBackends) == 0 {
				t.Errorf("checkpoint row: %+v", model)
			}
		case "q8_0", "ud-q6_k_xl":
			if model.Format != "gguf" || !slices.Contains(model.QualifiedBackends, "cpu") {
				t.Errorf("qualified GGUF row: %+v", model)
			}
		default:
			if model.Format != "gguf" || len(model.QualifiedBackends) != 0 || model.SuggestedAccelerator != "" {
				t.Errorf("download-only GGUF row: %+v", model)
			}
		}
	}
	mlx := got[1]
	if mlx.Repo != "mlx-community/embeddinggemma-2-8bit" || mlx.Revision != "7505ef2f8ddef45efef6d060865f27989b3c9cec" ||
		mlx.SuggestedAccelerator != "metal" || !slices.Equal(mlx.QualifiedBackends, []string{"cpu", "metal"}) {
		t.Fatalf("MLX catalog row: %+v", mlx)
	}
	got[1].Variant = "changed"
	got[1].QualifiedBackends[0] = "changed"
	if again := EmbeddingGemma2Models()[1]; again.Variant != "mlx-8bit" || again.QualifiedBackends[0] != "cpu" {
		t.Fatalf("caller mutated the catalog: %+v", again)
	}
}

func TestDownloadEmbeddingGemma2MLX(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	t.Setenv("XDG_CACHE_HOME", root)
	t.Setenv("LocalAppData", root)
	models, err := CacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	// Replace only the payloads. The public call still chooses the production
	// variant, revision, endpoint paths, and destination.
	at := slices.IndexFunc(embeddingGemma2Bundles, func(b embeddingGemma2Bundle) bool { return b.Variant == "mlx-8bit" })
	if at < 0 {
		t.Fatal("MLX bundle missing")
	}
	oldFiles := embeddingGemma2Bundles[at].files
	files := slices.Clone(oldFiles)
	for i := range files {
		contents := "pinned fixture " + files[i].Path
		files[i].Size = int64(len(contents))
		files[i].SHA256 = fmt.Sprintf("%x", sha256.Sum256([]byte(contents)))
	}
	embeddingGemma2Bundles[at].files = files
	t.Cleanup(func() { embeddingGemma2Bundles[at].files = oldFiles })

	const repo = "mlx-community/embeddinggemma-2-8bit"
	const rev = "7505ef2f8ddef45efef6d060865f27989b3c9cec"
	dest := filepath.Join(models, filepath.FromSlash(repo), rev, "mlx-8bit")
	base := "/" + repo + "/resolve/" + rev + "/"
	var requests atomic.Int64
	var corrupt atomic.Bool
	corrupt.Store(true)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		name, ok := strings.CutPrefix(r.URL.Path, base)
		if !ok || !slices.ContainsFunc(files, func(f File) bool { return f.Path == name }) {
			t.Errorf("unexpected download %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		if _, err := os.Stat(dest); !os.IsNotExist(err) {
			t.Errorf("bundle visible during transfer: %v", err)
		}
		contents := "pinned fixture " + name
		if name == "model.safetensors" && corrupt.Load() {
			contents = strings.Repeat("x", len(contents))
		}
		_, _ = fmt.Fprint(w, contents)
	}))
	defer srv.Close()
	oldCDN := hfCDN
	hfCDN = srv.URL
	t.Cleanup(func() { hfCDN = oldCDN })

	if got, err := DownloadEmbeddingGemma2(t.Context(), "mlx-8bit", nil); got != "" || err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("corrupt bundle returned %q, %v", got, err)
	}
	if entries, err := os.ReadDir(filepath.Dir(dest)); err != nil || len(entries) != 0 {
		t.Fatalf("failed transfer left cache entries %v, err %v", entries, err)
	}

	corrupt.Store(false)
	var done, total int64
	got, err := DownloadEmbeddingGemma2(t.Context(), " MLX-8BIT ", func(d, n int64) { done, total = d, n })
	if err != nil || got != dest {
		t.Fatalf("download returned %q, %v, want directory %q", got, err, dest)
	}
	if total != SetBytes(files) || done != total {
		t.Fatalf("progress reported %d of %d, want %d of %d", done, total, SetBytes(files), SetBytes(files))
	}
	if entries, err := os.ReadDir(filepath.Dir(dest)); err != nil || len(entries) != 1 || entries[0].Name() != "mlx-8bit" {
		t.Fatalf("published cache entries %v, err %v", entries, err)
	}
	for _, f := range files {
		path := filepath.Join(got, f.Path)
		if sum, err := fileSHA256(path); err != nil || sum != f.SHA256 {
			t.Fatalf("published %s = %s, %v", f.Path, sum, err)
		}
	}

	before := requests.Load()
	srv.Close()
	if cached, err := DownloadEmbeddingGemma2(t.Context(), "mlx8", nil); err != nil || cached != dest {
		t.Fatalf("offline alias returned %q, %v", cached, err)
	}
	modelPath := filepath.Join(dest, "model.safetensors")
	bad := strings.Repeat("x", len("pinned fixture model.safetensors"))
	if err := os.WriteFile(modelPath, []byte(bad), 0o600); err != nil {
		t.Fatal(err)
	}
	past := time.Unix(1, 0)
	if err := os.Chtimes(modelPath, past, past); err != nil {
		t.Fatal(err)
	}
	if cached, err := DownloadEmbeddingGemma2(t.Context(), "mlx8", nil); cached != "" || err == nil || !strings.Contains(err.Error(), "invalid model bundle") {
		t.Fatalf("changed cached file returned %q, %v", cached, err)
	}
	if requests.Load() != before {
		t.Fatalf("offline cache attempted %d additional requests", requests.Load()-before)
	}
}
