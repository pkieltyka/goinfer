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
	"strings"
	"sync"
	"sync/atomic"
	"testing"
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
	for _, f := range embeddingGemma2GoogleFiles {
		verify(f)
	}
	for _, f := range embeddingGemma2UnslothFiles {
		verify(f)
	}
}
