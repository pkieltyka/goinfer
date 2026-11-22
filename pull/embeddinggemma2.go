package pull

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	embeddingGemma2GoogleRepo     = "google/embeddinggemma-2"
	embeddingGemma2GoogleRevision = "f9972ac22e696a105c482202646979e33e4b5a1e"
	embeddingGemma2UnslothRepo    = "unsloth/embeddinggemma-2-GGUF"
	embeddingGemma2UnslothRev     = "031f0d4b35536f69ab3509d4893c923264fcf253"
)

var embeddingGemma2GoogleFiles = []File{
	{Path: "config.json", Size: 4455, SHA256: "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3"},
	{Path: "config_sentence_transformers.json", Size: 1565, SHA256: "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5"},
	{Path: "tokenizer.json", Size: 32170510, SHA256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"},
	{Path: "tokenizer_config.json", Size: 1599, SHA256: "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"},
	{Path: "model.safetensors", Size: 1488915288, SHA256: "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"},
}

var embeddingGemma2UnslothFiles = map[string]File{
	"bf16":       {Path: "embeddinggemma-2-BF16.gguf", Size: 557950240, SHA256: "f315cbbb30dd487e44d501c8902abe88808755e43753a96beed1964f0a48aa4f"},
	"f16":        {Path: "embeddinggemma-2-F16.gguf", Size: 557950240, SHA256: "0d4350867eddbb94af485501a3d07e9374c47779d88f8a39cd1abd6a8e2d0901"},
	"q8_0":       {Path: "embeddinggemma-2-Q8_0.gguf", Size: 309855520, SHA256: "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"},
	"ud-q6_k_xl": {Path: "embeddinggemma-2-UD-Q6_K_XL.gguf", Size: 248812032, SHA256: "dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3"},
	"ud-q5_k_xl": {Path: "embeddinggemma-2-UD-Q5_K_XL.gguf", Size: 210055680, SHA256: "a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41"},
	"ud-q4_k_xl": {Path: "embeddinggemma-2-UD-Q4_K_XL.gguf", Size: 175673856, SHA256: "ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493"},
}

// DownloadEmbeddingGemma2 fetches a pinned public model from Hugging Face into
// the user model cache. "google" returns a safetensors checkpoint directory.
// BF16, F16, Q8_0, UD-Q6_K_XL, UD-Q5_K_XL and UD-Q4_K_XL return the respective
// Unsloth text GGUF path. Variant names are case-insensitive. A verified cache
// hit works offline. Progress is optional and reports bytes across the bundle.
func DownloadEmbeddingGemma2(ctx context.Context, variant string, progress func(done, total int64)) (string, error) {
	variant = strings.ToLower(strings.TrimSpace(variant))
	repo, revision, files, filename := embeddingGemma2GoogleRepo, embeddingGemma2GoogleRevision, embeddingGemma2GoogleFiles, ""
	if variant != "google" {
		file, ok := embeddingGemma2UnslothFiles[variant]
		if !ok {
			return "", fmt.Errorf("unknown EmbeddingGemma 2 variant %q; use google, BF16, F16, Q8_0, UD-Q6_K_XL, UD-Q5_K_XL or UD-Q4_K_XL", variant)
		}
		repo, revision, files, filename = embeddingGemma2UnslothRepo, embeddingGemma2UnslothRev, []File{file}, file.Path
	}
	cache, err := CacheDir(repo)
	if err != nil {
		return "", err
	}
	plan := Plan{Repo: repo, Files: files, Bytes: SetBytes(files)}
	dest := filepath.Join(cache, revision, variant)
	dir, err := downloadPinnedBundle(ctx, plan, revision, dest, progress)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, filename), nil
}

// A private staging directory lets separate callers fetch the same bundle
// without sharing partial files. Only a fully verified bundle is published.
func downloadPinnedBundle(ctx context.Context, plan Plan, revision, dest string, progress func(done, total int64)) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if checkpointComplete(dest, plan.Files) {
		return dest, nil
	}
	if _, err := os.Lstat(dest); err == nil {
		if checkpointComplete(dest, plan.Files) {
			return dest, nil
		}
		return "", fmt.Errorf("%s exists with an invalid model bundle; move it aside to download again", dest)
	} else if !os.IsNotExist(err) {
		return "", err
	}
	parent := filepath.Dir(dest)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return "", err
	}
	staging, err := os.MkdirTemp(parent, ".download-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(staging)
	var report func(done, total int64, file string)
	if progress != nil {
		report = func(done, total int64, _ string) { progress(done, total) }
	}
	bundle, err := downloadCheckpointAtRevision(ctx, plan, revision, filepath.Join(staging, "model"), report)
	if err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.Rename(bundle, dest); err != nil {
		if checkpointComplete(dest, plan.Files) {
			return dest, nil
		}
		return "", fmt.Errorf("publishing model at %s: %w", dest, err)
	}
	return dest, nil
}
