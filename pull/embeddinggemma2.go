package pull

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

const (
	embeddingGemma2GoogleRepo     = "google/embeddinggemma-2"
	embeddingGemma2GoogleRevision = "f9972ac22e696a105c482202646979e33e4b5a1e"
	embeddingGemma2UnslothRepo    = "unsloth/embeddinggemma-2-GGUF"
	embeddingGemma2UnslothRev     = "031f0d4b35536f69ab3509d4893c923264fcf253"
	embeddingGemma2MLXRepo        = "mlx-community/embeddinggemma-2-8bit"
	embeddingGemma2MLXRev         = "7505ef2f8ddef45efef6d060865f27989b3c9cec"
)

var embeddingGemma2GoogleFiles = []File{
	{Path: "config.json", Size: 4455, SHA256: "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3"},
	{Path: "config_sentence_transformers.json", Size: 1565, SHA256: "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5"},
	{Path: "tokenizer.json", Size: 32170510, SHA256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"},
	{Path: "tokenizer_config.json", Size: 1599, SHA256: "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"},
	{Path: "model.safetensors", Size: 1488915288, SHA256: "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"},
}

var embeddingGemma2MLXFiles = []File{
	{Path: "config.json", Size: 5463, SHA256: "35e55812554bf172f9321f5c6a25d58810432db9eeeb38094723a04e8d3dd9af"},
	{Path: "config_sentence_transformers.json", Size: 1565, SHA256: "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5"},
	{Path: "tokenizer.json", Size: 32170510, SHA256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"},
	{Path: "model.safetensors", Size: 1234238431, SHA256: "6b7e97f9687ad422ab3625a1cdf6002892187d7dd04d01b2a09253fa1e5d4cfd"},
}

// EmbeddingGemma2Model describes a pinned download and its qualified text runtimes.
type EmbeddingGemma2Model struct {
	Variant  string // canonical selector for DownloadEmbeddingGemma2
	Repo     string // Hugging Face owner/repo
	Revision string
	Format   string // "safetensors" (directory) or "gguf" (file)

	// QualifiedBackends names text backends exercised for this checkpoint on representative hardware.
	// An empty list means the file is download-only for the EmbeddingGemma 2 loader.
	QualifiedBackends []string
	// SuggestedAccelerator is an optional selector for Encoder.UseAccelerator.
	// A caller's explicit choice takes priority.
	SuggestedAccelerator string
}

type embeddingGemma2Bundle struct {
	EmbeddingGemma2Model
	files    []File
	filename string // empty when the result is a directory
}

func embeddingGemma2GGUF(variant string, file File, qualified ...string) embeddingGemma2Bundle {
	return embeddingGemma2Bundle{
		EmbeddingGemma2Model: EmbeddingGemma2Model{
			Variant: variant, Repo: embeddingGemma2UnslothRepo, Revision: embeddingGemma2UnslothRev,
			Format: "gguf", QualifiedBackends: qualified,
		},
		files: []File{file}, filename: file.Path,
	}
}

var embeddingGemma2Bundles = []embeddingGemma2Bundle{
	{
		EmbeddingGemma2Model: EmbeddingGemma2Model{
			Variant: "google", Repo: embeddingGemma2GoogleRepo, Revision: embeddingGemma2GoogleRevision,
			Format: "safetensors", QualifiedBackends: []string{"cpu", "cuda", "metal", "vulkan"},
		},
		files: embeddingGemma2GoogleFiles,
	},
	{
		EmbeddingGemma2Model: EmbeddingGemma2Model{
			Variant: "mlx-8bit", Repo: embeddingGemma2MLXRepo, Revision: embeddingGemma2MLXRev,
			Format: "safetensors", QualifiedBackends: []string{"cpu", "metal"}, SuggestedAccelerator: "metal",
		},
		files: embeddingGemma2MLXFiles,
	},
	embeddingGemma2GGUF("bf16", File{Path: "embeddinggemma-2-BF16.gguf", Size: 557950240, SHA256: "f315cbbb30dd487e44d501c8902abe88808755e43753a96beed1964f0a48aa4f"}),
	embeddingGemma2GGUF("f16", File{Path: "embeddinggemma-2-F16.gguf", Size: 557950240, SHA256: "0d4350867eddbb94af485501a3d07e9374c47779d88f8a39cd1abd6a8e2d0901"}),
	embeddingGemma2GGUF("q8_0", File{Path: "embeddinggemma-2-Q8_0.gguf", Size: 309855520, SHA256: "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"}, "cpu", "cuda", "metal", "vulkan"),
	embeddingGemma2GGUF("ud-q6_k_xl", File{Path: "embeddinggemma-2-UD-Q6_K_XL.gguf", Size: 248812032, SHA256: "dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3"}, "cpu", "cuda", "metal", "vulkan"),
	embeddingGemma2GGUF("ud-q5_k_xl", File{Path: "embeddinggemma-2-UD-Q5_K_XL.gguf", Size: 210055680, SHA256: "a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41"}),
	embeddingGemma2GGUF("ud-q4_k_xl", File{Path: "embeddinggemma-2-UD-Q4_K_XL.gguf", Size: 175673856, SHA256: "ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493"}),
}

// EmbeddingGemma2Models lists canonical download variants in stable catalog order.
// Each call returns an independent copy and uses no network.
func EmbeddingGemma2Models() []EmbeddingGemma2Model {
	models := make([]EmbeddingGemma2Model, 0, len(embeddingGemma2Bundles))
	for _, bundle := range embeddingGemma2Bundles {
		model := bundle.EmbeddingGemma2Model
		model.QualifiedBackends = slices.Clone(model.QualifiedBackends)
		models = append(models, model)
	}
	return models
}

// DownloadEmbeddingGemma2 fetches a pinned model into the user model cache.
// "google" and "mlx-8bit" (alias "mlx8") return safetensors directories. Unsloth
// variants return their GGUF file paths. See EmbeddingGemma2Models for the list.
// Variant names are case-insensitive. A verified cache hit works offline.
// Progress is optional and reports bytes across the bundle.
func DownloadEmbeddingGemma2(ctx context.Context, variant string, progress func(done, total int64)) (string, error) {
	variant = strings.ToLower(strings.TrimSpace(variant))
	if variant == "mlx8" {
		variant = "mlx-8bit"
	}
	for _, bundle := range embeddingGemma2Bundles {
		if bundle.Variant != variant {
			continue
		}
		cache, err := CacheDir(bundle.Repo)
		if err != nil {
			return "", err
		}
		plan := Plan{Repo: bundle.Repo, Files: bundle.files, Bytes: SetBytes(bundle.files)}
		dest := filepath.Join(cache, bundle.Revision, bundle.Variant)
		dir, err := downloadPinnedBundle(ctx, plan, bundle.Revision, dest, progress)
		if err != nil {
			return "", err
		}
		return filepath.Join(dir, bundle.filename), nil
	}
	names := make([]string, 0, len(embeddingGemma2Bundles))
	for _, bundle := range embeddingGemma2Bundles {
		names = append(names, bundle.Variant)
	}
	return "", fmt.Errorf("unknown EmbeddingGemma 2 variant %q; use %s", variant, strings.Join(names, ", "))
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
