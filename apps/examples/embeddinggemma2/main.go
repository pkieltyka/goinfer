// Command embeddinggemma2 ranks Markdown documents by embedding similarity.
//
//	go run ./apps/examples/embeddinggemma2 <checkpoint-dir> <query> <doc.md> [doc.md ...]
//	go run ./apps/examples/embeddinggemma2 --download <variant>
package main

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
	"github.com/pkieltyka/goinfer/pull"
)

type match struct {
	path  string
	score float64
}

func main() {
	var err error
	switch {
	case len(os.Args) == 3 && os.Args[1] == "--download":
		err = download(os.Args[2])
	case len(os.Args) >= 4:
		err = rank(os.Args[1], os.Args[2], os.Args[3:])
	default:
		fmt.Fprintln(os.Stderr, "usage: embeddinggemma2 --download <variant> or embeddinggemma2 <checkpoint-dir> <query> <doc.md> [doc.md ...]")
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func download(variant string) error {
	path, err := pull.DownloadEmbeddingGemma2(context.Background(), variant, nil)
	if err != nil {
		return err
	}
	fmt.Println(path)
	return nil
}

func rank(checkpoint, query string, documents []string) error {
	encoder, err := embeddinggemma2.LoadEncoder(checkpoint)
	if err != nil {
		return err
	}
	queryVector, _, err := encoder.EmbedText(query, "query")
	if err != nil {
		return fmt.Errorf("query: %w", err)
	}
	matches := make([]match, 0, len(documents))
	for _, path := range documents {
		text, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		vector, _, err := encoder.EmbedText(string(text), "document")
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		// Both embeddings have unit length, so the dot product is cosine similarity.
		var score float64
		for i, v := range vector {
			score += float64(v) * float64(queryVector[i])
		}
		matches = append(matches, match{path, score})
	}
	slices.SortFunc(matches, func(a, b match) int {
		switch {
		case a.score > b.score:
			return -1
		case a.score < b.score:
			return 1
		default:
			return strings.Compare(a.path, b.path)
		}
	})
	for _, result := range matches {
		fmt.Printf("%.6f\t%s\n", result.score, result.path)
	}
	return nil
}
