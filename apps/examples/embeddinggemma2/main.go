// Command embeddinggemma2 ranks Markdown documents by embedding similarity.
//
//	go run ./apps/examples/embeddinggemma2 <checkpoint-dir> <query> <doc.md> [doc.md ...]
package main

import (
	"flag"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

type match struct {
	path  string
	score float64
}

func main() {
	device := flag.String("accelerator", "cpu", "cpu, auto, or a registered accelerator name")
	flag.Parse()
	args := flag.Args()
	if len(args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: embeddinggemma2 [-accelerator cpu|auto|name] <checkpoint-dir> <query> <doc.md> [doc.md ...]")
		os.Exit(2)
	}
	if err := rank(args[0], args[1], args[2:], *device); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func rank(checkpoint, query string, documents []string, device string) error {
	encoder, err := embeddinggemma2.LoadEncoder(checkpoint)
	if err != nil {
		return err
	}
	switch device {
	case "cpu":
	case "auto":
		choice := encoder.UseAutoAccelerator()
		for _, decline := range choice.Declined {
			fmt.Fprintf(os.Stderr, "%s: %v\n", decline.Selector, decline.Err)
		}
		fmt.Fprintln(os.Stderr, "embedding backend:", choice.Backend)
	default:
		if _, err := encoder.UseAccelerator(device); err != nil {
			return err
		}
	}
	if accelerator := encoder.Accelerator(); accelerator != nil {
		defer accelerator.Close()
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
