# Rank Markdown with EmbeddingGemma 2

Run from the goinfer checkout with a local Hugging Face checkpoint directory containing the model weights, tokenizer, and Sentence Transformers prompt configuration.

```sh
fixture=apps/examples/embeddinggemma2/testdata
go run ./apps/examples/embeddinggemma2 "$EMBEDDINGGEMMA2_DIR" \
  'I pulled a teammate’s new notes. How do I make the new wording findable by meaning?' \
  "$fixture"/*.md
```

The program prints cosine similarity followed by the file path, highest score first. For this fixture, expect `index-refresh.md` to rank first. The example uses separate checkpoint prompts for retrieval queries and documents. It embeds each whole file, including any front matter. To evaluate application behavior with chunking, BM25, and SQLite-backed indexing, use pbrain's retrieval suite.

The question is `a001` in pbrain's `internal/semantic/calibrate/testdata/retrieval/questions.json` at commit `b4def8fad8ea34101c369777604144fbbec2182a`. The four files are copied from that commit's `brain/decisions/` and `brain/notes/` fixture directories. Their license is in `testdata/LICENSE`. The expected rank was checked with `google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e`.
