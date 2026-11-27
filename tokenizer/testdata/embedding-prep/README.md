# Embedding text preparation reference

`reference.json` selects the `role`, `text`, `ids`, and optional `prepared_ids`
fields from the pbrain reference fixtures at `pkieltyka/pbrain` revision
`b4def8fad8ea34101c369777604144fbbec2182a`. The `row` field is the zero-based
index in the original fixture. The accompanying `LICENSE` is pbrain's MIT
license at that revision.

| Model | pbrain fixture | Rows | Hugging Face checkpoint | Exercised in-tree? |
| --- | --- | --- | --- | --- |
| EmbeddingGemma 2 | `internal/embed/gemma2/testdata/reference.json` | 7, 8, 9, 11, 12 | [`google/embeddinggemma-2` at `f9972ac22e696a105c482202646979e33e4b5a1e`](https://huggingface.co/google/embeddinggemma-2/tree/f9972ac22e696a105c482202646979e33e4b5a1e) | Yes (`TestEmbeddingPrepReference` in `embeddinggemma2`) |
| BGE-small-en-v1.5 | `internal/embed/bge/testdata/reference.json` | 0, 11, 13, 20 | [`BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | Archived reference rows only (caller/external boundary) |

pbrain generated the expected IDs with the Hugging Face tokenizers.

- **In-tree EmbeddingGemma 2 coverage (`embeddinggemma2/embedding_prep_test.go`):**
  `TestEmbeddingPrepReference` exercises the five `embeddinggemma-2` rows
  through `embeddinggemma2.Encoder.Tokenize` — covering the checkpoint's query
  and document prompts, BOS/EOS framing, multilingual text (German, Japanese),
  and recognition of `<eos>` within text. Row 11 also checks `prepared_ids`
  separately through `tokenizer.Tokenizer.EncodeSegments`, which encodes the
  text's literal `<eos>` as ordinary content and appends the framing EOS.
- **Archived BGE-small-en-v1.5 boundary:** The four `bge-small-en-v1.5` rows
  remain archived in `reference.json` with their original pbrain attribution
  and checkpoint pin (query instruction
  `"Represent this sentence for searching relevant passages: "`, normalization,
  multilingual input, and CLS/SEP framing). goinfer does not ship or import an
  external BGE WordPiece encoder or runtime, so the in-tree test verifies only
  that those four rows remain present in the fixture and leaves optional BGE
  tokenizer verification to external callers.

The selected inputs fit their model contexts. Coverage is limited to token IDs;
pbrain adapter truncation, vector parity, and retrieval quality are outside
this slice.

The tokenizer artifacts stay local. The test verifies their SHA-256 digests
before use and requires no weights or network access. The publishers identify
the model licenses as [Apache-2.0 for EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma)
and [MIT for BGE](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a).

From the goinfer root, download the two pinned EmbeddingGemma 2 tokenizer files
and run the opt-in test:

```sh
assets="${XDG_CACHE_HOME:-$HOME/.cache}/goinfer-embedding-prep"
gemma=https://huggingface.co/google/embeddinggemma-2/resolve/f9972ac22e696a105c482202646979e33e4b5a1e
mkdir -p "$assets/embeddinggemma-2"
curl -fLsS "$gemma/tokenizer.json" -o "$assets/embeddinggemma-2/tokenizer.json"
curl -fLsS "$gemma/config_sentence_transformers.json" -o "$assets/embeddinggemma-2/config_sentence_transformers.json"
python3 scripts/timing_lock.py run --shared --label embedding-prep-reference -- \
  go test ./embeddinggemma2 -count=1 -v -run '^TestEmbeddingPrepReference$' \
  -args -embedding-prep-assets "$assets"
```

The test skips when `-embedding-prep-assets` is absent and fails when an
explicitly supplied directory has missing or mismatched files.
