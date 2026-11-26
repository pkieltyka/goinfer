# Markdown embedding evaluation

This test fixture contains 15 original Markdown passages about fictional field operations and eight labeled queries. Every query has an explicit relevant passage and a plausible distractor. The ordered groups contain four, four, four and three passages. On the two pinned tokenizers the document groups contain 388, 1,023, 1,428 and 530 prepared tokens, with individual passages from 25 to 1,027 tokens.

[The manifest](testdata/v1/manifest.json) binds the raw corpus SHA-256, group order, relevance judgments, checkpoint revisions and file hashes. The parser preserves the UTF-8 Markdown body following each `<!-- passage:ID -->` delimiter line, including the final LF. The model-specific manifest entries also bind each document and query's prepared length and SHA-256 over its little-endian unsigned 32-bit IDs. Preparation uses EmbeddingGemma 2's named document and query prompts.

## Run the model-free tests

Run these commands from the repository root on Linux or macOS.

```sh
scripts/make_work.sh
python3 scripts/timing_lock.py run --shared --label eg2-markdown-metadata -- \
  go test -v ./backend/metal/internal/markdowneval
```

## Run the opt-in Metal evaluation

On a Mac that supports the explicit `metal-eg2-batch` selector, put the pinned Q8_0 GGUF file and MLX8 checkpoint directory on fast local storage. Pass their absolute paths. The test verifies both local checkpoints before loading each one.

```sh
scripts/make_work.sh
python3 scripts/timing_lock.py run --label eg2-markdown-eval -- \
  go test -v -count=1 ./backend/metal \
  -run '^TestEG2MetalMarkdownEval$' -args -eg2-markdown-eval \
  -eg2-batch-test-gguf /local/models/embeddinggemma-2-Q8_0.gguf \
  -eg2-batch-test-mlx /local/models/embeddinggemma-2-8bit
```

The test uses public `Tokenize`, `EmbedIDs` and `EmbedIDsBatch` paths. It checks preparation for both roles, then compares singleton document vectors to every packed, reordered and repeated result word-for-word at 768 and independently L2-normalized 512 dimensions. Reordering rotates the entire document sequence by one passage and restores corpus order for comparison. A batch counts as exercised only after one actual submit and wait. The test logs advisory batch allocation estimates separately.

Each run reports descriptive wall time and process-wide Go allocator byte deltas for loading, attaching, tokenizing and embedding. The packed value sums the four public batch calls. Hashing and output verification run outside those intervals. The allocator deltas describe cumulative Go heap traffic during each phase. Treat the timings as observations from the correctness schedule.

The report prints all ranked passage IDs and cosine scores at each width for the eight singleton queries. Hits carry the manifest's relevance, distractor or unjudged label. First relevant rank and recall at five describe this fixture. The test passes when preparation, execution and exact layout parity checks pass.
