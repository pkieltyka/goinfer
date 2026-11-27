# Owned Markdown replay v1

This backend-neutral workload represents document embedding with 219 ordered passages, 54 groups of four and a final group of three. The fictional station logs come from the original section bank and ordered recipes. They provide short, middle and long inputs for a direct prepared-ID replay. The labeled correctness fixture remains under the Metal module.

The two pinned tokenizers each prepare 113,707 document positions. The median is 531, with lengths from 39 to 2,044. The five length bands contain 22 up to 127, 25 from 128 to 383, 117 from 384 to 640, 51 from 641 to 879 and four from 1,500 to 2,048. Group totals range from 1,636 to 2,375. Metal's separate 2,748-row admission cap is larger than any group in this fixture.

## Shared input contract

[The manifest](manifest.json) orders the document IDs and groups and pins the bytes of the [corpus](corpus.md), [section bank](sections.md) and [recipes](recipes.json). Each corpus record starts with an exact `<!-- passage:watch-NNN -->` delimiter line. Pass the following UTF-8 Markdown body through the public document tokenizer as written, including its final LF. The prepared pin for the selected model gives each body's length and SHA-256 over the ordered IDs serialized as little-endian unsigned 32-bit words. Q8_0 supplies the document prefix `title: none | text: ` to the GGUF loader. The MLX8 directory supplies the same bytes under its named `document` prompt. The checkpoint files and revisions are pinned in the [labeled fixture manifest](../../../backend/metal/internal/markdowneval/testdata/v1/manifest.json).

Consumers need the Markdown corpus and manifest. Recipe section numbers are zero-based indexes into the section bank, where a line of three hyphens separates sections. The model-free test renders each recipe in Go and checks it against the committed bytes.

```sh
python3 scripts/timing_lock.py run --shared --label eg2-owned-fixture -- \
  go test ./embeddinggemma2 -run '^TestMarkdownReplayFixture$'
```

## Run the Metal baseline

Prepare a local copy of each pinned checkpoint on fast local storage and an empty output directory. Run from the repository root on a Mac that supports `metal-eg2-batch`.

```sh
scripts/make_work.sh
out=$(mktemp -d)
python3 scripts/timing_lock.py run --label eg2-owned-replay -- \
  go test -v -count=1 ./backend/metal -run '^TestEG2MetalMarkdownReplay$' -args \
  -eg2-replay-fixture "$PWD/testdata/embeddinggemma2-markdown-replay/replay-v1" \
  -eg2-replay-out "$out" \
  -eg2-batch-test-gguf /local/models/embeddinggemma-2-Q8_0.gguf \
  -eg2-batch-test-mlx /local/models/embeddinggemma-2-8bit
```

For each format the test verifies checkpoint hashes and prepared inputs, builds a singleton oracle on the actual batch attachment and checks every packed result at both 768 and independently L2-normalized 512. It probes rotation, a split and a different neighbour on the one-run and two-run groups. The off-clock group census records admitted rows, real dispatch counts, FFN runs and advisory memory alongside tracked allocation boundary samples. Output files include both-width vectors, per-row digests and the aggregate digests in corpus order.

After warming the largest group, the test clears RoPE tables before each of three resident replays. Arena and row scratch stay warm. Replay wall covers the public prepared calls through pooling. Preparation, attachment, fit checks, verification and export run outside that interval. The receipts include each wall time, useful positions per second and the number of completed commands. These observations describe the stated replay and cache state on the attached device.
