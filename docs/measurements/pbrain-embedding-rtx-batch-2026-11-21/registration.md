# EmbeddingGemma 2 CUDA document pairs: registration

Registered 2026-11-21, before any timed cell, at the batch branch commit that adds this file.

**Question.** On the frozen pbrain shared corpus (manifest SHA-256 `420988ad6653a9141a56ef61f12de91adc07255705426a3042f9a41cf413d732`), does encoding the 32 current documents in fixed adjacent pairs through one native CUDA forward (`ForwardBatch`) take less summed document time than one forward per document, with every output unchanged?

**Arms.** One binary, `testdata/main.go` built with `-tags cuda` at the commit that adds this file, on `google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e` from local disk.
- A: `-arm single`, a `Forward` per document.
- B: `-arm pairs`, documents 1+2, 3+4, … 31+32 in the manifest's current order, a `ForwardBatch` per pair.

The pairs are 16 fixed groups of 261 to 5,027 rows. The two long pairs are `time-window`+`discovery-field-guide` (2,176 rows) and `storage-field-guide`+`tokenizer-report` (5,027 rows). Queries and the first call `a001` run one at a time in both arms. Tokenization and pooling are inside every timed call, and both arms pool with the encoder's own arithmetic. No scratch reuse, packed bounds or default change is involved.

**Tier 0, already passed.** The tiny and real-checkpoint tests show every batched input bit-identical to the input alone. The real-checkpoint test is `TestEG2CUDA_realBatchParity`: 1 to 2,048 tokens around the 512-radius window, neighbours, and the golden's 48 texts.

**Tier 1 cells.** Five blocks of A B B A, 20 cells in total, each a fresh process. They hold the exclusive timing lock throughout, on the RTX PRO 6000 Blackwell Max-Q (driver 595.112.07). Before each cell the host must have a one-minute load of at most 1.0, at least 10 GiB available and at least 3 GiB free device memory. A cell started over the load bar waits; one started outside the fit margin stops the session.

**Exact output.** Every cell must report the same per-input ID, role, token count, framed-ID digest and vector digest for all 139 inputs, and the same top five and abstention for all 107 queries, in both arms. One mismatch stops the session as a correctness failure, with no timing read.

**Metric.** The primary metric is the summed document time of a cell: the 32 per-document calls in A, the 16 per-pair calls in B. Each adjacent A/B pair of cells gives one ratio A/B, 10 ratios in all. Secondary metrics, reported but not decisive:
- the warm query median, over the 106 queries after `a001`, which should not move;
- model load and CUDA attach;
- peak device memory from a sampler, run separately from the timed cells.

**Decision rule.** The bit-identical rule of `CLAUDE.md` ("Measurement discipline") applies, since batching replaces nothing and is opt-in:
- Median ratio at least 1.05 with at least 9 of 10 pairs favouring B: batching is a candidate for an opt-in integration proposal. Its memory and longest-input effects go to the operator.
- Median under 1.05, or fewer than 9 favouring B: parked.
- A query median that moves by more than 5 % in either direction: also reported, and investigated before any proposal.

**Stopping.** Stop on a correctness failure, after the 20 fixed cells, or after 10 minutes of waiting for a quiet window. There is no early stop on a promising or poor reading. Each cell takes about 6 s, so the session should need about 3 minutes of lock time plus idle waits.
