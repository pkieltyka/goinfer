# EmbeddingGemma 2 Q8 and Q6 verification

The tier-1 question is whether the pure-Go GGUF loader preserves the intended
text embedding on CPU and CUDA with the same artifact. Use
[Unsloth revision 031f0d4](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/tree/031f0d4b35536f69ab3509d4893c923264fcf253)
and verify full SHA-256 before use. Hold the model constant within each CPU/CUDA
comparison. The implementation expands weights to FP32 and uses the existing
CUDA singleton and native batch forwards. This run qualifies file compatibility and
correctness. It makes no native-quant speed claim and changes no model default.

Run a short query and prepared 511- and 514-token sequences through the actual
RTX PRO 6000 and the CPU. Require finite 768-dimensional singleton and batch
vectors, CUDA/CPU cosine at least 0.99999 and max absolute coordinate error at
most 3e-5. Check explicit GPU attachment and fail on shortage or device refusal.
Use the frozen pbrain manifest SHA-256
`420988ad6653a9141a56ef61f12de91adc07255705426a3042f9a41cf413d732`
with 32 documents and 107 queries. Require all prepared IDs to match the pinned
Google-tokenizer reference, and normalized CUDA vectors to match the same-file
CPU vectors at that tolerance. Record same-file llama.cpp parity separately
using its previously recorded vectors. Report cosine and discrepancies in ranking.

Report Recall@5 and MRR for the 47 answerable queries, plus per-query ranked
results, misses and false accepts at the frozen 0.72 threshold for all 107.
Compare with the existing same-session peer output by artifact for quality,
and with the frozen Google BF16 reference for changes caused by quantization.
No result selects a pbrain default. Stop after both real file checks and one full
corpus pass per file and backend. Budget about five minutes plus the timing
lock and fit checks, based on prior frozen-corpus Go runs. The local references,
prompts, and device driver identify the scope of claims.
