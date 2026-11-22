# embeddinggemma2: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `embeddinggemma2`, moved here verbatim by the comment diet
(`docs/completed/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. The sections marked as moved retain their original text,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestTiny_parity

Moved from `embeddinggemma2/model_test.go` (the comment above `TestTiny_parity`) on 2026-10-09.

```text
TestTiny_parity is Gate 1 (docs/completed/task-embeddinggemma2.md): on a random-weight fixture pinned with
transformers' own embedding_gemma2 modeling code, every layer's hidden state, the projected last hidden state, the
pooled vector before normalisation and the sentence embedding match the float32 reference to cosine 0.9999 (each on
its own, so a pooling or normalisation bug cannot pass on a matching hidden state), and the embedding is as accurate
as the reference's own float32 run: its max |diff| to the same weights run in float64 is at most 1.5x the float32
reference's, plus 1e-7 (float32 rounding in the reference is the same size as goinfer's, so an absolute bar on the
float32 output cannot tell them apart; the 1e-7 floor is the task doc's original band, about three float32 ulps at
these values, because where the reference's own error is a couple of ulps, 1.5x of it is a rounding coin toss that
differs by arch: linux/amd64, which does not fuse multiply-adds, read 1.06e-7 against arm64's 6.4e-8 on the one-token
case). On a miss the first divergent layer is named.
```

## TestReal_vision.V4

Moved from `embeddinggemma2/vision_real_test.go` (the comment above `TestReal_vision.V4`) on 2026-10-09.

```text
	V4: every case's end-to-end embedding (aikit's preprocessing and tower, then the encoder) has cosine >= 0.999 with
	    the reference, and the same embedding from HF's own pixels has cosine >= 0.9999. The pre-registered bar was
	    0.9999 end to end; the read landed in its ambiguous band (0.99924-0.99986) with the whole gap in the resize
	    (aikit bilinear, the reference bicubic; 1.000000000 from HF's pixels), and the owner accepted it on 2026-10-06
	    ("since we understand the difference i'm ok with it"). The loose bar covers only the resize: everything after it
	    is still held to 0.9999 through HF's pixels. V3 and V4 are read for bilinear.
```

## TestReal_vision.modes

Moved from `embeddinggemma2/vision_real_test.go` (the comment above `TestReal_vision.modes`) on 2026-10-09.

```text
The modes read: aikit's bilinear (V3/V4) and the reference's bicubic (R1/R2), both aikit's since 2026-10-06.
```

## EmbeddingGemma 2 text GGUF

Applications open an Unsloth Q8_0 or UD-Q6_K_XL text file with
`embeddinggemma2.LoadGGUFEncoder`. The caller supplies the query and document prefixes
and can require CUDA with `Encoder.UseAccelerator("cuda")`. The loader uses the
embedded tokenizer. Packed CUDA batches use the existing `Encoder.EmbedIDsBatch`
entry point. [Go usage guide](../use-from-go.md#embeddinggemma-2-text-gguf)

The in-tree GGUF reader checks the directory, tensor bounds and supported encodings.
The encoder then checks the architecture, quantization version, tensor shapes and
the storage type for each weight role. Q8_0 also contains FP32 and BF16 tensors.
UD-Q6_K_XL additionally uses Q8_0 for some projections. The loader expands
every tensor once into owned FP32 weights and keeps the existing embedding graph.
Generative GGUF loaders still expose aikit's GGUF type through the decoder API,
and the embedding forward still uses aikit's math. [Reader](../../internal/gguf/README.md) ·
[Unsloth artifact revision](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/tree/031f0d4b35536f69ab3509d4893c923264fcf253)

Keep the GGUF file digest and preprocessing configuration in the vector-cache
identity. Changing either calls for recomputing document and query vectors.
The current CUDA results verify correctness and retrieval for the two profiles.
Keeping the token table compressed and adding packed projection kernels are
separate opportunities to reduce runtime memory and test inference speed.
[Qualification and benchmark findings](../measurements/embeddinggemma2-gguf-2026-11-22/README.md) ·
[Reference GGML decoders](https://github.com/unslothai/llama.cpp/blob/8d60873dfb8619f2b5ed31adfbab07cccb40db5c/ggml/src/ggml-quants.c)
