- **EmbeddingGemma 2 from bytes.** `embeddinggemma2.NewModel` builds the text encoder from weights already in memory:
  the token table as the checkpoint stores it (bfloat16 or float32), kept as it is and widened one row per token, and a
  safetensors file of the other `language_model.*` tensors, widened once across the processors.
  `embeddinggemma2.NewEncoder` builds an `Encoder` from a model, a tokenizer and a prompt table, and
  `embeddinggemma2.LoadConfigBytes` reads a `config.json` from memory. A caller that has read and checked a
  checkpoint's files can now run it without reading them again by path. Vectors equal `Load`'s bit for bit.
