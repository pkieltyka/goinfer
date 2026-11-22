# PBA2 Vulkan embedding registration

The tier-1 decision is whether the optional direct Vulkan accelerator on an
RX 9060 XT improves hot EmbeddingGemma 2 document encoding over the Ryzen
9 7950X CPU on the same host. Use the pinned Google checkpoint
`f9972ac22e696a105c482202646979e33e4b5a1e` with the same BF16 weights widened
to FP32 and the frozen pbrain corpus manifest SHA-256
`420988ad6653a9141a56ef61f12de91adc07255705426a3042f9a41cf413d732`.
The manifest contains 32 documents and 107 queries. Use local model files,
one cgo-free executable, and four Go scheduler threads. Both arms time
tokenization, single-input embedding and pooling with the model resident.

Before timing, require fit and verified hardware identity, identical prepared
IDs, finite unit-norm vectors with CPU/Radeon cosine of at least 0.9999, and
identical ordered top-five and abstention results. Cover the held-out
511–514-token boundary cases and the longest document in untimed validation.

Run five ABBA blocks under the exclusive timing lock for ten matched pairs.
The primary metric is the sum of the 32 document latencies in each cell.
Pass when Vulkan wins at least nine pairs and the median paired CPU/Vulkan
ratio is at least 1.05. Require a median paired warm-query CPU/Vulkan ratio
of at least `1/1.05`. Park a result that misses either speed condition.
A correctness, device, fit or execution failure stops the registered run.
Log invalid environmental cells at the point they occurred and retry the
same position in order, up to four environmental retries. Stop after
20 clean cells. The earlier frozen CPU and GPU corpus runs put the wall
budget at roughly 8–12 minutes including correctness and idle waits.

Record memory, load, temperature, identity, versions and build checksums.
Use separate runs for the three-engine CPU/CUDA/Vulkan description and for
pbrain's fresh SQLite index, which also exercises document batching. Report
their timing and fidelity with their own provenance.
