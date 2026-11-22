# PBA2 Vulkan embedding registration

The tier-1 decision is whether the optional direct Vulkan accelerator on an
RX 9060 XT improves hot EmbeddingGemma 2 document encoding over the Ryzen
9 7950X CPU on the same host. Use the pinned Google checkpoint
`f9972ac22e696a105c482202646979e33e4b5a1e` with the same BF16 weights widened
to FP32 and the frozen pbrain corpus manifest SHA-256
`420988ad6653a9141a56ef61f12de91adc07255705426a3042f9a41cf413d732`.
Run the 32 current documents and all 107 queries from the manifest. Use local model files,
one cgo-free executable, and four Go scheduler threads. Both arms time
tokenization, single-input embedding and pooling with the model resident.

Before timing, require fit and verified hardware identity, identical prepared
IDs, finite 768-dimensional vectors with L2 norms within 1e-5 of one, and
CPU/Radeon cosine of at least 0.9999 for all 139 inputs. Require identical
ordered top-five and abstention results at the frozen 0.72 cutoff. Validate
the 511–514-token boundary cases and the 2,655-token document with the same
checkpoint and executable.

Run five ABBA blocks under the exclusive timing lock for ten matched pairs.
The primary metric is the sum of the 32 document latencies in each cell.
Pass when Vulkan wins at least nine pairs and the median paired CPU/Vulkan
ratio is at least 1.05. The warm-query metric is the
cell median of the 106 queries after its first query. Require a median
paired warm-query CPU/Vulkan ratio of at least `1/1.05`. Park a result that
misses either speed condition. A correctness, device, fit or execution
failure stops the run. Log otherwise successful cells voided by the
environment and retry the same position in order, up to four retries.
Stop after 20 clean cells or when the aggregate idle wait reaches 600 seconds. The earlier frozen CPU and GPU corpus runs put the wall
budget at roughly 8–12 minutes including correctness and idle waits.

Record memory, load, temperature, identity, versions and build checksums.
Use separate runs for the three-engine CPU/CUDA/Vulkan description and for
pbrain's fresh SQLite index, which also exercises document batching. Report
their timing and fidelity with their own provenance.
