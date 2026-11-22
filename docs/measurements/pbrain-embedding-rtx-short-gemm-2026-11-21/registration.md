# Short-query FP32 CUDA GEMM experiment

The [RTX forward profile](../pbrain-embedding-rtx-profile-2026-11-21/README.md) attributes most of the profiled 24-token query time to matrix multiplications. The correction at `44c5aa2ab` is the reference. Measure the current register-blocked FP32 GEMM and a one-element launch control first. Use the real matrix shapes and invocation counts from the pinned Google model. Matrix dimensions below use `N` outputs and `K` inputs.

| N | K | Launches per forward |
| ---: | ---: | ---: |
| 512 | 512 | 96 |
| 2,048 | 512 | 52 |
| 512 | 2,048 | 28 |
| 1,024 | 512 | 20 |
| 512 | 1,024 | 20 |
| 12,288 | 512 | 1 |
| 768 | 512 | 1 |

Use the measured shape-weighted result to choose a smaller FP32 tile and a candidate row cutoff. The 107 frozen queries span 11 to 36 prepared tokens with a median of 22. Include 76 and 148 rows as passage controls. The candidate preserves the ascending-K fused multiply-add chain and the bias operation. Stop on any raw bit mismatch or if the measured launch control leaves little matrix work to accelerate. Verify exact results for representative ragged and aligned raw matrices and for the real 139-input corpus, including 107 query rankings. Keep the candidate in an opt-in build and preserve the existing default dispatch.

Compare the two dispatches through the same binary using five ABBA quartets on the RTX PRO 6000 Blackwell Max-Q. Each cell loads an encoder and warms up the longest query once. It uses all 107 prepared IDs in export order and records the synchronized `EmbedIDs` interval including pooling. The exported IDs, model digest, binary digest, host load and memory, device and driver, each token and vector digest, and warm-query median and p95 go into the record. The median is primary. Require exact outputs in every cell. The existing lossless opt-in performance rule is median speed ratio at least 1.05 and at least nine of ten adjacent pairs in favor. The experiment controls its explicit reference and candidate selections. It supplies no default change.

Use the existing exclusive timing lock and per-cell idle and fit rules. Stop after 20 clean cells or 600 seconds of aggregate idle-gate waiting. Allow roughly one minute of cell execution plus idle waits. Keep the raw microbench and the short-query end-to-end evidence separate.

The existing-kernel control uses the 16×16 tiled GEMM plus the existing explicit bias-add kernel for inputs of at most 36 rows, with at most 2,048 outputs and an aligned input dimension. It keeps the original dispatch for the large initial PLE projection. Compare this control in its own 20-cell run. An independently built fused tile uses a separate comparison with the same reference and inputs. Treat any inputs outside the candidate's selected shape range as reference dispatches in both arms.
