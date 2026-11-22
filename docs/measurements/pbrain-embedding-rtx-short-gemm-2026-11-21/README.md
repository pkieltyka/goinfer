# EmbeddingGemma 2 short-query CUDA GEMM

The [short-query experiment](registration.md) builds on [RTX stage attribution](../pbrain-embedding-rtx-profile-2026-11-21/README.md). On 2026-11-21, the corrected CUDA encoder's register-blocked GEMM was measured on the RTX PRO 6000 Blackwell Max-Q Workstation Edition with driver 595.112.07. Kernel diagnostic `8f2355463` used five interleaved rounds of 2,000 launches and one terminal synchronization per sample. It cycled weights through 256 MB of resident copies. Host load was 0.98–1.06 during this diagnostic.

At 24 rows, the 512-output GEMM measured 40.68 µs for K=512, 80.09 µs for K=1,024, and 158.87 µs for K=2,048. The one-element launch control measured 6.56 µs. The 512×512 case returned 40.70 µs in the repeated reference arm. Across the full projection-shape mix, the weighted estimate was 12.93 ms for the current kernel and 7.61 ms for the existing tiled kernel before its separate bias add. The tiled kernel lost on the large initial PLE projection and at 524 rows. These are kernel diagnostics. The forward-pass comparison includes the complete encoder and pooling.

An opt-in Go dispatch reuses the 16×16 tiled GEMM and an explicit zero-bias add for at most 36 rows, at most 2,048 outputs, and aligned K. The RTX passed bit-equivalence for 14 raw shapes, including ragged dimensions and selector boundaries, and matched all 139 token/vector digests and 107 rankings and scores on the actual checkpoint. The initial diagnostic corpus pass had query medians of 15.85 ms and 10.81 ms. This pair established fit and correctness. A separate 20-cell comparison uses the same prepared IDs.

The post-correctness source also drains the CUDA queue before freeing scratch if the second launch fails. The RTX passed the planted nil-pipeline failure test, which covers both immediate drain and failed-drain retention through Close retry. Successful arithmetic is unchanged. Device logs and preserved source for the preflight and performance build are under `~/goinfer-logs/pbe2-rtx-short-gemm-2026-11-21`. The kernel-only diagnostic is under `~/goinfer-logs/eg2-small-gemm-2026-11-21`.

## Matched query timing with the existing tiled kernel

The 20-cell run completed from 18:23:38 to 18:26:50 EST, including 135 seconds of idle waiting. Every cell encoded the same 107 prepared queries on the same RTX and matched all output digests. The same cgo-free Go binary ran both arms. Median query latency across cells was **15.618 ms** with register-blocked GEMM and **10.483 ms** with the opt-in tiled dispatch. The median adjacent-pair ratio was **1.489×**, with 1.481×–1.505× across all ten pairs, all favoring the smaller tile. Median per-cell p95 was 15.892 ms and 10.751 ms respectively.

The timed executable SHA-256 is `8aee7021295eac4cf61e1c7667a03e075b141c97bc9667e1eaf137e8395f79fb`; its successful-path source is in the local `tiled-timed-source/` snapshot. The source-reviewed opt-in code and separately device-tested cleanup are at commit `8ff85dfcb` on `peter/eg2-short-gemm-tiled-control`. Raw cells, timing provenance, and paired ratios are in the local `tiled-cells/` directory. These results isolate the GEMM change from the earlier attention and batching experiments.

## Fused-tile selection

The separate selection run at `181ac9b24` repeated the exactness test on the same RTX and passed all 336 fused-candidate cases. The planted unfused implementation differed on all 84 input shapes.

The fastest new fused tile used 16 rows, 32 columns, and 128 inputs per K step. It measured 317.7 µs for the sliding-layer projection mix at 24 rows, versus 523.1 µs for the reference, a 1.65× kernel-only ratio. Its weighted whole-forward GEMM estimate was 7.894 ms. The existing tiled kernel's estimate was 7.687 ms before the separate bias kernel. The fused experiment stopped at its registered 2× selection rule. The measured forward-pass candidate is the existing tiled kernel with the explicit bias add.

Raw selection and exactness evidence is in `~/goinfer-logs/eg2-small-gemm-2026-11-21/select.log`, beside its start conditions and source revision.
