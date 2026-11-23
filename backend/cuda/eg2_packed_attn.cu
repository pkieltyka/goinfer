// eg2_packed_attn.cu -- packed variable-length bidirectional fused attention for
// EmbeddingGemma 2 (goinfer #63):
//
//   * Per-segment fixed local query/key tiles: each block owns one (head,
//     segment-local query tile [qLocal0, qLocal0 + BM)) from a host-built tile
//     descriptor table (tileSegStart, tileSegLen, tileQLocal0). Tiles never
//     cross segment boundaries and never read from adjacent segments.
//   * Bit-identical across singleton, split, pack, and reorder on the same
//     device: every key tile boundary, score reduction order, online softmax
//     step, and mask depends only on (segLen, qLocal0, window, BM, BN), with
//     zero dependence on segStart or batch packing.
//   * Direct Q/K/V row-major access with GQA (kvh = h / (nH / nKV)), eliminating
//     per-layer QKV packing copies and the fused [T, 3, nH, HD] buffer.
//   * Bidirectional full (window < 0) and symmetric sliding window (window >= 0,
//     inclusive |qLocal - kLocal| <= window), arbitrary float scale (EG2 uses
//     scale = 1.0f; general scaling supported).
//   * FP16 mma.sync.aligned.m16n8k8 tensor-core QK^T and PV with FP32
//     accumulation and FP32 online softmax; all scalar float MACs use explicit
//     __fmaf_rn / __fmul_rn / __fadd_rn intrinsics (FMA-lint compliant).
//   * WARP_D warps cooperate along HD so each warp keeps HD/WARP_D <= 128 output
//     columns in registers (0 bytes local-memory spill) and total shared memory
//     stays under the 48 KB default limit for HD=256 (34.5 KB) and HD=512
//     (44.5 KB).
//
// Regenerate with:  ./build_ptx.sh eg2_packed_attn

#include <cuda_fp16.h>

#define KPAD 8

__device__ __forceinline__ void mma_m16n8k8_f32(float &d0, float &d1, float &d2, float &d3,
                                                unsigned int a0, unsigned int a1, unsigned int b0,
                                                float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

union f32_vec4 {
    uint4 u;
    float f[4];
};

template <int HD, int BM, int BN, int WARP_D, int DK_SUB>
__device__ __forceinline__ void eg2_packed_attn_impl(
    const float* __restrict__ q,
    const float* __restrict__ kc,
    const float* __restrict__ vc,
    const int* __restrict__ tileSegStart,
    const int* __restrict__ tileSegLen,
    const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles,
    float* __restrict__ out) {
    static_assert(HD % 16 == 0, "HD must be a multiple of 16");
    static_assert(DK_SUB % 16 == 0, "DK_SUB must be a multiple of 16");
    static_assert((BN & (BN - 1)) == 0, "BN must be a power of two");
    constexpr int HD_WARP = HD / WARP_D;
    constexpr int OT_WARP = HD_WARP / 8;
    constexpr int NG = BN / 8;
    constexpr int DK_TOT = WARP_D * DK_SUB;
    constexpr int N_CHUNKS = HD_WARP / DK_SUB;
    constexpr int SUB_TILES = DK_SUB / 8;
    constexpr int HD_V4 = HD / 4;
    constexpr int DK_TOT_V4 = DK_TOT / 4;
    constexpr int DK_SUB_V4 = DK_SUB / 4;

    const int h = blockIdx.x;
    const int tileIdx = blockIdx.y;
    if (h >= nH || tileIdx >= totalTiles) return;

    const int segStart = tileSegStart[tileIdx];
    const int segLen = tileSegLen[tileIdx];
    const int qLocal0 = tileQLocal0[tileIdx];
    if (segLen <= 0 || qLocal0 >= segLen) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warpM = warp / WARP_D;
    const int warpD = warp - warpM * WARP_D;
    const int qRow0 = lane >> 2;
    const int qRow1 = qRow0 + 8;
    const int colBase = (lane & 3) * 2;
    const int dBaseWarp = warpD * HD_WARP;

    const int qDim = nH * HD;
    const int kvDim = nKV * HD;
    const int group = nH / nKV;
    const int kvh = h / group;

    constexpr int qStride = HD + KPAD;
    constexpr int kStride = DK_TOT + KPAD;
    constexpr int vStride = BN + KPAD;

    extern __shared__ __half sh[];
    __shared__ int tileVNonFinite;
    __half* Qsh = sh;                              // [BM][HD + KPAD]
    __half* KVsh = sh + (size_t)BM * qStride;      // overlays Ksh, Vtsh, and Ssh
    __half* Ksh = KVsh;                            // [BN][DK_TOT + KPAD]
    __half* Vtsh = KVsh;                           // [DK_TOT][BN + KPAD]
    float* Ssh = reinterpret_cast<float*>(KVsh);   // [WARP_M][WARP_D - 1][16][BN] when WARP_D > 1
    if (tid == 0) tileVNonFinite = 0;

    const int nqTile = (segLen - qLocal0 < BM) ? (segLen - qLocal0) : BM;
    const bool qVecOK = (reinterpret_cast<unsigned long long>(q) & 15ull) == 0ull;
    const bool kVecOK = (reinterpret_cast<unsigned long long>(kc) & 15ull) == 0ull;
    const bool vVecOK = (reinterpret_cast<unsigned long long>(vc) & 15ull) == 0ull;

    // Stage this block's segment-local Q tile [nqTile, HD] into Qsh as f16.
    // Rows >= nqTile are zeroed; no reads ever occur past segStart + segLen.
    if (qVecOK) {
        for (int idx = tid; idx < BM * HD_V4; idx += blockDim.x) {
            const int qr = idx / HD_V4;
            const int dv = idx - qr * HD_V4;
            uint2 packed = make_uint2(0u, 0u);
            if (qr < nqTile) {
                f32_vec4 v;
                const float* src = q + (long)(segStart + qLocal0 + qr) * qDim + h * HD + dv * 4;
                v.u = *reinterpret_cast<const uint4*>(src);
                __half2 a = __floats2half2_rn(v.f[0], v.f[1]);
                __half2 b = __floats2half2_rn(v.f[2], v.f[3]);
                packed.x = *reinterpret_cast<unsigned int*>(&a);
                packed.y = *reinterpret_cast<unsigned int*>(&b);
            }
            *reinterpret_cast<uint2*>(&Qsh[qr * qStride + dv * 4]) = packed;
        }
    } else {
        for (int idx = tid; idx < BM * HD; idx += blockDim.x) {
            const int qr = idx / HD;
            const int dd = idx - qr * HD;
            float val = 0.f;
            if (qr < nqTile) {
                val = q[(long)(segStart + qLocal0 + qr) * qDim + h * HD + dd];
            }
            Qsh[qr * qStride + dd] = __float2half_rn(val);
        }
    }
    __syncthreads();

    const int qLoc0 = qLocal0 + warpM * 16 + qRow0;
    const int qLoc1 = qLocal0 + warpM * 16 + qRow1;
    const bool has0 = qLoc0 < segLen;
    const bool has1 = qLoc1 < segLen;

    // Segment-local per-row inclusive key bounds [winLo, winHi].
    const int winLo0 = (window >= 0 && qLoc0 > window) ? (qLoc0 - window) : 0;
    const int winHi0 = (window >= 0 && qLoc0 + window < segLen - 1) ? (qLoc0 + window) : (segLen - 1);
    const int winLo1 = (window >= 0 && qLoc1 > window) ? (qLoc1 - window) : 0;
    const int winHi1 = (window >= 0 && qLoc1 + window < segLen - 1) ? (qLoc1 + window) : (segLen - 1);

    // Segment-local tile key range [kTileStart, kTileEnd): fixed by (segLen, qLocal0, window, BM, BN),
    // with ascending key tiles anchored at multiples of BN relative to sequence start.
    const int tileLastQ = qLocal0 + nqTile - 1;
    const int rawStart = (window >= 0 && qLocal0 > window) ? (qLocal0 - window) : 0;
    const int kTileStart = rawStart & ~(BN - 1);
    const int kTileEnd = (window >= 0 && tileLastQ + window + 1 < segLen) ? (tileLastQ + window + 1) : segLen;

    float mRun0 = -1e30f, mRun1 = -1e30f;
    float lRun0 = 0.f, lRun1 = 0.f;
    bool hadPrev0 = false, hadPrev1 = false;
    float acc[OT_WARP][4];
#pragma unroll
    for (int t = 0; t < OT_WARP; t++) {
        acc[t][0] = 0.f;
        acc[t][1] = 0.f;
        acc[t][2] = 0.f;
        acc[t][3] = 0.f;
    }

    for (int k0 = kTileStart; k0 < kTileEnd; k0 += BN) {
        const int nk = (kTileEnd - k0 < BN) ? (kTileEnd - k0) : BN;

        // ---- QK^T partial dot products across this warpD's HD_WARP slice ----
        float s[NG][4];
#pragma unroll
        for (int g = 0; g < NG; g++) {
            s[g][0] = 0.f;
            s[g][1] = 0.f;
            s[g][2] = 0.f;
            s[g][3] = 0.f;
        }

#pragma unroll
        for (int ch = 0; ch < N_CHUNKS; ch++) {
            __syncthreads();
            if (kVecOK) {
                for (int idx = tid; idx < nk * DK_TOT_V4; idx += blockDim.x) {
                    const int kk = idx / DK_TOT_V4;
                    const int dv = idx - kk * DK_TOT_V4;
                    const int wd = dv / DK_SUB_V4;
                    const int dOff = (dv - wd * DK_SUB_V4) * 4;
                    const int gD = wd * HD_WARP + ch * DK_SUB + dOff;
                    f32_vec4 v;
                    const float* src = kc + (long)(segStart + k0 + kk) * kvDim + kvh * HD + gD;
                    v.u = *reinterpret_cast<const uint4*>(src);
                    __half2 a = __floats2half2_rn(v.f[0], v.f[1]);
                    __half2 b = __floats2half2_rn(v.f[2], v.f[3]);
                    uint2 packed;
                    packed.x = *reinterpret_cast<unsigned int*>(&a);
                    packed.y = *reinterpret_cast<unsigned int*>(&b);
                    *reinterpret_cast<uint2*>(&Ksh[kk * kStride + dv * 4]) = packed;
                }
            } else {
                for (int idx = tid; idx < nk * DK_TOT; idx += blockDim.x) {
                    const int kk = idx / DK_TOT;
                    const int dd = idx - kk * DK_TOT;
                    const int wd = dd / DK_SUB;
                    const int dOff = dd - wd * DK_SUB;
                    const int gD = wd * HD_WARP + ch * DK_SUB + dOff;
                    const float kv = kc[(long)(segStart + k0 + kk) * kvDim + kvh * HD + gD];
                    Ksh[kk * kStride + dd] = __float2half_rn(kv);
                }
            }
            __syncthreads();

#pragma unroll
            for (int cSub = 0; cSub < SUB_TILES; cSub++) {
                const int gD = dBaseWarp + ch * DK_SUB + cSub * 8 + colBase;
                const int qOff0 = (warpM * 16 + qRow0) * qStride + gD;
                const int qOff1 = (warpM * 16 + qRow1) * qStride + gD;
                const unsigned int qa0 = *reinterpret_cast<const unsigned int*>(&Qsh[qOff0]);
                const unsigned int qa1 = *reinterpret_cast<const unsigned int*>(&Qsh[qOff1]);
                const int kD = warpD * DK_SUB + cSub * 8 + colBase;
#pragma unroll
                for (int g = 0; g < NG; g++) {
                    const int kIdx = g * 8 + (lane >> 2);
                    unsigned int b0 = 0u;
                    if (kIdx < nk) {
                        b0 = *reinterpret_cast<const unsigned int*>(&Ksh[kIdx * kStride + kD]);
                    }
                    mma_m16n8k8_f32(s[g][0], s[g][1], s[g][2], s[g][3],
                                    qa0, qa1, b0,
                                    s[g][0], s[g][1], s[g][2], s[g][3]);
                }
            }
        }

        // ---- Fixed-order cross-WARP_D score reduction (only when WARP_D > 1) ----
        if constexpr (WARP_D > 1) {
            constexpr int S_STRIDE = 16 * BN;
            const int baseM = warpM * (WARP_D - 1) * S_STRIDE;
            __syncthreads();
            if (warpD > 0) {
                float* dst = Ssh + baseM + (warpD - 1) * S_STRIDE;
#pragma unroll
                for (int g = 0; g < NG; g++) {
                    const int cA = g * 8 + colBase;
                    dst[qRow0 * BN + cA]     = s[g][0];
                    dst[qRow0 * BN + cA + 1] = s[g][1];
                    dst[qRow1 * BN + cA]     = s[g][2];
                    dst[qRow1 * BN + cA + 1] = s[g][3];
                }
            }
            __syncthreads();
            if (warpD == 0) {
                float* slot0 = Ssh + baseM;
#pragma unroll
                for (int wd = 1; wd < WARP_D; wd++) {
                    const float* src = Ssh + baseM + (wd - 1) * S_STRIDE;
#pragma unroll
                    for (int g = 0; g < NG; g++) {
                        const int cA = g * 8 + colBase;
                        s[g][0] = __fadd_rn(s[g][0], src[qRow0 * BN + cA]);
                        s[g][1] = __fadd_rn(s[g][1], src[qRow0 * BN + cA + 1]);
                        s[g][2] = __fadd_rn(s[g][2], src[qRow1 * BN + cA]);
                        s[g][3] = __fadd_rn(s[g][3], src[qRow1 * BN + cA + 1]);
                    }
                }
#pragma unroll
                for (int g = 0; g < NG; g++) {
                    const int cA = g * 8 + colBase;
                    slot0[qRow0 * BN + cA]     = s[g][0];
                    slot0[qRow0 * BN + cA + 1] = s[g][1];
                    slot0[qRow1 * BN + cA]     = s[g][2];
                    slot0[qRow1 * BN + cA + 1] = s[g][3];
                }
            }
            __syncthreads();
            if (warpD > 0) {
                const float* slot0 = Ssh + baseM;
#pragma unroll
                for (int g = 0; g < NG; g++) {
                    const int cA = g * 8 + colBase;
                    s[g][0] = slot0[qRow0 * BN + cA];
                    s[g][1] = slot0[qRow0 * BN + cA + 1];
                    s[g][2] = slot0[qRow1 * BN + cA];
                    s[g][3] = slot0[qRow1 * BN + cA + 1];
                }
            }
        }

        // ---- Scale, per-row segment/window mask, and FP32 online softmax ----
        float tileMax0 = -1e30f, tileMax1 = -1e30f;
        unsigned int validMask0 = 0u, validMask1 = 0u;
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const int kA = g * 8 + colBase;
            const int kB = kA + 1;
            const int klA = k0 + kA;
            const int klB = k0 + kB;
            const bool v0A = has0 && (kA < nk) && (klA >= winLo0) && (klA <= winHi0);
            const bool v0B = has0 && (kB < nk) && (klB >= winLo0) && (klB <= winHi0);
            const bool v1A = has1 && (kA < nk) && (klA >= winLo1) && (klA <= winHi1);
            const bool v1B = has1 && (kB < nk) && (klB >= winLo1) && (klB <= winHi1);
            const int bitA = g << 1;
            const int bitB = bitA | 1;
            if (v0A) validMask0 |= (1u << bitA);
            if (v0B) validMask0 |= (1u << bitB);
            if (v1A) validMask1 |= (1u << bitA);
            if (v1B) validMask1 |= (1u << bitB);

            const float a0 = v0A ? __fmul_rn(s[g][0], scale) : -1e30f;
            const float a1 = v0B ? __fmul_rn(s[g][1], scale) : -1e30f;
            const float b0 = v1A ? __fmul_rn(s[g][2], scale) : -1e30f;
            const float b1 = v1B ? __fmul_rn(s[g][3], scale) : -1e30f;
            s[g][0] = a0;
            s[g][1] = a1;
            s[g][2] = b0;
            s[g][3] = b1;
            if (v0A) tileMax0 = fmaxf(tileMax0, a0);
            if (v0B) tileMax0 = fmaxf(tileMax0, a1);
            if (v1A) tileMax1 = fmaxf(tileMax1, b0);
            if (v1B) tileMax1 = fmaxf(tileMax1, b1);
        }

        unsigned int rowAny0 = validMask0;
        unsigned int rowAny1 = validMask1;
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            tileMax0 = fmaxf(tileMax0, __shfl_xor_sync(0xffffffffu, tileMax0, off));
            tileMax1 = fmaxf(tileMax1, __shfl_xor_sync(0xffffffffu, tileMax1, off));
            rowAny0 |= __shfl_xor_sync(0xffffffffu, rowAny0, off);
            rowAny1 |= __shfl_xor_sync(0xffffffffu, rowAny1, off);
        }

        const bool hasValid0 = (rowAny0 != 0u);
        const bool hasValid1 = (rowAny1 != 0u);
        const float mNew0 = hasValid0 ? (hadPrev0 ? fmaxf(mRun0, tileMax0) : tileMax0) : mRun0;
        const float mNew1 = hasValid1 ? (hadPrev1 ? fmaxf(mRun1, tileMax1) : tileMax1) : mRun1;
        const float alpha0 = hasValid0 ? (hadPrev0 ? __expf(mRun0 - mNew0) : 0.f) : 1.f;
        const float alpha1 = hasValid1 ? (hadPrev1 ? __expf(mRun1 - mNew1) : 0.f) : 1.f;

        float sum0 = 0.f, sum1 = 0.f;
        unsigned int pFrag[NG][2];
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const int bitA = g << 1;
            const int bitB = bitA | 1;
            const float p0 = ((validMask0 >> bitA) & 1u) ? __expf(s[g][0] - mNew0) : 0.f;
            const float p1 = ((validMask0 >> bitB) & 1u) ? __expf(s[g][1] - mNew0) : 0.f;
            const float p2 = ((validMask1 >> bitA) & 1u) ? __expf(s[g][2] - mNew1) : 0.f;
            const float p3 = ((validMask1 >> bitB) & 1u) ? __expf(s[g][3] - mNew1) : 0.f;
            sum0 = __fadd_rn(sum0, __fadd_rn(p0, p1));
            sum1 = __fadd_rn(sum1, __fadd_rn(p2, p3));
            __half2 pa = __floats2half2_rn(p0, p1);
            __half2 pb = __floats2half2_rn(p2, p3);
            pFrag[g][0] = *reinterpret_cast<unsigned int*>(&pa);
            pFrag[g][1] = *reinterpret_cast<unsigned int*>(&pb);
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            sum0 = __fadd_rn(sum0, __shfl_xor_sync(0xffffffffu, sum0, off));
            sum1 = __fadd_rn(sum1, __shfl_xor_sync(0xffffffffu, sum1, off));
        }

        if (hasValid0) {
            lRun0 = __fmaf_rn(alpha0, lRun0, sum0);
            mRun0 = mNew0;
            hadPrev0 = true;
        }
        if (hasValid1) {
            lRun1 = __fmaf_rn(alpha1, lRun1, sum1);
            mRun1 = mNew1;
            hadPrev1 = true;
        }

#pragma unroll
        for (int t = 0; t < OT_WARP; t++) {
            acc[t][0] = __fmul_rn(acc[t][0], alpha0);
            acc[t][1] = __fmul_rn(acc[t][1], alpha0);
            acc[t][2] = __fmul_rn(acc[t][2], alpha1);
            acc[t][3] = __fmul_rn(acc[t][3], alpha1);
        }

        // ---- PV accumulation across this warpD's HD_WARP slice ----
#pragma unroll
        for (int ch = 0; ch < N_CHUNKS; ch++) {
            __syncthreads();
            if (vVecOK) {
                for (int idx = tid; idx < nk * DK_TOT_V4; idx += blockDim.x) {
                    const int kk = idx / DK_TOT_V4;
                    const int dv = idx - kk * DK_TOT_V4;
                    const int wd = dv / DK_SUB_V4;
                    const int dOff = (dv - wd * DK_SUB_V4) * 4;
                    const int gD = wd * HD_WARP + ch * DK_SUB + dOff;
                    f32_vec4 v;
                    const float* src = vc + (long)(segStart + k0 + kk) * kvDim + kvh * HD + gD;
                    v.u = *reinterpret_cast<const uint4*>(src);
                    const __half h0 = __float2half_rn(v.f[0]);
                    const __half h1 = __float2half_rn(v.f[1]);
                    const __half h2 = __float2half_rn(v.f[2]);
                    const __half h3 = __float2half_rn(v.f[3]);
                    if (((__half_as_ushort(h0) & 0x7c00u) == 0x7c00u) ||
                        ((__half_as_ushort(h1) & 0x7c00u) == 0x7c00u) ||
                        ((__half_as_ushort(h2) & 0x7c00u) == 0x7c00u) ||
                        ((__half_as_ushort(h3) & 0x7c00u) == 0x7c00u)) {
                        tileVNonFinite = 1;
                    }
                    const int dd = dv * 4;
                    Vtsh[(dd + 0) * vStride + kk] = h0;
                    Vtsh[(dd + 1) * vStride + kk] = h1;
                    Vtsh[(dd + 2) * vStride + kk] = h2;
                    Vtsh[(dd + 3) * vStride + kk] = h3;
                }
            } else {
                for (int idx = tid; idx < nk * DK_TOT; idx += blockDim.x) {
                    const int kk = idx / DK_TOT;
                    const int dd = idx - kk * DK_TOT;
                    const int wd = dd / DK_SUB;
                    const int dOff = dd - wd * DK_SUB;
                    const int gD = wd * HD_WARP + ch * DK_SUB + dOff;
                    const float vv = vc[(long)(segStart + k0 + kk) * kvDim + kvh * HD + gD];
                    const __half hv = __float2half_rn(vv);
                    if ((__half_as_ushort(hv) & 0x7c00u) == 0x7c00u) {
                        tileVNonFinite = 1;
                    }
                    Vtsh[dd * vStride + kk] = hv;
                }
            }
            __syncthreads();

            const int vNonFinite = tileVNonFinite;
            if (vNonFinite == 0) {
                // Fast path: every entry in Vtsh is finite, so 0 * Vtsh == 0 in MMA.
#pragma unroll
                for (int g = 0; g < NG; g++) {
                    const unsigned int a0 = pFrag[g][0];
                    const unsigned int a1 = pFrag[g][1];
                    const int kIdx = g * 8 + colBase;
#pragma unroll
                    for (int tSub = 0; tSub < SUB_TILES; tSub++) {
                        const int t = ch * SUB_TILES + tSub;
                        const int dIdx = warpD * DK_SUB + tSub * 8 + (lane >> 2);
                        unsigned int b0 = 0u;
                        if (kIdx + 1 < nk) {
                            b0 = *reinterpret_cast<const unsigned int*>(&Vtsh[dIdx * vStride + kIdx]);
                        } else if (kIdx < nk) {
                            __half2 hv = __halves2half2(Vtsh[dIdx * vStride + kIdx], __float2half_rn(0.f));
                            b0 = *reinterpret_cast<unsigned int*>(&hv);
                        }
                        mma_m16n8k8_f32(acc[t][0], acc[t][1], acc[t][2], acc[t][3],
                                        a0, a1, b0,
                                        acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
                    }
                }
            } else {
                // Guarded exceptional-V path: at least one V element in this union tile is
                // NaN, Inf, or overflows FP16. Accumulate in FP32 strictly over each row's
                // valid keys so masked rows never compute 0 * NaN/Inf while eligible rows
                // still expose nonfinite inputs.
                const int rowBase = lane & ~3;
#pragma unroll
                for (int g = 0; g < NG; g++) {
                    const int bitA = g << 1;
                    const int bitB = bitA | 1;
#pragma unroll 1
                    for (int srcCol = 0; srcCol < 4; srcCol++) {
                        const int srcLane = rowBase + srcCol;
                        const unsigned int m0 = __shfl_sync(0xffffffffu, validMask0, srcLane);
                        const unsigned int m1 = __shfl_sync(0xffffffffu, validMask1, srcLane);
                        const unsigned int pf0 = __shfl_sync(0xffffffffu, pFrag[g][0], srcLane);
                        const unsigned int pf1 = __shfl_sync(0xffffffffu, pFrag[g][1], srcLane);
                        const __half2 hp0 = *reinterpret_cast<const __half2*>(&pf0);
                        const __half2 hp1 = *reinterpret_cast<const __half2*>(&pf1);
                        const float p0A = __low2float(hp0);
                        const float p0B = __high2float(hp0);
                        const float p1A = __low2float(hp1);
                        const float p1B = __high2float(hp1);
                        const bool v0A = ((m0 >> bitA) & 1u) != 0u;
                        const bool v0B = ((m0 >> bitB) & 1u) != 0u;
                        const bool v1A = ((m1 >> bitA) & 1u) != 0u;
                        const bool v1B = ((m1 >> bitB) & 1u) != 0u;
                        const int kA = g * 8 + (srcCol << 1);
                        const int kB = kA + 1;
#pragma unroll
                        for (int tSub = 0; tSub < SUB_TILES; tSub++) {
                            const int t = ch * SUB_TILES + tSub;
                            const int dcol = dBaseWarp + t * 8 + colBase;
                            if (v0A) {
                                const float* vr = vc + (long)(segStart + k0 + kA) * kvDim + kvh * HD + dcol;
                                acc[t][0] = __fmaf_rn(p0A, vr[0], acc[t][0]);
                                acc[t][1] = __fmaf_rn(p0A, vr[1], acc[t][1]);
                            }
                            if (v0B) {
                                const float* vr = vc + (long)(segStart + k0 + kB) * kvDim + kvh * HD + dcol;
                                acc[t][0] = __fmaf_rn(p0B, vr[0], acc[t][0]);
                                acc[t][1] = __fmaf_rn(p0B, vr[1], acc[t][1]);
                            }
                            if (v1A) {
                                const float* vr = vc + (long)(segStart + k0 + kA) * kvDim + kvh * HD + dcol;
                                acc[t][2] = __fmaf_rn(p1A, vr[0], acc[t][2]);
                                acc[t][3] = __fmaf_rn(p1A, vr[1], acc[t][3]);
                            }
                            if (v1B) {
                                const float* vr = vc + (long)(segStart + k0 + kB) * kvDim + kvh * HD + dcol;
                                acc[t][2] = __fmaf_rn(p1B, vr[0], acc[t][2]);
                                acc[t][3] = __fmaf_rn(p1B, vr[1], acc[t][3]);
                            }
                        }
                    }
                }
                __syncthreads();
                if (tid == 0) tileVNonFinite = 0;
            }
        }
    }

    const float inv0 = (lRun0 == 0.f) ? 0.f : (1.f / lRun0);
    const float inv1 = (lRun1 == 0.f) ? 0.f : (1.f / lRun1);
#pragma unroll
    for (int t = 0; t < OT_WARP; t++) {
        const int dcol = dBaseWarp + t * 8 + colBase;
        if (has0) {
            float* o = out + (long)(segStart + qLoc0) * qDim + h * HD;
            o[dcol]     = __fmul_rn(acc[t][0], inv0);
            o[dcol + 1] = __fmul_rn(acc[t][1], inv0);
        }
        if (has1) {
            float* o = out + (long)(segStart + qLoc1) * qDim + h * HD;
            o[dcol]     = __fmul_rn(acc[t][2], inv1);
            o[dcol + 1] = __fmul_rn(acc[t][3], inv1);
        }
    }
}

extern "C" __global__ void eg2_packed_attn_hd256(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const int* __restrict__ tileSegStart, const int* __restrict__ tileSegLen, const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles, float* __restrict__ out) {
    eg2_packed_attn_impl<256, 32, 64, 2, 64>(
        q, kc, vc, tileSegStart, tileSegLen, tileQLocal0, nH, nKV, window, scale, totalTiles, out);
}

extern "C" __global__ void eg2_packed_attn_hd512(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const int* __restrict__ tileSegStart, const int* __restrict__ tileSegLen, const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles, float* __restrict__ out) {
    eg2_packed_attn_impl<512, 32, 32, 4, 32>(
        q, kc, vc, tileSegStart, tileSegLen, tileQLocal0, nH, nKV, window, scale, totalTiles, out);
}

extern "C" __global__ void eg2_packed_attn_hd16(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const int* __restrict__ tileSegStart, const int* __restrict__ tileSegLen, const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles, float* __restrict__ out) {
    eg2_packed_attn_impl<16, 32, 32, 1, 16>(
        q, kc, vc, tileSegStart, tileSegLen, tileQLocal0, nH, nKV, window, scale, totalTiles, out);
}

extern "C" __global__ void eg2_packed_attn_hd32(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const int* __restrict__ tileSegStart, const int* __restrict__ tileSegLen, const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles, float* __restrict__ out) {
    eg2_packed_attn_impl<32, 32, 32, 1, 32>(
        q, kc, vc, tileSegStart, tileSegLen, tileQLocal0, nH, nKV, window, scale, totalTiles, out);
}
