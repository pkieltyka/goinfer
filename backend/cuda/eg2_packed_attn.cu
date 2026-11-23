// eg2_packed_attn.cu -- packed variable-length bidirectional fused attention for
// EmbeddingGemma 2 (goinfer #63):
//
//   * Per-segment fixed local query/key tiles: each block owns one (head,
//     segment-local query tile [qLocal0, qLocal0 + BM)) from a host-built tile
//     descriptor table (tileSegStart, tileSegLen, tileQLocal0). Tiles never
//     cross segment boundaries and never read from adjacent segments.
//   * Bit-identical across singleton, split, pack, and reorder on the same
//     device: every key tile boundary (anchored at multiples of BN relative to
//     sequence start), score reduction order, online softmax step, and mask
//     depends only on (segLen, qLocal0, window, BM, BN), with zero dependence
//     on segStart or batch packing.
//   * Direct Q/K/V row-major access with GQA (kvh = h / (nH / nKV)), eliminating
//     per-layer QKV packing copies and the fused [T, 3, nH, HD] buffer.
//   * Bidirectional full (window < 0) and symmetric sliding window (window >= 0,
//     inclusive |qLocal - kLocal| <= window) with 64-bit window endpoint math
//     and -INFINITY tile-max initialization for general float scale support.
//   * Three-term split-FP16 (hi + lo: hi*hi + hi*lo + lo*hi) mma.sync.aligned.m16n8k8
//     tensor-core QK^T and PV with FP32 accumulation and FP32 online softmax
//     (retaining the half subnormal floor); all scalar float MACs use explicit
//     __fmaf_rn / __fmul_rn / __fadd_rn / __fsub_rn intrinsics (FMA-lint clean).
//   * Per-chunk exceptional-V guard: if any staged V element in a union tile is
//     NaN, Inf, or overflows FP16 (|v| >= 65520), it is zero-substituted in
//     VtshHi/Lo so the baseline MMA stays bit-identical on unaffected rows and
//     columns; admitted exceptional cells are patched in FP32 directly from vc
//     using the exact FP32 exp(score - mRun).
//   * Optional per-forward range flag (overflowFlag, nullable): if any active
//     staged finite Q, K, or V value overflows FP16 (|x| >= 65520), the CTA
//     latches atomicOr(overflowFlag, 1u) once at exit so integrated callers can
//     fail closed with a typed numeric-range error.
//   * WARP_D warps cooperate along HD so each warp keeps HD/WARP_D <= 128 output
//     columns in registers (0 bytes local-memory spill) and total shared memory
//     stays under the 48 KiB default limit for HD=256 (43.0 KiB) and HD=512
//     (42.5 KiB).
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

__device__ __forceinline__ bool f16_is_exceptional(__half h) {
    return (__half_as_ushort(h) & 0x7c00u) == 0x7c00u;
}

__device__ __forceinline__ bool f32_finite_overflows_f16(float x, __half hi) {
    return isfinite(x) && f16_is_exceptional(hi);
}

__device__ __forceinline__ bool f32_split_f16(float x, __half &hi, __half &lo) {
    hi = __float2half_rn(x);
    if (f16_is_exceptional(hi)) {
        lo = __float2half_rn(0.f);
        return isfinite(x);
    }
    lo = __float2half_rn(__fsub_rn(x, __half2float(hi)));
    return false;
}

// f32_split_f16_v_safe splits V into (hi, lo), substituting (0, 0) and returning
// true whenever hi would be NaN or Inf (including finite |x| >= 65520). If x was
// finite and overflowed FP16, finiteOverflow is set to 1.
__device__ __forceinline__ bool f32_split_f16_v_safe(float x, __half &hi, __half &lo, int &finiteOverflow) {
    hi = __float2half_rn(x);
    if (f16_is_exceptional(hi)) {
        if (isfinite(x)) finiteOverflow = 1;
        hi = __float2half_rn(0.f);
        lo = __float2half_rn(0.f);
        return true;
    }
    lo = __float2half_rn(__fsub_rn(x, __half2float(hi)));
    return false;
}

__device__ __forceinline__ unsigned int pack_halves(__half a, __half b) {
    __half2 h = __halves2half2(a, b);
    return *reinterpret_cast<unsigned int*>(&h);
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
    float* __restrict__ out,
    unsigned int* __restrict__ overflowFlag) {
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
    constexpr size_t qPlane = (size_t)BM * qStride;
    constexpr size_t kPlane = (size_t)BN * kStride;
    constexpr size_t vPlane = (size_t)DK_TOT * vStride;

    extern __shared__ __half sh[];
    __half* QshHi = sh;                                  // [BM][HD + KPAD]
    __half* QshLo = QshHi + qPlane;                      // [BM][HD + KPAD]
    __half* KVsh = QshLo + qPlane;                       // overlays Ksh, Vtsh, and Ssh
    __half* KshHi = KVsh;                                // [BN][DK_TOT + KPAD]
    __half* KshLo = KshHi + kPlane;                      // [BN][DK_TOT + KPAD]
    __half* VtshHi = KVsh;                               // [DK_TOT][BN + KPAD]
    __half* VtshLo = VtshHi + vPlane;                    // [DK_TOT][BN + KPAD]
    float* Ssh = reinterpret_cast<float*>(KVsh);         // [WARP_M][WARP_D - 1][16][BN] when WARP_D > 1

    const int nqTile = (segLen - qLocal0 < BM) ? (segLen - qLocal0) : BM;
    const bool qVecOK = (reinterpret_cast<unsigned long long>(q) & 15ull) == 0ull;
    const bool kVecOK = (reinterpret_cast<unsigned long long>(kc) & 15ull) == 0ull;
    const bool vVecOK = (reinterpret_cast<unsigned long long>(vc) & 15ull) == 0ull;

    int threadSawOverflow = 0;

    // Stage this block's segment-local Q tile [nqTile, HD] as split-f16 (hi + lo).
    // Rows >= nqTile are zeroed; no reads ever occur past segStart + segLen.
    if (qVecOK) {
        for (int idx = tid; idx < BM * HD_V4; idx += blockDim.x) {
            const int qr = idx / HD_V4;
            const int dv = idx - qr * HD_V4;
            uint2 hiPacked = make_uint2(0u, 0u);
            uint2 loPacked = make_uint2(0u, 0u);
            if (qr < nqTile) {
                f32_vec4 v;
                const float* src = q + (long)(segStart + qLocal0 + qr) * qDim + h * HD + dv * 4;
                v.u = *reinterpret_cast<const uint4*>(src);
                __half h0, l0, h1, l1, h2, l2, h3, l3;
                if (f32_split_f16(v.f[0], h0, l0)) threadSawOverflow = 1;
                if (f32_split_f16(v.f[1], h1, l1)) threadSawOverflow = 1;
                if (f32_split_f16(v.f[2], h2, l2)) threadSawOverflow = 1;
                if (f32_split_f16(v.f[3], h3, l3)) threadSawOverflow = 1;
                hiPacked.x = pack_halves(h0, h1);
                hiPacked.y = pack_halves(h2, h3);
                loPacked.x = pack_halves(l0, l1);
                loPacked.y = pack_halves(l2, l3);
            }
            const int off = qr * qStride + dv * 4;
            *reinterpret_cast<uint2*>(&QshHi[off]) = hiPacked;
            *reinterpret_cast<uint2*>(&QshLo[off]) = loPacked;
        }
    } else {
        for (int idx = tid; idx < BM * HD; idx += blockDim.x) {
            const int qr = idx / HD;
            const int dd = idx - qr * HD;
            float val = 0.f;
            if (qr < nqTile) {
                val = q[(long)(segStart + qLocal0 + qr) * qDim + h * HD + dd];
            }
            __half hi, lo;
            if (f32_split_f16(val, hi, lo)) threadSawOverflow = 1;
            const int off = qr * qStride + dd;
            QshHi[off] = hi;
            QshLo[off] = lo;
        }
    }
    __syncthreads();

    const int qLoc0 = qLocal0 + warpM * 16 + qRow0;
    const int qLoc1 = qLocal0 + warpM * 16 + qRow1;
    const bool has0 = qLoc0 < segLen;
    const bool has1 = qLoc1 < segLen;

    // Segment-local per-row inclusive key bounds [winLo, winHi] using 64-bit math.
    const long wL = (long)window;
    const long segLastL = (long)(segLen - 1);
    const int winLo0 = (window >= 0 && (long)qLoc0 > wL) ? (int)((long)qLoc0 - wL) : 0;
    const int winHi0 = (window >= 0 && (long)qLoc0 + wL < segLastL) ? (int)((long)qLoc0 + wL) : (segLen - 1);
    const int winLo1 = (window >= 0 && (long)qLoc1 > wL) ? (int)((long)qLoc1 - wL) : 0;
    const int winHi1 = (window >= 0 && (long)qLoc1 + wL < segLastL) ? (int)((long)qLoc1 + wL) : (segLen - 1);

    // Segment-local tile key range [kTileStart, kTileEnd): fixed by (segLen, qLocal0, window, BM, BN),
    // with ascending key tiles anchored at multiples of BN relative to sequence start.
    const long tileLastQL = (long)qLocal0 + (long)(nqTile - 1);
    const int rawStart = (window >= 0 && (long)qLocal0 > wL) ? (int)((long)qLocal0 - wL) : 0;
    const int kTileStart = rawStart & ~(BN - 1);
    const int kTileEnd = (window >= 0 && tileLastQL + wL + 1L < (long)segLen) ? (int)(tileLastQL + wL + 1L) : segLen;

    const float negInf = -__int_as_float(0x7f800000);
    float mRun0 = negInf, mRun1 = negInf;
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
                    __half h0, l0, h1, l1, h2, l2, h3, l3;
                    if (f32_split_f16(v.f[0], h0, l0)) threadSawOverflow = 1;
                    if (f32_split_f16(v.f[1], h1, l1)) threadSawOverflow = 1;
                    if (f32_split_f16(v.f[2], h2, l2)) threadSawOverflow = 1;
                    if (f32_split_f16(v.f[3], h3, l3)) threadSawOverflow = 1;
                    uint2 hiPacked = make_uint2(pack_halves(h0, h1), pack_halves(h2, h3));
                    uint2 loPacked = make_uint2(pack_halves(l0, l1), pack_halves(l2, l3));
                    const int off = kk * kStride + dv * 4;
                    *reinterpret_cast<uint2*>(&KshHi[off]) = hiPacked;
                    *reinterpret_cast<uint2*>(&KshLo[off]) = loPacked;
                }
            } else {
                for (int idx = tid; idx < nk * DK_TOT; idx += blockDim.x) {
                    const int kk = idx / DK_TOT;
                    const int dd = idx - kk * DK_TOT;
                    const int wd = dd / DK_SUB;
                    const int dOff = dd - wd * DK_SUB;
                    const int gD = wd * HD_WARP + ch * DK_SUB + dOff;
                    const float kv = kc[(long)(segStart + k0 + kk) * kvDim + kvh * HD + gD];
                    __half hi, lo;
                    if (f32_split_f16(kv, hi, lo)) threadSawOverflow = 1;
                    const int off = kk * kStride + dd;
                    KshHi[off] = hi;
                    KshLo[off] = lo;
                }
            }
            __syncthreads();

#pragma unroll
            for (int cSub = 0; cSub < SUB_TILES; cSub++) {
                const int gD = dBaseWarp + ch * DK_SUB + cSub * 8 + colBase;
                const int qOff0 = (warpM * 16 + qRow0) * qStride + gD;
                const int qOff1 = (warpM * 16 + qRow1) * qStride + gD;
                const unsigned int qaHi0 = *reinterpret_cast<const unsigned int*>(&QshHi[qOff0]);
                const unsigned int qaHi1 = *reinterpret_cast<const unsigned int*>(&QshHi[qOff1]);
                const unsigned int qaLo0 = *reinterpret_cast<const unsigned int*>(&QshLo[qOff0]);
                const unsigned int qaLo1 = *reinterpret_cast<const unsigned int*>(&QshLo[qOff1]);
                const int kD = warpD * DK_SUB + cSub * 8 + colBase;
#pragma unroll
                for (int g = 0; g < NG; g++) {
                    const int kIdx = g * 8 + (lane >> 2);
                    unsigned int kbHi = 0u, kbLo = 0u;
                    if (kIdx < nk) {
                        const int kOff = kIdx * kStride + kD;
                        kbHi = *reinterpret_cast<const unsigned int*>(&KshHi[kOff]);
                        kbLo = *reinterpret_cast<const unsigned int*>(&KshLo[kOff]);
                    }
                    mma_m16n8k8_f32(s[g][0], s[g][1], s[g][2], s[g][3],
                                    qaHi0, qaHi1, kbHi,
                                    s[g][0], s[g][1], s[g][2], s[g][3]);
                    mma_m16n8k8_f32(s[g][0], s[g][1], s[g][2], s[g][3],
                                    qaHi0, qaHi1, kbLo,
                                    s[g][0], s[g][1], s[g][2], s[g][3]);
                    mma_m16n8k8_f32(s[g][0], s[g][1], s[g][2], s[g][3],
                                    qaLo0, qaLo1, kbHi,
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
        float tileMax0 = negInf, tileMax1 = negInf;
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

            const float a0 = v0A ? __fmul_rn(s[g][0], scale) : negInf;
            const float a1 = v0B ? __fmul_rn(s[g][1], scale) : negInf;
            const float b0 = v1A ? __fmul_rn(s[g][2], scale) : negInf;
            const float b1 = v1B ? __fmul_rn(s[g][3], scale) : negInf;
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
        const float alpha0 = hasValid0 ? (hadPrev0 ? __expf(__fsub_rn(mRun0, mNew0)) : 0.f) : 1.f;
        const float alpha1 = hasValid1 ? (hadPrev1 ? __expf(__fsub_rn(mRun1, mNew1)) : 0.f) : 1.f;

        float sum0 = 0.f, sum1 = 0.f;
        unsigned int pFragHi[NG][2];
        unsigned int pFragLo[NG][2];
#pragma unroll
        for (int g = 0; g < NG; g++) {
            const int bitA = g << 1;
            const int bitB = bitA | 1;
            const float p0 = ((validMask0 >> bitA) & 1u) ? __expf(__fsub_rn(s[g][0], mNew0)) : 0.f;
            const float p1 = ((validMask0 >> bitB) & 1u) ? __expf(__fsub_rn(s[g][1], mNew0)) : 0.f;
            const float p2 = ((validMask1 >> bitA) & 1u) ? __expf(__fsub_rn(s[g][2], mNew1)) : 0.f;
            const float p3 = ((validMask1 >> bitB) & 1u) ? __expf(__fsub_rn(s[g][3], mNew1)) : 0.f;
            sum0 = __fadd_rn(sum0, __fadd_rn(p0, p1));
            sum1 = __fadd_rn(sum1, __fadd_rn(p2, p3));
            __half h0, l0, h1, l1, h2, l2, h3, l3;
            f32_split_f16(p0, h0, l0);
            f32_split_f16(p1, h1, l1);
            f32_split_f16(p2, h2, l2);
            f32_split_f16(p3, h3, l3);
            pFragHi[g][0] = pack_halves(h0, h1);
            pFragHi[g][1] = pack_halves(h2, h3);
            pFragLo[g][0] = pack_halves(l0, l1);
            pFragLo[g][1] = pack_halves(l2, l3);
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
            int threadSawExceptional = 0;
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
                    __half h0, l0, h1, l1, h2, l2, h3, l3;
                    if (f32_split_f16_v_safe(v.f[0], h0, l0, threadSawOverflow)) threadSawExceptional = 1;
                    if (f32_split_f16_v_safe(v.f[1], h1, l1, threadSawOverflow)) threadSawExceptional = 1;
                    if (f32_split_f16_v_safe(v.f[2], h2, l2, threadSawOverflow)) threadSawExceptional = 1;
                    if (f32_split_f16_v_safe(v.f[3], h3, l3, threadSawOverflow)) threadSawExceptional = 1;
                    const int dd = dv * 4;
                    VtshHi[(dd + 0) * vStride + kk] = h0;
                    VtshHi[(dd + 1) * vStride + kk] = h1;
                    VtshHi[(dd + 2) * vStride + kk] = h2;
                    VtshHi[(dd + 3) * vStride + kk] = h3;
                    VtshLo[(dd + 0) * vStride + kk] = l0;
                    VtshLo[(dd + 1) * vStride + kk] = l1;
                    VtshLo[(dd + 2) * vStride + kk] = l2;
                    VtshLo[(dd + 3) * vStride + kk] = l3;
                }
            } else {
                for (int idx = tid; idx < nk * DK_TOT; idx += blockDim.x) {
                    const int kk = idx / DK_TOT;
                    const int dd = idx - kk * DK_TOT;
                    const int wd = dd / DK_SUB;
                    const int dOff = dd - wd * DK_SUB;
                    const int gD = wd * HD_WARP + ch * DK_SUB + dOff;
                    const float vv = vc[(long)(segStart + k0 + kk) * kvDim + kvh * HD + gD];
                    __half hv, lv;
                    if (f32_split_f16_v_safe(vv, hv, lv, threadSawOverflow)) threadSawExceptional = 1;
                    VtshHi[dd * vStride + kk] = hv;
                    VtshLo[dd * vStride + kk] = lv;
                }
            }
            const int tileSawExceptional = __syncthreads_or(threadSawExceptional);

            // Always execute the exact baseline MMA on finite/zero-substituted Vtsh so
            // unaffected rows and columns preserve bit-identical results even when a
            // masked or sibling key carries NaN, Inf, or FP16-overflowing V.
#pragma unroll
            for (int g = 0; g < NG; g++) {
                const unsigned int pHi0 = pFragHi[g][0];
                const unsigned int pHi1 = pFragHi[g][1];
                const unsigned int pLo0 = pFragLo[g][0];
                const unsigned int pLo1 = pFragLo[g][1];
                const int kIdx = g * 8 + colBase;
#pragma unroll
                for (int tSub = 0; tSub < SUB_TILES; tSub++) {
                    const int t = ch * SUB_TILES + tSub;
                    const int dIdx = warpD * DK_SUB + tSub * 8 + (lane >> 2);
                    unsigned int vbHi = 0u, vbLo = 0u;
                    if (kIdx + 1 < nk) {
                        const int vOff = dIdx * vStride + kIdx;
                        vbHi = *reinterpret_cast<const unsigned int*>(&VtshHi[vOff]);
                        vbLo = *reinterpret_cast<const unsigned int*>(&VtshLo[vOff]);
                    } else if (kIdx < nk) {
                        const int vOff = dIdx * vStride + kIdx;
                        vbHi = pack_halves(VtshHi[vOff], __float2half_rn(0.f));
                        vbLo = pack_halves(VtshLo[vOff], __float2half_rn(0.f));
                    }
                    mma_m16n8k8_f32(acc[t][0], acc[t][1], acc[t][2], acc[t][3],
                                    pHi0, pHi1, vbHi,
                                    acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
                    mma_m16n8k8_f32(acc[t][0], acc[t][1], acc[t][2], acc[t][3],
                                    pHi0, pHi1, vbLo,
                                    acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
                    mma_m16n8k8_f32(acc[t][0], acc[t][1], acc[t][2], acc[t][3],
                                    pLo0, pLo1, vbHi,
                                    acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
                }
            }

            if (tileSawExceptional != 0) {
                // Patch ONLY the exact (admitted row, exceptional V[k, d]) cells in FP32
                // using the exact FP32 exp(score - mRun) (preserving tiny weights such as
                // exp(-20) * 1e20 that underflow both half P limbs). Masked rows and finite
                // columns are untouched and remain bit-identical to baseline.
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
                        const float s0A = __shfl_sync(0xffffffffu, s[g][0], srcLane);
                        const float s0B = __shfl_sync(0xffffffffu, s[g][1], srcLane);
                        const float s1A = __shfl_sync(0xffffffffu, s[g][2], srcLane);
                        const float s1B = __shfl_sync(0xffffffffu, s[g][3], srcLane);
                        const bool v0A = ((m0 >> bitA) & 1u) != 0u;
                        const bool v0B = ((m0 >> bitB) & 1u) != 0u;
                        const bool v1A = ((m1 >> bitA) & 1u) != 0u;
                        const bool v1B = ((m1 >> bitB) & 1u) != 0u;
                        const float p0A = v0A ? __expf(__fsub_rn(s0A, mRun0)) : 0.f;
                        const float p0B = v0B ? __expf(__fsub_rn(s0B, mRun0)) : 0.f;
                        const float p1A = v1A ? __expf(__fsub_rn(s1A, mRun1)) : 0.f;
                        const float p1B = v1B ? __expf(__fsub_rn(s1B, mRun1)) : 0.f;
                        const int kA = g * 8 + (srcCol << 1);
                        const int kB = kA + 1;
#pragma unroll
                        for (int tSub = 0; tSub < SUB_TILES; tSub++) {
                            const int t = ch * SUB_TILES + tSub;
                            const int dcol = dBaseWarp + t * 8 + colBase;
                            if (v0A) {
                                const float* vr = vc + (long)(segStart + k0 + kA) * kvDim + kvh * HD + dcol;
                                if (f16_is_exceptional(__float2half_rn(vr[0]))) acc[t][0] = __fmaf_rn(p0A, vr[0], acc[t][0]);
                                if (f16_is_exceptional(__float2half_rn(vr[1]))) acc[t][1] = __fmaf_rn(p0A, vr[1], acc[t][1]);
                            }
                            if (v0B) {
                                const float* vr = vc + (long)(segStart + k0 + kB) * kvDim + kvh * HD + dcol;
                                if (f16_is_exceptional(__float2half_rn(vr[0]))) acc[t][0] = __fmaf_rn(p0B, vr[0], acc[t][0]);
                                if (f16_is_exceptional(__float2half_rn(vr[1]))) acc[t][1] = __fmaf_rn(p0B, vr[1], acc[t][1]);
                            }
                            if (v1A) {
                                const float* vr = vc + (long)(segStart + k0 + kA) * kvDim + kvh * HD + dcol;
                                if (f16_is_exceptional(__float2half_rn(vr[0]))) acc[t][2] = __fmaf_rn(p1A, vr[0], acc[t][2]);
                                if (f16_is_exceptional(__float2half_rn(vr[1]))) acc[t][3] = __fmaf_rn(p1A, vr[1], acc[t][3]);
                            }
                            if (v1B) {
                                const float* vr = vc + (long)(segStart + k0 + kB) * kvDim + kvh * HD + dcol;
                                if (f16_is_exceptional(__float2half_rn(vr[0]))) acc[t][2] = __fmaf_rn(p1B, vr[0], acc[t][2]);
                                if (f16_is_exceptional(__float2half_rn(vr[1]))) acc[t][3] = __fmaf_rn(p1B, vr[1], acc[t][3]);
                            }
                        }
                    }
                }
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
    if (overflowFlag != nullptr && __syncthreads_or(threadSawOverflow) && tid == 0) {
        atomicOr(overflowFlag, 1u);
    }
}

extern "C" __global__ void eg2_packed_attn_hd256(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const int* __restrict__ tileSegStart, const int* __restrict__ tileSegLen, const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles, float* __restrict__ out,
    unsigned int* __restrict__ overflowFlag) {
    eg2_packed_attn_impl<256, 32, 32, 2, 32>(
        q, kc, vc, tileSegStart, tileSegLen, tileQLocal0, nH, nKV, window, scale, totalTiles, out, overflowFlag);
}

extern "C" __global__ void eg2_packed_attn_hd512(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const int* __restrict__ tileSegStart, const int* __restrict__ tileSegLen, const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles, float* __restrict__ out,
    unsigned int* __restrict__ overflowFlag) {
    eg2_packed_attn_impl<512, 16, 32, 4, 16>(
        q, kc, vc, tileSegStart, tileSegLen, tileQLocal0, nH, nKV, window, scale, totalTiles, out, overflowFlag);
}

extern "C" __global__ void eg2_packed_attn_hd16(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const int* __restrict__ tileSegStart, const int* __restrict__ tileSegLen, const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles, float* __restrict__ out,
    unsigned int* __restrict__ overflowFlag) {
    eg2_packed_attn_impl<16, 32, 32, 1, 16>(
        q, kc, vc, tileSegStart, tileSegLen, tileQLocal0, nH, nKV, window, scale, totalTiles, out, overflowFlag);
}

extern "C" __global__ void eg2_packed_attn_hd32(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const int* __restrict__ tileSegStart, const int* __restrict__ tileSegLen, const int* __restrict__ tileQLocal0,
    int nH, int nKV, int window, float scale, int totalTiles, float* __restrict__ out,
    unsigned int* __restrict__ overflowFlag) {
    eg2_packed_attn_impl<32, 32, 32, 1, 32>(
        q, kc, vc, tileSegStart, tileSegLen, tileQLocal0, nH, nKV, window, scale, totalTiles, out, overflowFlag);
}
