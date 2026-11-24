//go:build darwin

package metal

// FP32 matrix attention for the full and sliding text-layer layouts.
const eg2MatrixAttnExtraMSL = `
#define MA_BQ 8u
#define MA_BK 16u
#define MA_HD 512u
#define MA_TAIL_LD 520u

kernel void eg2_matrix_attn_hd512(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& T [[buffer(4)]], constant uint& nH [[buffer(5)]],
    constant uint& nKV [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float tail[MA_BQ * MA_TAIL_LD]; // 8 * 520 * 4 = 16,640 B
    threadgroup float partial[4u * 2u * 64u];   // 4 * 2 * 64 * 4 = 2,048 B

    const uint r0 = tg.x * MA_BQ;
    const uint h = tg.y;
    const uint group = nH / nKV;
    const uint g = h / group;
    const uint qStride = nH * MA_HD;
    const uint kvStride = nKV * MA_HD;
    const uint c0 = uint(sgid) * 128u;

    // Apple Silicon simdgroup_float8x8 lane map (matches tower_attn_body & attention_prefill_steel_split).
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;

    // 1. Load 16 persistent Q 8x8 FP32 fragments for this SIMD group's 128 channels.
    simdgroup_float8x8 qT[16];
    if (r0 + MA_BQ <= T) {
        device const float* qBase = q + r0 * qStride + h * MA_HD + c0;
        #pragma unroll
        for (ushort kk = 0; kk < 16; kk++) {
            simdgroup_load(qT[kk], qBase + uint(kk) * 8u, qStride);
        }
    } else {
        for (uint idx = uint(tid); idx < MA_BQ * (MA_HD / 4u); idx += 128u) {
            const uint r = idx >> 7;
            const uint d4 = (idx & 127u) * 4u;
            const uint row = r0 + r;
            *(threadgroup float4*)(tail + r * MA_TAIL_LD + d4) =
                (row < T) ? *(device const float4*)(q + row * qStride + h * MA_HD + d4) : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (ushort kk = 0; kk < 16; kk++) {
            simdgroup_load(qT[kk], tail + c0 + uint(kk) * 8u, MA_TAIL_LD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 2. Initialize 16 persistent O 8x8 FP32 accumulators and running softmax state.
    simdgroup_float8x8 oAcc[16];
    #pragma unroll
    for (ushort cc = 0; cc < 16; cc++) {
        oAcc[cc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float mRow = -INFINITY;
    float lRow = 0.0f;

    // 3. Stream keys in 16-key blocks (two 8-key subtiles).
    for (uint j0 = 0u; j0 < T; j0 += MA_BK) {
        const bool sub0Full = (j0 + 8u <= T);
        const bool sub1Full = (j0 + MA_BK <= T);

        simdgroup_float8x8 S0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 S1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        if (sub1Full) {
            // Regular 16-key tile: direct device simdgroup_load (K transposed), zero tail staging.
            device const float* kBase0 = k + j0 * kvStride + g * MA_HD + c0;
            device const float* kBase1 = kBase0 + 8u * kvStride;
            #pragma unroll
            for (ushort kk = 0; kk < 16; kk++) {
                simdgroup_float8x8 kTile0, kTile1;
                simdgroup_load(kTile0, kBase0 + uint(kk) * 8u, kvStride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S0, qT[kk], kTile0, S0);
                simdgroup_load(kTile1, kBase1 + uint(kk) * 8u, kvStride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S1, qT[kk], kTile1, S1);
            }
        } else if (sub0Full) {
            // Tail tile with complete sub=0 and partial sub=1 (e.g. T=2042: keys 2032..2039 direct, 2040..2041 via tail).
            device const float* kBase0 = k + j0 * kvStride + g * MA_HD + c0;
            #pragma unroll
            for (ushort kk = 0; kk < 16; kk++) {
                simdgroup_float8x8 kTile0;
                simdgroup_load(kTile0, kBase0 + uint(kk) * 8u, kvStride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S0, qT[kk], kTile0, S0);
            }
            const uint j1 = j0 + 8u;
            for (uint idx = uint(tid); idx < 8u * (MA_HD / 4u); idx += 128u) {
                const uint r = idx >> 7;
                const uint d4 = (idx & 127u) * 4u;
                const uint j = j1 + r;
                *(threadgroup float4*)(tail + r * MA_TAIL_LD + d4) =
                    (j < T) ? *(device const float4*)(k + j * kvStride + g * MA_HD + d4) : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (ushort kk = 0; kk < 16; kk++) {
                simdgroup_float8x8 kTile1;
                simdgroup_load(kTile1, tail + c0 + uint(kk) * 8u, MA_TAIL_LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S1, qT[kk], kTile1, S1);
            }
        } else {
            // Tail tile with partial sub=0 (1..7 keys left); sub=1 is wholly past T and masked below.
            for (uint idx = uint(tid); idx < 8u * (MA_HD / 4u); idx += 128u) {
                const uint r = idx >> 7;
                const uint d4 = (idx & 127u) * 4u;
                const uint j = j0 + r;
                *(threadgroup float4*)(tail + r * MA_TAIL_LD + d4) =
                    (j < T) ? *(device const float4*)(k + j * kvStride + g * MA_HD + d4) : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (ushort kk = 0; kk < 16; kk++) {
                simdgroup_float8x8 kTile0;
                simdgroup_load(kTile0, tail + c0 + uint(kk) * 8u, MA_TAIL_LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S0, qT[kk], kTile0, S0);
            }
        }

        // Barrier 1: publish partial scores AND synchronize all tail[] K reads before V tail reuse.
        simdgroup_store(S0, partial + (uint(sgid) * 2u + 0u) * 64u, 8u);
        simdgroup_store(S1, partial + (uint(sgid) * 2u + 1u) * 64u, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Sum the four channel slices in fixed ascending order (0 -> 1 -> 2 -> 3).
        float s[4];
        {
            threadgroup const float* p0 = partial + 0u * 64u + fm * 8u + fn;
            float e0 = p0[0], e1 = p0[1];
            e0 += p0[128u]; e1 += p0[128u + 1u];
            e0 += p0[256u]; e1 += p0[256u + 1u];
            e0 += p0[384u]; e1 += p0[384u + 1u];
            s[0] = e0;
            s[1] = e1;
        }
        {
            threadgroup const float* p1 = partial + 1u * 64u + fm * 8u + fn;
            float e0 = p1[0], e1 = p1[1];
            e0 += p1[128u]; e1 += p1[128u + 1u];
            e0 += p1[256u]; e1 += p1[256u + 1u];
            e0 += p1[384u]; e1 += p1[384u + 1u];
            s[2] = e0;
            s[3] = e1;
        }

        if (!sub1Full) {
            const uint js[4] = { j0 + fn, j0 + fn + 1u, j0 + 8u + fn, j0 + 9u + fn };
            #pragma unroll
            for (ushort e = 0; e < 4; e++) {
                if (js[e] >= T) {
                    s[e] = -INFINITY;
                }
            }
        }

        float bmax = max(max(s[0], s[1]), max(s[2], s[3]));
        bmax = max(bmax, simd_shuffle_xor(bmax, 1));
        bmax = max(bmax, simd_shuffle_xor(bmax, 8));

        float factor = 1.0f;
        float p[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (bmax > -INFINITY) {
            const float mNew = max(mRow, bmax);
            if (mRow > -INFINITY) {
                factor = (mNew > mRow) ? precise::exp(mRow - mNew) : 1.0f;
            } else {
                factor = 0.0f;
            }
            #pragma unroll
            for (ushort e = 0; e < 4; e++) {
                p[e] = (s[e] > -INFINITY) ? precise::exp(s[e] - mNew) : 0.0f;
            }
            mRow = mNew;
        }

        float bsum = (p[0] + p[1]) + (p[2] + p[3]);
        bsum += simd_shuffle_xor(bsum, 1);
        bsum += simd_shuffle_xor(bsum, 8);
        lRow = lRow * factor + bsum;

        simdgroup_float8x8 P0, P1;
        P0.thread_elements()[0] = p[0];
        P0.thread_elements()[1] = p[1];
        P1.thread_elements()[0] = p[2];
        P1.thread_elements()[1] = p[3];

        if (sub1Full) {
            device const float* vBase0 = v + j0 * kvStride + g * MA_HD + c0;
            device const float* vBase1 = vBase0 + 8u * kvStride;
            #pragma unroll
            for (ushort cc = 0; cc < 16; cc++) {
                oAcc[cc].thread_elements()[0] *= factor;
                oAcc[cc].thread_elements()[1] *= factor;
                simdgroup_float8x8 vTile0, vTile1;
                simdgroup_load(vTile0, vBase0 + uint(cc) * 8u, kvStride);
                simdgroup_multiply_accumulate(oAcc[cc], P0, vTile0, oAcc[cc]);
                simdgroup_load(vTile1, vBase1 + uint(cc) * 8u, kvStride);
                simdgroup_multiply_accumulate(oAcc[cc], P1, vTile1, oAcc[cc]);
            }
        } else if (sub0Full) {
            const uint j1 = j0 + 8u;
            for (uint idx = uint(tid); idx < 8u * (MA_HD / 4u); idx += 128u) {
                const uint r = idx >> 7;
                const uint d4 = (idx & 127u) * 4u;
                const uint j = j1 + r;
                *(threadgroup float4*)(tail + r * MA_TAIL_LD + d4) =
                    (j < T) ? *(device const float4*)(v + j * kvStride + g * MA_HD + d4) : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            device const float* vBase0 = v + j0 * kvStride + g * MA_HD + c0;
            #pragma unroll
            for (ushort cc = 0; cc < 16; cc++) {
                oAcc[cc].thread_elements()[0] *= factor;
                oAcc[cc].thread_elements()[1] *= factor;
                simdgroup_float8x8 vTile0, vTile1;
                simdgroup_load(vTile0, vBase0 + uint(cc) * 8u, kvStride);
                simdgroup_multiply_accumulate(oAcc[cc], P0, vTile0, oAcc[cc]);
                simdgroup_load(vTile1, tail + c0 + uint(cc) * 8u, MA_TAIL_LD);
                simdgroup_multiply_accumulate(oAcc[cc], P1, vTile1, oAcc[cc]);
            }
        } else {
            for (uint idx = uint(tid); idx < 8u * (MA_HD / 4u); idx += 128u) {
                const uint r = idx >> 7;
                const uint d4 = (idx & 127u) * 4u;
                const uint j = j0 + r;
                *(threadgroup float4*)(tail + r * MA_TAIL_LD + d4) =
                    (j < T) ? *(device const float4*)(v + j * kvStride + g * MA_HD + d4) : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (ushort cc = 0; cc < 16; cc++) {
                oAcc[cc].thread_elements()[0] *= factor;
                oAcc[cc].thread_elements()[1] *= factor;
                simdgroup_float8x8 vTile0;
                simdgroup_load(vTile0, tail + c0 + uint(cc) * 8u, MA_TAIL_LD);
                simdgroup_multiply_accumulate(oAcc[cc], P0, vTile0, oAcc[cc]);
            }
        }

        // Barrier 2: fence partial[] reads before next tile overwrites partial[].
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Final divide and guarded token-major write to this SIMD group's 128-channel slice of ctx.
    const uint myRow = r0 + uint(fm);
    if (myRow < T) {
        const float inv = lRow > 0.0f ? (1.0f / lRow) : 0.0f;
        device float* oDst = out + myRow * qStride + h * MA_HD + c0 + uint(fn);
        #pragma unroll
        for (ushort cc = 0; cc < 16; cc++) {
            oDst[uint(cc) * 8u] = oAcc[cc].thread_elements()[0] * inv;
            oDst[uint(cc) * 8u + 1u] = oAcc[cc].thread_elements()[1] * inv;
        }
    }
}
`

const eg2MatrixSlidingExtraMSL = `
#define MS_BQ 16u
#define MS_BK 16u
#define MS_HD 256u
#define MS_TAIL_LD 264u

kernel void eg2_matrix_sliding_hd256(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& T [[buffer(4)]], constant uint& nH [[buffer(5)]],
    constant uint& nKV [[buffer(6)]], constant uint& W [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float tail[8u * MS_TAIL_LD]; // 8 * 264 * 4 = 8,448 B
    threadgroup float partial[4u * 2u * 64u]; // 4 * 2 * 64 * 4 = 2,048 B

    const uint q0 = tg.x * MS_BQ;
    const uint h = tg.y;
    const uint group = nH / nKV;
    const uint g = h / group;
    const uint qStride = nH * MS_HD;
    const uint kvStride = nKV * MS_HD;
    const uint kvOff = g * MS_HD;

    const ushort rg = sgid >> 1; // 0 or 1 (8 query rows each)
    const ushort sl = sgid & 1;  // 0 or 1 (128 channels each)
    const uint r0 = q0 + uint(rg) * 8u;
    const uint c0 = uint(sl) * 128u;
    const bool rgActive = (r0 < T);

    // Apple Silicon simdgroup_float8x8 lane map.
    const ushort qid = lane >> 2;
    const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
    const uint myRow = r0 + uint(fm);

    // 1. Load 16 persistent Q 8x8 FP32 fragments for this SIMD group's 8 rows x 128 channels.
    simdgroup_float8x8 qT[16];
    if (q0 + MS_BQ <= T) {
        device const float* qBase = q + r0 * qStride + h * MS_HD + c0;
        #pragma unroll
        for (ushort kk = 0; kk < 16; kk++) {
            simdgroup_load(qT[kk], qBase + uint(kk) * 8u, qStride);
        }
    } else if (q0 + 8u <= T) {
        // rg=0 is complete (direct device load); rg=1 is either partial (q0+8 < T) or wholly inactive (q0+8 == T).
        if (rg == 0) {
            device const float* qBase = q + r0 * qStride + h * MS_HD + c0;
            #pragma unroll
            for (ushort kk = 0; kk < 16; kk++) {
                simdgroup_load(qT[kk], qBase + uint(kk) * 8u, qStride);
            }
        }
        if (q0 + 8u < T) {
            const uint r1 = q0 + 8u;
            for (uint idx = uint(tid); idx < 8u * (MS_HD / 4u); idx += 128u) {
                const uint r = idx >> 6;
                const uint d4 = (idx & 63u) * 4u;
                const uint row = r1 + r;
                *(threadgroup float4*)(tail + r * MS_TAIL_LD + d4) =
                    (row < T) ? *(device const float4*)(q + row * qStride + h * MS_HD + d4) : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rg == 1) {
                #pragma unroll
                for (ushort kk = 0; kk < 16; kk++) {
                    simdgroup_load(qT[kk], tail + c0 + uint(kk) * 8u, MS_TAIL_LD);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // rg=0 is partial (1..7 rows); rg=1 is wholly inactive (rgActive == false).
        for (uint idx = uint(tid); idx < 8u * (MS_HD / 4u); idx += 128u) {
            const uint r = idx >> 6;
            const uint d4 = (idx & 63u) * 4u;
            const uint row = q0 + r;
            *(threadgroup float4*)(tail + r * MS_TAIL_LD + d4) =
                (row < T) ? *(device const float4*)(q + row * qStride + h * MS_HD + d4) : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0) {
            #pragma unroll
            for (ushort kk = 0; kk < 16; kk++) {
                simdgroup_load(qT[kk], tail + c0 + uint(kk) * 8u, MS_TAIL_LD);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 2. Initialize 16 persistent O 8x8 FP32 accumulators and running softmax state.
    simdgroup_float8x8 oAcc[16];
    #pragma unroll
    for (ushort cc = 0; cc < 16; cc++) {
        oAcc[cc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float mRow = -INFINITY;
    float lRow = 0.0f;

    // 3. TG union sliding-window key bounds: [floor(max(0, q0 - W)/16)*16, min(T, min(T, q0 + 16) + W)).
    const uint qEnd = min(T, q0 + MS_BQ);
    const uint kStart = ((W > 0u && q0 > W) ? (q0 - W) : 0u) & ~15u;
    const uint kEnd = (W > 0u) ? min(T, qEnd + W) : T;

    for (uint j0 = kStart; j0 < kEnd; j0 += MS_BK) {
        const bool sub0Full = (j0 + 8u <= T);
        const bool sub1Full = (j0 + MS_BK <= T);

        // --- Stage 1: Partial Q·K^T ---
        if (sub1Full) {
            if (rgActive) {
                simdgroup_float8x8 S0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 S1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                device const float* kBase0 = k + j0 * kvStride + kvOff + c0;
                device const float* kBase1 = kBase0 + 8u * kvStride;
                #pragma unroll
                for (ushort kk = 0; kk < 16; kk++) {
                    simdgroup_float8x8 kTile0, kTile1;
                    simdgroup_load(kTile0, kBase0 + uint(kk) * 8u, kvStride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S0, qT[kk], kTile0, S0);
                    simdgroup_load(kTile1, kBase1 + uint(kk) * 8u, kvStride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S1, qT[kk], kTile1, S1);
                }
                simdgroup_store(S0, partial + (uint(sgid) * 2u + 0u) * 64u, 8u);
                simdgroup_store(S1, partial + (uint(sgid) * 2u + 1u) * 64u, 8u);
            }
        } else if (sub0Full) {
            const uint j1 = j0 + 8u;
            for (uint idx = uint(tid); idx < 8u * (MS_HD / 4u); idx += 128u) {
                const uint r = idx >> 6;
                const uint d4 = (idx & 63u) * 4u;
                const uint j = j1 + r;
                *(threadgroup float4*)(tail + r * MS_TAIL_LD + d4) =
                    (j < T) ? *(device const float4*)(k + j * kvStride + kvOff + d4) : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rgActive) {
                simdgroup_float8x8 S0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 S1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                device const float* kBase0 = k + j0 * kvStride + kvOff + c0;
                #pragma unroll
                for (ushort kk = 0; kk < 16; kk++) {
                    simdgroup_float8x8 kTile0, kTile1;
                    simdgroup_load(kTile0, kBase0 + uint(kk) * 8u, kvStride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S0, qT[kk], kTile0, S0);
                    simdgroup_load(kTile1, tail + c0 + uint(kk) * 8u, MS_TAIL_LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S1, qT[kk], kTile1, S1);
                }
                simdgroup_store(S0, partial + (uint(sgid) * 2u + 0u) * 64u, 8u);
                simdgroup_store(S1, partial + (uint(sgid) * 2u + 1u) * 64u, 8u);
            }
        } else {
            for (uint idx = uint(tid); idx < 8u * (MS_HD / 4u); idx += 128u) {
                const uint r = idx >> 6;
                const uint d4 = (idx & 63u) * 4u;
                const uint j = j0 + r;
                *(threadgroup float4*)(tail + r * MS_TAIL_LD + d4) =
                    (j < T) ? *(device const float4*)(k + j * kvStride + kvOff + d4) : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rgActive) {
                simdgroup_float8x8 S0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 S1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma unroll
                for (ushort kk = 0; kk < 16; kk++) {
                    simdgroup_float8x8 kTile0;
                    simdgroup_load(kTile0, tail + c0 + uint(kk) * 8u, MS_TAIL_LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S0, qT[kk], kTile0, S0);
                }
                simdgroup_store(S0, partial + (uint(sgid) * 2u + 0u) * 64u, 8u);
                simdgroup_store(S1, partial + (uint(sgid) * 2u + 1u) * 64u, 8u);
            }
        }

        // Barrier 1: publish partial scores AND synchronize all tail[] K reads before V tail staging.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // If tail key tile, cooperatively stage partial V subtile into tail[] across all 128 threads.
        if (!sub1Full) {
            const uint jBase = sub0Full ? (j0 + 8u) : j0;
            for (uint idx = uint(tid); idx < 8u * (MS_HD / 4u); idx += 128u) {
                const uint r = idx >> 6;
                const uint d4 = (idx & 63u) * 4u;
                const uint j = jBase + r;
                *(threadgroup float4*)(tail + r * MS_TAIL_LD + d4) =
                    (j < T) ? *(device const float4*)(v + j * kvStride + kvOff + d4) : float4(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- Stage 2: Sum slices 0 -> 1 within row group rg, exact per-row inclusive mask, online softmax, and P·V ---
        if (rgActive) {
            threadgroup const float* p0 = partial + (uint(rg) * 4u + 0u) * 64u + fm * 8u + fn;
            threadgroup const float* p1 = partial + (uint(rg) * 4u + 1u) * 64u + fm * 8u + fn;
            float s[4];
            s[0] = p0[0] + p0[128u];
            s[1] = p0[1] + p0[128u + 1u];
            s[2] = p1[0] + p1[128u];
            s[3] = p1[1] + p1[128u + 1u];

            const uint js[4] = { j0 + uint(fn), j0 + uint(fn) + 1u, j0 + 8u + uint(fn), j0 + 9u + uint(fn) };
            #pragma unroll
            for (ushort e = 0; e < 4; e++) {
                const uint jj = js[e];
                if (myRow >= T || jj >= T || (W > 0u && (jj + W < myRow || myRow + W < jj))) {
                    s[e] = -INFINITY;
                }
            }

            float bmax = max(max(s[0], s[1]), max(s[2], s[3]));
            bmax = max(bmax, simd_shuffle_xor(bmax, 1));
            bmax = max(bmax, simd_shuffle_xor(bmax, 8));

            float factor = 1.0f;
            float p[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (bmax > -INFINITY) {
                const float mNew = max(mRow, bmax);
                if (mRow > -INFINITY) {
                    factor = (mNew > mRow) ? precise::exp(mRow - mNew) : 1.0f;
                } else {
                    factor = 0.0f;
                }
                #pragma unroll
                for (ushort e = 0; e < 4; e++) {
                    p[e] = (s[e] > -INFINITY) ? precise::exp(s[e] - mNew) : 0.0f;
                }
                mRow = mNew;
            }

            float bsum = (p[0] + p[1]) + (p[2] + p[3]);
            bsum += simd_shuffle_xor(bsum, 1);
            bsum += simd_shuffle_xor(bsum, 8);
            lRow = lRow * factor + bsum;

            simdgroup_float8x8 P0, P1;
            P0.thread_elements()[0] = p[0];
            P0.thread_elements()[1] = p[1];
            P1.thread_elements()[0] = p[2];
            P1.thread_elements()[1] = p[3];

            if (sub1Full) {
                device const float* vBase0 = v + j0 * kvStride + kvOff + c0;
                device const float* vBase1 = vBase0 + 8u * kvStride;
                #pragma unroll
                for (ushort cc = 0; cc < 16; cc++) {
                    oAcc[cc].thread_elements()[0] *= factor;
                    oAcc[cc].thread_elements()[1] *= factor;
                    simdgroup_float8x8 vTile0, vTile1;
                    simdgroup_load(vTile0, vBase0 + uint(cc) * 8u, kvStride);
                    simdgroup_multiply_accumulate(oAcc[cc], P0, vTile0, oAcc[cc]);
                    simdgroup_load(vTile1, vBase1 + uint(cc) * 8u, kvStride);
                    simdgroup_multiply_accumulate(oAcc[cc], P1, vTile1, oAcc[cc]);
                }
            } else if (sub0Full) {
                device const float* vBase0 = v + j0 * kvStride + kvOff + c0;
                #pragma unroll
                for (ushort cc = 0; cc < 16; cc++) {
                    oAcc[cc].thread_elements()[0] *= factor;
                    oAcc[cc].thread_elements()[1] *= factor;
                    simdgroup_float8x8 vTile0, vTile1;
                    simdgroup_load(vTile0, vBase0 + uint(cc) * 8u, kvStride);
                    simdgroup_multiply_accumulate(oAcc[cc], P0, vTile0, oAcc[cc]);
                    simdgroup_load(vTile1, tail + c0 + uint(cc) * 8u, MS_TAIL_LD);
                    simdgroup_multiply_accumulate(oAcc[cc], P1, vTile1, oAcc[cc]);
                }
            } else {
                #pragma unroll
                for (ushort cc = 0; cc < 16; cc++) {
                    oAcc[cc].thread_elements()[0] *= factor;
                    oAcc[cc].thread_elements()[1] *= factor;
                    simdgroup_float8x8 vTile0;
                    simdgroup_load(vTile0, tail + c0 + uint(cc) * 8u, MS_TAIL_LD);
                    simdgroup_multiply_accumulate(oAcc[cc], P0, vTile0, oAcc[cc]);
                }
            }
        }

        // Barrier 2: fence partial[] and tail[] reads before next tile overwrites them.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Final divide and guarded token-major write to this SIMD group's 128-channel slice of ctx.
    if (myRow < T) {
        const float inv = lRow > 0.0f ? (1.0f / lRow) : 0.0f;
        device float* oDst = out + myRow * qStride + h * MS_HD + c0 + uint(fn);
        #pragma unroll
        for (ushort cc = 0; cc < 16; cc++) {
            oDst[uint(cc) * 8u] = oAcc[cc].thread_elements()[0] * inv;
            oDst[uint(cc) * 8u + 1u] = oAcc[cc].thread_elements()[1] * inv;
        }
    }
}
`
