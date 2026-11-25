//go:build darwin

package metal

const eg2FFNStagedMSL = `
#include <metal_stdlib>
using namespace metal;

template<uint NC>
inline void eg2_ffn_store(thread simdgroup_float8x8 (&acc)[2][NC], device float* C,
    threadgroup float* edge, uint M, uint N, uint rm, uint cn, ushort sg, ushort lane) {
    for (uint i = 0; i < 2; ++i) {
        const uint row = rm + i * 8;
        for (uint j = 0; j < NC; ++j) {
            simdgroup_float8x8 f = acc[i][j];
            for (uint x = 0; x < 2; ++x) {
                f.thread_elements()[x] = f.thread_elements()[x] + 0.0f;
            }
            if (row + 8 <= M) {
                simdgroup_store(f, C + row * N + cn + j * 8, N);
            } else if (row < M) {
                threadgroup float* tmp = edge + uint(sg) * 64;
                simdgroup_store(f, tmp, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint cell = lane; cell < (M - row) * 8; cell += 32)
                    C[(row + cell / 8) * N + cn + j * 8 + cell % 8] = tmp[cell];
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

kernel void eg2_ffn_gate_up(device const float* A [[buffer(0)]], device const float* W [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint ld = 36;
    threadgroup float staging[(32 + 64) * 36];
    threadgroup float* as = staging;
    threadgroup float* bs = staging + 32 * 36;
    const uint m0 = tg.y * 32, n0 = tg.x * 64;
    const uint sr = uint(sg / 2) * 16, sc = uint(sg % 2) * 32;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 4; ++j)
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < K; k += 32) {
        const uint k4 = uint(tid % 8) * 4;
        for (uint r = tid / 8; r < 32; r += 16) {
            float4 x(0.0f);
            if (m0 + r < M) x = *(device const float4*)(A + (m0 + r) * K + k + k4);
            *(threadgroup float4*)(as + r * ld + k4) = x;
        }
        for (uint r = tid / 8; r < 64; r += 16) {
            float4 x = *(device const float4*)(W + (n0 + r) * K + k + k4);
            *(threadgroup float4*)(bs + r * ld + k4) = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a[2], b[4];
            for (uint i = 0; i < 2; ++i)
                simdgroup_load(a[i], as + (sr + i * 8) * ld + kk, ld);
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(b[j], bs + (sc + j * 8) * ld + kk, ld, ulong2(0, 0), true);
            for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    eg2_ffn_store<4>(acc, C, staging, M, N, m0 + sr, n0 + sc, sg, lane);
}
`
