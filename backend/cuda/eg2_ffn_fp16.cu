// FP32 X[M,K] times FP16 W[N,K] transposed, with FP16 RNE inputs to MMA
// and FP32 Y[M,N]. K is a multiple of 16. A CTA owns 32 rows by 64 columns.
// Each output has the same ascending K16 instruction sequence in any row tile.
// Lane fragment layout follows NVIDIA PTX ISA "mma.m16n8k16 with floating point type".

namespace {
constexpr int rows = 32;
constexpr int cols = 64;
constexpr int depth = 64;
constexpr int pairs = depth / 2;
constexpr int pitch = pairs + 2;
constexpr int threads = 128;

__device__ __forceinline__ unsigned pack_pair(float low, float high) {
    unsigned result;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(high), "f"(low));
    return result;
}

__device__ __forceinline__ void mma(float (&v)[4], unsigned a0, unsigned a1,
                                   unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
} // namespace

extern "C" __global__ __launch_bounds__(threads)
void eg2_ffn_fp16(const float *x, const unsigned *w, float *out, unsigned *invalid_input, int m, int n, int k) {
    __shared__ unsigned a[rows][pitch];
    __shared__ unsigned b[cols][pitch];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int group = lane / 4;
    const int lane_pair = lane % 4;
    const int row_base = blockIdx.y * rows;
    const int col_base = blockIdx.x * cols;
    const int a_row = (warp / 2) * 16 + group;
    const int b_col = (warp % 2) * 32 + group;
    float accum[4][4] = {};

    for (int k0 = 0; k0 < k / 2; k0 += pairs) {
        bool invalid = false;
        for (int i = tid; i < rows * pairs; i += threads) {
            const int r = i / pairs;
            const int j = i % pairs;
            const int p = k0 + j;
            unsigned value = 0;
            if (row_base + r < m && p < k / 2) {
                const float *src = x + (row_base + r) * k + 2 * p;
                value = pack_pair(src[0], src[1]);
                invalid |= (value & 0x7c00u) == 0x7c00u ||
                           (value & 0x7c000000u) == 0x7c000000u;
            }
            a[r][j] = value;
        }
        if (__any_sync(0xffffffffu, invalid) && lane == 0)
            atomicOr(invalid_input, 1u);
        for (int i = tid; i < cols * pairs; i += threads) {
            const int c = i / pairs;
            const int j = i % pairs;
            const int p = k0 + j;
            b[c][j] = col_base + c < n && p < k / 2 ?
                w[(col_base + c) * (k / 2) + p] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < pairs; j += 8) {
            if (k0 + j >= k / 2)
                break;
            const int p = j + lane_pair;
            const unsigned a0 = a[a_row][p], a1 = a[a_row + 8][p];
            const unsigned a2 = a[a_row][p + 4], a3 = a[a_row + 8][p + 4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const unsigned *key = b[b_col + c * 8];
                mma(accum[c], a0, a1, a2, a3, key[p], key[p + 4]);
            }
        }
        __syncthreads();
    }

    const int r = row_base + a_row;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c = col_base + (warp % 2) * 32 + j * 8 + 2 * lane_pair;
        if (r < m && c < n) {
            out[r * n + c] = accum[j][0];
            if (c + 1 < n)
                out[r * n + c + 1] = accum[j][1];
        }
        if (r + 8 < m && c < n) {
            out[(r + 8) * n + c] = accum[j][2];
            if (c + 1 < n)
                out[(r + 8) * n + c + 1] = accum[j][3];
        }
    }
}
