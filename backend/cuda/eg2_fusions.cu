// Preserve the same evaluation and two output roundings as vit.cu's
// gelu_tanh followed by tower_base.cu's tower_mul.

__device__ __forceinline__ float gelu_gate(float x) {
    const double v = (double)x;
    const double cubic = __fma_rn(__dmul_rn(__dmul_rn(v, 0.044715), v), v, v);
    const double arg = __dmul_rn(0.7978845608028654, cubic);
    return __double2float_rn(__dmul_rn(__dmul_rn(0.5, v), __dadd_rn(1.0, tanh(arg))));
}

extern "C" __global__ void eg2_gelu_mul(float* __restrict__ gate,
                                        const float* __restrict__ up, int n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gate[i] = __fmul_rn(gelu_gate(gate[i]), up[i]);
    }
}

// ple starts at the selected layer's channels in the first packed row.
extern "C" __global__ void eg2_gelu_mul_ple(float* __restrict__ gate,
                                            const float* __restrict__ ple,
                                            int channels, long stride, int n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const long row = i / channels;
        const int channel = (int)(i % channels);
        gate[i] = __fmul_rn(gelu_gate(gate[i]), ple[row * stride + channel]);
    }
}
