#pragma once

// GELU rounds to FP32 before the caller multiplies by the up value.
__device__ __forceinline__ float gelu_gate(float x) {
    const double v = (double)x;
    const double cubic = __fma_rn(__dmul_rn(__dmul_rn(v, 0.044715), v), v, v);
    const double arg = __dmul_rn(0.7978845608028654, cubic);
    return __double2float_rn(__dmul_rn(__dmul_rn(0.5, v), __dadd_rn(1.0, tanh(arg))));
}
