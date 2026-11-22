// Diagnostic prototype from goinfer's attention_seg at 44c5aa2ab. Preserve the score,
// softmax, and value reduction order. Adjacent scoring threads read adjacent keys.
#define LNBLOCK 256

extern "C" __global__ void eg2_transpose_keys(const float* __restrict__ k, float* __restrict__ kt,
                                              int tokens, int width, int pitch)
{
    __shared__ float tile[32][33];
    const int lane = threadIdx.x;
    const int row = threadIdx.y;
    const int x = blockIdx.x * 32;
    const int y = blockIdx.y * 32;

    #pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        const int token = y + row + offset;
        if (x + lane < width && token < tokens)
            tile[row + offset][lane] = k[(long)token * width + x + lane];
    }
    __syncthreads();
    #pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        const int feature = x + row + offset;
        const int token = y + lane;
        if (feature < width && token < tokens)
            kt[(long)feature * pitch + token] = tile[lane][row + offset];
    }
}

extern "C" __global__ void eg2_attention_transposed(
    const float* __restrict__ q, const float* __restrict__ kt,
    const float* __restrict__ v, float* __restrict__ out,
    const int* __restrict__ segStart, const int* __restrict__ segEnd,
    int seq, int nH, int nKV, int hd, int pitch, float scale)
{
    int blk = blockIdx.x;
    if (blk >= nH * seq) return;
    int h = blk / seq, i = blk % seq;
    int hidden = nH * hd, off = h * hd;
    int s0 = segStart[i], s1 = segEnd[i], n = s1 - s0;
    const int kv = h / (nH / nKV);
    const int kvWidth = nKV * hd;
    extern __shared__ float sc[];

    const float* qi = q + (long)i * hidden + off;
    const float* kh = kt + (long)kv * hd * pitch;
    __shared__ float qs[512];
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = qi[d];
    __syncthreads();
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        const float* key = kh + s0 + t;
        float acc = 0.f;
        for (int d = 0; d < hd; ++d) acc += qs[d] * key[(long)d * pitch];
        sc[t] = acc * scale;
    }
    __syncthreads();

    __shared__ float smax[LNBLOCK];
    __shared__ double ssum[LNBLOCK];
    float m = -3.402823466e+38f;
    for (int t = threadIdx.x; t < n; t += blockDim.x) if (sc[t] > m) m = sc[t];
    smax[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o && smax[threadIdx.x + o] > smax[threadIdx.x]) smax[threadIdx.x] = smax[threadIdx.x + o];
        __syncthreads();
    }
    float mx = smax[0];
    __syncthreads();

    double su = 0.0;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        double e = exp((double)sc[t] - (double)mx);
        sc[t] = (float)e;
        su += e;
    }
    ssum[threadIdx.x] = su;
    __syncthreads();
    // Cross-thread SUM reduction: the order is fixed by blockDim.x (== host ViTBlock
    // == LNBLOCK), and f64 addition is not associative either, so this width is a
    // BIT-IDENTITY dependency, not a tuning knob. See cuda_vit.go's ViTBlock.
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) ssum[threadIdx.x] += ssum[threadIdx.x + o];
        __syncthreads();
    }
    double inv = 1.0 / ssum[0];
    __syncthreads();
    for (int t = threadIdx.x; t < n; t += blockDim.x) sc[t] = (float)((double)sc[t] * inv);
    __syncthreads();

    float* oi = out + (long)i * hidden + off;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < n; t++) acc += sc[t] * v[(long)(s0 + t) * kvWidth + kv * hd + d];
        oi[d] = acc;
    }
}
