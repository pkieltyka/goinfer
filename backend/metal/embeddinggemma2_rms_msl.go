//go:build darwin

package metal

// eg2RMSFusedMSL is the private #102 kernel family: the stock rmsnorm reduction,
// followed by a register-only FP32 epilogue that rounds the normalized word before
// the residual add. The no-contraction directive is deliberately last so it can
// affect only the helper definitions, never the reduction, rsqrt or the kernels.
const eg2RMSFusedMSL = `
#define EG2_RMS_TG 256
inline float eg2_rms_epilogue(float r, float x, float inv, float w);
inline float eg2_rms_epilogue_scale(float r, float x, float inv, float w, float s);

kernel void eg2_rmsnorm_add(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* r [[buffer(2)]], constant int& rows [[buffer(3)]], constant int& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]], uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)row * (uint)dim;
    device float* rr = r + (uint)row * (uint)dim;
    threadgroup float sm[EG2_RMS_TG];
    float acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float vv = xr[i]; acc += vv * vv; }
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = rsqrt(sm[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += tgsz) rr[i] = eg2_rms_epilogue(rr[i], xr[i], inv, w[i]);
}

kernel void eg2_rmsnorm_add_scale(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* r [[buffer(2)]], constant int& rows [[buffer(3)]], constant int& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]], constant float& s [[buffer(6)]], uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint tgsz [[threads_per_threadgroup]])
{
    device const float* xr = x + (uint)row * (uint)dim;
    device float* rr = r + (uint)row * (uint)dim;
    threadgroup float sm[EG2_RMS_TG];
    float acc = 0.0f;
    for (int i = tid; i < dim; i += tgsz) { float vv = xr[i]; acc += vv * vv; }
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tgsz >> 1; o > 0; o >>= 1) { if (tid < o) sm[tid] += sm[tid + o]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = rsqrt(sm[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += tgsz) rr[i] = eg2_rms_epilogue_scale(rr[i], xr[i], inv, w[i], s);
}

#pragma METAL fp contract(off)
inline float eg2_rms_epilogue(float r, float x, float inv, float w) {
    float n = (x * inv) * w;
    return r + n;
}
inline float eg2_rms_epilogue_scale(float r, float x, float inv, float w, float s) {
    float n = (x * inv) * w;
    float t = r + n;
    return t * s;
}
`

// newEG2RMS builds the optional fused residual-site pipelines for the explicit
// batch attachment. The released norm+add path stays the fallback whenever a
// pipeline is unavailable or would exceed the released static threadgroup bytes.
func newEG2RMS(d *Device, stockTG int) (Pipeline, Pipeline, bool) {
	var none Pipeline
	if stockTG <= 0 {
		return none, none, false
	}
	lib, err := d.CompileLibraryPrecise("#include <metal_stdlib>\nusing namespace metal;\n"+eg2RMSFusedMSL, MSL3_1)
	if err != nil {
		return none, none, false
	}
	defer d.ReleaseObject(lib)
	build := func(name string) (Pipeline, bool) {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil || p.ThreadExecutionWidth() != 32 || p.MaxTotalThreadsPerThreadgroup() < 256 ||
			p.StaticThreadgroupMemoryLength() <= 0 || p.StaticThreadgroupMemoryLength() > stockTG {
			if err == nil {
				d.ReleasePipeline(p)
			}
			return none, false
		}
		return p, true
	}
	plain, ok := build("eg2_rmsnorm_add")
	if !ok {
		return none, none, false
	}
	scaled, ok := build("eg2_rmsnorm_add_scale")
	if !ok {
		d.ReleasePipeline(plain)
		return none, none, false
	}
	return plain, scaled, true
}
