// DFA bottleneck attribution bench -- REAL data (dfatap6 val_00 loc/w bins).
//
// No ncu on board -> discriminator kernels instead:
//   T1 gather_real    : real addresses + real k=3.86 distribution (the engine case)
//   T2 gather_compact : same instruction stream, tap offsets folded into 2MB
//                       (L2-resident) -> isolates the DRAM contribution
//   T3 gather_k0      : counts all 0 -> launch + out-write floor
//   T4 plan_synth     : synthetic half logits with REAL validity density + real
//                       loc -> plan kernel cost (softmax+validity+emit)
//   T5 gather_h2      : half2 wide-load variant (2 ch/thread) -> ILP headroom probe
//   T6 stream_read    : 1GB uint4 grid-stride -> today's DRAM streaming ceiling
// Geometry = layers.0/p: A=1024 pts=500 N=512000 C=256 G=8 cams=3 levels=4.
// Host builds entries/counts from the real bins exactly like plan phase C
// (cam-major, level-minor, slot=rank) and cross-checks T4's counts+addresses.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// launchers from libdfa_plugin source (compiled together by the pgreal stage
// would duplicate symbols; declare instead -- no plugin TU needed here)
struct PgEnt { float4 wk; int4 off; float4 wg_lo, wg_hi; };   // 64B mirror of PgEnt
__device__ __forceinline__ float pg_sel8_l(const float4& lo, const float4& hi, int g) {
    return (g == 0) ? lo.x : (g == 1) ? lo.y : (g == 2) ? lo.z : (g == 3) ? lo.w
         : (g == 4) ? hi.x : (g == 5) ? hi.y : (g == 6) ? hi.z : hi.w;
}
extern "C" int dfa_pg_plan_cuda(void*, int*, const void*, int, const float*, const __half*,
                                const int*, const int*, int, int, int, int, int, int, int,
                                int, float, cudaStream_t);
extern "C" int dfa_pg_gather_cuda(__half*, const __half*, const void*, const int*,
                                  int, int, int, int, int, int, cudaStream_t);

static const int A = 1024, PTS = 500, N = A * PTS, C = 256, G = 8;
static const int CAMS = 3, LEVELS = 4, CS = CAMS * LEVELS, HW = 10880;
static const int SH[8] = {64, 128, 32, 64, 16, 32, 8, 16};      // H,W per level
static const int SSI[4] = {0, 8192, 10240, 10752};
static const char* KDIR = "/opt/m0/sd2/prof/kstat";
static const float MEAN_CORNERS = 3.952f;   // measured locally from the same bins

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct HEntry { float4 wk; int4 off; float4 wg_lo, wg_hi; };    // 64B, == PgEnt

static float timed_ms(std::function<void()> fn, int warm, int iters) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < warm; ++i) fn();
    CK(cudaDeviceSynchronize());
    float best = 1e30f, sum = 0.f;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(e0)); fn(); CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        best = best < ms ? best : ms; sum += ms;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return sum / iters;
}

// ---- wide-load probe: one thread per (row, channel-pair), __half2 taps -------
// timing-only probe (no bit-exactness requirement); per-entry math still fp32.
__global__ void gather_h2_kernel(__half* __restrict__ out, const __half* __restrict__ feat,
                                 const PgEnt* __restrict__ entries,
                                 const int* __restrict__ counts,
                                 int rows, int C, int G, int CS) {
    __shared__ PgEnt s_e[8][16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = idx < rows * (C / 2);
    int row = 0, cp = 0, k = 0;
    if (active) {
        row = idx / (C / 2); cp = idx % (C / 2);
        k = counts[row];
        const float* src = reinterpret_cast<const float*>(entries + (size_t)row * CS);
        for (int j = lane; j < k * 16; j += 32)
            reinterpret_cast<float*>(s_e[warp])[j] = src[j];
    }
    __syncwarp();
    if (!active) return;
    const int g = (cp * 2) / (C / G);
    const __half2* f2 = reinterpret_cast<const __half2*>(feat);
    float ax = 0.f, ay = 0.f;
    for (int j = 0; j < k; ++j) {
        const PgEnt ent = s_e[warp][j];
        const float wg = pg_sel8_l(ent.wg_lo, ent.wg_hi, g);
        float s1 = 0.f, s2 = 0.f;
        if (ent.wk.x != 0.f) { float2 v = __half22float2(f2[ent.off.x / 2 + cp]);
            s1 += ent.wk.x * v.x; s2 += ent.wk.x * v.y; }
        if (ent.wk.y != 0.f) { float2 v = __half22float2(f2[ent.off.y / 2 + cp]);
            s1 += ent.wk.y * v.x; s2 += ent.wk.y * v.y; }
        if (ent.wk.z != 0.f) { float2 v = __half22float2(f2[ent.off.z / 2 + cp]);
            s1 += ent.wk.z * v.x; s2 += ent.wk.z * v.y; }
        if (ent.wk.w != 0.f) { float2 v = __half22float2(f2[ent.off.w / 2 + cp]);
            s1 += ent.wk.w * v.x; s2 += ent.wk.w * v.y; }
        ax += s1 * wg; ay += s2 * wg;
    }
    reinterpret_cast<__half2*>(out)[idx] = __floats2half2_rn(ax, ay);
}

// ---- v3 probes: CPT channels/thread, block-cooperative single staging -------
// C=256 hardwired (bench geometry); entry-major accumulation order preserved
// (per channel: corners x,y,z,w with skips, then acc += s*wg) so outputs must
// be bit-identical to the h2 launcher path.
template <int CPT>
struct VecT { float v[CPT]; };

template <int CPT>
__device__ __forceinline__ VecT<CPT> ldv(const __half* p);

template <>
__device__ __forceinline__ VecT<2> ldv<2>(const __half* p) {
    VecT<2> r;
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(p));
    r.v[0] = f.x; r.v[1] = f.y;
    return r;
}
template <>
__device__ __forceinline__ VecT<4> ldv<4>(const __half* p) {
    VecT<4> r;
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    r.v[0] = a.x; r.v[1] = a.y; r.v[2] = b.x; r.v[3] = b.y;
    return r;
}
template <>
__device__ __forceinline__ VecT<8> ldv<8>(const __half* p) {
    VecT<8> r;
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    const float2 c = __half22float2(*reinterpret_cast<const __half2*>(&u.z));
    const float2 d = __half22float2(*reinterpret_cast<const __half2*>(&u.w));
    r.v[0] = a.x; r.v[1] = a.y; r.v[2] = b.x; r.v[3] = b.y;
    r.v[4] = c.x; r.v[5] = c.y; r.v[6] = d.x; r.v[7] = d.y;
    return r;
}

template <int CPT>
__device__ __forceinline__ void store_out(__half* out, size_t oidx, const float* acc);

template <>
__device__ __forceinline__ void store_out<2>(__half* out, size_t oidx, const float* acc) {
    reinterpret_cast<__half2*>(out)[oidx] = __floats2half2_rn(acc[0], acc[1]);
}
template <>
__device__ __forceinline__ void store_out<4>(__half* out, size_t oidx, const float* acc) {
    const __half2 a = __floats2half2_rn(acc[0], acc[1]);
    const __half2 b = __floats2half2_rn(acc[2], acc[3]);
    uint2 u;
    u.x = *reinterpret_cast<const unsigned*>(&a);
    u.y = *reinterpret_cast<const unsigned*>(&b);
    reinterpret_cast<uint2*>(out)[oidx] = u;
}
template <>
__device__ __forceinline__ void store_out<8>(__half* out, size_t oidx, const float* acc) {
    __half2 h[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) h[q] = __floats2half2_rn(acc[2 * q], acc[2 * q + 1]);
    uint4 u;
    u.x = *reinterpret_cast<const unsigned*>(&h[0]);
    u.y = *reinterpret_cast<const unsigned*>(&h[1]);
    u.z = *reinterpret_cast<const unsigned*>(&h[2]);
    u.w = *reinterpret_cast<const unsigned*>(&h[3]);
    reinterpret_cast<uint4*>(out)[oidx] = u;
}

template <int CPT, int UNROLL>
__global__ void gather_v3_kernel(__half* __restrict__ out, const __half* __restrict__ feat,
                                 const PgEnt* __restrict__ entries,
                                 const int* __restrict__ counts,
                                 int rows, int CS) {
    constexpr int C = 256, GSZ = 32;             // channels, group size (C/G)
    constexpr int TPR = C / CPT;                 // threads per row
    constexpr int ROWS_B = 256 / TPR;            // rows per block
    static_assert(256 % TPR == 0, "block must cover whole rows");
    __shared__ PgEnt s_e[ROWS_B][16];
    const int tid = threadIdx.x;
    const size_t gidx = (size_t)blockIdx.x * 256 + tid;
    const int row = (int)(gidx / TPR);
    const bool active = row < rows;
    const int row0 = blockIdx.x * ROWS_B;
    for (int r = 0; r < ROWS_B; ++r) {           // cooperative stage, 1x traffic
        const int rr = row0 + r;
        if (rr >= rows) break;
        const int kr = counts[rr];
        const float* src = reinterpret_cast<const float*>(entries + (size_t)rr * CS);
        for (int j = tid; j < kr * 16; j += 256)
            reinterpret_cast<float*>(s_e[r])[j] = src[j];
    }
    __syncthreads();
    if (!active) return;
    const int k = counts[row];
    const int c0 = (int)(gidx % TPR) * CPT;
    const int g = c0 / GSZ;
    const PgEnt* er = s_e[row - row0];
    float acc[CPT];
#pragma unroll
    for (int q = 0; q < CPT; ++q) acc[q] = 0.f;

    // UNROLL==1 body: per entry s = sum of corners, acc += s*wg (h2 order)
    auto do_entry = [&](const PgEnt& e, float wg) {
        float s[CPT];
#pragma unroll
        for (int q = 0; q < CPT; ++q) s[q] = 0.f;
        if (e.wk.x != 0.f) { const VecT<CPT> v = ldv<CPT>(feat + e.off.x + c0);
#pragma unroll
            for (int q = 0; q < CPT; ++q) s[q] = e.wk.x * v.v[q]; }
        if (e.wk.y != 0.f) { const VecT<CPT> v = ldv<CPT>(feat + e.off.y + c0);
#pragma unroll
            for (int q = 0; q < CPT; ++q) s[q] += e.wk.y * v.v[q]; }
        if (e.wk.z != 0.f) { const VecT<CPT> v = ldv<CPT>(feat + e.off.z + c0);
#pragma unroll
            for (int q = 0; q < CPT; ++q) s[q] += e.wk.z * v.v[q]; }
        if (e.wk.w != 0.f) { const VecT<CPT> v = ldv<CPT>(feat + e.off.w + c0);
#pragma unroll
            for (int q = 0; q < CPT; ++q) s[q] += e.wk.w * v.v[q]; }
#pragma unroll
        for (int q = 0; q < CPT; ++q) acc[q] += s[q] * wg;
    };

    if (UNROLL == 1) {
        for (int j = 0; j < k; ++j) {
            const PgEnt e = er[j];
            const float wg = pg_sel8_l(e.wg_lo, e.wg_hi, g);
            if (wg == 0.f) continue;
            do_entry(e, wg);
        }
    } else {
        int j = 0;
        for (; j + UNROLL <= k; j += UNROLL) {
            PgEnt e[UNROLL];
            float wg[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                e[u] = er[j + u];
                wg[u] = pg_sel8_l(e[u].wg_lo, e[u].wg_hi, g);
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                if (wg[u] != 0.f) do_entry(e[u], wg[u]);
        }
        for (; j < k; ++j) {
            const PgEnt e = er[j];
            const float wg = pg_sel8_l(e.wg_lo, e.wg_hi, g);
            if (wg == 0.f) continue;
            do_entry(e, wg);
        }
    }
    store_out<CPT>(out, (size_t)row * (C / CPT) + (c0 / CPT), acc);
}

// ---- int8-feat probe: same v3 CPT=4/U=1 mapping, int8 taps + dequant --------
// Models deployment where upstream writes int8 feat (per-tensor scale dscale);
// plugin reads 4 packed int8 per corner (uint32) and dequantizes.
__global__ void gather_v3_i8_kernel(__half* __restrict__ out,
                                    const signed char* __restrict__ feat8,
                                    const PgEnt* __restrict__ entries,
                                    const int* __restrict__ counts,
                                    int rows, int CS, float dscale) {
    constexpr int C = 256, GSZ = 32;
    constexpr int TPR = C / 4;
    constexpr int ROWS_B = 256 / TPR;
    __shared__ PgEnt s_e[ROWS_B][16];
    const int tid = threadIdx.x;
    const size_t gidx = (size_t)blockIdx.x * 256 + tid;
    const int row = (int)(gidx / TPR);
    const bool active = row < rows;
    const int row0 = blockIdx.x * ROWS_B;
    for (int r = 0; r < ROWS_B; ++r) {
        const int rr = row0 + r;
        if (rr >= rows) break;
        const int kr = counts[rr];
        const float* src = reinterpret_cast<const float*>(entries + (size_t)rr * CS);
        for (int j = tid; j < kr * 16; j += 256)
            reinterpret_cast<float*>(s_e[r])[j] = src[j];
    }
    __syncthreads();
    if (!active) return;
    const int k = counts[row];
    const int c0 = (int)(gidx % TPR) * 4;
    const int g = c0 / GSZ;
    const PgEnt* er = s_e[row - row0];
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int j = 0; j < k; ++j) {
        const PgEnt e = er[j];
        const float wg = pg_sel8_l(e.wg_lo, e.wg_hi, g);
        if (wg == 0.f) continue;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        if (e.wk.x != 0.f) {
            const unsigned u = *reinterpret_cast<const unsigned*>(feat8 + e.off.x + c0);
            const float f0 = (float)(int)(signed char)(u & 0xffu) * dscale;
            const float f1 = (float)(int)(signed char)((u >> 8) & 0xffu) * dscale;
            const float f2 = (float)(int)(signed char)((u >> 16) & 0xffu) * dscale;
            const float f3 = (float)(int)(signed char)((u >> 24) & 0xffu) * dscale;
            s0 = e.wk.x * f0; s1 = e.wk.x * f1; s2 = e.wk.x * f2; s3 = e.wk.x * f3;
        }
        if (e.wk.y != 0.f) {
            const unsigned u = *reinterpret_cast<const unsigned*>(feat8 + e.off.y + c0);
            const float f0 = (float)(int)(signed char)(u & 0xffu) * dscale;
            const float f1 = (float)(int)(signed char)((u >> 8) & 0xffu) * dscale;
            const float f2 = (float)(int)(signed char)((u >> 16) & 0xffu) * dscale;
            const float f3 = (float)(int)(signed char)((u >> 24) & 0xffu) * dscale;
            s0 += e.wk.y * f0; s1 += e.wk.y * f1; s2 += e.wk.y * f2; s3 += e.wk.y * f3;
        }
        if (e.wk.z != 0.f) {
            const unsigned u = *reinterpret_cast<const unsigned*>(feat8 + e.off.z + c0);
            const float f0 = (float)(int)(signed char)(u & 0xffu) * dscale;
            const float f1 = (float)(int)(signed char)((u >> 8) & 0xffu) * dscale;
            const float f2 = (float)(int)(signed char)((u >> 16) & 0xffu) * dscale;
            const float f3 = (float)(int)(signed char)((u >> 24) & 0xffu) * dscale;
            s0 += e.wk.z * f0; s1 += e.wk.z * f1; s2 += e.wk.z * f2; s3 += e.wk.z * f3;
        }
        if (e.wk.w != 0.f) {
            const unsigned u = *reinterpret_cast<const unsigned*>(feat8 + e.off.w + c0);
            const float f0 = (float)(int)(signed char)(u & 0xffu) * dscale;
            const float f1 = (float)(int)(signed char)((u >> 8) & 0xffu) * dscale;
            const float f2 = (float)(int)(signed char)((u >> 16) & 0xffu) * dscale;
            const float f3 = (float)(int)(signed char)((u >> 24) & 0xffu) * dscale;
            s0 += e.wk.w * f0; s1 += e.wk.w * f1; s2 += e.wk.w * f2; s3 += e.wk.w * f3;
        }
        a0 += s0 * wg; a1 += s1 * wg; a2 += s2 * wg; a3 += s3 * wg;
    }
    __half2* o2 = reinterpret_cast<__half2*>(out) + (size_t)row * (C >> 1) + (c0 >> 1);
    o2[0] = __floats2half2_rn(a0, a1);
    o2[1] = __floats2half2_rn(a2, a3);
}

// ---- int8 wide-mapping probes (user hypothesis: int8 + more ch/thread) ------
// CPT=8 (uint2 = 8 int8 per corner load) and CPT=16 (uint4 = 16 int8): the
// instruction-density argument for int8 — half the load instructions per
// channel vs half-precision at the same mapping width.
__device__ __forceinline__ void deq8(unsigned u, float dscale, float* f) {
    f[0] = (float)(int)(signed char)(u & 0xffu) * dscale;
    f[1] = (float)(int)(signed char)((u >> 8) & 0xffu) * dscale;
    f[2] = (float)(int)(signed char)((u >> 16) & 0xffu) * dscale;
    f[3] = (float)(int)(signed char)((u >> 24) & 0xffu) * dscale;
}

template <int CPT>
__global__ void gather_v3_i8w_kernel(__half* __restrict__ out,
                                     const signed char* __restrict__ feat8,
                                     const PgEnt* __restrict__ entries,
                                     const int* __restrict__ counts,
                                     int rows, int CS, float dscale) {
    constexpr int C = 256, GSZ = 32;
    constexpr int TPR = C / CPT;
    constexpr int ROWS_B = 256 / TPR;
    static_assert(CPT == 8 || CPT == 16, "probe widths");
    __shared__ PgEnt s_e[CPT == 8 ? 8 : 16][16];
    const int tid = threadIdx.x;
    const size_t gidx = (size_t)blockIdx.x * 256 + tid;
    const int row = (int)(gidx / TPR);
    const bool active = row < rows;
    const int row0 = blockIdx.x * ROWS_B;
    for (int r = 0; r < ROWS_B; ++r) {
        const int rr = row0 + r;
        if (rr >= rows) break;
        const int kr = counts[rr];
        const float* src = reinterpret_cast<const float*>(entries + (size_t)rr * CS);
        for (int j = tid; j < kr * 16; j += 256)
            reinterpret_cast<float*>(s_e[r])[j] = src[j];
    }
    __syncthreads();
    if (!active) return;
    const int k = counts[row];
    const int c0 = (int)(gidx % TPR) * CPT;
    const int g = c0 / GSZ;
    const PgEnt* er = s_e[row - row0];
    float acc[16];
#pragma unroll
    for (int q = 0; q < CPT; ++q) acc[q] = 0.f;
    for (int j = 0; j < k; ++j) {
        const PgEnt e = er[j];
        const float wg = pg_sel8_l(e.wg_lo, e.wg_hi, g);
        if (wg == 0.f) continue;
        float s[16];
#pragma unroll
        for (int q = 0; q < CPT; ++q) s[q] = 0.f;
        float* f = s;
        if (e.wk.x != 0.f) {
            if (CPT == 8) {
                const uint2 u = *reinterpret_cast<const uint2*>(feat8 + e.off.x + c0);
                deq8(u.x, dscale, f); deq8(u.y, dscale, f + 4);
            } else {
                const uint4 u = *reinterpret_cast<const uint4*>(feat8 + e.off.x + c0);
                deq8(u.x, dscale, f); deq8(u.y, dscale, f + 4);
                deq8(u.z, dscale, f + 8); deq8(u.w, dscale, f + 12);
            }
#pragma unroll
            for (int q = 0; q < CPT; ++q) s[q] = e.wk.x * f[q];
        }
        if (e.wk.y != 0.f) {
            if (CPT == 8) {
                const uint2 u = *reinterpret_cast<const uint2*>(feat8 + e.off.y + c0);
                deq8(u.x, dscale, f); deq8(u.y, dscale, f + 4);
            } else {
                const uint4 u = *reinterpret_cast<const uint4*>(feat8 + e.off.y + c0);
                deq8(u.x, dscale, f); deq8(u.y, dscale, f + 4);
                deq8(u.z, dscale, f + 8); deq8(u.w, dscale, f + 12);
            }
#pragma unroll
            for (int q = 0; q < CPT; ++q) s[q] += e.wk.y * f[q];
        }
        if (e.wk.z != 0.f) {
            if (CPT == 8) {
                const uint2 u = *reinterpret_cast<const uint2*>(feat8 + e.off.z + c0);
                deq8(u.x, dscale, f); deq8(u.y, dscale, f + 4);
            } else {
                const uint4 u = *reinterpret_cast<const uint4*>(feat8 + e.off.z + c0);
                deq8(u.x, dscale, f); deq8(u.y, dscale, f + 4);
                deq8(u.z, dscale, f + 8); deq8(u.w, dscale, f + 12);
            }
#pragma unroll
            for (int q = 0; q < CPT; ++q) s[q] += e.wk.z * f[q];
        }
        if (e.wk.w != 0.f) {
            if (CPT == 8) {
                const uint2 u = *reinterpret_cast<const uint2*>(feat8 + e.off.w + c0);
                deq8(u.x, dscale, f); deq8(u.y, dscale, f + 4);
            } else {
                const uint4 u = *reinterpret_cast<const uint4*>(feat8 + e.off.w + c0);
                deq8(u.x, dscale, f); deq8(u.y, dscale, f + 4);
                deq8(u.z, dscale, f + 8); deq8(u.w, dscale, f + 12);
            }
#pragma unroll
            for (int q = 0; q < CPT; ++q) s[q] += e.wk.w * f[q];
        }
#pragma unroll
        for (int q = 0; q < CPT; ++q) acc[q] += s[q] * wg;
    }
    if (CPT == 8) {
        __half2 h[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) h[q] = __floats2half2_rn(acc[2 * q], acc[2 * q + 1]);
        uint4 u;
        u.x = *reinterpret_cast<const unsigned*>(&h[0]);
        u.y = *reinterpret_cast<const unsigned*>(&h[1]);
        u.z = *reinterpret_cast<const unsigned*>(&h[2]);
        u.w = *reinterpret_cast<const unsigned*>(&h[3]);
        reinterpret_cast<uint4*>(out)[(size_t)row * (C / 8) + (c0 / 8)] = u;
    } else {
        __half2 h[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) h[q] = __floats2half2_rn(acc[2 * q], acc[2 * q + 1]);
        uint4* o4 = reinterpret_cast<uint4*>(out) + (size_t)row * (C / 8) + (c0 / 8);
        uint4 u0, u1;
        u0.x = *reinterpret_cast<const unsigned*>(&h[0]);
        u0.y = *reinterpret_cast<const unsigned*>(&h[1]);
        u0.z = *reinterpret_cast<const unsigned*>(&h[2]);
        u0.w = *reinterpret_cast<const unsigned*>(&h[3]);
        u1.x = *reinterpret_cast<const unsigned*>(&h[4]);
        u1.y = *reinterpret_cast<const unsigned*>(&h[5]);
        u1.z = *reinterpret_cast<const unsigned*>(&h[6]);
        u1.w = *reinterpret_cast<const unsigned*>(&h[7]);
        o4[0] = u0; o4[1] = u1;
    }
}


__global__ void stream_read_kernel(const uint4* __restrict__ p, size_t n4, float* sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (; i < n4; i += stride) {
        const uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) sink[0] = 1.f;   // never true; defeats DCE
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("[dev] %s SMs=%d clock=%dMHz mem=%zuMB L2=%dMB\n", prop.name,
           prop.multiProcessorCount, prop.clockRate / 1000,
           prop.totalGlobalMem >> 20, (int)(prop.l2CacheSize >> 20));

    // ---- load real bins ----
    std::vector<float> loc((size_t)N * CAMS * 2), w((size_t)N * CAMS * LEVELS * G);
    FILE* f = fopen((std::string(KDIR) + "/loc.bin").c_str(), "rb");
    if (!f || fread(loc.data(), 4, loc.size(), f) != loc.size()) { printf("loc.bin FAIL\n"); return 1; }
    fclose(f);
    f = fopen((std::string(KDIR) + "/w.bin").c_str(), "rb");
    if (!f || fread(w.data(), 4, w.size(), f) != w.size()) { printf("w.bin FAIL\n"); return 1; }
    fclose(f);
    printf("[data] loc/w loaded (%zu/%zu floats)\n", loc.size(), w.size());

    // ---- host entry build (plan phase C replica) ----
    std::vector<HEntry> ent_real((size_t)N * CS);
    std::vector<int> cnt(N);
    unsigned long long tot_entries = 0;
    for (int row = 0; row < N; ++row) {
        unsigned mask = 0;
        for (int cs = 0; cs < CS; ++cs) {
            const int cam = cs / LEVELS;
            const float wv = loc[(size_t)row * CAMS * 2 + cam * 2];
            const float hv = loc[(size_t)row * CAMS * 2 + cam * 2 + 1];
            if (!(wv > 0.f && wv < 1.f && hv > 0.f && hv < 1.f)) continue;
            const float* wg = &w[((size_t)row * CAMS + cam) * LEVELS * G + (cs % LEVELS) * G];
            for (int g = 0; g < G; ++g)
                if (wg[g] > 0.f) { mask |= 1u << cs; break; }
        }
        cnt[row] = __builtin_popcount(mask);
        tot_entries += cnt[row];
        int slot = 0;
        for (int cs = 0; cs < CS; ++cs) {
            if (!((mask >> cs) & 1u)) continue;
            const int cam = cs / LEVELS, lv = cs % LEVELS;
            const float wv = loc[(size_t)row * CAMS * 2 + cam * 2];
            const float hv = loc[(size_t)row * CAMS * 2 + cam * 2 + 1];
            const int H = SH[lv * 2], W = SH[lv * 2 + 1], ssi = SSI[lv];
            const float h_im = (float)((double)(hv * (float)H) - 0.5);
            const float w_im = (float)((double)(wv * (float)W) - 0.5);
            const int h_low = (int)floorf(h_im), w_low = (int)floorf(w_im);
            const int h_high = h_low + 1, w_high = w_low + 1;
            const float lh = h_im - h_low, lw = w_im - w_low;
            const float hh = 1 - lh, hw2 = 1 - lw;
            const int w_stride = C, h_stride = W * w_stride;
            const int base = cam * HW * C + ssi * C;
            HEntry e; memset(&e, 0, sizeof(e));
            e.wk.x = (h_low >= 0 && w_low >= 0) ? hh * hw2 : 0.f;
            e.off.x = (e.wk.x != 0.f) ? base + h_low * h_stride + w_low * w_stride : 0;
            e.wk.y = (h_low >= 0 && w_high <= W - 1) ? hh * lw : 0.f;
            e.off.y = (e.wk.y != 0.f) ? base + h_low * h_stride + w_high * w_stride : 0;
            e.wk.z = (h_high <= H - 1 && w_low >= 0) ? lh * hw2 : 0.f;
            e.off.z = (e.wk.z != 0.f) ? base + h_high * h_stride + w_low * w_stride : 0;
            e.wk.w = (h_high <= H - 1 && w_high <= W - 1) ? lh * lw : 0.f;
            e.off.w = (e.wk.w != 0.f) ? base + h_high * h_stride + w_high * w_stride : 0;
            const float* wg = &w[((size_t)row * CAMS + cam) * LEVELS * G + lv * G];
            e.wg_lo = make_float4(wg[0], wg[1], wg[2], wg[3]);
            e.wg_hi = make_float4(wg[4], wg[5], wg[6], wg[7]);
            ent_real[(size_t)row * CS + (slot++)] = e;
        }
    }
    printf("[host] entries=%llu mean_k=%.3f\n", tot_entries, (double)tot_entries / N);

    // compact variant: fold tap offsets into 2MB (L2-resident)
    const int COMPACT_ELEMS = 1 << 20;
    std::vector<HEntry> ent_cmp = ent_real;
    for (auto& e : ent_cmp) {
        if (e.wk.x != 0.f) e.off.x %= COMPACT_ELEMS;
        if (e.wk.y != 0.f) e.off.y %= COMPACT_ELEMS;
        if (e.wk.z != 0.f) e.off.z %= COMPACT_ELEMS;
        if (e.wk.w != 0.f) e.off.w %= COMPACT_ELEMS;
    }

    // synthetic half logits with real validity density: [A, ISP, G], i = cs*pts+pt
    const int ISP = CS * PTS;
    std::vector<__half> logits((size_t)A * ISP * G);
    __half_raw ninf_raw; ninf_raw.x = 0xFC00;
    const __half NEG_INF(ninf_raw);
    unsigned seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) * (1.f / 16777216.f); };
    auto randn = [&]() {
        return sqrtf(-2.f * logf(rnd() + 1e-9f)) * cosf(6.2831853f * rnd()) * 1.5f; };
    for (int a_i = 0; a_i < A; ++a_i)
        for (int pt = 0; pt < PTS; ++pt) {
            const int row = a_i * PTS + pt;
            for (int cs = 0; cs < CS; ++cs) {
                const int cam = cs / LEVELS;
                const float wv = loc[(size_t)row * CAMS * 2 + cam * 2];
                const float hv = loc[(size_t)row * CAMS * 2 + cam * 2 + 1];
                const bool guard = wv > 0.f && wv < 1.f && hv > 0.f && hv < 1.f;
                const float* wg = &w[((size_t)row * CAMS + cam) * LEVELS * G + (cs % LEVELS) * G];
                bool any = false;
                for (int g = 0; g < G; ++g) any |= (wg[g] > 0.f);
                __half* p = &logits[((size_t)a_i * ISP + cs * PTS + pt) * G];
                for (int g = 0; g < G; ++g)
                    p[g] = (guard && any) ? __float2half(randn()) : NEG_INF;
            }
        }
    std::vector<__half> loc_h((size_t)N * CAMS * 2);
    for (size_t i = 0; i < loc_h.size(); ++i) loc_h[i] = __float2half(loc[i]);

    // ---- device buffers ----
    const size_t ent_bytes = (size_t)N * CS * 64;
    HEntry *d_ent_real, *d_ent_cmp, *d_ent_ws; int *d_cnt, *d_cnt0;
    CK(cudaMalloc(&d_ent_real, ent_bytes)); CK(cudaMalloc(&d_ent_cmp, ent_bytes));
    CK(cudaMalloc(&d_ent_ws, ent_bytes));
    CK(cudaMalloc(&d_cnt, (size_t)N * 4)); CK(cudaMalloc(&d_cnt0, (size_t)N * 4));
    CK(cudaMemcpy(d_ent_real, ent_real.data(), ent_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ent_cmp, ent_cmp.data(), ent_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cnt, cnt.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_cnt0, 0, (size_t)N * 4));
    __half *d_feat, *d_out, *d_logits, *d_loc;
    CK(cudaMalloc(&d_feat, (size_t)CAMS * HW * C * 2));
    CK(cudaMalloc(&d_out, (size_t)N * C * 2));
    CK(cudaMalloc(&d_logits, logits.size() * 2));
    CK(cudaMalloc(&d_loc, loc_h.size() * 2));
    {   std::vector<__half> feat((size_t)CAMS * HW * C);
        for (size_t i = 0; i < feat.size(); ++i) feat[i] = __float2half(randn());
        CK(cudaMemcpy(d_feat, feat.data(), feat.size() * 2, cudaMemcpyHostToDevice)); }
    CK(cudaMemcpy(d_logits, logits.data(), logits.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_loc, loc_h.data(), loc_h.size() * 2, cudaMemcpyHostToDevice));
    int *d_sp, *d_ssi;
    CK(cudaMalloc(&d_sp, sizeof(SH))); CK(cudaMalloc(&d_ssi, sizeof(SSI)));
    CK(cudaMemcpy(d_sp, SH, sizeof(SH), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ssi, SSI, sizeof(SSI), cudaMemcpyHostToDevice));
    float* d_sink; CK(cudaMalloc(&d_sink, 4));
    const size_t STREAM_BYTES = (size_t)1 << 30;
    uint4* d_stream; CK(cudaMalloc(&d_stream, STREAM_BYTES));
    CK(cudaMemset(d_stream, 1, STREAM_BYTES));

    // T4 sanity: plan(synth) counts must equal host counts; addresses must match
    {
        dfa_pg_plan_cuda(d_ent_ws, d_cnt0, d_logits, 1, nullptr, d_loc,
                         d_sp, d_ssi, 1, CAMS, HW, C, LEVELS, N, A, G, 0.f, 0);
        CK(cudaDeviceSynchronize());
        std::vector<int> cnt2(N);
        CK(cudaMemcpy(cnt2.data(), d_cnt0, (size_t)N * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (int i = 0; i < N; ++i) if (cnt2[i] != cnt[i]) ++bad;
        printf("[sanity] plan counts mismatch rows = %zu / %d\n", bad, N);
        std::vector<HEntry> pe((size_t)1000 * CS);
        CK(cudaMemcpy(pe.data(), d_ent_ws, pe.size() * 64, cudaMemcpyDeviceToHost));
        size_t off_bad = 0, wk_bad = 0;
        for (int r = 0; r < 1000; ++r)
            for (int s = 0; s < cnt[r]; ++s) {
                const HEntry& a = pe[(size_t)r * CS + s];
                const HEntry& b = ent_real[(size_t)r * CS + s];
                if (memcmp(&a.off, &b.off, 16) != 0) ++off_bad;
                if (memcmp(&a.wk, &b.wk, 16) != 0) ++wk_bad;
            }
        printf("[sanity] plan vs host (first 1000 rows): off_bad=%zu wk_bad=%zu\n", off_bad, wk_bad);
        CK(cudaMemset(d_cnt0, 0, (size_t)N * 4));   // reset for T3 k0
    }

    // ---- timings ----
    auto t1 = timed_ms([&]() { dfa_pg_gather_cuda(d_out, d_feat, d_ent_real, d_cnt,
                                                   1, C, G, CAMS, LEVELS, N, 0); }, 10, 50);
    auto t2 = timed_ms([&]() { dfa_pg_gather_cuda(d_out, d_feat, d_ent_cmp, d_cnt,
                                                   1, C, G, CAMS, LEVELS, N, 0); }, 10, 50);
    auto t3 = timed_ms([&]() { dfa_pg_gather_cuda(d_out, d_feat, d_ent_real, d_cnt0,
                                                   1, C, G, CAMS, LEVELS, N, 0); }, 10, 50);
    auto t4 = timed_ms([&]() {
        dfa_pg_plan_cuda(d_ent_ws, d_cnt0, d_logits, 1, nullptr, d_loc,
                         d_sp, d_ssi, 1, CAMS, HW, C, LEVELS, N, A, G, 0.f, 0); }, 5, 20);
    const int h2_total = N * C / 2, h2_block = 256;
    const int h2_grid = (h2_total + h2_block - 1) / h2_block;
    auto t5 = timed_ms([&]() { gather_h2_kernel<<<h2_grid, h2_block>>>(
        d_out, d_feat, (PgEnt*)d_ent_real, d_cnt, N, C, G, CS); }, 10, 50);
    const size_t n4 = STREAM_BYTES / 16;
    auto t6 = timed_ms([&]() { stream_read_kernel<<<2048, 256>>>(d_stream, n4, d_sink); },
                       3, 10);

    // ---- v3 probes: CPT channels/thread x UNROLL, block-cooperative staging ----
    // reference output from the launcher (h2) path for bit-exact comparison
    dfa_pg_gather_cuda(d_out, d_feat, d_ent_real, d_cnt, 1, C, G, CAMS, LEVELS, N, 0);
    CK(cudaDeviceSynchronize());
    std::vector<__half> ref_out((size_t)N * C);
    CK(cudaMemcpy(ref_out.data(), d_out, ref_out.size() * 2, cudaMemcpyDeviceToHost));

    float t7[6]; int v3_cpt[6] = {2, 4, 4, 8, 8, 0}, v3_unr[6] = {1, 1, 2, 1, 2, 0};
    for (int vi = 0; vi < 5; ++vi) {
        const int cpt = v3_cpt[vi], unr = v3_unr[vi];
        const int tpr = C / cpt;
        const int grid = (int)(((size_t)N * tpr + 255) / 256);
        auto t = timed_ms([&]() {
            if (cpt == 2) gather_v3_kernel<2, 1><<<grid, 256>>>(d_out, d_feat, (PgEnt*)d_ent_real, d_cnt, N, CS);
            else if (cpt == 4 && unr == 1) gather_v3_kernel<4, 1><<<grid, 256>>>(d_out, d_feat, (PgEnt*)d_ent_real, d_cnt, N, CS);
            else if (cpt == 4) gather_v3_kernel<4, 2><<<grid, 256>>>(d_out, d_feat, (PgEnt*)d_ent_real, d_cnt, N, CS);
            else if (cpt == 8 && unr == 1) gather_v3_kernel<8, 1><<<grid, 256>>>(d_out, d_feat, (PgEnt*)d_ent_real, d_cnt, N, CS);
            else gather_v3_kernel<8, 2><<<grid, 256>>>(d_out, d_feat, (PgEnt*)d_ent_real, d_cnt, N, CS);
        }, 10, 50);
        t7[vi] = t;
        CK(cudaDeviceSynchronize());
        std::vector<__half> got(ref_out.size());
        CK(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));
        size_t nd = 0;
        int maxd = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const short a = *reinterpret_cast<const short*>(&got[i]);
            const short b = *reinterpret_cast<const short*>(&ref_out[i]);
            if (a != b) { ++nd; const int dd = abs(a - b); if (dd > maxd) maxd = dd; }
        }
        printf("[V3] CPT=%d U=%d      = %8.3f ms  bitdiff=%zu maxbits=%d\n",
               cpt, unr, t, nd, maxd);
    }

    const double feat_gb = (double)tot_entries * MEAN_CORNERS * C * 2 / 1e9;
    printf("[T1] gather_real    = %8.3f ms  (feat~%.2fGB -> %.0f GB/s eff)\n",
           t1, feat_gb, feat_gb / (t1 * 1e-3));
    printf("[T2] gather_compact = %8.3f ms  (L2-resident taps)\n", t2);
    printf("[T3] gather_k0      = %8.3f ms  (launch+write floor)\n", t3);
    printf("[T4] plan_synth     = %8.3f ms\n", t4);
    printf("[T5] gather_h2      = %8.3f ms  (wide-load probe)\n", t5);
    printf("[T6] stream 1GB     = %8.3f ms  -> %.0f GB/s ceiling\n",
           t6, (STREAM_BYTES / 1e9) / (t6 * 1e-3));
    float best7 = 1e30f; int bi = -1;
    for (int i = 0; i < 5; ++i) if (t7[i] < best7) { best7 = t7[i]; bi = i; }
    printf("[BEST-V3] CPT=%d U=%d = %.3f ms (vs launcher h2 %.3f, -%.0f%%)\n",
           v3_cpt[bi], v3_unr[bi], best7, t1, 100.f * (1.f - best7 / t1));

    // ---- T8: int8-feat probe (half the tap bytes, + dequant ALU) ----
    {
        const size_t feat8_elems = (size_t)CAMS * HW * C;
        signed char* d_feat8;
        CK(cudaMalloc(&d_feat8, feat8_elems));
        {   std::vector<signed char> f8(feat8_elems);
            unsigned s2 = 987654321u;
            for (size_t i = 0; i < f8.size(); ++i) {
                s2 = s2 * 1664525u + 1013904223u;
                f8[i] = (signed char)((s2 >> 16) & 0xffu);
            }
            CK(cudaMemcpy(d_feat8, f8.data(), f8.size(), cudaMemcpyHostToDevice));
        }
        const int tpr8 = C / 4;
        const int grid8 = (int)(((size_t)N * tpr8 + 255) / 256);
        auto t8 = timed_ms([&]() {
            gather_v3_i8_kernel<<<grid8, 256>>>(d_out, d_feat8, (PgEnt*)d_ent_real,
                                                d_cnt, N, CS, 0.05f);
        }, 10, 50);
        printf("[T8] v3 int8-feat   = %8.3f ms  (vs v3 half %.3f, %+.0f%%; feat bytes 4.0->2.0GB)\n",
               t8, best7, 100.f * (t8 / best7 - 1.f));
        // wide-mapping int8 probes (same 8B/16B load as half CPT=8, half the bytes)
        const int grid9 = (int)(((size_t)N * (C / 8) + 255) / 256);
        auto t9 = timed_ms([&]() {
            gather_v3_i8w_kernel<8><<<grid9, 256>>>(d_out, d_feat8, (PgEnt*)d_ent_real,
                                                    d_cnt, N, CS, 0.05f);
        }, 10, 50);
        printf("[T9] i8 CPT=8       = %8.3f ms  (vs half CPT=8 %.3f)\n", t9, t7[3]);
        const int grid10 = (int)(((size_t)N * (C / 16) + 255) / 256);
        auto t10 = timed_ms([&]() {
            gather_v3_i8w_kernel<16><<<grid10, 256>>>(d_out, d_feat8, (PgEnt*)d_ent_real,
                                                      d_cnt, N, CS, 0.05f);
        }, 10, 50);
        printf("[T10] i8 CPT=16     = %8.3f ms  (vs best half %.3f)\n", t10, best7);
        cudaFree(d_feat8);
    }
    printf("ATTRIB: DRAM part = %.2f ms; instr/latency part = %.2f ms\n",
           t1 - (t2 - t3), t2 - t3);
    printf("BENCH_REAL_DONE\n");
    return 0;
}
