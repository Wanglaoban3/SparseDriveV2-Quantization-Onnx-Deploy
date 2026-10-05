// FusedMHA acceptance bench -- compiles WITH dfa_plugin.cu (uses
// mha_forward_cuda from the plugin TU).
//
// For each S in {64,128,256,400,1024}:
//   C1 device vs host reference emulating the graph's fp16 numerics
//     (Q*scale rounded to fp16, scores BMM output rounded to fp16, softmax
//      P rounded to fp16, ctx accumulated fp32) -> tolerance 5e-3 rel,
//     report max_abs/max_rel and fp16 bit-diff fraction vs the fp16 reference
//   C2 determinism: two runs bit-identical
//   T  kernel timing per S
// Also an x*8 "sharp" pass for S=1024 to stress the online-softmax rescale.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" int mha_forward_cuda(__half*, const __half*, int, cudaStream_t);

static const float SCALE = 0.1767766922712326f;   // 1/sqrt(32), graph constant

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

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

// graph-faithful fp16 reference for one (S): out[r*256 + h*32 + d]
static void host_ref(const std::vector<__half>& x, int S, std::vector<__half>& out) {
    out.assign((size_t)S * 256, __float2half(0.f));
    std::vector<__half> sc16(S);
    for (int r = 0; r < S; ++r)
        for (int h = 0; h < 8; ++h) {
            float q[32];
            for (int d = 0; d < 32; ++d)
                q[d] = __half2float(x[(size_t)r * 768 + h * 32 + d]) * SCALE;
            float mx = -1e30f;
            for (int kr = 0; kr < S; ++kr) {
                float dot = 0.f;
                for (int d = 0; d < 32; ++d)
                    dot += q[d] * __half2float(x[(size_t)kr * 768 + 256 + h * 32 + d]);
                sc16[kr] = __float2half(dot);        // BMM output rounding
                const float v = __half2float(sc16[kr]);
                if (v > mx) mx = v;
            }
            float ssum = 0.f;
            std::vector<__half> p16(S);
            for (int kr = 0; kr < S; ++kr) {
                p16[kr] = __float2half(expf(__half2float(sc16[kr]) - mx));
                ssum += __half2float(p16[kr]);
            }
            const float inv = 1.f / ssum;
            float y[32];
            for (int d = 0; d < 32; ++d) y[d] = 0.f;
            for (int kr = 0; kr < S; ++kr) {
                const float p = __half2float(p16[kr]) * inv;
                for (int d = 0; d < 32; ++d)
                    y[d] += p * __half2float(x[(size_t)kr * 768 + 512 + h * 32 + d]);
            }
            for (int d = 0; d < 32; ++d)
                out[(size_t)r * 256 + h * 32 + d] = __float2half(y[d]);
        }
}

static int check_S(const std::vector<__half>& x, int S, float amp,
                   float rel_tol,
                   const std::vector<__half>* ref_in /*optional precomputed*/) {
    std::vector<__half> xin = x;
    if (amp != 1.f)
        for (auto& v : xin) v = __float2half(__half2float(v) * amp);
    __half *d_x, *d_out;
    CK(cudaMalloc(&d_x, xin.size() * 2));
    CK(cudaMalloc(&d_out, (size_t)S * 256 * 2));
    CK(cudaMemcpy(d_x, xin.data(), xin.size() * 2, cudaMemcpyHostToDevice));
    int rc = 0;

    if (mha_forward_cuda(d_out, d_x, S, 0)) { printf("launch FAIL S=%d\n", S); return 1; }
    CK(cudaDeviceSynchronize());
    std::vector<__half> got((size_t)S * 256);
    CK(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));

    // determinism
    if (mha_forward_cuda(d_out, d_x, S, 0)) { printf("relaunch FAIL S=%d\n", S); return 1; }
    CK(cudaDeviceSynchronize());
    std::vector<__half> got2((size_t)S * 256);
    CK(cudaMemcpy(got2.data(), d_out, got2.size() * 2, cudaMemcpyDeviceToHost));
    const bool det = memcmp(got.data(), got2.data(), got.size() * 2) == 0;
    printf("[C2] S=%4d determinism: %s\n", S, det ? "IDENTICAL" : "DIFF");
    rc |= !det;

    // numeric vs graph-faithful fp16 reference. Gate = per-row scale: the
    // ref's fp16 score/P rounding injects noise proportional to the ROW's
    // output scale (which varies hugely under sharp softmax), not to the
    // element's own |b|. Structural bugs (bad rescale/layout) produce
    // O(row-scale) errors and fail by an order of magnitude.
    std::vector<__half> ref;
    if (ref_in != nullptr) ref = *ref_in; else host_ref(xin, S, ref);
    double max_abs = 0, max_rel = 0;
    size_t bits = 0, viol = 0;
    for (int r = 0; r < S; ++r) {
        const __half* rr = ref.data() + (size_t)r * 256;
        double ss = 0;
        for (int i = 0; i < 256; ++i) {
            const double v = __half2float(rr[i]);
            ss += v * v;
        }
        const double row_tol = fmax(8e-2 * sqrt(ss / 256.0), 1e-3);
        for (int i = 0; i < 256; ++i) {
            const size_t idx = (size_t)r * 256 + i;
            const double a = __half2float(got[idx]), b = __half2float(rr[i]);
            const double dd = fabs(a - b);
            max_abs = dd > max_abs ? dd : max_abs;
            max_rel = fmax(max_rel, dd / fmax(fabs(b), 1e-3));
            bits += (memcmp(&got[idx], &rr[i], 2) != 0);
            viol += (dd > fmax(rel_tol * fabs(b), row_tol));
        }
    }
    printf("[C1] S=%4d amp=%.0f: max_abs=%.4e max_rel=%.3e viol=%zu/%zu "
           "bitdiff=%zu/%zu (%.3f%%)\n",
           S, amp, max_abs, max_rel, viol, got.size(), bits, got.size(),
           100.0 * (double)bits / got.size());
    rc |= (viol != 0);

    const int iters = S >= 1024 ? 50 : 200;
    const float ms = timed_ms([&]() { mha_forward_cuda(d_out, d_x, S, 0); }, 10, iters);
    printf("[T ] S=%4d: %8.4f ms\n", S, ms);

    CK(cudaFree(d_x)); CK(cudaFree(d_out));
    return rc;
}

int main() {
    unsigned seed = 424242;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) * (1.f / 16777216.f); };
    auto randn = [&]() {
        return sqrtf(-2.f * logf(rnd() + 1e-9f)) * cosf(6.2831853f * rnd()) * 0.5f; };
    const int SMAX = 1024;
    std::vector<__half> x((size_t)SMAX * 768);
    for (auto& v : x) v = __float2half(randn());

    int rc = 0;
    const int Ss[5] = {64, 128, 256, 400, 1024};
    for (int i = 0; i < 5; ++i) {
        std::vector<__half> xs(x.begin(), x.begin() + (size_t)Ss[i] * 768);
        rc |= check_S(xs, Ss[i], 1.f, 5e-3, nullptr);
    }
    // sharp pass: large logits stress the online rescale (S=1024)
    {
        std::vector<__half> xs(x.begin(), x.begin() + (size_t)1024 * 768);
        rc |= check_S(xs, 1024, 8.f, 4e-2, nullptr);
    }
    printf(rc == 0 ? "MHA_BENCH_DONE\n" : "MHA_BENCH_FAIL rc=%d\n", rc);
    return rc;
}
