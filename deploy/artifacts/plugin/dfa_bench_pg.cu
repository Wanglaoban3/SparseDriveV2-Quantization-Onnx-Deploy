// Pg (plan+gather+inline-softmax) micro-benchmark: correctness vs a naive
// reference softmax + the v1 native kernel, run-to-run determinism, and
// per-phase perf under two loc distributions (worst=all-in-bounds,
// real=1-of-3 cams visible approximating the frustum mask).
//
// Build (board):
//   nvcc -O3 -std=c++17 -arch=sm_87 -I /usr/src/tensorrt/include \
//        dfa_plugin.cu dfa_bench_pg.cu -o /usr/local/bin/dfa_bench_pg \
//        -lcudart -lnvinfer
// Usage: dfa_bench_pg [real|worst] [iters] [eps]
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <random>
#include <vector>

extern "C" void dfa_forward_half_cuda(
    __half* output, const __half* feat,
    const int* spatial_shape, const int* scale_start_index,
    const float* loc_f, const __half* loc_h,
    const float* w_f, const __half* w_h,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, cudaStream_t stream);

extern "C" int dfa_pg_forward_cuda(
    __half* out, const __half* feat,
    const int* spatial_shape, const int* scale_start_index,
    const float* loc_f, const __half* loc_h,
    const void* logits_v, int logits_is_half,
    void* workspace,
    int bs, int cams, int hw, int C, int levels, int N, int A, int G,
    float eps, cudaStream_t stream);
extern "C" int dfa_pg_plan_cuda(
    void* entries_v, int* counts,
    const void* logits_v, int logits_is_half,
    const float* loc_f, const __half* loc_h,
    const int* spatial_shape, const int* scale_start_index,
    int bs, int cams, int hw, int C, int levels, int N, int A, int G,
    float eps, cudaStream_t stream);
extern "C" int dfa_pg_gather_cuda(
    __half* out, const __half* feat,
    const void* entries_v, const int* counts,
    int bs, int C, int G, int cams, int levels, int N,
    cudaStream_t stream);
extern "C" size_t dfa_pg_workspace_bytes(int rows, int cams, int levels);

#define CK(x) do { cudaError_t s_ = (x); if (s_ != cudaSuccess) { \
    printf("CUDA error %s @%d\n", cudaGetErrorString(s_), __LINE__); return 1; } } while (0)

// ---- naive reference softmax (obviously correct, deterministic) -------------
// one 128-thread block per (anchor, group); writes w in the v1 plugin layout
__global__ void ref_softmax_kernel(const float* logits, float* w_out,
                                   int A, int pts, int cams, int levels,
                                   int G) {
    const int id = blockIdx.x;
    const int g = id % G;
    const int a = id / G;
    const int CS = cams * levels;
    const int ISP = CS * pts;
    const float* lrow = logits + (size_t)a * ISP * G + g;
    __shared__ float red[128];
    float mx = -3.0e38f;
    for (int i = threadIdx.x; i < ISP; i += blockDim.x)
        mx = fmaxf(mx, lrow[(size_t)i * G]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    float sum = 0.f;
    for (int i = threadIdx.x; i < ISP; i += blockDim.x)
        sum += expf(lrow[(size_t)i * G] - mx);
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    sum = red[0];
    for (int i = threadIdx.x; i < ISP; i += blockDim.x) {
        const float wv = expf(lrow[(size_t)i * G] - mx) / sum;
        const int cs = i / pts, pt = i % pts;
        const int cam = cs / levels, level = cs % levels;
        w_out[((size_t)a * pts + pt) * cams * levels * G +
              (size_t)cam * levels * G + level * G + g] = wv;
    }
}

static float timed_ms(int iters, int warmup, const std::function<void()>& fn,
                      cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) fn();
    CK(cudaStreamSynchronize(stream));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "real";
    const int iters = argc > 2 ? atoi(argv[2]) : 20;
    const float eps = argc > 3 ? (float)atof(argv[3]) : 0.0f;
    // fixed deployment geometry (layers.0 p_deform)
    const int bs = 1, A = 1024, pts = 500, N = A * pts;
    const int cams = 3, levels = 4, C = 256, G = 8;
    const int ss_h[8] = {64, 128, 32, 64, 16, 32, 8, 16};
    const int ssi_v[4] = {0, 8192, 10240, 10752};
    const int hw = 10880;

    printf("dfa_bench_pg mode=%s iters=%d eps=%g  (A=%d pts=%d N=%d)\n",
           mode, iters, eps, A, pts, N);
    CK(cudaSetDevice(0));
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    // ---- host gen ----
    std::mt19937 rng(12345);
    std::normal_distribution<float> n01(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.05f, 0.95f);
    std::uniform_real_distribution<float> uout(-1.6f, 1.6f);
    std::uniform_real_distribution<float> ulg(-8.f, 2.f);

    std::vector<__half> feat_h((size_t)bs * cams * hw * C);
    for (auto& v : feat_h) v = __float2half(n01(rng));
    std::vector<__half> loc_h((size_t)N * cams * 2);
    std::vector<float> logits_h((size_t)A * cams * levels * pts * G);
    const int CS = cams * levels;
    // per (n, cam) visibility, then fill loc + -inf logits consistently
    std::vector<unsigned char> vis((size_t)N * cams, 0);
    for (size_t n = 0; n < (size_t)N; ++n) {
        for (int cam = 0; cam < cams; ++cam) {
            bool in;
            if (strcmp(mode, "worst") == 0) {
                in = true;
            } else {                    // real-like: each cam independently
                in = (rng() % 3) != 0;  // ~2/3 cams masked out per point
            }
            vis[n * cams + cam] = in;
            const float wv = in ? u01(rng) : uout(rng);
            const float hv = in ? u01(rng) : uout(rng);
            loc_h[n * cams * 2 + cam * 2] = __float2half(wv);
            loc_h[n * cams * 2 + cam * 2 + 1] = __float2half(hv);
        }
    }
    for (int a = 0; a < A; ++a)
        for (int cs = 0; cs < CS; ++cs) {
            const int cam = cs / levels;
            for (int pt = 0; pt < pts; ++pt) {
                const size_t n = (size_t)a * pts + pt;
                for (int g = 0; g < G; ++g) {
                    logits_h[((size_t)a * CS + cs) * pts * G + (size_t)pt * G + g] =
                        vis[n * cams + cam] ? ulg(rng) : -INFINITY;
                }
            }
        }

    // ---- device alloc ----
    __half *feat, *loc, *out_ref, *out_pg, *out_pg2;
    float *logits, *w_ref;
    int* ss; int* ssi; void* ws; void* ws2;
    CK(cudaMalloc(&feat, feat_h.size() * 2));
    CK(cudaMalloc(&loc, loc_h.size() * 2));
    CK(cudaMalloc(&logits, logits_h.size() * 4));
    CK(cudaMalloc(&w_ref, logits_h.size() * 4));   // same element count
    CK(cudaMalloc(&out_ref, (size_t)N * C * 2));
    CK(cudaMalloc(&out_pg, (size_t)N * C * 2));
    CK(cudaMalloc(&out_pg2, (size_t)N * C * 2));
    CK(cudaMalloc(&ss, sizeof(ss_h)));
    CK(cudaMalloc(&ssi, sizeof(ssi_v)));
    const size_t wsz = dfa_pg_workspace_bytes(N, cams, levels);
    CK(cudaMalloc(&ws, wsz));
    CK(cudaMalloc(&ws2, wsz));
    printf("workspace: %.1f MB\n", wsz / 1048576.0);
    CK(cudaMemcpy(feat, feat_h.data(), feat_h.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(loc, loc_h.data(), loc_h.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(logits, logits_h.data(), logits_h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ss, ss_h, sizeof(ss_h), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ssi, ssi_v, sizeof(ssi_v), cudaMemcpyHostToDevice));

    // ---- reference: naive softmax -> v1 native kernel ----
    ref_softmax_kernel<<<A * G, 128, 0, stream>>>(
        logits, w_ref, A, pts, cams, levels, G);
    dfa_forward_half_cuda(out_ref, feat, ss, ssi, nullptr, loc, w_ref, nullptr,
                          bs, cams, hw, C, levels, N, G, stream);
    CK(cudaStreamSynchronize(stream));

    // ---- pg path ----
    CK(cudaMemset(ws, 0, wsz));
    if (dfa_pg_forward_cuda(out_pg, feat, ss, ssi, nullptr, loc,
                            logits, 0, ws, bs, cams, hw, C, levels, N, A, G,
                            eps, stream)) {
        printf("pg forward FAILED\n"); return 1;
    }
    CK(cudaStreamSynchronize(stream));

    // ---- compare ----
    {
        std::vector<__half> r(N * C), p(N * C);
        CK(cudaMemcpy(r.data(), out_ref, r.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(p.data(), out_pg, p.size() * 2, cudaMemcpyDeviceToHost));
        double max_abs = 0, sum_abs = 0;
        size_t n_bad = 0, n_nz = 0;
        for (size_t i = 0; i < r.size(); ++i) {
            const double dv = __half2float(r[i]) - __half2float(p[i]);
            const double ad = fabs(dv);
            sum_abs += ad;
            if (ad > max_abs) max_abs = ad;
            if (ad > 2e-3f) ++n_bad;
            if (__half2float(p[i]) != 0.f) ++n_nz;
        }
        printf("compare pg vs ref: max_abs=%.3e mean_abs=%.3e n(|d|>2e-3)=%zu  (nonzero outs=%zu/%zu)\n",
               max_abs, sum_abs / r.size(), n_bad, n_nz, r.size());
        printf("  %s (max_abs threshold 2e-3)\n", max_abs < 2e-3 ? "PASS" : "FAIL");
    }

    // ---- determinism: second pg run into fresh workspace ----
    {
        CK(cudaMemset(ws2, 0, wsz));
        if (dfa_pg_forward_cuda(out_pg2, feat, ss, ssi, nullptr, loc,
                                logits, 0, ws2, bs, cams, hw, C, levels, N, A, G,
                                eps, stream)) {
            printf("pg forward #2 FAILED\n"); return 1;
        }
        CK(cudaStreamSynchronize(stream));
        std::vector<__half> a(N * C), b(N * C);
        CK(cudaMemcpy(a.data(), out_pg, a.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), out_pg2, b.size() * 2, cudaMemcpyDeviceToHost));
        printf("determinism (two runs bitwise): %s\n",
            memcmp(a.data(), b.data(), a.size() * 2) == 0 ? "MATCH" : "DIFFER");
    }

    // ---- entries stats ----
    {
        std::vector<int> cnt(N);
        const size_t cnt_off = (size_t)N * CS * 64;   // counts live after entries
        CK(cudaMemcpy(cnt.data(), (char*)ws + cnt_off, N * 4,
                      cudaMemcpyDeviceToHost));
        double k = 0;
        for (int v : cnt) k += v;
        printf("entries/point: avg=%.2f max=%d\n", k / N,
               *std::max_element(cnt.begin(), cnt.end()));
    }

    // ---- perf ----
    auto fn_native = [&] {
        dfa_forward_half_cuda(out_ref, feat, ss, ssi, nullptr, loc, w_ref,
                              nullptr, bs, cams, hw, C, levels, N, G, stream);
    };
    auto fn_pg = [&] {
        dfa_pg_forward_cuda(out_pg, feat, ss, ssi, nullptr, loc, logits, 0, ws,
                            bs, cams, hw, C, levels, N, A, G, eps, stream);
    };
    auto fn_plan = [&] {
        dfa_pg_plan_cuda(ws, (int*)((char*)ws + (size_t)N * CS * 64), logits, 0,
                         nullptr, loc, ss, ssi, bs, cams, hw, C, levels, N, A, G,
                         eps, stream);
    };
    auto fn_gather = [&] {
        dfa_pg_gather_cuda(out_pg, feat, ws, (int*)((char*)ws + (size_t)N * CS * 64),
                           bs, C, G, cams, levels, N, stream);
    };
    printf("native v1 kernel : %8.2f ms\n", timed_ms(iters, 3, fn_native, stream));
    printf("pg total         : %8.2f ms\n", timed_ms(iters, 3, fn_pg, stream));
    printf("pg plan only     : %8.2f ms\n", timed_ms(iters, 3, fn_plan, stream));
    printf("pg gather only   : %8.2f ms\n", timed_ms(iters, 3, fn_gather, stream));
    return 0;
}
