// SparseDriveV2 DFA kernel micro-benchmark (plan Task 7 Step 2).
// Replicates the plugin's fp16-engine enqueue path phase by phase:
//   [1] cast_half_to_float on feat  (bs,cams,HW,C) fp16 -> fp32 workspace
//   [2] dfa_forward_cuda core kernel with fp16 loc / fp16 w direct read
//   [3] cast_float_to_half on output (bs,pts,C) fp32 workspace -> fp16
// Real shapes: cams=3, levels=4 (HW=10880: 64x128+32x64+16x32+8x16), C=256,
// groups=8, pts in {512000, 64000, 32000}. 100 timed iters per shape after 3
// warmups, cudaEvent timing per phase, all device-resident (no h2d in loop).
// Build:  nvcc -O3 -std=c++14 -I /usr/src/tensorrt/include dfa_plugin.cu dfa_bench.cu \
//             -o /usr/local/bin/dfa_bench -lcudart -lnvinfer
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" void dfa_forward_cuda(
    float* output, const float* feat, const int8_t* feat_i8, const float* feat_scale,
    const int* spatial_shape, const int* scale_start_index,
    const float* loc_f, const __half* loc_h,
    const float* w_f, const __half* w_h,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, cudaStream_t stream);

extern "C" void dfa_forward_half_cuda(
    __half* output, const __half* feat,
    const int* spatial_shape, const int* scale_start_index,
    const float* loc_f, const __half* loc_h,
    const float* w_f, const __half* w_h,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, cudaStream_t stream);

// Same bodies as the plugin's internal cast kernels (anonymous namespace there,
// not linkable across TUs).
__global__ void bench_cast_h2f(const __half* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
__global__ void bench_cast_f2h(const float* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

static const int bs = 1, cams = 3, levels = 4, C = 256, groups = 8;
static const int ss[8] = {64, 128, 32, 64, 16, 32, 8, 16};
static const int ssi[4] = {0, 8192, 10240, 10752};
static const int HW = 64 * 128 + 32 * 64 + 16 * 32 + 8 * 16;  // 10880

int run_shape(int pts, int iters) {
    const int feat_e = cams * HW * C;
    const int loc_e = pts * cams * 2;
    const int w_e = pts * cams * levels * groups;
    const int out_e = pts * C;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uf(-4.f, 4.f), u01(0.02f, 0.98f), up(0.1f, 1.f);
    std::vector<__half> h_feat(feat_e), h_loc(loc_e), h_w(w_e);
    for (int i = 0; i < feat_e; ++i) h_feat[i] = __float2half(uf(rng));
    for (int i = 0; i < loc_e; ++i) h_loc[i] = __float2half(u01(rng));
    for (int i = 0; i < w_e; ++i) h_w[i] = __float2half(up(rng));

    __half *d_feat, *d_loc, *d_w, *d_out_h, *d_out_h2;
    float *d_feat_f, *d_out_f;
    int *d_ss, *d_ssi;
    std::vector<__half> h_a(out_e), h_b(out_e);
    if (cudaMalloc(&d_feat, feat_e * 2) != cudaSuccess ||
        cudaMalloc(&d_loc, (size_t)loc_e * 2) != cudaSuccess ||
        cudaMalloc(&d_w, (size_t)w_e * 2) != cudaSuccess ||
        cudaMalloc(&d_out_h, (size_t)out_e * 2) != cudaSuccess ||
        cudaMalloc(&d_out_h2, (size_t)out_e * 2) != cudaSuccess ||
        cudaMalloc(&d_feat_f, feat_e * 4) != cudaSuccess ||
        cudaMalloc(&d_out_f, (size_t)out_e * 4) != cudaSuccess ||
        cudaMalloc(&d_ss, 8 * 4) != cudaSuccess ||
        cudaMalloc(&d_ssi, 4 * 4) != cudaSuccess) {
        printf("cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(d_feat, h_feat.data(), feat_e * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_loc, h_loc.data(), (size_t)loc_e * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), (size_t)w_e * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ss, ss, 8 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ssi, ssi, 4 * 4, cudaMemcpyHostToDevice);

    cudaEvent_t ev[6];
    for (auto& e : ev) cudaEventCreate(&e);
    float acc[3] = {0, 0, 0}, accn = 0;
    for (int it = 0; it < iters + 3; ++it) {
        cudaEventRecord(ev[0]);
        bench_cast_h2f<<<(feat_e + 511) / 512, 512>>>(d_feat, d_feat_f, feat_e);
        cudaEventRecord(ev[1]);
        dfa_forward_cuda(d_out_f, d_feat_f, nullptr, nullptr, d_ss, d_ssi,
                         nullptr, d_loc, nullptr, d_w,
                         bs, cams, HW, C, levels, pts, groups, 0);
        cudaEventRecord(ev[2]);
        bench_cast_f2h<<<(out_e + 511) / 512, 512>>>(d_out_f, d_out_h, out_e);
        cudaEventRecord(ev[3]);
        // native fp16-IO path (plan A): single kernel, no casts
        cudaEventRecord(ev[4]);
        dfa_forward_half_cuda(d_out_h2, d_feat, d_ss, d_ssi,
                              nullptr, d_loc, nullptr, d_w,
                              bs, cams, HW, C, levels, pts, groups, 0);
        cudaEventRecord(ev[5]);
        cudaEventSynchronize(ev[5]);
        if (it == 3) {  // one-shot bitwise equivalence of the two paths
            if (cudaMemcpy(h_a.data(), d_out_h, (size_t)out_e * 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(h_b.data(), d_out_h2, (size_t)out_e * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
                printf("D2H failed\n");
                return 1;
            }
            int bad = 0;
            if (std::memcmp(h_a.data(), h_b.data(), (size_t)out_e * 2) != 0) {
                for (int i = 0; i < out_e; ++i) {
                    unsigned short x, y;
                    std::memcpy(&x, &h_a[i], 2);
                    std::memcpy(&y, &h_b[i], 2);
                    if (x != y) {
                        ++bad;
                        if (bad <= 3)
                            printf("  diff@%d casted=%f native=%f\n", i,
                                   __half2float(h_a[i]), __half2float(h_b[i]));
                    }
                }
            }
            printf("BITWISE casted-vs-native: %s (%d/%d halfs differ)\n",
                   bad ? "MISMATCH" : "MATCH", bad, out_e);
        }
        if (it >= 3) {
            float ms01, ms12, ms23, ms45;
            cudaEventElapsedTime(&ms01, ev[0], ev[1]);
            cudaEventElapsedTime(&ms12, ev[1], ev[2]);
            cudaEventElapsedTime(&ms23, ev[2], ev[3]);
            cudaEventElapsedTime(&ms45, ev[4], ev[5]);
            acc[0] += ms01;
            acc[1] += ms12;
            acc[2] += ms23;
            accn += ms45;
        }
    }
    printf("SHAPE pts=%d cast_in=%.4f core=%.4f cast_out=%.4f total=%.4f native=%.4f ms (mean of %d)\n",
           pts, acc[0] / iters, acc[1] / iters, acc[2] / iters,
           (acc[0] + acc[1] + acc[2]) / iters, accn / iters, iters);
    for (auto& e : ev) cudaEventDestroy(e);
    cudaFree(d_feat); cudaFree(d_loc); cudaFree(d_w); cudaFree(d_out_h);
    cudaFree(d_out_h2); cudaFree(d_feat_f); cudaFree(d_out_f);
    cudaFree(d_ss); cudaFree(d_ssi);
    return cudaGetLastError() != cudaSuccess ? 1 : 0;
}

int main() {
    int rc = 0;
    for (int pts : {512000, 64000, 32000}) rc |= run_shape(pts, 100);
    printf("BENCH_%s\n", rc ? "FAIL" : "DONE");
    return rc;
}
