// Standalone unit test for the DFA forward kernel.
// Cases: (A) all fp32, (B) int8 feat, (C) fp16 loc/w, (D) int8 feat + fp16 loc/w.
// Each GPU path is compared against a double-precision CPU reference that mirrors
// the kernel indexing exactly (including fp16 rounding of loc/w where applicable).
// Build:  nvcc -O3 -std=c++14 -I <trt-include> dfa_plugin.cu dfa_i8_test.cu -o dfa_i8_test -lcudart
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void dfa_forward_cuda(
    float* output, const float* feat, const int8_t* feat_i8, const float* feat_scale,
    const int* spatial_shape, const int* scale_start_index,
    const float* loc_f, const __half* loc_h,
    const float* w_f, const __half* w_h,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, cudaStream_t stream);

static const int bs = 1, cams = 2, levels = 2, C = 16, groups = 4, pts = 2000;
static const int ss[4] = {16, 24, 8, 12};
static const int HW = ss[0] * ss[1] + ss[2] * ss[3];
static const int ssi[2] = {0, ss[0] * ss[1]};

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-4.f, 4.f), u01(0.02f, 0.98f), up(0.1f, 1.f);

    std::vector<float> feat(cams * HW * C), feat_q(cams * HW * C), scale(C);
    std::vector<int8_t> feat_i8(cams * HW * C);
    std::vector<float> loc(pts * cams * 2), w(pts * cams * levels * groups);
    std::vector<__half> loc_h(loc.size()), w_h(w.size());

    for (int c = 0; c < C; ++c) {
        float amax = 0.f;
        for (int i = 0; i < cams * HW; ++i) amax = std::max(amax, std::fabs(feat[i * C + c] = u(rng)));
        scale[c] = std::max(amax, 1e-6f) / 127.f;
        for (int i = 0; i < cams * HW; ++i) {
            int q = (int)std::lround(feat[i * C + c] / scale[c]);
            q = std::max(-127, std::min(127, q));
            feat_i8[i * C + c] = (int8_t)q;
            feat_q[i * C + c] = (float)q * scale[c];
        }
    }
    for (size_t i = 0; i < loc.size(); ++i) { loc[i] = u01(rng); loc_h[i] = __float2half(loc[i]); }
    for (size_t i = 0; i < w.size(); ++i) { w[i] = up(rng); w_h[i] = __float2half(w[i]); }

    float *d_feat, *d_feat_q, *d_scale, *d_loc, *d_w, *d_out;
    int8_t* d_i8;
    __half *d_loc_h, *d_w_h;
    int *d_ss, *d_ssi;
    cudaMalloc(&d_feat, feat.size() * 4); cudaMalloc(&d_feat_q, feat_q.size() * 4);
    cudaMalloc(&d_i8, feat_i8.size()); cudaMalloc(&d_scale, scale.size() * 4);
    cudaMalloc(&d_ss, 16); cudaMalloc(&d_ssi, 8);
    cudaMalloc(&d_loc, loc.size() * 4); cudaMalloc(&d_w, w.size() * 4);
    cudaMalloc(&d_loc_h, loc.size() * 2); cudaMalloc(&d_w_h, w.size() * 2);
    cudaMalloc(&d_out, pts * C * 4);
    cudaMemcpy(d_feat, feat.data(), feat.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat_q, feat_q.data(), feat_q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_i8, feat_i8.data(), feat_i8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, scale.data(), scale.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ss, ss, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ssi, ssi, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_loc, loc.data(), loc.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_loc_h, loc_h.data(), loc.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_h, w_h.data(), w.size() * 2, cudaMemcpyHostToDevice);

    // CPU reference with optional int8 feat and fp16-rounded loc / w
    auto cpu_ref = [&](std::vector<float>& out, bool use_i8, bool use_fp16_loc, bool use_fp16_w) {
        out.assign(pts * C, 0.f);
        for (int n = 0; n < pts; ++n)
            for (int c = 0; c < C; ++c) {
                double sum = 0;
                int g = c / (C / groups);
                for (int cam = 0; cam < cams; ++cam) {
                    double lw0 = loc[(n * cams + cam) * 2], lh0 = loc[(n * cams + cam) * 2 + 1];
                    if (use_fp16_loc) {
                        lw0 = __half2float(__float2half((float)lw0));
                        lh0 = __half2float(__float2half((float)lh0));
                    }
                    if (!(lw0 > 0 && lw0 < 1 && lh0 > 0 && lh0 < 1)) continue;
                    for (int s = 0; s < levels; ++s) {
                        int h = ss[s * 2], wd = ss[s * 2 + 1];
                        double h_im = lh0 * h - 0.5, w_im = lw0 * wd - 0.5;
                        int h_l = (int)std::floor(h_im), w_l = (int)std::floor(w_im);
                        int h_h2 = h_l + 1, w_h2 = w_l + 1;
                        double lhf = h_im - h_l, lwf = w_im - w_l;
                        double hh = 1 - lhf, hw = 1 - lwf;
                        int w_stride = C, h_stride = wd * C;
                        int base = cam * HW * C + ssi[s] * C + c;
                        double wgt = w[((n * cams + cam) * levels + s) * groups + g];
                        if (use_fp16_w) wgt = __half2float(__float2half((float)wgt));
                        double acc = 0;
                        auto tap = [&](int hh_, int ww_) -> double {
                            bool ok = hh_ >= 0 && hh_ <= h - 1 && ww_ >= 0 && ww_ <= wd - 1;
                            if (!ok) return 0.0;
                            int idx = base + hh_ * h_stride + ww_ * w_stride;
                            return use_i8 ? (double)feat_i8[idx] * (double)scale[c] : (double)feat[idx];
                        };
                        acc += hh * hw * tap(h_l, w_l);
                        acc += hh * lwf * tap(h_l, w_h2);
                        acc += lhf * hw * tap(h_h2, w_l);
                        acc += lhf * lwf * tap(h_h2, w_h2);
                        sum += acc * wgt;
                    }
                }
                out[n * C + c] = (float)sum;
            }
    };

    std::vector<float> ref_a, ref_b, ref_c, ref_d, ref_e, ref_f;
    cpu_ref(ref_a, false, false, false);  // A: all fp32
    cpu_ref(ref_b, true, false, false);   // B: int8 feat
    cpu_ref(ref_c, false, true, true);    // C: fp16 loc/w
    cpu_ref(ref_d, true, true, true);     // D: int8 feat + fp16 loc/w
    cpu_ref(ref_e, true, true, false);    // E: int8 feat + fp16 loc only
    cpu_ref(ref_f, true, false, true);    // F: int8 feat + fp16 w only

    std::vector<float> got(pts * C);
    int ok = 1;
    auto run_case = [&](const char* tag, float* d_out, std::vector<float>& host,
                        const float* f, const int8_t* fi8, const float* fs,
                        const float* lf, const __half* lh, const float* wf, const __half* wh,
                        const std::vector<float>& cpu) {
        cudaMemset(d_out, 0, pts * C * 4);
        dfa_forward_cuda(d_out, f, fi8, fs, d_ss, d_ssi, lf, lh, wf, wh,
                         bs, cams, HW, C, levels, pts, groups, 0);
        cudaMemcpy(host.data(), d_out, pts * C * 4, cudaMemcpyDeviceToHost);
        double maxabs = 0, rel2 = 0, n2 = 0;
        for (int i = 0; i < pts * C; ++i) {
            maxabs = std::max(maxabs, (double)std::fabs(host[i] - cpu[i]));
            rel2 += std::pow(host[i] - cpu[i], 2);
            n2 += std::pow(cpu[i], 2);
        }
        bool pass = maxabs < 2e-3;
        printf("%s: maxabs=%.3e relL2=%.3e %s\n", tag, maxabs,
               std::sqrt(rel2 / std::max(n2, 1e-12)), pass ? "OK" : "FAIL");
        ok &= pass ? 1 : 0;
    };

    run_case("A all-fp32             ", d_out, got, d_feat, nullptr, nullptr, d_loc, nullptr, d_w, nullptr, ref_a);
    run_case("B int8-feat            ", d_out, got, nullptr, d_i8, d_scale, d_loc, nullptr, d_w, nullptr, ref_b);
    run_case("C fp16 loc/w           ", d_out, got, d_feat, nullptr, nullptr, nullptr, d_loc_h, nullptr, d_w_h, ref_c);
    run_case("D int8-feat+fp16 loc+w ", d_out, got, nullptr, d_i8, d_scale, nullptr, d_loc_h, nullptr, d_w_h, ref_d);
    run_case("E int8-feat+fp16 loc   ", d_out, got, nullptr, d_i8, d_scale, nullptr, d_loc_h, d_w, nullptr, ref_e);
    run_case("F int8-feat+fp16 w     ", d_out, got, nullptr, d_i8, d_scale, d_loc, nullptr, nullptr, d_w_h, ref_f);

    printf("TEST_%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
