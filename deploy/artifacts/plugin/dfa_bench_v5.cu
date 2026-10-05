// DFA v5 (sumfusion) acceptance bench -- REAL data (kstat loc/w bins).
//
// libdfa_sd.so v5 changes ONLY the gather: the per-anchor point sum that the
// graph used to do (Reshape [1,A,pts,C] + Cast + ReduceSum axis 2, a 262MB
// fp16 round trip for layers.0/p) is fused into the kernel -- out is now
// [bs, A, C] instead of [bs, N, C]. Plan (entries/counts) is untouched.
// dfa_bench_v4.cu is frozen against the v4 ABI (its gather call has no
// partial/A args). This bench checks the v5 pair with the production
// launchers, two geometries:
//   case A = layers.0/p real geometry: A=1024 pts=500 (SPLIT=1, direct write)
//   case B = A=128 pts=500 (SPLIT=6, fp32 partials + deterministic finalize)
// Checks per case:
//   S1 plan counts == host counts (all N rows)          [plan unchanged]
//   S5 gather out[A,C] vs host fp32 per-anchor reference
//      (per-row expression identical to the v4 N1 host replica; gate:
//       max_rel <= 5e-3 OR max_abs <= 1e-2 -- a grouping bug is O(1))
//   D  determinism: second gather bit-identical
//   T5 gather timing
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" int dfa_pg_plan_cuda(void*, int*, const void*, int, const float*, const __half*,
                                const int*, const int*, int, int, int, int, int, int, int,
                                int, float, cudaStream_t);
extern "C" int dfa_pg_gather_cuda(__half*, const __half*, const void*, const int*, void*,
                                  int, int, int, int, int, int, int, cudaStream_t);

static const int C = 256, G = 8;
static const int CAMS = 3, LEVELS = 4, CS = CAMS * LEVELS, HW = 10880;
static const int SH[8] = {64, 128, 32, 64, 16, 32, 8, 16};      // H,W per level
static const int SSI[4] = {0, 8192, 10240, 10752};
static const char* KDIR = "/opt/m0/sd2/prof/kstat";
static const int N_FULL = 1024 * 500;
static const int RANCH = 32;       // anchors with a full numeric reference

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct E48 { float4 wk; int4 off; __half2 wg[4]; };   // 48B mirror of dfa_pg::Entry

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

static float half2f(__half2 h, int lane) { return __half2float(lane ? h.y : h.x); }

static int g_rc = 0;

// Returns S1/S5/D results for one geometry; loc/w are the full kstat bins
// (rows [0, N) used), feat/logits synthetic.
static void run_case(const char* tag, int A, int PTS,
                     const std::vector<float>& loc_full,
                     const std::vector<float>& w_full,
                     const std::vector<__half>& feat) {
    const int N = A * PTS;
    printf("==== case %s: A=%d pts=%d N=%d (expect SPLIT from A)\n", tag, A, PTS, N);
    std::vector<float> loc(loc_full.begin(), loc_full.begin() + (size_t)N * CAMS * 2);
    std::vector<float> w(w_full.begin(), w_full.begin() + (size_t)N * CAMS * LEVELS * G);

    std::vector<__half> loc_h((size_t)N * CAMS * 2);
    for (size_t i = 0; i < loc_h.size(); ++i) loc_h[i] = __float2half(loc[i]);
    auto locv = [&](int row, int cam, int k) {
        return __half2float(loc_h[((size_t)row * CAMS + cam) * 2 + k]);
    };

    const int ISP = CS * PTS;
    std::vector<__half> logits((size_t)A * ISP * G);
    __half_raw ninf_raw; ninf_raw.x = 0xFC00;
    const __half NEG_INF(ninf_raw);
    unsigned seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) * (1.f / 16777216.f); };
    auto randn = [&]() {
        return sqrtf(-2.f * logf(rnd() + 1e-9f)) * cosf(6.2831853f * rnd()) * 1.5f; };
    std::vector<int> cnt(N);
    std::vector<unsigned short> masks(N);
    const int RROWS = RANCH * PTS;
    std::vector<E48> ent_ref((size_t)RROWS * CS);
    for (int a_i = 0; a_i < A; ++a_i)
        for (int pt = 0; pt < PTS; ++pt) {
            const int row = a_i * PTS + pt;
            unsigned mask = 0;
            for (int cs = 0; cs < CS; ++cs) {
                const int cam = cs / LEVELS;
                const float wv = locv(row, cam, 0);
                const float hv = locv(row, cam, 1);
                const bool guard = wv > 0.f && wv < 1.f && hv > 0.f && hv < 1.f;
                const float* wg = &w[((size_t)row * CAMS + cam) * LEVELS * G + (cs % LEVELS) * G];
                bool any = false;
                for (int g = 0; g < G; ++g) any |= (wg[g] > 0.f);
                __half* p = &logits[((size_t)a_i * ISP + cs * PTS + pt) * G];
                for (int g = 0; g < G; ++g)
                    p[g] = (guard && any) ? __float2half(randn()) : NEG_INF;
                if (guard && any) mask |= 1u << cs;
            }
            cnt[row] = __builtin_popcount(mask);
            masks[row] = (unsigned short)mask;
        }
    // Anchor-level per-group softmax stats over the WHOLE ISP axis -- what the
    // plan kernel's phase A computes (per (anchor, group), denominator spans
    // all cs*pt samples) and phase C consumes (wg = exp(l-m)/s). MUST run
    // AFTER the logits fill loop.
    std::vector<double> stat_m((size_t)RANCH * G), stat_s((size_t)RANCH * G);
    for (int a_i = 0; a_i < RANCH; ++a_i) {
        double m[G], s[G];
        for (int g = 0; g < G; ++g) { m[g] = -1e300; s[g] = 0.0; }
        for (int i = 0; i < ISP; ++i) {
            const __half* lp = &logits[((size_t)a_i * ISP + i) * G];
            for (int g = 0; g < G; ++g) {
                const double v = __half2float(lp[g]);
                if (v == -INFINITY) continue;         // exp(-inf - m) = 0
                if (v > m[g]) {
                    s[g] *= exp(m[g] - v);
                    m[g] = v;
                }
                s[g] += exp(v - m[g]);
            }
        }
        for (int g = 0; g < G; ++g) {
            stat_m[(size_t)a_i * G + g] = m[g];
            stat_s[(size_t)a_i * G + g] = s[g];
        }
    }
    // detail entries for the first RROWS rows, with the anchor-level softmax wg
    for (int row = 0; row < RROWS; ++row) {
        const int a_i = row / PTS, pt = row % PTS;
        int slot = 0;
        for (int cs = 0; cs < CS; ++cs) {
            if (!((masks[row] >> cs) & 1u)) continue;
            const int cam = cs / LEVELS, lv = cs % LEVELS;
            const float wv = locv(row, cam, 0);
            const float hv = locv(row, cam, 1);
            const int H = SH[lv * 2], W = SH[lv * 2 + 1], ssi = SSI[lv];
            const float h_im = (float)((double)(hv * (float)H) - 0.5);
            const float w_im = (float)((double)(wv * (float)W) - 0.5);
            const int h_low = (int)floorf(h_im), w_low = (int)floorf(w_im);
            const int h_high = h_low + 1, w_high = w_low + 1;
            const float lh = h_im - h_low, lw = w_im - w_low;
            const float hh = 1 - lh, hw2 = 1 - lw;
            const int w_stride = C, h_stride = W * w_stride;
            const int base = cam * HW * C + ssi * C;
            E48 e; memset(&e, 0, sizeof(e));
            e.wk.x = (h_low >= 0 && w_low >= 0) ? hh * hw2 : 0.f;
            e.off.x = (e.wk.x != 0.f) ? base + h_low * h_stride + w_low * w_stride : 0;
            e.wk.y = (h_low >= 0 && w_high <= W - 1) ? hh * lw : 0.f;
            e.off.y = (e.wk.y != 0.f) ? base + h_low * h_stride + w_high * w_stride : 0;
            e.wk.z = (h_high <= H - 1 && w_low >= 0) ? lh * hw2 : 0.f;
            e.off.z = (e.wk.z != 0.f) ? base + h_high * h_stride + w_low * w_stride : 0;
            e.wk.w = (h_high <= H - 1 && w_high <= W - 1) ? lh * lw : 0.f;
            e.off.w = (e.wk.w != 0.f) ? base + h_high * h_stride + w_high * w_stride : 0;
            const __half* lp = &logits[((size_t)a_i * ISP + cs * PTS + pt) * G];
            __half2* wh = e.wg;
            for (int g = 0; g < G; ++g) {
                const double v = __half2float(lp[g]);
                const double wgd = (v == -INFINITY)
                    ? 0.0
                    : exp(v - stat_m[(size_t)a_i * G + g]) /
                      stat_s[(size_t)a_i * G + g];
                const __half h = __float2half((float)wgd);
                if (g & 1) wh[g >> 1].y = h; else wh[g >> 1].x = h;
                if (row == 18 && slot == 0 && g < 4)
                    printf("[D2] row18 g=%d v=%.4e stat_m=%.3f stat_s=%.3e "
                           "wgd=%.4e half=0x%04x\n",
                           g, v, stat_m[(size_t)a_i * G + g],
                           stat_s[(size_t)a_i * G + g], wgd,
                           *(unsigned short*)&h);
            }
            ent_ref[(size_t)row * CS + (slot++)] = e;
        }
    }
    {
        const E48& e0 = ent_ref[18 * CS];
        printf("[Q ] ent_ref[18].wk.x=%e wg hex=%08x %08x %08x %08x\n",
               e0.wk.x, *(unsigned*)&e0.wg[0], *(unsigned*)&e0.wg[1],
               *(unsigned*)&e0.wg[2], *(unsigned*)&e0.wg[3]);
        printf("[Q ] stat a0: m0=%.3f m1=%.3f s0=%.3e s1=%.3e\n",
               stat_m[0], stat_m[1], stat_s[0], stat_s[1]);
        printf("[Q ] masks[18]=0x%04x masks[0]=0x%04x masks[1]=0x%04x\n",
               masks[18], masks[0], masks[1]);
        for (int cs = 0; cs < CS; ++cs) {
            const __half* lp = &logits[(size_t)cs * PTS + 18];
            printf("[Q ] a0 cs=%d pt=18 lg0=%08x lg1=%08x (inf=%d%d)\n",
                   cs, *(unsigned*)&lp[0], *(unsigned*)&lp[1],
                   __half2float(lp[0]) == -INFINITY,
                   __half2float(lp[1]) == -INFINITY);
        }
        for (int r2 : {0, 1, 2, 500, 501})
            printf("[Q ] ent_ref[%d] wg=%08x %08x %08x %08x\n", r2,
                   *(unsigned*)&ent_ref[(size_t)r2 * CS].wg[0],
                   *(unsigned*)&ent_ref[(size_t)r2 * CS].wg[1],
                   *(unsigned*)&ent_ref[(size_t)r2 * CS].wg[2],
                   *(unsigned*)&ent_ref[(size_t)r2 * CS].wg[3]);
    }

    // ---- device buffers ----
    const size_t ent_bytes = (size_t)N * CS * 48 + (size_t)N * 4 + 256;
    E48* d_ent; int* d_cnt;
    CK(cudaMalloc(&d_ent, ent_bytes));
    CK(cudaMemset(d_ent, 0, ent_bytes));         // kill tail-slot garbage variable
    CK(cudaMalloc(&d_cnt, (size_t)N * 4));
    __half *d_feat, *d_out, *d_logits, *d_loc;
    CK(cudaMalloc(&d_feat, (size_t)CAMS * HW * C * 2));
    CK(cudaMalloc(&d_out, (size_t)A * C * 2));
    CK(cudaMalloc(&d_logits, logits.size() * 2));
    CK(cudaMalloc(&d_loc, loc_h.size() * 2));
    CK(cudaMemcpy(d_feat, feat.data(), feat.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_logits, logits.data(), logits.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_loc, loc_h.data(), loc_h.size() * 2, cudaMemcpyHostToDevice));
    int *d_sp, *d_ssi;
    CK(cudaMalloc(&d_sp, sizeof(SH))); CK(cudaMalloc(&d_ssi, sizeof(SSI)));
    CK(cudaMemcpy(d_sp, SH, sizeof(SH), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ssi, SSI, sizeof(SSI), cudaMemcpyHostToDevice));

    auto plan = [&]() {
        dfa_pg_plan_cuda(d_ent, d_cnt, d_logits, 1, nullptr, d_loc,
                         d_sp, d_ssi, 1, CAMS, HW, C, LEVELS, N, A, G, 0.f, 0);
    };
    // v5 gather: partial=nullptr is safe when SPLIT==1; for SPLIT>1 the
    // launcher gets a real buffer from the workspace in production -- here
    // we allocate worst-case fp32 partials (split computed host-side).
    int SPLIT, CH;
    {
        int split = (768 + A - 1) / A;
        const int mx = (PTS + 15) / 16;
        if (split > mx) split = mx;
        if (split > PTS) split = PTS;
        if (split < 1) split = 1;
        SPLIT = split; CH = (PTS + split - 1) / split;
    }
    float* d_part = nullptr;
    if (SPLIT > 1) CK(cudaMalloc(&d_part, (size_t)A * SPLIT * C * 4));
    printf("[g ] SPLIT=%d CH=%d warps=%d\n", SPLIT, CH, A * SPLIT);

    auto gather = [&]() {
        dfa_pg_gather_cuda(d_out, d_feat, d_ent, d_cnt, d_part,
                           1, C, G, CAMS, LEVELS, N, A, 0);
    };

    plan();
    CK(cudaDeviceSynchronize());
    int rc = 0;
    {
        std::vector<int> cnt2(N);
        CK(cudaMemcpy(cnt2.data(), d_cnt, (size_t)N * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (int i = 0; i < N; ++i) if (cnt2[i] != cnt[i]) ++bad;
        printf("[S1] counts mismatch rows = %zu / %d\n", bad, N);
        rc |= (bad != 0);
    }

    // ---- S5 + D ----
    gather();
    CK(cudaDeviceSynchronize());
    std::vector<__half> got((size_t)A * C);
    CK(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));
    gather();
    CK(cudaDeviceSynchronize());
    std::vector<__half> got2((size_t)A * C);
    CK(cudaMemcpy(got2.data(), d_out, got2.size() * 2, cudaMemcpyDeviceToHost));
    const bool det = memcmp(got.data(), got2.data(), got.size() * 2) == 0;
    printf("[D ] determinism: %s\n", det ? "IDENTICAL" : "DIFF");
    rc |= !det;
    {
        double mx = 0;
        for (size_t i = 0; i < got.size(); ++i)
            mx = fmax(mx, fabs(__half2float(got[i])));
        printf("[Q ] max|got|=%.4e got[a0][0..3]=%.4e %.4e %.4e %.4e\n",
               mx, __half2float(got[0]), __half2float(got[1]),
               __half2float(got[2]), __half2float(got[3]));
    }

    {
        double max_abs = 0, max_rel = 0;
        size_t viol = 0;
        const int CG = C / G;
        for (int a_i = 0; a_i < RANCH; ++a_i) {
            float ref[C];
            for (int ch = 0; ch < C; ++ch) ref[ch] = 0.f;
            for (int pt = 0; pt < PTS; ++pt) {
                const int row = a_i * PTS + pt;
                const E48* er = &ent_ref[(size_t)row * CS];
                const int k = cnt[row];
                for (int ch = 0; ch < C; ++ch) {
                    const int g = ch / CG;
                    float acc = 0.f;
                    for (int j = 0; j < k; ++j) {
                        const E48& e = er[j];
                        const float wg = half2f(e.wg[g >> 1], g & 1);
                        if (wg == 0.f) continue;
                        float s = 0.f;
                        if (e.wk.x != 0.f) s += e.wk.x * __half2float(feat[e.off.x + ch]);
                        if (e.wk.y != 0.f) s += e.wk.y * __half2float(feat[e.off.y + ch]);
                        if (e.wk.z != 0.f) s += e.wk.z * __half2float(feat[e.off.z + ch]);
                        if (e.wk.w != 0.f) s += e.wk.w * __half2float(feat[e.off.w + ch]);
                        acc += s * wg;
                    }
                    ref[ch] += acc;
                }
            }
            double ss = 0;
            for (int ch = 0; ch < C; ++ch) ss += ref[ch] * ref[ch];
            const double row_tol = 2e-2 * sqrt(ss / C);
            for (int ch = 0; ch < C; ++ch) {
                const double b = ref[ch];
                const double d = fabs(__half2float(got[(size_t)a_i * C + ch]) - b);
                max_abs = d > max_abs ? d : max_abs;
                max_rel = fmax(max_rel, d / fmax(fabs(b), 1e-3));
                viol += (d > fmax(5e-3 * fabs(b), row_tol));
            }
        }
        printf("[S5] out[A,C] vs host fp32 anchor-sum: max_abs=%.4e max_rel=%.3e viol=%zu/%d\n",
               max_abs, max_rel, viol, RANCH * C);
        rc |= (viol != 0);

        // ---- forensics: isolate kernel-vs-replica attribution ----
        {
            const int ROWS_F = RANCH * PTS;
            std::vector<E48> dent((size_t)ROWS_F * CS);
            CK(cudaMemcpy(dent.data(), d_ent, dent.size() * 48, cudaMemcpyDeviceToHost));
            // valid-slot-only divergence (tail slots are zeroed, not evidence)
            int nbad_rows = 0, first_bad = -1, first_bad_slot = -1;
            for (int row = 0; row < ROWS_F; ++row)
                for (int j = 0; j < cnt[row]; ++j)
                    if (memcmp(&dent[(size_t)row * CS + j], &ent_ref[(size_t)row * CS + j], 48) != 0) {
                        ++nbad_rows;
                        if (first_bad < 0) { first_bad = row; first_bad_slot = j; }
                        break;
                    }
            printf("[F ] device-vs-replica VALID slots (rows<%d): bad=%d first=(row %d slot %d)\n",
                   ROWS_F, nbad_rows, first_bad, first_bad_slot);
            if (first_bad >= 0) {
                const int row = first_bad, slot = first_bad_slot;
                const E48& de = dent[(size_t)row * CS + slot];
                const E48& he = ent_ref[(size_t)row * CS + slot];
                printf("[F ] row %d slot %d (cnt=%d) DEV wk=(%e,%e,%e,%e) off=(%d,%d,%d,%d) wg=%08x %08x %08x %08x\n",
                       row, slot, cnt[row], de.wk.x, de.wk.y, de.wk.z, de.wk.w,
                       de.off.x, de.off.y, de.off.z, de.off.w,
                       *(unsigned*)&de.wg[0], *(unsigned*)&de.wg[1],
                       *(unsigned*)&de.wg[2], *(unsigned*)&de.wg[3]);
                printf("[F ]                              HOS wk=(%e,%e,%e,%e) off=(%d,%d,%d,%d) wg=%08x %08x %08x %08x\n",
                       he.wk.x, he.wk.y, he.wk.z, he.wk.w,
                       he.off.x, he.off.y, he.off.z, he.off.w,
                       *(unsigned*)&he.wg[0], *(unsigned*)&he.wg[1],
                       *(unsigned*)&he.wg[2], *(unsigned*)&he.wg[3]);
            }
            // anchor-sum recomputed FROM DEVICE entry bits vs device out
            double kmax = 0;
            int kworst_a = -1, kworst_c = -1;
            for (int a_i = 0; a_i < RANCH; ++a_i)
                for (int ch = 0; ch < C; ++ch) {
                    float s2 = 0.f;
                    for (int pt = 0; pt < PTS; ++pt) {
                        const int row = a_i * PTS + pt;
                        const E48* er = &dent[(size_t)row * CS];
                        const int k = cnt[row];
                        const int gg = ch / CG;
                        float acc = 0.f;
                        for (int j = 0; j < k; ++j) {
                            const E48& e = er[j];
                            const float wg = half2f(e.wg[gg >> 1], gg & 1);
                            if (wg == 0.f) continue;
                            float s = 0.f;
                            if (e.wk.x != 0.f) s += e.wk.x * __half2float(feat[e.off.x + ch]);
                            if (e.wk.y != 0.f) s += e.wk.y * __half2float(feat[e.off.y + ch]);
                            if (e.wk.z != 0.f) s += e.wk.z * __half2float(feat[e.off.z + ch]);
                            if (e.wk.w != 0.f) s += e.wk.w * __half2float(feat[e.off.w + ch]);
                            acc += s * wg;
                        }
                        s2 += acc;
                    }
                    const double d2 = fabs(__half2float(got[(size_t)a_i * C + ch]) - s2);
                    if (d2 > kmax) { kmax = d2; kworst_a = a_i; kworst_c = ch; }
                }
            printf("[F ] kernel check vs DEVICE entries: max|dev-host(devent)|=%.4e (a=%d ch=%d)\n",
                   kmax, kworst_a, kworst_c);
        }
    }

    const float t5 = timed_ms(gather, 10, 50);
    printf("[T5] gather_v5 = %8.3f ms\n", t5);

    CK(cudaFree(d_ent)); CK(cudaFree(d_cnt)); CK(cudaFree(d_feat));
    CK(cudaFree(d_out)); CK(cudaFree(d_logits)); CK(cudaFree(d_loc));
    if (d_part) CK(cudaFree(d_part));
    g_rc |= rc;
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("[dev] %s SMs=%d L2=%dMB\n", prop.name, prop.multiProcessorCount,
           (int)(prop.l2CacheSize >> 20));

    std::vector<float> loc((size_t)N_FULL * CAMS * 2), w((size_t)N_FULL * CAMS * LEVELS * G);
    FILE* f = fopen((std::string(KDIR) + "/loc.bin").c_str(), "rb");
    if (!f || fread(loc.data(), 4, loc.size(), f) != loc.size()) { printf("loc.bin FAIL\n"); return 1; }
    fclose(f);
    f = fopen((std::string(KDIR) + "/w.bin").c_str(), "rb");
    if (!f || fread(w.data(), 4, w.size(), f) != w.size()) { printf("w.bin FAIL\n"); return 1; }
    fclose(f);

    std::vector<__half> feat((size_t)CAMS * HW * C);
    unsigned seed = 777;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) * (1.f / 16777216.f); };
    for (auto& v : feat) v = __float2half(rnd() * 2.f - 1.f);

    run_case("A", 1024, 500, loc, w, feat);   // layers.0/p: SPLIT=1 path
    run_case("B", 128, 500, loc, w, feat);    // SPLIT>1 + finalize path
    run_case("C", 400, 80, loc, w, feat);     // layers.1/t geometry

    printf(g_rc == 0 ? "BENCH_V5_DONE\n" : "BENCH_V5_FAIL rc=%d\n", g_rc);
    return g_rc;
}
