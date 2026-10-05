// SparseDriveV2 DeformableAggregation TensorRT plugin (IPluginV2DynamicExt)
// ONNX node: op_type="DeformableAggregation", domain="sparsedrivev2"
//   inputs : [0] feat   (bs, cams, HW_total, C)   fp32/fp16
//            [1] ss     (levels, 2)              int32   (H,W per FPN level)
//            [2] ssi    (levels,)                int32   (start offset per level)
//            [3] loc    (bs, N, cams, 2)         fp32/fp16  (normalized coords)
//            [4] w      (bs, N, cams, levels, groups) fp32/fp16
//   output : (bs, N, C)                          fp32/fp16
// Math is identical to navsim/agents/sparsedrive/ops/src/deformable_aggregation_cuda.cu.
// fp16 IO (feat half + out half) runs the native kernel (plan A): half taps are
// up-cast exactly, accumulation and final rounding are fp32/fp16 as before, so
// outputs are bitwise-equal to the old cast->fp32->cast path. Other combos
// (int8 feat, fp32 out) still go through the internal up-cast/staging path.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t s_ = (call);                                        \
        if (s_ != cudaSuccess) {                                        \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(s_),  \
                   __FILE__, __LINE__);                                 \
            return 1;                                                   \
        }                                                                \
    } while (0)

namespace {

// ---------------- reference kernels (forward only) ----------------
__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr) {
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1 - lh, hw = 1 - lw;

    const int w_stride = num_embeds;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

    float v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    if (h_low >= 0 && w_low >= 0) v1 = bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr];
    if (h_low >= 0 && w_high <= width - 1) v2 = bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr];
    if (h_high <= height - 1 && w_low >= 0) v3 = bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr];
    if (h_high <= height - 1 && w_high <= width - 1) v4 = bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr];

    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

// half-storage variant (native fp16 IO path, plan A): feat stays __half in
// memory; each tap is up-cast to fp32 exactly like cast_half_to_float did, so
// per-tap values and the fp32 accumulation order are identical to the old
// cast->fp32->cast path (bitwise-equal outputs when compiled alike).
__device__ float bilinear_sampling_h(
    const __half *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr) {
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1 - lh, hw = 1 - lw;

    const int w_stride = num_embeds;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

    float v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    if (h_low >= 0 && w_low >= 0) v1 = __half2float(bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr]);
    if (h_low >= 0 && w_high <= width - 1) v2 = __half2float(bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr]);
    if (h_high <= height - 1 && w_low >= 0) v3 = __half2float(bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr]);
    if (h_high <= height - 1 && w_high <= width - 1) v4 = __half2float(bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr]);

    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}
// feat_scale[channel_index]; loc/w stay float. Compute is fp32.
__device__ float bilinear_sampling_i8(
    const int8_t *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr, const float *feat_scale, const int &channel_index) {
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1 - lh, hw = 1 - lw;

    const int w_stride = num_embeds;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

    const float s = feat_scale[channel_index];
    float v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    if (h_low >= 0 && w_low >= 0) v1 = (float)bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr] * s;
    if (h_low >= 0 && w_high <= width - 1) v2 = (float)bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr] * s;
    if (h_high <= height - 1 && w_low >= 0) v3 = (float)bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr] * s;
    if (h_high <= height - 1 && w_high <= width - 1) v4 = (float)bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr] * s;

    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

__global__ void deformable_aggregation_kernel(
    const int num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int8_t* mc_ms_feat_i8,
    const float* feat_scale,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const __half* sample_location_h,
    const float* weights,
    const __half* weights_h,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    float *output_ptr = output + idx;
    const int channel_index = idx % num_embeds;
    const int groups_index = channel_index / (num_embeds / num_groups);
    idx /= num_embeds;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int batch_index = idx;

    const int value_cam_stride = num_feat * num_embeds;
    const int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams << 1;
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset =
        (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride + groups_index;

    float result = 0;
    for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 2) {
        float loc_w, loc_h;
        if (sample_location_h != nullptr) {
            loc_w = __half2float(sample_location_h[loc_offset]);
            loc_h = __half2float(sample_location_h[loc_offset + 1]);
        } else {
            loc_w = sample_location[loc_offset];
            loc_h = sample_location[loc_offset + 1];
        }

        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
                const int scale_offset = scale_start_index[scale_index] * num_embeds;
                const int spatial_shape_ptr = scale_index << 1;
                const int h = spatial_shape[spatial_shape_ptr];
                const int w = spatial_shape[spatial_shape_ptr + 1];

                const float h_im = loc_h * h - 0.5;
                const float w_im = loc_w * w - 0.5;

                const int value_ptr = value_offset + scale_offset + value_cam_stride * cam_index;
                const int weight_ptr =
                    weight_offset + scale_index * num_groups + weight_cam_stride * cam_index;
                const float wgt = weights_h != nullptr
                                      ? __half2float(weights_h[weight_ptr])
                                      : weights[weight_ptr];
                if (mc_ms_feat_i8 != nullptr) {
                    result += bilinear_sampling_i8(mc_ms_feat_i8, h, w, num_embeds, h_im, w_im,
                                                   value_ptr, feat_scale, channel_index) * wgt;
                } else {
                    result += bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_ptr) * wgt;
                }
            }
        }
    }
    *output_ptr = result;
}

// Native fp16-IO kernel (plan A): feat read as __half (per-tap exact up-cast,
// same fp32 accumulation order as the fp32 kernel), output written directly as
// __half — eliminates the workspace cast kernels entirely.
__global__ void deformable_aggregation_kernel_h(
    const int num_kernels,
    __half* output,
    const __half* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const __half* sample_location_h,
    const float* weights,
    const __half* weights_h,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    __half *output_ptr = output + idx;
    const int channel_index = idx % num_embeds;
    const int groups_index = channel_index / (num_embeds / num_groups);
    idx /= num_embeds;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int batch_index = idx;

    const int value_cam_stride = num_feat * num_embeds;
    const int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams << 1;
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset =
        (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride + groups_index;

    float result = 0;
    for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 2) {
        float loc_w, loc_h;
        if (sample_location_h != nullptr) {
            loc_w = __half2float(sample_location_h[loc_offset]);
            loc_h = __half2float(sample_location_h[loc_offset + 1]);
        } else {
            loc_w = sample_location[loc_offset];
            loc_h = sample_location[loc_offset + 1];
        }

        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
                const int scale_offset = scale_start_index[scale_index] * num_embeds;
                const int spatial_shape_ptr = scale_index << 1;
                const int h = spatial_shape[spatial_shape_ptr];
                const int w = spatial_shape[spatial_shape_ptr + 1];

                const float h_im = loc_h * h - 0.5;
                const float w_im = loc_w * w - 0.5;

                const int value_ptr = value_offset + scale_offset + value_cam_stride * cam_index;
                const int weight_ptr =
                    weight_offset + scale_index * num_groups + weight_cam_stride * cam_index;
                const float wgt = weights_h != nullptr
                                      ? __half2float(weights_h[weight_ptr])
                                      : weights[weight_ptr];
                result += bilinear_sampling_h(mc_ms_feat, h, w, num_embeds, h_im, w_im,
                                              value_ptr) * wgt;
            }
        }
    }
    *output_ptr = __float2half(result);
}

// Generalized launcher usable from the plugin and from standalone unit tests.
// Exactly one of feat / feat_i8 is non-null; feat_scale required when feat_i8 != null.
// Exactly one of loc_f / loc_h and one of w_f / w_h is non-null (fp32 or fp16 direct read).
// Must live OUTSIDE the anonymous namespace below: extern "C" inside an unnamed
// namespace keeps internal linkage and the unit-test TU cannot link the symbol.
}  // anonymous namespace (closed so dfa_forward_cuda gets external linkage)
extern "C" void dfa_forward_cuda(
    float* output, const float* feat, const int8_t* feat_i8, const float* feat_scale,
    const int* spatial_shape, const int* scale_start_index,
    const float* loc_f, const __half* loc_h,
    const float* w_f, const __half* w_h,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, cudaStream_t stream) {
    const int num_kernels = batch_size * num_pts * num_embeds;
    deformable_aggregation_kernel
        <<< (int)ceil(((double)num_kernels / 512)), 512, 0, stream >>>
        (num_kernels, output, feat, feat_i8, feat_scale, spatial_shape, scale_start_index,
         loc_f, loc_h, w_f, w_h, batch_size, num_cams, num_feat, num_embeds,
         num_scale, num_pts, num_groups);
}

// Native fp16-IO launcher (feat __half, output __half; loc/w half or float).
extern "C" void dfa_forward_half_cuda(
    __half* output, const __half* feat,
    const int* spatial_shape, const int* scale_start_index,
    const float* loc_f, const __half* loc_h,
    const float* w_f, const __half* w_h,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_pts, int num_groups, cudaStream_t stream) {
    const int num_kernels = batch_size * num_pts * num_embeds;
    deformable_aggregation_kernel_h
        <<< (int)ceil(((double)num_kernels / 512)), 512, 0, stream >>>
        (num_kernels, output, feat, spatial_shape, scale_start_index,
         loc_f, loc_h, w_f, w_h, batch_size, num_cams, num_feat, num_embeds,
         num_scale, num_pts, num_groups);
}

namespace {  // anonymous namespace reopened (kernels/helpers stay internal-linkage)

// ---------------- cast helpers ----------------
__global__ void cast_half_to_float_kernel(const __half* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
__global__ void cast_float_to_half_kernel(const float* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

inline int elems(const nvinfer1::Dims& d) {
    int p = 1;
    for (int i = 0; i < d.nbDims; ++i) p *= d.d[i];
    return p;
}

inline bool linear_ok(const nvinfer1::PluginTensorDesc& d) {
    return d.format == nvinfer1::TensorFormat::kLINEAR;
}

// ---------------- plugin ----------------
class DeformableAggregationPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    DeformableAggregationPlugin() = default;
    DeformableAggregationPlugin(const void* data, size_t length) { (void)data; (void)length; }
    ~DeformableAggregationPlugin() override = default;

    const char* getPluginType() const noexcept override { return "DeformableAggregation"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)outputIndex; (void)nbInputs;
        nvinfer1::DimsExprs out{};
        out.nbDims = 3;
        out.d[0] = inputs[0].d[0];                  // bs
        out.d[1] = inputs[3].d[1];                  // N
        out.d[2] = inputs[0].d[3];                  // C
        return out;
    }

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) noexcept override {
        // layout: in {feat=0, ss=1, ssi=2, loc=3, w=4[, feat_scale=5]}, out {nbInputs}
        // nbInputs==6: int8-feat contract with per-C scale input; nbInputs==5: legacy fp32/fp16
        const int out_pos = nbInputs;  // output index
        if (pos == 1 || pos == 2) {  // int32 shape tensors
            return inOut[pos].type == nvinfer1::DataType::kINT32 &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        }
        if (nbInputs == 6 && pos == 5) {  // per-channel dequant scales, fp32
            return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        }
        bool isFloat = (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
                        inOut[pos].type == nvinfer1::DataType::kHALF) &&
                       inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        bool isInt8 = inOut[pos].type == nvinfer1::DataType::kINT8 &&
                      inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        if (pos == 0) {
            if (nbInputs == 6) return isFloat || isInt8;  // int8-feat contract
            // legacy path: force HALF feat so TRT picks the native fp16 kernel
            // (fp32 combos made the built engine stage fp32 and forfeit the
            // half-bandwidth gain; board evidence 2026-10-05)
            return inOut[pos].type == nvinfer1::DataType::kHALF &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        }
        if (pos == 3 || pos == 4) return isFloat;  // loc/w: fp32 or fp16 direct-read, independent
        // pos == out_pos: follows feat dtype (int8 feat computes fp32)
        if (inOut[0].type == nvinfer1::DataType::kINT8)
            return inOut[out_pos].type == nvinfer1::DataType::kFLOAT && linear_ok(inOut[out_pos]);
        return isFloat && inOut[out_pos].type == inOut[0].type;
    }

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {
        (void)nbInputs; (void)nbOutputs;
        // static dims at build time
        m_bs = in[0].max.d[0];
        m_cams = in[0].max.d[1];
        m_hw = in[0].max.d[2];
        m_c = in[0].max.d[3];
        m_levels = in[1].max.d[0];
        m_n = in[3].max.d[1];
        m_groups = in[4].max.d[4];
    }

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* in, int nbInputs,
        const nvinfer1::PluginTensorDesc* out, int nbOutputs) const noexcept override {
        (void)nbInputs; (void)nbOutputs;
        // fp32 staging for any fp16 tensor we must up-cast (feat/loc/w/out)
        size_t bytes = 0;
        auto stage = [&](const nvinfer1::PluginTensorDesc& d) {
            size_t e = 1;
            for (int i = 0; i < d.dims.nbDims; ++i) e *= (size_t)d.dims.d[i];
            bytes += e * sizeof(float) + 256;
        };
        // fp32 staging for fp16 feat and fp16 output (feat int8 dequantizes in-kernel;
        // loc/w are read directly in fp16, no staging)
        if (in[0].type == nvinfer1::DataType::kHALF) stage(in[0]);
        if (out[0].type == nvinfer1::DataType::kHALF) stage(out[0]);
        return bytes;
    }

    int enqueue(const nvinfer1::PluginTensorDesc* in, const nvinfer1::PluginTensorDesc* out,
                const void* const* inputs, void* const* outputs, void* workspace,
                cudaStream_t stream) noexcept override {
        uint8_t* ws = (uint8_t*)workspace;
        auto take = [&](size_t n) { void* p = ws; ws += (n + 255) / 256 * 256; return p; };
        auto prod = [](const nvinfer1::Dims& d) {
            size_t e = 1;
            for (int i = 0; i < d.nbDims; ++i) e *= (size_t)d.d[i];
            return e;
        };

        const float* feat = nullptr;
        const int8_t* feat_i8 = nullptr;
        const float* feat_scale = nullptr;
        {
            static bool combo_logged = false;
            if (!combo_logged) {
                combo_logged = true;
                fprintf(stderr, "[dfa] combo feat_type=%d out_type=%d (0=f32 1=half 2=i8)\n",
                        (int)in[0].type, (int)out[0].type);
            }
        }
        if (in[0].type == nvinfer1::DataType::kINT8) {
            feat_i8 = (const int8_t*)inputs[0];
            feat_scale = (const float*)inputs[5];
        } else if (in[0].type == nvinfer1::DataType::kHALF &&
                   out[0].type == nvinfer1::DataType::kHALF) {
            // Native fp16 IO path (plan A): no workspace casts. Per-tap math is
            // identical to the cast->fp32->cast path (exact half up-cast, same
            // fp32 accumulation order, same final rounding).
            dfa_forward_half_cuda(
                (__half*)outputs[0], (const __half*)inputs[0],
                (const int*)inputs[1], (const int*)inputs[2],
                in[3].type == nvinfer1::DataType::kHALF ? nullptr : (const float*)inputs[3],
                in[3].type == nvinfer1::DataType::kHALF ? (const __half*)inputs[3] : nullptr,
                in[4].type == nvinfer1::DataType::kHALF ? nullptr : (const float*)inputs[4],
                in[4].type == nvinfer1::DataType::kHALF ? (const __half*)inputs[4] : nullptr,
                in[0].dims.d[0], in[0].dims.d[1], in[0].dims.d[2],
                in[0].dims.d[3], in[1].dims.d[0], in[3].dims.d[1],
                in[4].dims.d[4], stream);
            return cudaGetLastError() != cudaSuccess ? 1 : 0;
        } else if (in[0].type == nvinfer1::DataType::kHALF) {
            size_t e = prod(in[0].dims);
            float* feat_f = (float*)take(e * sizeof(float));
            cast_half_to_float_kernel<<<(int)((e + 511) / 512), 512, 0, stream>>>(
                (const __half*)inputs[0], feat_f, (int)e);
            feat = feat_f;
        } else {
            feat = (const float*)inputs[0];
        }

        // loc / w: fp32 direct, fp16 direct-read, or staged when fp16 feat forced fp16 elsewhere
        const float* tensors_f[2] = {nullptr, nullptr};
        const __half* tensors_h[2] = {nullptr, nullptr};
        const nvinfer1::PluginTensorDesc* descs[2] = {&in[3], &in[4]};
        const void* raws[2] = {inputs[3], inputs[4]};
        for (int t = 0; t < 2; ++t) {
            if (descs[t]->type == nvinfer1::DataType::kHALF) {
                tensors_h[t] = (const __half*)raws[t];
            } else {
                tensors_f[t] = (const float*)raws[t];
            }
        }

        const bool half_out = (out[0].type == nvinfer1::DataType::kHALF);
        size_t o_e = prod(out[0].dims);
        float* outp = half_out ? (float*)take(o_e * sizeof(float)) : (float*)outputs[0];

        dfa_forward_cuda(outp, feat, feat_i8, feat_scale,
                         (const int*)inputs[1], (const int*)inputs[2],
                         tensors_f[0], tensors_h[0], tensors_f[1], tensors_h[1],
                         in[0].dims.d[0], in[0].dims.d[1], in[0].dims.d[2],
                         in[0].dims.d[3], in[1].dims.d[0], in[3].dims.d[1],
                         in[4].dims.d[4], stream);

        if (half_out) {
            cast_float_to_half_kernel<<<(int)((o_e + 511) / 512), 512, 0, stream>>>(
                outp, (__half*)outputs[0], (int)o_e);
        }
        return cudaGetLastError() != cudaSuccess ? 1 : 0;
    }

    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override {
        (void)index; (void)nbInputs;
        // int8-feat path computes in fp32 and outputs fp32 (fp16 handled via combos of loc/w)
        if (inputTypes[0] == nvinfer1::DataType::kHALF) return nvinfer1::DataType::kHALF;
        return nvinfer1::DataType::kFLOAT;
    }

    const char* getPluginNamespace() const noexcept override { return m_ns.c_str(); }
    void setPluginNamespace(const char* ns) noexcept override { m_ns = ns ? ns : ""; }

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }

    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override { (void)buffer; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* p = new DeformableAggregationPlugin(*this);
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }

private:
    std::string m_ns;
    int m_bs = 1, m_cams = 3, m_hw = 0, m_c = 0, m_levels = 4, m_n = 0, m_groups = 8;
};

class DeformableAggregationPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const noexcept override { return "DeformableAggregation"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &m_fields;
    }
    nvinfer1::IPluginV2* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        (void)name; (void)fc;
        auto* p = new DeformableAggregationPlugin();
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override {
        (void)name; (void)serialData; (void)serialLength;
        auto* p = new DeformableAggregationPlugin(serialData, serialLength);
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }
    void setPluginNamespace(const char* ns) noexcept override { m_ns = ns ? ns : ""; }
    const char* getPluginNamespace() const noexcept override { return m_ns.c_str(); }

private:
    // 必须与 ONNX node domain 一致（parser 按 name/version/namespace 查 creator）；
    // registerCreator() 调 setPluginNamespace() 覆盖此值；生效 ns 由
    // 尾部 DfaMultiDomainRegistrar 的注册顺序决定（"" 必须先注册）。
    std::string m_ns{""};
    nvinfer1::PluginFieldCollection m_fields{0, nullptr};
};

}  // namespace

// Dual-domain registration. Board-proven facts (TRT 8.6.1.2):
//   1. the ONNX parser queries getPluginCreator(name, ver, "") — the EMPTY
//      namespace, regardless of the node's ONNX domain (the domain only
//      supplies the opset version);
//   2. PluginRegistry::registerCreator dedups by (name, version) and keeps
//      the FIRST registration — so "" must be registered first; the second
//      call is logged as duplicate and dropped.
namespace {
struct DfaMultiDomainRegistrar {
    DfaMultiDomainRegistrar() {
        auto* registry = getPluginRegistry();
        registry->registerCreator(*new DeformableAggregationPluginCreator(), "");
        registry->registerCreator(*new DeformableAggregationPluginCreator(), "sparsedrivev2");
    }
};
static DfaMultiDomainRegistrar g_dfa_multi_domain_registrar;
}

// ============================================================================
// Pg variant: plan+gather with inline softmax  (op "DeformableAggregationPg")
// ----------------------------------------------------------------------------
// Board evidence 2026-10-05: the kernel above walks the fixed (cams x levels)
// schedule per output element and relies on guards to skip out-of-picture
// samples; with real inputs most warps do no useful work (effective bandwidth
// ~1/3 of peak) and the upstream Softmax+Cast+Reshape+Transpose chain sweeps
// a 196MB fp32 tensor several times inside the Myelin region. This variant:
//
//   K1 plan (one 256-thread block per (batch, anchor)):
//     A. inline softmax over the CS*pts axis per (anchor, group): fp32 math,
//        deterministic fixed reduction tree (per-lane online max/sum, warp
//        butterfly, block merge in warp order). Replaces the graph Softmax.
//     B. entry validity = any-group weight > DFA_PG_EPS (exact 0 weights from
//        the -inf mask vanish at EPS=0).
//     B2. AND with the bilinear camera guard (0<loc_w<1, 0<loc_h<1), build
//        per-point 12-bit masks + counts.
//     C. emit 64B entries {float4 corner weights, int4 corner element offsets,
//        float4x2 group weights} into fixed per-point slots, in (cam-major,
//        level-minor) order -- exactly the v1 kernel's accumulation order.
//   K2 gather (one warp per (row, group), lane = channel-in-group):
//     walks only the row's valid entries; per-entry math is expression-
//     identical to deformable_aggregation_kernel_h (same wk formulas, same
//     fp32 FMA sequence, same half->float tap reads), so the result equals
//     the v1 path except for ulp-level softmax differences.
//
// Interface change vs "DeformableAggregation": input[4] is the PRE-softmax
// logits [bs, A, cams*levels*pts, G] (cs-major, pt-minor; fp32 or fp16)
// instead of post-softmax w [bs, N, cams, levels, G]. All upstream Softmax /
// Cast / Reshape / Transpose nodes feeding w are removed by graph surgery.
//
// Hard geometry (build fails otherwise): G <= 8, C % G == 0, C/G <= 32,
// A * pts == N, CS = cams*levels <= 16, pts <= 4096.
// Workspace: rows * CS * 64B entries (fixed slots, only valid ones written)
//            + rows * 4B counts.
// ============================================================================

namespace dfa_pg {

// Pruning threshold on the summed (pre-division) group weight of an entry.
// 0 = drop only exact zeros (the -inf mask); the v1 accumulation order is
// then preserved bit-for-bit for every surviving entry.
#ifndef DFA_PG_EPS
#define DFA_PG_EPS 0.0f
#endif

struct Entry {          // 48B, 16B-aligned (v4; was 64B with fp32 wg_lo/wg_hi)
    float4 wk;          // bilinear corner weights; 0 for guarded-out corners
    int4  off;          // corner element offsets (channel 0); gather adds lane
    __half2 wg_h[4];    // softmax group weights 0..7 as half (fp32 softmax math
                        // happens in plan; half rounding is v4's accepted noise)
};

__device__ __forceinline__ float pg_sel8h(const __half2* w, int g) {
    const __half2 h = w[g >> 1];
    return __half2float((g & 1) ? h.y : h.x);
}

// ---- K1: plan -------------------------------------------------------------
// phases A (softmax stats), B (validity bits), B2 (masks+counts), C (emit)
template <typename LT, bool G8>
__global__ void dfa_pg_plan_kernel(
    Entry* __restrict__ entries,
    int* __restrict__ counts,
    const LT* __restrict__ logits,
    const float* __restrict__ loc_f,
    const __half* __restrict__ loc_h,
    const int* __restrict__ spatial_shape,
    const int* __restrict__ scale_start_index,
    int A, int pts, int N, int cams, int levels, int G,
    int hw, int C, float eps) {
    const int ba = blockIdx.x;
    const int b = ba / A;
    const int a = ba % A;
    const int CS = cams * levels;
    const int ISP = CS * pts;
    const int nwarps = blockDim.x >> 5;          // 8
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int S = (ISP + nwarps - 1) / nwarps;   // i-slice per warp

    extern __shared__ float pgsm[];
    char* smc = reinterpret_cast<char*>(pgsm);
    float* s_m = reinterpret_cast<float*>(smc); smc += G * 4;              // [G]
    float* s_s = reinterpret_cast<float*>(smc); smc += G * 4;              // [G]
    float* s_inv = reinterpret_cast<float*>(smc); smc += G * 4;            // [G] v4: 1/s_s
    float* s_part = reinterpret_cast<float*>(smc); smc += nwarps * G * 2 * 4;
    float* s_ss = reinterpret_cast<float*>(smc); smc += levels * 3 * 4;    // h,w,ssi
    unsigned short* s_mask = reinterpret_cast<unsigned short*>(smc);       // [pts]
    smc += pts * 2;
    smc = reinterpret_cast<char*>((reinterpret_cast<size_t>(smc) + 3) & ~size_t(3));
    unsigned* s_okw = reinterpret_cast<unsigned*>(smc);                    // [nwarps*wpw]
    const int wpw = (S + 31) >> 5;               // u32 words per warp slice

    for (int i = threadIdx.x; i < levels * 3; i += blockDim.x) {
        const int lv = i / 3;
        s_ss[i] = (i % 3 == 0) ? (float)spatial_shape[lv * 2]
                : (i % 3 == 1) ? (float)spatial_shape[lv * 2 + 1]
                               : (float)scale_start_index[lv];
    }

    const LT* lrow = logits + (size_t)(b * A + a) * ISP * G;
    const int beg = warp * S;
    const int end = min(beg + S, ISP);

    // ---- phase A: online max/sum per group over this warp's i-slice ----
    // Validity bits ride along for free: with the -inf mask an entry is useful
    // iff any group logit is finite (exp(-inf-m)=0 exactly; eps=0). Entries
    // whose every group underflows at the FINAL max may be marked valid here
    // (running max is smaller) -- their recomputed wg is then exactly 0 and
    // gather skips them, so outputs are unchanged. Replaces the old phase B
    // second full pass over the logits.
    float m[8], s[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) { m[g] = -3.0e38f; s[g] = 0.f; }
    for (int it = 0; it < S; it += 32) {
        const int i = beg + it + lane;
        bool ok = false;
        if (i < end) {
            float v[8];
            if (G8) {
                const char* p = reinterpret_cast<const char*>(lrow) +
                                (size_t)i * G * sizeof(LT);
                if (sizeof(LT) == 4) {
                    const float4 lo = __ldg(reinterpret_cast<const float4*>(p));
                    const float4 hi = __ldg(reinterpret_cast<const float4*>(p) + 1);
                    v[0] = lo.x; v[1] = lo.y; v[2] = lo.z; v[3] = lo.w;
                    v[4] = hi.x; v[5] = hi.y; v[6] = hi.z; v[7] = hi.w;
                } else {
                    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
                    const __half* h = reinterpret_cast<const __half*>(&u);
#pragma unroll
                    for (int g = 0; g < 8; ++g) v[g] = __half2float(h[g]);
                }
            } else {
                for (int g = 0; g < G; ++g)
                    v[g] = (sizeof(LT) == 4) ? (float)lrow[i * G + g]
                           : __half2float(reinterpret_cast<const __half*>(lrow)[i * G + g]);
            }
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                if (g >= G) break;
                if (v[g] != -INFINITY) ok = true;
                const float mn = fmaxf(m[g], v[g]);
                s[g] = s[g] * __expf(m[g] - mn) + __expf(v[g] - mn);
                m[g] = mn;
            }
        }
        const unsigned bits = __ballot_sync(0xffffffffu, ok);
        if (lane == 0 && it / 32 < wpw) s_okw[warp * wpw + it / 32] = bits;
    }
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g >= G) break;
        for (int off = 16; off > 0; off >>= 1) {
            const float mo = __shfl_xor_sync(0xffffffffu, m[g], off);
            const float so = __shfl_xor_sync(0xffffffffu, s[g], off);
            const float mn = fmaxf(m[g], mo);
            s[g] = s[g] * __expf(m[g] - mn) + so * __expf(mo - mn);
            m[g] = mn;
        }
        if (lane == 0) {
            s_part[(warp * G + g) * 2] = m[g];
            s_part[(warp * G + g) * 2 + 1] = s[g];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            if (g >= G) break;
            float mm = -3.0e38f, ss = 0.f;
            for (int w = 0; w < nwarps; ++w) {   // fixed order: deterministic
                const float mo = s_part[(w * G + g) * 2];
                const float so = s_part[(w * G + g) * 2 + 1];
                const float mn = fmaxf(mm, mo);
                ss = ss * __expf(mm - mn) + so * __expf(mo - mn);
                mm = mn;
            }
            if (lane == 0) {
                s_m[g] = mm; s_s[g] = ss;
                // v4: one IEEE divide per (block, group) here replaces ~1.55M
                // divides in phase C (8 per emitted entry)
                s_inv[g] = (ss > 0.f) ? 1.f / ss : 0.f;
            }
        }
    }
    __syncthreads();

    // ---- phase B2: warp 0 builds per-point masks (AND camera guard) + counts
    if (warp == 0) {
        for (int pt = lane; pt < pts; pt += 32) {
            const int n = a * pts + pt;
            const int row = b * N + n;
            unsigned mask = 0;
            for (int cs = 0; cs < CS; ++cs) {
                const int i = cs * pts + pt;
                const int w2 = i / S;
                const int li = i - w2 * S;
                const bool bit = (s_okw[w2 * wpw + li / 32] >> (li & 31)) & 1u;
                if (!bit) continue;
                const int cam = cs / levels;
                float wv, hv;
                if (loc_h != nullptr) {
                    wv = __half2float(loc_h[(row * cams + cam) * 2]);
                    hv = __half2float(loc_h[(row * cams + cam) * 2 + 1]);
                } else {
                    wv = loc_f[(row * cams + cam) * 2];
                    hv = loc_f[(row * cams + cam) * 2 + 1];
                }
                if (wv > 0.f && wv < 1.f && hv > 0.f && hv < 1.f)
                    mask |= 1u << cs;
            }
            s_mask[pt] = (unsigned short)mask;
            counts[row] = __popc(mask);
        }
    }
    __syncthreads();

    // ---- phase C: emit entries (cam-major, level-minor; slot = rank) ----
    const int chunks = (pts + 31) >> 5;
    for (int task = warp; task < CS * chunks; task += nwarps) {
        const int cs = task / chunks;
        const int chunk = task % chunks;
        const int pt = chunk * 32 + lane;
        if (pt >= pts) continue;
        const unsigned mask = s_mask[pt];
        if (!((mask >> cs) & 1u)) continue;
        const int slot = __popc(mask & ((1u << cs) - 1u));
        const int cam = cs / levels;
        const int level = cs % levels;
        const int n = a * pts + pt;
        const int row = b * N + n;

        float wv, hv;
        if (loc_h != nullptr) {
            wv = __half2float(loc_h[(row * cams + cam) * 2]);
            hv = __half2float(loc_h[(row * cams + cam) * 2 + 1]);
        } else {
            wv = loc_f[(row * cams + cam) * 2];
            hv = loc_f[(row * cams + cam) * 2 + 1];
        }
        // expression-identical to deformable_aggregation_kernel_h: h_im keeps
        // the double-promoted `- 0.5` of the reference implementation
        const int h = (int)s_ss[level * 3];
        const int w = (int)s_ss[level * 3 + 1];
        const int ssi = (int)s_ss[level * 3 + 2];
        const float h_im = hv * h - 0.5;
        const float w_im = wv * w - 0.5;
        const int h_low = floorf(h_im);
        const int w_low = floorf(w_im);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;
        const float lh = h_im - h_low;
        const float lw = w_im - w_low;
        const float hh = 1 - lh, hw2 = 1 - lw;
        const int w_stride = C;
        const int h_stride = w * w_stride;
        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w_low * w_stride;
        const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
        const int base = b * cams * hw * C + cam * hw * C + ssi * C;

        float4 wk;
        int4 off;
        wk.x = (h_low >= 0 && w_low >= 0) ? hh * hw2 : 0.f;
        off.x = (wk.x != 0.f) ? base + h_low_ptr_offset + w_low_ptr_offset : 0;
        wk.y = (h_low >= 0 && w_high <= w - 1) ? hh * lw : 0.f;
        off.y = (wk.y != 0.f) ? base + h_low_ptr_offset + w_high_ptr_offset : 0;
        wk.z = (h_high <= h - 1 && w_low >= 0) ? lh * hw2 : 0.f;
        off.z = (wk.z != 0.f) ? base + h_high_ptr_offset + w_low_ptr_offset : 0;
        wk.w = (h_high <= h - 1 && w_high <= w - 1) ? lh * lw : 0.f;
        off.w = (wk.w != 0.f) ? base + h_high_ptr_offset + w_high_ptr_offset : 0;

        // group weights: recompute from logits + block softmax stats.
        // v4: division by s_s replaced by multiply with precomputed s_inv
        // (~1 ulp fp32 difference), then rounded to half in the 48B entry.
        float wv8[8];
        {
            const int i = cs * pts + pt;
            const char* p = reinterpret_cast<const char*>(lrow) +
                            (size_t)i * G * sizeof(LT);
            if (G8) {
                if (sizeof(LT) == 4) {
                    const float4 lo = __ldg(reinterpret_cast<const float4*>(p));
                    const float4 hi = __ldg(reinterpret_cast<const float4*>(p) + 1);
                    wv8[0] = lo.x; wv8[1] = lo.y; wv8[2] = lo.z; wv8[3] = lo.w;
                    wv8[4] = hi.x; wv8[5] = hi.y; wv8[6] = hi.z; wv8[7] = hi.w;
                } else {
                    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
                    const __half* hh4 = reinterpret_cast<const __half*>(&u);
#pragma unroll
                    for (int g = 0; g < 8; ++g) wv8[g] = __half2float(hh4[g]);
                }
            } else {
                for (int g = 0; g < G; ++g)
                    wv8[g] = (sizeof(LT) == 4) ? (float)lrow[i * G + g]
                             : __half2float(reinterpret_cast<const __half*>(lrow)[i * G + g]);
            }
#pragma unroll
            for (int g = 0; g < 8; ++g)
                wv8[g] = (g < G) ? __expf(wv8[g] - s_m[g]) * s_inv[g] : 0.f;
        }

        Entry e;
        e.wk = wk; e.off = off;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            e.wg_h[k] = __halves2half2(wv8[2 * k], wv8[2 * k + 1]);
        entries[(size_t)row * CS + slot] = e;
    }
}

// ---- K2: gather (v5, sumfusion) --------------------------------------------
// v4 left rows = (anchor, point) independent and let the graph sum the pts
// rows per anchor (Reshape [1,A,pts,C] + ReduceSum axis 2) -- a 262MB fp16
// round trip for layers.0/p (plugin writes it, the reduce kernel re-reads it
// at ~168GB/s = streaming ceiling, and myelin cannot fuse across the plugin
// boundary; measured 1.57+0.20+0.11 = 1.88ms across the three DFA calls).
// v5 fuses the anchor sum into the gather: warp per (anchor, chunk of pts
// rows), lane owns 8 channels (uint4 corner taps, warp covers the whole
// 512B corner row in one transaction), fp32 register accumulation, ONE fp16
// write per anchor. Chunk count scales with A to keep >=768 warps in flight
// (sm_87: 16 SM x 48 warps); SPLIT>1 writes fp32 partials that a tiny
// deterministic finalize kernel reduces (no atomics -> run-to-run identical).
// Plan (entries/counts, Entry48) is untouched -- identical stream as v4.
__global__ void dfa_pg_gather_kernel_v5(
    __half* __restrict__ out, const __half* __restrict__ feat,
    const Entry* __restrict__ entries, const int* __restrict__ counts,
    float* __restrict__ partial,
    int A, int pts, int SPLIT, int CH, int C, int G, int CS) {
    __shared__ Entry s_e[8][16];                 // warp-private, kr<=CS<=16
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gwid = blockIdx.x * 8 + warp;      // global warp = (b, a, sp)
    if (gwid >= A * SPLIT) return;               // whole warp exits; syncwarp only
    const int sp = gwid % SPLIT;
    const int a = (gwid / SPLIT) % A;
    const int b = gwid / (SPLIT * A);
    const int p0 = sp * CH;
    const int p1 = min(p0 + CH, pts);
    const int c0 = lane * 8;                     // C==256, G==8 (check_geom)
    const int g = lane >> 2;                     // c0 / (C/G)
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    const size_t abase = ((size_t)b * A + a) * (size_t)pts;
    for (int p = p0; p < p1; ++p) {
        const int row = (int)(abase + p);
        const int kr = counts[row];
        const float4* src4 = reinterpret_cast<const float4*>(
            entries + (size_t)row * CS);
        float4* se4 = reinterpret_cast<float4*>(s_e[warp]);
        for (int j = lane; j < kr * 3; j += 32) se4[j] = src4[j];
        __syncwarp();
        const float4* er4 = reinterpret_cast<const float4*>(s_e[warp]);
        for (int j = 0; j < kr; ++j) {
            const float4* e4 = er4 + j * 3;      // 3x LDS.128 per entry (v4)
            const float4 ewk = e4[0];
            const int4 eoff = *reinterpret_cast<const int4*>(&e4[1]);
            const float wg = pg_sel8h(reinterpret_cast<const __half2*>(&e4[2]), g);
            if (wg == 0.f) continue;             // never fires when mask is per-(row,cs)
            float s[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) s[i] = 0.f;
            if (ewk.x != 0.f) {
                const uint4 u = *reinterpret_cast<const uint4*>(feat + eoff.x + c0);
                const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(h2[i]);
                    s[2 * i] += ewk.x * f.x; s[2 * i + 1] += ewk.x * f.y;
                }
            }
            if (ewk.y != 0.f) {
                const uint4 u = *reinterpret_cast<const uint4*>(feat + eoff.y + c0);
                const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(h2[i]);
                    s[2 * i] += ewk.y * f.x; s[2 * i + 1] += ewk.y * f.y;
                }
            }
            if (ewk.z != 0.f) {
                const uint4 u = *reinterpret_cast<const uint4*>(feat + eoff.z + c0);
                const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(h2[i]);
                    s[2 * i] += ewk.z * f.x; s[2 * i + 1] += ewk.z * f.y;
                }
            }
            if (ewk.w != 0.f) {
                const uint4 u = *reinterpret_cast<const uint4*>(feat + eoff.w + c0);
                const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(h2[i]);
                    s[2 * i] += ewk.w * f.x; s[2 * i + 1] += ewk.w * f.y;
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += s[i] * wg;
        }
        __syncwarp();                            // s_e reuse by next row
    }
    if (SPLIT == 1) {
        __half* orow = out + ((size_t)b * A + a) * C + c0;
#pragma unroll
        for (int i = 0; i < 8; ++i) orow[i] = __float2half(acc[i]);
    } else {
        float* prow = partial + ((size_t)(b * A + a) * SPLIT + sp) * C + c0;
#pragma unroll
        for (int i = 0; i < 8; ++i) prow[i] = acc[i];
    }
}

// SPLIT>1 epilogue: sum the fp32 chunk partials (fixed order) -> fp16 out.
__global__ void dfa_pg_gather_finalize_kernel(
    __half* __restrict__ out, const float* __restrict__ partial,
    int A, int SPLIT, int C) {
    const int a = blockIdx.x;                    // grid = bs*A
    const int t = threadIdx.x;                   // half2 slot, block = C/2
    float s0 = 0.f, s1 = 0.f;
    const float* p = partial + (size_t)a * SPLIT * C + t * 2;
    for (int sp = 0; sp < SPLIT; ++sp) {
        s0 += p[(size_t)sp * C];
        s1 += p[(size_t)sp * C + 1];
    }
    reinterpret_cast<__half2*>(out + (size_t)a * C)[t] = __floats2half2_rn(s0, s1);
}

}  // namespace dfa_pg

// ---- C launchers (bench + plugin) -------------------------------------------
static inline size_t pg_workspace_bytes_impl(
        int rows, int cams, int levels) {
    const int CS = cams * levels;
    return (size_t)rows * CS * sizeof(dfa_pg::Entry) + (size_t)rows * 4 + 256;
}

// v5 chunking: warps = A*SPLIT >= 768 (16 SM x 48 warps on sm_87) when pts
// allows, chunks >= 16 rows to amortize per-row staging.
static inline void pg_split_calc(int A, int pts, int* SPLIT, int* CH) {
    int split = (768 + A - 1) / A;
    const int max_split = (pts + 15) / 16;
    if (split > max_split) split = max_split;
    if (split > pts) split = pts;
    if (split < 1) split = 1;
    *SPLIT = split;
    *CH = (pts + split - 1) / split;
}

static inline size_t pg_partial_bytes(int bs, int A, int pts, int C) {
    int SPLIT, CH;
    pg_split_calc(A, pts, &SPLIT, &CH);
    return SPLIT > 1 ? (size_t)bs * A * SPLIT * C * 4 : 0;
}

static inline size_t pg_workspace_bytes_v5(
        int bs, int N, int A, int cams, int levels, int C) {
    return pg_workspace_bytes_impl((size_t)bs * N, cams, levels) +
           pg_partial_bytes(bs, A, N / A, C);
}

static inline int pg_smem_bytes(int ISP, int pts, int levels, int G, int nwarps) {
    const int S = (ISP + nwarps - 1) / nwarps;
    const int wpw = (S + 31) >> 5;
    size_t bytes = (size_t)(3 * G + 2 * nwarps * G + levels * 3) * 4;   // v4: +s_inv[G]
    bytes += (size_t)pts * 2 + 4;
    bytes += (size_t)nwarps * wpw * 4;
    return (int)bytes;
}

// Computes phase layout; returns 0 ok, !=0 geometry violation (msg printed).
static inline int pg_check_geom(int A, int N, int cams, int levels,
                                int C, int G, int pts) {
    if (A * pts != N) { printf("[dfapg] bad A*pts!=N (%d*%d vs %d)\n", A, pts, N); return 1; }
    if (G > 8 || G <= 0) { printf("[dfapg] bad G=%d\n", G); return 1; }
    if (C != 256 || G != 8) { printf("[dfapg] v5 gather needs C=256 G=8 (got %d,%d)\n", C, G); return 1; }
    if (C % G != 0 || C / G > 32) { printf("[dfapg] bad C/G=%d/%d\n", C, G); return 1; }
    if ((C / G) % 4 != 0) { printf("[dfapg] C/G%%4 breaks v3 quarter mapping\n"); return 1; }
    if (C < 256 || 256 % (C / 4) != 0) { printf("[dfapg] bad C=%d for v3 gather (s_e[4], whole rows/block)\n", C); return 1; }
    if (C % 32 != 0) { printf("[dfapg] bad C%%32=%d\n", C % 32); return 1; }
    if (cams * levels > 16) { printf("[dfapg] CS=%d>16\n", cams * levels); return 1; }
    if (pts > 4096) { printf("[dfapg] pts=%d>4096\n", pts); return 1; }
    return 0;
}

extern "C" int dfa_pg_plan_cuda(
    void* entries_v, int* counts,
    const void* logits_v, int logits_is_half,
    const float* loc_f, const __half* loc_h,
    const int* spatial_shape, const int* scale_start_index,
    int bs, int cams, int hw, int C, int levels, int N, int A, int G,
    float eps, cudaStream_t stream) {
    const int pts = N / A;
    const int CS = cams * levels;
    const int ISP = CS * pts;
    const int nwarps = 8;
    const int smem = pg_smem_bytes(ISP, pts, levels, G, nwarps);
    dim3 grid(bs * A), block(256);
    if (!logits_is_half) {
        if (G == 8)
            dfa_pg::dfa_pg_plan_kernel<float, true><<<grid, block, smem, stream>>>(
                (dfa_pg::Entry*)entries_v, counts, (const float*)logits_v,
                loc_f, loc_h, spatial_shape, scale_start_index,
                A, pts, N, cams, levels, G, hw, C, eps);
        else
            dfa_pg::dfa_pg_plan_kernel<float, false><<<grid, block, smem, stream>>>(
                (dfa_pg::Entry*)entries_v, counts, (const float*)logits_v,
                loc_f, loc_h, spatial_shape, scale_start_index,
                A, pts, N, cams, levels, G, hw, C, eps);
    } else {
        if (G == 8)
            dfa_pg::dfa_pg_plan_kernel<__half, true><<<grid, block, smem, stream>>>(
                (dfa_pg::Entry*)entries_v, counts, (const __half*)logits_v,
                loc_f, loc_h, spatial_shape, scale_start_index,
                A, pts, N, cams, levels, G, hw, C, eps);
        else
            dfa_pg::dfa_pg_plan_kernel<__half, false><<<grid, block, smem, stream>>>(
                (dfa_pg::Entry*)entries_v, counts, (const __half*)logits_v,
                loc_f, loc_h, spatial_shape, scale_start_index,
                A, pts, N, cams, levels, G, hw, C, eps);
    }
    return cudaGetLastError() != cudaSuccess ? 1 : 0;
}

extern "C" int dfa_pg_gather_cuda(
    __half* out, const __half* feat,
    const void* entries_v, const int* counts, void* partial_v,
    int bs, int C, int G, int cams, int levels, int N, int A,
    cudaStream_t stream) {
    const int pts = N / A;
    int SPLIT, CH;
    pg_split_calc(A, pts, &SPLIT, &CH);
    const int tasks = bs * A * SPLIT;
    const int grid = (tasks + 7) / 8;
    dfa_pg::dfa_pg_gather_kernel_v5<<<grid, 256, 0, stream>>>(
        out, feat, (const dfa_pg::Entry*)entries_v, counts, (float*)partial_v,
        A, pts, SPLIT, CH, C, G, cams * levels);
    if (SPLIT > 1)
        dfa_pg::dfa_pg_gather_finalize_kernel<<<bs * A, C / 2, 0, stream>>>(
            out, (const float*)partial_v, A, SPLIT, C);
    return cudaGetLastError() != cudaSuccess ? 1 : 0;
}

extern "C" int dfa_pg_forward_cuda(
    __half* out, const __half* feat,
    const int* spatial_shape, const int* scale_start_index,
    const float* loc_f, const __half* loc_h,
    const void* logits_v, int logits_is_half,
    void* workspace,
    int bs, int cams, int hw, int C, int levels, int N, int A, int G,
    float eps, cudaStream_t stream) {
    const int rows = bs * N;
    uint8_t* ws = (uint8_t*)workspace;
    void* entries = ws;
    int* counts = (int*)(ws + (size_t)rows * cams * levels * sizeof(dfa_pg::Entry));
    void* partial = ws + (size_t)rows * cams * levels * sizeof(dfa_pg::Entry) +
                    (size_t)rows * 4 + 256;
    if (dfa_pg_plan_cuda(entries, counts, logits_v, logits_is_half,
                         loc_f, loc_h, spatial_shape, scale_start_index,
                         bs, cams, hw, C, levels, N, A, G, eps, stream)) return 1;
    return dfa_pg_gather_cuda(out, feat, entries, counts, partial,
                              bs, C, G, cams, levels, N, A, stream);
}

extern "C" size_t dfa_pg_workspace_bytes(int rows, int cams, int levels) {
    return pg_workspace_bytes_impl(rows, cams, levels);
}

// ==== FusedMHA: 8 heads x head_dim 32, B=1, S from input dims, no mask ======
// Input  x [1, S, 768] half: 768 = Q|K|V three contiguous 256 segments, each
//        segment [h*32 + d] (S-major), i.e. Q(s,h,d)=x[s*768 + h*32 + d],
//        K at +256, V at +512.
// Output y [1, S, 256] half: y(s,h,d) = sum_k softmax_k(q·k*scale) * v(k,h,d).
// Replaces per attention block: Reshape -> Transpose -> Gather x3 -> Mul(scale)
// -> MatMul(scores) -> Softmax -> MatMul(ctx) -> Transpose -> Reshape.
//
// v2 (flash-tile): block = 8 warps = 8 query rows of ONE head; K/V tiles of
// 128 k-rows are staged once into double-buffered smem per block, so L2
// re-read amplification drops from S-1x (v1 warp-per-row read global) to
// S/8-1x. v1 measured 3.63 ms on S=1024 at only ~292 GB/s effective
// (latency/occupancy bound on 1 GB of L2 traffic); v2 cuts that traffic 8x.
// - smem layout is chunk-transposed [stage][K/V][chunk c][row]: the 32 lanes
//   reading the SAME 16B chunk of 32 consecutive rows land on consecutive
//   addresses -> conflict-free LDS.128 (row-major 64B rows would 16-way
//   bank-conflict at stride 64B).
// - software pipeline: global->register prefetch of tile t+1 issued before
//   computing tile t from smem, stores + one __syncthreads per tile. No
//   warp may exit early (breaks the barrier); rows >= S just skip compute.
// - numerics are bit-identical to v1: same lane->k-row map, same fp32
//   accumulation order (chunk c, dims 2e/2e+1), same tile-128 online
//   softmax with warp shfl reductions, scale pinned 1/sqrt(32).
namespace mha_sd {

constexpr int MHA_TILE = 128;   // k-rows staged per tile

__device__ __forceinline__ float mha_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float mha_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void __launch_bounds__(256, 2) mha_fwd_kernel(
    __half* __restrict__ out, const __half* __restrict__ x, int S) {
    __shared__ uint4 buf[2][2][4][MHA_TILE];   // [stage][K=0/V=1][chunk][row]

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;         // query row (8 rows / block)
    const int h = blockIdx.y;                    // head
    const int nt = (S + MHA_TILE - 1) / MHA_TILE;

    // per-thread slice of a tile: 4 16B chunks; idx -> (kv, c, row)
    uint4 u[4];
    auto load_g2r = [&](int t) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = (int)threadIdx.x + i * 256;
            const int kr = t * MHA_TILE + (idx & (MHA_TILE - 1));
            if (kr < S) {
                const int kv = idx >> 9;
                const int c = (idx >> 7) & 3;
                u[i] = *reinterpret_cast<const uint4*>(
                    x + (size_t)kr * 768 + (kv ? 512 : 256) + h * 32 + c * 8);
            }
        }
    };
    auto store_r2s = [&](int t, int stage) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = (int)threadIdx.x + i * 256;
            const int kr = t * MHA_TILE + (idx & (MHA_TILE - 1));
            if (kr < S) {
                const int kv = idx >> 9;
                const int c = (idx >> 7) & 3;
                buf[stage][kv][c][idx & (MHA_TILE - 1)] = u[i];
            }
        }
    };

    float m = -3.0e38f, s = 0.f, y[32];
#pragma unroll
    for (int d = 0; d < 32; ++d) y[d] = 0.f;
    float q[32];
    if (r < S) {
        const __half* qrow = x + (size_t)r * 768 + h * 32;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 uu = *reinterpret_cast<const uint4*>(qrow + j * 8);
            const __half2* h2 = reinterpret_cast<const __half2*>(&uu);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __half22float2(h2[e]);
                q[j * 8 + 2 * e] = f.x;
                q[j * 8 + 2 * e + 1] = f.y;
            }
        }
    }

    load_g2r(0);
    store_r2s(0, 0);
    __syncthreads();
    for (int t = 0; t < nt; ++t) {
        if (t + 1 < nt) load_g2r(t + 1);         // overlap with compute below
        if (r < S) {
            float sc[4], p[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) sc[j] = -3.0e38f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kr = t * MHA_TILE + lane + j * 32;
                if (kr >= S) break;              // kr monotonic in j
                float dot = 0.f;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 uu = buf[t & 1][0][c][lane + j * 32];
                    const __half2* h2 = reinterpret_cast<const __half2*>(&uu);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = __half22float2(h2[e]);
                        dot = fmaf(q[c * 8 + 2 * e], f.x, dot);
                        dot = fmaf(q[c * 8 + 2 * e + 1], f.y, dot);
                    }
                }
                sc[j] = dot * 0.1767766922712326f;   // = 1/sqrt(32)
            }
            float mt = sc[0];
#pragma unroll
            for (int j = 1; j < 4; ++j) mt = fmaxf(mt, sc[j]);
            mt = mha_warp_max(mt);
            float st = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[j] = (sc[j] == -3.0e38f) ? 0.f : __expf(sc[j] - mt);
                st += p[j];
            }
            st = mha_warp_sum(st);
            const float mn = fmaxf(m, mt);
            const float c1 = (m == -3.0e38f) ? 0.f : __expf(m - mn);
            const float c2 = __expf(mt - mn);
            s = s * c1 + st * c2;
#pragma unroll
            for (int d = 0; d < 32; ++d) y[d] *= c1;
#pragma unroll
            for (int j = 0; j < 4; ++j) p[j] *= c2;  // rescale new tile
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kr = t * MHA_TILE + lane + j * 32;
                if (kr >= S) break;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 uu = buf[t & 1][1][c][lane + j * 32];
                    const __half2* h2 = reinterpret_cast<const __half2*>(&uu);
                    const float pj = p[j];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = __half22float2(h2[e]);
                        y[c * 8 + 2 * e] = fmaf(pj, f.x, y[c * 8 + 2 * e]);
                        y[c * 8 + 2 * e + 1] =
                            fmaf(pj, f.y, y[c * 8 + 2 * e + 1]);
                    }
                }
            }
            m = mn;
        }
        if (t + 1 < nt) store_r2s(t + 1, (t + 1) & 1);
        __syncthreads();
    }
    if (r < S) {
        // y[d] is lane-local (each lane owns 4 k-rows/tile); reduce across
        // the warp before normalizing -- s/m are already warp-uniform.
        for (int d = 0; d < 32; ++d) y[d] = mha_warp_sum(y[d]);
        const float inv = (s > 0.f) ? 1.f / s : 0.f;
        __half2* orow = reinterpret_cast<__half2*>(out) + (size_t)r * 128 + h * 16;
        if (lane < 16)
            orow[lane] = __floats2half2_rn(y[2 * lane] * inv, y[2 * lane + 1] * inv);
    }
}

}  // namespace mha_sd

extern "C" int mha_forward_cuda(
    __half* out, const __half* x, int S, cudaStream_t stream) {
    if (S <= 0) return 1;
    const dim3 block(256), grid((S + 7) / 8, 8);
    mha_sd::mha_fwd_kernel<<<grid, block, 0, stream>>>(out, x, S);
    return cudaGetLastError() != cudaSuccess ? 1 : 0;
}

namespace {

class DeformableAggregationPgPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    DeformableAggregationPgPlugin() = default;
    DeformableAggregationPgPlugin(const void* data, size_t length) {
        (void)data; (void)length;
    }
    ~DeformableAggregationPgPlugin() override = default;

    const char* getPluginType() const noexcept override {
        return "DeformableAggregationPg";
    }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)outputIndex; (void)nbInputs; (void)exprBuilder;
        nvinfer1::DimsExprs out{};
        out.nbDims = 3;
        out.d[0] = inputs[0].d[0];                  // bs
        out.d[1] = inputs[4].d[1];                  // A (per-anchor sum fused in-plugin)
        out.d[2] = inputs[0].d[3];                  // C
        return out;
    }

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) noexcept override {
        // in {feat=0, ss=1, ssi=2, loc=3, logits=4}, out {5}
        (void)nbInputs; (void)nbOutputs;
        if (pos == 1 || pos == 2)
            return inOut[pos].type == nvinfer1::DataType::kINT32 &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        if (pos == 0)  // half-only: fp32 combos forfeit the half gain (A lesson)
            return inOut[pos].type == nvinfer1::DataType::kHALF &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        if (pos == 3 || pos == 4)
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
                    inOut[pos].type == nvinfer1::DataType::kHALF) &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        // output: HALF (feat is forced HALF)
        return inOut[pos].type == nvinfer1::DataType::kHALF &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {
        (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
    }

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* in, int nbInputs,
        const nvinfer1::PluginTensorDesc* out, int nbOutputs) const noexcept override {
        (void)nbInputs; (void)out; (void)nbOutputs;
        const int bs = in[3].dims.d[0];
        const int N = in[3].dims.d[1];
        const int A = in[4].dims.d[1];
        const int cams = in[0].dims.d[1];
        const int levels = in[1].dims.d[0];
        const int C = in[0].dims.d[3];
        return pg_workspace_bytes_v5(bs, N, A, cams, levels, C);
    }

    int enqueue(const nvinfer1::PluginTensorDesc* in, const nvinfer1::PluginTensorDesc* out,
                const void* const* inputs, void* const* outputs, void* workspace,
                cudaStream_t stream) noexcept override {
        (void)out;
        static bool pg_probe = false;
        if (!pg_probe) {
            pg_probe = true;
            fprintf(stderr,
                    "[dfapg] combo feat=%d loc=%d logits=%d out=%d (0=f32 1=half)\n",
                    (int)in[0].type, (int)in[3].type, (int)in[4].type,
                    (int)out[0].type);
        }
        const int bs = in[0].dims.d[0];
        const int cams = in[0].dims.d[1];
        const int hw = in[0].dims.d[2];
        const int C = in[0].dims.d[3];
        const int levels = in[1].dims.d[0];
        const int N = in[3].dims.d[1];
        const int A = in[4].dims.d[1];
        const int G = in[4].dims.d[in[4].dims.nbDims - 1];
        const int ISP = in[4].dims.d[in[4].dims.nbDims - 2];
        const int pts = N / A;
        if (pg_check_geom(A, N, cams, levels, C, G, pts) ||
            ISP != cams * levels * pts) {
            fprintf(stderr, "[dfapg] geometry mismatch, ISP=%d\n", ISP);
            return 1;
        }
        return dfa_pg_forward_cuda(
            (__half*)outputs[0], (const __half*)inputs[0],
            (const int*)inputs[1], (const int*)inputs[2],
            in[3].type == nvinfer1::DataType::kHALF ? nullptr : (const float*)inputs[3],
            in[3].type == nvinfer1::DataType::kHALF ? (const __half*)inputs[3] : nullptr,
            inputs[4], in[4].type == nvinfer1::DataType::kHALF ? 1 : 0,
            workspace, bs, cams, hw, C, levels, N, A, G,
            DFA_PG_EPS, stream);
    }

    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override {
        (void)index; (void)nbInputs;
        return inputTypes[0] == nvinfer1::DataType::kHALF
                   ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    }

    const char* getPluginNamespace() const noexcept override { return m_ns.c_str(); }
    void setPluginNamespace(const char* ns) noexcept override { m_ns = ns ? ns : ""; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override { (void)buffer; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* p = new DeformableAggregationPgPlugin(*this);
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }

private:
    std::string m_ns;
};

class DeformableAggregationPgPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const noexcept override {
        return "DeformableAggregationPg";
    }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &m_fields;
    }
    nvinfer1::IPluginV2* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        (void)name; (void)fc;
        auto* p = new DeformableAggregationPgPlugin();
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override {
        (void)name; (void)serialData; (void)serialLength;
        auto* p = new DeformableAggregationPgPlugin(serialData, serialLength);
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }
    void setPluginNamespace(const char* ns) noexcept override { m_ns = ns ? ns : ""; }
    const char* getPluginNamespace() const noexcept override { return m_ns.c_str(); }

private:
    std::string m_ns{""};
    nvinfer1::PluginFieldCollection m_fields{0, nullptr};
};

// same dual-domain registration discipline as the v1 plugin: "" first
// (onnxparser queries the empty namespace; dedup keeps the first).
struct PgMultiDomainRegistrar {
    PgMultiDomainRegistrar() {
        auto* registry = getPluginRegistry();
        registry->registerCreator(*new DeformableAggregationPgPluginCreator(), "");
        registry->registerCreator(*new DeformableAggregationPgPluginCreator(),
                                  "sparsedrivev2");
    }
};
static PgMultiDomainRegistrar g_pg_multi_domain_registrar;

class FusedMhaPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    FusedMhaPlugin() = default;
    FusedMhaPlugin(const void* data, size_t length) {
        (void)data; (void)length;
    }
    ~FusedMhaPlugin() override = default;

    const char* getPluginType() const noexcept override { return "FusedMHA"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)outputIndex; (void)nbInputs; (void)exprBuilder;
        nvinfer1::DimsExprs out{};
        out.nbDims = 3;
        out.d[0] = inputs[0].d[0];
        out.d[1] = inputs[0].d[1];
        out.d[2] = exprBuilder.constant(256);
        return out;
    }

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) noexcept override {
        (void)nbInputs; (void)nbOutputs;
        return inOut[pos].type == nvinfer1::DataType::kHALF &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {
        (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
    }

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* in, int nbInputs,
        const nvinfer1::PluginTensorDesc* out, int nbOutputs) const noexcept override {
        (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
        return 0;
    }

    int enqueue(const nvinfer1::PluginTensorDesc* in, const nvinfer1::PluginTensorDesc* out,
                const void* const* inputs, void* const* outputs, void* workspace,
                cudaStream_t stream) noexcept override {
        (void)out; (void)workspace;
        const int S = in[0].dims.d[1];
        if (in[0].dims.d[2] != 768) {
            fprintf(stderr, "[fusedmha] bad last dim %d\n", in[0].dims.d[2]);
            return 1;
        }
        return mha_forward_cuda((__half*)outputs[0], (const __half*)inputs[0],
                                S, stream);
    }

    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override {
        (void)index; (void)nbInputs;
        return inputTypes[0];
    }

    const char* getPluginNamespace() const noexcept override { return m_ns.c_str(); }
    void setPluginNamespace(const char* ns) noexcept override { m_ns = ns ? ns : ""; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override { (void)buffer; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* p = new FusedMhaPlugin(*this);
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }

private:
    std::string m_ns;
};

class FusedMhaPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const noexcept override { return "FusedMHA"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &m_fields;
    }
    nvinfer1::IPluginV2* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        (void)name; (void)fc;
        auto* p = new FusedMhaPlugin();
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override {
        (void)name; (void)serialData; (void)serialLength;
        auto* p = new FusedMhaPlugin(serialData, serialLength);
        p->setPluginNamespace(m_ns.c_str());
        return p;
    }
    void setPluginNamespace(const char* ns) noexcept override { m_ns = ns ? ns : ""; }
    const char* getPluginNamespace() const noexcept override { return m_ns.c_str(); }

private:
    std::string m_ns{""};
    nvinfer1::PluginFieldCollection m_fields{0, nullptr};
};

struct MhaMultiDomainRegistrar {
    MhaMultiDomainRegistrar() {
        auto* registry = getPluginRegistry();
        registry->registerCreator(*new FusedMhaPluginCreator(), "");
        registry->registerCreator(*new FusedMhaPluginCreator(), "sparsedrivev2");
    }
};
static MhaMultiDomainRegistrar g_mha_multi_domain_registrar;

}  // namespace
