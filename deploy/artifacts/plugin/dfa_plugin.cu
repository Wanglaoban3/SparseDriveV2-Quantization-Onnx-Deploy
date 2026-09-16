// SparseDriveV2 DeformableAggregation TensorRT plugin (IPluginV2DynamicExt)
// ONNX node: op_type="DeformableAggregation", domain="sparsedrivev2"
//   inputs : [0] feat   (bs, cams, HW_total, C)   fp32/fp16
//            [1] ss     (levels, 2)              int32   (H,W per FPN level)
//            [2] ssi    (levels,)                int32   (start offset per level)
//            [3] loc    (bs, N, cams, 2)         fp32/fp16  (normalized coords)
//            [4] w      (bs, N, cams, levels, groups) fp32/fp16
//   output : (bs, N, C)                          fp32/fp16
// Math is identical to navsim/agents/sparsedrive/ops/src/deformable_aggregation_cuda.cu.
// fp16 inputs are up-cast to fp32 internally, kernels run fp32, output is down-cast.

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

// int8-storage variant: feat values are int8, dequantized per embed-channel with
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

// Generalized launcher usable from the plugin and from standalone unit tests.
// Exactly one of feat / feat_i8 is non-null; feat_scale required when feat_i8 != null.
// Exactly one of loc_f / loc_h and one of w_f / w_h is non-null (fp32 or fp16 direct read).
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
        if (pos == 0) return isFloat || isInt8;  // feat: fp32/fp16/int8
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
        if (in[0].type == nvinfer1::DataType::kINT8) {
            feat_i8 = (const int8_t*)inputs[0];
            feat_scale = (const float*)inputs[5];
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
    std::string m_ns{""};
    nvinfer1::PluginFieldCollection m_fields{0, nullptr};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(DeformableAggregationPluginCreator);
