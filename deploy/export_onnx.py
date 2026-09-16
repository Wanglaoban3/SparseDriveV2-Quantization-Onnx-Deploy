"""Export SparseDriveV2 to ONNX with sparsedrivev2::DeformableAggregation custom op.

Run inside an env where the repo + ckpt are reachable and torch.cuda works
(tracing executes the real CUDA deformable op).

Usage: python export_onnx.py [--out model.onnx] [--calib N]
"""
import argparse
import os
import sys

import numpy as np
import torch

DEPLOY_DIR = os.environ.get("SD_DEPLOY_DIR", os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

OUT = os.environ.get("TRT_DEPLOY_DIR", os.path.join(DEPLOY_DIR, "artifacts"))
os.makedirs(OUT, exist_ok=True)

# force nn.MultiheadAttention onto the decomposed (exportable) path handled below.
# aten::unflatten has no symbolic in torch 2.0.1; emulate with reshape (identical math).
def _unflatten_reshape(self, dim, sizes):
    shape = list(self.shape)
    d = self.ndim + dim if dim < 0 else dim
    new_shape = shape[:d] + list(sizes) + shape[d + 1:]
    return self.reshape(new_shape)

torch.Tensor.unflatten = _unflatten_reshape

from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
import navsim.agents.sparsedrive.ops.deformable_aggregation as damod

METRICS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
           "time_to_collision_within_bound", "comfort", "ego_progress"]

# When set ((256,) float array), the DFA symbolic emits a per-channel QuantizeLinear on
# the feat input (axis=3) plus the scale as a Constant 6th plugin input, so the exported
# ONNX natively carries feat-INT8 (plugin dequantizes in-kernel). No post-editing needed.
FEAT_INT8_SCALE = None


def make_symbolic():
    """symbolic for DeformableAggregationFunction; shapes read from traced values."""

    def _cast32(g, t):
        return g.op("Cast", t, to_i=3)  # 3 = ONNX INT32

    def symbolic(g, feat, ss, ssi, loc, w):
        feat_in = feat
        extra = []
        if FEAT_INT8_SCALE is not None:
            scale_t = torch.tensor(FEAT_INT8_SCALE, dtype=torch.float32)
            scale_c = g.op("Constant", value_t=scale_t)
            zp_c = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
            feat_in = g.op("QuantizeLinear", feat, scale_c, zp_c, axis_i=3)
            extra = [scale_c]
        out = g.op("sparsedrivev2::DeformableAggregation",
                   feat_in, _cast32(g, ss), _cast32(g, ssi), loc, w, *extra)
        # dtype only; static shapes are patched into value_info after export
        try:
            out.setType(feat.type().with_dtype(torch.float))
        except Exception:
            pass
        return out

    return symbolic


def calibrate_feat_int8_scale(wrapper, ds, calib_idx):
    """Per-embed-channel amax of the DFA feat input, maxed over all call sites/samples."""
    amax = None
    orig = damod.DeformableAggregationFunction.forward

    def patched(ctx, feat, ss, ssi, loc, w):
        nonlocal amax
        m = feat.abs().amax(dim=(0, 1, 2)).float().cpu().numpy()
        amax = m if amax is None else np.maximum(amax, m)
        return orig(ctx, feat, ss, ssi, loc, w)

    damod.DeformableAggregationFunction.forward = staticmethod(patched)
    with torch.no_grad():
        for i in calib_idx:
            inp, _, _ = to_inputs(ds, i)
            wrapper(*inp)
    damod.DeformableAggregationFunction.forward = orig
    return (amax / 127.0).astype(np.float32)


class QuantizableMHA(torch.nn.Module):
    """nn.MultiheadAttention with nn.Linear projections so ModelOpt/PTQ can quantize them.
    Mathematically identical to batch_first=True self-attention MHA."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.in_proj = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, *a, **k):
        B, L, E = query.shape
        qkv = self.in_proj(query).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scaling) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, E)
        return self.out_proj(out), None


def swap_mha_for_quantizable(module):
    """Replace every nn.MultiheadAttention with QuantizableMHA (weights copied 1:1)."""
    import torch.nn as nn
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            assert child.batch_first and child._qkv_same_embed_dim and child.bias_k is None
            q = QuantizableMHA(child.embed_dim, child.num_heads).to(child.in_proj_weight.device)
            q.in_proj.weight.data.copy_(child.in_proj_weight.data)
            q.in_proj.bias.data.copy_(child.in_proj_bias.data)
            q.out_proj.weight.data.copy_(child.out_proj.weight.data)
            q.out_proj.bias.data.copy_(child.out_proj.bias.data)
            setattr(module, name, q)
        else:
            swap_mha_for_quantizable(child)
    return module


class FrozenVocabHead(torch.nn.Module):
    """Precomputes vocabulary embeddings/constants so they become ONNX initializers
    instead of being recomputed every frame."""

    def __init__(self, head):
        super().__init__()
        self.decoder = head.decoder
        with torch.no_grad():
            path_vocab = head.path_vocab.data[None]
            vel_vocab = head.vel_vocab.data[None]
            traj_vocab = head.traj_vocab.data[None]
            traj_mask = head.traj_mask.data[None]
            self.register_buffer("f_path_embed", head.path_pos_embed(path_vocab.flatten(-2, -1)))
            self.register_buffer("f_vel_embed", head.vel_pos_embed(vel_vocab))
            self.register_buffer("f_path_vocab", path_vocab)
            self.register_buffer("f_vel_vocab", vel_vocab)
            self.register_buffer("f_traj_vocab", traj_vocab)
            self.register_buffer("f_traj_mask", traj_mask)

    def forward(self, camera_feature, status_encoding, targets):
        decoder_outputs = self.decoder(
            (self.f_path_embed, self.f_vel_embed, self.f_path_vocab,
             self.f_vel_vocab, self.f_traj_vocab, self.f_traj_mask),
            (camera_feature, status_encoding, targets),
        )
        return decoder_outputs


def apply_deploy_optimizations(model):
    """Frozen vocab + quantizable MHA. Idempotent-ish; call once after checkpoint load."""
    swap_mha_for_quantizable(model)
    model._trajectory_head = FrozenVocabHead(model._trajectory_head).to(next(model.parameters()).device)
    return model


LOG_NAMES = ["2021.06.28.16.29.11_veh-38_01415_01821", "2021.10.11.02.57.41_veh-50_01522_02088"]


def build_dataset(cfg):
    from navsim.planning.training.dataset import CacheOnlyDataset
    from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder, SparseDriveTargetBuilder
    return CacheOnlyDataset(
        cache_path=os.path.join(ROOT, "exp", "data_cache_mini"),
        feature_builders=[SparseDriveFeatureBuilder(cfg)],
        target_builders=[SparseDriveTargetBuilder(cfg)],
        log_names=LOG_NAMES,
    )


def to_inputs(ds, i):
    features, _targets, token = ds[i]
    cf = features["camera_feature"]
    imgs = cf["imgs"].unsqueeze(0).float().cuda()
    proj = cf["projection_mat"].unsqueeze(0).float().cuda()
    iwh = torch.as_tensor(np.asarray(cf["image_wh"]), dtype=torch.float32).unsqueeze(0).cuda()
    status = features["status_feature"].unsqueeze(0).float().cuda()
    return (imgs, proj, iwh, status), token


class ExportModel(torch.nn.Module):
    """Flatten dict-of-tensors model interface into 4 positional tensor inputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs, projection_mat, image_wh, status_feature, targets=None):
        camera_feature = {"imgs": imgs, "projection_mat": projection_mat, "image_wh": image_wh}
        output, _ = self.model({"camera_feature": camera_feature, "status_feature": status_feature}, targets)
        outs = [output["trajectory"], output["traj_scores"]]
        outs += [output[f"metric_{m}"] for m in METRICS]
        return tuple(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "sparsedrive_fp32.onnx"))
    ap.add_argument("--calib", type=int, default=16, help="dump N calibration samples (0=skip)")
    args = ap.parse_args()

    cfg = SparseDriveConfig()
    cfg.dataset_version = "v1"
    cfg.metrics = list(METRICS)
    cfg.velocity_filter_num = [64, 20]
    cfg.path_filter_num = [128, 20]

    agent = SparseDriveAgent(config=cfg, lr=1e-4, checkpoint_path=os.path.join(ROOT, "ckpt", "sparsedrive_navsimv1_92p2.ckpt"))
    agent.initialize()
    model = agent._sparsedrive_model.cuda().eval()
    apply_deploy_optimizations(model)

    # register custom-op symbolic (harmless outside export)
    damod.DeformableAggregationFunction.symbolic = staticmethod(make_symbolic())

    wrapper = ExportModel(model).cuda().eval()
    # MHA fastpath (_native_multi_head_attention) is not exportable; its only gate in
    # torch 2.0.1 is module.training. dropout==0 so train() changes nothing numerically.
    import torch.nn as _nn
    for m in wrapper.modules():
        if isinstance(m, _nn.MultiheadAttention):
            m.train()

    # ---- real sample from feature cache ----
    from navsim.planning.training.dataset import CacheOnlyDataset
    from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder, SparseDriveTargetBuilder

    ds = CacheOnlyDataset(
        cache_path=os.path.join(ROOT, "exp", "data_cache_mini"),
        feature_builders=[SparseDriveFeatureBuilder(cfg)],
        target_builders=[SparseDriveTargetBuilder(cfg)],
        log_names=["2021.06.28.16.29.11_veh-38_01415_01821", "2021.10.11.02.57.41_veh-50_01522_02088"],
    )
    print("calib pool size:", len(ds))

    def to_inputs(i):
        features, _targets, token = ds[i]
        cf = features["camera_feature"]
        imgs = cf["imgs"].unsqueeze(0).float().cuda()
        proj = cf["projection_mat"].unsqueeze(0).float().cuda()
        iwh = torch.as_tensor(np.asarray(cf["image_wh"]), dtype=torch.float32).unsqueeze(0).cuda()
        status = features["status_feature"].unsqueeze(0).float().cuda()
        return imgs, proj, iwh, status, token

    # torch reference outputs (for later engine parity)
    with torch.no_grad():
        example = to_inputs(0)
        ref = wrapper(*example[:4])
    names = ["trajectory", "traj_scores"] + [f"metric_{m}" for m in METRICS]
    print("reference outputs:", {n: tuple(t.shape) for n, t in zip(names, ref)})

    # ---- calibration dump ----
    if args.calib > 0:
        calib_dir = os.path.join(OUT, "calib")
        os.makedirs(calib_dir, exist_ok=True)
        step = max(1, len(ds) // args.calib)
        k = 0
        for i in range(0, len(ds), step):
            if k >= args.calib:
                break
            imgs, proj, iwh, status, token = to_inputs(i)
            np.savez(os.path.join(calib_dir, f"calib_{k:02d}.npz"),
                     imgs=imgs.cpu().numpy(), proj=proj.cpu().numpy(),
                     iwh=iwh.cpu().numpy(), status=status.cpu().numpy(), token=token)
            k += 1
        print("calib samples dumped:", k)

    # ---- export ----
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example[:4],
            args.out,
            input_names=["imgs", "projection_mat", "image_wh", "status_feature"],
            output_names=names,
            opset_version=17,
            do_constant_folding=False,  # mixed-device constants crash torch's folder; TRT re-folds anyway
            training=torch.onnx.TrainingMode.PRESERVE,  # keep MHA in train() (dropout=0) to dodge non-exportable fastpath
        )
    print("exported:", args.out)

    # ---- structural sanity + patch value_info for DFA outputs ----
    import onnx
    m = onnx.load(args.out)
    dfa_nodes = [n_ for n_ in m.graph.node if n_.op_type == "DeformableAggregation"]
    print("DFA nodes in graph:", len(dfa_nodes))
    assert len(dfa_nodes) == 3, "expected 3 DeformableAggregation nodes"

    expected = [512000, 64000, 32000]  # static N per call site, in graph order
    vi_missing = []
    for nd, n_exp in zip(dfa_nodes, sorted(expected, key=lambda x: -x)):
        oname = nd.output[0]
        has_vi = any(v.name == oname for v in m.graph.value_info)
        print(f"  {oname}: value_info present={has_vi}")
        if not has_vi:
            vi_missing.append((oname, n_exp))
    for oname, n_exp in vi_missing:
        vi = m.graph.value_info.add()
        vi.name = oname
        t = vi.tensor_type
        t.elem_type = onnx.TensorProto.FLOAT
        for d in (1, n_exp, 256):
            dim = t.shape.dim.add()
            dim.dim_value = d
    onnx.save(m, args.out)
    print("value_info patched:", len(vi_missing))
    print("EXPORT_OK")


if __name__ == "__main__":
    main()
