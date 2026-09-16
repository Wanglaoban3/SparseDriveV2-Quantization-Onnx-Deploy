"""Evaluate INT8 quantization loss inside the DeformableAggregation op.

Simulates an int8-storage plugin: feat / loc / w are round-tripped through
per-tensor (or per-channel) symmetric int8, then the fp32 kernel runs on the
dequantized values -- numerically identical to int8-load + fp32-compute.

Reports:
  1. DFA-output-level error (rel L2 / cosine) per call site
  2. End-to-end drift on held-out samples (trajectory / scores / metric logits)
vs the fp32 DFA baseline.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ptq_pipeline as pp
import navsim.agents.sparsedrive.ops.deformable_aggregation as damod


def q_identity(x):
    return x


def q_fixed_range(x, lo, hi):
    """fixed-range quantization with clipping (for tensors whose valid range is known)"""
    scale = (hi - lo) / 254.0
    return torch.clamp(torch.round((x - lo) / scale), 0, 254) * scale + lo


def q_loc(x):
    """fixed [0,1] range: valid coords live in (0,1); outliers are kernel-excluded"""
    return q_fixed_range(x, 0.0, 1.0)


def q_w(x):
    """fixed [0,1] range: softmax weights"""
    return q_fixed_range(x, 0.0, 1.0)


def q_per_tensor(x):
    amax = x.abs().amax().clamp_min(1e-12)
    scale = amax / 127.0
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


def q_per_channel_c(x):
    """per-channel over the embed dim (last axis), the DFA reduction axis."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


def q_per_channel_cam_c(x):
    """per (cam, channel) scale: cam dimension is dim-1 of feat."""
    amax = x.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


def q_per_channel_spatial_true(x):
    """TRUE per-C-channel: one scale per embed channel (amax over bs/cams/HW)."""
    amax = x.abs().amax(dim=(0, 1, 2), keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


def q_per_point_fp16scale(x):
    """per-point scale stored as fp16 (what a plugin would actually keep):
    quantize with fp32 scale then round the scale to fp16 precision."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 127.0).half().float()
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


state = {"mode": "off", "feat_q": q_per_tensor}
stats = {}
_orig_fwd = damod.DeformableAggregationFunction.forward


def patched_fwd(ctx, feat, ss, ssi, loc, w):
    y_fp = _orig_fwd(ctx, feat, ss, ssi, loc, w)
    if state["mode"] != "int8":
        return y_fp
    fq = state["feat_q"](feat)
    y_q = _orig_fwd(ctx, fq, ss, ssi, state["q_loc"](loc), state["q_w"](w))
    key = f"N{int(loc.shape[1])}"
    d = y_fp.double()
    q = y_q.double()
    rel = ((y_q - y_fp).norm() / y_fp.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(
        q.flatten(), y_fp.double().flatten(), dim=0).item()
    s = stats.setdefault(key, dict(rel=[], cos=[], mx=[]))
    s["rel"].append(rel)
    s["cos"].append(cos)
    s["mx"].append((y_q - y_fp).abs().max().item())
    return y_q


def main():
    cfg, wrapper, ds = pp.build()
    damod.DeformableAggregationFunction.forward = staticmethod(patched_fwd)
    val_idx = pp.VAL_IDX[:8]

    variants = [
        ("w_int8_only", q_identity, q_identity, q_w),
        ("loc_int8_only", q_identity, q_loc, q_identity),
        ("w+loc_int8", q_identity, q_loc, q_w),
        ("feat+w_int8", q_per_tensor, q_identity, q_w),
        ("all_int8", q_per_tensor, q_loc, q_w),
    ]
    print("=== fp32 baseline pass ===")
    state["mode"] = "off"
    with torch.no_grad():
        ref, gts = pp.run_batch(wrapper, ds, val_idx)
    ref_out = [list(o) for o in ref]

    for name, qf, q_loc_fn, q_w_fn in variants:
        stats.clear()
        state["mode"] = "int8"
        state["feat_q"] = qf
        state["q_loc"], state["q_w"] = q_loc_fn, q_w_fn
        print(f"=== int8 variant: {name} ===")
        outs = []
        with torch.no_grad():
            for i in val_idx:
                inp, _, _ = pp.to_inputs(ds, i)
                o = wrapper(*inp)
                outs.append([t.float().cpu().numpy() for t in o])
        state["mode"] = "off"
        for k, s in sorted(stats.items()):
            print(f"  DFA site {k}: relL2={np.mean(s['rel']):.5f}  "
                  f"cosine={np.mean(s['cos']):.6f}  maxabs={np.mean(s['mx']):.5f}")
        cmp = pp.compare([list(o) for o in ref_out], outs)
        gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(outs, gts)]))
        print(f"  end-to-end vs fp32-DFA: {pp.compare(ref_out, outs)}  GT={gt:.4f}")

    state["mode"] = "off"
    ref_gt = float(np.mean([np.abs(r[0][0] - g).mean() for r, g in zip(ref, gts)]))
    print(f"fp32 GT={ref_gt:.4f}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
