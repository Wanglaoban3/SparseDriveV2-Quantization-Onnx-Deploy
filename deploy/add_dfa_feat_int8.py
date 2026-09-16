"""Add feat-INT8 support to an exported SparseDriveV2 ONNX:
  - calibrates a per-embed-channel (256,) fp32 scale from the calib samples
  - inserts a per-channel QuantizeLinear (axis=3) before each DFA node's feat input
  - appends the scale constant as the 6th plugin input

Plugin contract: inputs = [feat(int8), ss(int32), ssi(int32), loc(f32/f16),
w(f32/f16), feat_scale((C,) f32)].
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, DEPLOY_DIR)

import ptq_pipeline as pp
import navsim.agents.sparsedrive.ops.deformable_aggregation as damod

Q_BITS = 127.0


def calibrate_scale(wrapper, ds, calib_idx):
    """Per-channel amax of the DFA feat input across all call sites and samples."""
    amax = None
    orig = damod.DeformableAggregationFunction.forward

    def patched(ctx, feat, ss, ssi, loc, w):
        nonlocal amax
        m = feat.abs().amax(dim=(0, 1, 2)).float().cpu().numpy()  # (C,)
        amax = m if amax is None else np.maximum(amax, m)
        return orig(ctx, feat, ss, ssi, loc, w)

    damod.DeformableAggregationFunction.forward = staticmethod(patched)
    with torch.no_grad():
        for i in calib_idx:
            inp, _, _ = pp.to_inputs(ds, i)
            wrapper(*inp)
    damod.DeformableAggregationFunction.forward = orig
    scale = (amax / Q_BITS).astype(np.float32)
    return scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(DEPLOY_DIR, "artifacts", "sparsedrive_int8_qdq.onnx"))
    ap.add_argument("--out", default=os.path.join(DEPLOY_DIR, "artifacts", "sparsedrive_int8_qdq_feat8.onnx"))
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args()

    cfg, wrapper, ds = pp.build()
    calib_idx = pp.CALIB_IDX[: args.samples]
    print("calibrating feat scale on", len(calib_idx), "samples...")
    scale = calibrate_scale(wrapper, ds, calib_idx)
    print("feat scale: shape", scale.shape, "min", float(scale.min()), "max", float(scale.max()))

    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(args.onnx)
    g = m.graph

    scale_name = "dfa_feat_scale"
    zp_name = "dfa_feat_zp"
    g.initializer.append(numpy_helper.from_array(scale, scale_name))
    g.initializer.append(numpy_helper.from_array(np.zeros(scale.shape, dtype=np.int8), zp_name))

    dfa_nodes = [n for n in g.node if n.op_type == "DeformableAggregation"]
    print("DFA nodes:", len(dfa_nodes))
    assert dfa_nodes, "no DFA nodes found"

    new_nodes = []
    for k, n in enumerate(dfa_nodes):
        feat_in = n.input[0]
        q_out = f"{feat_in}/feat_i8_{k}"
        qn = helper.make_node(
            "QuantizeLinear", [feat_in, scale_name, zp_name], [q_out],
            name=f"{n.name}/feat_quant", axis=3)
        new_nodes.append((n, qn, q_out))

    # insert Q right before the first DFA node's position; all DFA feats come from
    # deformable_format ops upstream, so appending Qs before their consumers is topologically safe
    for n, qn, q_out in new_nodes:
        idx = next(i for i, nd in enumerate(g.node) if nd.name == n.name)
        g.node.insert(idx, qn)
        n.input[0] = q_out
        n.input.append(scale_name)  # 6th plugin input: (C,) fp32 scale

    onnx.save(m, args.out)
    print("saved:", args.out)

    # structural check
    m2 = onnx.load(args.out)
    cnt_q = sum(1 for n in m2.graph.node if n.op_type == "QuantizeLinear" and "feat_quant" in (n.name or ""))
    for n in m2.graph.node:
        if n.op_type == "DeformableAggregation":
            assert len(n.input) == 6, n.name
    print("feat Q nodes:", cnt_q, "| all DFA nodes have 6 inputs")
    print("FEAT8_DONE")


if __name__ == "__main__":
    main()
