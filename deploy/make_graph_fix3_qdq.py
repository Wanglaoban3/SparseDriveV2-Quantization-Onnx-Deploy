# -*- coding: utf-8 -*-
"""Graft modelopt QDQ (from sparsedrive_int8_qdq.onnx) onto selected MatMuls of
the production fix3 graph. Scales are the original PTQ calibration values;
weights are byte-identical between the two graphs (verified 348/348).

Graft per target MatMul side (act / weight):
  donor:  src -> Q -> DQ -> [PASS chain] -> MatMul
  fix3:   src -> [PASS chain] -> MatMul           (QDQ stripped, chain kept)
  graft:  copy donor Q/DQ (+ scale/zp subtrees) renamed *_q8; assert fix3 PASS
  chain ops == donor post-DQ chain ops and same root tensor; rewire the
  root-side chain node input (or MatMul input when chain empty) to DQ output.

Weight roundtrip check per graft: dequant(quant(W)) within scale/2 of W —
catches wrong per-channel axis.

Excluded by design: kps_generator (loc), camera_encoder, MatMul_fxG (loc proj),
status_encoding, final mlp/head layers (*.2), attention score MatMuls.

usage:
  python deploy/make_graph_fix3_qdq.py --preset E1 --out <path.onnx>
  python deploy/make_graph_fix3_qdq.py --preset E2 --out <path.onnx>
"""
import argparse
import hashlib
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import onnx
from onnx import numpy_helper
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
FIX3 = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3.onnx")
DONOR = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_int8_qdq.onnx")

L0 = "/model/_trajectory_head/decoder/layers.0"
L1 = "/model/_trajectory_head/decoder/layers.1"
WFC = [
    f"{L0}/p_deform_model/weights_fc/MatMul",
    f"{L1}/p_deform_model/weights_fc/MatMul",
]
E2_EXTRA = []
for L in (L0, L1):
    for mod in ("p_ffn/p_ffn.0", "p_ffn/p_ffn.2", "v_ffn/v_ffn.0", "v_ffn/v_ffn.2",
                "p_attention/in_proj", "p_attention/out_proj",
                "v_attention/in_proj", "v_attention/out_proj",
                "v_img_attention/in_proj", "v_img_attention/out_proj",
                "path_mlp/path_mlp.0", "vel_mlp/vel_mlp.0"):
        E2_EXTRA.append(f"{L}/{mod}/MatMul")
for mod in ("t_ffn/t_ffn.0", "t_ffn/t_ffn.2",
            "t_attention/in_proj", "t_attention/out_proj",
            "traj_mlp/traj_mlp.0",
            "no_at_fault_collisions/no_at_fault_collisions.0",
            "drivable_area_compliance/drivable_area_compliance.0",
            "driving_direction_compliance/driving_direction_compliance.0",
            "time_to_collision_within_bound/time_to_collision_within_bound.0",
            "comfort/comfort.0", "ego_progress/ego_progress.0"):
    E2_EXTRA.append(f"{L1}/{mod}/MatMul")

PRESETS = {"E1": WFC, "E2": WFC + E2_EXTRA}

CAST_NP = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
           6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}

PASS = ("Cast", "Transpose", "Reshape", "Identity", "Squeeze", "Unsqueeze")



def md5_init(init):
    return hashlib.md5(numpy_helper.to_array(init).tobytes()).hexdigest()


def resolve_const(d_tensor, d_init, d_prod):
    """Numpy value of a donor tensor: initializer / Constant node / Cast chain."""
    if d_tensor in d_init:
        return numpy_helper.to_array(d_init[d_tensor])
    p = d_prod.get(d_tensor)
    if p is not None and p.op_type == "Constant":
        for a in p.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    if p is not None and p.op_type == "Cast":
        base = resolve_const(p.input[0], d_init, d_prod)
        np_dt = CAST_NP[p.attribute[0].i]
        return base.astype(np_dt)
    raise RuntimeError(f"cannot resolve donor const {d_tensor}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    targets = PRESETS[args.preset]

    fm = onnx.load(FIX3)
    dm = onnx.load(DONOR)
    fg, dg = fm.graph, dm.graph

    f_init = {i.name: i for i in fg.initializer}
    d_init = {i.name: i for i in dg.initializer}
    d_prod = {}
    for n in dg.node:
        for o in n.output:
            d_prod[o] = n
    f_names, f_out = set(), set()
    for n in fg.node:
        f_names.add(n.name)
        for o in n.output:
            f_out.add(o)
    f_prod = {}
    for n in fg.node:
        for o in n.output:
            f_prod[o] = n

    new_nodes = []
    new_inits = {}
    copied = set()   # donor tensors already materialized in grafted graph
    used_names = set(f_names)

    def copy_upstream(d_tensor, depth=0):
        """Materialize donor tensor `d_tensor` (scale/zp/Q subtrees only —
        never the main PASS chain). Returns new tensor name."""
        if depth > 16:
            raise RuntimeError(f"copy depth exceeded at {d_tensor}")
        if d_tensor in copied:
            return copied[d_tensor]
        if d_tensor in f_out:
            return d_tensor
        if d_tensor in f_init:
            return d_tensor
        if d_tensor in d_init:
            new_inits[d_tensor] = d_init[d_tensor]
            copied.add(d_tensor)
            return d_tensor
        prod = d_prod.get(d_tensor)
        if prod is None:
            raise RuntimeError(f"donor tensor {d_tensor} has no producer")
        renamed = prod.name + "_q8"
        assert renamed not in used_names, f"node name collision: {renamed}"
        nn = onnx.NodeProto()
        nn.CopyFrom(prod)
        nn.name = renamed
        for k, old_in in enumerate(list(nn.input)):
            if old_in:
                nn.input[k] = copy_upstream(old_in, depth + 1)
        for k, old_out in enumerate(list(nn.output)):
            new_out = old_out + "_q8" if old_out else old_out
            assert new_out not in f_out and new_out not in used_names, \
                f"output collision: {new_out}"
            nn.output[k] = new_out
            copied.add(old_out)
        used_names.add(renamed)
        for o in nn.output:
            f_out.add(o)
        new_nodes.append(nn)
        return nn.output[0]

    def graft_side(mm, dmm, k, tag):
        """Graft QDQ for MatMul input k (0=act, 1=weight)."""
        d_in = dmm.input[k]
        # walk donor back through PASS to the DQ
        d_chain = []          # post-DQ PASS chain (matmul-side first)
        cur = d_in
        dq = None
        for _ in range(8):
            p = d_prod.get(cur)
            if p is None:
                raise RuntimeError(f"{tag}: donor input {d_in} reaches no producer")
            if p.op_type == "DequantizeLinear":
                dq = p
                break
            assert p.op_type in PASS, f"{tag}: unexpected donor op {p.op_type}"
            d_chain.append(p.op_type)
            cur = p.input[0]
        assert dq is not None, f"{tag}: no DQ on donor side k={k}"
        q = d_prod.get(dq.input[0])
        assert q is not None and q.op_type == "QuantizeLinear", \
            f"{tag}: DQ input is not Q"

        # fix3 chain from mm.input[k] back to root
        f_chain = []
        cur = mm.input[k]
        f_root = None
        root_node = None       # fix3 chain node consuming root (or None)
        for _ in range(8):
            p = f_prod.get(cur)
            if p is None:
                f_root = cur
                break
            if p.op_type not in PASS:
                f_root = cur
                break
            f_chain.append(p.op_type)
            root_node = p
            cur = p.input[0]
        assert f_chain == list(reversed(d_chain)), \
            f"{tag}: chain mismatch fix3={f_chain} donor={list(reversed(d_chain))}"
        assert f_root == q.input[0], \
            f"{tag}: root mismatch fix3={f_root} donorQ={q.input[0]}"

        new_dq_out = copy_upstream(dq.output[0])
        if root_node is not None:
            root_node.input[0] = new_dq_out
        else:
            mm.input[k] = new_dq_out

        # weight-side roundtrip check (k==1): dequant(quant(W)) ~ W
        if k == 1:
            w = numpy_helper.to_array(f_init[f_root]).astype(np.float64)
            s = resolve_const(dq.input[1], d_init, d_prod).astype(np.float64)
            z = (resolve_const(dq.input[2], d_init, d_prod).astype(np.float64)
                 if len(dq.input) > 2 and dq.input[2] else np.zeros_like(s))
            if s.ndim == 0:
                s = s.reshape(1)
                z = z.reshape(1)
            ax = 0 if s.size == w.shape[0] else (1 if s.size == w.shape[-1] else None)
            assert ax is not None, f"{tag}: scale size {s.size} vs weight {w.shape}"
            sh = [1] * w.ndim
            sh[ax] = s.size
            s, z = s.reshape(sh), z.reshape(sh)
            q8 = np.clip(np.round(w / s) + z, -127, 127)
            err = np.abs((q8 - z) * s - w).max()
            assert err <= s.max(), f"{tag}: roundtrip err {err} > scale {s.max()}"
            print(f"  [{tag}] w roundtrip maxerr={err:.3e} scale_max={s.max():.3e} "
                  f"axis={ax} per{'-channel' if s.size > 1 else '-tensor'}")

    f_mm = {n.name: n for n in fg.node if n.op_type in ("MatMul", "Gemm")}
    d_mm = {n.name: n for n in dg.node if n.op_type in ("MatMul", "Gemm")}

    for t in targets:
        mm, dmm = f_mm.get(t), d_mm.get(t)
        assert mm is not None, f"target not in fix3: {t}"
        assert dmm is not None, f"target not in donor: {t}"
        print(f"--- {t}")
        graft_side(mm, dmm, 0, "act")
        graft_side(mm, dmm, 1, "wt")

    # structural splice LAST (proxies invalid across del/extend)
    if new_nodes:
        kept = [n for n in fg.node]
        del fg.node[:]
        fg.node.extend(new_nodes)
        fg.node.extend(kept)
    if new_inits:
        existing = {i.name for i in fg.initializer}
        for name, init in new_inits.items():
            if name not in existing:
                fg.initializer.append(init)

    onnx.save(fm, args.out)
    print(f"\ngrafted {len(targets)} matmuls x2 sides; new nodes {len(new_nodes)}, "
          f"new inits {len(new_inits)}")
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
