# -*- coding: utf-8 -*-
"""Inventory MatMul/Gemm nodes in fix3 graph; match against donor int8_qdq graph.

Outputs (stdout + deploy/artifacts/reports/myelin_matmul_inventory.md):
  - per-MatMul: name, MxKxN, weight bytes, FLOPs, region tag, donor QDQ presence
    (weight scale shape/axis, act scale), donor match by node name.
Region tags follow the 4 myelin ForeignNode regions seen in prof_real_pg.log.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import onnx
from onnx import numpy_helper, TensorProto
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
FIX3 = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3.onnx")
DONOR = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_int8_qdq.onnx")
OUT = os.path.join(_ROOT, "deploy/artifacts/reports/myelin_matmul_inventory.md")


def region_of(name):
    n = name
    if "camera_encoder" in n:
        return "cam_enc"
    if "p_deform_model" in n:
        return "p_deform(l0)" if "layers.0" in n else "p_deform(l1)"
    if "v_img_attention" in n:
        return "v_img_attn"
    if "t_deform_model" in n:
        return "t_deform"
    if "_trajectory_head" in n or "trajectory_head" in n:
        # heads region = layers.1 Reshape_10/Flatten .. Reshape_11/12 (heads 融合区)
        return "heads"
    return "other"


def dims_of(g, tensor_name, inits):
    if tensor_name in inits:
        return list(inits[tensor_name].dims)
    for vi in list(g.value_info) + list(g.output) + list(g.input):
        if vi.name == tensor_name and vi.type.HasField("tensor_type"):
            tt = vi.type.tensor_type
            if tt.elem_type != 0 and len(tt.shape.dim) > 0:
                return [d.dim_value if d.HasField("dim_value") else -1 for d in tt.shape.dim]
    return None


PASS_THROUGH = ("Cast", "Transpose", "Reshape", "Identity", "Squeeze", "Unsqueeze")


def find_dq_upstream(nodes_by_output, tensor_name, dq_by_output, inits, max_hops=5):
    """Walk upstream through pass-through ops; return first DQ node feeding this
    tensor (directly or through pass-throughs), else None."""
    cur = tensor_name
    for _ in range(max_hops):
        if cur in dq_by_output:
            return dq_by_output[cur]
        prod = nodes_by_output.get(cur)
        if prod is None:
            return None
        if prod.op_type not in PASS_THROUGH or not prod.input:
            return None
        cur = prod.input[0]
    return None


def matmul_rows(path, tag):
    m = onnx.load(path)
    m = onnx.shape_inference.infer_shapes(m)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    nodes_by_output = {}
    for n in g.node:
        for o in n.output:
            nodes_by_output[o] = n
    q_by_input = {}      # Q.input[0] -> Q node
    dq_by_output = {}    # DQ.output[0] -> DQ node
    for n in g.node:
        if n.op_type == "QuantizeLinear":
            q_by_input[n.input[0]] = n
        elif n.op_type == "DequantizeLinear":
            dq_by_output[n.output[0]] = n
    rows = []
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        wname = n.input[1] if len(n.input) > 1 else None
        wshape = dims_of(g, wname, inits) if wname else None
        yshape = dims_of(g, n.output[0], inits)
        xshape = dims_of(g, n.input[0], inits)
        wdq = find_dq_upstream(nodes_by_output, wname, dq_by_output, inits) if wname else None
        w_scale = None
        if wdq is not None:
            sname = wdq.input[1]
            w_scale = (sname, list(inits[sname].dims) if sname in inits else "?")
        xdq = find_dq_upstream(nodes_by_output, n.input[0], dq_by_output, inits) \
            if n.input[0] not in inits else None
        rows.append(dict(tag=tag, name=n.name, op=n.op_type, x=xshape, w=wshape, y=yshape,
                         wq=(wdq is not None), w_scale=w_scale, xq=(xdq is not None)))
    return rows, m


fix3_rows, fix3m = matmul_rows(FIX3, "fix3")
donor_rows, _ = matmul_rows(DONOR, "donor")
donor_by_name = {r["name"]: r for r in donor_rows}

lines = ["# myelin MatMul inventory (fix3 vs donor int8_qdq)\n"]
seen = []
for r in fix3_rows:
    y = r["y"] or []
    x = r["x"] or []
    w = r["w"] or []
    MN = 1
    for d_ in y:
        MN *= max(d_, 1)
    K = x[-1] if x else (w[-1] if w else 0)
    wbytes = 4 if not w else 4
    for d_ in w:
        wbytes *= max(d_, 1)
    flop = 2.0 * MN * max(K, 0)
    d = donor_by_name.get(r["name"])
    r2 = dict(r)
    r2["donor"] = "Y" if d and d["wq"] else ("xq_only" if d and d["xq"] else "-")
    r2["donor_wscale"] = d["w_scale"] if d else None
    r2["donor_xq"] = d["xq"] if d else None
    r2["MB"] = wbytes / 1e6
    r2["MF"] = flop / 1e6
    r2["region"] = region_of(r["name"])
    seen.append(r2)

seen.sort(key=lambda r: (-r["MF"], -r["MB"]))
lines.append("| region | name | x | w | y | wMB | MFLOP | donorWQ | donorXQ | donor_wscale |")
lines.append("|---|---|---|---|---|---:|---:|---|---|---|")
for r in seen:
    lines.append("| {} | {} | {} | {} | {} | {:.1f} | {:.0f} | {} | {} | {} |".format(
        r["region"], r["name"], r["x"], r["w"], r["y"], r["MB"], r["MF"],
        r["donor"], r["donor_xq"], r["donor_wscale"]))

# region subtotals (weight MB + FLOPs) for myelin regions only
lines.append("\n## region subtotals (fix3 MatMul only)\n")
lines.append("| region | nMM | wMB | GFLOP |")
lines.append("|---|---:|---:|---:|")
agg = {}
for r in seen:
    a = agg.setdefault(r["region"], [0, 0.0, 0.0])
    a[0] += 1
    a[1] += r["MB"]
    a[2] += r["MF"] / 1000.0
for k in sorted(agg, key=lambda k: -agg[k][2]):
    lines.append("| {} | {} | {:.1f} | {:.2f} |".format(k, *agg[k]))

text = "\n".join(lines) + "\n"
print(text)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("saved ->", OUT)
