# -*- coding: utf-8 -*-
"""Dump donor QDQ upstream/downstream structure for target MatMuls and check
name alignment with fix3. Read-only inspection."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import onnx
from onnx import numpy_helper
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DONOR = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_int8_qdq.onnx")
FIX3 = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3.onnx")

TARGETS = [
    "/model/_trajectory_head/decoder/layers.0/p_deform_model/weights_fc/MatMul",
    "/model/_trajectory_head/decoder/layers.1/p_deform_model/weights_fc/MatMul",
]

dm = onnx.load(DONOR)
fm = onnx.load(FIX3)
dg, fg = dm.graph, fm.graph
d_init = {i.name: i for i in dg.initializer}
f_init = {i.name: i for i in fg.initializer}
d_node = {}
for n in dg.node:
    d_node[n.name] = n
    for o in n.output:
        d_node.setdefault("out:" + o, n)
f_node = {}
for n in fg.node:
    f_node[n.name] = n
    for o in n.output:
        f_node.setdefault("out:" + o, n)

# producer lookup by tensor output name (separate maps)
d_prod = {}
for n in dg.node:
    for o in n.output:
        d_prod[o] = n
f_prod = {}
for n in fg.node:
    for o in n.output:
        f_prod[o] = n

PASS = ("Cast", "Transpose", "Reshape", "Identity", "Squeeze", "Unsqueeze")


def chain_back(g_prod, inits, tensor, depth=0, max_depth=8):
    """Human-readable chain from an initializer to `tensor`."""
    n = g_prod.get(tensor)
    if n is None:
        kind = "INIT" if tensor in inits else "EXTERNAL"
        sh = list(inits[tensor].dims) if tensor in inits else "?"
        return ["  " * depth + f"{kind} {tensor} {sh}"]
    ins = " | ".join(str(list(inits[i].dims) if i in inits else i) for i in n.input)
    lines = ["  " * depth + f"{n.op_type} {n.name}  in=[{ins}]"]
    if n.op_type in PASS and depth < max_depth:
        lines += chain_back(g_prod, inits, n.input[0], depth + 1, max_depth)
    return lines


print("=== donor weight-side chains ===")
for t in TARGETS:
    n = d_node[t]
    print(f"\n--- {t}")
    print("  input0 (act):", n.input[0], " in fix3?", n.input[0] in f_node or n.input[0] in f_init)
    print("  weight chain:")
    for ln in chain_back(d_prod, d_init, n.input[1]):
        print(ln)
    # QDQ on weight?
    wsrc = n.input[1]
    hops = 0
    cur = wsrc
    while hops < 8:
        p = d_prod.get(cur)
        if p is None:
            break
        if p.op_type in ("QuantizeLinear", "DequantizeLinear"):
            s = p.input[1] if len(p.input) > 1 else None
            z = p.input[2] if len(p.input) > 2 else None
            print(f"    {p.op_type}: scale={s} {list(d_init[s].dims) if s in d_init else '?'}"
                  f" zp={z} {list(d_init[z].dims) if z in d_init and z else ''}")
        if p.op_type in PASS:
            cur = p.input[0]
            hops += 1
        else:
            break
    # activation side upstream 1 hop
    ap = d_prod.get(n.input[0])
    print("  act producer:", ap.op_type if ap else "EXTERNAL/INIT", ap.name if ap else "")
    # output side downstream
    for o in n.output:
        consumers = [m for m in dg.node if o in m.input]
        print(f"  out {o} consumers: {[c.op_type + ':' + c.name for c in consumers]}")

print("\n=== fix3 side ===")
for t in TARGETS:
    n = f_node[t]
    print(f"\n--- {t}")
    print("  input0:", n.input[0], "producer:",
          f_prod.get(n.input[0]).op_type if f_prod.get(n.input[0]) else "EXTERNAL/INIT")
    for ln in chain_back(f_prod, f_init, n.input[1]):
        print("  wchain:", ln)
    wsrc = n.input[1]
    p = f_node.get(wsrc)
    hops = 0
    cur = wsrc
    while hops < 8:
        p = f_prod.get(cur)
        if p is None:
            break
        if p.op_type in PASS:
            cur = p.input[0]
            hops += 1
        else:
            break
    print("  weight root:", cur, "INIT?", cur in f_init,
          list(f_init[cur].dims) if cur in f_init else "?")
    # is the same weight initializer present in donor under same name?
    print("  same-name init in donor:", cur in d_init,
          list(d_init[cur].dims) if cur in d_init else "?")
    for o in n.output:
        consumers = [m for m in fg.node if o in m.input]
        print(f"  out {o} consumers: {[c.op_type + ':' + c.name for c in consumers]}")
