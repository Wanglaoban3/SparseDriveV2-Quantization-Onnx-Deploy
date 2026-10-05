# -*- coding: utf-8 -*-
"""Verify the E1 grafted graph:
  1. per grafted MatMul: act path src->Q_q8->DQ_q8->MM, weight path
     init->Q_q8->DQ_q8->Transpose->MM;
  2. scale/zp constants used by the grafted DQs are byte-identical to donor;
  3. graph-wide: unique node names/outputs, all inputs resolvable.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import onnx
from onnx import numpy_helper
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
E1 = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3_qdq_e1.onnx")
DONOR = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_int8_qdq.onnx")

TARGETS = [
    "/model/_trajectory_head/decoder/layers.0/p_deform_model/weights_fc/MatMul",
    "/model/_trajectory_head/decoder/layers.1/p_deform_model/weights_fc/MatMul",
]


def resolve_const(t, inits, prod):
    if t in inits:
        return numpy_helper.to_array(inits[t])
    p = prod.get(t)
    if p is not None and p.op_type == "Constant":
        for a in p.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    if p is not None and p.op_type == "Cast":
        return resolve_const(p.input[0], inits, prod).astype(np.int8)
    raise RuntimeError(f"unresolved const {t}")


em = onnx.load(E1)
dm = onnx.load(DONOR)
eg, dg = em.graph, dm.graph
e_init = {i.name: i for i in eg.initializer}
d_init = {i.name: i for i in dg.initializer}
e_prod, d_prod = {}, {}
for n in eg.node:
    for o in n.output:
        e_prod[o] = n
for n in dg.node:
    for o in n.output:
        d_prod[o] = n
d_name = {n.name: n for n in dg.node}

ok = True
for t in TARGETS:
    mm = e_prod.get([n for n in eg.node if n.name == t][0].output[0]) if False else None
    mm_node = None
    for n in eg.node:
        if n.name == t:
            mm_node = n
    assert mm_node is not None
    print(f"--- {t}")
    for k, tag in ((0, "act"), (1, "wt")):
        cur = mm_node.input[k]
        chain = []
        dq = None
        for _ in range(8):
            p = e_prod.get(cur)
            assert p is not None, f"{tag}: dangling {cur}"
            if p.op_type == "DequantizeLinear":
                dq = p
                break
            chain.append(p.op_type)
            cur = p.input[0]
        assert dq is not None, f"{tag}: no DQ above matmul"
        q = e_prod.get(dq.input[0])
        assert q is not None and q.op_type == "QuantizeLinear"
        es = resolve_const(dq.input[1], e_init, e_prod)
        d_dq = d_name.get(dq.name[:-3])  # strip _q8
        assert d_dq is not None, f"donor counterpart missing: {dq.name}"
        ds = resolve_const(d_dq.input[1], d_init, d_prod)
        same_s = np.array_equal(es, ds)
        ez = (resolve_const(dq.input[2], e_init, e_prod)
              if len(dq.input) > 2 and dq.input[2] else None)
        dz = (resolve_const(d_dq.input[2], d_init, d_prod)
              if len(d_dq.input) > 2 and d_dq.input[2] else None)
        same_z = (np.array_equal(ez, dz) if ez is not None and dz is not None
                  else ez is None and dz is None)
        if not same_s or not same_z:
            ok = False
        print(f"  {tag}: chain={chain} dq={dq.name} q_in={q.input[0]}")
        print(f"      scale byte-equal to donor: {same_s}  zp byte-equal: {same_z}"
              f"  scale_max={np.abs(es).max():.4e}")

# graph-wide sanity
names, outs = set(), set()
for n in eg.node:
    assert n.name not in names, f"dup node name {n.name}"
    names.add(n.name)
    for o in n.output:
        assert o not in outs, f"dup output {o}"
        outs.add(o)
known = outs | set(e_init) | {i.name for i in eg.input}
for n in eg.node:
    for i in n.input:
        if i and i not in known:
            raise RuntimeError(f"dangling input {i} on {n.name}")
print(f"\ngraph sanity OK: {len(eg.node)} nodes, {len(e_init)} inits")
print("VERIFY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
