# -*- coding: utf-8 -*-
"""Strip QDQ pairs for given scopes (default: backbone + trajectory_head = ALL)
-> fully-float graph (build with --fp16 only, no --int8)."""
import os
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import onnx

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
ART = os.path.join(_ROOT, "deploy/artifacts")
SRC = os.path.join(ART, "sparsedrive_int8_qdq_folded_pre_rewrite.onnx")
DST = os.path.join(ART, "sparsedrive_fp16_graph.onnx")
SCOPES = ["_backbone", "_trajectory_head"]

m = onnx.load(SRC)
g = m.graph
consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)
producer = {o: n for n in g.node for o in n.output}


def in_scope(name):
    return any(s in name for s in SCOPES)


dqs = [n for n in g.node if n.op_type == "DequantizeLinear" and in_scope(n.name)]
dq_removed = q_removed = rewired = 0
for n in dqs:
    src = n.input[0]
    p = producer.get(src)
    paired_q = None
    if p is not None and p.op_type == "QuantizeLinear":
        paired_q = p
        src = p.input[0]
    out = n.output[0]
    for c in consumers.get(out, []):
        for k, x in enumerate(c.input):
            if x == out:
                c.input[k] = src
                rewired += 1
    g.node.remove(n)
    dq_removed += 1
    if paired_q is not None:
        if not any(o in c.input for c in g.node for o in paired_q.output):
            g.node.remove(paired_q)
            q_removed += 1

nq = sum(1 for n in g.node if n.op_type == "QuantizeLinear")
ndq = sum(1 for n in g.node if n.op_type == "DequantizeLinear")
print("nodes=%d Q=%d DQ=%d (removed dq=%d q=%d rewired=%d)" % (len(g.node), nq, ndq, dq_removed, q_removed, rewired))
assert nq == 0 and ndq == 0, "graph still has QDQ — scope list incomplete"
onnx.save(m, DST)
import hashlib
print("saved -> %s md5=%s" % (DST, hashlib.md5(open(DST, "rb").read()).hexdigest()))
print("FLOAT_GRAPH_DONE")
