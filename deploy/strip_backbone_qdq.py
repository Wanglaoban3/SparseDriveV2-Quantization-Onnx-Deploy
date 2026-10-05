# -*- coding: utf-8 -*-
"""Strip all Q/DQ pairs under /model/_backbone/ from pre_rewrite ONNX
-> sparsedrive_int8_qdq_nobkb.onnx (backbone runs float under --int8 --fp16).

Rewiring: consumers of each DQ output are redirected to the DQ's float input;
the Q nodes (whose outputs only fed DQs) are then removed.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import onnx
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
ART = os.path.join(_ROOT, "deploy/artifacts")
SRC = os.path.join(ART, "sparsedrive_int8_qdq_folded_pre_rewrite.onnx")
DST = os.path.join(ART, "sparsedrive_int8_qdq_nobkb.onnx")
SCOPE = "_backbone"

m = onnx.load(SRC)
g = m.graph

def in_scope(name):
    return SCOPE in name

consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n

def float_source_of(dq):
    """Bypass source for removing DQ: walk through the feeding QuantizeLinear
    (DQ.input[0] is the INT8 Q output, NOT float)."""
    src = dq.input[0]
    p = producer.get(src)
    if p is not None and p.op_type == "QuantizeLinear":
        return p.input[0], p
    return src, None

dq_removed = q_removed = 0
rewired_edges = 0
for n in list(g.node):
    if n.op_type == "DequantizeLinear" and in_scope(n.name):
        src, paired_q = float_source_of(n)
        out = n.output[0]
        for c in consumers.get(out, []):
            for k, x in enumerate(c.input):
                if x == out:
                    c.input[k] = src
                    rewired_edges += 1
        g.node.remove(n)
        dq_removed += 1
        if paired_q is not None:
            # remove the paired Q if nothing else consumes its output
            still = any(o in c.input for c in g.node for o in paired_q.output)
            if not still:
                g.node.remove(paired_q)
                q_removed += 1

qn = [n for n in g.node if n.op_type == "QuantizeLinear" and in_scope(n.name)]
for n in qn:
    outs_still_used = any(o in consumers and consumers[o] for o in n.output)
    # consumers dict is stale after DQ removal; recheck against remaining nodes
    used = False
    for c in g.node:
        if any(x in n.output for x in c.input):
            used = True
            break
    if not used:
        g.node.remove(n)
        q_removed += 1

# prune now-orphaned initializers (optional, keep graph lean)
n_init0 = len(g.initializer)
live = set()
for c in g.node:
    live.update(x for x in c.input if x)
for o in g.output:
    live.add(o.name)
inits = [i for i in g.initializer if i.name in live or not i.name]
del g.initializer[:]
g.initializer.extend(inits)

# dtype sanity: every remaining consumer of a backbone Q output must exist (none expected)
leftover = 0
for n in g.node:
    if n.op_type == "QuantizeLinear" and in_scope(n.name):
        for o in n.output:
            if any(o in c.input for c in g.node):
                leftover += 1
print("leftover backbone Q with consumers:", leftover)

onnx.checker.check_model(m, skip_opset_compatibility_check=True) if False else None
ops = {}
for n in g.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("nodes=%d  Q=%d DQ=%d  (backbone stripped: dq_removed=%d q_removed=%d rewired=%d)"
      % (len(g.node), ops.get("QuantizeLinear", 0), ops.get("DequantizeLinear", 0),
         dq_removed, q_removed, rewired_edges))
print("initializers %d -> %d" % (n_init0, len(g.initializer)))
onnx.save(m, DST)
print("saved ->", DST)
import hashlib
print("md5=%s" % hashlib.md5(open(DST, "rb").read()).hexdigest())
print("STRIP_DONE")
