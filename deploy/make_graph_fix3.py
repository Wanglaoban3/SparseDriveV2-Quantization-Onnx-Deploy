# -*- coding: utf-8 -*-
"""Graph surgery fix2 -> fix3: fold the pre-DFA Softmax chain into the plugin.

For every DeformableAggregation node the w input (in4) is produced by
  Reshape_5 <- Transpose_2 <- Reshape_3 <- Cast_6 <- Softmax <- Reshape_2 (logits)
fix3 rewires the plugin input to the logits tensor (Reshape_2 output), deletes
the 5-node chain (Reshape_5/Transpose_2/Reshape_3/Cast_6/Softmax — Reshape_2
STAYS, it produces the plugin input), and renames the op to
"DeformableAggregationPg" (new plugin type in the same libdfa_sd.so; the v1
type stays for old engines).

Board-observed geometry (fix2 graph):
  layers.0/p_deform: A=1024, pts=500,  logits (1,1024,6000,8)
  layers.1/p_deform: A=128,  pts=4000, logits (1,128,48000,8)
  layers.1/t_deform: A=400,  pts=1280, logits (1,400,15360,8)
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import onnx
from onnx import numpy_helper
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
SRC = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix2.onnx")
DST = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3.onnx")

m = onnx.load(SRC)
g = m.graph

prod = {}   # tensor name -> node
uses = {}   # tensor name -> [nodes]
for nd in g.node:
    for o in nd.output:
        if o: prod[o] = nd
    for inp in nd.input:
        if inp: uses.setdefault(inp, []).append(nd)

DELETE = ["Reshape_5", "Transpose_2", "Reshape_3", "Cast_6", "Softmax"]

delnames = set()
dfa_names = [n.name for n in g.node if n.op_type == "DeformableAggregation"]
for nm0 in dfa_names:
    dfa = next(n for n in g.node if n.name == nm0)   # fresh proxy each time
    pfx = dfa.name.rsplit("/", 1)[0]
    print(f"== {dfa.name}")
    w_in = dfa.input[4]
    assert prod[w_in].op_type == "Reshape" and prod[w_in].name == pfx + "/Reshape_5", \
        f"unexpected w producer {prod[w_in].name}"
    # single-consumer verification along the chain (pre-rewire)
    for nm in DELETE:
        full = pfx + "/" + nm
        nd = prod.get(full + "_output_0")
        assert nd is not None and nd.name == full, f"missing node {full}"
        consumers = uses.get(full + "_output_0", [])
        assert len(consumers) == 1, f"{full}_output_0 has {len(consumers)} consumers"
    # logits producer must be Reshape_2, still consumed only by Softmax
    r2_name = pfx + "/Reshape_2"
    r2 = prod[r2_name + "_output_0"]
    assert len(uses[r2_name + "_output_0"]) == 1
    tgt2 = None
    for inp in r2.input[1:]:
        if inp in prod and prod[inp].op_type == "Constant":
            tgt2 = numpy_helper.to_array(prod[inp].attribute[0].t).tolist()
        elif inp in {i.name for i in g.initializer}:
            tgt2 = numpy_helper.to_array(
                next(i for i in g.initializer if i.name == inp)).tolist()
    print(f"   Reshape_2 target = {tgt2}")
    # rewire now; structural deletion deferred until all mutations are done
    # (protobuf proxies go stale across del g.node[:] / extend)
    dfa.input[4] = r2_name + "_output_0"
    dfa.op_type = "DeformableAggregationPg"
    delnames.update(pfx + "/" + nm for nm in DELETE)
    print(f"   rewired in4 -> {r2_name}_output_0, op -> {dfa.op_type}")

before = len(g.node)
kept = [n for n in g.node if n.name not in delnames]
del g.node[:]
g.node.extend(kept)
removed = before - len(g.node)

assert removed == 15, f"expected to remove 15 nodes, removed {removed}"
n_pg = len([n for n in g.node if n.op_type == "DeformableAggregationPg"])
n_v1 = len([n for n in g.node if n.op_type == "DeformableAggregation"])
print(f"\nPg nodes: {n_pg}, v1 nodes left: {n_v1}")

known = {i.name for i in g.initializer} | {i.name for i in g.input} | \
        {o.name for o in g.output}
for nd in g.node:
    for o in nd.output:
        known.add(o)
missing = []
for nd in g.node:
    for inp in nd.input:
        if inp and inp not in known:
            missing.append((nd.name, inp))
assert not missing, f"dangling inputs: {missing[:5]}"
print("structural check OK (no dangling inputs)")
onnx.save(m, DST)
print("saved:", DST)
