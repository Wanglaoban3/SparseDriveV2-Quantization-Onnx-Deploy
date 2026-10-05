# -*- coding: utf-8 -*-
"""fix3_mha -> fix3_sumf：DFA 插件输出直连 output_proj（anchor 求和已入插件）。

每块删除链（侦察 inspect_reducesum/_tmp_rs_probe 定案）：
  DeformableAggregationPg[1,A*pts,256] -> Reshape_6 [1,A,pts,256]
  -> Cast(->f32) -> ReduceSum_1(axes=[2], keepdims=0) -> output_proj/MatMul
插件 getOutputDimensions 已改为 [bs, A, C]（v5 gather 内部求和），图上把
中间三个节点删掉、MatMul 直接吃插件输出，262MB fp16 写+读往返消失。
"""
import io
import sys

import onnx
import numpy as np
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
SRC = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3_mha.onnx")
DST = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3_sumf.onnx")

BLKS = [
    "layers.0/p_deform_model",
    "layers.1/p_deform_model",
    "layers.1/t_deform_model",
]
PRE = "/model/_trajectory_head/decoder/"

M = onnx.load(SRC)
g = M.graph
producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n
consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n.name)


def const_of(tname, depth=0):
    if depth > 4:
        return None
    for i in g.initializer:
        if i.name == tname:
            return onnx.numpy_helper.to_array(i)
    p = producer.get(tname)
    if p is None:
        return None
    if p.op_type == "Constant":
        for a in p.attribute:
            if a.name == "value":
                return onnx.numpy_helper.to_array(a.t)
    if p.op_type == "Concat":
        parts = [const_of(i, depth + 1) for i in p.input]
        if all(x is not None for x in parts):
            return np.concatenate([np.asarray(x).ravel() for x in parts])
    if p.op_type == "Transpose":
        return const_of(p.input[0], depth + 1)
    return None


def sole_consumer(tname, want):
    cs = consumers.get(tname, [])
    assert len(cs) == 1, f"{tname}: consumers {cs} != [{want}]"
    assert cs[0] == want, f"{tname}: consumer {cs[0]} != {want}"


def consumer_node(tname):
    cs = consumers.get(tname, [])
    assert len(cs) == 1, f"{tname}: consumers {cs}"
    return next(n for n in g.node if n.name == cs[0])


n0 = len(g.node)
deleted = set()
new_vi = []
for blk in BLKS:
    pg = next(n for n in g.node if n.name == PRE + blk + "/DeformableAggregation")
    pout = pg.output[0]
    r6n = consumer_node(pout)
    assert r6n.name == PRE + blk + "/Reshape_6" and r6n.op_type == "Reshape", r6n.name
    tgt = const_of(r6n.input[1])
    assert tgt is not None and len(tgt) == 4 and tgt[0] == 1 and tgt[3] == 256, tgt
    A, pts = int(tgt[1]), int(tgt[2])
    cst = consumer_node(r6n.output[0])
    assert cst.op_type == "Cast" and cst.name.startswith(PRE + blk + "/Cast"), cst.name
    rs1n = consumer_node(cst.output[0])
    assert rs1n.name == PRE + blk + "/ReduceSum_1" and rs1n.op_type == "ReduceSum", rs1n.name
    axes = const_of(rs1n.input[1])
    assert axes is not None and list(axes) == [2], axes
    kd = [a.i for a in rs1n.attribute if a.name == "keepdims"]
    assert kd == [0], kd
    mmn = consumer_node(rs1n.output[0])
    assert mmn.name == PRE + blk + "/output_proj/MatMul" and mmn.op_type == "MatMul", mmn.name
    assert mmn.input[0] == rs1n.output[0]
    w = const_of(mmn.input[1])
    wshape = None if w is None else w.shape
    wet = None
    for i in g.initializer:
        if i.name == mmn.input[1]:
            wet = i.data_type
    print(f"{blk}: A={A} pts={pts} -> out [1,{A},256]; w={wshape} et={wet}")
    assert wshape == (256, 256), wshape
    # 重接 + 删除
    before = list(mmn.input)
    mmn.input[0] = pout
    deleted |= {r6n.name, cst.name, rs1n.name}
    # 插件输出 value_info 更新（fp16，[1,A,256]）
    new_vi.append((pout, A, wet if wet in (1, 10) else 10))

# 按 value_info 顺序替换插件输出形状
def _set_dims(vi, A):
    tt = vi.type.tensor_type
    tt.elem_type = 10
    del tt.shape.dim[:]
    for d in (1, A, 256):
        dd = tt.shape.dim.add()
        dd.dim_value = d

vis = {v.name: v for v in g.value_info}
for pout, A, _ in new_vi:
    assert pout in vis, pout
    _set_dims(vis[pout], A)

# 死代码清除（保护插件与图输出，到不动点）
PROTECTED = {"DeformableAggregationPg", "FusedMHA", "DeformableAggregation"}
graph_outputs = {o.name for o in g.output}
changed = True
dead = 0
nodes = [n for n in g.node if n.name not in deleted]
while changed:
    changed = False
    cons = {}
    for n in nodes:
        for i in n.input:
            cons.setdefault(i, []).append(n.name)
    keep = []
    for n in nodes:
        if n.op_type in PROTECTED:
            keep.append(n)
            continue
        if all(len(cons.get(o, [])) == 0 and o not in graph_outputs
               for o in n.output):
            dead += 1
            changed = True
            continue
        keep.append(n)
    nodes = keep

del g.node[:]
g.node.extend(nodes)

onnx.save(M, DST)
print(f"saved -> {DST}")

# ---- 验证 ----
pos = {}
for i, n in enumerate(g.node):
    for o in n.output:
        pos[o] = i
bad = 0
for i, n in enumerate(g.node):
    for inp in n.input:
        if inp in pos and pos[inp] >= i:
            bad += 1
            if bad <= 8:
                print(f"VIOLATION: node[{i}] {n.name} input {inp} @{pos[inp]}")
assert bad == 0
onnx.checker.check_model(M)
print(f"nodes: {n0} -> {len(g.node)} (deleted {n0 - len(g.node)}, dead {dead})")
assert len(g.node) == n0 - len(deleted) - dead
for pout, A, _ in new_vi:
    cs = consumers.get(pout, [])
    print(f"  {pout[-60:]} -> {cs}")
print("SUMF_SURGERY_OK")
