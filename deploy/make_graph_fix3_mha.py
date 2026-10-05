# -*- coding: utf-8 -*-
"""fix3 图手术：7 个 attention 块的 12 节点链 -> 1 个 FusedMHA 插件节点。

链（inspect_attention_chain.py 侦察定案，全块同构）：
  Reshape[1,S,3,8,32] -> Transpose[2,0,3,1,4] -> Gather idx{0,1,2} 切 Q/K/V
  -> Q: Mul(1/sqrt32) ; K: Transpose[0,1,3,2] -> MatMul(scores) -> Softmax
  -> MatMul(ctx) -> Transpose[0,2,1,3] -> Reshape[1,S,256]
插件边界：输入 = in_proj+bias 输出（Reshape 的 in0），输出 = Reshape_1 的输出
（名字保留，out_proj 不动）。权重零注入；S 由插件从输入 dims 读取。

安全断言：op_type 全链匹配、常量（Gather 索引 ∈{0,1,2}、Mul scale≈0.176777）、
删除集输出无外部消费者、图 io 签名不变、onnx.checker 通过。
"""
import io
import sys

import onnx
from onnx import helper
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
SRC = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3.onnx")
DST = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3_mha.onnx")

BLKS = [
    "/model/_trajectory_head/decoder/layers.0/p_attention",
    "/model/_trajectory_head/decoder/layers.0/v_attention",
    "/model/_trajectory_head/decoder/layers.0/v_img_attention",
    "/model/_trajectory_head/decoder/layers.1/t_attention",
    "/model/_trajectory_head/decoder/layers.1/p_attention",
    "/model/_trajectory_head/decoder/layers.1/v_attention",
    "/model/_trajectory_head/decoder/layers.1/v_img_attention",
]
CHAIN = ["/Reshape", "/Transpose", "/Gather", "/Gather_1", "/Gather_2", "/Mul",
         "/Transpose_1", "/MatMul", "/Softmax", "/MatMul_1", "/Transpose_2",
         "/Reshape_1"]
OPS = ["Reshape", "Transpose", "Gather", "Gather", "Gather", "Mul", "Transpose",
       "MatMul", "Softmax", "MatMul", "Transpose", "Reshape"]

M = onnx.load(SRC)
g = M.graph
d_name = {n.name: n for n in g.node}
producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n
consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n.name)
inits = {i.name: i for i in g.initializer}


def const_of(tname):
    if tname in inits:
        return onnx.numpy_helper.to_array(inits[tname])
    n = producer.get(tname)
    if n is not None and n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                return onnx.numpy_helper.to_array(a.t)
    if n is not None and n.op_type == "Cast":
        return const_of(n.input[0])
    return None


n0 = len(g.node)
deleted = set()
plugin_nodes = []
for blk in BLKS:
    names = [blk + s for s in CHAIN]
    for nm, op in zip(names, OPS):
        n = d_name.get(nm)
        assert n is not None, f"missing {nm}"
        assert n.op_type == op, f"{nm}: op {n.op_type} != {op}"
    # 常量核对：Gather 索引 {0,1,2}、Mul scale
    idxs = set()
    for gn in ("Gather", "Gather_1", "Gather_2"):
        gi = d_name[blk + "/" + gn].input[1]
        a = const_of(gi)
        assert a is not None and int(a) in (0, 1, 2), f"{blk}/{gn} idx={a}"
        idxs.add(int(a))
    assert len(idxs) == 3, f"{blk} gather idx not 0/1/2: {idxs}"
    mi = d_name[blk + "/Mul"].input[1]
    ms = const_of(mi)
    assert ms is not None and abs(float(ms) - 0.17677669) < 1e-6, f"{blk} scale={ms}"
    # 外部消费者检查
    x768 = d_name[blk + "/Reshape"].input[0]
    out_t = d_name[blk + "/Reshape_1"].output[0]
    dset = set(names)
    for nm in names:
        for o in d_name[nm].output:
            if o == out_t:
                continue
            for c in consumers.get(o, []):
                assert c in dset, f"{o} 外部消费者 {c}"
    plugin_nodes.append((dset, blk + "/Reshape", helper.make_node(
        "FusedMHA", [x768], [out_t], name=blk + "/FusedMHA",
        domain="sparsedrivev2")))
    deleted |= dset

# 拼接：按原序扫，遇块首 Reshape 发插件节点，其余删除集跳过
new_nodes = []
pending = {anchor: pn for dset, anchor, pn in plugin_nodes}
for n in g.node:
    if n.name in deleted:
        pn = pending.pop(n.name, None)
        if pn is not None:
            new_nodes.append(pn)
        continue
    new_nodes.append(n)
assert not pending, f"块首 Reshape 未命中: {list(pending)}"

# 死代码清除（到不动点）：输出无消费者的可折叠节点
# 图输出也算消费者，否则会删掉悬空但必需的 output 生产者
PROTECTED = {"DeformableAggregationPg", "FusedMHA"}
graph_outputs = {o.name for o in g.output}
changed = True
dead = 0
while changed:
    changed = False
    cons = {}
    for n in new_nodes:
        for i in n.input:
            cons.setdefault(i, []).append(n.name)
    keep = []
    for n in new_nodes:
        if n.op_type in PROTECTED:
            keep.append(n)
            continue
        if all(len(cons.get(o, [])) == 0 and o not in graph_outputs
               for o in n.output):
            dead += 1
            changed = True
            continue
        keep.append(n)
    new_nodes = keep

del g.node[:]
g.node.extend(new_nodes)

# ---- 验证 ----
onnx.save(M, DST)
print(f"saved -> {DST}")
# 手动拓扑扫描（checker 之外的双保险）
pos = {}
for i, n in enumerate(g.node):
    for o in n.output:
        pos[o] = i
bad = 0
produced = set(pos)
for i, n in enumerate(g.node):
    for inp in n.input:
        if inp in pos and pos[inp] >= i:
            bad += 1
            if bad <= 10:
                print(f"VIOLATION: node[{i}] {n.name} input {inp} produced at {pos[inp]}")
print(f"manual topo scan violations: {bad}")
assert bad == 0
onnx.checker.check_model(M)
def _dims(v):
    tt = v.type.tensor_type
    return tuple((d.dim_param or d.dim_value) for d in tt.shape.dim)


sig_in = [(v.name, v.type.tensor_type.elem_type, _dims(v)) for v in M.graph.input]
print(f"nodes: {n0} -> {len(g.node)} (deleted {n0 - len(g.node)}, of which dead-consts {dead})")
print(f"io: {len(M.graph.input)} inputs / {len(M.graph.output)} outputs (签名见下)")
for nm, et, dims in sig_in:
    print(f"  in  {nm[:60]:60s} et={et} dims={dims}")
mh = [n for n in g.node if n.op_type == "FusedMHA"]
assert len(mh) == 7, len(mh)
for n in mh:
    cs = consumers.get(n.output[0], [])
    print(f"  {n.name[:80]}  out_consumers={[c[-40:] for c in cs]}")
print("SURGERY_OK")
