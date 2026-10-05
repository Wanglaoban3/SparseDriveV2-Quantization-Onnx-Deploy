# -*- coding: utf-8 -*-
"""ReduceSum_1 融合损失侦察：axes/dims/上下游链/流量估算。

用法: python inspect_reducesum.py [onnx 路径，默认 fix3_mha]
"""
import io
import sys

import onnx
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
SRC = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3_mha.onnx")

M = onnx.load(SRC)
g = M.graph

# dims：value_info + io + initializers（不跑 shape_inference：插件 op 会挡路）
dims = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    if vi.type.tensor_type.HasField("shape"):
        dd = []
        for d in vi.type.tensor_type.shape.dim:
            dd.append(d.dim_param or d.dim_value)
        dims[vi.name] = (vi.type.tensor_type.elem_type, dd)

producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n
consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

def short(nm):
    return nm.replace("/model/_trajectory_head/decoder/", "").replace(
        "/model/_trajectory_head/", "").replace("/model/", "")

def show_tensor(nm):
    et, dd = dims.get(nm, (None, None))
    e = {1: "f32", 10: "f16", 7: "i64", 6: "i32"}.get(et, et)
    sz = 1
    known = dd is not None and all(isinstance(x, int) for x in dd) and dd
    if known:
        for x in dd:
            sz *= x
    b = sz * {1: 4, 10: 2, 6: 4, 7: 8}.get(et, 0) if known else 0
    return f"{nm} [{e}:{','.join(map(str, dd)) if dd else '?'}]" + (f" ({b/1e6:.2f}MB)" if b else "")

hits = [n for n in g.node if "ReduceSum" in n.name]
print(f"graph: {len(g.node)} nodes; ReduceSum nodes: {len(hits)}; "
      f"value_info coverage: {len(g.value_info)}\n")
for n in hits:
    print(f"=== {n.name}")
    for a in n.attribute:
        v = onnx.helper.get_attribute_value(a)
        print(f"    attr {a.name} = {list(v) if isinstance(v, list) else v}")
    # 上游链：最多 6 跳，遇到分支 >1 停
    cur = n.input[0]
    for hop in range(6):
        print(f"    up[{hop}]: {show_tensor(cur)}")
        p = producer.get(cur)
        if p is None:
            print(f"      (producer: initializer/graph input)")
            break
        ins = ", ".join(show_tensor(i) for i in p.input)
        print(f"      <- {short(p.name)} [{p.op_type}] ({ins})")
        if len(consumers.get(cur, [])) > 1:
            print(f"      (fanout {len(consumers[cur])}, stop up-walk)")
            break
        cur = p.input[0] if p.input else None
        if cur is None:
            break
    # 下游
    for c in consumers.get(n.output[0], []):
        outs = ", ".join(show_tensor(o) for o in c.output)
        print(f"    down: {short(c.name)} [{c.op_type}] -> {outs}")
    print()
