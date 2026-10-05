# -*- coding: utf-8 -*-
"""侦察 fix3 ONNX 的 attention 块完整节点链（in_proj -> BMM -> softmax -> BMM -> out_proj）。"""
import io
import sys

import onnx
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
M = onnx.load(os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3.onnx"))
g = M.graph
init = {i.name: i for i in g.initializer}
vi = {v.name: v for v in list(g.input) + list(g.output)}

d_prod = {}
consumers = {}
d_name = {}
for n in g.node:
    d_name[n.name] = n
    for o in n.output:
        d_prod[o] = n
    for i in n.input:
        consumers.setdefault(i, []).append(n)

ANCHORS = [
    "/model/_trajectory_head/decoder/layers.0/p_attention/MatMul",
    "/model/_trajectory_head/decoder/layers.1/t_attention/MatMul",
    "/model/_trajectory_head/decoder/layers.0/v_img_attention/MatMul",
]

for anchor in ANCHORS:
    n = d_name[anchor]
    blk = anchor.rsplit("/MatMul", 1)[0]
    print("=" * 110)
    print(f"ANCHOR scores BMM: {n.name}")
    print(f"  inQ: {n.input[0][:80]}\n  inK: {n.input[1][:80]}")

    # ---- forward: scores -> ... -> out_proj（只走本块名字前缀的 consumer）----
    cur = n.output[0]
    for hop in range(14):
        cands = [c for c in consumers.get(cur, []) if c.name.startswith(blk)]
        if not cands:
            print(f"  fwd[{hop}] <{cur[:80]}> -> 非本块 consumer: "
                  f"{[c.name[:60] for c in consumers.get(cur, [])]}")
            break
        c = cands[0]
        print(f"  fwd[{hop}] [{c.op_type}] {c.name[len(blk):]}  "
              f"in={[i.rsplit('/', 1)[-1][:40] for i in c.input]}  "
              f"out={c.output[0].rsplit('/', 1)[-1][:40]}")
        cur = c.output[0]
        if "out_proj" in c.name:
            break

    # ---- backward: K 侧 Transpose 链与 Q 侧 Mul(scale) 链，线性主链 ----
    for side, t in (("Q", n.input[0]), ("K", n.input[1])):
        print(f"  ---- backward {side}: ----")
        cur = t
        for hop in range(10):
            p = d_prod.get(cur)
            if p is None:
                src = init.get(cur) or vi.get(cur)
                dims = list(src.dims) if src is not None and hasattr(src, "dims") else "?"
                print(f"    [{hop}] <{cur[-70:]}> initializer/graph dims={dims}")
                break
            print(f"    [{hop}] [{p.op_type}] {p.name[len(blk):] if p.name.startswith(blk) else p.name[-60:]}  "
                  f"in={[i.rsplit('/', 1)[-1][:40] for i in p.input]}")
            cur = p.input[0]
    print()
