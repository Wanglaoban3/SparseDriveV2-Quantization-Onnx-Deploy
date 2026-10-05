# -*- coding: utf-8 -*-
"""钉死 attention 布局常量：Reshape shape / Transpose perm / Gather idx / scale 值。"""
import io
import sys

import numpy as np
import onnx
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
M = onnx.load(os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3.onnx"))
g = M.graph
init = {i.name: i for i in g.initializer}
d_prod = {}
for n in g.node:
    for o in n.output:
        d_prod[o] = n


def const_of(tname):
    """解析 Constant/Cast/initializer 链 -> numpy。"""
    n = d_prod.get(tname)
    if n is not None and n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                return onnx.numpy_helper.to_array(a.t)
    if n is not None and n.op_type == "Cast":
        return const_of(n.input[0])
    if tname in init:
        return onnx.numpy_helper.to_array(init[tname])
    return None


blk = "/model/_trajectory_head/decoder/layers.0/p_attention"
names = {
    "Reshape(shape)": f"{blk}/Concat_output_0",
    "Reshape_1(shape)": f"{blk}/Concat_1_output_0",
    "Gather(Q idx)": f"{blk}/Constant_output_0",
    "Gather_1(K idx)": f"{blk}/Constant_2_output_0",
    "Gather_2(V idx)": f"{blk}/Constant_1_output_0",
    "Mul(scale Q)": f"{blk}/Constant_4_output_0",
}
for label, t in names.items():
    a = const_of(t)
    print(f"{label:20s} = {a if a is not None else '<missing>'}")

# 三条 Transpose 的 perm
for tn in ("Transpose", "Transpose_1", "Transpose_2"):
    n = d_prod[f"{blk}/{tn}_output_0"]
    perm = next(a.ints for a in n.attribute if a.name == "perm")
    print(f"Transpose {tn:12s} perm = {list(perm)}  (in={n.input[0].rsplit('/',1)[-1]})")

# in_proj 权重与输出 dtype
n = d_prod[f"{blk}/in_proj/MatMul_output_0"]
print("in_proj MatMul in:", [i.rsplit('/', 1)[-1] for i in n.input])
w = None
for i in n.input:
    a = const_of(i)
    if a is not None and getattr(a, "ndim", 0) == 2:
        w = a
        print("in_proj W shape:", a.shape, "dtype:", a.dtype)
# Softmax 属性
sm = d_prod[f"{blk}/MatMul_output_0"] if f"{blk}/MatMul_output_0" in d_prod else None
