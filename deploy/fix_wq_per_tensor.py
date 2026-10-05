# -*- coding: utf-8 -*-
"""权重 QDQ 全部改 per-tensor（v1 板上验证过的形态）。

背景：v2 plain 配置权重 per-channel（axis=0），其中 44 个打在转置前权重
（[out,in]→Transpose→FC/Conv），TRT 8.6.12 对转置后 per-channel 轴不支持 →
头部分类输出系统性失真（metric 头膨胀 3-6x、score 收缩，traj top-mode 反而
无恙）。v1 全 per-tensor 权重板上门禁通过。修法：把权重 Q 的 scale
initializer 换成标量（amax/127 对称）、zp 标量 0，并删除 Q/DQ 的 axis 属性。
f32 权重原样保留，Q 在运行时自行量化——数值可 ORT 链级重放验证。
改前备份 .bak_pre_pertensor。
"""
import io
import shutil
import sys
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np
import onnx
from onnx import numpy_helper
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
PATH = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_int8_qdq_folded_pre_rewrite.onnx")
BAK = PATH + ".bak_pre_pertensor"
EMBED_MAX = 127.0

shutil.copyfile(PATH, BAK)
m = onnx.load(PATH)
g = m.graph
inits = {i.name: i for i in g.initializer}
idx = {i.name: k for k, i in enumerate(g.initializer)}
producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n
consumer = {}
for n in g.node:
    for x in n.input:
        consumer.setdefault(x, []).append(n)

n_fix = 0
prefixes = Counter()
for n in g.node:
    if n.op_type != "QuantizeLinear" or n.input[0] not in inits:
        continue
    w = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
    scale = max(float(np.abs(w).max()), 1e-8) / EMBED_MAX
    scale_name, zp_name = n.input[1], n.input[2]
    # scale/zp 换标量（新名字避免与激活 QDQ 共享的标量冲突/同名覆盖）
    ns, nz = scale_name + "_pt", zp_name + "_pt"
    g.initializer.append(numpy_helper.from_array(np.array(scale, np.float32), name=ns))
    g.initializer.append(numpy_helper.from_array(np.array(0, np.int8), name=nz))
    n.input[1], n.input[2] = ns, nz
    for a in list(n.attribute):
        if a.name == "axis":
            n.attribute.remove(a)
    dq = consumer.get(n.output[0], [None])[0]
    if dq is not None and dq.op_type == "DequantizeLinear":
        dq.input[1], dq.input[2] = ns, nz
        for a in list(dq.attribute):
            if a.name == "axis":
                dq.attribute.remove(a)
    # 记录这个权重属于哪个模块前缀
    prefixes[n.input[0].split("/")[2][:40] if n.input[0].count("/") > 2
             else n.input[0][:40]] += 1
    n_fix += 1

print("converted %d weight QDQ to per-tensor" % n_fix)
for k, v in prefixes.most_common(12):
    print("   %-42s %d" % (k, v))

onnx.checker.check_model(m)
onnx.save(m, PATH)
print("saved, backup=%s" % BAK)
