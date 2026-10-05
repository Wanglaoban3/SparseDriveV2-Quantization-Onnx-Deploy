# -*- coding: utf-8 -*-
"""手术：把 3 个 DFA 节点 ss/ssi 的 Cast(to=INT8) 翻成 Cast(to=INT32)。

背景：export_onnx.py _cast32 曾误写 to_i=3（INT8），ssi 值域上万千被截断，
且插件 int32 契约在 TRT format 协商必死（"could not find any supported
formats"）。i64→i32 对本模型量级无损。改前备份 .bak_pre_i32。
"""
import io
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import onnx
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
PATH = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_int8_qdq_folded.onnx")
BAK = PATH + ".bak_pre_i32"

shutil.copyfile(PATH, BAK)
m = onnx.load(PATH)
g = m.graph

producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n

flips = []
for dfa in (n for n in g.node if n.op_type == "DeformableAggregation"):
    for k in (1, 2):  # ss, ssi
        p = producer.get(dfa.input[k])
        assert p is not None and p.op_type == "Cast", (dfa.name, k, p)
        to = next(a.i for a in p.attribute if a.name == "to")
        assert to == 3, f"{p.name} to={to} (期望误写的 3)"
        for a in p.attribute:
            if a.name == "to":
                a.i = 6
        flips.append(p.name)

print("flipped %d casts to INT32:" % len(flips))
for f in flips:
    print("  ", f)
assert len(flips) == 6, flips

# 确认全图再无 Cast(to=INT8)
left = [n.name for n in g.node if n.op_type == "Cast"
        and any(a.name == "to" and a.i == 3 for a in n.attribute)]
print("remaining Cast(to=i8):", left)
assert not left

onnx.checker.check_model(m)
onnx.save(m, PATH)
print("saved (backup at %s)" % BAK)
