# -*- coding: utf-8 -*-
"""同款手术打到 pre_rewrite 备份（fallback 构建候选）。"""
import io
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import onnx
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
PATH = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_int8_qdq_folded_pre_rewrite.onnx")
BAK = PATH + ".bak_pre_i32"

shutil.copyfile(PATH, BAK)
m = onnx.load(PATH)
g = m.graph
producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n

flips = 0
for dfa in (n for n in g.node if n.op_type == "DeformableAggregation"):
    for k in (1, 2):
        p = producer.get(dfa.input[k])
        assert p is not None and p.op_type == "Cast", (dfa.name, k, p)
        to = next(a.i for a in p.attribute if a.name == "to")
        assert to == 3, f"{p.name} to={to}"
        for a in p.attribute:
            if a.name == "to":
                a.i = 6
        flips += 1
assert flips == 6, flips
left = [n.name for n in g.node if n.op_type == "Cast"
        and any(a.name == "to" and a.i == 3 for a in n.attribute)]
assert not left, left
onnx.checker.check_model(m)
onnx.save(m, PATH)
print("pre_rewrite: flipped %d casts, saved (backup %s)" % (flips, BAK))
