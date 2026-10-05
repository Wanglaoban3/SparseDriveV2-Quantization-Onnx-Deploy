# -*- coding: utf-8 -*-
"""E1 int8 实验归因：E1 profile vs 基线 pg profile 的逐 kernel 差异 + 桶归类。"""
import io
import json
import sys
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
BASE = os.path.join(_ROOT, "deploy/artifacts/prof/prof_real_pg.json")
E1 = os.path.join(_ROOT, "deploy/artifacts/prof/pull_qdq/prof_real_qdq_e1.json")


def load(p):
    out = {}
    for e in json.load(open(p)):
        if "name" in e and "averageMs" in e:
            out[e["name"]] = e["averageMs"]
    return out


pg, e1 = load(BASE), load(E1)
tot_pg, tot_e1 = sum(pg.values()), sum(e1.values())
print(f"total kernel time: pg={tot_pg:.3f}  e1={tot_e1:.3f}  delta={tot_e1-tot_pg:+.3f} ms\n")

gone = set(pg) - set(e1)
new = set(e1) - set(pg)
common = set(pg) & set(e1)

print(f"== 消失的 kernel ({len(gone)}), 消失时间合计 {sum(pg[n] for n in gone):.3f} ms:")
for n in sorted(gone, key=lambda x: -pg[x]):
    if pg[n] > 0.001:
        print(f"  -{pg[n]:8.4f}  {n[:120]}")

print(f"\n== 新增的 kernel ({len(new)}), 新增时间合计 {sum(e1[n] for n in new):.3f} ms:")
for n in sorted(new, key=lambda x: -e1[x]):
    if e1[n] > 0.001:
        print(f"  +{e1[n]:8.4f}  {n[:120]}")

deltas = sorted(((e1[n] - pg[n], n) for n in common), key=lambda x: x[0])
print(f"\n== 同名 kernel 变化 top10 (变慢):")
for dv, n in deltas[-10:]:
    if dv > 0.002:
        print(f"  {pg[n]:8.4f} -> {e1[n]:8.4f}  (d={dv:+7.4f})  {n[:110]}")
print(f"\n== 同名 kernel 变化 top10 (变快):")
for dv, n in deltas[:10]:
    if dv < -0.002:
        print(f"  {pg[n]:8.4f} -> {e1[n]:8.4f}  (d={dv:+7.4f})  {n[:110]}")

# 归类汇总
def bucket(n):
    nl = n.lower()
    if "reformatting" in nl:
        return "reformat"
    if "quantize" in nl or "dequantize" in nl:
        return "quantize/dequantize"
    if "deformableaggregation" in nl:
        return "DFA插件"
    if "foreignnode" in nl:
        return "myelin ForeignNode"
    if "img_backbone" in nl or "img_neck" in nl or "/model/_backbone" in nl:
        return "backbone/neck"
    if "fusedmha" in nl or "attention" in nl:
        return "attention"
    return "其他"


b_pg, b_e1 = {}, {}
for n, v in pg.items():
    b_pg.setdefault(bucket(n), 0.0)
    b_pg[bucket(n)] += v
for n, v in e1.items():
    b_e1.setdefault(bucket(n), 0.0)
    b_e1[bucket(n)] += v
print("\n== 桶对比 (pg -> e1):")
for b in sorted(set(b_pg) | set(b_e1), key=lambda x: -(b_e1.get(x, 0))):
    print(f"  {b:22s} {b_pg.get(b,0):8.3f} -> {b_e1.get(b,0):8.3f}  "
          f"(d={b_e1.get(b,0)-b_pg.get(b,0):+7.3f})")
