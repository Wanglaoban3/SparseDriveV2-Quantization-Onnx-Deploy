# -*- coding: utf-8 -*-
"""A/B 全图 profile 对比：pg（现役 v4）vs mha（FusedMHA 插件版）。

用法: python cmp_prof_mha.py [pg.json] [mha.json]
输出: FusedMHA kernel 实测合计、逐 kernel Δ（按名匹配）、消失/新增 kernel、
      foreign node 结构变化、总 kernel 时间差。
"""
import io
import json
import sys
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
PG = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(_ROOT, "deploy/artifacts/prof/prof_real_pg.json")
MHA = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.join(_ROOT, "deploy/artifacts/prof/prof_real_mha.json")


def load(p):
    out = {}
    for e in json.load(open(p)):
        if "name" in e and "averageMs" in e:
            out[e["name"]] = e["averageMs"]
    return out


pg, mha = load(PG), load(MHA)

fmha = [(v, n) for n, v in mha.items() if "fusedmha" in n.lower() or "FusedMHA" in n]
s_fmha = sum(v for v, _ in fmha)
print(f"== FusedMHA kernels (mha profile): {len(fmha)} entries, sum={s_fmha:.4f} ms")
for v, n in sorted(fmha, reverse=True):
    print(f"  {v:8.4f}  {n[:120]}")

names_pg, names_mha = set(pg), set(mha)
gone = sorted(names_pg - names_mha)
new = sorted(names_mha - names_pg)
print(f"\n== kernels gone from pg profile ({len(gone)}):")
for n in gone:
    print(f"  -{pg[n]:8.4f}  {n[:130]}")
print(f"\n== kernels new in mha profile ({len(new)}):")
for n in new:
    print(f"  +{mha[n]:8.4f}  {n[:130]}")

common = names_pg & names_mha
deltas = [(mha[n] - pg[n], pg[n], mha[n], n) for n in common]
deltas.sort()
print(f"\n== biggest decreases (pg -> mha):")
for dv, a, b, n in deltas[:12]:
    print(f"  {a:8.4f} -> {b:8.4f}  (d={dv:+8.4f})  {n[:110]}")
print(f"\n== biggest increases (pg -> mha):")
for dv, a, b, n in reversed(deltas[-12:]):
    print(f"  {a:8.4f} -> {b:8.4f}  (d={dv:+8.4f})  {n[:110]}")

tot_pg, tot_mha = sum(pg.values()), sum(mha.values())
print(f"\n== total kernel time: pg={tot_pg:.3f}  mha={tot_mha:.3f}  "
      f"delta={tot_mha - tot_pg:+.3f} ms")
print(f"== FusedMHA sum = {s_fmha:.4f} ms; net myelin/other delta = "
      f"{(tot_mha - tot_pg) - s_fmha:+.4f} ms")
