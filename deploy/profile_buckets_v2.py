# -*- coding: utf-8 -*-
"""SparseDriveV2: bucket a trtexec --dumpProfile table into module groups, per-iter ms.

v2 model buckets (plan Task 7):
  img_backbone / img_neck / dfa_plugin / decoder_attn / heads / reformat / other
Parses the trtexec stdout '=== Profile ===' text table (v1-proven format:
'[I]  <total_ms>  <avg_ms>  <med_ms>  <pct>  <name>'); pass the prof log file(s).

usage: python deploy/profile_buckets_v2.py <prof.log> [<prof2.log> ...] [--out report.md]
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

ROW = re.compile(r"(?:\[.*?\])?\s*\[I\]\s+([0-9.]+)\s+([0-9.]+)\s+"
                 r"([0-9.]+)\s+([0-9.]+)\s+(.+?)\s*$")

ORDER = ["dfa_plugin", "myelin_fused", "img_backbone", "reformat", "other"]


def bucket_of(name):
    n = name.lower()
    if n == "total":
        return "_total"
    if "reformatting copynode" in n:
        # 喂 DFA 插件的 reformat 行名字里也带 DeformableAggregation，先判 reformat
        return "reformat"
    if "deformableaggregation" in n:
        return "dfa_plugin"
    if "foreignnode" in n or "myelin" in n:
        # decoder/attention/heads 大多被 Myelin 融合进这些区域（深层归属看 layerinfo）
        return "myelin_fused"
    if "img_backbone" in n or "backbone" in n or "img_neck" in n or "neck" in n or "fpn" in n:
        return "img_backbone"
    return "other"


def parse(path):
    buckets = {}
    total = None
    nrows = 0
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = ROW.match(ln)
        if not m:
            continue
        avg_ms = float(m.group(2))
        name = m.group(5)
        if name == "Total":
            total = avg_ms
            continue
        nrows += 1
        b = bucket_of(name)
        buckets.setdefault(b, [0.0, 0])
        buckets[b][0] += avg_ms
        buckets[b][1] += 1
    return buckets, total, nrows


def main():
    argv = sys.argv[1:]
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        out_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    args = argv
    lines = ["# M4 profile buckets (SparseDriveV2)\n"]
    for path in args:
        buckets, total, nrows = parse(path)
        s = sum(v[0] for v in buckets.values())
        lines.append(f"## {path}\n")
        lines.append(f"rows={nrows} total_row={total} sum_of_layers={s:.3f} ms\n")
        lines.append("| bucket | ms/iter | % | layers |")
        lines.append("|---|---:|---:|---:|")
        for b in ORDER + sorted(k for k in buckets if k not in ORDER):
            if b not in buckets:
                continue
            ms, cnt = buckets[b]
            lines.append(f"| {b} | {ms:.3f} | {100.0 * ms / s:.1f}% | {cnt} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
