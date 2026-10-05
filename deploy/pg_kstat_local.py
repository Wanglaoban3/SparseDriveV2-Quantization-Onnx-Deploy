# -*- coding: utf-8 -*-
"""Exact entry-count statistics from real dumped DFA inputs (val_00), local."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
D = os.path.join(_ROOT, "deploy/artifacts/prof/kstat")
loc = np.fromfile(os.path.join(D, "loc.bin"), dtype=np.float32).reshape(512000, 3, 2)
w = np.fromfile(os.path.join(D, "w.bin"), dtype=np.float32).reshape(512000, 3, 4, 8)

lw = loc[:, :, 0]
lh = loc[:, :, 1]
guard = (lw > 0) & (lw < 1) & (lh > 0) & (lh < 1)          # (N, cams)
wnz = (w > 0).any(axis=3)                                   # (N, cams, levels)
valid = guard[:, :, None] & wnz                             # (N, cams, levels)
k = valid.reshape(512000, 12).sum(axis=1)

print("cam guard pass rate      : %.4f" % guard.mean())
print("w==0 exact entry rate    : %.4f" % ((w == 0).mean()))
print("valid (guard & w>0) rate : %.4f  -> mean k = %.3f" % (valid.mean(), k.mean()))
for q in (1, 5, 25, 50, 75, 95, 99):
    print("  k p%02d = %d" % (q, np.percentile(k, q)))
print("k histogram:", np.bincount(k.astype(np.int64), minlength=13).tolist())

# also: weight mass concentration — how much softmax mass lives outside top-1 cs
wsum = w.sum(axis=(1, 2))                                    # (N, G) = 1 per group
wmax = w.max(axis=(1, 2))
print("w max per (n,g): mean=%.4f  p50=%.4f" % (wmax.mean(), np.median(wmax)))
