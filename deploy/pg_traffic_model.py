# -*- coding: utf-8 -*-
"""Exact feat-tap traffic model from real loc/w (val_00), replicating plan phase C
corner-weight computation (fp32, double-promoted -0.5).

Ground-truth numbers produced (2026-10-05, dfatap6 val_00):
  valid entries 1,976,676 (mean k=3.861), uniform 25%/level
  nonzero-corner histogram [0, 302, 47245, 0, 1929129] -> mean 3.952 corners/entry
  nonzero groups per valid entry = 8.000 (mask is per-(row,cs), broadcast over G)
  feat tap traffic 3.999 GB; entries w+r 0.25 GB; plan logits 0.23 GB
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
D = os.path.join(_ROOT, "deploy/artifacts/prof/kstat")
loc = np.fromfile(os.path.join(D, "loc.bin"), dtype=np.float32).reshape(512000, 3, 2)
w = np.fromfile(os.path.join(D, "w.bin"), dtype=np.float32).reshape(512000, 3, 4, 8)

SHAPES = [(64, 128), (32, 64), (16, 32), (8, 16)]   # (H, W) per level

lw = loc[:, :, 0]                                    # (N, cams)
lh = loc[:, :, 1]
guard = (lw > 0) & (lw < 1) & (lh > 0) & (lh < 1)
wnz = (w > 0).any(axis=3)                            # (N, cams, levels)
valid = guard[:, :, None] & wnz
nz_groups = (w > 0).sum(axis=3)                      # (N, cams, levels) per entry

tot_entries = int(valid.sum())
tot_corners = 0
corner_hist = np.zeros(5, dtype=np.int64)            # nonzero corners per entry
nzg_sum = 0
level_valid = np.zeros(4, dtype=np.int64)
for lv, (H, W) in enumerate(SHAPES):
    h_im = (lh.astype(np.float64) * H - 0.5).astype(np.float32)
    w_im = (lw.astype(np.float64) * W - 0.5).astype(np.float32)
    h_low = np.floor(h_im).astype(np.int32)
    w_low = np.floor(w_im).astype(np.int32)
    fr_h = (h_im - h_low).astype(np.float32)
    fr_w = (w_im - w_low).astype(np.float32)
    hh = 1.0 - fr_h
    hw2 = 1.0 - fr_w
    c_x = (h_low >= 0) & (w_low >= 0)
    c_y = (h_low >= 0) & (w_low + 1 <= W - 1)
    c_z = (h_low + 1 <= H - 1) & (w_low >= 0)
    c_w = (h_low + 1 <= H - 1) & (w_low + 1 <= W - 1)
    nc = (c_x.astype(np.int8) + c_y + c_z + c_w)       # (N, cams)
    m = valid[:, :, lv]
    tot_corners += int(nc[m].sum())
    for v in range(5):
        corner_hist[v] += int((nc[m] == v).sum())
    nzg_sum += int(nz_groups[:, :, lv][m].sum())
    level_valid[lv] = int(m.sum())

print("valid entries            : %d  (mean k=%.3f)" % (tot_entries, tot_entries / 512000))
print("valid by level           :", level_valid.tolist(),
      "level0 share=%.1f%%" % (100.0 * level_valid[0] / tot_entries))
print("nonzero-corner histogram :", corner_hist.tolist())
print("mean corners per entry   : %.3f" % (tot_corners / tot_entries))
print("mean nonzero groups/entry: %.3f (of 8)" % (nzg_sum / tot_entries))

C = 256
feat_bytes = tot_corners * C * 2                        # 64B per (corner, group-warp)
print("feat tap traffic model   : %.3f GB" % (feat_bytes / 1e9))
print("  at 165 GB/s -> %.1f ms;  at 129 GB/s -> %.1f ms"
      % (feat_bytes / 165e9 * 1e3, feat_bytes / 129e9 * 1e3))

ent_bytes = tot_entries * 64 * 2
print("entries w+r traffic      : %.2f GB" % (ent_bytes / 1e9))
logits_bytes = (512000 * 12 * 8 * 2) * 2 + tot_entries * 16
print("logits traffic (plan)    : %.2f GB" % (logits_bytes / 1e9))
tot = feat_bytes + ent_bytes + logits_bytes
print("TOTAL traffic model      : %.2f GB -> floor %.1f ms @165GB/s"
      % (tot / 1e9, tot / 165e9 * 1e3))
