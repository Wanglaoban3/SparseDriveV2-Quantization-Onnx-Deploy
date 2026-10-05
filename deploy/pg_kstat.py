# -*- coding: utf-8 -*-
"""Exact entry-count statistics from real dumped DFA inputs (val_00).
k(n) = # of (cam,level) pairs with cam-guard pass AND any-group w > 0."""
import numpy as np

D = "/opt/m0/sd2/outs/dfatap6/val_00/"
loc = np.fromfile(D + "_model__trajectory_head_decoder_layers.0_p_deform_model_Reshape_4_output_0.bin",
                  dtype=np.float32).reshape(512000, 3, 2)
w = np.fromfile(D + "_model__trajectory_head_decoder_layers.0_p_deform_model_Reshape_5_output_0.bin",
                dtype=np.float32).reshape(512000, 3, 4, 8)

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
