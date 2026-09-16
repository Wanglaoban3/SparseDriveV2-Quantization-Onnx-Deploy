"""Generate FP32 reference inputs/outputs for board-side engine accuracy checks.

For each held-out validation sample (the same 24 used in every quantization
report) saves one npz with:
  inputs  : imgs / proj / iwh / status   (exactly the engine's 4 inputs)
  refs    : ref_traj / ref_scores / ref_metric_0..5   (FP32 model outputs)
  gt      : gt_traj                       (dataset ground-truth trajectory)

Output: deploy/artifacts/engine_ref/val_XX.npz  (large, git-ignored; copy to
the target device together with artifacts/engine_infer_check.py).
"""
import os

import numpy as np

import torch

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
OUT = os.environ.get("TRT_DEPLOY_DIR", os.path.join(DEPLOY_DIR, "artifacts"))
REF_DIR = os.path.join(OUT, "engine_ref")

import ptq_pipeline as pp

cfg, wrapper, ds = pp.build()
os.makedirs(REF_DIR, exist_ok=True)

for n, i in enumerate(pp.VAL_IDX):
    inp, gt, token = pp.to_inputs(ds, i)
    with torch.no_grad():
        o = wrapper(*inp)
    d = {k: t.float().cpu().numpy() for k, t in zip(("imgs", "proj", "iwh", "status"), inp)}
    d["ref_traj"] = o[0].float().cpu().numpy()
    d["ref_scores"] = o[1].float().cpu().numpy()
    for j, m in enumerate(o[2:]):
        d["ref_metric_%d" % j] = m.float().cpu().numpy()
    d["gt_traj"] = gt.numpy().astype(np.float32)
    np.savez_compressed(os.path.join(REF_DIR, "val_%02d.npz" % n), **d)
    print("val_%02d <- token %s" % (n, token))
print("done ->", REF_DIR)
