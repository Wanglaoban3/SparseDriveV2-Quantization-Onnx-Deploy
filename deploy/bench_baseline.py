"""Baseline latency of current (pre-optimization) fp32 model."""
import os, sys, time
import numpy as np
import torch

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import ptq_pipeline as pp

cfg, wrapper, ds = pp.build()
inp, _, _ = pp.to_inputs(ds, 0)

with torch.no_grad():
    for _ in range(5):
        wrapper(*inp)
    torch.cuda.synchronize()
    t0 = time.time()
    N = 30
    for _ in range(N):
        wrapper(*inp)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N * 1000
print(f"BASELINE fp32 forward: {dt:.1f} ms/frame")
