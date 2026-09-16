"""Verify deploy optimizations: outputs match pre-optimization fp32 reference; measure latency."""
import os, sys, time
import numpy as np
import torch

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = os.environ.get("TRT_DEPLOY_DIR", os.path.join(DEPLOY_DIR, "artifacts"))

import ptq_pipeline as pp

cfg, wrapper, ds = pp.build()

d = np.load(os.path.join(OUT, "sample_inputs.npz"))
ref = np.load(os.path.join(OUT, "reference_traj_fp32.npz"))["traj_0"]
imgs = torch.tensor(d["imgs"], device="cuda")
proj = torch.tensor(d["proj"], device="cuda")
iwh = torch.tensor(d["iwh"], device="cuda")
status = torch.tensor(d["status"], device="cuda")

with torch.no_grad():
    out = wrapper(imgs, proj, iwh, status)
traj_new = out[0].float().cpu().numpy()
diff = np.abs(traj_new - ref)
print(f"parity vs pre-optimization fp32: max|dTraj|={diff.max():.2e} m, mean|dTraj|={diff.mean():.2e} m")

with torch.no_grad():
    for _ in range(5):
        wrapper(imgs, proj, iwh, status)
    torch.cuda.synchronize()
    t0 = time.time()
    N = 30
    for _ in range(N):
        wrapper(imgs, proj, iwh, status)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N * 1000
print(f"OPTIMIZED fp32 forward: {dt:.1f} ms/frame  (baseline was 82.8 ms)")
