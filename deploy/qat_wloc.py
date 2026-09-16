"""QAT for w/loc INT8 inside DFA + real PDM metric supervision.

- STE fake-quant (fixed [0,1]) on the DFA sampling weights (w) and coordinates (loc)
- Short fine-tune with the model's own losses: path/velocity/trajectory + REAL PDM
  metric losses (metric caches exist for all 138 mini tokens; scorer runs serially
  in-process to avoid platform spawn-pool quirks)
- Reports how much of the w/loc int8 drift is recovered on held-out samples.
"""
import copy
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ptq_pipeline as pp
import navsim.agents.sparsedrive.ops.deformable_aggregation as damod
import navsim.agents.sparsedrive.blocks as blocks_mod
import navsim.agents.sparsedrive.custom_decoder as cd_mod
from navsim.agents.sparsedrive.scorer import get_pdm_score_v1 as gmod

QAT_STATE = {"on": False}


class STEFixedRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        s = (hi - lo) / 254.0
        return torch.clamp(torch.round((x - lo) / s), 0, 254) * s + lo

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


_orig_DAF = blocks_mod.DAF


def daf_qat(col, ss, ssi, loc, w, depth_prob=None, depth=None):
    if QAT_STATE["on"]:
        loc = STEFixedRange.apply(loc, 0.0, 1.0)
        w = STEFixedRange.apply(w, 0.0, 1.0)
    return _orig_DAF(col, ss, ssi, loc, w, depth_prob, depth)


def main():
    torch.manual_seed(0)
    cfg, wrapper, ds = pp.build()
    blocks_mod.DAF = daf_qat

    # serial in-process PDM scorer (bypass platform spawn pool)
    gmod._init_pool()

    def serial_score(trajectory, metric_cache_path):
        traj_np = trajectory.detach().cpu().numpy()
        return [gmod._pdm_worker((p, traj_np[b])) for b, p in enumerate(metric_cache_path)]

    cd_mod.get_pdm_score_v1 = serial_score

    # token -> metric cache relative path (matches the loss code's construction:
    # <p>.split('/') -> insert 'unknown' before last -> + '/metric_cache.pkl')
    token2cache = {}
    for p in glob.glob(os.path.join(ROOT, "exp", "metric_cache_mini_v1", "*", "*", "*", "metric_cache.pkl")):
        parts = p[len(ROOT) + 1:].replace("\\", "/").split("/")  # exp/metric_cache_mini_v1/<log>/unknown/<token>/metric_cache.pkl
        token2cache[parts[-2]] = "/".join(parts[:-3])           # exp/metric_cache_mini_v1/<log>  ('unknown' is re-inserted by the loss code)
    print("metric caches indexed:", len(token2cache))

    def fwd_train(m, i):
        features, targets, token = ds[i]
        cf = features["camera_feature"]
        f = {"camera_feature": {"imgs": cf["imgs"].unsqueeze(0).float().cuda(),
                                "projection_mat": cf["projection_mat"].unsqueeze(0).float().cuda(),
                                "image_wh": torch.as_tensor(np.asarray(cf["image_wh"])).unsqueeze(0).float().cuda()},
             "status_feature": features["status_feature"].unsqueeze(0).float().cuda()}
        targets = {k: (v.unsqueeze(0).float().cuda() if torch.is_tensor(v) else v)
                   for k, v in targets.items()}
        if token in token2cache:
            targets["token_path"] = [token2cache[token] + "/" + token]
        return m(f, targets)

    print("[1/4] fp32 reference (QAT off)...")
    QAT_STATE["on"] = False
    ref, gts = pp.run_batch(wrapper, ds, pp.VAL_IDX)
    ref_gt = float(np.mean([np.abs(r[0][0] - g).mean() for r, g in zip(ref, gts)]))

    print("[2/4] baseline drift with w/loc int8 (no QAT)...")
    QAT_STATE["on"] = True
    base_out, _ = pp.run_batch(wrapper, ds, pp.VAL_IDX)
    base_cmp = pp.compare(ref, base_out)
    base_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(base_out, gts)]))
    print("    ", json.dumps(base_cmp), "GT:", round(base_gt, 4))

    print("[3/4] QAT (STE on w/loc, full real losses incl PDM metrics)...")
    qmodel = wrapper
    qmodel.train()
    for mod in qmodel.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.eval()
        if type(mod).__name__ == "SparseBackbone":
            mod.use_grid_mask = False
    opt = torch.optim.AdamW([p for p in qmodel.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.0)
    batch, steps = 4, 24
    pool = list(range(len(ds)))
    step = 0
    while step < steps:
        np.random.shuffle(pool)
        for b0 in range(0, len(pool) - batch + 1, batch):
            if step >= steps:
                break
            opt.zero_grad()
            loss_sum = 0.0
            for i in pool[b0:b0 + batch]:
                _, loss_dict = fwd_train(qmodel.model, i)
                loss = loss_dict["loss"] / batch
                loss.backward()
                loss_sum += float(loss)
            opt.step()
            step += 1
            if step % 4 == 0:
                print(f"    step {step}/{steps} loss~{loss_sum:.4f}")
    qmodel.eval()

    print("[4/4] post-QAT drift (w/loc int8 still active)...")
    QAT_STATE["on"] = True
    qout, _ = pp.run_batch(qmodel, ds, pp.VAL_IDX)
    qat_cmp = pp.compare(ref, qout)
    qat_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(qout, gts)]))
    print("    w/loc-int8 before QAT:", json.dumps(base_cmp), "GT:", round(base_gt, 4))
    print("    w/loc-int8 after  QAT:", json.dumps(qat_cmp), "GT:", round(qat_gt, 4))
    print(f"    fp32 GT={ref_gt:.4f}")

    QAT_STATE["on"] = False
    fout, _ = pp.run_batch(qmodel, ds, pp.VAL_IDX)
    f_cmp = pp.compare(ref, fout)
    f_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(fout, gts)]))
    print("    trained-fp32 vs original fp32:", json.dumps(f_cmp), "GT:", round(f_gt, 4))

    json.dump(dict(fp32_gt=ref_gt, base=base_cmp, base_gt=base_gt,
                   qat=qat_cmp, qat_gt=qat_gt, trained_fp32=f_cmp, trained_fp32_gt=f_gt),
              open(os.path.join(pp.OUT, "qat_wloc_report.json"), "w"), indent=2)
    torch.save(qmodel.state_dict(), os.path.join(pp.OUT, "sparsedrive_int8_wloc_qat.state_dict.pt"))
    print("QAT_WLOC_DONE")


if __name__ == "__main__":
    main()
