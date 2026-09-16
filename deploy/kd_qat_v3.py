"""KD-QAT v3: maximize INT8 coverage.
Student = FULL int8 (all mtq modules, no fallbacks) + DFA feat int8 (per-C scale)
+ w/loc STE int8. Teacher = fp32. Pure normalized-KD loss on outputs (stable,
reference: kd_qat_v2), lr 2e-6, keep-best vs the PTQ-full-int8 starting point.
Incumbent config (feat int8 + 12 fallbacks, w/loc fp16) drift quoted from kd_qat_v2.
"""
import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import export_onnx as eo
import ptq_pipeline as pp
import modelopt.torch.quantization as mtq
import navsim.agents.sparsedrive.ops.deformable_aggregation as damod
import navsim.agents.sparsedrive.blocks as blocks_mod
import onnx
from onnx import numpy_helper


class STEFixedRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        s = (hi - lo) / 254.0
        return torch.clamp(torch.round((x - lo) / s), 0, 254) * s + lo

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class STEPerC(torch.autograd.Function):
    """per-C-channel symmetric int8 round-trip with straight-through grad."""

    @staticmethod
    def forward(ctx, x, scale):
        return torch.clamp(torch.round(x / scale), -127, 127) * scale

    @staticmethod
    def backward(ctx, grad):
        return grad, None


QAT = {"on": False, "feat_scale": None}
_orig_DAF = blocks_mod.DAF


def daf_qat(col, ss, ssi, loc, w, depth_prob=None, depth=None):
    if QAT["on"]:
        col = STEPerC.apply(col, QAT["feat_scale"])
        loc = STEFixedRange.apply(loc, 0.0, 1.0)
        w = STEFixedRange.apply(w, 0.0, 1.0)
    return _orig_DAF(col, ss, ssi, loc, w, depth_prob, depth)


def load_feat_scale():
    m = onnx.load(os.path.join(pp.OUT, "sparsedrive_int8_qdq_feat8.onnx"))
    for init in m.graph.initializer:
        if init.name == "dfa_feat_scale":
            return torch.tensor(numpy_helper.to_array(init), dtype=torch.float32).cuda()
    raise RuntimeError("dfa_feat_scale not found")


def main():
    torch.manual_seed(0)
    QAT["feat_scale"] = load_feat_scale()
    blocks_mod.DAF = daf_qat

    teacher = pp.build()[1]
    teacher.cuda().eval()
    print("[1/5] teacher fp32 reference...")
    with torch.no_grad():
        ref, gts = pp.run_batch(teacher, ds := pp.build()[2], pp.VAL_IDX)
    ref_gt = float(np.mean([np.abs(r[0][0] - g).mean() for r, g in zip(ref, gts)]))

    print("[2/5] student: FULL int8 (no fallbacks) + feat/w/loc int8...")
    def calib_loop(m):
        for i in pp.CALIB_IDX:
            inp, _, _ = pp.to_inputs(ds, i)
            m(*inp)

    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_DEFAULT_CFG, calib_loop)
    student.cuda().eval()
    QAT["on"] = True  # w/loc STE on from here on
    qout, _ = pp.run_batch(student, ds, pp.VAL_IDX)
    ptq_cmp = pp.compare(ref, qout)
    ptq_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(qout, gts)]))
    print("    PTQ full-int8 baseline:", json.dumps(ptq_cmp), "GT:", round(ptq_gt, 4))

    print("[3/5] KD-QAT (normalized KD, lr 2e-6, 16 steps, keep-best)...")
    state_backup = copy.deepcopy(student.state_dict())
    heads = lambda s, t: sum(
        torch.nn.functional.mse_loss(a, b) / b.var().clamp_min(1e-6)
        for a, b in zip([s[1]] + list(s[2:]), [t[1]] + list(t[2:]))) / 7.0
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2e-6, weight_decay=0.0)
    batch, steps = 4, 16
    pool = list(pp.CALIB_IDX)
    step = 0
    while step < steps:
        np.random.shuffle(pool)
        for b0 in range(0, len(pool) - batch + 1, batch):
            if step >= steps:
                break
            opt.zero_grad()
            loss_sum = 0.0
            for i in pool[b0:b0 + batch]:
                inp, _, _ = pp.to_inputs(ds, i)
                with torch.no_grad():
                    t_out = teacher(*inp)
                s_out = student(*inp)
                loss = heads(s_out, t_out) / batch
                loss.backward()
                loss_sum += float(loss)
            opt.step()
            step += 1
            if step % 4 == 0:
                print(f"    step {step}/{steps} kd~{loss_sum:.4f}")

    print("[4/5] post-KD eval + keep-best...")
    qout2, _ = pp.run_batch(student, ds, pp.VAL_IDX)
    kd_cmp = pp.compare(ref, qout2)
    kd_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(qout2, gts)]))
    print("    full-int8 PTQ:", json.dumps(ptq_cmp), "GT:", round(ptq_gt, 4))
    print("    full-int8 KD :", json.dumps(kd_cmp), "GT:", round(kd_gt, 4))
    print("    incumbent (feat-int8 + 12 fallbacks, w/loc fp16): metric_mae 0.656, GT 1.0925 (kd_qat_v2)")

    def score(c, gt):
        return c["traj_l1"] + c["metric_mae"] + abs(gt - ref_gt)

    kd_better = score(kd_cmp, kd_gt) < score(ptq_cmp, ptq_gt)
    final_cmp, final_gt = (kd_cmp, kd_gt) if kd_better else (ptq_cmp, ptq_gt)
    final_tag = "kd" if kd_better else "ptq"
    if not kd_better:
        student.load_state_dict(state_backup)
        print("    KD regressed vs PTQ-full -> keep PTQ-full weights")
    print(f"    FINAL full-int8 ({final_tag}):", json.dumps(final_cmp), "GT:", round(final_gt, 4))
    json.dump(dict(ref_gt=ref_gt, ptq_full=ptq_cmp, ptq_full_gt=ptq_gt,
                   kd=kd_cmp, kd_gt=kd_gt, final_tag=final_tag, final=final_cmp, final_gt=final_gt),
              open(os.path.join(pp.OUT, "kd_qat_v3_report.json"), "w"), indent=2)
    torch.save(student.state_dict(), os.path.join(pp.OUT, "sparsedrive_fullint8.state_dict.pt"))
    print("[5/5] done (export of full-int8 onnx skipped; rerun export pipeline with this state if adopted)")


if __name__ == "__main__":
    main()
