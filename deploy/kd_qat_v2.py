"""KD-QAT v2: normalized loss, smaller lr, keep-best-vs-PTQ guarantee.
Final deliverable = whichever of {PTQ, PTQ+KD} validates better."""
import copy
import json
import os
import sys

import numpy as np
import torch

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = os.environ.get("TRT_DEPLOY_DIR", os.path.join(DEPLOY_DIR, "artifacts"))

import export_onnx as eo
import ptq_pipeline as pp
import modelopt.torch.quantization as mtq
import navsim.agents.sparsedrive.ops.deformable_aggregation as damod


def kd_loss(student_out, teacher_out):
    """Mean of per-head normalized MSEs (each head contributes equally, O(1) scale)."""
    heads = [student_out[1:2][0]]  # traj_scores
    ts = [teacher_out[1]]
    heads = [student_out[1]]
    ts = [teacher_out[1]]
    heads += list(student_out[2:])
    ts += list(teacher_out[2:])
    total = 0.0
    for s, t in zip(heads, ts):
        var = t.var().clamp_min(1e-6)
        total = total + torch.nn.functional.mse_loss(s, t) / var
    return total / len(heads)


def main():
    torch.manual_seed(0)
    cfg, teacher, ds = pp.build()
    teacher.cuda().eval()

    print("[1/6] teacher fp32 reference...")
    ref, gts = pp.run_batch(teacher, ds, pp.VAL_IDX)
    ref_gt = float(np.mean([np.abs(r[0][0] - g).mean() for r, g in zip(ref, gts)]))

    print("[2/6] calibrate + fallbacks...")
    def forward_loop(m):
        for i in pp.CALIB_IDX:
            inp, _, _ = pp.to_inputs(ds, i)
            m(*inp)

    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_DEFAULT_CFG, forward_loop)
    student.cuda().eval()
    report = json.load(open(os.path.join(OUT, "ptq_report_final.json")))
    groups = dict(pp.quant_groups(student.model))
    for name in report.get("fallback", []):
        if name in groups:
            pp.set_group(groups[name], False)

    qout, _ = pp.run_batch(student, ds, pp.VAL_IDX)
    ptq_cmp = pp.compare(ref, qout)
    ptq_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(qout, gts)]))
    print("    PTQ baseline:", json.dumps(ptq_cmp), "GT:", round(ptq_gt, 4))

    print("[3/6] KD-QAT v2 (normalized loss, lr 2e-6, 16 steps)...")
    state_backup = copy.deepcopy(student.state_dict())
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
                loss = kd_loss(s_out, t_out) / batch
                loss.backward()
                loss_sum += float(loss)
            opt.step()
            step += 1
            if step % 4 == 0:
                print(f"    step {step}/{steps} kd_loss~{loss_sum:.4f}")

    print("[4/6] post-KD validation + keep-best...")
    qout2, _ = pp.run_batch(student, ds, pp.VAL_IDX)
    kd_cmp = pp.compare(ref, qout2)
    kd_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(qout2, gts)]))
    print("    PTQ:", json.dumps(ptq_cmp), "GT:", round(ptq_gt, 4))
    print("    KD :", json.dumps(kd_cmp), "GT:", round(kd_gt, 4))

    def score(cmp_, gt):
        return cmp_["traj_l1"] + cmp_["metric_mae"] + abs(gt - ref_gt)

    use_kd = score(kd_cmp, kd_gt) < score(ptq_cmp, ptq_gt)
    final_cmp, final_gt, tag = (kd_cmp, kd_gt, "ptq+kd") if use_kd else (ptq_cmp, ptq_gt, "ptq")
    if not use_kd:
        student.load_state_dict(state_backup)
        print("    KD regressed -> keeping PTQ model")
    else:
        print("    KD improved -> keeping KD model")
    print(f"    FINAL ({tag}):", json.dumps(final_cmp), "GT:", round(final_gt, 4))
    json.dump(dict(fp32_gt=ref_gt, ptq=ptq_cmp, ptq_gt=ptq_gt, kd=kd_cmp, kd_gt=kd_gt, final_tag=tag,
                   final=final_cmp, final_gt=final_gt),
              open(os.path.join(OUT, "kd_qat_report.json"), "w"), indent=2)
    torch.save(student.state_dict(), os.path.join(OUT, "sparsedrive_int8_final.state_dict.pt"))

    print("[5/6] QDQ ONNX export...")
    import torch.nn as _nn
    damod.DeformableAggregationFunction.symbolic = staticmethod(eo.make_symbolic())
    for m in student.modules():
        if isinstance(m, _nn.MultiheadAttention):
            m.train()
    onnx_path = os.path.join(OUT, "sparsedrive_int8_qdq.onnx")
    inp, _, _ = pp.to_inputs(ds, pp.VAL_IDX[0])
    with torch.no_grad():
        torch.onnx.export(
            student, inp, onnx_path,
            input_names=["imgs", "projection_mat", "image_wh", "status_feature"],
            output_names=["trajectory", "traj_scores"] + [f"metric_{m}" for m in eo.METRICS],
            opset_version=17,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.PRESERVE,
        )
    import onnx
    import collections
    g = onnx.load(onnx_path)
    ops = collections.Counter(n.op_type for n in g.graph.node)
    used = set()
    for n in g.graph.node:
        used.update(n.input)
    ok_inputs = all(i.name in used for i in g.graph.input)
    print(f"    QDQ={ops.get('QuantizeLinear')} DFA={ops.get('DeformableAggregation')} inputs_all_used={ok_inputs}")
    print("KD_V2_DONE ->", onnx_path)


if __name__ == "__main__":
    main()
