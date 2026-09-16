"""SparseDriveV2 PTQ pipeline (ModelOpt, single navsim env):
  calibrate -> validate vs fp32 -> per-module sensitivity -> sensitive fallback -> QDQ ONNX export.

All accuracy checks use a small held-out sample batch (fake-quant inference), as agreed.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

DEPLOY_DIR = os.environ.get("SD_DEPLOY_DIR", os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = os.environ.get("TRT_DEPLOY_DIR", os.path.join(DEPLOY_DIR, "artifacts"))
os.makedirs(OUT, exist_ok=True)

import export_onnx as eo  # applies unflatten patch; provides ExportModel/METRICS/make_symbolic

import modelopt.torch.quantization as mtq
from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder, SparseDriveTargetBuilder
from navsim.planning.training.dataset import CacheOnlyDataset

LOGS = ["2021.06.28.16.29.11_veh-38_01415_01821", "2021.10.11.02.57.41_veh-50_01522_02088"]
CALIB_IDX = list(range(0, 16)) + list(range(69, 85))       # 32 samples, both logs
VAL_IDX = list(range(16, 28)) + list(range(85, 97))        # 24 held-out samples


def build():
    cfg = SparseDriveConfig()
    cfg.dataset_version = "v1"
    cfg.metrics = list(eo.METRICS)
    cfg.velocity_filter_num = [64, 20]
    cfg.path_filter_num = [128, 20]
    agent = SparseDriveAgent(config=cfg, lr=1e-4,
                             checkpoint_path=os.path.join(ROOT, "ckpt", "sparsedrive_navsimv1_92p2.ckpt"))
    agent.initialize()
    model = agent._sparsedrive_model.cuda().eval()
    eo.apply_deploy_optimizations(model)  # frozen vocab + quantizable MHA
    wrapper = eo.ExportModel(model).cuda().eval()
    ds = CacheOnlyDataset(cache_path=os.path.join(ROOT, "exp", "data_cache_mini"),
                          feature_builders=[SparseDriveFeatureBuilder(cfg)],
                          target_builders=[SparseDriveTargetBuilder(cfg)],
                          log_names=LOGS)
    return cfg, wrapper, ds


def to_inputs(ds, i):
    features, targets, token = ds[i]
    cf = features["camera_feature"]
    imgs = cf["imgs"].unsqueeze(0).float().cuda()
    proj = cf["projection_mat"].unsqueeze(0).float().cuda()
    iwh = torch.as_tensor(np.asarray(cf["image_wh"]), dtype=torch.float32).unsqueeze(0).cuda()
    status = features["status_feature"].unsqueeze(0).float().cuda()
    gt_traj = targets["trajectory"].float()
    return (imgs, proj, iwh, status), gt_traj, token


@torch.no_grad()
def run_batch(wrapper, ds, idx):
    """Returns (outs list of tuples, gt list)."""
    outs, gts = [], []
    for i in idx:
        inp, gt, _ = to_inputs(ds, i)
        with torch.no_grad():
            o = wrapper(*inp)
        outs.append([t.float().cpu().numpy() for t in o])
        gts.append(gt.numpy())
    return outs, gts


def compare(ref, q):
    """ref/q: lists of output tuples. Output tuple: [traj(1,8,3), scores(1,400), 6x metric(1,400)]."""
    match, traj_l1, score_mae, metric_mae = [], [], [], []
    gt_l1_ref, gt_l1_q = [], []
    for r, qq in zip(ref, q):
        match.append(int(np.argmax(r[1]) == np.argmax(qq[1])))
        traj_l1.append(np.abs(r[0] - qq[0]).mean())
        score_mae.append(np.abs(r[1] - qq[1]).mean())
        metric_mae.append(np.mean([np.abs(a - b).mean() for a, b in zip(r[2:], qq[2:])]))
    return dict(
        argmax_match=float(np.mean(match)),
        traj_l1=float(np.mean(traj_l1)),
        traj_l1_max=float(np.max(traj_l1)),
        score_mae=float(np.mean(score_mae)),
        metric_mae=float(np.mean(metric_mae)),
    )


def drift(m):
    """Composite scalar drift metric (lower better)."""
    return m["traj_l1"] + m["metric_mae"]


def quant_groups(model):
    """One group per quantized module (its input+weight quantizers)."""
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer  # noqa
    groups = []
    for name, mod in model.named_modules():
        if hasattr(mod, "weight_quantizer") and hasattr(mod, "input_quantizer"):
            groups.append((name, mod))
    return groups


def set_group(mod, enable):
    for a in ("input_quantizer", "weight_quantizer"):
        q = getattr(mod, a, None)
        if q is None:
            continue
        q.enable() if enable else q.disable()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "calib_validate", "sensitivity", "export"])
    ap.add_argument("--val-n", type=int, default=24)
    ap.add_argument("--sens-top", type=int, default=12, help="fallback count for most sensitive modules")
    args = ap.parse_args()

    cfg, wrapper, ds = build()
    n_total = len(ds)
    val_idx = [i for i in VAL_IDX if i < n_total][: args.val_n]
    calib_idx = [i for i in CALIB_IDX if i < n_total]
    print(f"calib={len(calib_idx)} val={len(val_idx)} (pool {n_total})")

    # ---- fp32 reference ----
    print("[1/5] fp32 reference pass...")
    ref, gts = run_batch(wrapper, ds, val_idx)

    # fp32 vs GT distance (planning accuracy baseline)
    ref_gt = float(np.mean([np.abs(r[0][0] - g).mean() for r, g in zip(ref, gts)]))
    print(f"    fp32 mean|traj-GT| = {ref_gt:.4f} m")

    # ---- calibrate ----
    print("[2/5] PTQ calibration (max) over calib set...")
    def forward_loop(m):
        for i in calib_idx:
            inp, _, _ = to_inputs(ds, i)
            m(*inp)

    qwrapper = mtq.quantize(copy.deepcopy(wrapper), mtq.INT8_DEFAULT_CFG, forward_loop)
    qwrapper.cuda().eval()

    # ---- validate fake-quant vs fp32 ----
    print("[3/5] fake-quant validation...")
    qout, _ = run_batch(qwrapper, ds, val_idx)
    base_cmp = compare(ref, qout)
    base = drift(base_cmp)
    q_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(qout, gts)]))
    print("    ALL-INT8:", json.dumps(base_cmp))
    print(f"    quant mean|traj-GT| = {q_gt:.4f} m (fp32 {ref_gt:.4f})")
    json.dump(dict(all_int8=base_cmp, all_int8_gt=q_gt, fp32_gt=ref_gt),
              open(os.path.join(OUT, "ptq_report_base.json"), "w"), indent=2)

    # ---- per-module sensitivity ----
    print("[4/5] per-module sensitivity (this is the slow part)...")
    groups = quant_groups(qwrapper.model)
    print(f"    {len(groups)} quantized modules")
    sens = []
    for gi, (name, mod) in enumerate(groups):
        set_group(mod, False)
        out_d, _ = run_batch(qwrapper, ds, val_idx)
        set_group(mod, True)
        cmp_d = compare(ref, out_d)
        gain = base - drift(cmp_d)  # >0: disabling this module reduces drift
        sens.append(dict(name=name, gain=float(gain), drift=float(drift(cmp_d)),
                         argmax=float(cmp_d["argmax_match"]), traj_l1=float(cmp_d["traj_l1"])))
        if gi % 10 == 0:
            print(f"    [{gi}/{len(groups)}] {name}: gain={gain:.4f}")

    sens.sort(key=lambda x: -x["gain"])
    json.dump(sens, open(os.path.join(OUT, "sensitivity.json"), "w"), indent=2)
    print("    top-sensitive:")
    for s in sens[:10]:
        print(f"      {s['name']}: gain={s['gain']:.4f} drift_if_off={s['drift']:.4f}")

    # ---- fallback: keep top-K sensitive modules in fp16 (disable their int8) ----
    print(f"[5/5] fallback: disable int8 on top-{args.sens_top} sensitive modules...")
    gmap = {n: m for n, m in groups}
    fallback_names = set()
    for s in sens[: args.sens_top]:
        if s["gain"] > 0:
            fallback_names.add(s["name"])
            set_group(gmap[s["name"]], False)
    n_fallback = sum(1 for s in sens[: args.sens_top] if s["gain"] > 0)
    print(f"    fallback modules: {n_fallback}")

    fout, _ = run_batch(qwrapper, ds, val_idx)
    final_cmp = compare(ref, fout)
    f_gt = float(np.mean([np.abs(q[0][0] - g).mean() for q, g in zip(fout, gts)]))
    print("    FINAL (int8+fallback):", json.dumps(final_cmp))
    print(f"    final mean|traj-GT| = {f_gt:.4f} m")
    json.dump(dict(final=final_cmp, final_gt=f_gt, fallback=sorted(fallback_names),
                   sensitivity=sens), open(os.path.join(OUT, "ptq_report_final.json"), "w"), indent=2)

    # ---- QDQ ONNX export ----
    print("exporting QDQ ONNX...")
    import torch.nn as _nn
    for m in qwrapper.modules():
        if isinstance(m, _nn.MultiheadAttention):
            m.train()
    onnx_path = os.path.join(OUT, "sparsedrive_int8_qdq.onnx")
    with torch.no_grad():
        inp, _, _ = to_inputs(ds, val_idx[0])
        torch.onnx.export(
            qwrapper, inp, onnx_path,
            input_names=["imgs", "projection_mat", "image_wh", "status_feature"],
            output_names=["trajectory", "traj_scores"] + [f"metric_{m}" for m in eo.METRICS],
            opset_version=17,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.PRESERVE,
        )
    import onnx
    g = onnx.load(onnx_path)
    ops = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("ONNX node stats:", {k: v for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:12]})
    print("QDQ:", ops.get("QuantizeLinear", 0), "QuantizeLinear /", ops.get("DequantizeLinear", 0), "DequantizeLinear")
    print("DFA nodes:", ops.get("DeformableAggregation", 0))
    print("PIPELINE_DONE ->", onnx_path)


if __name__ == "__main__":
    main()
