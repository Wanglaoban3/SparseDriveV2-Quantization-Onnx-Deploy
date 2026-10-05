# -*- coding: utf-8 -*-
"""引擎输出 vs FP32 参考基准对拍（24 样本）+ 门禁判定。

公式契约（ptq_pipeline.py:137 对齐）：
  traj_l1    = mean|engine_traj - ref_traj|
  score_mae  = mean|engine_scores - ref_scores|
  metric_mae = mean_i( mean|engine_metric_i - ref_metric_i| )   i=0..5
  argmax_match = argmax(engine_scores) == argmax(ref_scores)
  gt_dist    = mean|engine_traj[0][0] - gt_traj|                gt (8,3)

输出名解析：从 dump 目录的 manifest.tsv 按 binding 顺序解析——
dims=1,8,3 的是 trajectory；名字含 traj_scores 的是分数；其余 6 个 (1,400)
按 binding 顺序映射 ref_metric_0..5（make_engine_reference 按引擎输出序落盘）。
"""
import argparse
import json
import os
import sys

import numpy as np


def sanitize(name: str) -> str:
    return name.lstrip("/")


def read_dump_manifest(dump_dir: str):
    """返回 [(name, dtype, dims, file)] 按 manifest 行序（binding 序）。"""
    with open(os.path.join(dump_dir, "manifest.tsv"), "r", encoding="utf-8") as f:
        rows = []
        for ln in f.read().strip().split("\n"):
            name, dtype, dims, file = ln.split("\t")
            rows.append((name, dtype, dims, file))
    return rows


def resolve_outputs(dump_dir: str):
    """按 manifest binding 序解析 8 个输出 -> {traj, scores, metric_0..5}。"""
    rows = read_dump_manifest(dump_dir)
    out = {}
    metric_by_name = {}
    metric_order = []
    for name, dtype, dims, file in rows:
        arr = np.fromfile(os.path.join(dump_dir, sanitize(file)), dtype=np.float32)
        key = sanitize(name)
        if dims == "1,8,3":
            out["trajectory"] = arr.reshape(1, 8, 3)
        elif "traj_scores" in key or key.endswith("scores"):
            out["scores"] = arr.reshape(1, 400)
        elif dims == "1,400":
            metric_order.append((key, arr.reshape(1, 400)))
        else:
            raise AssertionError(f"未知输出张量 name={name} dims={dims}")
    assert "trajectory" in out and "scores" in out, out.keys()
    assert len(metric_order) == 6, f"metric 头数量 {len(metric_order)} != 6"
    # ref_metric_i 按 make_engine_reference 的模型输出序落盘（MNAMES 序）；
    # 引擎 binding 序不保证一致（trtexec 可能按名字排序）——按名字配对，
    # 名字不可识别时才退回 binding 序并告警。
    mnames = ["no_at_fault_collisions", "drivable_area_compliance",
              "driving_direction_compliance", "time_to_collision_within_bound",
              "comfort", "ego_progress"]
    named = 0
    for i, (key, arr) in enumerate(metric_order):
        hit = next((m for m in mnames if m in key), None)
        if hit is not None:
            out[f"metric_{mnames.index(hit)}"] = arr
            named += 1
        else:
            out[f"metric_{i}"] = arr
    if named < 6:
        print(f"[warn] {dump_dir}: 仅 {named}/6 个 metric 头按名识别，其余按 binding 序回退")
    return out


def sample_metrics(eng: dict, ref: dict) -> dict:
    traj_l1 = float(np.abs(eng["trajectory"] - ref["ref_traj"]).mean())
    score_mae = float(np.abs(eng["scores"] - ref["ref_scores"]).mean())
    per_head = [float(np.abs(eng[f"metric_{i}"] - ref[f"ref_metric_{i}"]).mean())
                for i in range(6)]
    metric_mae = float(np.mean(per_head))
    argmax_match = bool(int(np.argmax(eng["scores"])) == int(np.argmax(ref["ref_scores"])))
    # 契约（ptq_pipeline.py:137）：r 为原始输出元组，r[0] = trajectory (1,8,3)，
    # r[0][0] = (8,3) 整条轨迹；对 gt (8,3) 求均值。引擎侧 trajectory 已 reshape
    # 成 (1,8,3)，对应量是 [0]（整条轨迹），不是 [0][0]（首路点）。
    gt_dist = float(np.abs(eng["trajectory"][0] - ref["gt_traj"]).mean())
    return {"traj_l1": traj_l1, "score_mae": score_mae, "metric_mae": metric_mae,
            "argmax_match": argmax_match, "gt_dist": gt_dist,
            "metric_mae_per_head": [round(v, 4) for v in per_head]}


def aggregate(samples: list) -> dict:
    keys = ("traj_l1", "score_mae", "metric_mae", "argmax_match", "gt_dist")
    return {k: float(np.mean([s[k] for s in samples])) for k in keys}


def judge(agg: dict, gates: dict) -> dict:
    return {
        "metric_mae": "PASS" if agg["metric_mae"] <= gates["metric_mae_max"] else "FAIL",
        "traj_l1": "PASS" if agg["traj_l1"] <= gates["traj_l1_max"] else "FAIL",
        "gt_dist": "PASS" if abs(agg["gt_dist"] - gates["gt_dist_fp32"]) <= gates["gt_dist_max_delta"] else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", required=True, help="含 val_XX 子目录的根")
    ap.add_argument("--ref-dir", default=os.path.join("deploy", "artifacts", "engine_ref"))
    ap.add_argument("--gates", default=os.path.join("deploy", "artifacts", "reports", "gates.json"))
    ap.add_argument("--out", default=os.path.join("deploy", "artifacts", "reports", "m2_align.json"))
    args = ap.parse_args()

    with open(args.gates, "r", encoding="utf-8") as f:
        gates = json.load(f)

    dump_dirs = sorted(d for d in os.listdir(args.dump_root)
                       if d.startswith("val_") and os.path.isdir(os.path.join(args.dump_root, d)))
    assert dump_dirs, f"no val_* under {args.dump_root}"

    samples, lines = [], []
    for d in dump_dirs:
        eng = resolve_outputs(os.path.join(args.dump_root, d))
        ref = dict(np.load(os.path.join(args.ref_dir, d + ".npz")))
        m = sample_metrics(eng, ref)
        m["sample"] = d
        samples.append(m)
        lines.append(f"{d} traj_l1={m['traj_l1']:.4f} score_mae={m['score_mae']:.4f} "
                     f"metric_mae={m['metric_mae']:.4f} argmax={int(m['argmax_match'])} "
                     f"gt={m['gt_dist']:.4f}")

    agg = aggregate(samples)
    verdict = judge(agg, gates)
    all_pass = all(v == "PASS" for v in verdict.values())

    rep = {"aggregate": agg, "verdict": verdict, "gates": gates,
           "n_samples": len(samples), "per_sample": samples}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)

    print("\n".join(lines))
    print("==== aggregate ====")
    print({k: round(v, 4) for k, v in agg.items()})
    print("==== gates ====")
    print(gates)
    print("==== verdict ====")
    print(verdict, "=> ALL PASS" if all_pass else "=> GATE FAIL")
    sys.exit(0 if all_pass else 2)


if __name__ == "__main__":
    sys.exit(main())
