# -*- coding: utf-8 -*-
"""Score board-dumped trajectories with the navsim PDMS chain (Task 6 Step 3).

Protocol mirrors pdms_eval_quant.eval_model exactly (PDMSimulator+PDMScorer,
proposal_sampling 40x0.1s, agent trajectory 8x0.5s), but consumes the board's
trajectory.bin files instead of running a model. No GPU/model needed.

Usage (navsim env):
  python deploy/eval_board_pdms.py --traj-dir deploy/artifacts/board_pdms_traj \
      --out deploy/artifacts/reports/m3_board_pdms
Outputs: <out>.csv (per scene), <out>.json (summary + gate), <out>.md
"""
import argparse
import dataclasses
import json
import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402
from navsim.navsim_v1.common.dataclasses import Trajectory  # noqa: E402
from navsim.navsim_v1.common.dataloader import MetricCacheLoader  # noqa: E402
from navsim.navsim_v1.evaluate.pdm_score import pdm_score  # noqa: E402
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer  # noqa: E402

METRIC_CACHE = os.environ.get("SD_METRIC_CACHE", os.path.join(ROOT, "exp", "metric_cache_mini_v1"))
ART = os.path.join(DEPLOY_DIR, "artifacts")
DEV_FAKEQUANT_PDMS = 0.7471
DEV_FP32_PDMS = 0.7440
GATE_DELTA = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, help="含 <token>/trajectory.bin 的根目录")
    ap.add_argument("--out", default=os.path.join(ART, "reports", "m3_board_pdms"))
    args = ap.parse_args()

    mcache = MetricCacheLoader(Path(METRIC_CACHE))
    traj_root = Path(args.traj_dir)
    # 行序必须 = ds.tokens 迭代序（dev csv 无 token 列，按行位对齐），tokens.txt 由
    # dump_board_inputs.py 落盘；缺失时退化为字母序并禁用 dev 对齐。
    tokens_file = traj_root / "tokens.txt"
    if tokens_file.exists():
        tokens = [t.strip() for t in tokens_file.read_text().splitlines() if t.strip()]
        assert len(tokens) == 138 and all((traj_root / t / "trajectory.bin").exists() for t in tokens), \
            "tokens.txt 与 trajectory.bin 目录不一致"
    else:
        tokens = sorted(d.name for d in traj_root.iterdir()
                        if d.is_dir() and (d / "trajectory.bin").exists())
        tokens = [t for t in tokens if t in mcache.tokens]
        print("WARN: no tokens.txt, using sorted order; dev positional alignment disabled")
    print(f"scenes to score: {len(tokens)}")

    simulator = PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1))
    scorer = PDMScorer(simulator.proposal_sampling)
    fut_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    rows = []
    for n, tok in enumerate(tokens):
        poses = np.fromfile(traj_root / tok / "trajectory.bin", dtype=np.float32).reshape(8, 3)
        trajectory = Trajectory(poses=poses, trajectory_sampling=fut_sampling)
        with lzma.open(mcache.metric_cache_paths[tok], "rb") as f:
            cache = pickle.load(f)
        r = pdm_score(metric_cache=cache, model_trajectory=trajectory,
                      future_sampling=simulator.proposal_sampling,
                      simulator=simulator, scorer=scorer)
        rows.append(dataclasses.asdict(r))
        if (n + 1) % 20 == 0 or n + 1 == len(tokens):
            print(f"  [{n + 1}/{len(tokens)}] running PDMS={pd.DataFrame(rows)['score'].mean():.4f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "token", tokens)
    board_pdms = float(df["score"].mean())

    # dev per-scene agreement（fakequant / fp32 dev runs）。dev csv 无 token 列，
    # 行序 = pdms_eval_quant 的 ds.tokens 迭代序 == tokens.txt 序 → 按行位对齐。
    agree = {}
    for tag, csv in (("fakequant", os.path.join(ART, "pdms_int8_qat.csv")),
                     ("fp32", os.path.join(ART, "pdms_fp32.csv"))):
        if not (os.path.exists(csv) and tokens_file.exists()):
            continue
        dev = pd.read_csv(csv)
        if "token" in dev.columns:
            m = df.merge(dev[["token", "score"]], on="token", suffixes=("_board", "_dev"))
        else:
            assert len(dev) == len(tokens), \
                f"dev csv rows {len(dev)} != tokens {len(tokens)}, 不能按行位对齐"
            m = df.copy()
            m["score_dev"] = dev["score"].values[:len(m)]
        agree[tag] = dict(
            n=len(m),
            mean_abs_delta=float((m["score"] - m["score_dev"]).abs().mean()),
            exact_rate=float((m["score"] - m["score_dev"]).abs().lt(1e-6).mean()),
            within_0p05=float((m["score"] - m["score_dev"]).abs().le(0.05).mean()),
        )

    delta = board_pdms - DEV_FAKEQUANT_PDMS
    summary = dict(
        protocol="navsim v1 PDMS, PDMSimulator+PDMScorer(40x0.1s), agent traj 8x0.5s; "
                 f"board engine trajectories, {len(tokens)} mini scenes",
        board_pdms=board_pdms,
        dev_fakequant_pdms=DEV_FAKEQUANT_PDMS,
        dev_fp32_pdms=DEV_FP32_PDMS,
        delta=delta,
        gate=f"|board - {DEV_FAKEQUANT_PDMS}| <= {GATE_DELTA}",
        verdict="PASS" if abs(delta) <= GATE_DELTA else "FAIL",
        per_scene_agreement=agree,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out + ".csv", index=False)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(args.out + ".md", "w", encoding="utf-8", newline="\n") as f:
        f.write("# M3 board PDMS\n\n```json\n%s\n```\n" % json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"EVAL_PDMS_DONE -> {args.out}.{{csv,json,md}}")


if __name__ == "__main__":
    main()
