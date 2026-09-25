"""Paired PDMS A/B on the navsim mini split: base IL checkpoint vs RL-finetuned checkpoint.

Same protocol as deploy/pdms_eval_quant.py (which produced the 0.7440 FP32 baseline):
navsim v1 PDMS, PDMSimulator+PDMScorer(40x0.1s), agent trajectory 8x0.5s, 138 mini scenes,
identical scene list and identical deploy graph optimizations for both models — only the
weights differ (RL s0+s1 fine-tuned traj_mlp + metric_heads).

Usage:
  python deploy/pdms_eval_finetuned.py [--limit N]
Outputs (deploy/artifacts/):
  pdms_finetuned_report.json      summary + per-metric means + deltas
  pdms_base.csv / pdms_finetuned.csv / pdms_finetuned_delta.csv   per-scene rows
"""
import argparse
import dataclasses
import json
import lzma
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = os.environ.get("TRT_DEPLOY_DIR", os.path.join(DEPLOY_DIR, "artifacts"))

import ptq_pipeline as pp
import export_onnx as eo
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.navsim_v1.common.dataclasses import Trajectory
from navsim.navsim_v1.common.dataloader import MetricCacheLoader
from navsim.navsim_v1.evaluate.pdm_score import pdm_score
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder, SparseDriveTargetBuilder
from navsim.planning.training.dataset import CacheOnlyDataset

METRIC_CACHE = os.environ.get("SD_METRIC_CACHE", os.path.join(ROOT, "exp", "metric_cache_mini_v1"))
BASE_CKPT = os.path.join(ROOT, "ckpt", "sparsedrive_navsimv1_92p2.ckpt")
RL_CKPT = os.environ.get(
    "SD_RL_CKPT",
    os.path.join(ROOT, "exp", "sparsedrive_rl_s0s1_mini_full", "2026.09.25.15.46.04", "periodic_pdm_ckpts", "ep0003.ckpt"),
)
FUT_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)


def build_with_ckpt(ckpt_path: str):
    """Identical to pp.build() except the checkpoint — same config, same deploy graph."""
    cfg = SparseDriveConfig()
    cfg.dataset_version = "v1"
    cfg.metrics = list(eo.METRICS)
    cfg.velocity_filter_num = [64, 20]
    cfg.path_filter_num = [128, 20]
    agent = SparseDriveAgent(config=cfg, lr=1e-4, checkpoint_path=ckpt_path)
    agent.initialize()
    model = agent._sparsedrive_model.cuda().eval()
    eo.apply_deploy_optimizations(model)
    wrapper = eo.ExportModel(model).cuda().eval()
    ds = CacheOnlyDataset(cache_path=os.path.join(ROOT, "exp", "data_cache_mini"),
                          feature_builders=[SparseDriveFeatureBuilder(cfg)],
                          target_builders=[SparseDriveTargetBuilder(cfg)],
                          log_names=pp.LOGS)
    return wrapper, ds


@torch.no_grad()
def eval_model(model, ds, tokens, token2idx, simulator, scorer, mcache, tag, limit=None):
    if limit:
        tokens = tokens[:limit]
    rows, t0 = [], time.time()
    for n, tok in enumerate(tokens):
        inp, gt, _ = pp.to_inputs(ds, token2idx[tok])
        out = model(*inp)
        poses = out[0][0].float().cpu().numpy()
        trajectory = Trajectory(poses=poses, trajectory_sampling=FUT_SAMPLING)
        with lzma.open(mcache.metric_cache_paths[tok], "rb") as f:
            cache = pickle.load(f)
        r = pdm_score(metric_cache=cache, model_trajectory=trajectory,
                      future_sampling=simulator.proposal_sampling,
                      simulator=simulator, scorer=scorer)
        rows.append(dataclasses.asdict(r))
        if (n + 1) % 25 == 0 or n + 1 == len(tokens):
            cur = pd.DataFrame(rows)["score"].mean()
            print(f"  [{tag}] {n + 1}/{len(tokens)}  running PDMS={cur:.4f}  "
                  f"({(time.time() - t0) / (n + 1):.2f} s/scene)", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: only first N scenes")
    args = ap.parse_args()

    print(f"base ckpt: {BASE_CKPT}")
    print(f"rl   ckpt: {RL_CKPT}")

    mcache = MetricCacheLoader(Path(METRIC_CACHE))
    simulator = PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1))
    scorer = PDMScorer(simulator.proposal_sampling)

    print("[1/2] base IL model...")
    wrapper_base, ds = build_with_ckpt(BASE_CKPT)
    token2idx = {t: i for i, t in enumerate(ds.tokens)}
    tokens = [t for t in ds.tokens if t in mcache.tokens]
    print(f"scenes with feature cache + metric cache: {len(tokens)}")
    df_base = eval_model(wrapper_base, ds, tokens, token2idx, simulator, scorer, mcache, "base", args.limit or None)

    print("[2/2] RL-finetuned model (s0+s1)...")
    wrapper_rl, _ = build_with_ckpt(RL_CKPT)
    df_rl = eval_model(wrapper_rl, ds, tokens, token2idx, simulator, scorer, mcache, "rl", args.limit or None)

    base_means = {k: float(v) for k, v in df_base.mean().items()}
    rl_means = {k: float(v) for k, v in df_rl.mean().items()}

    delta = df_rl.copy()
    for col in df_base.columns:
        delta[col + "_delta"] = df_rl[col] - df_base[col]
    d = delta["score_delta"]
    summary = dict(
        n_scenes=int(len(d)),
        base=base_means,
        rl=rl_means,
        deltas={k: rl_means[k] - base_means[k] for k in base_means},
        scene_diff=dict(
            identical=int((d.abs() < 1e-9).sum()),
            improved=int((d > 1e-9).sum()),
            degraded=int((d < -1e-9).sum()),
        ),
        worst_scenes=d.nsmallest(5).index.tolist(),
        best_scenes=d.nlargest(5).index.tolist(),
    )
    print(json.dumps(summary, indent=1))

    if not args.limit:
        df_base.to_csv(os.path.join(OUT, "pdms_base.csv"), index=False)
        df_rl.to_csv(os.path.join(OUT, "pdms_finetuned.csv"), index=False)
        delta.to_csv(os.path.join(OUT, "pdms_finetuned_delta.csv"), index=False)
        json.dump(summary, open(os.path.join(OUT, "pdms_finetuned_report.json"), "w"), indent=2)
        print("saved ->", os.path.join(OUT, "pdms_finetuned_report.json"))


if __name__ == "__main__":
    main()
