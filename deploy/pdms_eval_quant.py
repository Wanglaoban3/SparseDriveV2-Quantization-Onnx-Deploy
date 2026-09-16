"""End-to-end PDMS comparison: FP32 vs final INT8 (fake-quant) on the same mini split.

Protocol mirrors navsim run_pdm_score_navtest_v1_fast exactly (PDMSimulator + PDMScorer,
proposal_sampling 40x0.1s, agent trajectory 8x0.5s). The quantized model is the torch-side
source of the delivered QDQ ONNX: ModelOpt INT8 + Top-12 sensitive-layer fallback + KD-QAT
weights (artifacts/sparsedrive_int8_final.state_dict.pt).

Usage:
  python deploy/pdms_eval_quant.py [--limit N]
Outputs:
  deploy/artifacts/pdms_report.json          (summary, both models)
  deploy/artifacts/pdms_fp32.csv / pdms_int8_qat.csv  (per-scene rows)
"""
import argparse
import copy
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
import modelopt.torch.quantization as mtq
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.navsim_v1.common.dataclasses import Trajectory
from navsim.navsim_v1.common.dataloader import MetricCacheLoader
from navsim.navsim_v1.evaluate.pdm_score import pdm_score
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer

METRIC_CACHE = os.environ.get("SD_METRIC_CACHE", os.path.join(ROOT, "exp", "metric_cache_mini_v1"))
FUT_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)


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
        if (n + 1) % 10 == 0 or n + 1 == len(tokens):
            cur = pd.DataFrame(rows)["score"].mean()
            print(f"  [{tag}] {n + 1}/{len(tokens)}  running PDMS={cur:.4f}  "
                  f"({(time.time() - t0) / (n + 1):.2f} s/scene)", flush=True)
    df = pd.DataFrame(rows)
    return {k: float(v) for k, v in df.mean().items()}, df


def rebuild_quantized(wrapper, ds):
    """The exact final quantized model: INT8 + Top-12 fallback + KD-QAT weights."""
    def forward_loop(m):
        for i in pp.CALIB_IDX:
            inp, _, _ = pp.to_inputs(ds, i)
            m(*inp)

    student = mtq.quantize(copy.deepcopy(wrapper), mtq.INT8_DEFAULT_CFG, forward_loop)
    student.cuda().eval()
    report = json.load(open(os.path.join(OUT, "ptq_report_final.json")))
    groups = dict(pp.quant_groups(student.model))
    n_off = 0
    for name in report.get("fallback", []):
        if name in groups:
            pp.set_group(groups[name], False)
            n_off += 1
    sd = torch.load(os.path.join(OUT, "sparsedrive_int8_final.state_dict.pt"), map_location="cpu")
    student.load_state_dict(sd)
    print(f"  quantized model ready: int8 + {n_off} fallback modules, QAT weights loaded")
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: only first N scenes")
    args = ap.parse_args()

    cfg, wrapper, ds = pp.build()
    mcache = MetricCacheLoader(Path(METRIC_CACHE))
    token2idx = {t: i for i, t in enumerate(ds.tokens)}
    tokens = [t for t in ds.tokens if t in mcache.tokens]
    print(f"scenes with feature cache + metric cache: {len(tokens)}")

    simulator = PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1))
    scorer = PDMScorer(simulator.proposal_sampling)

    print("[1/2] FP32 pass...")
    fp32_means, fp32_df = eval_model(wrapper, ds, tokens, token2idx, simulator, scorer,
                                     mcache, "fp32", args.limit or None)
    print("  FP32:", json.dumps(fp32_means, indent=1))

    print("[2/2] INT8+fallback+QAT (final deliverable, fake-quant) pass...")
    student = rebuild_quantized(wrapper, ds)
    q_means, q_df = eval_model(student, ds, tokens, token2idx, simulator, scorer,
                               mcache, "int8_qat", args.limit or None)
    print("  INT8+QAT:", json.dumps(q_means, indent=1))

    if not args.limit:
        fp32_df.to_csv(os.path.join(OUT, "pdms_fp32.csv"), index=False)
        q_df.to_csv(os.path.join(OUT, "pdms_int8_qat.csv"), index=False)
        json.dump(dict(
            protocol="navsim v1 PDMS, PDMSimulator+PDMScorer(40x0.1s), agent traj 8x0.5s, "
                     f"{len(tokens)} mini scenes; quant model = INT8+Top12fallback+KD-QAT fake-quant",
            n_scenes=len(tokens),
            fp32=fp32_means,
            int8_qat=q_means,
            deltas={k: q_means[k] - fp32_means[k] for k in fp32_means},
        ), open(os.path.join(OUT, "pdms_report.json"), "w"), indent=2)
        print("saved ->", os.path.join(OUT, "pdms_report.json"))


if __name__ == "__main__":
    main()
