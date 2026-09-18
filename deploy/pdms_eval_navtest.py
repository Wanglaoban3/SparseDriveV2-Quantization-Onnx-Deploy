"""navtest (full OpenScene test split, 136 logs / ~12k scenes) PDMS evaluation of all 4 configs.

Same protocol as pdms_eval_quant.py (navsim v1 PDMSimulator + PDMScorer). The quantized
candidates are calibrated on the SAME mini-split calibration samples as the delivered QDQ
ONNX (exp/data_cache_mini, pp.CALIB_IDX) — only the evaluation split changes.

Usage (one long run; resume-safe, checkpoints every 200 scenes):
  python deploy/pdms_eval_navtest.py                    # all 4 configs sequentially
  python deploy/pdms_eval_navtest.py --config fp32,qat  # subset
  python deploy/pdms_eval_navtest.py --limit 8          # smoke test on first 8 scenes
  python deploy/pdms_eval_navtest.py --report           # print merged table only
Requires exp/data_cache_navtest (feature cache) + SD_METRIC_CACHE (default
exp/metric_cache_navtestv1). Writes:
  deploy/artifacts/pdms_navtest_report.json
  deploy/artifacts/pdms_navtest_<config>.csv   (+ .partial.csv while running)
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

import pandas as pd
import torch

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = os.environ.get("TRT_DEPLOY_DIR", os.path.join(DEPLOY_DIR, "artifacts"))

import ptq_pipeline as pp
import pdms_eval_quant as peq
import pdms_eval_configs as pec
from navsim.navsim_v1.common.dataloader import MetricCacheLoader
from navsim.navsim_v1.evaluate.pdm_score import pdm_score

NAVTEST_CACHE = os.path.join(ROOT, "exp", "data_cache_navtest")
METRIC_CACHE = os.environ.get("SD_METRIC_CACHE",
                              os.path.join(ROOT, "exp", "metric_cache_navtestv1"))
CHECKPOINT_EVERY = 200


def navtest_logs():
    """136 log names from the navtest scene filter yaml (no yaml dep needed)."""
    p = os.path.join(ROOT, "navsim", "planning", "script", "config", "common",
                     "train_test_split", "scene_filter", "navtest.yaml")
    logs, in_block = [], False
    for line in open(p):
        s = line.strip()
        if s.startswith("log_names:"):
            in_block = True
            continue
        if in_block:
            if s.startswith("- "):
                # yaml entries may be quoted: strip surrounding ' or "
                logs.append(s[2:].strip().strip("'\""))
            elif s:
                break
    assert logs, f"no log_names parsed from {p}"
    return logs


def make_ds(cfg, cache_path, log_names):
    return pp.CacheOnlyDataset(cache_path=cache_path,
                               feature_builders=[pp.SparseDriveFeatureBuilder(cfg)],
                               target_builders=[pp.SparseDriveTargetBuilder(cfg)],
                               log_names=log_names)


@torch.no_grad()
def eval_navtest(model, ds, tokens, token2idx, simulator, scorer, mcache, tag,
                 limit=None):
    """Resume-safe eval: skips tokens already in the partial CSV, checkpoints it."""
    partial = os.path.join(OUT, f"pdms_navtest_{tag}.partial.csv")
    done, rows = set(), []
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        rows = prev.to_dict("records")
        done = set(prev["token"])
        print(f"  [{tag}] resuming: {len(done)} scenes already done", flush=True)
    if limit:
        tokens = tokens[:limit]
    todo = [t for t in tokens if t not in done]
    t0, n0 = time.time(), len(todo)
    print(f"  [{tag}] {n0} scenes to run", flush=True)
    for n, tok in enumerate(todo):
        inp, _, _ = pp.to_inputs(ds, token2idx[tok])
        out = model(*inp)
        poses = out[0][0].float().cpu().numpy()
        trajectory = peq.Trajectory(poses=poses, trajectory_sampling=peq.FUT_SAMPLING)
        with lzma.open(mcache.metric_cache_paths[tok], "rb") as f:
            cache = pickle.load(f)
        r = pdm_score(metric_cache=cache, model_trajectory=trajectory,
                      future_sampling=simulator.proposal_sampling,
                      simulator=simulator, scorer=scorer)
        d = dataclasses.asdict(r)
        d["token"] = tok
        rows.append(d)
        if (n + 1) % 50 == 0 or n + 1 == n0:
            cur = pd.DataFrame(rows)["score"].mean()
            rate = (time.time() - t0) / (n + 1)
            eta_min = rate * (n0 - n - 1) / 60
            print(f"  [{tag}] {n + 1}/{n0}  running PDMS={cur:.4f}  "
                  f"{rate:.2f} s/scene  eta {eta_min:.0f} min", flush=True)
        if (n + 1) % CHECKPOINT_EVERY == 0 and not limit:
            pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows)
    if not limit:
        df.to_csv(partial, index=False)
    if "token" not in df.columns:
        return {}, df
    return {k: float(v) for k, v in df.drop(columns=["token"]).mean().items()}, df


def build_model(tag, wrapper, ds_mini):
    if tag == "fp32":
        return wrapper
    if tag == "qat":
        return peq.rebuild_quantized(wrapper, ds_mini)
    ptq, protected = pec.build_models(wrapper, ds_mini)
    return {"ptq": ptq, "protected": protected}[tag]


def merge_report(tag, means, n_scenes):
    path = os.path.join(OUT, "pdms_navtest_report.json")
    rep = json.load(open(path)) if os.path.exists(path) else {}
    cfgs = rep.get("configs", {})
    cfgs[tag] = means
    out = dict(
        protocol="navsim v1 PDMS, PDMSimulator+PDMScorer(40x0.1s), agent traj 8x0.5s, "
                 f"navtest (OpenScene test split, {n_scenes} scenes); quant candidates "
                 "calibrated on the mini split like the delivered QDQ ONNX",
        n_scenes=n_scenes,
        configs=cfgs,
    )
    if "fp32" in cfgs:
        out["deltas_vs_fp32"] = {k: cfgs[k]["score"] - cfgs["fp32"]["score"]
                                 for k in cfgs}
    json.dump(out, open(path, "w"), indent=2)
    print("saved ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fp32,ptq,protected,qat",
                    help="comma list of: fp32,ptq,protected,qat")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: only first N scenes")
    ap.add_argument("--report", action="store_true", help="print merged table only")
    args = ap.parse_args()

    if args.report:
        rep = json.load(open(os.path.join(OUT, "pdms_navtest_report.json")))
        for k, v in rep["configs"].items():
            print(f"{k:24s} PDMS={v['score']:.4f}")
        print("deltas:", json.dumps(rep.get("deltas_vs_fp32", {}), indent=1))
        return

    logs = navtest_logs()
    print(f"navtest logs: {len(logs)}", flush=True)
    cfg, wrapper, ds_mini = pp.build()
    ds = make_ds(cfg, NAVTEST_CACHE, logs)
    mcache = MetricCacheLoader(Path(METRIC_CACHE))
    token2idx = {t: i for i, t in enumerate(ds.tokens)}
    tokens = [t for t in ds.tokens if t in mcache.tokens]
    print(f"scenes with feature cache + metric cache: {len(tokens)}", flush=True)

    simulator = peq.PDMSimulator(peq.TrajectorySampling(num_poses=40, interval_length=0.1))
    scorer = peq.PDMScorer(simulator.proposal_sampling)

    for tag in [t.strip() for t in args.config.split(",")]:
        print(f"[{tag}] building model...", flush=True)
        model = build_model(tag, wrapper, ds_mini)
        print(f"[{tag}] evaluating on navtest...", flush=True)
        means, df = eval_navtest(model, ds, tokens, token2idx, simulator, scorer,
                                 mcache, tag, args.limit or None)
        print(f"  [{tag}] PDMS={means['score']:.4f}", json.dumps(means), flush=True)
        if args.limit:
            continue
        final = os.path.join(OUT, f"pdms_navtest_{tag}.csv")
        df.to_csv(final, index=False)
        partial = os.path.join(OUT, f"pdms_navtest_{tag}.partial.csv")
        if os.path.exists(partial):
            os.remove(partial)
        merge_report(tag, means, len(tokens))


if __name__ == "__main__":
    main()
