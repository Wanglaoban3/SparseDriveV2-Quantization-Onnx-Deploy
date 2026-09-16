"""Dataset-level (PDMS) evaluation of the quantization configs on the full mini split.

Adds the two missing configs to the PDMS evidence chain:
  ptq_all_int8        — ModelOpt INT8, no fallback            (measured here, 138 scenes)
  ptq_protected_top12 — INT8 + Top-12 sensitive-layer fallback (measured here, 138 scenes)
  fp32 / qat_final    — already measured in pdms_report.json    (reused as-is)

Usage:
  python deploy/pdms_eval_configs.py [--limit N]
Writes:
  deploy/artifacts/pdms_configs_report.json  (merged 4-config table)
  deploy/artifacts/pdms_ptq_all_int8.csv / pdms_ptq_protected_top12.csv (per-scene)
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = os.environ.get("TRT_DEPLOY_DIR", os.path.join(DEPLOY_DIR, "artifacts"))

import ptq_pipeline as pp
import pdms_eval_quant as peq
import modelopt.torch.quantization as mtq


def build_models(wrapper, ds):
    def forward_loop(m):
        for i in pp.CALIB_IDX:
            inp, _, _ = pp.to_inputs(ds, i)
            m(*inp)

    ptq = mtq.quantize(copy.deepcopy(wrapper), mtq.INT8_DEFAULT_CFG, forward_loop)
    ptq.cuda().eval()
    protected = copy.deepcopy(ptq)
    report = json.load(open(os.path.join(OUT, "ptq_report_final.json")))
    groups = dict(pp.quant_groups(protected.model))
    n_off = 0
    for name in report.get("fallback", []):
        if name in groups:
            pp.set_group(groups[name], False)
            n_off += 1
    print(f"protected model: {n_off} fallback modules disabled", flush=True)
    return ptq, protected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: only first N scenes")
    args = ap.parse_args()

    cfg, wrapper, ds = pp.build()
    mcache = peq.MetricCacheLoader(Path(peq.METRIC_CACHE))
    token2idx = {t: i for i, t in enumerate(ds.tokens)}
    tokens = [t for t in ds.tokens if t in mcache.tokens]
    print(f"scenes with feature cache + metric cache: {len(tokens)}")

    simulator = peq.PDMSimulator(peq.TrajectorySampling(num_poses=40, interval_length=0.1))
    scorer = peq.PDMScorer(simulator.proposal_sampling)

    ptq, protected = build_models(wrapper, ds)
    out = {}
    for tag, model in [("ptq_all_int8", ptq), ("ptq_protected_top12", protected)]:
        print(f"[{tag}] full split...", flush=True)
        means, df = peq.eval_model(model, ds, tokens, token2idx, simulator, scorer, mcache, tag,
                                   args.limit or None)
        out[tag] = means
        print(f"  {tag}:", json.dumps(means), flush=True)
        if not args.limit:
            df.to_csv(os.path.join(OUT, f"pdms_{tag}.csv"), index=False)
    if not args.limit:
        prev = json.load(open(os.path.join(OUT, "pdms_report.json")))
        out["fp32"] = prev["fp32"]
        out["qat_final"] = prev["int8_qat"]
        json.dump(dict(
            protocol="navsim v1 PDMS, PDMSimulator+PDMScorer(40x0.1s), agent traj 8x0.5s, "
                     f"{len(tokens)} mini scenes; fp32/qat_final reused from pdms_report.json",
            n_scenes=len(tokens),
            configs=out,
            deltas_vs_fp32={k: out[k]["score"] - out["fp32"]["score"] for k in out},
        ), open(os.path.join(OUT, "pdms_configs_report.json"), "w"), indent=2)
        print("saved ->", os.path.join(OUT, "pdms_configs_report.json"))


if __name__ == "__main__":
    main()
