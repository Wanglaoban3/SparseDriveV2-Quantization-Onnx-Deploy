"""PDMS-based per-layer sensitivity scan (post-processed end-to-end standard).

Companion to the metric-MAE scan in `ptq_pipeline.py --stage sensitivity`:
same base model (all-INT8 PTQ), same scene set (the 24 held-out samples, never
seen by calibration), but the readout is the **post-processed PDMS** — i.e. the
argmax-selected trajectory is scored by the navsim PDM simulator/scorer, exactly
like the dataset-level evaluations (pdms_eval_quant / pdms_eval_configs).

For each of the 106 quantizable modules:
  exclude it from INT8 -> PDMS_excluded (24 scenes)
  gain_pdms = PDMS_excluded - PDMS_all_int8   (>0: excluding this layer helps)

Usage:
  python deploy/pdms_sensitivity.py [--smoke]
Writes:
  deploy/artifacts/sensitivity_pdms.json / sensitivity_pdms.csv
  (config reference PDMS on the same 24 scenes is included)
Runtime: ~1 h single GPU (110 configs x 24 scenes); --smoke ≈ 2 min.
"""
import argparse
import copy
import json
import os
import sys
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
import modelopt.torch.quantization as mtq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 modules x 3 scenes sanity run")
    args = ap.parse_args()

    cfg, wrapper, ds = pp.build()
    mcache = peq.MetricCacheLoader(Path(peq.METRIC_CACHE))
    token2idx = {t: i for i, t in enumerate(ds.tokens)}
    val_tokens = [ds.tokens[i] for i in pp.VAL_IDX if i < len(ds)]
    assert all(t in mcache.tokens for t in val_tokens), "held-out scenes missing metric cache"
    n_scenes = 3 if args.smoke else len(val_tokens)
    scenes = val_tokens[:n_scenes]
    print(f"held-out scenes for scan: {n_scenes}")

    simulator = peq.PDMSimulator(peq.TrajectorySampling(num_poses=40, interval_length=0.1))
    scorer = peq.PDMScorer(simulator.proposal_sampling)

    def ev(model, tag):
        return peq.eval_model(model, ds, scenes, token2idx, simulator, scorer, mcache, tag)[0]

    print("[ref] fp32...", flush=True)
    fp32_m = ev(wrapper, "fp32")

    def forward_loop(m):
        for i in pp.CALIB_IDX:
            inp, _, _ = pp.to_inputs(ds, i)
            m(*inp)

    print("[ref] all-int8 PTQ...", flush=True)
    student = mtq.quantize(copy.deepcopy(wrapper), mtq.INT8_DEFAULT_CFG, forward_loop)
    student.cuda().eval()
    base_m = ev(student, "allint8")
    base_score = base_m["score"]
    print(f"  fp32={fp32_m['score']:.4f}  all_int8={base_score:.4f}  "
          f"gap={base_score - fp32_m['score']:+.4f}", flush=True)

    groups = pp.quant_groups(student.model)
    if args.smoke:
        groups = groups[:2]
    print(f"[scan] {len(groups)} modules x {n_scenes} scenes ...", flush=True)

    mae_gain = {s["name"]: s["gain"]
                for s in json.load(open(os.path.join(OUT, "sensitivity.json")))}
    rows = []
    for gi, (name, mod) in enumerate(groups):
        pp.set_group(mod, False)
        m = ev(student, name)
        pp.set_group(mod, True)
        rows.append(dict(name=name, pdms_if_excluded=m["score"],
                         gain_pdms=m["score"] - base_score,
                         gain_mae_ref=mae_gain.get(name)))
        if gi % 5 == 0 or gi + 1 == len(groups):
            print(f"  [{gi + 1}/{len(groups)}] {name}: pdms={m['score']:.4f} "
                  f"gain={rows[-1]['gain_pdms']:+.4f}", flush=True)

    result = dict(
        protocol="PDMS sensitivity scan: exclude one module from all-INT8 PTQ, readout = "
                 "post-processed PDMS on held-out scenes (navsim v1 simulator+scorer); "
                 "gain_pdms = PDMS_excluded - PDMS_all_int8",
        n_scenes=n_scenes,
        fp32=fp32_m,
        all_int8=base_m,
        rows=rows,
    )

    # config references on the same scenes (rounded through the same readout)
    full_groups = dict(pp.quant_groups(student.model))
    report = json.load(open(os.path.join(OUT, "ptq_report_final.json")))
    for name in report.get("fallback", []):
        if name in full_groups:
            pp.set_group(full_groups[name], False)
    prot_m = ev(student, "protected")
    if not args.smoke:
        sd = torch.load(os.path.join(OUT, "sparsedrive_int8_final.state_dict.pt"),
                        map_location="cpu")
        student.load_state_dict(sd)
        qat_m = ev(student, "qat_final")
        result["config_reference"] = dict(ptq_protected_top12=prot_m, qat_final=qat_m)

    rows.sort(key=lambda r: -r["gain_pdms"])
    json.dump(result, open(os.path.join(OUT, "sensitivity_pdms.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity_pdms.csv"), index=False)
    print("saved -> sensitivity_pdms.json / sensitivity_pdms.csv", flush=True)
    print("top-10 by gain_pdms:")
    for r in rows[:10]:
        print(f"  {r['name']}: gain_pdms={r['gain_pdms']:+.4f} "
              f"pdms_if_excluded={r['pdms_if_excluded']:.4f} (mae_ref {r['gain_mae_ref']:+.4f})")


if __name__ == "__main__":
    main()
