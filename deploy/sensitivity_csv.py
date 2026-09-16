"""Convert artifacts/sensitivity.json (per-module quantization sensitivity scan)
into a human-readable CSV, sorted by metric gain (how much metric MAE improves
when this module is excluded from INT8).

Columns:
  rank       - 1 = most sensitive (highest gain)
  module     - torch module name in the SparseDriveModel
  gain       - metric_mae(all-int8) - metric_mae(all-int8 except this module)
               >0 : excluding this module REDUCES drift (it is sensitive)
  drift      - end-to-end metric MAE measured while this module is excluded
  argmax     - trajectory argmax agreement vs FP32 while this module is excluded
  traj_l1    - trajectory L1 (m) while this module is excluded
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "artifacts", "sensitivity.json")
DST = os.path.join(HERE, "artifacts", "sensitivity.csv")

sens = json.load(open(SRC, encoding="utf-8"))
sens.sort(key=lambda x: -x["gain"])
with open(DST, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["rank", "module", "gain(metric_mae)", "drift_if_off(metric_mae)",
                "argmax_match_if_off", "traj_l1_if_off(m)"])
    for i, s in enumerate(sens, 1):
        w.writerow([i, s["name"], "%.6f" % s["gain"], "%.6f" % s["drift"],
                    "%.4f" % s["argmax"], "%.6f" % s["traj_l1"]])
print("wrote %s (%d modules)" % (DST, len(sens)))
