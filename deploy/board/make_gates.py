# -*- coding: utf-8 -*-
"""从 kd_qat_report.json 的 final 块（plain 配置 24 样本 fake-quant）生成门禁文件。

门禁规则（spec §7 / 计划 Global Constraints）：
  metric_mae_max = metric_mae_ref * 1.25
  traj_l1_max    = traj_l1_ref + 0.03   (ref + 3cm)
  gt_dist        以 FP32 锚点 fp32_gt 为基准，板端 |gt_dist - fp32_gt| <= 0.06
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "deploy", "artifacts", "kd_qat_report.json")
DST = os.path.join(ROOT, "deploy", "artifacts", "reports", "gates.json")


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        rep = json.load(f)
    final = rep["final"]
    fp32_gt = float(rep["fp32_gt"])

    gates = {
        "metric_mae_ref": round(float(final["metric_mae"]), 4),
        "metric_mae_max": round(float(final["metric_mae"]) * 1.25, 4),
        "traj_l1_ref": round(float(final["traj_l1"]), 4),
        "traj_l1_max": round(float(final["traj_l1"]) + 0.03, 4),
        "score_mae_ref": round(float(final["score_mae"]), 4),
        "argmax_ref": round(float(final["argmax_match"]), 4),
        "gt_dist_fp32": round(fp32_gt, 4),
        "gt_dist_max_delta": 0.06,
    }

    # 硬断言：数值必须落在计划钉死的量级内（防止误读别的配置块）
    assert abs(gates["metric_mae_ref"] - 0.6556) < 0.001, gates
    assert abs(gates["traj_l1_ref"] - 0.0950) < 0.001, gates
    assert abs(gates["score_mae_ref"] - 0.4499) < 0.001, gates
    assert abs(gates["gt_dist_fp32"] - 1.0923) < 0.001, gates

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(gates, f, indent=2, ensure_ascii=False)
    print(json.dumps(gates, indent=2))
    print(f"written: {DST}")


if __name__ == "__main__":
    sys.exit(main())
