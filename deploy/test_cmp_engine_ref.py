# -*- coding: utf-8 -*-
"""cmp_engine_ref 指标函数单元测试（standalone，先于实现运行必须失败）。

公式契约（ptq_pipeline.py:137 逐字对齐 + 计划 Task 5 Step 2）：
- traj_l1    = mean|traj - ref_traj|          (元素级)
- score_mae  = mean|scores - ref_scores|
- metric_mae = mean_i( mean|metric_i - ref_metric_i| )   6 头先各自 mean 再平均
- argmax_match = argmax(traj_scores) == argmax(ref_scores)
- gt_dist    = mean|traj[0][0] - gt|           gt 是 (8,3)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mk():
    rng = np.random.RandomState(7)
    ref = {
        "ref_traj": rng.rand(1, 8, 3).astype(np.float32),
        "ref_scores": rng.rand(1, 400).astype(np.float32),
        **{f"ref_metric_{i}": rng.rand(1, 400).astype(np.float32) for i in range(6)},
        "gt_traj": rng.rand(8, 3).astype(np.float32),
    }
    eng = {
        "trajectory": ref["ref_traj"] + np.float32(0.01),          # 全体 +0.01
        "scores": ref["ref_scores"] + np.float32(0.001),
        **{f"metric_{i}": ref[f"ref_metric_{i}"] + np.float32(0.02) for i in range(6)},
    }
    return eng, ref


def test_perfect_match():
    from deploy import cmp_engine_ref as cmp_mod
    eng, ref = _mk()
    eng["trajectory"] = ref["ref_traj"].copy()
    eng["scores"] = ref["ref_scores"].copy()
    for i in range(6):
        eng[f"metric_{i}"] = ref[f"ref_metric_{i}"].copy()
    m = cmp_mod.sample_metrics(eng, ref)
    assert m["traj_l1"] == 0.0 and m["score_mae"] == 0.0 and m["metric_mae"] == 0.0
    assert m["argmax_match"] is True
    assert m["gt_dist"] == float(np.abs(ref["ref_traj"][0][0] - ref["gt_traj"]).mean())


def test_known_offsets():
    from deploy import cmp_engine_ref as cmp_mod
    eng, ref = _mk()
    m = cmp_mod.sample_metrics(eng, ref)
    assert abs(m["traj_l1"] - 0.01) < 1e-6, m
    assert abs(m["score_mae"] - 0.001) < 1e-6, m
    assert abs(m["metric_mae"] - 0.02) < 1e-6, m
    # gt_dist 用 engine 轨迹（ref+0.01）
    expect = float(np.abs((ref["ref_traj"][0][0] + np.float32(0.01)) - ref["gt_traj"]).mean())
    assert abs(m["gt_dist"] - expect) < 1e-6


def test_aggregate_and_gates():
    from deploy import cmp_engine_ref as cmp_mod
    agg = cmp_mod.aggregate([{"traj_l1": 0.1, "score_mae": 0.2, "metric_mae": 0.7,
                              "argmax_match": True, "gt_dist": 1.10},
                             {"traj_l1": 0.3, "score_mae": 0.4, "metric_mae": 1.0,
                              "argmax_match": False, "gt_dist": 1.20}])
    assert abs(agg["traj_l1"] - 0.2) < 1e-9
    assert abs(agg["metric_mae"] - 0.85) < 1e-9
    assert abs(agg["argmax_match"] - 0.5) < 1e-9
    gates = {"metric_mae_max": 0.8195, "traj_l1_max": 0.125,
             "gt_dist_fp32": 1.0923, "gt_dist_max_delta": 0.06}
    verdict = cmp_mod.judge(agg, gates)
    assert verdict["metric_mae"] == "FAIL" and verdict["traj_l1"] == "FAIL"
    assert verdict["gt_dist"] == "PASS"  # 1.15 - 1.0923 = 0.0577 <= 0.06


if __name__ == "__main__":
    fns = [test_perfect_match, test_known_offsets, test_aggregate_and_gates]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            failed += 1
            print("ERROR", fn.__name__, "->", type(e).__name__, e)
    print(f"{len(fns) - failed}/{len(fns)} pass")
    sys.exit(1 if failed else 0)
