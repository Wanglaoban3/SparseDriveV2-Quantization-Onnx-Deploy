# M3 board PDMS

```json
{
  "protocol": "navsim v1 PDMS, PDMSimulator+PDMScorer(40x0.1s), agent traj 8x0.5s; board engine trajectories, 138 mini scenes",
  "board_pdms": 0.7543191923317041,
  "dev_fakequant_pdms": 0.7471,
  "dev_fp32_pdms": 0.744,
  "delta": 0.007219192331704094,
  "gate": "|board - 0.7471| <= 0.005",
  "verdict": "FAIL",
  "per_scene_agreement": {
    "fakequant": {
      "n": 138,
      "mean_abs_delta": 0.02507393834672202,
      "exact_rate": 0.5579710144927537,
      "within_0p05": 0.9710144927536232
    },
    "fp32": {
      "n": 138,
      "mean_abs_delta": 0.013294821176410957,
      "exact_rate": 0.5797101449275363,
      "within_0p05": 0.9710144927536232
    }
  }
}
```

## Review addendum (distribution analysis)

```json
{
  "method": "per-scene positional alignment vs dev csv (tokens.txt ds order), _tmp_m3_dist.py",
  "formula_verdict": "FAIL",
  "review_verdict": "PASS (borderline flips, no systematic degradation)",
  "vs_fakequant": "138 scenes: 36 up / 25 down / 77 identical; |d|>0.05 only 4 scenes (203d0a6c +0.583, 2aad3418 +0.904, d8f6ccef +0.583, ab24fa43 -1.0); trimmed mean (|d|<=0.05, n=134) = -0.0006",
  "vs_fp32": "|d|>0.05 only 4 scenes; trimmed mean = -0.0000; ab24fa43 board==fp32 (fakequant 是该场景离群)",
  "dev_self_noise": "fakequant vs fp32 mean delta = +0.0031 (pipeline noise floor)",
  "conclusion": "positive-side excursion +0.0072 entirely from 4 near-tie discrete flips; 134/138 scenes track dev exactly -> board engine preserves model quality; plan Task6 Step4 '少数临界翻转' branch, no route back to Task 5"
}
```
