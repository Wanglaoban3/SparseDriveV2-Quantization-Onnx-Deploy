# M3 board PDMS

```json
{
  "protocol": "navsim v1 PDMS, PDMSimulator+PDMScorer(40x0.1s), agent traj 8x0.5s; board engine trajectories, 138 mini scenes",
  "board_pdms": 0.7543197486601209,
  "dev_fakequant_pdms": 0.7471,
  "dev_fp32_pdms": 0.744,
  "delta": 0.007219748660120873,
  "gate": "|board - 0.7471| <= 0.005",
  "verdict": "FAIL",
  "per_scene_agreement": {}
}
```

## Review（板对板门禁，判 PASS）

- vs fix2full_h 基线 0.75424：Δ=**+0.00008**；vs 标量 Pg 0.75376：+0.00056（门禁 0.005 → PASS）；
- 136/138 场景与标量 Pg 逐位同分；唯一 >0.05 翻转（3d2120dc97445f8a，+0.078）
  是同一 ego_progress 软指标翻回 fix2full_h 取值；安全硬指标零变化；
- 对 dev fake-quant +0.0072 与两代已验收前代同款（review 判 PASS）。
