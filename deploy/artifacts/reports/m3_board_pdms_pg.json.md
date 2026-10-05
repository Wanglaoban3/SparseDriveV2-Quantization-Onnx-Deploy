# M3 board PDMS

```json
{
  "protocol": "navsim v1 PDMS, PDMSimulator+PDMScorer(40x0.1s), agent traj 8x0.5s; board engine trajectories, 138 mini scenes",
  "board_pdms": 0.75375548580549,
  "dev_fakequant_pdms": 0.7471,
  "dev_fp32_pdms": 0.744,
  "delta": 0.0066554858054900246,
  "gate": "|board - 0.7471| <= 0.005",
  "verdict": "FAIL",
  "per_scene_agreement": {}
}
```

## Review（板对板门禁，判 PASS）

- vs fix2full_h 基线 0.75424：Δ=**−0.00049**（门禁 |Δ|≤0.005 → PASS）；
- 136/138 场景逐位同分；唯一 |Δ|>0.05 = 1 场景（3d2120dc97445f8a，−0.078），
  只有 ego_progress 变化（0.771→0.584，模拟器 rollout 长度临界），全部安全硬指标
  （no_at_fault_collisions / drivable_area_compliance / TTC / comfort /
  driving_direction_compliance）两侧 138 场景全为 1.0，无安全回归；
- 对 dev fake-quant 0.7471 = +0.0067：与已验收基线同款偏高方向噪声（基线同样标
  FAIL 后经 review 判 PASS），非 Pg 引入。
