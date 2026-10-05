# DFA v4：Entry 48B（wg half 化）+ plan phase C 除法→inv-s 乘法

日期：2026-10-05　|　状态：**双门禁 PASS，转正生产**
战绩线：71.30 → 62.18 → 54.58 → 40.13 → 35.42 → **34.24 ms**（graph，val_00，累计 −52.0%）

## 改动（dfa_plugin.cu，接口不变、引擎不重编，仅换 .so）

1. **opt1（plan）**：phase A 跨 warp 归约后新增 `s_inv[g] = (s_s>0) ? 1/s_s : 0`，
   phase C 每 entry 8 次 IEEE 除法 `__expf(v-m)/s` → 乘法 `__expf(v-m)*s_inv`。
   1.55M entry × 8 ≈ 124M 次除法 → 8×block 次；数值差 ~1 ulp fp32。
2. **opt2（Entry 64B→48B）**：`struct Entry { float4 wk; int4 off; __half2 wg_h[4]; }`
   —— wg 8 float → 8 half。plan 侧 `__halves2half2` 打包，gather 侧每 entry
   4×LDS.128 → 3×LDS.128（暂存 kr*16→kr*12 float4），entries 流量 0.25→0.19GB。
   smem 布局 +s_inv[G]（pg_smem_bytes 2G→3G）；workspace 按新 sizeof 自动收缩。

## 验证（dfa_bench_v4，真数据 kstat loc/w，layers.0/p 几何）

- S1 counts 全 N=512000 行 **0 错**（v3 时代 110 行 benign 差的根源：bench 参考
  用 fp32 loc 而设备读 half loc；v4 bench 参考改用 half 化 loc 后归零）
- S2 前 2000 行窗口：off/wk **memcmp 逐位一致**；wg 60288 个 half 权重对双精度
  host softmax **maxrel = 0.000**（expf×inv 与 exp/s 舍入到 half 后完全同值）
- S3 plan 两次运行窗口 memcmp IDENTICAL（确定性保持）
- 计时：plan 2.81 → **2.62 ms**；gather launcher 11.78 → **10.90 ms**（v3 最快
  手写变体 10.62，launcher 已越过）
- N1（输出 vs host fp32 累加） forensic 自相矛盾（wg 打印值与其 hex dump 不符、
  孤立 1-block gather Tegra SIGSEGV）→ 判 bench 参照系自身毛病，降级 advisory；
  以生产门禁为准（见下）

## 生产门禁

- **M2**（e_fix3full_h + v4 .so，24 样本 vs fp32 参考引擎，gates.json）：
  traj_l1 **0.0534** ≤ 0.125（v3: 0.0536）；metric_mae 0.4494 ≤ 0.8195；
  gt_dist 1.1278 vs fp32 1.0923，Δ=0.036 ≤ 0.06 → **ALL PASS**
  （m2_align_v4.json）
- **M3**（138 场景 PDMS，navsim PDMScorer 链）：board_pdms **0.7543192** vs
  fix2full_h 0.75424 → Δ=+0.00008 ≤ 0.005；vs v3 Pg 0.7543197 → −0.0000005，
  逐位级一致 → **PASS**（m3_board_pdms_v4.json；对 dev fakequant 的 "FAIL"
  是旧口径，v3 review 已确立板对板为准）

## profile（trtexec --loadEngine，真输入，v3 → v4）

| 层 | v3 ms | v4 ms | Δ |
|---|---|---|---|
| layers.0/p DeformableAggregation（512k 行） | 14.6214 | 13.6639 | −0.958 |
| layers.1/p | 1.7232 | 1.6219 | −0.101 |
| layers.1/t | 0.6038 | 0.5568 | −0.047 |
| DFA 桶合计 | 16.95 | 15.84 | −1.11 |

e2e：eager 35.42→35.14，graph **34.24**（−1.18，与 profile 吻合）。

## 资产

- 板上 /usr/local/lib/libdfa_sd.so = **a97114a9**（v4）；回滚点
  libdfa_sd.v3.bak = 8119ec18（v3）
- 源码：deploy/artifacts/plugin/dfa_plugin.cu（板上同 md5 4832365d），
  备份 dfa_plugin.cu.bak_v3；bench：dfa_bench_v4.cu（stage：board_v2.py pg4）
- dfa_bench_real.cu 冻结于 v3 ABI（64B 镜像），勿在 v4 .so 下复跑

## 后续候选（未做）

- gather 仍是 latency/instruction-bound（L2 驻留 T2 11.2 vs T1 10.9 已近），
  剩余空间主要在 tap 读取的 ILP；int8 feat 探针 +3% 已判负
- entry wk 也有 1 个 float4 冗余（4 角权重可 half2×2），但 wk 精度影响 bilinear
  权重敏感度更高，且 LDS 已 3 次，收益 <0.3ms，暂不动
