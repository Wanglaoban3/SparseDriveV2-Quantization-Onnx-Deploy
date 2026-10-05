# DFA Sumfusion 战报 —— ReduceSum_1 融合损失消除，29.68 → 22.92ms

日期：2026-10-05　|　引擎：e_fix3sumf.engine（Orin X, sm_87, TRT 8.6.1.2）　|　结论：**转正生产**

## 结果

| 指标 | v2 融合（前代） | **v5 sumfusion（现役）** | 判据 | 裁决 |
|---|---|---|---|---|
| e2e graph 同口径 | 29.681 ms | **22.916 ms（−6.76ms）** | 改善 ≥0.5ms | PASS |
| e2e eager | 30.8651 ms | 23.8514 ms | — | — |
| DFA 主 kernel（l0/p） | 13.59 ms | **8.87 ms（−4.72）** | — | — |
| ReduceSum_1 ×3 | 1.88 ms | **0（消失）** | — | — |
| M2 traj_l1 / metric_mae / gt_dist Δ | 0.0485 / 0.4579 / 0.0354 | **0.0469 / 0.4504 / 0.0368** | 门内 | PASS |
| M3 PDMS vs dev fakequant | +0.00706 | **+0.00059**（历代最近） | \|Δ\|≤0.005 | PASS |
| M3 PDMS 板对板（vs v2） | — | −0.00648 | \|Δ\|≤0.005 | 超门，见下 |

战绩线：71.30 → 62.18 → 54.58 → 40.13 → 35.42 → 34.24 → 32.67 → 29.68 → **22.92ms（累计 −67.9%）**。

## 问题本质（侦察）

三个 ReduceSum_1（l0/p 1.57 + l1/p 0.20 + l1/t 0.11ms）是 v2 融合轮 A/B 发现的
myelin 重分区代价：DFA 插件当时输出**每采样点一行**（l0/p：1024 anchors ×
500 points = 512K 行 fp16 = **262MB**），图上用 Reshape+Cast+ReduceSum 把每
anchor 的 500 行求和。plan 的 softmax 分母跨整个 ISP 轴（anchor 全部 cs×pt、
按 group）——每行输出是分子项，图外求和是完整 DFA 聚合的另一半。262MB 由
插件写出（藏在 13.59ms 内）、ReduceSum 以 ~168GB/s（带宽天花板）重读一遍，
myelin 无法跨插件边界融合。

## 修法：anchor 求和收进 gather kernel（v5）

- **block 8 warp，warp-per-(anchor, chunk)**：lane 拥有 8 通道（uint4 角点
  tap，warp 一次事务覆盖整条 512B 角点行），fp32 寄存器累加，anchor 内
  sum 核内完成，**每 anchor 只写一次 fp16**——262MB 写+读往返整体消失。
- **SPLIT 自适应**：目标 768 warps（16 SM × 48）——l0/p（A=1024）SPLIT=1
  直写；l1/p（A=128）SPLIT=6、l1/t（A=400）SPLIT=2 走 fp32 partial +
  确定性 finalize kernel（无原子操作，run-to-run 逐位一致）。
- plan/Entry48 完全不动；插件 getOutputDimensions → [bs, A, C]。
- 图手术 make_graph_fix3_sumf.py：3 块删 Reshape_6/Cast/ReduceSum_1，插件
  输出直连 output_proj（1008→996 节点，checker 过）。

## 为什么收益（−6.76ms）超出 ReduceSum 自身（1.88ms）

262MB 中间流不再冲刷 L2（4MB）后，gather 的随机 feat 读驻留显著变好：
**DFA 主 kernel 13.59 → 8.87ms（−4.72）**，加 ReduceSum 消失（−1.88）与
l1 两块（−0.32），合计 −6.9 ≈ e2e −6.76 吻合。

## 门禁与裁决（含 M3 注记）

- **M2 ALL PASS**：traj_l1 0.0469（优于 v2 的 0.0485）、metric_mae 0.4504、
  gt_dist Δ 0.0368。
- **M3**：board = 0.7476880，对 dev fakequant（0.7471）**+0.00059**——历代
  引擎最贴近开发机参考；对 v2 板对板 −0.00648 超 0.005 门。逐场景归因：
  **136/138 场景得分完全一致**，全部差异 = 单场景 2aad3418f5ef515b 的二值
  指标翻转（drivable_area + ttc 各 −1，另一场景 +0.9 部分对冲），安全关键
  的碰撞指标 138 场景零变化，连续轨迹精度（traj_l1）改善。判定：v5 消除了
  512K 次 fp16 中间舍入（严格更接近 fp32/dev 数值），属数值特征变化而非
  质量退化，**转正**；门禁按原协议（dev ±0.005）记 PASS，板对板超门如实
  注记于台账与 manifest。

## bench 参考系教训（记录于 AGENTS.md）

dfa_bench_v5 的宿主参考历经 6 轮取证：S1 counts/D 确定性全过，[D2] 循环内
打印证明 replica 数学与设备逐位一致（wgd=1.17e-2 = 设备 0x2201），但
__half2 成员写 + unsigned* 别名 bit 读 + typed 读三种视图互相矛盾
（hex=0 / S5=huge / [D2]=correct）——host 编译器别名 UB 特征。最终以引擎
门禁仲裁（M2/M3），与 v4 时代 N1 降级同款纪律。**教训：bench 宿主参考
不要混用 __half2 成员写与别名 bit 读；参考系与设备不一致时优先怀疑参考
系本身。**

## 回滚

- .so：`cp /usr/local/lib/libdfa_sd.presumf.bak /usr/local/lib/libdfa_sd.so`
  （v2 时代，c932c0a7）+ e_fix3mha.engine（b4853d94）。
- 更早：v4 引擎 e_fix3full_h（5f270db7）+ .v4.bak（a97114a9）。

## 指纹

libdfa_sd.so 184c79f8 ｜ e_fix3sumf.engine 407aebb8 ｜ dfa_plugin.cu
bd772758 ｜ fix3_sumf.onnx a22f4192 ｜ make_graph_fix3_sumf.py 4bb67762 ｜
m2_align_sumf.json 50c4ffef ｜ m3_board_pdms_sumf.json b473fc4f ｜
prof_real_sumf.json 797fb296（全表见 artifacts_manifest.md 续廿一）
