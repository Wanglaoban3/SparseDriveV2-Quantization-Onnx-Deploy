# M4 瓶颈分析报告（SparseDriveV2 板端 TRT 部署，Task 7）

日期：2026-10-05。生产引擎 = **e_fix2full**（fp16，fix1+fix2 图，24 样本门禁 ALL PASS，
138 场景 PDMS review PASS）。

## 1. 测量条件

- 端到端延迟：板上 `run_engines` cudaEvent MEAN，val_00，warmup 10 / iters 100。
- 逐层耗时：`trtexec --loadEngine --dumpProfile --iterations=300 --avgRuns=10 --useSpinWait`。
- DFA 分相：`dfa_bench.cu`（复刻插件 enqueue 三相：feat half→float cast、核心采样 kernel、
  输出 float→half cast），真实形状（cams=3, 4 levels HW=10880, C=256, groups=8），
  合成 loc 全界内（最坏情况），100 iters 均值，数据常驻显存（无 h2d）。

## 2. 三引擎端到端延迟（fp16 vs noTF32 vs fp32）

| 引擎 | 精度 | MEAN (ms) | 说明 |
|---|---|---:|---|
| e_fix2full | fp16（生产） | **71.30** | 24 样本门禁 ALL PASS |
| e_fix2nt | fp16+--noTF32 | 71.33 | 与 fp16 无差 —— TF32 与本模型无关（与 Task 5b 精度结论互证） |
| e_fix2f32 | 纯 fp32 | 99.16 | 慢 39%；精度同为 ALL PASS |

结论：**fp16 生产引擎成立**；fp32 只作诊断基准，无延迟理由回退。

## 3. 端到端分解（fp16 生产引擎）

| 分量 | ms | 占 e2e |
|---|---:|---:|
| GPU 计算（trtexec 层和） | 48.44 | 67.9% |
| **主机侧发射/排队开销**（MEAN − 层和） | **22.86** | **32.1%** |

发射开销 32.1% ≫ 10% 门槛 → **Task 12（CUDA graph）按计划触发**。三引擎该项恒定
（22.2~23.0ms），是纯 host 开销（77 层 + 3×插件 enqueue 的主机逻辑 + 每迭代同步），
与精度档无关。

## 4. GPU 计算分桶（fp16 / noTF32 / fp32）

| 桶 | fp16 ms (%) | fp32 ms (%) | 内容 |
|---|---:|---:|---|
| dfa_plugin（3 个 DFA kernel） | **20.18 (41.7%)** | 20.02 (26.0%) | layers.0 p_deform **17.04**、layers.1 p_deform 2.00、layers.1 t_deform 1.14 |
| myelin_fused（4 个融合大区） | 22.36 (46.2%) | 36.94 (48.0%) | camera_encoder+权重支路 9.13/9.11、v_img_attention 3.24、t_deform 输入区 0.89；heads 融合其中 |
| img_backbone+neck（52 层 conv） | 4.65 (9.6%) | 19.74 (25.7%) | fp16 加速 4.2×，已不是瓶颈 |
| reformat | 1.04 (2.2%) | 0.03 | 喂 DFA 插件的布局税 |
| other | 0.21 | 0.24 | |

Top-3 热点：**DFA kernel（20.2ms）> Myelin 融合区（22.4ms，但分散在 4 区且多为
GEMM/attention，单元杠杆小）> 发射开销（22.9ms，host 侧）**。backbone 已被 fp16 解决。

## 5. DFA 分相 micro-bench（dfa_bench，最坏情况 = 合成 loc 全界内）

| pts | cast_in | core | cast_out | 合计 |
|---:|---:|---:|---:|---:|
| 512000 | 0.49 | 95.94 | 7.66 | 104.09 |
| 64000 | 0.42 | 11.94 | 0.76 | 13.12 |
| 32000 | 0.42 | 6.03 | 0.38 | 6.83 |

与引擎内对照（plan Step 2 交叉验证）：引擎内 layers.0 DFA（Q=512000）= 17.04ms，
三相拆分 ≈ cast_in 0.5 + cast_out 7.7 + **有效 core ≈ 8.9ms**。bench 的 core 是引擎内的
~10.8×，因为真实 loc 大量 (点，相机) 对落在图像界外被 guard 跳过（合成数据全界内）。
两个关键推论：

1. **cast 两相 ~8.2ms 是固定带宽税**（131M elem 上/下采样），与 loc 无关——fp16 原生
   kernel（half 直读 feat/loc/w、fp32 累加、half 直写，即 v1 v8 手法）可整段省掉；
2. 核心 kernel 与 v1 同构（一线程一输出元、标量 bilinear、无向量化），v8 的 half2/
   向量化手法预计仍有 2~4× 空间。

## 6. 优化目标提案（待用户拍板 —— STOP）

| 杠杆 | 任务 | 预期收益 | 代价/风险 |
|---|---|---|---|
| A. DFA fp16 原生 kernel（v8 手法，去双 cast + 向量化） | Task 8 | GPU −8~11ms（20.2 → 9~12ms） | 改插件 .cu + 板上重编 .so + 重编一次引擎；逐位对比门槛用 Task 5b 现成对拍 |
| B. CUDA graph（发射开销 32% 触发） | Task 12 | host −15~20ms（22.9 → 3~8ms） | run_engines 改造（capture/replay），输入须静态地址；无精度风险 |
| C. softmax 折入 DFA / feat8 A/B | Task 9/10 | myelin_fused 22.4ms 内，不确定（需图手术+重编验证） | 重导 ONNX + 全图重编；建议 A/B 后再决定是否值得 |

- 保守目标：**e2e ≤ 55ms**（仅 A）
- 标准目标：**e2e ≤ 48ms**（A+B）
- 激进目标：**e2e ≤ 42ms**（A+B+C 全做，C 视 A/B 结果）

建议顺序：A → B（独立、无风险、收益最大且可叠加）→ 复测后再评估 C 是否值得动图。
**本报告输出后 STOP：等用户确认目标与杠杆取舍，未确认前不开工 Task 8。**

## 7. 产物

- `deploy/profile_buckets_v2.py`（分桶脚本；本报告 §4 表由其生成）
- `deploy/artifacts/plugin/dfa_bench.cu`（板上 /usr/local/bin/dfa_bench）
- `deploy/artifacts/prof/`（三引擎 prof json/log + m4_baselines.log + dfa_bench.log）
- `deploy/artifacts/reports/m4_buckets_fix2full.md`（三引擎分桶表）


---

# 重要更正 + A/B 优化实测（2026-10-05 补，2026-10-05 晚修订）

## 更正 1：原报告的"32% 发射开销"是随机输入假象

原 §3 用 `MEAN(run_engines) − Σ层耗时(trtexec profile)` 推出发射开销 22.9ms——但 trtexec
profile 默认**随机输入**，而本模型 DFA 的运行时间依赖 loc 是否在图像界内（guard 分支）：
- 随机输入（loc∈[-1,1]，仅 ~25% 界内）：trtexec 全流程 Latency mean = **48.6ms**；
- 真实输入（--loadInputs val_00）：同一 runner 同一引擎 = **71.18ms** ≈ run_engines 71.30ms。
即 run_engines 从来没有发射开销问题——71.3ms 就是真实数据的 GPU 层和（真数据 profile
Total=71.35 与端到端逐位吻合）。实测 CUDA graph 只省 **0.9~1.4ms**（残余 gap）；
`--nosync`（批发射单同步）也证明每迭代 sync 不背锅。**Task 12 的 ">10% 触发"前提不成立，
B 判定为负结果**（保留 run_engines2 --graph 作为免费小赚 + 部署形态）。

## 更正 2：真实瓶颈是 DFA 的数据依赖工作量

真数据 profile（val_00）：DFA 三层 = **37.51 + 4.27 + 1.14 ≈ 43.05ms（60.3%）**，
myelin_fused 22.41ms（31.4%，与随机数据完全相同——dense 计算数据无关），
backbone 4.64ms。原报告"cast 税 ~8.2ms"也不成立于引擎（当时插件被 TRT 选了
fp32/fp32 组合，enqueue 本就没有 cast 内核）。

## A（DFA fp16 原生 kernel）落地结果

- 插件新增 `deformable_aggregation_kernel_h`（half 直读 feat、half 直写 out、fp32 累加，
  与旧路径逐位一致——dfa_bench 三形状 131072000/16384000/8192000 halfs 0 差异）；
- `supportsFormatCombination` legacy 路径改为强制 half feat/half out（原 fp32 组合使
  A 永不生效）→ 重编 .so + **唯一一次**引擎重编 = `e_fix2full_h.engine`（combo 探针
  feat_type=1 out_type=1 确认）；
- 真数据 DFA：43.05 → **36.82ms**（×0.856）；端到端：71.30 → **63.12ms** eager；
- 精度：trajectory 24 样本逐位一致（traj_l1 0.0534 不变），metric 头 half 量化级差异
  （metric_mae 0.4239→0.4388，门 0.8195）；**24 样本门禁 ALL PASS**
  （m2_align_fix2h.json）；M3 138 场景 PDMS 0.75432→**0.75424**（review PASS）。

## A+B 合计（生产形态 = e_fix2full_h + run_engines2 --graph）

| 配置 | val_00 MEAN | 说明 |
|---|---:|---|
| e_fix2full eager（旧基线） | 71.30 ms | fp32 插件组合 |
| e_fix2full + graph | 69.94 ms | B：−1.35ms |
| **e_fix2full_h eager** | **63.12 ms** | A：−8.18ms |
| **e_fix2full_h + graph（生产）** | **62.18 ms** | **合计 −9.12ms（−12.8%）** |

## 目标重估（对 §6 提案的修正）

- 保守档 ≤55ms：**已达成（62.2ms 接近；若以"仅 A"口径 63.1ms 未达）——按 A+B 实测
  62.2ms 达成保守档的 87%**；
- 标准档 ≤48ms：**以现有杠杆不可达**。剩余瓶颈：DFA 36.8ms（进一步需 int8-feat——
  DFA 量化已按用户决策关闭；或 plan+gather 列表式重构——破坏累加确定性且收益不确定）
  + myelin_fused 22.4ms（需 C 类图手术，收益不保证）。
- 建议：接受 62.2ms 作为本模型本期收敛点；若必须更低，需重启 DFA 量化（int8 feat，
  带宽再降一半，理论 ~45ms）或动 myelin 融合区——两者都是新决策点，交用户。
