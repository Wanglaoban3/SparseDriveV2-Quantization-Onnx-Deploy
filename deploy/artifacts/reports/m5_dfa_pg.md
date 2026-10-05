# M5 · DFA 深度优化（Pg：plan+gather+softmax 内联）战报

日期：2026-10-05　引擎：`e_fix3full_h.engine`（板上 md5 `5f270db7b41cb7e6b8ab6aa2f0327965`）
插件：`libdfa_sd.so`（同文件含 v1 `DeformableAggregation:1` + 新 `DeformableAggregationPg:1`
两个 creator，旧引擎回滚不受影响；重编前备份 `/usr/local/lib/libdfa_sd.prePg.bak`）

## 一、结论速览

| 指标 | e_fix2full_h（A+B 生产形态） | **e_fix3full_h（本次）** | Δ |
|---|---:|---:|---:|
| 真数据 eager（val_00, 100 iter） | 63.12 ms | 55.50 ms | −7.6 |
| 真数据 **graph（生产形态）** | **62.18 ms** | **54.58 ms** | **−7.6（−12.2%）** |
| 相对最初浮点基线 71.30 ms | −12.8% | **−23.5%** | |
| myelin_fused 桶 | 21.4 ms | 14.42 ms | −7.0 |
| dfa_plugin 桶（3 层） | 36.82 ms | 36.07 ms | −0.75 |
| img_backbone / reformat | 4.6 / 0.1 | 4.64 / 0.12 | 不变 |

逐层（真数据）：layers.0/p **31.35ms**、layers.1/p 3.64ms、layers.1/t 1.07ms。

门禁：**M2 ALL PASS**（`reports/m2_align_pg.json`：traj_l1 0.0536 / metric_mae 0.4423 /
gt_dist 1.1275，与 fix2h 基线 0.0534/0.4388/1.1278 在噪声内重合）。
M3 138 场景 PDMS：**PASS**（板对板）。e_fix3full_h 0.75376 vs fix2full_h 基线 0.75424，Δ=−0.00049（门禁 |Δ|≤0.005）；136/138 场景逐位同分，唯一 |Δ|>0.05 的 1 个场景（3d2120dc97445f8a，−0.078）只动了 ego_progress 软指标（模拟器 rollout 长度临界），碰撞/可行驶区域/TTC/comfort/方向合规等安全硬指标 138 场景全部无回归。对 dev fake-quant 0.7471 为 +0.0067（与基线同为偏高方向的噪声，基线同款 FAIL 标签经 review 判 PASS）。见 `reports/m3_board_pdms_pg.{json,md}`。

## 二、这轮做了什么

1. **图手术 fix2→fix3**（`deploy/make_graph_fix3.py`）：三个 DFA 节点的 w 输入上游
   `Softmax→Cast_6→Reshape_3→Transpose_2→Reshape_5` 五节点链（在 196MB fp32 张量上扫了
   数遍）整体删除，插件改吃 logits（Reshape_2 输出，`[1,A,CS·pts,8]`，cs 主序 pt 内序）。
   几何：layers.0/p A=1024·pts=500；layers.1/p A=128·pts=4000；layers.1/t A=400·pts=1280。
2. **Pg 插件**（`deploy/artifacts/plugin/dfa_plugin.cu` 追加段）：
   - K1 plan（block/anchor）：内联 softmax（fp32 online max-sum、固定归约树、`__expf`，
     logits 按 32B/线程向量化读）→ 条目有效性（组权和 > eps）∧ 相机 guard → 64B 条目
     `{float4 wk, int4 off, float4×2 wg}` 按每点固定 12 槽落位，**cam 主序 level 内序**
     与 v1 内核累加顺序完全一致；
   - K2 gather（线程/(row,channel)，与 v1 同构）：block 按 warp 暂存本行条目进 smem，
     只走 k 条有效条目；每条目数学与 v1 内核逐表达式相同（含 `loc_h*h-0.5` 的 double
     提升复刻）。
   - eps=DFA_PG_EPS=0：只剪精确 0，不做幅值剪枝，最大限度保数值。
3. **bench**（`dfa_bench_pg.cu`，板上两分布）：对拍参考路径（朴素 softmax+原生内核）
   max_abs=7.6e-6；**双跑逐位一致**（确定性成立）；worst(k=12) gather 76.5ms ≈ native
   77.0ms（已达 ~165GB/s ≈ DRAM 峰值 80%，纯带宽极限）；real 按 k 严格缩放。

## 三、剪枝前提被真数据证伪（重要修正）

用 dfatap6 真实 dump（val_00 的 loc/w）实测（`deploy/pg_kstat_local.py`）：

- **mean k = 3.86**（直方图只有 0/4/8：p50=4 = 恰 1 相机可见 × 4 层）；
- **w 的 -inf 掩码与 loc guard 完全重合**（valid rate 32.17% == guard pass rate，
  w==0 率 54.4% 全部落在 guard 之外）。

即：**老内核的 guard 从来没有为无效条目花过有意义的钱**（每条无效 cs 只是一次地址计算
+分支，无访存）。此前"17% 有效负载、大量 warp 空转"的判断是把随机输入 profile 的统计
错记到了真数据上——真数据方向相反（anchor 本来就设计在视锥内，真数据比随机输入更慢）。
DFA 侧 36.82→36.07ms（−0.75ms）就是全部剪枝收益；**本轮 −7.6ms 几乎全部来自 softmax
链移出 Myelin 区**。bench 最坏情况还表明 gather 已跑在 DRAM 峰值 80%，k=12 时无油水。

## 四、对用户问题的回答（half / v1 手法）

运行时组合探针：`[dfapg] combo feat=1 loc=1 logits=1 out=1`——**feat/loc/logits/输出
已经全部是 half I/O**。v1 项目的优化手法（logits uint4 向量读、内联 online softmax、
`__expf`、条目化 gather、warp 合并 feat 读）本轮已全部移植。唯一保留 fp32 的是累加器
（逐位验收纪律 + 累加不在瓶颈路径，half 累加不省字节）。

## 五、下一步的真实决策点（按证据排序）

1. **myelin_fused 14.42ms 里的 GEMM 量化**（用户最初的直觉，方向正确）：
   softmax 链移除后剩下的是 camera_encoder/weights_fc 等 Linear 栈，int8 tensor core
   是现役最大杠杆；病灶①只实锤过 backbone conv，Linear int8 忠实性从未单独验证，且
   fix1/fix2/fix3 图链已把当年的混淆因素全部排除。需要重量化+重导出+门禁重跑。
2. **DFA 微观 ILP 优化**：k=3.86 下 gather 有效流量仅 ~4GB/31ms ≈ 129GB/s，仍是延迟型
   而非带宽型；双条目 interleave、tap 预取等或可再挤 10-20%，工程收益比不确定。
3. **DFA int8-feat**：halve tap 字节；但当前延迟主导，字节减半的兑现率存疑，且需重过
   全套精度门禁。

## 六、（续）瓶颈归因 + gather 向量化：36.07 → 21.67ms，e2e 40.13ms

第五节的决策点 2/3 基于一个**错误前提**（"已贴带宽地板 87%"）。用户追问
"30 多 ms 太慢，必须先搞清瓶颈"后做了真数据归因，推翻了它。

### 归因（dfa_bench_real，真 loc/w，六对照）

| 对照 | ms | 含义 |
|---|---:|---|
| T1 gather 真数据 | 28.15 | 与引擎逐层 31.35 吻合（+plan 3.25） |
| T2 tap 折进 L2（2MB） | 26.40 | DRAM 贡献仅 ~1.7ms |
| T3 k=0 地板 | 8.81 | 512k block 调度/发射地板 |
| T4 plan | 3.25 | softmax 两遍扫 + emit |
| T5 half2 探针（未调优） | 15.64 | **ILP 是大头** |
| T6 流式天花板 | 151GB/s | 当日实测 |

**结论：gather 是延迟/指令型，不是带宽型**。每线程 2B tap 读 + 131M 线程串行
k 循环；T1−(T2−T3)=10.6ms DRAM、17.6ms 指令/延迟。流量模型
（pg_traffic_model.py）：feat 角点 4.00GB（89%）/ entries 0.25 / logits 0.23；
掩码按 (row,cs) 广播，有效 entry 8 group 全非零。

### 实施（接口不变，仅重编 .so，引擎不重编）

1. gather 一线程一 (row, 通道对)：`__half2` tap 读，线程数与发射指令减半；
   每通道 FMA 序与标量版逐表达式相同。
2. plan 砍 phase B：旧逻辑整遍重读 logits（98MB）算 wsum>eps；有效性并入
   phase A（eps=0 ⟺ 任意 logit 有限；wg 在 phase C 重算，gather 对 wg==0
   跳过，输出不变）。

### 结果与门禁

- bench_pg：对拍 7.6e-6、双跑逐位一致；worst(k=12) gather 43.42（标量 76.5，−43%）。
- 真数据 e2e：eager 55.50→**41.08**，graph 54.58→**40.13ms**（对最初 71.30 累计 −43.7%）。
- 分桶：dfa_plugin 36.07→**21.67**（52.8%），myelin 14.415 不变（sanity ✓）。
- M2 pg24（h2 版重跑）**ALL PASS** 且三项指标与标量版同值
  （0.0536/0.4423/1.1275）。输出与标量版非逐位一致：half 特征 1-ulp 编译级
  差异被打分头放大（traj_scores max ~0.08），trajectory 逐位相同——门禁是
  指标级，与 fix2h→fix3 先例一致。M3 复验见 reports/m3_board_pdms_pg_h2.*。

### 更新后的决策点

1. **myelin 14.42ms 的 Linear int8**（不变，另一半主线）。
2. gather 二阶段：4ch/线程、block 级一次暂存（去掉 8 warp 重复读 entries）、
   k 循环展开——探针 15.6ms 未调优，或再挤 3-6ms；**挤到 ~12ms 后 gather 才
   真正带宽型，届时 int8-feat（tap 字节减半）才值得过门禁**。
3. plan 还剩 ~2ms（phase C 的 logits 重读 31MB + emit）。

## 七、（续）gather v3：16.95ms，e2e 35.42ms（累计 −50.3%）

第六节决策点 2 实施：bench_real 加 5 个 v3 变体实板选型——CPT=4/不展开赢
（10.68ms vs h2 16.13，−34%；**输出与 h2 逐位相同**）；手动 k 展开反而慢
（寄存器压力），CPT=8 无增益。生产化：一线程一 (row, 4 通道)、uint2 8B
tap 读、block=256 线程=4 行、条目 block 协同暂存一次（去 8 warp 重复读）。

- bench_pg：对拍/确定性全过；合成 k=8 gather 31.28→21.21（−32%）。
- val_00 引擎 dump 与 h2 版 8 输出逐位相同 ⟹ M2/M3 结论继承
  （M2 0.0536/0.4423/1.1275 ALL PASS、M3 0.75432 PASS）。
- e2e graph 40.13→**35.42ms**；DFA 桶 21.67→**16.95ms**；myelin 14.41 不变。

剩余：DFA ~14ms（gather 贴近本 mapping 地板：k0 ~4.4 + DRAM ~6；plan ~2.8
里 phase C logits 重读 31MB 是下一小口）+ myelin 14.41（Linear int8 主线）。

### 补测：int8-feat（用户问，dfa_bench_real T8）

v3 内核 + int8 tap（每角点 uint32 读 4×int8 + dequant）：**10.94ms vs half
10.61ms（+3% 更慢）**。字节减半省的 DRAM 时间 ~1.7ms < dequant 指令开销
~2ms；feat DRAM 只占 gather ~4.4/10.6ms。与 v1 板测"+0.7ms 更慢"结论一致，
**int8-feat 线关闭**（现役内核下无性能收益，只有精度与门禁成本）。

宽映射补测（用户假设的完全体 int8+更宽线程映射）：

| 变体 | ms | vs |
|---|---:|---|
| half CPT=4（现役） | 10.62 | 基线 |
| half CPT=8 | 10.66 | 映射宽度已饱和 |
| **int8 CPT=8**（同 1 条载入/角点，字节减半） | **10.64** | **打平** |
| int8 CPT=16（载入指令密度再翻倍） | 12.37 | **慢 16%（寄存器溢出）** |

i8@CPT=8 与 half@CPT=8 载入指令数相同、字节减半 → 打平证明**字节数无关紧要**；
i8@CPT=16 证明**指令密度再翻倍被寄存器压力反噬**。映射/发射杠杆在 CPT=4~8
饱和，约束已转移到 per-entry 循环体（LDS 条目 + FMA 链），与数据类型正交。
