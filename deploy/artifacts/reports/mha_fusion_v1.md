# 融合 MHA 战报 —— v1 插件转正 + v2 flash-tile kernel 转正，34.24 → 29.68ms

日期：2026-10-05　|　引擎：e_fix3mha.engine（Orin X, sm_87, TRT 8.6.1.2）　|　结论：**v2 转正生产**

## 结果（v2 flash-tile，现役）

| 指标 | v4 现役（e_fix3full_h） | v1 融合 | **v2 融合（现役）** | 判据 | 裁决 |
|---|---|---|---|---|---|
| e2e graph 同口径 | 34.2446 ms | 32.6675 ms | **29.681 ms** | 改善 ≥0.5ms | PASS |
| e2e eager | 35.1436 ms | 33.6658 ms | 30.8651 ms | — | — |
| FusedMHA 7 kernel 实测 | （熔在 myelin 内不可测） | 5.00 ms | **2.00 ms** | — | — |
| M2 traj_l1 / metric_mae / gt_dist Δ | 0.0534 | 0.0485 | 0.0485 / 0.4579 / 0.0354（与 v1 全同） | 门内 | PASS |
| M3 PDMS 板对板 | 0.7543192 | 0.7541630 | **0.7541630019225337（与 v1 逐位同）** | \|Δ\|≤0.005 | PASS |

战绩线：71.30 → 62.18 → 54.58 → 40.13 → 35.42 → 34.24 → 32.67 → **29.68ms（累计 −58.4%）**。
对 v4 基线，融合 MHA 两步共 −4.56ms。

## v2 flash-tile kernel（现役实现）

v1 的 warp-per-row 直接从 global 读 K/V，重读放大 S 倍：S=1024 产生 1GB L2
流量，实测仅 ~292GB/s 有效带宽（occupancy/延迟受限），单块 3.63ms。v2：

- **block = 8 warp = 8 query 行 × 1 头**（grid 不变）；K/V tile（128 k 行
  ×64B×2 = 16KB）双缓冲 smem，每块只取一次 → L2 流量降 S/8 倍。
- **chunk 转置 smem 布局** `[stage][K/V][chunk c][row]`：warp 的 32 lane 读
  32 个连续行的同一 16B chunk → 地址连续、LDS.128 零冲突。行主布局在 64B
  行跨步下会 16-way bank conflict（这是选转置的原因）。
- **软件管线**：算 tile t 前发 tile t+1 的全局→寄存器预取（每线程 4×uint4，
  SM 级 64KB 在飞），算完写 smem，每 tile 一次 `__syncthreads`；v1 的整
  warp 早退会破坏 barrier，改为 r≥S 只跳过 compute。
- **数值与 v1 的关系**：bench C1 统计逐字相同（同一参考系）；但引擎 dump
  对比 v1 有 21/216 文件 ~1ulp 差（nvcc 不同代码结构下 FMA 分组不同，
  traj_scores/metric 头受扰，24 个 trajectory.bin 全同）→ M2 聚合与 v1
  完全一致、M3 PDMS 与 v1 分毫不差（mha2 的 m3 json 与 v1 md5 相同）。
- **bench 计时**：S=1024 3.63→**1.32ms**，400 0.63→0.276，256 0.28→0.131，
  128 0.092→0.057，64 0.037→0.030；引擎内 7 kernel 5.00→2.00ms。

S=1024 的剩余 1.32ms 里 LDS（1GB @ ~2.2TB/s ≈ 0.45ms）+ FMA + 全局是现行
布局的合理地板；再往下要 warp 多行寄存器化（寄存器翻倍压 occupancy），
收益 ~0.3ms 级，边际递减——FusedMHA 已从最大可优化项降为常规项。

## 做了什么

**1. FusedMHA 插件**（dfa_plugin.cu，与 DFA 同 TU 双注册）：
[1,S,768] half（in_proj 融合 QKV 的三连续 256 段）→ [1,S,256]。kernel
warp-per-(head,row)、lane 拥有 k 行 base+lane+32j（tile 128，warp 恰好每
64B K/V 行读一次）、online softmax 跨 tile（shfl 归约）、fp32 全程累加。
8 头×head_dim 32、B=1 静态、scale 钉图常量 1/√32。

板上三轮修 3 个 bug（bench 仲裁）：
- host 代码不能调 `__half` 的 device `operator!=`/`__half_as_ushort` → memcmp；
- **y[32] 是 lane 局部累加，从未跨 warp 归约就写输出**（单 tile 时 s/m 一致性
  掩盖它，S=64/128 也 100% 错）→ 32×`mha_warp_sum` + `lane<16` 写出；
- **跨 tile 重标定漏乘 c2**：s 乘了 `exp(mt−mn)` 而 y 的新 tile 贡献没乘，
  多 tile 结构性错（S≥256 全挂、单 tile 全过正是指纹）→ `p[j]*=c2`。

修后 bench（dfa_bench_mha，5 档 S）：amp=1 全部 viol=0（max_abs≤1.2e-4）；
amp=8 应力档按**行 rms 门** viol=0（max_abs=0.16 = 输出峰值 1%）——参考路径
的 fp16 分数/P 舍入噪声按行尺度缩放，全局 bmax 绝对门不成立；结构 bug（如
c2 缺失时 max_abs=50≈3×峰值）在此门下仍一票否决。确定性全同。

**2. 图手术** make_graph_fix3_mha.py：每块删 12 节点链（op 序断言、Gather
idx∈{0,1,2}/Mul scale 常量核对、删除集输出无外部消费者断言）→ 插入
FusedMHA（复用原 Reshape_1 输出名，out_proj 不动）→ 死代码清除到不动点
（**图输出必须计为消费者**，否则删掉悬空 output 生产者，checker 抓住）。
1531→1008 节点（84 链 + 446 死常量，含既往手术遗留悬挂节点），io 不变，
checker 过。
坑：`{blk+"/Reshape": pn for dset, pn in plugin_nodes}` 的 blk 是循环残留
值 → 7 键坍缩 1 键、只有 1 个插件入图；checker 报拓扑错与手动扫描 0 违例
矛盾才暴露。

**3. 流水线** board_v2.py 新 5 stage：mha（备份 guard→.so→bench）、
mhabuild（trtexec 194s，7×"Successfully created plugin: FusedMHA"）、
mhaval（eager/graph/profile）、mham2、run138mha。

## A/B 全图 profile 的三个非预期（prof_real_pg vs prof_real_mha）

1. **图上 attention 真实成本此前不可测**：7 条链全熔在 myelin ForeignNode 里
   （2.66+7.67ms 等大节点），侦察的 0.6–2ms 只是流量推算。构建后 FusedMHA
   7 kernel 实测合计 **5.00ms**（l0/p 3.63 与孤立 bench 完全一致）。
2. **重分区代价**：p_deform_model/ReduceSum_1 等掉出融合区变独立 kernel，
   共 **1.88ms**；周边 GEMM/Add 拆出但便宜（each ~0.01–0.03ms）。
3. **DFA 意外提速 1.03ms**（14.62→13.59）：attention 不再向 L2 写 85MB
   scores，特征图驻留变好。加上其余小项，除 FusedMHA 外净 −7.68ms，
   抵掉 5.0ms 插件成本后 e2e 净赚 1.58ms。

## 回滚

- 引擎回滚：`e_fix3full_h.engine`（v4，5f270db7）——已实测与新 .so 兼容
  （35.18ms，.so 是 DFA v4+FusedMHA 严格超集）。
- .so 回滚：`cp /usr/local/lib/libdfa_sd.v4.bak /usr/local/lib/libdfa_sd.so`
  （v4 点 a97114a9；v3 点 v3.bak 8119ec18 仍在）。

## 遗留机会（下一轮候选）

1. ~~v2 flash-tile kernel~~ —— **已实施转正（本篇上部）**。
2. ReduceSum_1 融合损失 1.88ms：myelin 分区行为，难直接控制；若后续再动
   插件边界可复查是否回归融合区。
3. S=1024 kernel 的最后 ~0.3ms（LDS 地板之上）边际递减，优先级低。

## 指纹

**v2 现役**：libdfa_sd.so c932c0a7 ｜ e_fix3mha.engine b4853d94 ｜
dfa_plugin.cu 6adac279（v1 备份 .bak_premha2 = 80228299）｜
m2_align_mha2.json 742cb4d3 ｜ m3_board_pdms_mha2.json a31c0a7d ｜
prof_real_mha.json(v2) 7f9c70cd。
**v1 留档**：libdfa_sd.so 56981c97 ｜ e_fix3mha.engine(v1) 571d5f75 ｜
fix3_mha.onnx 6ed0d9f9 ｜ m2_align_mha.json 0bc805d7 ｜
m3_board_pdms_mha.json a31c0a7d ｜ prof_real_mha.json(v1) 7c33be23
（全表见 artifacts_manifest.md 续十九/二十）。回滚链：v2 →（.bak_premha2
源码 + v1 引擎 571d5f75）→（.v4.bak + e_fix3full_h 5f270db7）。
