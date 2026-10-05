MD5 的 deploy\board\board_common.py 哈希:
2e1a5d090facd0373c8b754a51140ba3
CertUtil: -hashfile 命令成功完成。
MD5 的 deploy\board\board_v2.py 哈希:
3958f9c4e0c6534945da441036b215b6
CertUtil: -hashfile 命令成功完成。
MD5 的 deploy\artifacts\reports\gates.json 哈希:
4455eae4d8592bd91f9e70e1153e56ce
CertUtil: -hashfile 命令成功完成。
--- task2: folded+rewritten plain QDQ 2026-10-05 
MD5 的 deploy\artifacts\sparsedrive_int8_qdq_folded.onnx 哈希:
9eddb497028414a17ccff1e4def52a20
CertUtil: -hashfile 命令成功完成。
MD5 的 deploy\artifacts\reports\gates.json 哈希:
4455eae4d8592bd91f9e70e1153e56ce
CertUtil: -hashfile 命令成功完成。
--- task5b/6: fix2 graph + M2 ALL PASS + M3 board PDMS 2026-10-05 
MD5 deploy\artifacts\sparsedrive_fp16_graph_fix1.onnx:
251ef23b87b7fc9c518232eb6b67cc9a
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\sparsedrive_fp16_graph_fix2.onnx:
2d1b076b97f93c1fd88e0345ea6a73ad
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m2_align_fix2.json:
41df01fb2497312d5bf28c8a856c593c
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m2_align_fix2_fp32.json:
a2f96295005a74cb95882942b3474813
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m3_board_pdms.json:
dcdc3bac9feed08cbef4706f287762e2
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m3_board_pdms.csv:
ed13b980f384eb3a659d8686058ac9d0
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m3_board_pdms.md:
5031b4a55439939637c4d5a3491349ab
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\engine_inputs\mini138\tokens.txt:
e0a6e211ec744d7b702ba0064e80941c
CertUtil: -hashfile 命令成功完成。
MD5 deploy\dump_board_inputs.py:
4a6cd109dd7238ef0789808b1677c6e1
CertUtil: -hashfile 命令成功完成。
MD5 deploy\eval_board_pdms.py:
6423b717fdb2629b9d5693a5d49252cf
CertUtil: -hashfile 命令成功完成。
MD5 deploy\cmp_engine_ref.py:
3ad62a963003897d4a2188748a7ca881
CertUtil: -hashfile 命令成功完成。
MD5 deploy\cmp_engine_ref.py.bak_pre_sd2:
6bef30d8d4082792b758b491c9181b47
CertUtil: -hashfile 命令成功完成。
MD5 deploy\board\board_v2.py:
fba991156c7e9e458582be6eeb2b49c0
CertUtil: -hashfile 命令成功完成。
mini138: 138 token dirs (tokens.txt ds order), board_outs/mini138 1243 files pulled
board engines: e_fix2full(fp16, production) / e_fix2f32 / e_fix2nt @ /opt/m0/sd2/engine
M3 result: board_pdms=0.7543 vs fakequant 0.7471 (delta +0.0072, formula FAIL) -> review PASS: 134/138 scenes track dev (trimmed mean -0.0006), 4 borderline flips only, no systematic degradation (see m3_board_pdms.json review block)

--- task7: M4 bottleneck analysis 2026-10-05 
MD5 deploy\profile_buckets_v2.py:
5be79d61469da07f1f7468742659d519
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\plugin\dfa_bench.cu:
f51fae9f02ecd3c0cce5df809b1bb0eb
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m4_bottleneck.md:
9feca2240770bfb1b264fa466adc1510
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m4_buckets_fix2full.md:
7a2e911e81476f18fa131322a197fc21
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\prof\m4_baselines.log:
db9e0477e2174639d1e2877f1678eaf7
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\prof\dfa_bench.log:
ce6b5579aef50a8e01b5092450cf1465
CertUtil: -hashfile 命令成功完成。
engines latency: e_fix2full 71.30 / e_fix2nt 71.33 / e_fix2f32 99.16 ms; launch overhead 22.9ms (32%) -> Task12 CUDA graph triggered; GPU split: DFA 20.18 + myelin_fused 22.36 + backbone 4.65 + reformat 1.04 ms

--- task8/12: A(DFA half native)+B(cuda graph) closed 2026-10-05 
MD5 deploy\artifacts\plugin\dfa_plugin.cu:
2e64321e2cc4b59060f8c7596b15411d
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\plugin\dfa_bench.cu:
3df0f5d31e73d94ecfdcb9c9a86b84dd
CertUtil: -hashfile 命令成功完成。
MD5 deploy\run_engines2.cpp:
135b9b2c486468b27d6dc949b46fb2b7
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m2_align_fix2h.json:
b313d30c0b2c0347cb356b10413f3f63
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m3_board_pdms_h.json:
5bdffee77af9f258bb4f60d964f4d66c
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\reports\m4_bottleneck.md:
dacdf3611d89a8335221db606ded6568
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\prof\prof_real.json:
7be63a5cc2b5f939e8836bc4962d27b5
CertUtil: -hashfile 命令成功完成。
MD5 deploy\artifacts\prof\prof_real_h.json:
cd25830cca703afe078d7f22b6543a23
CertUtil: -hashfile 命令成功完成。
result: e_fix2full_h(half-only combo) + run_engines2 --graph = 62.18ms (was 71.30, -12.8%); DFA real-data 43.05->36.82ms; M2 ALL PASS (0.0534/0.4388/1.1278); M3 PDMS 0.75424 review PASS; B = -0.9~1.4ms only; 48ms target unreachable with current levers (DFA int8-feat or C surgery = new decision)

### 2026-10-05 Pg（plan+gather+softmax 内联）追加指纹
- deploy/make_graph_fix3.py —— fix2→fix3 图手术（softmax 链并入插件 x3 节点）
- deploy/artifacts/sparsedrive_fp16_graph_fix3.onnx —— 手术产物（15 节点删除、3 节点改 Pg）
- deploy/artifacts/plugin/dfa_plugin.cu —— 追加 DeformableAggregationPg（K1 plan + K2 gather + 注册器）
- deploy/artifacts/plugin/dfa_bench_pg.cu —— 对拍/确定性/两分布计时
- 板上：engine/e_fix3full_h.engine（5f270db7b41cb7e6b8ab6aa2f0327965）、
  /usr/local/lib/libdfa_sd.so（51c06da4fde385e2354b9c24dafacdd6，v1+Pg 双 creator）、
  /usr/local/lib/libdfa_sd.prePg.bak（重编前备份）、/usr/local/bin/dfa_bench_pg
- reports/m5_dfa_pg.md —— 本轮战报（54.58ms graph，−12.2%）
- reports/m2_align_pg.json —— M2 ALL PASS
- prof/prof_real_pg.{json,log,buckets.txt}、prof/kstat/{loc,w}.bin —— 真数据 profile 与 k 统计
- board_outs/pg24/ —— 24 样本 dump（M2 输入）

### 2026-10-05 h2 gather 向量化（DFA 归因战役）追加指纹
- deploy/pg_traffic_model.py —— 真数据精确流量模型（feat 4.00GB/entries/logits）
- deploy/artifacts/plugin/dfa_bench_real.cu —— 真数据六对照归因 bench（板上 /usr/local/bin/dfa_bench_real）
- deploy/artifacts/plugin/dfa_plugin.cu —— gather half2 向量化 + plan phase B 并入 phase A
- 板上：/usr/local/lib/libdfa_sd.so（a6ea087c4d23f9ef12fb670656f7937b，h2 版；
  libdfa_sd.prePg.bak=标量 Pg 版）；引擎 e_fix3full_h.engine 不变（5f270db7…）
- reports/m2_align_pg_h2.json —— M2 ALL PASS（0.0536/0.4423/1.1275，与标量版同值）
- prof/prof_real_pg.log —— h2 版真数据逐层（dfa 21.671 / myelin 14.415 / backbone 4.647）
- board_outs/pg24/ —— h2 版 24 样本 dump（覆盖标量版）

### 2026-10-05 gather v3（4ch/线程 + block 一次暂存）追加指纹
- deploy/artifacts/plugin/dfa_bench_real.cu —— 加 gather_v3_kernel<CPT,UNROLL> 五变体选型
- deploy/artifacts/plugin/dfa_plugin.cu —— v3 gather（uint2 tap、block 协同暂存）+ 几何约束
- 板上：/usr/local/lib/libdfa_sd.so（8119ec18c000d0e0f6f4a744bcb90039，v3 版）
- prof/prof_real_pg.log —— v3 版真数据逐层（dfa 16.948 / myelin 14.412 / backbone 4.649）
- board_outs/v3chk —— val_00 v3 dump（与 pg24 h2 版逐位相同）

### 2026-10-05 myelin Linear int8 三剂量判负（路线关闭）追加指纹
- deploy/make_graph_fix3_qdq.py（e4ee9faf16ba2850d7bd5d4c357ec06c） —— QDQ 嫁接工具（donor scale 移植 + roundtrip 校验）
- deploy/verify_qdq_graft.py（c130654cf25f878dff00b1eaa5a004a5） —— 嫁接验证（路径结构 + scale/zp 字节比对）
- deploy/inspect_matmul_fix3.py（ed83828ddfe384e74bbc47482b3ed84d） —— fix3 MatMul 全量清单（形状/分区/donor 对齐）
- deploy/inspect_donor_chain.py（0d48009098759f1512de6c40635a583c） —— donor QDQ 链路探查
- deploy/board/board_v2.py（80f9a624ac0fe9e9af16704c603cb341） —— 新增 _make_qdq_stages 工厂 + qdq* stage
- deploy/artifacts/reports/myelin_matmul_inventory.md（6a924150d685d5762bf8fa989d94cce2） —— 79 MatMul 清单报告
- deploy/artifacts/reports/m6_myelin_int8.md（593a3ce5b8ce61b5aee3ce9f0161cf56） —— 本轮战报
- deploy/artifacts/sparsedrive_fp16_graph_fix3_qdq_e1.onnx（a6024a9c86cf7378b7a76725755a7078） —— E1 嫁接图（留档）
- deploy/artifacts/prof/qdq_e1/qdqv_e1_prof.log（96df52eaa52e14c225021a9353698898） —— E1 真数据 profile（myelin 18.571）
- deploy/artifacts/prof/qdq_e1/esd2_prof.log（becb5783d7ec7ab6238ae0d59ac26994） —— e_sd2 全图 int8 profile（myelin 50.991）

## 续十七：DFA v4（Entry 48B + inv-s 乘法），35.42→34.24ms，M2/M3 PASS（2026-10-05）

| 产物 | 路径 | md5 |
|---|---|---|
| libdfa_sd.so (v4, DFA Entry48+inv_s) | /usr/local/lib/libdfa_sd.so | a97114a91abc4103306559bb3adc5892 |
| libdfa_sd.v3.bak (回滚点=v3) | /usr/local/lib/libdfa_sd.v3.bak | 8119ec18c000d0e0f6f4a744bcb90039 |
| dfa_plugin.cu (v4 源, 本地=板上一致) | deploy/artifacts/plugin/dfa_plugin.cu | 4832365d8ab5e0e8bc02ff5653fec31d |
| dfa_plugin.cu.bak_v3 (v3 源备份) | deploy/artifacts/plugin/dfa_plugin.cu.bak_v3 | 4403a57fe72b9de98ffff5fb1f94b548 |
| dfa_bench_v4.cu (v4 验收 bench) | deploy/artifacts/plugin/dfa_bench_v4.cu | 3e25edb4ed27a0501a857ddc0882d96b |
| dfa_bench_v4 (板) | /usr/local/bin/dfa_bench_v4 | - |
| m2_align_v4.json (M2 ALL PASS) | deploy/artifacts/reports/m2_align_v4.json | 7cd69ce41d2595beb8a7c585be409a80 |
| m3_board_pdms_v4.json (M3 PASS) | deploy/artifacts/reports/m3_board_pdms_v4.json | e574cb7adedadd7745dba868c8850380 |
| dfa_v4_entry48_invs.md (战报) | deploy/artifacts/reports/dfa_v4_entry48_invs.md | 68b7b2baba9dcce1e6f3035ab34a69a1 |
| prof_real_pg.json (v4 pull) | deploy/artifacts/prof/pull_v4/prof_real_pg.json | dba5dc5f7ace61bc32697e19e69729e9 |

## 续十八：融合 MHA 侦察与方案（2026-10-05）

| 产物 | 路径 | md5 |
|---|---|---|
| mha_fusion_recon.md (MHA 方案) | deploy/artifacts/reports/mha_fusion_recon.md | 2693fe8b5ca2a585ecb79183376befbd |
| inspect_attention_chain.py | deploy/inspect_attention_chain.py | 1b3fddbf833895c7abd732d6643e0652 |
| inspect_attention_layout.py | deploy/inspect_attention_layout.py | 8a68f0297a9b8f1e27dca0448bc754e2 |


## 续十九（2026-10-05）：融合 MHA v1

| 产物 | md5 | 说明 |
|---|---|---|
| deploy/artifacts/plugin/dfa_plugin.cu | 802282996dd2e1018deefc2d8e0f8612 | DFA v4 + FusedMHA 插件（mha_sd kernel，3 bug 修后 bench 全绿） |
| deploy/artifacts/plugin/dfa_bench_mha.cu | 0a2faf7951a2bef7b54be73f2444f5d8 | MHA bench：5 档 S 图保真 fp16 参考对拍+确定性+计时+amp8 应力 |
| deploy/artifacts/sparsedrive_fp16_graph_fix3_mha.onnx | 6ed0d9f92c71e62ecdc7b25e2751032f | 7 块 12 节点链→FusedMHA，1531→1008 节点，checker 过 |
| deploy/make_graph_fix3_mha.py | b3f682da9b48c319ab260d1a0e82a98e | 图手术脚本（常量核对/消费者断言/死清除） |
| 板 /usr/local/lib/libdfa_sd.so | 56981c97dc1638f872b8cb6115c6cd79 | 现役：DFA v4+FusedMHA 超集 |
| 板 /usr/local/lib/libdfa_sd.v4.bak | a97114a91abc4103306559bb3adc5892 | DFA v4 回滚点 |
| 板 /opt/m0/sd2/engine/e_fix3mha.engine | 571d5f7561ae3aa9200c19546dbbdfd6 | **新生产引擎**（trtexec 194s，7×FusedMHA） |
| 板 /opt/m0/sd2/engine/e_fix3full_h.engine | 5f270db7b41cb7e6b8ab6aa2f0327965 | v4 引擎回滚点（已验证兼容新 .so） |
| deploy/artifacts/prof/pull_mha/prof_real_mha.json | 7c33be2374eeed69033383a814a2571b | 真数据逐层 profile（FusedMHA 5.00ms 实测） |
| deploy/artifacts/reports/m2_align_mha.json | 0bc805d7b4149128110a42b440018585 | M2 ALL PASS（0.0485/0.4579/Δ0.0354） |
| deploy/artifacts/reports/m3_board_pdms_mha.json | a31c0a7deb7e3c340c0f31f96ecfcd55 | M3 板对板 Δ−0.000156 PASS（0.7541630） |
| deploy/cmp_prof_mha.py | b94cd4f646c813d567ff61fa9cc08ee6 | A/B profile 对比脚本 |
| 板 /usr/local/bin/dfa_bench_mha | 23344af0dfd396f61e5fb61edc6670a5 | 板上 bench 二进制 |

e2e：34.2446→32.6675ms（graph 同口径，−1.58ms）；战绩线 71.30→32.67（−54.2%）。


## 续二十（2026-10-05）：MHA v2 flash-tile kernel

| 产物 | md5 | 说明 |
|---|---|---|
| deploy/artifacts/plugin/dfa_plugin.cu | 6adac279634506f07a6f35c4e35771ca | mha_fwd_kernel v2：smem K/V tile 双缓冲 + chunk 转置无冲突布局 + 预取管线 |
| deploy/artifacts/plugin/dfa_plugin.cu.bak_premha2 | 802282996dd2e1018deefc2d8e0f8612 | v1 warp-per-row 版备份（=续十九指纹） |
| 板 /usr/local/lib/libdfa_sd.so | c932c0a7e741665ec0f823c68d429e5e | 现役：DFA v4 + FusedMHA v2 |
| 板 /opt/m0/sd2/engine/e_fix3mha.engine | b4853d944ac404bac68da1ee4706d2e7 | **新生产引擎**（trtexec 30.6s 热缓存） |
| 板 /usr/local/bin/dfa_bench_mha | ca31ec8145e1943ba872ab1c325716dd | 板上 bench（v2） |
| deploy/artifacts/prof/pull_mha2/prof_real_mha.json | 7f9c70cdf8c02cebab234309c5e2c187 | v2 真数据逐层 profile（FusedMHA 2.00ms） |
| deploy/artifacts/reports/m2_align_mha2.json | 742cb4d3c324569934660329f094d2c3 | M2 ALL PASS（聚合与 v1 融合版全同） |
| deploy/artifacts/reports/m3_board_pdms_mha2.json | a31c0a7deb7e3c340c0f31f96ecfcd55 | M3 板对板 PASS（与 v1 融合版 md5 相同=输出逐位一致） |

e2e：32.6675→29.681ms（graph 同口径）；对 v4 基线 −4.56ms；战绩线 71.30→29.68（−58.4%）。


## 续廿一（2026-10-05）：ReduceSum_1 融合损失消除（DFA sumfusion v5）

| 产物 | md5 | 说明 |
|---|---|---|
| deploy/artifacts/plugin/dfa_plugin.cu | bd772758df2b95f2b81959b3026a6360 | gather v5：anchor 求和入核（warp-per-(anchor,chunk)+fp32 寄存器累加+SPLIT 自适应+确定性 finalize） |
| deploy/artifacts/plugin/dfa_bench_v5.cu | d57eeaffc2a73632e722181998c844e2 | v5 bench（S1/D + 三几何 case；宿主参考系不可信已注记，仲裁=M2/M3） |
| deploy/artifacts/sparsedrive_fp16_graph_fix3_sumf.onnx | a22f4192b74a8bb074ddd7bc87d82475 | 3 块删 Reshape_6/Cast/ReduceSum_1，插件输出直连 output_proj |
| deploy/make_graph_fix3_sumf.py | 4bb67762467c7cda1a26c4a208394e06 | sumfusion 图手术脚本 |
| deploy/inspect_reducesum.py | 7e5fa7f5339966d8ae101d004bdf12ab | ReduceSum 侦察工具 |
| 板 /usr/local/lib/libdfa_sd.so | 184c79f8595984ff9092b9e1b5de6333 | 现役：DFA v5(sumfusion) + FusedMHA v2 |
| 板 /usr/local/lib/libdfa_sd.presumf.bak | c932c0a7e741665ec0f823c68d429e5e | 回滚点：v2 时代 .so |
| 板 /opt/m0/sd2/engine/e_fix3sumf.engine | 407aebb874d47e277858aa0e70f1280f | **新生产引擎**（trtexec 29s 热缓存） |
| 板 /usr/local/bin/dfa_bench_v5 | d7fa308000fae2df6f575b3037529457 | 板上 bench 二进制 |
| deploy/artifacts/prof/pull_sumf/prof_real_sumf.json | 797fb2962911aa4b758131db80a77966 | 真数据逐层 profile（DFA 13.59→8.87，ReduceSum×3 消失） |
| deploy/artifacts/reports/m2_align_sumf.json | 50c4ffef5156df06a666a0ae16031e2c | M2 ALL PASS（traj_l1 0.0469 优于 v2） |
| deploy/artifacts/reports/m3_board_pdms_sumf.json | b473fc4f6f0e1cd5060f87f0aae04311 | M3：dev 协议 +0.0006 PASS；板对板 −0.00648 超门=单场景二值翻转已注记 |

e2e：29.681→22.916ms（graph 同口径，−6.76ms）；对 v4 基线 −11.33ms；战绩线 71.30→22.92（−67.9%）。


## 续廿二（2026-10-05）：backbone int8 诊断实验（净零，量化路线关闭）

| 产物 | md5 | 说明 |
|---|---|---|
| deploy/make_graph_fix3_sumf_bint8.py | （见文件） | backbone QDQ 嫁接脚本（fresh initializer scale + fp32 权重 + mini-topo 拼接） |
| deploy/artifacts/sparsedrive_fp16_graph_fix3_sumf_bint8.onnx | （见文件） | 33 conv 量化实验图 |
| deploy/cmp_prof_qdq.py | （见文件） | E1 回归归因工具（逐 kernel Δ） |
| 板 /opt/m0/sd2/engine/e_fix3sumf_b8.engine | c7d3eeb0a78a5c9f767a94897455de09 | 实验引擎（非生产） |
| 板 /opt/m0/sd2/prof/prof_real_b8.json | （见文件） | b8 逐层 profile |

结论：int8 conv 本体 2.3× 但被 act-Q(0.43)+reformat(0.41)+maxpool(0.08) 抵消，
e2e 22.911 vs 22.916 净零。生产维持 e_fix3sumf。历史 E1 回归 +4.16ms 归因=
myelin 对量化 GEMM 区域重规划（非 reformat）。


## 续廿二更正（同日晚，折叠误读澄清）

- e_fix3sumf_b8.engine md5 更正为 **f3591ba8**（b8v5 终版：fp32 权重 +
  axis=0 + int8 zp，c7d3eeb0 为中间版）。
- 机理更正：profile 融合组名 "weight + QuantizeLinear + Conv" ≠ 未折叠
  （TRT 以源节点命名融合组，e_sd2 同款）；精确对账 int8 conv 2.296→1.636ms
  （−0.660，−29%）被 act-Q 0.448 + reformat 增量 0.357 + maxpool 0.08 抵消
  （+0.225 ≈ 实测 +0.223），e2e 22.9025 vs 22.916 净零。生产维持 e_fix3sumf。
