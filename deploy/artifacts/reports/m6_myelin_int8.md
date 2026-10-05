# M6 战报：myelin Linear int8 —— 三剂量实测，路线关闭（2026-10-05 晚）

## 结论

**myelin（14.41ms，四个融合大区）的 int8 量化在本栈（TRT 8.6.1.2 + sm_87 Orin X）上
是净负收益，无论以何种剂量施加。** 路线关闭。生产引擎维持 e_fix3full_h
（fix3 图 + fp16，e2e graph 35.42ms）不变。

## 背景

myelin 桶 = trajectory_head decoder 的 4 个 Myelin ForeignNode 融合区，
14.41ms（41.5% of graph）：

| 区 | ms | 内容 |
|---|---:|---|
| A：backbone→layers.0/p_deform/Reshape_2 | 3.26 | camera_encoder + **weights_fc(l0) 25.2 GFLOP**（全图第一大 GEMM，[3072,256]×[256,16000]）+ kps/loc 链 |
| B：camera_encoder→layers.1/p_deform/Reshape_2 | 7.67 | layers.0 attention/ffn/mlp + **weights_fc(l1) 3.1 GFLOP** + l1 p_deform 准备 |
| C：v_img_attention→layers.1/t_deform/Reshape_2 | 2.66 | attention 投影 + t_deform 链 |
| D：layers.1 heads 区 | 0.83 | traj_mlp + 6 个 metric 头 |

总 GEMM ~36 GFLOP，14.4ms → 有效 ~2.5 TFLOPS：一半是 GEMM，一半是
K=32 的批量 attention score（形状残废）与 LN/softmax/elementwise 链（带宽型）。
int8 理论天花板 ~3ms。

## 嫁接方案（无需重新校准）

deploy/make_graph_fix3_qdq.py：从 sparsedrive_int8_qdq.onnx（modelopt PTQ 图）
把目标 MatMul 的 Q/DQ 子树（per-channel 权重 scale + per-tensor 激活 scale）
逐字节移植进 fix3 图。权重 348/348 与现役图一致（此前已验证），校准值直接复用。
排除项：kps_generator 与 MatMul_fxG（产出 loc，精度敏感）、camera_encoder、
各 mlp/metric 头最终层（.2，直接打门禁指标）、attention score 批量 MatMul。
每个嫁接做权重 roundtrip 校验（dequant(quant(W)) 误差 ≤ scale/2，
per-channel axis=0 全部正确）+ scale/zp 与 donor 逐字节比对
（deploy/verify_qdq_graft.py，PASS）。

## 实测（真数据 val_00，100 iters）

| 实验 | 配置 | myelin | e2e graph | vs 基线 |
|---|---|---:|---:|---|
| 基线 | fix3 --fp16 | 14.41 | 35.42 | — |
| **E1** | weights_fc×2 QDQ，--int8 --fp16 | **18.57** | **39.57** | **+4.15ms** |
| e1n | 同上，无 timing cache | ~18.6 | 39.59 | 缓存无关 |
| e1f | 同图只 --fp16 | — | — | 本机 TRT 拒建（QDQ 必须 --int8，Task 4 同坑） |
| **e_sd2（全图 int8）** | modelopt 原生全图 QDQ | **50.99** | — | vs 同构 fp16（fix2 era 22.4）**+28.6ms（2.3×）** |

分区看 E1：区 A +2.84ms、区 B +1.31ms、未嫁接的 C/D 恒 0 ✓——
伤害严格跟着 QDQ 走。B 区里 0.5ms 级的小 GEMM 倒贴 1.31ms，说明这不是
"int8 GEMM 不够快"，而是 **QDQ 插入破坏了区级融合计划**：
bias Add 不再吃进 GEMM epilogue，weights_fc 输出（49M 元素，98MB fp16）
每多一遍读写 ≈1.3ms，与实测增量吻合。全图 int8（e_sd2）下 reformat 节点
0.1→2.9ms（12→27 个），同一机制在所有区放大。

对照组发现：backbone conv int8（cuDNN 路径）不受影响——e_sd2 的
img_backbone 桶 4.03ms vs fp16 4.65ms（还略快）。**塌的只有 Myelin 融合区**。

## 判读

1. 本机 Myelin 在 sm_87 上对 int8 QDQ 区没有可用的 IMMA 映射路径：
   requant 搬运 + epilogue 断裂的代价稳定大于 int8 GEMM 的理论收益
   （该收益本来就被 K=32 attention / 带宽型链路稀释到 ~3ms）。
2. "myelin 14.4ms Linear int8 是下一顿正餐"的原路线假设**被实测否定**。
3. e2e 35.42ms 之下，部署侧剩余空间：DFA plan phase C logits 重读
   （~0.8-1ms）、Entry 64B→48B（LDS −25%，~1ms）；myelin 与 backbone
   分别被证明 fp16/cuDNN 已饱和。不改模型结构（如导出融合 MHA）的话，
   现实终点 ~33-34ms。

## 产物

- deploy/make_graph_fix3_qdq.py、verify_qdq_graft.py、
  inspect_matmul_fix3.py、inspect_donor_chain.py
- deploy/artifacts/reports/myelin_matmul_inventory.md（fix3 全部 79 个
  MatMul 形状/分区/donor 对齐清单）
- deploy/artifacts/sparsedrive_fp16_graph_fix3_qdq_e1.onnx（E1 图，留档）
- deploy/artifacts/prof/qdq_e1/{qdqv_e1_prof.log, esd2_prof.log}
- board_v2.py 新增 stage：qdqbuild/qdqval/qdqm2（e1/e2/e1n/e1f 变体）
- 板上 e_fix3full_h_qdq_* 实验引擎与 e1 onnx 已清理（盘 88%）
- 生产资产不变：e_fix3full_h.engine + libdfa_sd.so(8119ec18)
