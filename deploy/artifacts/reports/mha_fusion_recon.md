# 融合 MHA 导出尝试 —— 侦察与方案（准备稿）

日期：2026-10-05　|　状态：侦察完备，待实施
基线：v4 DFA 转正后 **34.24 ms**（graph，val_00）；myelin ForeignNode 栈 ~14.4ms

## 1. 拓扑侦察（fix3 图，三层同构）

每个 attention 块（自研 MHA 工具链 inspect_attention_chain.py / inspect_attention_layout.py）：

```
x[1,S,256] → in_proj MatMul[256,768]+bias → x768[1,S,768]
  → Reshape [1,S,3,8,32] → Transpose perm[2,0,3,1,4] → [3,8,S,32]
  → Gather idx0/1/2 切 Q/K/V（各 [1,8,S,32]）
  → Q: Mul(0.17677669 = 1/√32)           K: Transpose[0,1,3,2]
  → scores = MatMul(Q,Kᵀ) [1,8,S,S]
  → Softmax(-1)                            ← 无 mask！
  → ctx = MatMul(P,V) [1,8,S,32]
  → Transpose[0,2,1,3] → Reshape [1,S,256] → out_proj MatMul[256,256]
```

关键事实：
- **in_proj 已是融合 QKV**：768 = Q|K|V 三个连续 256 段（S-major：x768[s, seg, 256]）
- **无 attention mask**（decoder self-attn），scale 精确 = 1/√32
- **8 头 × head_dim=32，B=1 静态**（t_attention 的 reshape 形状 [1,400,3,8,32]
  把上游动态维钉死为 1）
- Q/K/V 切分用 Gather(index) 而非 Split——myelin 未将其识别为标准 MHA 模式

7 个块清单（scores 形状 → fp16 材料化大小）：

| 块 | S | scores [1,8,S,S] | 材料化 |
|---|---|---|---|
| l0/p_attention | 1024 | 16.8MB | 最大头 |
| l0/v_attention, l0/v_img_attention | 256 | 1.05MB ×2 | |
| l1/t_attention | 400 | 2.56MB | |
| l1/p_attention | 128 | 0.26MB | |
| l1/v_attention, l1/v_img_attention | 64 | 0.07MB ×2 | |

## 2. 收益核算

- scores 材料化流量：写(BMM)+读+写(softmax)+读(ctxBMM) = 4×16.8MB ≈ 67MB
  （仅 l0/p）→ 全部块合计 ~85MB @151GB/s ≈ **0.56ms 纯流量下限**
- K=32 批量 BMM 形状残废（M=8×S, K=32, N=S/32, 8 batch）：tensor core 利用率差
- 每块 8 个节点（Gather×3/Mul/Transp×2/BMM×2/Softmax）→ 1 个 kernel，
  launch 与中间 buffer 生命周期开销全消
- **预期：保守 0.5-1ms，乐观 1.5-2ms**（相对 34.24）

## 3. 路线选型

- **A. TRT 自带 FMHA plugin**（libnvinfer_plugin dis_fused_mha）：head_dim=32
  支持矩阵不明（主线要求 64）、S=400 非 64 倍数、输入布局胶水复杂 → 弃
- **C. 仅 ONNX micro-fusion**（折叠 Gather/Mul/scale）：myelin 已做常量折叠，
  改不动 BMM/softmax 分离的本质 → 弃
- **B. 自研 fused MHA 插件（推荐）**：head_dim 32 + S∈{64..1024} 小集合、无
  mask、fp16 I/O，完全是 DFA 插件的技术复刻（online softmax 在 plan kernel 已
  写过一遍）；基建全在（.so 双 creator、graph surgery、bench、M2/M3 门禁）

## 4. 插件设计草案（路线 B）

- **边界**：输入 = in_proj+bias 输出 `x768 [1,S,768]`（half），输出 = 合并布局
  `[1,S,256]`（half）；in_proj/out_proj/残差/LN 留在 TRT
- **替换节点集**（每块 8 个）：Gather×3、Q 的 Mul、Transpose×2（Q 布局与 Kᵀ）、
  scores MatMul、Softmax、ctx MatMul_1、Transpose_2、Reshape_1
- **kernel 骨架**：grid=(⌈S/64⌉, 8头)，block=128 线程；
  Q tile [64,32] smem；S≤256 时 K/V 全量驻留 smem（≤32KB），S=1024 分 128 行
  tile 走 online softmax（fp32 累加，同 DFA plan 的 m/s 双变量法）；
  scores 用 FFMA（K=32，MMA 收益有限可二期）；输出 half2 直接写合并布局
- **权重不入插件**：零新常量注入，插件 fields 只要 S/H/D 三元组
- 图手术：在 fix3 图上按块名前缀定位 8 节点链 → 删除 → 插入 Plugin 节点，
  复用 make_graph_fix3 的 kept-nodes 拼接模式

## 5. 风险与对策

1. **myelin 分区被切开（最大风险）**：插入插件节点会改变 ForeignNode 边界，
   weights_fc 等 GEMM 的现有融合可能退化——myelin int8 战役的直接教训
   （E1 嫁接 +4.15ms 倒贴）。对策：**A/B 全图 profile 对比**（v4 vs v4+mha），
   只看 attention 链改善不算数，e2e 与全图层耗时必须不劣化
2. 数值：fp32 accum + online softmax，偏差预期 half-ULP 级（同 v4 wg-half 量级），
   M2（traj_l1≤0.125，现值 0.0534）/M3（PDMS Δ≤0.005，新基线 0.7543192）门禁不变
3. Gather 索引常量未逐块核对（侦察了 3 块均同构）——实施脚本按常量值断言
   idx∈{0,1,2} 全 7 块校验后再动刀

## 6. 实施步骤（复用 pg4 流水线）

1. mha_plugin.cu：kernel + creator（"DeformableMHA" 类命名，注册进 libdfa_sd.so，
   双 creator 兼容旧引擎）+ dfa_bench_mha.cu（随机+真数据对拍）
2. make_graph_fix3_mha.py：7 块图手术 → fix3_mha.onnx → trtexec 构建
   e_fix3mha.engine（对照引擎 e_fix3full_h 复用）
3. 板上：pg4 模式新 stage（push/编 .so/bench 对拍）→ pgval 计时 + 全图 profile
   A/B → M2 → M3
4. 判据：e2e 改善 ≥0.5ms 且 M2/M3 PASS → 转正；myelin 分区劣化 → 回滚 .so+engine
