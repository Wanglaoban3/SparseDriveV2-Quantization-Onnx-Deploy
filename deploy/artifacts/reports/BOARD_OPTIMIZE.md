# SparseDriveV2 板端延迟优化战役（71.30 → 22.92 ms，−67.9%）

本文件记录把 SparseDriveV2 在车端 Orin 上的单帧端到端延迟从 71.30ms 压到 22.92ms 的
完整过程：每一步做了什么、带来多少收益、怎么验证的、踩过什么坑、哪些路线被证伪关闭。
所有数字均为板上实测，证据文件（逐层 profile / 门禁报告 / 战报）随仓库分发，
 见 `deploy/artifacts/reports/` 与 `deploy/artifacts/prof/`。

## 1. 测试环境与设备

| 项 | 值 |
|---|---|
| 目标设备 | NVIDIA Orin（sm_87，16× Ampere SM，L2 4MB，aarch64），AGX Orin 级量产域控 |
| TensorRT | 8.6.1.2（`/usr/src/tensorrt/bin/trtexec`） |
| CUDA / 编译器 | CUDA 11.4（nvcc），插件 `-O3 -std=c++17 -arch=sm_87` |
| 板端 python | 3.8（仅 TRT bindings，无 numpy 也能跑 parse 检查） |
| 生产引擎 | fp16 全图 + 自定义插件（DFA v5 + FusedMHA v2），CUDA Graph 模式计时 |
| 延迟口径 | `run_engines2`（自研 C++ runner）cudaEvent 100 iters MEAN，warmup 10，
  真实输入（24 个 held-out 样本之一）；`--graph` 为 CUDA Graph 捕获模式 |
| 逐层 profile | `trtexec --loadEngine --dumpProfile --loadInputs=<真实输入>`，100 iters × avg 100 |
| 工作目录 | 板上 `/opt/m0/sd2`；产物只进 `/usr/local/{bin,lib}`（工作目录可能 noexec） |

宿主机只做图手术与量化（纯 Python，无 GPU 依赖）；所有 engine 编译、延迟与精度
验证都在目标板上完成。

## 2. 方法论：三轮门禁 + 回滚链

每轮优化固定走同一流程，**任何一步不过门禁就不转正**：

1. **归因**：trtexec 逐层 profile（真实输入）+ 必要时自研 bench（对拍 + 确定性 + 计时），
   找到最大可优化桶；
2. **改造**：改 CUDA 插件（`dfa_plugin.cu`）或 ONNX 图手术（`deploy/make_graph_*.py`）；
3. **bench 门禁**：插件级数值对拍（fp16/fp32 参考系）+ 确定性（两次运行逐位一致）；
4. **M2 精度门禁**：24 个 held-out 样本逐样本 dump，`traj_l1 / score MAE / metric MAE`
   对照基准（门限 traj_l1≤0.125、metric MAE Δ≤0.06 等，见 `reports/m2_align_*.json`）；
5. **M3 端到端门禁**：138 场景（OpenScene mini 分片 0）板上推理 dump → navsim PDMS
   评分（`deploy/eval_board_pdms.py`），与基准协议 Δ≤0.005，超门必须逐场景归因
   （见 `reports/m3_board_pdms_*.json/csv`）；
6. **转正或回滚**：`.so` 与引擎均有 md5 备份链（`libdfa_sd.{v3,v4,presumf}.bak` →
   `e_fix3mha` → `e_fix3full_h`），一键回退。

## 3. 战绩线总表

| # | 轮次 | 内容 | e2e (ms) | Δ | 证据 |
|---|---|---|---|---|---|
| 0 | 基线 | e_fix2full（fp16 引擎，eager；fp32 引擎为 99.16） | 71.30 | — | `m4_bottleneck.md` |
| 1 | R1 | DFA fp16 原生 IO（w/loc 直读半精度）+ CUDA Graph（launch 开销 22.9ms/32% 消除） | 62.18 | −9.12 | `m4_bottleneck.md` |
| 2 | R2 | Pg：softmax+plan+gather 全并入 DFA 插件（fix3 图手术，3 节点换 Pg） | 54.58 | −7.60 | `m5_dfa_pg.md` |
| 3 | R3 | gather half2 向量化 + plan phase B 并入 phase A | 40.13 | −14.45 | 台账/prof |
| 4 | R4 | gather v3：uint2 tap、4 通道/线程、block 协同暂存 | 35.42 | −4.71 | `prof/prof_real_pg.log` |
| 5 | R5 | DFA v4：Entry 64B→48B（wg half 化）+ phase C 除法→乘法 | 34.24 | −1.18 | `dfa_v4_entry48_invs.md` |
| 6 | R6 | FusedMHA v1：7 块 × 12 节点注意力链 → 单插件 kernel（flash 风格） | 32.67 | −1.58 | `mha_fusion_v1.md` |
| 7 | R7 | FusedMHA v2：smem K/V tile 双缓冲 flash-tile（chunk 转置布局 + 预取管线） | 29.68 | −2.99 | `mha_fusion_v1.md` |
| 8 | R8 | sumfusion v5：anchor 级求和收进 gather 核（消 262MB ReduceSum 往返） | 22.92 | −6.76 | `dfa_sumfusion.md` |

累计 **71.30 → 22.92ms（−67.9%）**。每轮都过了 M2+M3 才转正；M3 的逐场景 CSV 全部留档。

## 4. 各轮要点

### R1：DFA half-native + CUDA Graph（71.30 → 62.18）
profile 显示 launch 开销 22.9ms（32%）——几百个小 kernel 的启动税；DFA 插件此前把
w/loc staging 成 fp32 再算。改为 fp16 直读（kernel 内 half 访问）+ `run_engines2 --graph`
CUDA Graph 捕获。两者正交，各贡献约一半。

### R2：Pg（softmax/plan/gather 并入插件，54.58）
原先"插件 gather → 图上 Reshape/Cast/Softmax → 再 gather"三段往返；把 softmax 与
采样 plan（top-k 压缩 + entry 打包）并入插件后，中间大张量不再落全局显存。

### R3+R4：gather 向量化两轮（40.13 → 35.42）
真数据流量模型（`deploy/pg_traffic_model.py`：feat 4.0GB/帧）证明瓶颈在访存宽度和
调度：half2 向量化 + plan 相位合并（R3），再换 uint2 tap、每线程 4 通道、block 协同
暂存（R4）。R4 后 gather 输出与标量版**逐位一致**（`board_outs/v3chk` 对拍）。

### R5：DFA v4 Entry 压缩（34.24）
采样 entry 从 64B 压到 48B（权重 half 化）+ softmax 分母除法换乘法。带宽敏感性
直接兑现。M2 与 v3 同值，M3 PASS。

### R6+R7：FusedMHA v1→v2（32.67 → 29.68）
decoder 两层共 7 块注意力（12 节点链/块：QKV 投影 + bmm + softmax + bmm）融合为单个
`FusedMHA` 插件节点。v1 warp-per-row；v2 改 flash-tile（8 query 行 × 1 头/block，
smem K/V tile 双缓冲 32KB，chunk 转置布局保证 LDS.128 无冲突 + 预取管线）。
v2 数值与 v1 **逐位一致**，7 kernel 合计 2.0ms，已达 LDS 带宽地板（l0/p 1.32ms）。
见 `mha_fusion_recon.md` / `mha_fusion_v1.md`。

### R8：sumfusion v5（22.92）
DFA 输出 [1, A×pts, 256] 的 anchor 级 ReduceSum 在图上产生 **262MB** fp16 写+读往返
（262MB / 4MB L2 ≈ 65 轮全量驱逐，殃及整条流水线）。v5 把求和收进 gather 核：
warp-per-(anchor,chunk)、fp32 寄存器累加、SPLIT 按 anchor 数自适应 + 确定性 finalize，
插件输出直接是 [bs,A,C]。图上删除 Reshape_6/Cast/ReduceSum_1 ×3 块。
M2 traj_l1 0.0469 优于上一版；M3 dev 协议 +0.0006 PASS。见 `dfa_sumfusion.md`。

## 5. 负结果与关闭的路线（同样重要）

### 5.1 myelin 区 Linear INT8（三剂量判负）
把 ModelOpt 校准 QDQ 嫁接回 fix3 图的 MatMul（E1: weights_fc；E2: 全注意力/FFN）：
E1 **+4.16ms**（回归 100% 在 myelin ForeignNode 内部——reformat 全程 0.115ms 无辜，
是 myelin 对含 QDQ 区域的重新规划），E2 曾把 e2e 推回 39.57ms。全部回滚，
路线关闭。见 `m6_myelin_int8.md` / `myelin_matmul_inventory.md`。

### 5.2 backbone INT8（净零，机理已对账）
33 个 backbone conv 嫁接校准 QDQ（A8W8，权重 fp32 + per-channel axis=0 + int8 zp）：
逐 conv 精确对账——**int8 conv 真实生效**（融合组 1.636ms vs fp16 2.296ms，−0.660ms/−29%），
但被 act-Q 独立 kernel（0.448ms，30 个 Q 仅 17 个独立出现，TRT 8.6.1/Orin 不把 Q 融进
ReLU epilogue）+ 精度边界 reformat 增量（0.357ms）+ maxpool（0.08ms）全额抵消：
−0.660 + 0.885 = +0.225ms，与实测 backbone/neck +0.223ms 分毫吻合，e2e 净零。

**判坑提醒**：引擎 profile 里 `weight + QuantizeLinear + Conv` 式融合组名 **≠ 未折叠**
——TRT 用源节点命名融合组（对照：老 ModelOpt 引擎 e_sd2 的 layerinfo 同款 68/114 层，
现役 fp16 引擎的 `{ForeignNode[...weight+Shuffle...]}` 同理）。判断是否折叠要看逐层
时间与 layerinfo 精度，不能看层名。

尺度分析：增益 ∝ 计算量（O(C²)），量化税 ∝ 激活字节（O(C)）——当前 backbone
（ResNet-34 级，conv 仅占 e2e ~10%）下税大于益；conv 占比 >20% 的更大 backbone 或
Q-fusion 更好的 TRT 版本上才值得重启。

### 5.3 其他
- fp32 引擎对照 99.16ms（fp16 全图收益 −28ms，早已锁定）；
- 不用 timing cache 的构建对照 e_fix2nt：71.33 ≈ 71.30（timing cache 只影响构建时间，
  不影响运行时）；
- 早期 w（softmax 采样权重）INT8：相对误差 19–25%，精度上不可行（见根 README
  BOARD_DEPLOY 四轮记录）。

## 6. 终态延迟分布（22.92ms）

| 桶 | 时间 | 说明 |
|---|---|---|
| DFA 聚合 ×3 | 10.67ms | 数据依赖 gather，两小调用 1.8ms |
| myelin ForeignNode ×4 | 5.85ms | anchor/keypoint 编码、投影、weights MLP（含 98MB logits GEMM） |
| FusedMHA ×2 | 1.60ms | LDS 地板 |
| backbone+neck conv | ~3.9ms | TRT fp16 融合已做满，int8 净零 |
| 杂项（Reshape/Transpose/reformat） | ~0.9ms | 大半可经布局协商消除（见 FUTURE_WORK） |

`prof/pull_sumf/prof_real_sumf.json` 为逐层证据。

## 7. 复现：板上全流程 stage 清单

宿主机（任意 Linux + Python3，无 GPU 要求）通过 `deploy/board/board_v2.py` 驱动板端；
凭据只从环境变量注入（`BOARD_HOST` / `BOARD_PASS`，可 `BOARD_USER`，默认 root），
**不写入任何文件**：

```bash
export BOARD_HOST=<board-ip> BOARD_PASS=<password>
python3 deploy/board/board_v2.py <stage>
```

| 阶段 | stage 序列 | 作用 |
|---|---|---|
| 0 环境 | `check` | 连通性 + TRT/CUDA 清点 + 后台/CRLF 规范自检 |
| 1 插件 | `plugin` | push 插件源码 → nvcc 编 `libdfa_sd.so` → 6 用例 kernel 单测 |
| 2 输入 | （宿主机）`prep_engine_inputs.py` / `dump_board_inputs.py` | 生成 24 held-out 样本与 138 场景输入树 |
| 3 首编 | `push_onnx` → `parse_check` → `build_engine` → `baseline` → `profile` | 推 ONNX（双向 md5）→ 插件契约解析检查 → trtexec 编引擎 → 延迟基线 → 逐层 profile |
| 4 评测 | `dump24` → `push138`/`run138`/`fetch138` | 24 样本 dump（M2）与 138 场景 dump（M3 PDMS） |
| 每轮优化 | `pgbench`/`pg4`/`mha`/`mha5`（插件轮）或 `pgbuild`/`mhabuild`/`mha5build`（换图轮）→ `pgval`/`mhaval`/`mha5val` → `pgm2`/`mham2`/`mha5m2` → `run138pg`/`run138mha`/`run138sumf` | bench 门禁 → 重编引擎 → 计时+profile → M2 → M3 |
| 归因 | `pgreal`、宿主机 `cmp_prof_mha.py` / `cmp_prof_qdq.py` / `profile_buckets_v2.py` | 真数据六对照 bench、A/B profile 逐 kernel 归因 |

构建纪律：**不整引擎反复重编**——`--timingCacheFile` 热缓存下 30s 级；
ONNX 推送双向 md5；所有后台任务 DONE/FAIL 双 marker 轮询；文本落板强制 LF。

## 8. 产物谱系（生产链回滚点）

```
插件 .so：libdfa_sd.v3.bak → libdfa_sd.v4.bak → libdfa_sd.presumf.bak → libdfa_sd.so（现役 184c79f8…，DFA v5 + FusedMHA v2）
引　　擎：e_fix2full → e_fix3full_h(5f270db7…) → e_fix3mha(b4853d94…) → e_fix3sumf(407aebb8…，现役 22.92ms)
```

逐轮产物 md5 指纹见 `reports/artifacts_manifest.md`。
