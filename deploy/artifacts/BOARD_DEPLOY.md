# SparseDriveV2 板端 TRT 部署说明

## 交付物清单（SparseDriveV2/deploy/artifacts\）
- `sparsedrive_fp32.onnx`          FP32 ONNX，含3个自定义节点 sparsedrivev2::DeformableAggregation（opset 17, bs=1静态shape）
- `sparsedrive_int8_qdq.onnx`      INT8显式量化ONNX（QuantizeLinear/DequantizeLinear QDQ节点，ModelOpt PTQ，敏感层已回退）
- `calib/calib_00.npz .. 15.npz`   校准样本（imgs/proj/iwh/status，uint8范围已归一化float32）
- `plugin/dfa_plugin.cu`           DFA算子 TRT 插件（IPluginV2DynamicExt，fp32/fp16 IO，内部fp32计算）
- `plugin/CMakeLists.txt`          插件编译脚本
- `build_engine.py`                引擎构建脚本（fp16/int8-implicit/int8-explicit均可）
- `ptq_report_*.json`              量化精度报告（开发机实测）
- `sensitivity.json`               逐层敏感度排序（int8退回fp16的依据）

## 引擎输入/输出
输入（bs=1, NCHW, float32）:
| 名称 | 形状 | 说明 |
|---|---|---|
| imgs | (1,3,3,256,512) | 3相机(cam_l0,cam_f0,cam_r0)已归一化图像 |
| projection_mat | (1,3,4,4) | lidar2img（含resize/crop增广补偿） |
| image_wh | (1,3,2) | 各相机(宽,高) |
| status_feature | (1,8) | driving_command(4)+ego_vel(2)+ego_acc(2) |

输出:
| 名称 | 形状 | 说明 |
|---|---|---|
| trajectory | (1,8,3) | 选中轨迹 8个位姿(x,y,θ)，0.5s间隔 |
| traj_scores | (1,400) | 400条组合轨迹得分 |
| metric_* ×6 | (1,400) | no_at_fault_collisions/drivable_area_compliance/driving_direction_compliance/ttc/comfort/ego_progress 的logits |

后处理（板端或上位机）: trajectory 直接就是最终规划输出（argmax选择已在图内完成）。

## 板端编译步骤
1. 编译插件:
   ```
   cmake -B build -DTENSORRT_ROOT=<trt> -DCMAKE_CUDA_ARCHITECTURES=<板子sm>
   cmake --build build -j
   ```
   TRT 8.x/9.x: IPluginV2DynamicExt 直接可用。TRT 10: 若需要 IPluginV3 需小改（接口适配约30行）。
2. 构建 engine（插件先加载）:
   ```
   python build_engine.py --onnx sparsedrive_int8_qdq.onnx --out engine_int8.plan \
       --plugin build/libdfa_plugin.so --fp16
   ```
   （QDQ模型用--fp16即可：Q/DQ节点会被builder融合成INT8，DFA插件跑FP16/FP32。）
   隐式量化备选：`--onnx sparsedrive_fp32.onnx --int8 --calib-dir calib`
3. 推理: 常规 enqueueV2 流程，输出中 trajectory 即结果。

## 量化精度结论（fake-quant 实测，24个held-out样本；优化后图：冻结词表+可量化MHA+deform去重）
| 配置 | argmax一致率 | 轨迹平均偏差 | score MAE | metric MAE | 相对GT距离 |
|---|---|---|---|---|---|
| FP32基线 | - | - | - | - | 1.0923m |
| 全INT8(PTQ) | 20.8% | 9.2cm | 0.683 | 1.012 | 1.0834m |
| INT8+敏感层回退(12层) | 33.3% | 9.9cm | 0.461 | 0.703 | 1.1194m |
| +蒸馏QAT(保留最优) | 20.8% | 9.5cm | 0.450 | **0.656** | **1.0925m** |

- 规划质量结论：最终量化模型相对GT的规划距离1.0925m vs FP32的1.0923m，**差异0.2mm量级**
- **端到端PDMS验证**（138场景同口径，`deploy/pdms_eval_quant.py` + `deploy/pdms_eval_configs.py`）：
  FP32 0.7440 / 全INT8 0.7432 / 保护PTQ 0.7533 / 最终交付（fake-quant）0.7471——四配置全部在
  临界场景翻转噪声内，最终交付与FP32逐分一致111/138场景，数据集指标与原版持平，
  证据见 `pdms_report.json`、`pdms_configs_report.json` + `pdms_*.csv`
- **navtest全量验收**（OpenScene test split，12,146场景，`deploy/pdms_eval_navtest.py`；
  量化候选仍用mini校准样本，与交付QDQ ONNX同源）：FP32 **0.9141** / 全INT8 PTQ 0.9131（−0.0009）/
  保护PTQ 0.9138（−0.0002）/ **最终交付 0.9138（−0.0003）**——大样本下排序恢复单调，
  量化损耗全部压到0.001量级，mini 138场景的"反升"确认为小样本临界场景翻转噪声。
  **最终交付在官方navtest协议下与FP32相比精度无损**，证据见 `pdms_navtest_report.json` + `pdms_navtest_*.csv`
- argmax翻转发生在得分接近的相似轨迹之间（400选1），几何影响很小（traj_l1≈9.5cm）
- MHA投影已通过模块替换纳入INT8量化（169→189对Q/DQ节点）
- 敏感层Top：_status_encoding（收益极小的tiny Linear）、v_attention/v_img_attention投影、backbone.layer1卷积（见sensitivity.json，已回退FP16）
- DFA自定义节点保持FP16/FP32；蒸馏QAT用fp32 teacher自蒸馏（无标签），keep-best保证不劣于PTQ

## 编译前优化清单（本轮已完成，输出等价性已验证 max|Δtraj|=0）
1. 冻结词表嵌入预计算：path/vel_pos_embed两个MLP和25MB词表repeat从每帧计算折叠为ONNX常量初始化器
2. deformable_format/img_value提升到decoder级只算一次（原来每层各算一遍，2×→1×）
3. nn.MultiheadAttention → QuantizableMHA（等价Linear堆叠），使注意力投影可INT8量化
4. 图规模：Constant 732→407、Unsqueeze 233→119、Transpose 99→70（静态shape下TRT还会进一步折叠）

## DFA算子INT8量化评估（四轮实测，deploy/dfa_int8_eval.py + deploy/qat_wloc.py + deploy/kd_qat_v3.py）
### 第一轮：feat/loc/w全int8 → ❌ 否决
- DFA输出相对L2误差20~25%（cosine 0.97），端到端metric MAE≈0.40（与全模型QDQ总漂移0.656相当）
- 根因：双线性插值误差不抵消；loc含相机外投影离群值，数据驱动amax被撑爆会导致输出全零（须固定[0,1]量化域+裁剪）

### 第二轮：仅feat做int8（loc/w保持FP16/FP32）→ ✅ 通过（8个held-out样本）
| 量化方案 | DFA输出relL2 | cosine | 端到端metric MAE | argmax一致 | GT距离 |
|---|---|---|---|---|---|
| per-tensor（1个scale） | 2.2~2.5% | 0.9997 | 0.0495 | 100% | 1.3855m |
| **per-C（256个scale）** | **1.5~1.7%** | **0.99986** | **0.0411** | **100%** | **1.3889m(=FP32)** |
| per-point（fp16 scale） | 0.47~0.50% | 0.99999 | 0.0104 | 100% | 1.3889m(=FP32) |
| per(cam,C) | 1.4~1.5% | 0.9999 | 0.0227 | 87.5% | 1.3855m |

参考：同批样本全模型QDQ总漂移为metric MAE 0.656——feat-int8的增量误差仅为其1/16。

### 第二轮补充：DFA各输入int8误差分解（8样本，fake-quant模拟int8存储）
| 量化对象 | DFA输出relL2 | cosine | 端到端metric MAE | GT距离 | 结论 |
|---|---|---|---|---|---|
| **仅w**（196MB，最大张量） | **19~25%** | 0.968~0.981 | **0.412** | 1.3999m | ❌ 误差主要来源 |
| 仅loc（12MB） | 3.3~3.7% | 0.9995 | 0.072 | 1.3889m(=FP32) | ✅ 可接受 |
| w+loc | 19~25% | 0.968 | 0.408 | 1.4011m | ❌ |
| feat+w | 19~25% | 0.981 | 0.402 | 1.4418m | ❌ |
| 全部 | 20~25% | 0.979 | 0.396 | 1.4412m | ❌ |

- **w是精度杀手**：softmax权重绝对值小（常<0.25），固定[0,1]域int8步长1/254的绝对误差在小权重上相对误差被放大，直接乘在采样值上不抵消
- **w的正确省带宽方式是FP16直读**（196→98MB，2×节省，fp16对[0,1]权重精度足够）；当前插件对fp16的w是staging到fp32再算，如需带宽可加fp16直读分支（kernel改动约10行）
- loc int8几乎免费（GT与FP32一致），但张量小(12MB)收益有限

### 第三轮：w/loc int8 + QAT尝试 → ❌ QAT无法救回（数据见 qat_wloc_report.json）
在w/loc插入STE伪量化、用真实PDM metric监督（token路径映射到138个mini指标缓存）+path/vel/traj损失
微调24步（lr 1e-5, bs 4, 138样本）：
| 模型 | metric MAE | traj_l1 | GT距离 |
|---|---|---|---|
| FP32原模型 | — | — | 1.0923m |
| w/loc int8（QAT前） | 0.812 | 15.1cm | 1.1341m |
| w/loc int8 + QAT | **1.905** | **49.1cm** | 0.8533m |
| QAT后模型的FP32路径 | 1.933 | 53.2cm | 0.8998m |

- QAT后漂移不降反升（metric MAE 0.81→1.91），且FP32路径本身也被带坏（GT 0.90 vs 原FP32输出1.09）
- 根因：50M预训练模型 + 仅138个mini场景微调 → 灾难性遗忘；loss量级(20~35)相对lr过猛
- **最终结论：w/loc不做int8，保持FP16直读**（已实现，误差~1e-6量级）。
  w/loc的int8+QAT只有拿到完整navtrain数据与正式训练配置才值得重启

### 第四轮：全INT8（取消12层回退）+ 蒸馏QAT → ❌ 不如现役配置（kd_qat_v3_report.json）
| 配置 | metric MAE | traj_l1 | GT距离 |
|---|---|---|---|
| 全INT8 PTQ | 1.198 | 17.7cm | 1.1275m |
| 全INT8 + KD蒸馏16步 | 1.166 | 16.3cm | 1.1210m |
| **现役配置（feat-int8+12回退，w/loc FP16）** | **0.656** | **9.5cm** | **1.0925m** |

- 蒸馏仅挽回~3%，全INT8的metric MAE仍是现役配置的1.8倍 → **12层回退必须保留**
- 现役配置（`sparsedrive_int8_qdq_feat8.onnx`）确认为最终精度最优解

## DFA算子feat-INT8实现（已实现并通过kernel单元测试）
- **交付物**：`sparsedrive_int8_qdq_feat8.onnx`（在QDQ模型基础上，3个DFA节点的feat输入前各插入
  per-channel QuantizeLinear(axis=3)，scale常量`dfa_feat_scale`(256,fp32)作为第6个插件输入）
- **插件**（`plugin/dfa_plugin.cu`）已扩展：
  - `supportsFormatCombination`支持feat=kINT8组合（loc/w/output为fp32或fp16，int8-feat时输出fp32）
  - kernel内按`feat_scale[channel]`反量化，**不做staging**（保住带宽收益）
  - **loc/w支持FP16直读**（kernel内直接读half，不再staging回fp32；w 196→98MB、loc 12→3MB，均2×带宽节省）
  - 输入精度可独立组合：feat∈{fp32,fp16,int8}×loc/w∈{fp32,fp16}；输出精度跟随feat
  - 兼容两种输入数：nbInputs=6（int8-feat契约）/ nbInputs=5（旧版fp32/fp16模型）
- **验证**：
  - 量化损失（fake-quant，8样本）：per-C方案DFA输出relL2 1.5~1.7%、cosine 0.99986；端到端metric MAE 0.041
    （仅为全模型QDQ漂移0.656的1/16）；GT距离与FP32完全一致；argmax一致率100%
  - kernel单元测试（原生CUDA，`plugin/dfa_i8_test.cu`，Linux/跨平台构建见 `plugin/Makefile`）：
    6个用例全PASS——all-fp32 / int8-feat / fp16-loc&w / int8+fp16组合，
    与CPU双精度参考最大偏差 7.6e-6（纯浮点舍入量级）
- 板端使用：`python build_engine.py --onnx sparsedrive_int8_qdq_feat8.onnx --out engine.plan --fp16`
  （feat的int8 Q节点由builder与上游卷积融合；若板子希望feat以fp16传输则用
  `sparsedrive_int8_qdq.onnx`，插件两种模型通用）

## 操作级优化评估总表
| 项 | 结论 |
|---|---|
| 词表嵌入冻结+deform_format提升 | ✅ 已实施（输出逐位一致，图Constant 732→407） |
| MHA→可量化Linear堆叠 | ✅ 已实施（注意力投影进INT8，QDQ 169→189） |
| 导出期Conv+BN常量折叠 | ⛔ 保持关闭：仍有混合device的shape-cat使torch折叠崩溃（deformable_format两处CPU常量已修复，残留其他）；TRT build自带BN折叠，运行时无差 |
| TopK/Gather/打分公式 | TRT原生层+pointwise epilogue融合，无需手动处理 |
| 转置/布局 | TRT按tactic自动选布局，静态shape下shape图算子编译期折叠，无需手动 |

## 量化敏感层逐层影响（证据链）

**测量方法**：以全 INT8（无回退）配置为基线（同批 24 样本上 metric MAE ≈ 1.104，
即下表首行 gain+drift 之和），每次仅把一个模块排除出量化，在 24 个 held-out 样本上
重测端到端指标；`gain = 基线metric_MAE − 排除该模块后的metric_MAE`，>0 表示该模块敏感
（把它排除能降低漂移）。106 个可量化模块全部扫描。

**数据文件（随仓库分发）**：
- `sensitivity.json` — 原始逐模块结果（106 项：name / gain / drift / argmax / traj_l1）
- `sensitivity.csv` — 人类可读版（按 gain 降序），由 `deploy/sensitivity_csv.py` 生成
- `ptq_report_final.json` — 回退清单（Top-12）与最终指标
- 复现：`python deploy/ptq_pipeline.py --stage sensitivity` →
  `python deploy/sensitivity_csv.py`
- 证据链以 `sensitivity.json/csv` + `ptq_report_final.json` 为准。

**Top-15 敏感模块**（gain 单位为 metric MAE；完整 106 行见 sensitivity.csv）：

| rank | 模块 | gain | 排除后metric MAE | 排除后argmax一致 |
|---|---|---|---|---|
| 1 | `_status_encoding`（自车状态编码） | 0.1969 | 0.908 | 0.167 |
| 2 | `layers.0.v_attention.out_proj` | 0.0449 | 1.060 | 0.125 |
| 3 | `layers.1.v_img_attention.in_proj` | 0.0269 | 1.078 | 0.125 |
| 4 | `layers.0.v_img_attention.out_proj` | 0.0211 | 1.083 | 0.167 |
| 5 | `layers.1.p_attention.in_proj` | 0.0169 | 1.088 | 0.292 |
| 6 | `backbone.layer1.1.conv1` | 0.0130 | 1.091 | 0.292 |
| 7 | `backbone.layer1.2.conv1` | 0.0125 | 1.092 | 0.250 |
| 8 | `backbone.layer2.1.conv2` | 0.0112 | 1.093 | 0.208 |
| 9 | `layers.1.v_ffn.0` | 0.0102 | 1.094 | 0.167 |
| 10 | `layers.1.metric_heads.no_at_fault_collisions.2` | 0.0097 | 1.095 | 0.208 |
| 11 | `layers.1.path_mlp.2` | 0.0088 | 1.096 | 0.208 |
| 12 | `layers.1.t_deform_model.camera_encoder.3` | 0.0075 | 1.097 | 0.208 |
| 13 | `layers.1.t_deform_model.weights_fc` | 0.0074 | 1.097 | 0.208 |
| 14 | `backbone.layer3.0.conv2` | 0.0047 | 1.100 | 0.250 |
| 15 | `layers.1.vel_mlp.0` | 0.0031 | 1.101 | 0.208 |

**读法**：第 1 名遥遥领先（自车状态编码，输入仅 8 维但对轨迹影响最大）；
2–5 名集中在 decoder 的注意力投影；backbone 只有浅层个别卷积进入前列——
这解释了为什么「Top-12 回退」基本都落在 status_encoding + 注意力投影 + 浅层卷积上。
gain≈0 的其余 90+ 模块对 INT8 不敏感，全部保持量化。

### PDMS 端到端复核（第二标准，deploy/pdms_sensitivity.py）

metric-MAE 是 **logits 级**漂移，灵敏度高但不是部署指标；用**后处理后的 PDMS**（argmax 轨迹过
navsim PDM 仿真+打分，与数据集评测完全同口径）逐模块复核：基座=全INT8，逐个排除单个模块，
在 24 个校准集外样本上重测（`sensitivity_pdms.json/csv`，含 MAE 增益交叉参照列）。

| 模块（MAE 标准排名） | MAE gain | PDMS gain（排除该层） | PDMS 排名 |
|---|---|---|---|
| `_status_encoding`（MAE #1） | 0.197 | **−0.0019（≈0）** | 末段 |
| `layers.0.v_attention.out_proj`（#2） | 0.045 | −0.0001 | 101/106 |
| `layers.1.v_img_attention.in_proj`（#3） | 0.027 | −0.0000 | 84/106 |
| `layers.1.p_attention.in_proj`（#5） | 0.017 | +0.0030 | 4/106 |
| `backbone.layer1.1.conv1`（#6） | 0.013 | +0.0032 | 3/106 |

- 同批 24 样本上：FP32 PDMS 0.7939 vs 全INT8 0.7897（总差距仅 −0.0042）；单层排除的最大增益
  仅 **+0.0048**，在 24 场景采样噪声（SE≈0.02）以内，且 PDMS 排序与 MAE 排序基本不相关
- **结论**：logits 漂移传导不到场景级得分——逐层量化对数据集指标均无感；MAE 标准继续用作
  回退层的"定位器"（灵敏度高、计算便宜），Top-12 回退是保险而非必需。全量 138 场景四配置
  对照（FP32 0.7440 / 全INT8 0.7432 / 保护PTQ 0.7533 / QAT 0.7471）见
  `pdms_configs_report.json`

## 引擎构建前的解析验证（无需GPU）

插件源码以 `REGISTER_TENSORRT_PLUGIN` 静态注册；目标设备上编译
（CMake 或 `plugin/Makefile`），随后即可在**无 GPU 的机器上**先行验证
ONNX 与插件契约是否匹配：

```bash
cd deploy/artifacts/plugin && make TENSORRT_ROOT=<trt-headers> TRT_LIB_DIR=<trt-libs> CUDA_ARCH=<sm>
python3 trt_parse_check.py libdfa_plugin.so ../sparsedrive_int8_qdq_feat8_folded.onnx
```

脚本验证三件事：插件 creator 注册（name/namespace/version）、3 个
`sparsedrivev2::DeformableAggregation` 节点全部成功解析为插件（任何不匹配都会
导致 parse 失败）、网络 I/O 符合契约（4 输入 / 8 输出）。真正的 engine 构建
需要 CUDA 设备，用 `build_engine.py` 在目标机器完成。

## 引擎精度验证（开发机出基准 → 板端对齐）

**第 1 步 · 开发机生成 FP32 参考基准**（需 GPU + torch + navsim 环境，一次性）：

```bash
python deploy/make_engine_reference.py
# → deploy/artifacts/engine_ref/val_XX.npz，共 24 个 held-out 样本
```

每个 npz 内含：engine 的 4 个输入（`imgs/proj/iwh/status`）、FP32 参考输出
（`ref_traj/ref_scores/ref_metric_0..5`）与 GT 轨迹（`gt_traj`）。`engine_ref/`
不入库，可随时重新生成。

**第 2 步 · 板端精度对齐**（仅依赖 tensorrt + pycuda + numpy，无需 torch/数据集）：
把 `engine_ref/` 与 `engine_infer_check.py` 拷到目标机，与 engine 放一起：

```bash
python3 engine_infer_check.py --engine engine_int8.plan \
    --plugin libdfa_plugin.so --ref-dir engine_ref
```

逐样本打印 `traj_l1 / score MAE / metric MAE / argmax一致 / GT距离`，末尾汇总
24 样本均值与平均延迟。**验收标准**：argmax 一致率、traj L1、metric MAE 与
fake-quant 报告（`ptq_report_final.json`）同量级，即证明量化语义在 engine 中
完整保持；若 metric MAE 显著变大，优先排查 DFA 插件的 scale 输入与 Q/DQ 的
TRT tactic 选择。

## 已知限制
- engine构建需要CUDA设备，开发环境不满足，故未在开发机实跑；
  fake-quant精度与TRT INT8实测精度经验上一致（同QDQ语义），目标设备build后按
  上一节流程用 `engine_infer_check.py` + `engine_ref/` 做对齐验证。
- 导出图bs=1静态。多batch需改导出脚本dynamic_axes并让插件支持动态bs（已支持动态）。

## 编译前图结构清理（simplify_graph 折叠 pass）
- **动机**：导出时 do_constant_folding=False（规避 torch 混合 device 折叠崩溃），图里留下大量
  静态 shape 下的形状推导管线（Shape/Gather/Unsqueeze/Cast/Concat 等）。TRT build 自身也会折叠，
  但显式清理可缩短 build 时间、减少中间张量与图噪声。
- **为何不用 onnxsim**：onnxsim 靠 ORT 会话求值可折叠子图，而 ORT 无法加载含未注册
  sparsedrivev2::DeformableAggregation 的图。自研 deploy/simplify_graph.py：
  onnx shape inference + numpy 常量求值，只折叠 IEEE 精确/纯数据搬移类算子；
  Q/DQ 与自定义算子为硬边界（不折叠、不穿越）。
- **结果**（feat8 模型）：节点 2811 -> 1909（-32%）。Unsqueeze 304->55，Cast 227->29，
  ConstantOfShape 9->0，Where 12->3，Equal 12->3，Shape 40->25，Transpose 95->87；
  8 个 fallback 权重的 Transpose 一并折叠为常量。
- **等价性保证**（结构级断言全过）：
  - 192 Q / 189 DQ 节点签名逐一不变（量化语义未动）
  - Conv/MatMul/Gemm/LayerNorm/BatchNorm 的 (op, 输入) 签名逐一不变；保留 initializer 哈希不变
  - 3 个 DFA 节点输入输出不变；被移除的 initializer 均可追溯到折叠产物（provenance 断言）
  - 折叠不含 MatMul/Conv/LN，数值逐位等价；onnx checker 通过
- **使用**：板端编译推荐输入 sparsedrive_int8_qdq_feat8_folded.onnx；
  python deploy/simplify_graph.py <in.onnx> <out.onnx> 可对任意导出产物重跑。

## QDQ 结构标准化（qdq_onnx_rewrite pass，2026-09-18）
- **动机**：torch 侧导出的 QDQ 把每个 Q/DQ 的 scale/zp 放在图体内 Constant 节点（695 个、
  其中 220 个内容重复），权重表达为"浮点 initializer + 图内 QuantizeLinear+DequantizeLinear"，
  与标准 ONNX 量化产物（onnxruntime quantize_static / modelopt.onnx.quantization）的
  "int8 initializer → DQ + 共享 scale/zp initializer"样式不一致，Netron 可读性差。
- **变换**（deploy/qdq_onnx_rewrite.py，纯表示层）：94 个权重 Q 按 ONNX 规范离线量化
  （saturate(round(x/scale)+zp)，round half-to-even，含 per-axis 广播）为 int8 initializer；
  全部 Constant 迁移为 initializer 并按内容去重（695→475）；剪枝失去引用的浮点权重（94 个）；
  严格拓扑重排序。结果：节点 1909→1120，Constant 节点 695→0。
- **等价性验证**（无需 GPU / 无需自定义算子库）：
  - 94 条折叠链在 ORT 中以最小图重放（原 Q→DQ vs 新 int8→DQ，同一组 scale/zp/axis）
    **全部逐位相等（max|Δ|=0.0）**——以 ORT 自身的舍入/广播实现为裁判；
  - 568 个迁移后的 scale/zp initializer 与原 Constant 张量字节相等（另有 6 个本就是 initializer）；
  - onnx checker 通过（含拓扑排序校验）。
- **产物**：sparsedrive_int8_qdq_feat8_folded.onnx（+38MB，为 94 份 int8 权重副本的体积，
  源于共享权重在图内按 site 复制量化；TRT 解析无影响）；原文件保留为
  *_pre_rewrite.onnx；报告 *_rewrite_report.json。
