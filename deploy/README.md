# SparseDriveV2 部署工具集（导出 / PTQ量化 / 蒸馏QAT / TRT插件）

本目录收纳 SparseDriveV2 的全部部署相关工作：ONNX导出（含自定义DFA算子）、ModelOpt量化、
敏感层分析与蒸馏QAT，以及板端TensorRT部署交付物（`artifacts/`）。

## 环境
单个conda环境即可（`navsim`，Python 3.9 + torch 2.0.1+cu118）：
- ModelOpt 0.11（`pip install --no-deps nvidia-modelopt==0.11.0` + `pydantic`，兼容torch 2.0.1）
- onnx / onnxruntime（清华源）
- navsim / nuplan 以 editable 方式安装（`pip install --no-deps -e`）
- 需要的环境变量：`NAVSIM_DEVKIT_ROOT` / `OPENSCENE_DATA_ROOT` / `NAVSIM_EXP_ROOT` / `NUPLAN_MAPS_ROOT`
  （本地可用脚本/`.bashrc` 自行设置）

所有脚本路径基于自身位置自动解析（仓库根 = `deploy/` 的上一级），无需改动；
输出统一写入 `deploy/artifacts/`。可用环境变量 `SD_ROOT` / `TRT_DEPLOY_DIR` 覆盖。

## 流程（按序执行）

### 0. 数据与特征缓存（一次性）
```
# 特征缓存：138 场景（OpenScene mini 分片0的两个日志），
#   按 navsim 标准 run caching 流程（见上游 docs/train_eval.md），agent 选 sparsedrive
# v1 指标缓存：navsim run metric_caching（需要 NUPLAN_MAPS_ROOT 指向 nuplan maps）
```
数据布局：`$OPENSCENE_DATA_ROOT/navsim_logs/mini/*.pkl`、`sensor_blobs/mini/<log>/<CAM>/*.jpg`。
（本仓库的 train_test_split `mini_shard0` 已限定到分片0覆盖的两个日志。）

### 1. FP32 ONNX 导出（含自定义算子）
```
python deploy/export_onnx.py
```
- `DeformableAggregationFunction` 挂 `symbolic`，导出为 `sparsedrivev2::DeformableAggregation`
  自定义节点（TRT侧按 plugin(name=op_type, namespace=domain) 自动解析为插件）
- 部署优化已内置：词表嵌入冻结为常量初始化器（`FrozenVocabHead`）、
  `nn.MultiheadAttention` → 等价 `QuantizableMHA`（使注意力投影可量化）、
  `deform_value` 提升到decoder级只算一次（与原实现输出逐位一致）
- 顺带导出 `artifacts/calib/calib_*.npz` 校准样本

### 2. PTQ + 敏感层分析 + 回退
```
python deploy/ptq_pipeline.py
```
ModelOpt `mtq`（INT8_DEFAULT_CFG，max校准32样本）→ 24个held-out样本上fake-quant验证 →
逐模块敏感度扫描（禁用单模块量化对漂移指标的改善排序）→ Top-12敏感层回退FP16。
逐模块结果落盘 `artifacts/sensitivity.json`（106模块的gain/drift/argmax/traj_l1）；
`python deploy/sensitivity_csv.py` 生成人类可读的 `artifacts/sensitivity.csv`（按gain降序）。

### 3. 蒸馏QAT + 最终QDQ导出
```
python deploy/kd_qat_v2.py
```
fp32 teacher 自蒸馏 `traj_scores` + 6个metric logits（无标签，归一化损失，lr 2e-6，16步），
`keep-best` 策略保证不劣于PTQ；最终导出 `artifacts/sparsedrive_int8_qdq.onnx`
（189对Q/DQ节点 + 3个DFA自定义节点，结构校验：全部图输入均被使用）。

### 4. DFA feat-INT8（可选增强）
```
python deploy/add_dfa_feat_int8.py     # 校准per-C scale + 插入Q节点 + 生成 feat8 ONNX
python deploy/dfa_int8_eval.py         # 量化损失评估（各方案对比）
```
插件已支持feat=INT8输入（kernel内按scale[C]反量化）；kernel单元测试见 artifacts/plugin/dfa_i8_test.cu。

### 5. 编译前图折叠（清理形状管线）
```bash
python deploy/simplify_graph.py artifacts/sparsedrive_int8_qdq_feat8.onnx artifacts/sparsedrive_int8_qdq_feat8_folded.onnx
```
静态 shape 下把导出残留的形状推导管线折叠为常量（节点数 -32%），Q/DQ 与 DFA 节点不动；
结构级断言保证量化语义不变。板端编译推荐使用 folded 版本。

### 6. 验证/基准工具
```
python deploy/verify_opt.py       # 当前图输出 vs 已存FP32参考（应逐位一致）+ 延迟
python deploy/bench_baseline.py   # fp32 延迟基准
```

## 交付物（deploy/artifacts/）
- sparsedrive_fp32.onnx / sparsedrive_int8_qdq.onnx / sparsedrive_int8_qdq_feat8.onnx
- sparsedrive_int8_qdq_feat8_folded.onnx（图折叠版，板端编译推荐输入）
- plugin/（dfa_plugin.cu 含feat-int8分支 + kernel单元测试 + CMake/Makefile）
- calib/、sensitivity.json + sensitivity.csv（逐层量化敏感度证据链）、ptq/kd报告、BOARD_DEPLOY.md（板端部署说明）

## 板端TensorRT部署
见 `artifacts/BOARD_DEPLOY.md`。要点：
- `artifacts/plugin/dfa_plugin.cu` 实现DFA算子（已对TRT 8.6头文件编译验证），
  CMake构建后 `ctypes.CDLL` 加载即可被ONNX parser自动关联
- `artifacts/build_engine.py` 构建FP16/INT8引擎；INT8显式量化直接用QDQ ONNX，
  隐式量化备选走熵校准器
- 精度对齐闭环：开发机 `deploy/make_engine_reference.py` 生成24样本FP32基准
  （`artifacts/engine_ref/`，不入库）→ 板端 `artifacts/engine_infer_check.py`
  对engine逐样本验证（仅需tensorrt+pycuda+numpy），详见BOARD_DEPLOY
- 引擎输入：imgs(1,3,3,256,512) / projection_mat(1,3,4,4) / image_wh(1,3,2) / status_feature(1,8)；
  输出：trajectory(1,8,3)（图内已完成argmax选择）+ traj_scores + 6×metric logits

## 量化精度（fake-quant 实测，24个held-out样本）
| 配置 | 轨迹偏差 | score MAE | metric MAE | 相对GT距离 |
|---|---|---|---|---|
| FP32 | — | — | — | 1.0923m |
| 全INT8 | 9.2cm | 0.683 | 1.012 | 1.0834m |
| INT8+12层回退 | 9.9cm | 0.461 | 0.703 | 1.1194m |
| +蒸馏QAT（最终） | 9.5cm | 0.450 | **0.656** | **1.0925m** |

早期实验脚本（v1蒸馏QAT、真标签QAT）未随仓库分发，其失败结论作为负结果记录在 `artifacts/BOARD_DEPLOY.md`；主流程用 `kd_qat_v2.py`。

## 端到端 PDMS（数据集指标 vs 原版）
```bash
python deploy/pdms_eval_quant.py      # FP32 与 最终INT8(fake-quant) 各跑一遍138场景
python deploy/pdms_eval_configs.py    # 补测 全INT8 PTQ / 保护PTQ(Top-12回退) 的138场景
python deploy/pdms_sensitivity.py     # 敏感层PDMS复核：106模块 × 24校准集外样本（约1h）
```
同一批 138 场景、与 navsim `run_pdm_score_navtest_v1_fast` 完全同口径（PDMSimulator+PDMScorer，
40×0.1s proposal，agent 轨迹 8×0.5s）：

| 配置 | PDMS | Δ vs FP32 | 备注 |
|---|---|---|---|
| FP32（原版） | 0.7440 | — | 历史 agent 链路基线 0.739（±0.005 为链路/场景翻转噪声） |
| 全 INT8 PTQ | 0.7432 | −0.0008 | |
| 保护 PTQ（Top-12 回退） | 0.7533 | +0.0092 | 临界场景翻转，量级仍在噪声内 |
| **INT8+回退+蒸馏QAT（最终交付）** | **0.7471** | +0.0031 | 与 FP32 逐分一致 111/138 场景 |

**敏感层双标准**：`ptq_pipeline.py --stage sensitivity` 的 metric-MAE 漂移用于**排序选回退层**
（logits 级灵敏度高）；`pdms_sensitivity.py` 用**后处理后的 PDMS** 复核（逐个排除模块、24 个
校准集外样本、与数据集评测同口径打分）。复核结论：单层 PDMS 增益全部 ≤0.005 且与 MAE 排序
基本不相关（MAE 头号层 `_status_encoding` 的 PDMS 增益 ≈0）——逐层量化对数据集指标均无感，
Top-12 回退是保险而非必需。

证据落盘：`artifacts/pdms_report.json`、`pdms_configs_report.json`、
`sensitivity_pdms.json/csv` 及逐场景 `pdms_*.csv`。
