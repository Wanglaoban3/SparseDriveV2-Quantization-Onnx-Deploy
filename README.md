<div align="center">
<h2>SparseDriveV2 Quantization &amp; ONNX Deployment</h2>
<p><b>W8A8 INT8 量化 · 自定义算子 ONNX 导出 · TensorRT DFA 插件</b><br>
让 SparseDriveV2 (ECCV 2026) 走上板端的完整工具链</p>
<p>基于 <a href="https://github.com/swc-17/SparseDriveV2">swc-17/SparseDriveV2</a> 构建，<br>
补齐「训练/评测仓库 → 车载部署」的最后一公里</p>
</div>

---

## 这是什么

上游仓库解决了 SparseDriveV2 的**训练与评测**；本仓库在其之上增加**部署侧**的全部工作：

- **导出**：把无法被 ONNX 表达的 DeformableAggregation CUDA 算子注册为自定义节点
  （`sparsedrivev2::DeformableAggregation`），**不做 GridSample 近似替换**，保留原生部署语义；
  同时内置一组结构优化（词表常量化、QKV 打包、静态 shape、导出后图折叠）
- **量化**：NVIDIA ModelOpt PTQ(W8A8) + 103 模块敏感度扫描 + Top-12 回退 + 无标签自蒸馏 QAT；
  DFA 采样特征（feat）走 INT8，插件 kernel 内反量化
- **插件**：纯 CUDA C++ 的 TensorRT 插件，feat∈{fp32, fp16, int8} × loc/w∈{fp32, fp16} 任意组合，
  附 6 用例单元测试
- **跨平台**：全部为纯 Python + CMake，无平台专属产物；量化精度在开发期 GPU 上完成验证，
  engine 编译与推理精度测试在 Linux + CUDA 设备上执行

## 相对原仓库的改动

### 1. 模型代码（`navsim/agents/sparsedrive/`，均不影响训练语义）

| 文件 | 改动 | 目的 |
|---|---|---|
| `ops/deformable_aggregation.py` | `torch._shape_as_tensor` → 显式 shape 常量；零张量构造改 `new_zeros` | ONNX 导出兼容（原写法导出即崩） |
| `custom_decoder.py` | 输出增加 `traj_scores` 与 6 个 PDM metric logits；metric loss 以存在 `token_path` 为条件 | 量化漂移评估信号 + 蒸馏监督头 |
| 其余模型/骨干/词表代码 | 无改动 | GridMask 自带 training 守卫，eval 下即恒等 |

### 2. 导出与图优化（`deploy/export_onnx.py`、`deploy/simplify_graph.py`）

- `DeformableAggregationFunction.symbolic` → `g.op` 自定义节点；TRT 侧按
  `plugin(name=op_type, namespace=domain)` 自动解析为 CUDA 插件
- 部署优化内置：**词表嵌入预计算为常量**（`FrozenVocabHead`，省 ~0.5 GFLOPs/帧）、
  `nn.MultiheadAttention` → QKV 打包的等价 Linear 实现（投影可量化 + 利于 TRT fused-MHA）、
  `deform_value` 提升至 decoder 级只计算一次（输出逐位一致）、静态 bs=1 图
- 导出后常量/形状管线折叠：节点 **2811 → 1909（−32%）**，带结构级等价断言
  （192 Q / 189 DQ 签名逐一不变、Conv/MatMul/Gemm 结构不变、保留权重哈希不变；
  onnxsim 因自定义算子无法被 ORT 加载而不可用，故自研折叠 pass）

### 3. INT8 量化（`deploy/ptq_pipeline.py`、`deploy/kd_qat_v2.py`）

- ModelOpt `mtq`（INT8_DEFAULT_CFG，max 校准 32 样本）→ 24 个 held-out 样本漂移验证 →
  **103 个模块逐个敏感度扫描** → Top-12 敏感层回退 FP16
- fp32 teacher 自蒸馏 QAT：对 `traj_scores` + 6 个 metric logits 做归一化 KD（无标签、lr 2e-6、
  keep-best 保证不劣于 PTQ 起点）
- DFA feat INT8：per-channel(256) scale 校准，Q 节点插在插件输入前，scale 作为插件第 6 输入

### 4. TensorRT DFA 插件（`deploy/artifacts/plugin/`）

- `dfa_plugin.cu`：feat∈{fp32,fp16,int8} × loc/w∈{fp32,fp16} 任意组合；int8 feat 在 kernel 内
  反量化（零 staging）；fp16 w/loc 直读（采样带宽 −50%）
- 6 用例 kernel 单元测试 vs CPU 双精度参考：**最大偏差 7.6e-6**（纯浮点舍入量级）
- CMake 构建，与具体 TRT 版本解耦（TRT 8.6 头文件编译验证通过）

## 指标

> 环境：NAVSIM v1（OpenScene mini 分片 0，2 个日志 / 138 场景）。FP32 基线为原生 CUDA 推理；
> 漂移对比在 24 个从未参与校准/训练的 held-out 样本上进行，另有 138 场景端到端 PDMS 对比。

**端到端量化精度**（核心表）：

| 配置 | 轨迹 L1 | score MAE | metric MAE | 相对 GT 距离 |
|---|---|---|---|---|
| FP32 基线 | — | — | — | 1.0923 m |
| 全 INT8（无回退） | 9.2 cm | 0.683 | 1.012 | 1.0834 m |
| INT8 + Top-12 回退 | 9.9 cm | 0.461 | 0.703 | 1.1194 m |
| **+ 蒸馏 QAT（最终交付）** | 9.5 cm | 0.450 | **0.656** | **1.0925 m（≈FP32）** |

**其他关键数字**：

- 端到端 PDMS 对比（138 场景同口径，`deploy/pdms_eval_quant.py`）：FP32 **0.744** vs
  最终 INT8 交付（fake-quant）**0.747** —— 111/138 场景逐分一致（14 升 / 13 降），
  量化在数据集指标上与原版持平；历史 agent 链路基线 0.739，链路间 ±0.005 为临界场景翻转噪声
  （上游全量 navtest 为 92.22，mini 子集分布不同，只用于横向对比）
  （证据：`deploy/artifacts/pdms_report.json` + 逐场景 `pdms_fp32.csv` / `pdms_int8_qat.csv`）
- DFA feat-INT8（per-C）增量：端到端 metric MAE **+0.041**（仅为全模型 QDQ 漂移的 1/16），
  DFA 输出 cosine 0.99986，轨迹 argmax 一致率 100%
- **负结果同样记录在案**（详见 `deploy/artifacts/BOARD_DEPLOY.md`）：
  - w（softmax 采样权重）INT8 相对误差 19–25%，不可行——小权重 × 固定 int8 步长的相对误差放大
  - 取消 12 层回退的全 INT8 配置漂移为现役配置的 **1.8×**
  - 仅 138 个 mini 样本无法支撑 QAT 微调（灾难性遗忘），蒸馏法也只能挽回 ~3%——
    完整数据集上的 QAT 是后续工作

## 快速开始

环境与数据准备完整说明见 [`deploy/README.md`](deploy/README.md)。
单 conda 环境即可（Python 3.9 + torch 2.0.1 + ModelOpt 0.11 + onnx），权重从
[上游 HuggingFace](https://huggingface.co/wenchaosun/SparseDriveV2) 下载。

```bash
# FP32 导出（注册自定义 DFA 算子 + 内置部署优化）
python deploy/export_onnx.py

# PTQ 校准 + 敏感度扫描 + Top-12 回退 + QDQ 导出
python deploy/ptq_pipeline.py

# 蒸馏 QAT + 最终 QDQ
python deploy/kd_qat_v2.py

# DFA feat-INT8 增强 + 图折叠（推荐的板端编译输入）
python deploy/add_dfa_feat_int8.py
python deploy/simplify_graph.py \
    deploy/artifacts/sparsedrive_int8_qdq_feat8.onnx \
    deploy/artifacts/sparsedrive_int8_qdq_feat8_folded.onnx

# 板端：CMake 编译插件 → 构建 engine
cmake -S deploy/artifacts/plugin -B build-plugin && cmake --build build-plugin
python deploy/artifacts/build_engine.py \
    --onnx deploy/artifacts/sparsedrive_int8_qdq_feat8_folded.onnx --fp16
```

大文件（`*.onnx`、`*.pt`、`calib/`）不入库，由上述脚本重新生成；`deploy/artifacts/` 内保留
全部实验报告 JSON、**逐层量化敏感度证据链（sensitivity.json / sensitivity.csv，106 模块）**、
插件源码与单元测试、参考输出 `reference_traj_fp32.npz` 与
`sample_inputs.npz`（可离线复跑 `verify_opt.py` 对齐验证）。

## 板端部署

详见 [`deploy/artifacts/BOARD_DEPLOY.md`](deploy/artifacts/BOARD_DEPLOY.md)：
插件契约（输入布局 / dtype 组合 / 第 6 输入 scale）、四轮量化实验完整记录、engine 构建要点、
精度对齐闭环（开发机 make_engine_reference.py 出基准 → 板端 engine_infer_check.py 验证）。
已知限制：engine 编译与实测需在 Linux + CUDA 目标设备完成，全流程脚本仓库内齐备。

## Roadmap

- [ ] **Linux 板端 engine 实测**：build + 24 样本精度对齐 + 延迟，在任意 Linux + CUDA 设备
  （含目标板卡）上执行，脚本与对齐基准仓库内齐备
- [ ] **完整 navtrain 数据上的 QAT**（蒸馏管线已就绪）：解锁 loc INT8（已证明损失可忽略）、
  冲击 w INT8 与全 INT8 覆盖
- [ ] **2:4 结构化稀疏**（Ampere 及以上，ModelOpt sparsity → TRT sparse tactic）
- [ ] **ONNX Runtime 自定义算子**（CPU/CUDA EP 回退，非 TRT 平台可运行）
- [ ] DFA per-point scale（实测 relL2 0.47%，作为高精度可选档）

## License

本项目沿用上游的 [Apache-2.0](LICENSE) 协议。

## 致谢

- [swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2)（ECCV 2026）——模型与上游工程；
  原始 README 见 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)
- [NAVSIM](https://github.com/autonomousvision/navsim) / [nuplan-devkit](https://github.com/motional/nuplan-devkit) ——评测框架
- [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 、[TensorRT](https://developer.nvidia.com/tensorrt)

如果你觉得本项目有帮助，也请给上游 Star 并引用：

```bibtex
@article{sun2026sparsedrivev2,
  title={SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving},
  author={Sun, Wenchao and Lin, Xuewu and Chen, Keyu and Pei, Zixiang and Li, Xiang and Shi, Yining and Zheng, Sifa},
  journal={arXiv preprint arXiv:2603.29163},
  year={2026}
}
```
