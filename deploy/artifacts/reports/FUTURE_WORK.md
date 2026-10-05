# FUTURE_WORK：剩余优化建议（按预期收益排序）

截至 22.92ms（见 [BOARD_OPTIMIZE.md](BOARD_OPTIMIZE.md)）结束时，逐桶剩余空间与
候选方案如下。这些是**已经归因、但尚未实施**的方向，按"预期收益 × 实现代价"排序；
作者暂不打算继续，欢迎接手。

当前延迟构成：DFA 10.67 / myelin ForeignNode 5.85 / backbone+neck conv ~3.9 /
FusedMHA 1.60 / 杂项 ~0.9（ms，合计 22.92）。

## A. 消灭 98MB logits 往返（上限 ~1-2ms，6-9%）——最大剩余单点

`weights_fc` GEMM 在 myelin 区产出 [1024,6000,8] fp16 ≈ 98MB logits，写一次、
DFA 插件的 plan 阶段再整读一次做 softmax——纯 DRAM 流量约 196MB ≈ 1.5ms，
而 GEMM 本身只有 ~25 GFLOP。

- **A1（便宜，先做）plan 稀疏读**：softmax 分母只需 valid 项；投影 mask 是小张量，
  plan 改为按 mask 前缀/计数稀疏读 logits。先用现成 cnt 统计测 valid 占比
  （`dfa_bench_v5.cu` 里已有计数插桩）——valid 若只有几十个百分点，plan 读侧
  直接砍掉大半。不动 GEMM，M2 门禁可验。
- **A2（重）GEMM+softmax+pack 全融合**：flash 式分块 GEMM + 在线 softmax，
  直接产出 entry，写侧 98MB 也消掉。需自写 fp16 GEMM 或 cublasLt+epilogue，
  工程量大，M2/M3 全过门。
- 注意 E1 教训：在 myelin 区插 QDQ 会触发重规划（+4.16ms），这条路线必须走
  插件/图手术，不能走量化。

## B. DFA 本体内部再抠（~0.5-1ms？）

v5 之后没有再拆过 plan/gather/finalize 的账。先做半天级 phase 拆分计时
（8.87ms 的 layers.0 调用里排序占多少、访存占多少），再决定打哪里。候选：
entry 宽度进一步压缩（48B → 更紧编码）、softmax 尾部权重剪枝（动数值，必须
M2/M3 全过门）、两个小调用（1.34/0.46ms）的 SPLIT 特调。

## C. 布局协商消拷贝（~0.5ms）

`Reshape_1 链 0.38ms`、`Transpose_1 链 0.21ms` 这类 profile 条目实为 layout
转换拷贝。让插件接受生产者原生布局理论上可消，但 myelin 会重规划（E1/E2 教训），
需小步实测：先只动一处，profile 确认无连带回归再推广。

## D. 不建议再投入的

- **FusedMHA**：2.0ms/7 kernel 已是 LDS 带宽地板（l0+membar 1.32ms），无算法级空间。
- **backbone/neck INT8**：净零已定案且机理对账闭环（int8 conv −0.660ms 被量化税
  +0.885ms 抵消，见 BOARD_OPTIMIZE §5.2）。重启条件：conv 占 e2e >20% 的更大
  backbone，或 Q-fusion 更好的 TRT 版本（10.x 把 act-Q 融进 ReLU epilogue 的比例
  高得多）；届时也应重新校准 + 重过 M2/M3。
- **CUDA Graph 差距**：eager 与 graph 已只差 0.02ms，无油水。
- 训练侧长期项（完整 navtrain QAT、loc INT8、2:4 稀疏、ORT 自定义算子）见根
  README Roadmap。

## 尺度参考：这条管线还剩多少可优化空间

22.92ms 里的结构地板：DFA 是数据依赖 gather（采样集合由输入决定）、conv ~3.9ms、
MHA 1.6ms——A+B+C 全部吃满的理论极限约 **20ms（再 −13%）**。再往下必须动模型
结构（更小 backbone / 蒸馏 / 剪枝），那是训练侧行为，不是部署侧能解决的。
