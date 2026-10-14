# PithTrain 深入学习文档

> 学习日期：2026-10-14。Fork：`jasonchen505/pith-train`，上游：`mlc-ai/pith-train`（Apache-2.0，v0.1.3）。
> 本文档在 fork 的 `notes/learning` 分支，`main` 未动。本文只做学习记录，不认领 issue，不提 PR。

## 1. 一句话定位

PithTrain 是 MLC（CMU，Tianqi Chen 团队）出的 MoE 训练系统，卖点是 **agent-native**：约 11K 行 Python（README 自述），没有插件注册表、没有重型编译扩展，代码量小到一个 agent 或一个人能端到端读完，同时保持生产级性能（4D 并行、计算通信重叠、FP8 训练）。

论文：arXiv:2605.31463《PithTrain: A Compact and Agent-Native MoE Training System》，博客 2026-06-01。Pipeline 调度派生自 DeepSeek 的 DualPipe（`pithtrain/pipeline/` 下保留了 MIT LICENSE 与 NOTICE，改动时署名要留着）。

## 2. 仓库状态（2026-10-14 核实）

- 上游：355⭐，39 forks，创建于 2026-03-21，默认分支 `main`，HEAD 为 `87208d9`（2026-10-09，#108 给 Qwen3.5 加 packed sequences）。
- `open_issues_count = 15`，拆开是 **4 个 issue + 11 个 PR**（GitHub 把 PR 也计入该数；“15 个 issue”是误读）。
- Fork 的 `main` 头是 `5187f030`（2026-10-04，把上游 main merge 进 fork），**落后上游 1 个 commit（#108）**。
- Fork 的 `main` 历史里有一个个人笔记 commit `99088365`（2026-07-15，`add notes`），根目录多出上游没有的 5 个个人 md（面试准备、Megatron 对比、8×4090 复现计划、增量学习笔记等）+ `paper/`（arXiv-2605.31463v1）。**上游贡献分支必须从上游 `main` 起，绝不能带这些个人文件进 PR diff。**
- 核心作者集中在 CMU：Hao Kang（`haok1402`，绝大多数 PR）、GaokaiZhang、akaashrp、kldxbjz、Ruihang Lai、Haozhan Tang。外部 PR 只有 #78 一例，且自 7 月 20 日起未合入（见 §10）。

## 3. 设计哲学：四个原则（CONTRIBUTING 明文）

这是评审 PR 的标尺，贡献前必须先对齐：

1. **Compact over comprehensive**：不追求广覆盖，改动取最小解。只对部分用户有用的功能，先问该不该进这个仓库。
2. **Python-native**：一个语言能导航完。Kernel 走 Triton 这类 Python DSL；DeepGEMM、FlashAttention 这类外部库只在确有必要时用，且包一层薄薄的 Python operator。树内不做 C++/CUDA 构建步骤。
3. **Minimal implicit indirection**：调用点读代码就能知道跑的是什么。不搞插件注册表、字符串 key 分发表、运行时 spec 解析。每个模型是一个自包含文件、自己搭 layer，接受少量重复换取局部可读性。
4. **Ship skills for recurring procedures**：重复性流程要随仓库发一个 agent skill（`.agents/skills/`），带明确前置条件和可验证的 PASS/FAIL。

大改动（pipeline 调度、并行 mesh、模型契约、新增依赖）**先开 issue 讨论**，再写代码；小而自包含的 fix 可直接 PR。

## 4. 三层架构与目录地图

| 层 | 目录 | 职责 |
|---|---|---|
| Application | `pithtrain/tasks/` | 端到端流程：pretrain_lm、tokenize_corpus、convert_checkpoint |
| Engine | `pithtrain/{pipeline,models,modules,contexts}` | 主体：pipeline 调度、模型、分布式/训练基建 |
| Operator | `pithtrain/operators/` | Triton / 库支持的 fused kernel |

```
pithtrain/
├── tasks/        # launch(cfg) 入口
├── pipeline/     # DualPipeV 调度 + F/B 重叠（dualpipev.py 971 行，execution.py）
├── models/       # 一模型一文件 + interface.py 契约（qwen3 345 行，qwen35 502 行）
├── modules/      # distributed.py 164 行，training.py 569 行，checkpoint.py 565 行，
│                 # optimizer.py 252 行，dataset.py，load_balance.py，logging.py
├── contexts/     # 进程全局运行时状态（distributed / training / logging 三个模块）
└── operators/    # ep_dispatch.py 664 行，token_scatter.py 390 行，grouped_linear.py 285 行，
                  # ring_attention.py，deepgemm_quantize.py，flash_attn_v4.py 等
tests/            # 单卡单元测试 + tests/test_dualpipev.py（22K，含 .sh 多卡集成）
examples/         # pretrain_lm（5 个模型配置）/ tokenize_corpus / convert_checkpoint
benchmarks/       # operators（ring attention、fp8）+ pretraining（5 个模型）
tools/memory_estimator/  # 给定并行 mesh 的峰值内存模拟器
.agents/skills/   # 10 个 agent skill
docs/             # architecture.md（163 行）、user-guide.md（152 行）
```

## 5. 核心抽象：5-stage decoder layer

整个 engine 围绕一个切分：一个 transformer decoder layer 按专家并行通信边界切成 5 个 stage，让 pipeline 把一个 micro-batch 的计算与另一个的通信重叠起来。

| Stage | 内容 | 运行位置 |
|---|---|---|
| 1 Attention | LayerNorm → Attention → LayerNorm → 专家路由（top-k） | 计算流 |
| 2 Dispatch | all-to-all：token 发往专家所在 rank | 通信流 |
| 3 MLP | 在收到的 token 上跑专家 / MLP | 计算流 |
| 4 Combine | all-to-all：专家输出 gather 回源 rank | 通信流 |
| 5 Aggregate | 专家输出按 router 权重加权 + 残差 | 计算流 |

契约在 `pithtrain/models/interface.py`（117 行）：

- `LayerProtocol`：`forward_stage1`（返回 hidden + residual + `RoutingInfo`，含 topk 权重、专家索引、dispatch/combine splits）、`forward_stage3`（专家计算）、`forward_stage5`（聚合）、`reference_forward`（朴素 forward，供正确性校验）。
- `ModelProtocol`：`forward_prolog`（首 stage：embed）、`forward_epilog`（末 stage：final norm + lm_head）、`forward_posemb`（RoPE）、`reference_forward`。
- Stage 2/4 是框架所有的，layer 不实现，只交路由元数据，调度器驱动 all-to-all。

## 6. DualPipeV：pipeline 引擎

`pithtrain/pipeline/dualpipev.py`（971 行）+ `execution.py`（stage 实现、`ExecutionCtx`、dispatch/combine helper、`WeightGradStore`）：

- **V 形排布**：模型切成 `2 × pp_size` 个 chunk，rank `r` 持有 chunk `r` 和 `2×pp_size−1−r`。每 rank 同时在 forward 和 backward 扫掠，减小 pipeline bubble。层数除不尽时，边上 chunk（带 embedding / norm+lm_head）少分几层以平衡内存。
- `DualPipeV.step()` 编排跨 micro-batch 的重叠 F/B；`overlapped_forward_backward()` 是单个 micro-batch 对的交错循环；支持 `forward_only=True`（推理用）。
- `WeightGradStore`：延迟 wgrad，服务 zero-bubble 调度。
- `layer_partition()`：把 decoder 层分到各 pipeline stage。
- 显存相关的新动向在 open PR #106：forward-only step 里逐层释放 activation。

## 7. 并行：MoE parallel folding（最值得读透的部分）

`pithtrain/modules/distributed.py`（164 行）是 rank 的唯一事实来源。`setup_device_mesh` 在**同一批 rank 上建两个 device mesh**：

- `attn_mesh`：`(pp, dp, cp)`，attention 视角；
- `expt_mesh`：`(pp, dp, ep)`，专家视角。

一个 pipeline stage 拥有 `world_size // pp` 个 rank，两种视角只是两种分解：`dp × cp == ep × expt_dp == world_size // pp`。PP 是两者共享的唯一轴，且在两个 mesh 里都最外层，这是正确性论证的核心：attention 一半和专家一半的 layer 落在同一 device 上。

关键性质：

- `cp`、`ep` 各自只需整除 stage 大小，**互不整除也行，EP 可以横跨 CP**。
- CP 组和 EP 组都是**连续 rank 块**，各自独立待在 NVLink 上，互不花对方的代价。
- **`dp_rank` 单独决定一个 rank 读什么数据**，`ep_rank` 只命名它托管的专家和 all-to-all 对端，不决定数据。MoE 层把 `(B, S, hidden)` 压平成 token 独立路由，所以 DP 切 batch 还是 CP 切 sequence，对它没有区别，过渡零通信。
- FSDP 分两类参数：**expt**（routed expert 权重）沿 expert 视角的 `dp` 轴分片；**attn**（其余一切）沿展平的 `dp × cp` 分片。每层两次 `fully_shard`，FSDP2 的 module walk 在已分片子模块处停下，两类参数落在不相交的规约组里。`hsdp_replica > 1` 时两组各拆成 N 个 replica，组内分片、跨 replica 全规约。
- **梯度规约是 plain-sum，不是 average**：两类参数规约组大小不同，但各自的 sum 已经是真实全局梯度（专家组每个成员都经 all-to-all 收到了整个 EP 组的 token）。`apply_fsdp` 给所有参数 `set_gradient_divide_factor(1.0)`，训练 step 里只做一次除以全局 token 数的归一化。若用 FSDP2 默认 average，专家梯度会大 `ep_size` 倍。**这是 #96 式回归最容易藏身的地方**（见 §11）。

`setup_default_process_group` 还装了一个 fail-fast excepthook，并把 `DistributedCfg.timeout`（默认 15 分钟）作为 collective 超时，死一个 rank 不让对端空等。

## 8. CP：两种 sequence layout（Qwen3.5 hybrid 的关键）

`pithtrain/operators/cp_sequence.py`，两种布局：

- **zigzag**：softmax attention 用，rank `r` 持 `2×cp_size` 个块中的第 `r` 和 `2×cp_size−r−1` 块，平衡 causal 注意力的计算量。
- **contiguous**：线性注意力 recurrence 用，块 `2r` 和 `2r+1`，rank 顺序即 sequence 顺序。

Hybrid 模型（Qwen3.5）在两类层边界做一次 all-to-all 转换，提升到每个线性注意力层 run 的两端。不变量：**hidden 在模型边界和每个 attention 层处是 zigzag，在线性注意力 run 内部是 contiguous**；norm、MoE、残差都是 per-token，对布局无感。

`zigzag_spans(cp_rank, cp_size, seq_len)` 是 zigzag 划分的唯一定义，返回该 rank 持有的两个全局 token 区间，loader 和 `forward_posemb` 都调它。**别处重述这份算术会静默训错而不报错**，这是 AGENTS.md 点名的坑。

## 9. FP8 训练

`TrainingCfg.fp8` 一个开关翻转整个模型：setup 时把 `training.Linear` / `training.GroupedLinear` 绑定到 FP8 或 BF16 类。DeepGEMM 做 128 元素 block-scaled FP8 GEMM（E8M0 幂 2 scale；Blackwell 上是原生 MXFP8 PTX，Hopper 上模拟）。量化是 `operators/deepgemm_quantize.py` 里的自定义 Triton kernel，`fp8_weight_cache.py` 持每步版本号，让 linear 层在 micro-batch 间复用已量化的权重。

## 10. Checkpoint：canonical vs localized

`pithtrain/modules/checkpoint.py`（565 行），基于 PyTorch DCP：

- **Canonical（磁盘）**：与并行无关，全局层名，专家逐个存；
- **Localized（内存）**：当前 PP/EP 布局下的排布，专家按 EP rank 堆叠、带 DualPipeV 前缀。
- 保存 localized→canonical，加载 canonical→localized。磁盘格式与并行无关，所以 checkpoint 可 **reshard**：换 PP/EP/DP 布局续训。
- HF 导入路径产出只有模型的 checkpoint（无 optimizer/scheduler），非严格加载。
- 正在进行的 open PR 都围着这块转：#109（safetensors 流式导入 FSDP 分片训练）、#110（HF 导入时 bound 内存占用）、#111（uneven FSDP shard + HSDP 的专家 checkpoint 修复）。三者同属一个作者圈（#110/#111 GaokaiZhang/akaashrp），互相可能有依赖，动手前要先看它们的状态，别重复造。

## 11. 模型：四个家族，一模型一文件

`pithtrain/models/`：DeepSeek-V2-Lite（`deepseek_v2.py`，MLA 在模型文件内实现）、Qwen3 MoE（`qwen3_moe.py`，345 行，文档点名的“完整可读实现”）、Qwen3.5 MoE（`qwen35_moe.py`，502 行，含 `Qwen35MoeGatedDeltaNet` 线性注意力 + hybrid 布局）、GPT-OSS 20B/120B（`gpt_oss.py`）。

新模型按 `add-new-model` skill 走：模型文件实现两个 Protocol + `reference_forward`，用 `training.Linear` / `training.GroupedLinear` 建层以保持 FP8/BF16 无关；复制现有模型当模板，**不要抽象共享基类**（no-indirection 原则）；接线（`setup_model` 注册、`apply_fsdp`、examples 下 `script.py` + `config.json`）；可选 checkpoint converter；测试从 pp=1/ep=1 带到 pp=2/ep=2。

历史教训：#96（2026-09-20）——#95 把 `clip_grad_norm_` 移到 `modules/optimizer.py` 时，`@torch.no_grad()` 装饰器被留在了后面的 `train_step` 上，导致之后每次 pretraining step 的 forward/backward 都在 no_grad 下跑，参数永远收不到梯度，日志里 gradient-norm 恒 0.0、loss 随数据漂移。修复是把装饰器放回 `clip_grad_norm`。这类“装饰器/状态放错函数”的回归，是这个仓库真实发生过的、代价最高的 bug 形态。

## 12. 训练编排

`pithtrain/tasks/pretrain_lm.py`（343 行）：`PretrainLMCfg` 组合 `DistributedCfg` + `TrainingCfg` + `LoggingCfg` + 自己的 `dataset` 路径。`launch` 依次调 `setup_logging`、`setup_distributed`、`setup_dataset`、`setup_training`，`find_checkpoint` 找到就续训。

`train_step` 的要点：

- `get_global_batch`：dp_rank 决定读哪部分数据；每个 pipeline rank 读同样的样本（offset 只跟 step、数据、context rank 有关，与 pipeline rank 无关），DualPipeV 不用 broadcast 就知道 shape；只有首 rank 消费 tensor（V 形下它同时持有 embedding 和 loss）。CP 下按 `zigzag_spans` 读 front/back 两块拼进预分配 host buffer，一次 HtoD。
- `objective`：交叉熵，返回 micro-batch token 求和后的 loss；step 除以全局非忽略 token 数得到 token 加权均值。CP>1 时 loss 还要在 cp_group 上 all_reduce 再除 `cp_size`。
- **唯一一次梯度归一化**：FSDP plain-sum 之后，`p.grad.mul_(1 / (num_tokens × dp × cp))`。注释明言：该式成立仅当每 rank 计的 token 数相同，pretraining 成立，带 `-100` mask 的 packed 数据不成立。
- 优化器是 Muon + AdamW 组合（`modules/optimizer.py`，Muon 经 Newton-Schulz 正交化，`zeropower_via_newtonschulz5`），调度器 WSD（`make_wsd_scheduler`，start_lr / warmup_ratio / final_lr）。
- Load-balance loss：`lb_coef × E × dot(f, p)`，日志里把系数除掉，1.0 为完美平衡（Megatron-LM 约定）。
- 每步结尾延迟 GC（`gc.collect()`），避免 cyclic 回收在 forward/backward 中间触发。

## 13. Packed sequences / SFT 载荷

Pipeline 端到端 **BSHD**：hidden 始终 `(B, S, hidden)`，embedding、norm、attention、MoE、残差、P2P、loss 全是 BSHD。Packed（变长 SFT）训练强制 `micro_batch_size == 1`：一个 micro-batch 是 `(1, S, hidden)`，`S` 为总 token 数，多文档沿 `S` 轴 concat，`cu_seqlens`（int32 累计偏移 `[0, …, S]`）标边界。残差流语义上是 `TD`，但刻意保持 3-D，使所有假设 BSHD 的模块零改动，且 `B=1` 仍是切 micro-batch 的缝。

- **THD 只限 attention kernel 内**：attention 把 `(1, S, H, D)` squeeze 成 `(S, H, D)` 交给 FA4 的 varlen 入口（`flash_attn_v4.py:flash_attn_varlen_func`），再 unsqueeze 回来。按 `cu_seqlens` 是否为 None 分发 dense causal vs block-diagonal varlen。`max_seqlen` 传 `S`（compile 安全的上界；FA4 按 `cu_seqlens` 调度工作，过分配免费且避免逐调用 device sync）。
- **位置**：`forward_posemb(S, cu_seqlens)` 在 packed 时按文档重置位置；RoPE 相对 + block-diagonal mask，所以 `cp_size == 1` 时重置与连续全局位置 bitwise 相同，CP 下才 load-bearing。
- **Loss**：边界 / prompt token 用 `-100` mask（无单独 loss-mask tensor），token 加权。
- `cu_seqlens` 挂在每个 `Microbatch` 上，随 pipeline 以显式 forward 参数传递（`rotary_posemb` 的兄弟，不是模块属性），只被 attention 和 `forward_posemb` 消费。
- **#108（2026-10-09，fork 落后的那一个 commit）**：给 Qwen3.5 加 packed sequences 支持 + packed Gated DeltaNet 测试，并删掉了 AGENTS.md 里对应的 stale packing caveat。读 fork 的 AGENTS.md 时注意：其中“packed 数据管线是 deferred follow-up，尚无训练入口设置 cu_seqlens”一段在上游已被 #108 改写，不要按旧版理解。

## 14. SFT / RL 扩展：slime 是正在发生的方向

- #95（2026-09-20）：SFT through slime。把 checkpoint load/save 与 appstate 从 pretrain task 抽出；简化分布式 setup 使已有 process group 的宿主能复用；YaRN mscale 修正；`training.model` 改接模型目录而非 config.json；`clip_grad_norm` 移到 optimizer 旁（即 #96 的起因）。
- #102（2026-10-04）：colocated RL through slime。`gate.router_replay` 把 `(num_tokens, top_k)` 映射到专家索引，eager 里解析、作为参数传给 gate，使编译后的 gate 看到 graph input 而非追踪 Python 状态；DualPipeV 把飞行中的 micro-batch 发布为 `training.current_microbatch`，供 replay store 按 (micro-batch, layer) key 路由；`iter_canonical_parameters` 逐个 gather 全量参数（额外内存只一个参数，供 RL 调用方推权重进推理引擎；`full_tensor` 是 collective，所有 rank 必须消费完整个迭代器）；`offload_training_state` / `reload_training_state` 把本 rank 参数分片与 optimizer 状态移到 pinned host 内存并缩掉 storage（pinned buffer 进程内池化，offload 21.76 GiB/rank 在 PP=2/EP=8 下从 11.79s 降到 1.09s，reload 0.84s）。
- 配套的 `pith-rl` 是独立仓库（不在本 repo 内），PithTrain 侧只暴露 hook。#49（2026-06-03 问 RL/SFT/OPD 计划）在 #95/#102 之后事实性地被回答了大半，但 issue 仍开着，4 条评论，未确认维护者是否打算关闭。
- Open PR #105：给 RL 目标加 fused log-prob + entropy operator，与 #102 同一方向。

## 15. Operators

`pithtrain/operators/`，每个 operator 都在同一模块内带 PyTorch reference 实现（这是测试对比的对象，也是该层的契约）：

- `ep_dispatch.py`（664 行）：专家并行 token dispatch 的 fused Triton kernel + 编排，带去重（`fused_dedup_prepare_dispatch`、`prepare_dispatch`、`build_expert_idxs`、`adjust_expand_idx`）。
- `token_scatter.py`（390 行）：grouped GEMM 之前按专家分组 token 的 Triton scatter kernel。
- `ring_attention.py`：zigzag ring attention（标准 + MLA-aware 变体）。
- `deepgemm_quantize.py`：DeepGEMM 风格 FP8 量化 fused Triton kernel。
- `cp_sequence.py`：两种 CP 布局定义、转换、以及 depthwise conv 需要的“前一 rank 的末尾 token”获取。
- `grouped_linear.py`（285 行）：BF16 grouped linear；`linear.py`：FP8/DeepGEMM 路径；`flash_attn_v4.py`：FA4 包装；`gated_delta_rule.py`：Qwen3.5 线性注意力；`all_to_all.py`：可微 collective 包装；另有 `silu_mul.py`、`clamped_swiglu.py`、`indexed_bias_add.py`、`cross_entropy.py` 等小 fused kernel。
- 包装方式：很多 kernel 包成 PyTorch custom op（手写 autograd），使 `torch.compile` 把它当作一个不透明节点；全部 transformer 计算以 `torch.compile(fullgraph=True)` 编译（MoE 专家计算除外，其 per-expert shape 在 EP 下数据依赖）。fullgraph 是刻意的：静默 graph break 会变成编译错误。

## 16. Agent skills：仓库的一等公民

`.agents/skills/` 共 10 个：`add-memory-prints`、`add-new-model`、`analyze-nsys-profile`、`capture-nsys-profile`、`estimate-memory`、`launch-with-slurm`、`setup-benchmark-inputs`、`validate-correctness`、`validate-performance`、`wandb-tracking`。另有 `.github/review-prompts/` 四个评审维度：correctness、performance、compactness、consistency（每份约 3K），以及按需触发的 Claude review（PR 评论 `@claude review`）。

AGENTS.md 的要求：匹配到 skill 就调用 skill，不要重新推导流程；跟着 skill 走发现它过时/错误/不完整时，**surface 出来给 review，不要静默改**；做出的可重复流程值得沉淀时，提议新 skill。

## 17. 测试与验证

- 单卡单元测试：`pytest tests/...`（kernel、op、layer protocol）。FP8 正确性用归一化平方误差 `calc_diff`，阈值通常 `< 1e-3`，不是精确相等。
- 多卡集成：`bash tests/test_dualpipev.sh`（torchrun 启动 DualPipeV + FSDP，约 4 GPU，pp=2 ep=2）。
- 坑：`F.grouped_mm` 可能往 padding 行（`grouped_mm_offs[-1]` 之外）写 NaN，比较前先截到 `[:actual_M]`。
- 数值类改动（kernel、调度、FP8、精度）要用 `validate-correctness` skill 对比分支与基线的逐 step loss 曲线，并在 PR 里报告结果。
- 测试在缺依赖时优雅跳过（无 CUDA、缺可选包、GPU 不够），所以在能真正跑通的硬件上跑与改动相关的测试。

## 18. 贡献姿态（提交前必查）

- **无 PR 模板，无 CONTRIBUTING 之外的格式要求，无 DCO / CLA**（与 InferSim 的 CLA、ServerlessLLM 的 Signed-off-by 都不同）。但 PR 正文应按 CONTRIBUTING 清单自查：最小改动、无新增隐式 indirection、ruff 通过、相关测试通过、动 engine 则跑 `test_dualpipev.sh`、数值改动做 loss-curve parity、新配置项写 field docstring、用户可见变化在 PR 里注明。
- **CI 只有 `precommit-check`**：push 到 main 与 PR 时跑 pre-commit（ruff check + format）。**没有跑单元测试的 CI**，测试全靠本地。这意味着：提交前本地验证是唯一的防线，reviewer 也会按这个预期看你的 PR（是否自带测试证据）。
- 另有两个 Claude workflow（`claude.yml`、`claude-code-review.yml`），review 按需触发，不拦合并。
- License：Apache-2.0。动 `pithtrain/pipeline/` 时保留 DeepSeek DualPipe 的 MIT 署名与根目录 NOTICE。
- 硬件门槛：Hopper（SM90）或 Blackwell（SM100），CUDA ≥ 13.0，Python ≥ 3.12，`uv` 管理依赖。`torch` 钉在 `>=2.13.0,<2.14`（#85 曾 pin 快速变动的依赖并 bound torch < 2.14），DeepGEMM 从 git rev 钉死。

## 19. Issue / PR 全景（4 issues + 11 PRs）

### Open issues（4）

| # | 标题 | 状态判断 |
|---|---|---|
| #81 | 提议：找最优正确配置的 agent skill（“time to best correct config”） | 2026-07-30 由外部用户 DAN-DNA-AND 提出，0 评论，**无人回应**。大提案，按 CONTRIBUTING 应先 issue 讨论；但它本身就是讨论帖，未见维护者表态。 |
| #80 | 有计划支持 VLM 吗？ | 2026-07-30 提问，1 条评论（评论内容未读），未见路线图回应。 |
| #79 | Qwen3-30B-A3B 在 PithTrain EP4/PP2 vs TorchTitan FSDP8 的吞吐差异 | 2026-07-22，7 条评论。报告者贴了完整复现配置与 load-balance 数据，并澄清：TorchTitan 最优配方 19.52k TPS/GPU 是 best-recipe 对比；**同几何（MBS1/GBS1024）下 TorchTitan 约 12.05k**，PithTrain 17.5k。Issue 仍开着，未见维护者定论。#81 由此讨论衍生。 |
| #49 | 未来是否扩展到 post-train（RL/SFT/OPD） | 2026-06-03，4 条评论。#95/#102 事实上推进了 SFT 与 colocated RL，但 issue 仍开着，未确认是否该关闭。 |

### Open PRs（11，按标题与作者归纳；正文未逐条拉取）

| # | 标题 | 作者 / 时间 | 备注 |
|---|---|---|---|
| #111 | Fix expert checkpoints for uneven FSDP shards and HSDP | GaokaiZhang，10-06 | checkpoint 方向，与 #109/#110 同簇 |
| #110 | Bound memory usage during HF checkpoint import | akaashrp，10-05 | checkpoint 方向 |
| #109 | Stream HF safetensors directly into FSDP-sharded training | Rishith18，10-02 | checkpoint 方向；作者是核心圈外的新面孔 |
| #107 | Support packed sequences under CP for Qwen3 | GaokaiZhang，09-30 | CP + packed 方向，与 #108 衔接 |
| #106 | Release activations layer by layer in forward-only steps | GaokaiZhang，09-30 | 显存方向 |
| #105 | Add a fused log-prob and entropy operator for RL objectives | GaokaiZhang，09-30 | RL 方向 |
| #101 | Add native Qwen3-Omni Thinker text pretraining（draft） | kldxbjz，09-27 | Omni 三件套之一，draft |
| #100 | Prepare staged Omni data and connect it to pretraining（draft） | kldxbjz，09-24 | Omni 三件套之一，draft |
| #98 | Scaffold native Qwen3-Omni correctness comparison（draft） | kldxbjz，09-22 | Omni 三件套之一，draft |
| #83 | [Feature] Add Nemotron-CC v2 dataset support | akaashrp，08-06 | 数据集方向，开了两个月未合入 |
| #78 | Add tirx silu_mul kernel | yagneekp，07-20 | 唯一外部作者 PR，近三个月未合入；外部 kernel PR 在此仓库的存活率是未知数，动手前应先看它卡在哪里 |

格局：核心团队以每月数个 PR 的速度推进 checkpoint / CP-packed / RL / Omni 四条线；外部声音（#81、#80、#78）回应慢。#78 的停滞是外部贡献者最需要先弄清的信号。

## 20. 与相关项目的关系

- **DeepSeek DualPipe**：pipeline 调度的源头，本仓库是其“agent-native 重写 + overlap”。动 pipeline 先读 DualPipe 的原始设计与本仓库的差异（V 形、5-stage、WeightGradStore）。
- **Megatron-LM**：load-balance 指标约定（1.0 = 完美平衡）沿用 Megatron；用户已有 Megatron 对比笔记（fork main 上的个人文件）。
- **TorchTitan**：#79 里作为参照系出现。PithTrain 的定位不是“比 TorchTitan 快”，而是“在紧凑可读的前提下达到生产级性能”。
- **verl / uni-agent / pith-rl**：RL 训练栈。PithTrain 不做 RL 训练器本身，只经 slime 暴露 colocated RL 的 hook（router replay、权重同步、状态 offload），训练编排在 `pith-rl`。与 uni-agent（把任意 agent harness 接入 RL）的关系是互补：一个管训练引擎，一个管 agent 侧。
- **Automodel / Molt**：通用训练库 vs MoE 专用紧凑系统。PithTrain 的 4D 并行与 folding 是 MoE 专属设计，不与 Automodel 的通用 SPMD 直接对标。

## 21. 值得注意的判断

- **紧凑性是硬约束，不是口号**。新模型、新功能、新 skill 都要先过“该不该进仓库”这一关。#83（Nemotron-CC 数据集）两个月未合入，很可能就是这条标尺在起作用。
- **状态放错函数的 bug 形态**（#96）比逻辑 bug 更值得防：装饰器、`training.current_microbatch` 这类进程全局状态、FSDP 的 divide factor，都是“一行放错地方，全局静默训错”的位置。读代码时对这类位置要格外敏感。
- **两个“唯一定义”**：`zigzag_spans`（CP 划分）、训练 step 里唯一一次梯度归一化。任何与之重复或绕过它的实现都是 smell。
- **CI 不跑测试**：本地验证证据就是 PR 的可信度本身。没有 GPU 时，能做的只有 CPU 可跑的部分（py_compile、ruff、纯逻辑测试）+ 明确标注未跑的部分。
- **Fork 落后 1 个 commit（#108）**：任何基于 fork main 的工作都要先意识到 AGENTS.md 里的 packing 描述已过期；贡献分支必须从上游 `87208d9` 起。

## 22. 后续可看的方向（仅观察，未认领，未动手）

- #81 的 skill 提案：需要维护者先定 correctness contract 与初始搜索空间，0 回应状态下直接实现大概率偏离预期。
- #80 的 VLM：需维护者给路线图方向，单凭提问贴无法确定接口。
- #78 的停滞原因：先读它的 review/CI 状态，才能判断外部 kernel PR 在此仓库的真实门槛。
- #109/#110/#111 的 checkpoint 簇：先确认三者之间的依赖与合并顺序，避免撞车。
- #107 与 #108 的衔接：#108 落地 Qwen3.5 packed，#107 做 Qwen3 的 CP 下 packed，两者边界是否留有未覆盖的组合（如 Qwen3.5 的 CP 下 packed）需读代码确认。

