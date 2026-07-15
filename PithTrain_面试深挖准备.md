# PithTrain 深度面试准备手册

> 基于 PithTrain 框架源码逐模块精读 + NeurIPS 2026 论文精读，面向 LLM 算法实习岗位的 LLM & Agent 应用/后训练方向面试准备。
> 建议在自我介绍部分以 PithTrain 为轴心展开，能够 "讲清楚一个模块为什么这么设计、遇到了什么问题、如何解决、trade-off 是什么"。

---

## 目录

1. [自我介绍模板](#1-自我介绍模板)
2. [Part I：LLM 基础核心（必考）](#2-part-i-llm-基础核心必考)
3. [Part II：MoE 架构（重点）](#3-part-ii-moe-架构重点)
4. [Part III：DualPipeV 流水线并行（框架核心）](#4-part-iii-dualpipev-流水线并行框架核心)
5. [Part IV：4D 并行策略（PP×DP×CP×EP）](#5-part-iv-4d-并行策略ppxdpcxep)
6. [Part V：FP8 训练与低精度优化](#6-part-v-fp8-训练与低精度优化)
7. [Part VI：Ring Attention 与 Context Parallelism](#7-part-vi-ring-attention-与-context-parallelism)
8. [Part VII：优化器与训练技巧（Muon / WSD）](#8-part-vii-优化器与训练技巧muon--wsd)
9. [Part VIII：Checkpointing 与 Resharding](#9-part-viii-checkpointing-与-resharding)
10. [Part IX：Kernel 优化与 Triton 实践](#9-part-ix-kernel-优化与-triton-实践)
11. [Part X：Agent-Native 设计哲学（论文核心贡献）](#10-part-x-agent-native-设计哲学论文核心贡献)
12. [Part XI：后训练方向延伸](#11-part-xi-后训练方向延伸)
13. [Part XII：系统级深挖题（区分度最高的题）](#12-part-xii-系统级深挖题区分度最高的题)
14. [Part XIII：实现题（手写代码/伪代码）](#13-part-xiii-实现题手写代码伪代码)
15. [面试策略总结](#14-面试策略总结)

---

## 1. 自我介绍模板

> **核心原则**：用 PithTrain 作为"抓手"，把零散的知识点串成一个有深度的故事线。
> **加分项**：如果能提到论文中提出的 Agent-Task Efficiency 概念，说明你不仅读了代码，还读了论文。

**参考话术（2-3分钟）：**

> "我最近在参与 CMU 开源的 PithTrain 项目，这是一个 ~11K 行的 MoE 训练框架，核心创新是 DualPipeV——一种 V 形双向流水线并行策略，可以把 Transformer 的每一层拆成 5 个阶段（Attention/Dispatch/MLP/Combine/Aggregate），实现计算与通信的细粒度重叠。
>
> 我在项目中主要做了三件事：第一，深入理解了 MoE Expert Parallelism 的 token dispatch/combine 流程，包括 EP 通信中的 token 去重（dedup）算法和自定义 Triton kernel 优化；第二，研究了 4D 并行（PP×DP×CP×EP）设备网格的设计，以及 FSDP2 在 pipeline 并行下的集成方式；第三，分析了 FP8 训练在 Hopper/Blackwell 上的 block scaling 方案。
>
> 我最近也在读 PithTrain 的 NeurIPS 2026 论文，里面提出了一个很有意思的概念——Agent-Task Efficiency（ATE），即用 AI coding agent 来理解、操作和扩展训练框架的效率。PithTrain 通过四个设计原则（代码紧凑、Python-native、无隐式间接、内置 agent skills）实现了比 Megatron 和 TorchTitan 更高的 ATE（agent turns 减少 62%，active GPU time 减少 64%）。
>
> 我对 LLM 预训练和后训练（SFT/RLHF）都很感兴趣，关注过 MoE 负载均衡、梯度裁剪、学习率调度等训练技巧的实际效果。PithTrain 虽然代码量小，但把 pipeline bubble、all-to-all overlap、wgrad 延迟释放这些系统细节都实现得很干净，很适合用来理解大模型训练的全链路。"

**关键信息点**（面试官会从自我介绍里挑出来追问）：
- "DualPipeV 具体做了什么？" → Part III
- "MoE 的 token dedup 是什么？" → Part II
- "FP8 block scaling 怎么工作？" → Part V
- "Pipeline bubble 怎么解决？" → Part III
- "Muon 优化器和 AdamW 有什么区别？" → Part VII
- "Agent-Task Efficiency 是什么？" → Part X
- "ATE-Bench 怎么评估 agent 效率？" → Part X

---

## 2. Part I：LLM 基础核心（必考）

### 2.1 Transformer 架构

**必考题 1：请描述标准 Transformer Decoder Layer 的计算流程**

**应回答的层次：**
```
hidden_states
  → LayerNorm (RMSNorm)
  → Attention: Q = W_q · x, K = W_k · x, V = W_v · x
    → RoPE (Rotary Position Embedding)
    → Softmax(QK^T / √d) · V
    → O = W_o · output
  → Residual: x + O
  → LayerNorm (RMSNorm)
  → MLP: gate(x) ⊙ up(x) → down_proj(SiLU(gate) ⊙ up)
  → Residual: x + MLP(x)
```

**可深挖点：**
- Q：为什么用 Pre-LN 而不是 Post-LN？**A**：Pre-LN 的梯度更稳定，训练初期不会出现梯度爆炸。Post-LN 在深层网络中梯度容易消失，需要 warmup 或 careful initialization。
- Q：RMSNorm 和 LayerNorm 的区别？**A**：RMSNorm 去掉均值中心化，只做 RMS 缩放，计算量减半。PithTrain 中所有 norm 都是 `nn.RMSNorm`。
- Q：SwiGLU 和标准 MLP 的区别？**A**：SwiGLU 用 `SiLU(gate) ⊙ up` 代替 `SiLU(gate + up)`，PithTrain 的 `silu_mul` 操作符实现了这个 fused kernel。

**对应 PithTrain 源码：**
- `pithtrain/models/qwen3_moe.py:Qwen3MoeDecoderLayer._forward_attn_compute()` — LN + Attn + LN
- `pithtrain/operators/silu_mul.py` — fused SiLU × multiplication
- `pithtrain/models/qwen3_moe.py:Qwen3MoeMLP.forward()` — SwiGLU MLP

---

### 2.2 Attention 机制

**必考题 2：Grouped Query Attention (GQA) 和 Multi-Head Attention (MHA) 的区别**

**应回答：**
- MHA：Q、K、V 都有 N_heads 个头，计算量 O(N²·d)
- GQA：Q 有 N_q_heads 个头，K、V 共享 N_kv_heads 个（N_kv_heads < N_q_heads）
- 效果：减少 KV cache 大小，加速 decode 阶段，精度损失很小
- PithTrain 中 Qwen3 使用 GQA：`num_attention_heads=32, num_key_value_heads=8`

**深挖点：**
- Q：KV cache 的显存公式是什么？**A**：`2 × batch × seq_len × N_kv_heads × head_dim × dtype_bytes`
- Q：FlashAttention 的核心优化是什么？**A**：tiling + recomputation，将 O(N²) 的显存降到 O(N)，通过分块计算 attention 并在 backward 时重计算

**对应 PithTrain 源码：**
- `pithtrain/models/qwen3_moe.py:Qwen3MoeAttention` — GQA with `num_key_value_groups`
- `pithtrain/operators/flash_attn_v4.py` — FlashAttention v4 wrapper

---

### 2.3 Position Embedding

**必考题 3：RoPE（Rotary Position Embedding）的原理**

**应回答：**
```
q_m = (W_q · x) * cos(m·θ) - rotate_half(W_q · x) * sin(m·θ)
k_m = (W_k · x) * cos(m·θ) - rotate_half(W_k · x) * sin(m·θ)
```
- 将位置信息编码到 Q、K 的旋转中，而非加在 embedding 上
- 绝对位置 → 相对位置注意力
- θ_i = base^(-2i/d)，i 是维度索引
- PithTrain 中缓存 cos/sin 表（`Qwen3MoeRotaryEmbedding`），避免重复计算

**深挖点：**
- Q：RoPE 的外推性怎么解决？**A**：YaRN (Yet another RoPE extensioN) — 对高频/低频维度分别做不同的缩放因子，DeepSeek-V2 采用。PithTrain 的 DeepSeek 模型实现了完整的 YaRN。
- Q：为什么 DeepSeek 用 `q_norm` 和 `k_norm`？**A**：训练不稳定时 Q/K 的范数会爆炸，加 RMSNorm 约束可以稳定训练。PithTrain 的 `Qwen3MoeAttention` 也有 `q_norm` 和 `k_norm`。

---

## 3. Part II：MoE 架构（重点）

### 3.1 MoE 基本概念

**必考题 4：MoE（Mixture of Experts）是什么？和 Dense 模型比有什么优势？**

**应回答：**
```
MoE Layer:
  x → Router(Gate) → top-k 专家选择 → Token Dispatch → Experts 计算 → Combine → x + output

Dense: 每层所有参数都参与计算
MoE:   只有 top-k 专家参与计算，总参数量大但激活参数量小

Qwen3-30B-A3B: 总参 30B，激活参 3B（每 token 只激活 1/10 的参数）
```

**优势：**
1. 更大的模型容量（总参数量），相同的计算量
2. 专家可以 specialize 到不同的 token 类型
3. 可以通过 EP（Expert Parallelism）分布到多卡

**挑战：**
1. Load balance：专家分配不均导致部分 GPU 空闲
2. 通信开销：EP 的 all-to-all 通信
3. 训练稳定性：路由噪声、 expert collapse

**对应 PithTrain 源码：**
- `pithtrain/models/qwen3_moe.py:Qwen3MoeGate` — top-k 路由
- `pithtrain/models/qwen3_moe.py:Qwen3MoeMoE` — MoE block
- `pithtrain/modules/load_balance.py` — 3 种负载均衡 loss

---

### 3.2 Top-k 路由与负载均衡

**必考题 5：MoE 的 Top-k 路由如何工作？Load Balance Loss 有哪些类型？**

**PithTrain 中 Gate 的计算流程（`Qwen3MoeGate.compute`）：**
```
1. logits = W_gate · hidden_states  (shape: [N_tokens, num_experts])
2. scores = softmax(logits)          (float32)
3. topk_weight, topk_idx = topk(scores, k=num_experts_per_tok)
4. if norm_topk_prob: topk_weight /= sum(topk_weight)  # 归一化
5. if training: lb_loss = load_balance_loss_fn(scores, topk_idx)
6. topk_weight = MoELoadBalanceLossInjector(topk_weight, lb_loss)  # 直通梯度注入
```

**PithTrain 实现的 3 种 Load Balance Loss：**

| 类型 | 公式 | 同步范围 | 积累方式 |
|------|------|----------|----------|
| micro-batch | `coef × E × Σ(f_i × P_i)` | 无（单 rank） | 每个 micro-batch 独立 |
| global-batch | 同上 | DP × EP all-reduce | 梯度累积步内累加 |
| sequence | 同上 | CP all-reduce | 按 sequence 独立计算后平均 |

其中 `f_i` = 选中 expert i 的 token 比例，`P_i` = expert i 的平均路由概率。

**深挖点：**
- Q：`MoELoadBalanceLossInjector` 为什么用自定义 autograd Function 而不是直接加 loss？**A**：直通操作符（straight-through estimator）——forward 时返回 topk_weight 不变，backward 时注入 `∂lb_loss/∂topk_weight = 1`。这样 load balance loss 的梯度会直接加到 topk_weight 上，影响 router 的参数更新，而不会改变前向的计算结果。
- Q：Global-batch loss 为什么需要 DP × EP all-reduce？**A**：每个 DP rank 看到的数据不同，如果不同步，每个 rank 计算的 f_i 是局部的，不能反映全局负载。all-reduce 确保所有 rank 看到相同的 expert 选择频率。
- Q：`_expert_token_counts` 为什么不用 `torch.bincount`？**A**：`bincount` 会先调用 `input.max().item()` 同步获取最大值，在 GPU 上有同步开销。PithTrain 用 `scatter_add_` 实现异步版本。

**对应 PithTrain 源码：**
- `pithtrain/modules/load_balance.py:MicroBatchLoadBalanceLoss` (L38-68)
- `pithtrain/modules/load_balance.py:GlobalBatchLoadBalanceLoss` (L71-134)
- `pithtrain/modules/load_balance.py:SequenceLevelLoadBalanceLoss` (L137-198)
- `pithtrain/modules/load_balance.py:MoELoadBalanceLossInjector` (L21-35)

---

### 3.3 Expert Parallelism 与 Token Dispatch

**必考题 6：EP（Expert Parallelism）的 all-to-all 通信流程是什么？Token Dedup 解决了什么问题？**

**EP 通信流程（`moe_ep_prepare_dispatch`）：**

```
Rank i 上的 token 经过 router 后，每个 token 被分配到 k 个 experts。
如果多个 token 被分配到同一个 EP rank（不同 expert），它们需要发送到同一个 rank。

Step 1: 计算每个 EP rank 收到多少 token（去重后）
         tokens_per_ep_rank = Σ(expert ∈ rank_i) count(expert)
         dedup_tokens_per_gpu = 每个 rank 的独立 token 数（去重）

Step 2: Metadata all-to-all（piggyback）
         把 tokens_per_expert 和 dedup_tokens 打包一起发

Step 3: 对每个 token，计算它在去重后的排序位置（expand_idx）

Step 4: 真正的 token all-to-all（发送去重后的 tokens）
         + expand_idx all-to-all

Step 5: 接收后，用 expand_idx 恢复完整 token 序列
```

**Token Dedup 的价值：**
- 一个 token 如果被分配到 rank 0 的两个 experts，不 dedup 需要发 2 次
- Dedup 后只发 1 次，减少通信量（特别是 expert 数量多时效果显著）

**深挖点：**
- Q：为什么不用 PyTorch 原生的 scatter/argsort 而要写 Triton kernel？**A**：EP dispatch 是每层每个 micro-batch 都要执行的热路径。PyTorch 的 scatter + argsort + nonzero + searchsorted 有 ~22 次小 kernel  launch，每次有 CPU launch overhead。3 个 Triton kernel 把整个流程 fuse 成 3 次 launch：
  1. `_dedup_bincount_kernel` — 无原子操作的并行 bincount（用 `tl.histogram`）
  2. `_reduce_and_prefix_sum_kernel` — 单 CTA 做 cross-CTA 规约 + 前缀和
  3. `_dedup_scatter_expand_kernel` — counting sort O(n) 替代 argsort O(n log n)
- Q：`tl.histogram` 为什么比 atomic add 好？**A**：`tl.histogram` 是 warp-level reduction，每个 warp 内部用 shared memory 做 reduction，不需要全局原子操作，避免了 warp divergence 和 contention。
- Q：send_meta 的设计为什么要 piggyback？**A**：本来需要两次 all-to-all（一次发 token counts，一次发 tokens），把 dedup counts 和 token counts interleave 到一个 `(ep_size, experts_per_rank + 1)` 的 tensor 里，只需要一次 all-to-all。

**对应 PithTrain 源码：**
- `pithtrain/operators/ep_dispatch.py:fused_dedup_prepare_dispatch` (L270-419)
- `pithtrain/operators/ep_dispatch.py:moe_ep_prepare_dispatch` (L578-681)
- `pithtrain/operators/ep_dispatch.py:_dedup_bincount_kernel` (L36-103)
- `pithtrain/operators/ep_dispatch.py:_reduce_and_prefix_sum_kernel` (L106-174)

---

## 4. Part III：DualPipeV 流水线并行（框架核心）

### 4.1 Pipeline Parallelism 的基本问题

**必考题 7：Pipeline Parallelism 的 bubble 问题是什么？**

**应回答：**
```
标准 1F1B (one-forward-one-backward):

Step 1:  F0      (其他 rank 空闲)
Step 2:  F0 F1
Step 3:  F0 F1 F2
...
Step PP: F0...F_{PP-1}     ← 所有 rank 都在做 forward
Step PP+1: B0 F1...F_PP    ← rank 0 开始 backward，其他继续 forward
...
Step 2PP-1: B0...B_{PP-2} F_PP
Step 2PP: B0...B_{PP-1}    ← bubble 结束

Bubble 时间占比 = (PP-1) / (PP + num_microbatches - 1)
```

**解决方案方向：**
1. 增大 microbatch 数量
2. 1F1B（交错执行）
3. DualPipeV（双向 + 通信重叠）

---

### 4.2 DualPipeV 的 V 形布局

**必考题 8：DualPipeV 的 V 形布局和 5 阶段分解是什么？**

**V 形布局：**
```
Standard PP:  model 切 PP 块，每个 rank 拿连续的块
DualPipeV:    model 切成 2×PP 块，rank r 拿块 r 和块 2·PP-1-r

PP=4 的例子（model 有 8 个 chunks）：
rank 0: chunk 0, chunk 7
rank 1: chunk 1, chunk 6
rank 2: chunk 2, chunk 5
rank 3: chunk 3, chunk 4

好处：每个 rank 既做前向也做后向，不会出现"只有前向或只有后向"的 idle rank
```

**5 阶段分解（核心创新）：**
```
每个 Transformer Layer 被拆成 5 个阶段，cut 在 EP 通信边界：

Stage 1 (Attention):   LayerNorm → Attention → LayerNorm → Expert Routing  [compute]
Stage 2 (Dispatch):    All-to-all: 发 tokens 到 experts 所在的 rank        [comm]
Stage 3 (MLP):         Expert/MLP 计算                                       [compute]
Stage 4 (Combine):     All-to-all: 把 expert outputs 聚合回原 rank          [comm]
Stage 5 (Aggregate):   Weighted sum + residual connection                    [compute]

Stage 2 和 4 在独立的 CUDA comm stream 上运行，
可以和另一个 micro-batch 的 compute 重叠！
```

**深挖点：**
- Q：5 阶段为什么不合并成更少？**A**：MoE 的 EP 通信（all-to-all）是必须的，但通信量相对小（token 数量远小于 hidden_dim）。如果把通信埋到更大的 compute 阶段里，通信就 hide 不住了。拆成 5 阶段让每个 compute 阶段足够小，通信可以被下一个 micro-batch 的 compute 完全覆盖。
- Q：`IntermediateTensors` 预分配的作用？**A**：pipeline 循环中每次 forward/backward 都要分配 tensor，频繁的 CUDA 内存分配和释放有 overhead。预分配 `IntermediateTensors` 并复用，实现 zero-allocation pipeline execution。注意是 `in-place` 修改引用。
- Q：Zero Bubble 和 WeightGradStore 是什么？**A**：在标准 pipeline 中，wgrad（weight gradient）在 backward 的 compute 阶段计算。Zero Bubble 策略把 wgrad 的计算延迟到 idle 时间片，让 bubble 更小。`WeightGradStore` 是一个延迟队列，先存 wgrad 的计算函数，等到 scheduler 有空时才真正执行。

**对应 PithTrain 源码：**
- `pithtrain/dualpipe/dualpipev.py:DualPipeV` — 主调度器（8 步算法）
- `pithtrain/dualpipe/dualpipev.py:DualPipeV.step()` (L418-589) — 8 步调度
- `pithtrain/dualpipe/overlap.py:overlapped_forward_backward()` (L67-390) — 核心 F/B 重叠循环
- `pithtrain/dualpipe/execution.py` — 5 阶段的 stage1_f/b ~ stage5_f/b
- `pithtrain/dualpipe/utils.py:WeightGradStore` — wgrad 延迟
- `pithtrain/dualpipe/utils.py:FP8WeightCacheControl` — FP8 weight cache

---

### 4.3 DualPipeV 的 8 步调度算法

**必考题 9：请描述 DualPipeV.step() 的 8 步调度算法**

```
Step 1 (nF0):        预热 — 只有 phase0（第一个 V-chunk）做 forward
                     共 (PP - pp_rank - 1) × 2 次

Step 2 (nF0F1):      双 forward — phase0 和 phase1 交替 forward
                     共 (pp_rank + 1) 次迭代，每次 2 个 forward

Step 3 (nB1W1F1):    开始 backward — phase1 backward + wgrad + phase1 forward
                     共 (PP - pp_rank - 1) 次迭代（Zero Bubble）

Step 4 (nF0B1F1B0):  主循环 — F/B 重叠（核心！）
                     phase0 forward + phase1 backward（overlapped）
                     phase1 forward + phase0 backward（overlapped）
                     共 (num_chunks - 2×PP + pp_rank + 1) 次

Step 5 (nB1F1B0):    phase1 backward + F/B overlap
                     共 (PP - pp_rank - 1) 次

Step 6 (nB1B0):      纯 backward
                     共 (pp_rank + 1) 次（后半段 Zero Bubble）

Step 7 (nWB0):       剩余 wgrad + backward（Zero Bubble）

Step 8 (nW):         纯 wgrad 释放
```

**深挖点：**
- Q：Step 4 中的 `overlapped_forward_backward` 具体做了什么？**A**：在一个函数调用中，同时执行 module0 的 forward 和 module1 的 backward。具体来说：module0 的 Stage 1 forward → module1 的 Stage 5 backward → module0 的 Stage 2 forward → module1 的 Stage 4 backward → ... 在一个循环中交替执行，中间用 CUDA events 保证 compute stream 和 comm stream 的正确同步。
- Q：为什么 `num_chunks >= 2 * pp_size`？**A**：V 形布局需要至少 2×PP 个 chunks（每个 rank 有两个 chunks，forward 和 backward 各需要一个完整的 V）。否则某些阶段没有足够的 micro-batch 可以调度。
- Q：FSDP 的 post_backward 为什么手动调用？**A**：正常 `tensor.backward()` 会触发 FSDP 的 autograd hooks，但 pipeline 用的是手动 backward（`decoder_layer_backward`），不经过 autograd engine。所以 FSDP 的 post-backward callback 不会自动触发，需要在 step() 末尾手动调用 `run_post_backward()`。

**对应 PithTrain 源码：**
- `pithtrain/dualpipe/dualpipev.py:step()` L478-544 — 8 步实现
- `pithtrain/dualpipe/dualpipev.py:_forward_backward_chunk()` L312-321 — Step 4 核心
- `pithtrain/dualpipe/dualpipev.py:run_post_backward()` L565-588 — FSDP 手动回调

---

## 5. Part IV：4D 并行策略（PP×DP×CP×EP）

### 5.1 4D 设备网格

**必考题 10：PP、DP、CP、EP 分别解决什么问题？它们如何组合？**

| 维度 | 解决什么 | 机制 | 通信模式 |
|------|----------|------|----------|
| **PP** (Pipeline Parallel) | 模型太大放不下 | 每层切一块到不同 rank | P2P isend/irecv |
| **EP** (Expert Parallel) | MoE expert 太多 | 每个 rank 管一部分 experts | All-to-all |
| **CP** (Context Parallel) | 序列太长 | 把 sequence 维度切多块 | Ring All-to-All |
| **DP** (Data Parallel) |  batch 太小 | 每份数据跑一个副本 | All-reduce |

**组合规则：**
```
world_size = PP × DP × CP × EP
DP = world_size / (PP × CP × EP)  ← DP 是自动推导的

设备网格顺序：(PP, DP, CP, EP) outer → inner
CP 和 EP 放最内层：它们的通信最频繁（ring attention 每层一次，MoE all-to-all 每层两次）
保持在内层可以让通信在 NVLink 域内完成
```

**深挖点：**
- Q：FSDP 在 MoE 和非 MoE 参数上怎么 shard？**A**：
  - MoE experts：每个 EP rank 有自己的 experts，跨 DP×CP 冗余。FSDP 沿 `(DP, CP)` shard。
  - 非 MoE（attention、router、embedding、norm、lm_head）：跨 EP 也冗余。FSDP 沿 `(DP, CP, EP)` shard。
  - HSDP 模式：沿 DP 复制，在 `(CP, EP)` 内 shard。适合单卡 DP replica 能放下整个模型的情况。

**对应 PithTrain 源码：**
- `pithtrain/modules/distributed.py:setup_device_mesh()` (L165-191) — 4D mesh
- `pithtrain/modules/training.py:apply_fsdp()` (L350-414) — MoE vs non-MoE sharding

---

### 5.2 Context Parallelism 与 Zigzag 序列切分

**必考题 11：Zigzag Context Parallelism 的设计动机是什么？**

```
标准 CP (cp_size=4，sequence 切成 4 块):
  rank 0: [0, S/4)
  rank 1: [S/4, S/2)
  rank 2: [S/2, 3S/4)
  rank 3: [3S/4, S)

问题：causal attention 的工作量不均衡
  rank 0 的 K/V 来自 rank 3（远端），Q 只需 attend 到本地前 S/4
  rank 3 的 K/V 来自 rank 3（本地），Q 需要 attend 到前面 3S/4
  → rank 3 的 attention compute 是 rank 0 的 3 倍！

Zigzag CP (cp_size=4):
  chunk:  0  1  2  3  4  5  6  7
  rank:   0  1  2  3  3  2  1  0

  rank 0 拿 chunk 0（轻）+ chunk 7（重），工作量均衡
  rank 1 拿 chunk 1（轻）+ chunk 6（重）
  ...
  rank 3 拿 chunk 3（重）+ chunk 4（轻）

每个 rank 的 causal attention 计算量相同！
```

**深挖点：**
- Q：Zigzag 下 KV 怎么传递？**A**：Q 留在本地不动，K/V 按 +1 方向环形传递。每步 rank r 的 K/V 来自 `(r - step) mod cp_size` 对应的 rank。
- Q：数据加载时怎么做 zigzag sharding？**A**：`get_global_batch` 中 `front_offset = cp_rank * block`，`back_offset = (2*cp_size - cp_rank - 1) * block`，把两个不连续的 chunk 拼在一起。

**对应 PithTrain 源码：**
- `pithtrain/operators/ring_attention.py` (L1-49) — Zigzag 设计说明
- `pithtrain/tasks/pretrain_lm.py:get_global_batch()` (L70-126) — zigzag 数据加载

---

## 6. Part V：FP8 训练与低精度优化

### 6.1 FP8 基础

**必考题 12：FP8 训练的动机和挑战是什么？**

**动机：**
- H100/B200 有 FP8 tensor core，计算吞吐是 BF16 的 2 倍
- 显存减半，可以增大 batch size 或 model size

**挑战：**
- FP8 (E4M3) 的动态范围远小于 BF16：max ≈ 448 vs  BF16 max ≈ 65504
- 直接 cast 会溢出/下溢 → 需要 per-block scaling

**PithTrain 的方案（`deepgemm_fp8_quantize.py`）：**
```
Block Scaling:
  - 每 128 个元素为一个 block
  - 计算 block 的 amax（绝对最大值）
  - scale = ceil(amax / 448) → 最接近的 2 的幂（保证 cast 是精确的）
  - E8M0 scale (SM100+): 用 PTX cvt.rp 指令直接得到 exponent
  - FP32 scale (SM90): 用 IEEE-754 bit manipulation 取幂
  - 量化: x_fp8 = x / scale  (cast to E4M3)
  - 反量化: x_bf16 = x_fp8 * scale
```

**深挖点：**
- Q：为什么用 128-element block？**A**：DeepGEMM 库的 GEMM kernel 要求 block size = 128（K 维）。太大的 block 会导致精度损失（同 block 内不同 magnitude 的数值被同一个 scale 压缩），太小的 block 会增加 overhead。
- Q：Power-of-2 scale 的好处？**A**：量化 `x / 2^e` 和反量化 `x_fp8 * 2^e` 都是精确的 IEEE-754 操作，没有舍入误差。只有量化前的除法有舍入，且误差被 FP8 的精度限制自然 bound。
- Q：`torch.compile(fullgraph=True)` 在 MoE 上为什么禁用？**A**：MoE 的 per-expert shapes 在 EP 下是 data-dependent 的（每个 rank 收到的 token 数不同），会导致 graph break。PithTrain 用 `@torch.compiler.disable` 标记 EP dispatch 相关函数。

**对应 PithTrain 源码：**
- `pithtrain/operators/deepgemm_fp8_quantize.py:_compute_fp8_scale` (L39-81) — scale 计算
- `pithtrain/layers/deepgemm_fp8_linear.py` — FP8 Linear 和 Grouped Linear 实现
- `pithtrain/layers/factory.py` — BF16/FP8 切换

---

## 7. Part VI：Ring Attention 与 Context Parallelism

### 7.1 Zigzag Ring Attention 算法

**必考题 13：Ring Attention 如何实现超长序列的 causal attention？**

```
Ring Attention (cp_size = N):

每个 rank 持有一个长度为 S/N 的 sequence chunk。
Q 不动，K/V 按 +1 方向每步传一个 rank。

Step 0: rank r 用自己的 K/V 做 causal attention
Step 1: rank r 收到 rank (r-1) 的 K/V，和本地 Q 做 non-causal attention
        （因为 Q 的 block 0 不需要 attend 到这些 K/V）
Step 2: rank r 收到 rank (r-2) 的 K/V，只有 Q 的 block 1 attend 到这些 K/V
...
Step N-1: 所有 K/V 都旋转过一轮

Zigzag 变体：每个 rank 持有两个 chunks（front + back），K/V 包含两个部分，
            attention 的分块逻辑更复杂但计算量均衡。
```

**P2P 重叠：**
```
每个 ring step：
  1. 先 post 下一个 step 的 isend/irecv（batch_isend_irecv）
  2. 当前 step 的 K/V 已经就位，执行 flash attention
  3. 等待通信完成（next step 的 K/V 就绪）

通信和计算完全重叠！
```

**深挖点：**
- Q：为什么 backward 要两个 ring？**A**：Forward 只需要 Q 和旋转的 K/V。Backward 需要 dK 和 dV，它们也需要按相反方向旋转回到原位（因为 dK 的计算分散在所有 rank 上）。所以 backward 同时运行两个 ring：K/V 正向旋转（和 forward 一样），dK/dV 反向旋转。每个 rank 只保存一份 K/V，通过 partial dK/dV 累积得到完整的梯度。
- Q：FlashAttention 的 causal mask 在 zigzag 下怎么处理？**A**：分三种情况（ring_attention.py 注释的 `step == 0 / 1 <= s <= r / s > r`），根据 K/V 来源的 chunk 位置决定 causal mask 的起始位置。

**对应 PithTrain 源码：**
- `pithtrain/operators/ring_attention.py` — 标准 + MLA-aware 变体
- `pithtrain/operators/ring_attention.py:ring_attention_func` — forward ring
- `pithtrain/operators/ring_attention.py:mla_ring_attention_func` — MLA 变体

---

### 7.2 MLA (Multi-head Latent Attention)

**必考题 14：DeepSeek 的 MLA 是什么？和 GQA 有什么区别？**

```
标准 Attention:  Q, K, V 都是 d_model 维 → 投影到多头
GQA:              Q 有 N_q 头，K/V 共享 N_kv 头 → KV cache 减少
MLA (DeepSeek):   Q, K 先投影到低维 latent 空间，再做 RoPE，再投影回多头

具体流程：
  Q 路径：  x → W_dq → [latent_q] → RoPE → W_q → Q_heads
  K 路径：  x → W_dk → [latent_k] → RoPE → W_k → K_heads
  V 路径：  x → W_dv → V_heads

KV Cache 只存 latent 表示（低维），推理时才投影到 K/V heads
→ KV cache 大小减少到 1/√(compression_ratio)
→ 用 ckv/cqr 表示 compressed KV / compressed Q
```

**PithTrain 中 MLA 的实现特点：**
- DeepSeek-V2-Lite 实现了 MLA，包括 `kv_a_proj_with_mqa` 和 `kv_b_proj`
- Ring attention 有专门的 `mla_ring_attention_func`，处理 latent 的旋转和分块
- 使用 `gated_delta_rule` 算子做 latent KV 的压缩

**对应 PithTrain 源码：**
- `pithtrain/models/deepseek_v2_lite.py` — DeepSeek MLA 实现
- `pithtrain/operators/ring_attention.py:mla_ring_attention_func` — MLA ring attention
- `pithtrain/operators/gated_delta_rule.py` — latent KV 压缩

---

## 8. Part VII：优化器与训练技巧（Muon / WSD）

### 8.1 Muon 优化器

**必考题 15：Muon 优化器是什么？为什么 PithTrain 用它？**

```
Muon (Momentum Orthogonalized by Newton-Schulz):
  核心思想：对 2D 权重矩阵做 Newton-Schulz 正交化，然后 SGD with momentum

  更新公式：
    update = (1-β) · grad + β · momentum_buffer
    M = zeropower_via_newtonschulz5(update)  # 正交化
    param = param * (1 - lr * wd) - lr * scale_factor(M) · M

  scale_factor = 0.2 * max(out_dim, in_dim)^0.5  # 修正尺度

和 AdamW 的区别：
  AdamW: 对每个参数独立维护一阶/二阶矩 → 对 2D 权重来说，不同参数的学习率不一致
  Muon:   所有 2D 权重统一做正交化 → 隐式地在权重空间做白化，收敛更稳定
```

**PithTrain 的参数分类策略（`is_muon_param`）：**
```
Muon 优化：2D 隐藏权重
  ✓ attention q/k/v/o projections
  ✓ MLA projections (kv_a_proj_with_mqa, kv_b_proj)
  ✓ Dense MLP gate/up/down projections
  ✓ 3D 堆叠 expert 权重（stacked [E, out, in]）

AdamW 优化：其余参数
  ✗ 1D 参数（norm weights, biases, sinks）
  ✗ embedding / lm_head
  ✗ MoE gate/router（本质是分类器，不是隐层权重）
  ✗ 2D stacked expert biases
```

**深挖点：**
- Q：Newton-Schulz 5 迭代为什么够用？**A**：NS 迭代将矩阵的奇异值推向 1。5 次迭代对于大多数 LLM 的权重矩阵已经足够（奇异值谱相对集中），而且 Triton kernel 可以做 batched（对 stacked expert 权重一次性处理多个 expert）。
- Q：为什么 weight decay 对 1D norm 参数也生效？**A**：`weight_decay=0.1` 应用于所有参数，包括 RMSNorm 的 gamma。这可以防止每层的输出 RMS 不断膨胀——因为残差连接让输出是各层的叠加，如果没有 decay，深层输出的范数会指数增长。

**对应 PithTrain 源码：**
- `pithtrain/modules/optimizer.py` — Muon 实现 + `zeropower_via_newtonschulz5`
- `pithtrain/modules/training.py:is_muon_param()` (L39-56) — 参数分类
- `pithtrain/modules/training.py:make_muon_optimizer()` (L59-76) — Muon + AdamW 组合

---

### 8.2 Learning Rate 调度

**必考题 16：Warmup-Stable-Decay (WSD) 学习率调度是什么？**

```
WSD 三段式：
  1. Warmup:   linear from start_lr → lr  (warmup_steps = warmup_ratio × max_steps)
  2. Stable:   保持 lr 不变                (stable_steps = max_steps - warmup - decay)
  3. Decay:    从 lr → final_lr            (decay_steps = decay_ratio × max_steps)
               可以是 cosine decay 或 linear decay

为什么需要 warmup？
  - 训练初期参数随机，梯度范数很大，直接上大 lr 会导致数值不稳定
  - warmup 让模型先"稳定"下来再开始大范围探索

为什么需要 decay？
  - 训练后期应该收敛到局部最优，降低 lr 有助于精细调整
```

**对应 PithTrain 源码：**
- `pithtrain/modules/training.py:make_wsd_scheduler()` (L87-126)

---

## 9. Part VIII：Checkpointing 与 Resharding

### 9.1 Canonical vs Localized 格式

**必考题 17：PithTrain 的 checkpoint 格式设计为什么需要 resharding？**

```
问题：PP 把模型切成多个 stage，每个 rank 的 FQN 里有 DualPipeV 的 module.{N}. 前缀
      EP 把 experts 按 rank 堆叠（stacked [experts_per_rank, ...]）

Localized (运行时):
  module.0.layers.1.mlp.experts.gate_proj.weight  → shape [experts_per_rank, ...]

Canonical (磁盘):
  layers.1.mlp.experts.3.gate_proj.weight          → shape [..., ...] (单个 expert)
  没有 module.{N}. 前缀（用全局 layer ID）

Resharding 的作用：
  保存时：Localized → Canonical（strip prefix, expand stacked experts）
  加载时：Canonical → Localized（add prefix, stack experts per EP rank）

好处：checkpoint 与 PP/EP 配置无关 → 可以 resume 不同的并行布局
```

**深挖点：**
- Q：为什么磁盘格式要展开 stacked experts？**A**：如果磁盘存的是 `[experts_per_rank, ...]` 的 tensor，换个 EP size 就无法加载。展开成 `experts.0.weight, experts.1.weight, ...` 后，每个 expert 独立，可以按需组合。
- Q：HuggingFace 导入时如何检测 model-only checkpoint？**A**：`load_checkpoint` 读取 checkpoint metadata，如果所有 key 都以 `app.model.` 开头（没有 `app.optimizer.` 或 `app.scheduler.`），就认为 model_only=True，用 `set_model_state_dict(strict=False)` 非严格加载。

**对应 PithTrain 源码：**
- `pithtrain/modules/checkpoint.py:to_canonical_model` / `to_localized_model` — 模型格式转换
- `pithtrain/modules/checkpoint.py:to_canonical_optim` / `to_localized_optim` — 优化器格式转换
- `pithtrain/tasks/pretrain_lm.py:AppState` (L166-230) — DCP 集成

---

## 10. Part IX：Kernel 优化与 Triton 实践

### 10.1 自定义 Triton Kernel 设计原则

**必考题 18：PithTrain 为什么大量使用自定义 Triton kernel？设计原则是什么？**

```
PithTrain 的 kernel 选择逻辑：

1. 热路径才写 kernel
   EP dispatch, FP8 quantize, token scatter, ring attention K/V exchange
   → 每层每 micro-batch 必执行，累计调用次数 = 层数 × micro-batches

2. PyTorch reference impl 必保留
   每个 operator 都有一个 PyTorch 版本的参考实现，用于 correctness test
   test 模式：kernel output vs reference output，用 normalized squared error 比较

3. Fuse 小操作，减少 launch overhead
   silu_mul: SiLU(x) * y 在一个 kernel 里做
   FP8 quantize: abs → amax → scale → cast → transpose 在一个 kernel 里做

4. 减少动态内存分配
   pre-allocate output buffers，用 in-place 操作填充
   EP dispatch 的 pre-allocated dispatch_token_idxs / idxs / expand_idx
```

### 10.2 EP Dispatch Kernel 深入

**必考题 19：EP dispatch 的 3 个 Triton kernel 分别解决了什么问题？**

```
Kernel 1: _dedup_bincount_kernel
  输入：topk_ids [m, k]
  输出：per_cta_expert_hist [num_ctas, NE_PADDED]
        per_cta_gpu_hist [num_ctas, EP_PADDED]
  解决：并行统计每个 expert 被选多少次，以及每个 EP rank 收到多少独立 token
  优化：tl.histogram（warp-level reduction）替代 atomic add

Kernel 2: _reduce_and_prefix_sum_kernel
  输入：Kernel 1 的输出
  输出：expert_starts, gpu_starts, dedup_counters, sort_counters, send_meta
  解决：cross-CTA hist 规约 + 前缀和 + 清零计数器 + 构建 send_meta
  优化：单 CTA，完全 unroll EP_SIZE 和 EXPERTS_PER_RANK

Kernel 3: _dedup_scatter_expand_kernel
  输入：topk_ids, expert_starts, gpu_starts, 清零的 counters
  输出：dispatch_token_idxs, idxs, expand_idx
  解决：two-pass - pass 1 做 dedup scatter，pass 2 做 counting sort + expand_idx
  优化：counting sort O(n·k) 替代 argsort O(n·k·log(n·k))
```

---

## 11. Part X：Agent-Native 设计哲学（论文核心贡献）

> **来源**：NeurIPS 2026 论文 §3.1 (Agent-Native Design Principles) + §4 (ATE-Bench)
> 这是 PithTrain 区别于 Megatron/DeepSpeed/TorchTitan 的**核心创新点**，面试中问到 "为什么用 PithTrain" 或 "和 Megatron 比有什么优势" 时必答。

### 11.1 四个 Agent-Native 设计原则

**必考题 20：PithTrain 的四个 Agent-Native 设计原则是什么？和传统框架有什么不同？**

```
PithTrain 的四条设计原则（论文 §3.1）：

1. Compact codebase（代码紧凑）
   - PithTrain: ~11K LoC（纯 Python）
   - Megatron: ~149K LoC, DeepSpeed: ~167K LoC, TorchTitan: ~38K LoC
   - 紧凑的好处：agent 可以在单个 context window 内读完整个代码库

2. Python-native（Python 原生）
   - 全栈 Python，只有 custom kernel 用 Triton（Python DSL）
   - Megatron: 依赖 TransformerEngine（C++/CUDA 扩展）
   - DeepSpeed: 大量 in-tree CUDA 扩展
   - 好处：agent 不需要跨语言，traceback 可读，不需要编译扩展

3. No implicit indirection（无隐式间接调用）
   - 不用 plugin registry、runtime spec、string-keyed resolution
   - 每个 model 是一个自包含的文件（qwen3_moe.py, deepseek_v2_lite.py）
   - 直接 import，静态阅读就能知道调用链
   - 好处：grep 一次定位，不需要追踪 runtime 解析

4. Agent skills（任务专用 agent 技能）
   - 可复用的 procedural playbooks（add-new-model, capture-nsys-profile, validate-correctness 等）
   - 每个 skill 有：明确范围、前置条件、可验证的 PASS/FAIL 结果
   - 好处：agent 不需要每次重新推导流程，直接加载验证过的 playbook
```

**对比表格（论文 Table 1）：**

| 框架 | Compact | Python-native | No implicit indirection | Agent skills |
|------|---------|---------------|------------------------|--------------|
| Megatron | ✗ 149K | ✗ | ✗ | △ |
| DeepSpeed | ✗ 167K | ✗ | ✗ | ✗ |
| TorchTitan | △ 38K | ✓ | △ | △ |
| **PithTrain** | ✓ ~11K | ✓ | ✓ | ✓ |

**深挖点：**
- Q：Implicit indirection 具体指什么？**A**：指通过 runtime spec 或 plugin registry 延迟决定调用哪个实现。例如 Megatron 的 `TransformerLayer` 从配置文件中读取 spec，在运行时决定实例化哪些 submodule。好处是代码复用，坏处是"看代码看不到实际跑的是什么"。
- Q：Agent skills 和普通的 README 有什么区别？**A**：Three properties——specific scope（明确触发条件）、explicit prerequisites（前置条件检查）、verifiable success（可验证的 PASS/FAIL）。普通的 README 描述不完整，没有前置条件检查，结果也不能自动验证。
- Q：Code compactness 的边界在哪里？**A**：论文提到 "compactness as a constraint on growth"——PithTrain 可以增长，但新增必须遵守四条原则。不追求 broad model coverage（如 LLaMA、BERT 等），只支持当前需要的 MoE 模型。

---

### 11.2 Agent-Task Efficiency (ATE) 与 ATE-Bench

**必考题 21：什么是 Agent-Task Efficiency？ATE-Bench 怎么设计？**

```
Agent-Task Efficiency (ATE):
  定义：用 coding agent 来理解、操作和扩展训练框架的成本
  度量维度（5 个，无单标度，全部独立报告）：
    1. Session Duration（会话总时长）
    2. Active GPU Time（GPU 实际工作时间）
    3. Agent Turns（agent 和环境的交互轮数）
    4. Per-Turn Context（每轮输入 token 数）
    5. Output Tokens（输出 token 数）

  ← 这是论文提出的新指标！传统 benchmark 只比 throughput

ATE-Bench 设计思路（论文 §4）：
  - 固定 agent（Claude Code Opus 4.7），固定任务，变化框架
  - 这和传统 benchmark（SWE-bench, HumanEval）相反：传统是固定代码库，变化 agent
  
  三类任务：
    1. Q&A（12 题）：只读，回答关于框架的问题（如 "device mesh 怎么构建？"）
    2. Operate and Profile（4 题）：运行框架，instrument，profile
    3. New Feature（4 题）：集成新架构（Diff Attention, DynMoE, MoBA, MoE++）
```

**关键评估结果（论文 §5）：**

| 指标 | Megatron | TorchTitan | PithTrain | PithTrain 提升 |
|------|----------|------------|-----------|---------------|
| Agent Turns (Q&A) | 33-54 | 4-28 | 6-21 | **↓ 62%** vs Megatron |
| Active GPU Time (New Feature) | 33.7-58.7 min | 40.3-94.4 min | 27.6-41.9 min | **↓ 44%** vs Megatron |
| Session Duration (New Feature) | 47-88 min | 50-140 min | 38-63 min | **↓ 35%** vs Megatron |

**训练吞吐对比（论文 Table 2）：**

| Model | Hardware | Megatron | TorchTitan | **PithTrain** |
|-------|----------|----------|------------|---------------|
| GPT-OSS-20B | 8×B200 | 129.5K tok/s | --- | **140.9K** |
| Qwen3-30B-A3B | 8×B200 | 106.2K | OOM | **134.5K** |
| Qwen3-30B-A3B | 8×H100 | 126.7K | 90.5K | **124.9K** |
| DeepSeek-V2-Lite | 8×H100 | 107.3K | 74.1K | **114.6K** |

**深挖点：**
- Q：ATE-Bench 为什么不包括 "cross-model propagation" 任务？**A**：论文 §4 提到，这种任务中 Megatron 的 implicit indirection 可能反而降低 agent 成本（改一处，多个模型都生效）。PithTrain 的 flat structure 在这种场景下反而需要更多编辑。这是未来工作。
- Q：Skills ablation 的结果说明了什么？**A**：论文 §5.3，`validate-correctness` skill 让 agent turns 从 114 降到 34（70%↓），`capture-nsys-profile` 从 75 降到 36（52%↓）。Skills 不减少 GPU 时间（因为任务固定的 GPU work），但大幅减少 agent 的推理 overhead。
- Q：TorchTitan 为什么在某些任务上更差？**A**：论文 §5.2 case study 指出 TorchTitan 的 agent 经常 OOM（内存压力调试），导致反复的 debug-edit 循环。PithTrain 的内存效率更高（FP8 + DualPipeV），agent 不需要处理 OOM。

---

### 11.3 论文核心贡献总结

**必考题 22：PithTrain 论文的三个贡献是什么？**

```
1. PithTrain 系统本身
   - ~11K 行 Python MoE 训练框架
   - DualPipeV + 5 阶段 overlap + FP8 + 4D 并行
   - 吞吐量匹配 Megatron（4/5 配置超越，1/5 差 1.4%）

2. 四个 Agent-Native 设计原则
   - 首次系统性地提出 "为 AI agent 设计的训练框架" 应该是什么样子
   - 对比了 Megatron/DeepSpeed/TorchTitan 在各原则上的符合度

3. ATE 指标 + ATE-Bench + 实证研究
   - 定义了 Agent-Task Efficiency 为新的框架评估维度
   - 构建了 20 任务的 benchmark suite
   - 实证：PithTrain 在 agent 效率上显著优于生产框架
```

---

## 12. Part XI：后训练方向延伸

### 12.1 SFT (Supervised Fine-tuning)

**常见问题：SFT 和 Pretraining 的区别？在 PithTrain 中如何实现 SFT？**

```
SFT vs Pretraining：
  1. 数据：Pretraining 用大规模无标注语料，SFT 用标注的 instruction-response 对
  2. 目标：Pretraining 是 next-token prediction，SFT 同上但数据质量更高
  3. 长度：SFT 通常 sequence 更长（完整的对话/instruction + response）
  4. LR：SFT 用更小的 lr，通常只训练 few steps

在 PithTrain 中实现 SFT：
  - 复用 PretrainLMCfg，只需换 dataset 和 lr
  - 数据集：`ConcatDataset` + `MemmapDataset`，可以加载任意 tokenized 数据
  - SFT 通常需要 gradient checkpointing → PithTrain 目前没有显式的 activation checkpointing
    但可以通过减小 micro_batch_size 和增大 accumulate_steps 间接控制内存
```

### 12.2 RLHF / GRPO

**常见问题：PPO 和 GRPO 的区别？MoE 模型做 RLHF 有什么特殊考虑？**

```
PPO (Proximal Policy Optimization):
  - 需要 4 个模型：policy, reference, reward, critic
  - clip 目标: min(r·A, clip(r, 1-ε, 1+ε)·A)
  - 计算量：每个 rollout 需要 forward policy + reference + reward，backward policy

GRPO (Group Relative Policy Optimization):
  - 去掉 critic，用一组（同一 prompt 的多个 response）的相对回报估计 advantage
  - A = (R - mean(R_group)) / std(R_group)
  - 计算量：只需要 policy + reward，更轻量

MoE 模型 RLHF 的特殊考虑：
  1. Expert collapse：RL 的 reward signal 可能加剧负载不均衡 → 需要保持 lb_loss
  2. 推理 vs 训练差异：SFT 用 greedy/beam search，RL 用 sampling
     MoE 的 top-k 路由在 sampling 下和 training 一致（因为路由是确定性的）
  3. PP 和 RL 的兼容性：PP 的 pipeline 调度在 RL 的 per-sample reward 场景下
     需要确保 loss 计算在每个 micro-batch 上独立
```

### 12.3 Load Balancing 在后训练中的重要性

**常见问题：为什么 MoE 后训练中 load balance 特别重要？**

```
Pretraining：
  - 数据分布广泛，router 自然会学到相对均衡的分配
  - 但如果某些 expert "specialize" 到稀有 token，其余 expert 负载不均

SFT：
  - 数据量小，多样性差，更容易出现 expert collapse
  - 某些 expert 完全闲置，浪费参数 → 需要更强的 lb_loss

RLHF：
  - Reward hacking 可能导致模型集中输出某些 token 类型
  - 某些 expert 被过度使用 → 通信瓶颈（EP all-to-all 的不均衡）
  - 建议：global-batch lb_loss + 较大的 lb_coef（如 1e-2 ~ 1e-1）

PithTrain 的 lb_loss 选择：
  - pretraining: "global-batch" 或 "sequence"
  - SFT/RLHF: "global-batch"（跨 DP×EP 同步，确保全局均衡）
  - benchmark: force_balance（round-robin router）
```

---

## 13. Part XII：系统级深挖题（区分度最高的题）

### 12.1 Pipeline Bubble 的定量分析

**题目：对于一个 PP=4, num_chunks=8 的 DualPipeV，bubble 占多少比例？和标准 1F1B 比如何？**

**分析：**
```
标准 1F1B:
  总步数 = 2 × PP + num_chunks - 1 = 2×4 + 8 - 1 = 15
  Bubble（无计算步数）= PP - 1 = 3
  Bubble 占比 = 3/15 = 20%

DualPipeV:
  总步数 ≈ 2 × PP + num_chunks = 2×4 + 8 = 16（更紧密的调度）
  但每个 step 内有 compute + comm 重叠，实际 bubble 更小
  理论 bubble 占比 ≈ (PP-1)/(num_chunks + PP)  →  3/12 = 25%... 不对

  更准确的公式：
  DualPipeV 的 idle time ≈ (PP - 1) × (t_f + t_b) - (PP - 1) × overlap_time
  当 comm 完全隐藏在 compute 后面时，bubble 趋近于 0

  实际上 PithTrain 的 bubble 时间 ≈ PP - 1 个 micro-batch 的 compute time
  当 num_chunks >> PP 时，bubble 占比 ≈ (PP-1) / num_chunks
```

### 12.2 All-to-All 通信瓶颈分析

**题目：MoE 训练中 EP all-to-all 的通信量是多少？怎么优化？**

```
通信量计算：
  每个 token 被 dispatch 到 k 个 experts（k = num_experts_per_tok）
  k = 1（Qwen3），2（DeepSeek），6（GPT-OSS 等）

  不 dedup：每个 rank 发送 m × k 个 tokens（m = local tokens）
  Dedup 后：发送 dedup_m 个 tokens（每个 rank 只发一次独立 token）

  通信量 = dedup_tokens_per_gpu × hidden_dim × 2 bytes (BF16)
  对于 Qwen3-30B-A3B (k=1, ep=8, hidden=2048)：
    每 rank 约 (batch_size × seq_len / 8) × 2048 × 2 bytes

优化手段：
  1. Token Dedup（PithTrain 的核心优化）→ 减少约 30-50% 通信量
  2. Communication 放在独立 stream → 和 compute 重叠
  3. NVLink domain：CP 和 EP 放在 mesh 最内层 → 通信在 NVLink 内
  4. Pinned buffer + async H2D：避免同步 CUDA 拷贝

瓶颈判断：
  如果 all-to-all 时间 > compute 时间 → bubble 增加
  解决：增大 num_chunks（更多 micro-batches 可以重叠 comm）
  或者：增大 EP（减少 per-rank token 数，但增加 all-to-all 参与方）
```

### 12.3 FSDP + Pipeline 的梯度同步

**题目：PithTrain 中 FSDP 和 DualPipeV 的梯度同步如何正确工作？**

```
核心挑战：
  正常训练：module.backward() → autograd → FSDP hooks 自动触发 gradient sync
  Pipeline：  手动 backward (decoder_layer_backward) → 不走 autograd → FSDP hooks 不触发

PithTrain 的解决方案（dualpipev.py:run_post_backward）：
  1. step() 开始时：设置 set_is_last_backward(False) 和 suppress callback
     避免每次 run_backward 都触发 post_backward（~150-250μs overhead）
  2. Pipeline 循环结束后：手动调用 run_post_backward()
     - set_is_last_backward(True)
     - set_reshard_after_backward(True)
     - 手动调用 _root_post_backward_final_callback()
  3. 梯度规约（all-reduce）在 FSDP state 内部完成

为什么要手动处理 wgrad 的 dtype？
  FSDP 的 reduce_dtype = float32（mixed precision policy）
  Muon 的 momentum buffer 也是 float32
  但 wgrad 延迟（WeightGradStore）期间，可能和其他 fp16/bf16 的梯度混在一起
  → accumulate_unsharded_grad_if_needed() + to_accumulated_grad_if_needed()
     确保所有 wgrad 在 reduce 前是统一的 dtype
```

---

## 14. Part XIII：实现题（手写代码/伪代码）

### 13.1 手写 Top-k 路由

**题目：实现一个简单的 MoE top-k 路由（单机，无 EP）**

```python
import torch
import torch.nn.functional as F

def moe_topk_routing(hidden_states, gate_weight, num_experts, k):
    """
    hidden_states: [N, D]  (N = batch * seq_len)
    gate_weight:   [num_experts, D]
    返回: topk_idx [N, k], topk_weight [N, k]
    """
    logits = F.linear(hidden_states, gate_weight)  # [N, num_experts]
    scores = logits.softmax(dim=-1, dtype=torch.float32)
    topk_weight, topk_idx = torch.topk(scores, k=k, dim=-1, sorted=False)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)  # norm_topk_prob
    return topk_idx, topk_weight

def compute_load_balance_loss(scores, topk_idx, num_experts, k, coef=0.01):
    """Micro-batch load balance loss."""
    N = scores.shape[0]
    flat_idx = topk_idx.view(-1)
    tokens_per_expert = torch.bincount(flat_idx, minlength=num_experts).float()
    f = tokens_per_expert / (N * k)          # expert selection fraction
    p = scores.mean(dim=0)                    # average router probability
    return coef * num_experts * torch.dot(f, p)
```

### 13.2 手写 Pipeline Bubble 模拟

**题目：用 Python 模拟 PP=4, num_chunks=6 的 1F1B 调度，计算 bubble 比例**

```python
def simulate_1f1b(pp_size, num_chunks):
    """
    模拟 1F1B pipeline 调度。
    返回: (total_steps, bubble_steps, bubble_ratio)
    """
    # 每个 rank 的状态
    class Rank:
        def __init__(self, idx):
            self.idx = idx
            self.f_chunks = []   # 待 forward 的 chunks
            self.b_chunks = []   # 待 backward 的 chunks
    
    ranks = [Rank(i) for i in range(pp_size)]
    
    # 初始化：每个 chunk 从 rank 0 开始 forward
    for c in range(num_chunks):
        ranks[0].f_chunks.append(c)
    
    schedule = []  # (step, [(rank, action, chunk), ...])
    
    # 简化模拟：每个 step 每个 rank 执行一个 action
    step = 0
    while any(r.f_chunks or r.b_chunks for r in ranks):
        actions = []
        for r in ranks:
            if r.f_chunks:
                chunk = r.f_chunks.pop(0)
                actions.append((r.idx, 'F', chunk))
                # 传给下一个 rank
                if r.idx + 1 < pp_size:
                    ranks[r.idx + 1].f_chunks.append(chunk)
            elif r.b_chunks:
                chunk = r.b_chunks.pop(0)
                actions.append((r.idx, 'B', chunk))
            else:
                actions.append((r.idx, 'idle', None))
        schedule.append((step, actions))
        step += 1
    
    # 统计 bubble（idle steps）
    # ...
    return schedule
```

### 13.3 手写 Simplified Ring Attention

**题目：用伪代码描述 zigzag ring attention 的核心逻辑**

```
function zigzag_ring_attention(Q, K, V, cp_size, cp_rank):
    block = seq_len // (2 * cp_size)
    front_block = Q[:, :block, :]         # 本地 front chunk
    back_block  = Q[:, block:, :]         # 本地 back chunk
    
    # 构建本地 K/V（来自两个 chunks）
    local_k = K[front_start:front_start+block] + K[back_start:back_start+block]
    local_v = V[front_start:front_start+block] + V[back_start:back_start+block]
    
    output = zeros_like(Q)
    
    for step in range(cp_size):
        if step == 0:
            # 用自己的 K/V
            k_current, v_current = local_k, local_v
            mask = causal_mask(2*block)          # 标准 causal
            output += flash_attention(Q, k_current, v_current, mask)
        elif step <= cp_rank:
            # K/V 来自低 rank，只有 front block 在 causal mask 内
            k_recv, v_recv = recv_from(cp_rank - step)
            mask = causal_mask(block)             # 只看 front block
            output[:, :block] += flash_attention(
                Q[:, :block], k_recv[:, :block], v_recv[:, :block], mask
            )
        else:
            # K/V 来自高 rank，只有 back block attend
            k_recv, v_recv = recv_from(cp_size - (step - cp_rank))
            output[:, block:] += flash_attention(
                Q[:, block:], k_recv, v_recv, non_causal_mask
            )
        
        # 异步发送当前 K/V 到下一个 rank
        async_send(local_k, local_v, dst=(cp_rank + 1) % cp_size)
        # 异步接收下一步的 K/V
        k_recv_next, v_recv_next = async_recv_from(src=(cp_rank - 1) % cp_size)
        wait()  # 确保通信完成再进入下一步
```

---

## 15. 面试策略总结

### 自我介绍时的"钩子"

在自我介绍中埋 2-3 个技术钩子，引导面试官问你想好的题目：

| 你想引导的题目 | 在自我介绍中的埋点 |
|---------------|-------------------|
| MoE / EP dispatch | "研究了 MoE 的 expert parallelism，包括 token 去重和自定义 Triton kernel" |
| DualPipeV | "深入理解了 DualPipeV 的 5 阶段分解和 V 形双向调度" |
| FP8 | "分析了 FP8 block scaling 在 Hopper/Blackwell 上的差异" |
| Ring Attention | "研究了 zigzag context parallelism 和 KV 环形传递" |
| Muon | "对比了 Muon 和 AdamW 在不同参数类型上的效果" |

### 高频追问链

```
"你用了 MoE"
  → "MoE 的 router 怎么训练的？" → load balance loss
  → "EP 的 all-to-all 通信量有多大？" → token dedup, communication volume
  → "如果不做 dedup 会怎样？" → 通信翻倍，带宽瓶颈

"你提到了 pipeline parallelism"
  → "bubble 怎么解决？" → DualPipeV V-shape + 5-stage overlap
  → "PP 和 DP 怎么配合？" → FSDP sharding, gradient sync
  → "pipeline 并行下 checkpoint 怎么做？" → canonical vs localized format

"你用了 FP8"
  → "FP8 和 BF16 的精度差异？" → block scaling, amax, dynamic range
  → "Blackwell 和 Hopper 的 FP8 有什么不同？" → E8M0 vs FP32 scales
  → "FP8 训练会 loss 吗？怎么处理？" → loss scaling 的替代方案

"你做了 Agent 方向"
  → "agent 怎么调用工具？" → function calling, tool use
  → "agent 的 planning 怎么做？" → chain-of-thought, react, reflexion
  → "agent 的安全性问题？" → prompt injection, sandboxing
```

### 区分度最高的 3 道题

如果面试官追问到以下题目，说明你在 deep dive 模式，要答好：

1. **"DualPipeV 的 overlapped_forward_backward 中，compute stream 和 comm stream 的同步机制是什么？"**
   - 答案：`ExecutionCtx.fwd_event` / `bwd_event`（CUDA events） + `fwd_comm_work` / `bwd_comm_work`（Work handles）
   - Stage 2 (Dispatch) forward 在 comm stream 上，但需要在 Stage 1 compute 完成后启动
   - 做法：Stage 1 compute 在 comp_stream 上 record `fwd_event`，comm_stream wait 该 event

2. **"EP dispatch 的 token dedup 算法中，counting sort 为什么比 argsort 好？复杂度差多少？"**
   - 答案：Counting sort O(m·k)，argsort O(m·k·log(m·k))，其中 m = num_tokens, k = top_k
   - 对于典型值 m=4096, k=1-6，差距是 10-100x
   - Counting sort 需要 prefix sum 和 O(m·k) 的输出空间，但 m·k 很小（几 KB）

3. **"MoE 的 load balance loss 通过 straight-through estimator 注入梯度，这样做和直接加 loss 有什么区别？"**
   - 答案：MoELoadBalanceLossInjector 的 forward 返回 topk_weight 不变，backward 注入 `∂lb_loss/∂topk_weight = 1`
   - 区别：如果直接加 `total_loss = task_loss + lb_loss`，topk_weight 会同时受到 task_loss 和 lb_loss 的梯度影响，可能导致路由退化
   - Straight-through 让 lb_loss 的梯度直接调制 topk_weight 而不经过 softmax 的 chain rule，相当于在 routing 决策层直接施加平衡压力

### 准备时间分配建议

| 主题 | 建议准备时间 | 重要性 | 备注 |
|------|------------|--------|------|
| Transformer / Attention / RoPE / Norm | 1-2 天 | ★★★★★ 必考 | 基础中的基础 |
| MoE 路由 + Load Balance | 2-3 天 | ★★★★★ 重点 | 面试官会深挖 dedup + lb loss |
| DualPipeV 调度算法 | 2-3 天 | ★★★★★ 框架核心 | 8 步调度 + F/B overlap |
| 4D 并行（PP/DP/CP/EP） | 1-2 天 | ★★★★☆ | Zigzag CP 容易考 |
| FP8 训练 | 1 天 | ★★★★☆ | Block scaling + E8M0 |
| Ring Attention + Zigzag | 1 天 | ★★★★☆ | 算法逻辑要能手绘 |
| Agent-Native + ATE-Bench | 1 天 | ★★★★☆ | 论文核心，区分度最高 |
| Muon 优化器 | 0.5 天 | ★★★☆☆ | Newton-Schulz 原理 |
| Checkpoint resharding | 0.5 天 | ★★★☆☆ | Canonical vs Localized |
| Triton kernel 基础 | 1 天 | ★★★☆☆ | tl.histogram, counting sort |
| MLA | 0.5 天 | ★★★☆☆ | DeepSeek 特有 |
| 后训练 (SFT/RLHF) | 1 天 | ★★★☆☆ | DynMoE, GRPO |

### 论文面试题补充（NeurIPS 2026 论文特有）

面试官如果看过论文，可能会问以下问题：

1. **"ATE-Bench 的三类任务分别考察什么？你的项目在哪个维度最有优势？"**
   - Q&A 考察代码可理解性（compact + no indirection → 少 turns，少 context）
   - Operate/Profile 考察可操作性（Python-native → 可读 error，快速 debug）
   - New Feature 考察可扩展性（skills + flat structure → 快速集成）
   - PithTrain 在 New Feature 上优势最大（Active GPU Time ↓64%），因为 agent 更快迭代到可运行状态

2. **"你怎么看待 'agent-native' 作为框架设计目标？它会成为未来的趋势吗？"**
   - 这是一个开放性问题，可以从几个角度回答：
     - AI coding agent 的能力在快速增长（Claude Code, Cursor 等）
     - 框架的用户将越来越多地通过 agent 交互（自然语言 → 代码修改）
     - 但 production 需求（性能、覆盖度）不会消失 → 可能需要 "dual-mode" 框架
     - PithTrain 的取舍（compact over coverage）适合研究/原型，大规模生产可能仍需要 Megatron

3. **"ATE-Bench 的 Q12（Distributed Checkpoint Serialization）的具体答案是什么？"**
   - 论文附录 Q12 的答案在 PithTrain 中：
     - 使用 PyTorch DCP（`torch.distributed.checkpoint`）
     - 分布式保存：每个 rank 写自己的 shard（`dcp.save`）
     - 序列化逻辑在 `pithtrain/tasks/pretrain_lm.py:save_checkpoint()` (L261-299)
     - 使用 `StateDictOptions(cpu_offload=True)` 避免 GPU all-gather
     - 每个 rank 只写自己拥有的 expert keys（canonical format）

---

## 附录：关键源码路径速查

| 概念 | 关键文件 | 行号 |
|------|---------|------|
| Model Protocol | `pithtrain/models/interface.py` | 全部 |
| Qwen3 MoE Layer | `pithtrain/models/qwen3_moe.py` | L421-646 (DecoderLayer) |
| MoE Gate + LB Loss | `pithtrain/models/qwen3_moe.py:Qwen3MoeGate` | L183-264 |
| DeepSeek MLA | `pithtrain/models/deepseek_v2_lite.py` | L1-728 |
| DualPipeV Scheduler | `pithtrain/dualpipe/dualpipev.py:DualPipeV.step` | L418-589 |
| F/B Overlap Loop | `pithtrain/dualpipe/overlap.py:overlapped_forward_backward` | L67-390 |
| 5 Stage Execution | `pithtrain/dualpipe/execution.py` | L75-606 |
| 4D Device Mesh | `pithtrain/modules/distributed.py:setup_device_mesh` | L165-191 |
| FSDP Apply | `pithtrain/modules/training.py:apply_fsdp` | L350-414 |
| Muon Optimizer | `pithtrain/modules/optimizer.py` | 全部 |
| WSD Scheduler | `pithtrain/modules/training.py:make_wsd_scheduler` | L87-126 |
| EP Dispatch Kernel | `pithtrain/operators/ep_dispatch.py` | L36-419 |
| FP8 Quantize | `pithtrain/operators/deepgemm_fp8_quantize.py` | L39-81 |
| Ring Attention | `pithtrain/operators/ring_attention.py` | 全部 |
| LB Loss (3 types) | `pithtrain/modules/load_balance.py` | L38-198 |
| Checkpoint Resharding | `pithtrain/modules/checkpoint.py` | 全部 |
| Training Loop | `pithtrain/tasks/pretrain_lm.py:train_step` | L337-495 |
| Batch Loading (Zigzag) | `pithtrain/tasks/pretrain_lm.py:get_global_batch` | L70-126 |
