# PithTrain 技术面试五类能力应对手册

> 基于 PithTrain 项目源码与 NeurIPS 2026 论文的深度分析，针对技术面试中五类核心能力考察，准备应对策略、回答模板与可深挖细节。
> 与《PithTrain_面试深挖准备.md》互补：后者侧重"知识点是什么"，本手册侧重"面试官怎么考、你怎么答"。

---

## 目录

1. [能力一：底层原理理解能力](#1-能力一底层原理理解能力)
2. [能力二：实验和方案验证能力](#2-能力二实验和方案验证能力)
3. [能力三：问题定位能力](#3-能力三问题定位能力)
4. [能力四：工程落地能力](#4-能力四工程落地能力)
5. [能力五：业务与实际场景理解](#5-能力五业务与实际场景理解)
6. [综合模拟：五类能力串联回答](#6-综合模拟五类能力串联回答)

---

## 1. 能力一：底层原理理解能力

### 1.1 面试官考察方式

这类问题面试官不满足于"是什么"，会连续追问：

```
"为什么需要这个设计？"
  → "它解决什么具体问题？"
    → "如果不用会怎样？"
      → "它有什么局限性？"
        → "如果有局限性，你会怎么改进？"
```

**关键**：面试官想听的是 **problem → solution → trade-off → improvement** 的完整链条，而不是教科书定义。

---

### 1.2 应对策略：三层回答法

```
第一层（概念）：用一句话说清它是什么
第二层（动机+trade-off）：解决什么问题，为什么选这个方案而不是其他，代价是什么
第三层（局限性+改进）：哪里不够好，你会怎么改
```

---

### 1.3 PithTrain 高频原理题与回答模板

#### 题目 1：DualPipeV 的 V 形布局为什么比标准 Pipeline Parallelism 好？

**第一层（概念）：**
标准 PP 把模型切成连续的 `pp_size` 块，rank r 只拿第 r 块。DualPipeV 切成 `2*pp_size` 块，rank r 拿第 r 块和第 `2*pp_size-1-r` 块，形成 V 形。

**第二层（动机+trade-off）：**
```
问题：标准 PP 的 bubble 时间占比 = (PP-1) / (num_chunks + PP - 1)
      当 PP 大、num_chunks 小时，bubble 可以占到 30-50%

为什么不用更大的 num_chunks？
  - 每个 micro-batch 需要独立的前向/后向激活内存
  - num_chunks 翻倍 → 激活内存翻倍 → 可能 OOM

DualPipeV 的解法：
  - V 形让每个 rank 既有"前半段"的 forward chunk，也有"后半段"的 backward chunk
  - 这样 rank 不会出现"只有 forward 没有 backward"的 idle 状态
  - 配合 5 阶段分解 + comm stream 重叠，bubble 时间 ≈ PP-1 个 micro-batch 的 compute time
  - 当 num_chunks >> PP 时，bubble 占比趋近于 0

Trade-off：
  - V 形增加了通信复杂度：rank r 需要和 rank r-1（forward 输入）以及 rank r+1（backward 输入）通信
  - 5 阶段分解增加了实现复杂度：每个 layer 的 forward/backward 被拆成 5 个 stage
  - 代码量增加：但 PithTrain 用 pre-allocated IntermediateTensors 和 in-place 修改控制住了内存 overhead
```

**第三层（局限性+改进）：**
```
局限性：
  1. 需要 num_chunks >= 2*PP，小 batch 场景下内存 overhead 大
  2. V 形要求模型层数能被 2*PP 整除（edge stages 的 layer 数不同）
  3. EP all-to-all 的通信时间如果超过 compute 时间，5 阶段 overlap 的收益会打折扣

改进方法：
  1. 对于小 batch，可以降低 PP 增大 DP，或者用 activation checkpointing 换 PP
  2. layer_partition.py 已经处理了非整除情况（edge stages 少放层）
  3. 如果 compute-comm overlap 不足，可以：
     a) 增大 num_chunks（更多 micro-batch 可以重叠）
     b) 用更大的 EP（减少 per-rank token 数，缩短 all-to-all 时间）
     c) 或者把 Stage 2/4 的通信进一步拆细（但当前 5 阶段已经是最细粒度之一）
```

**对应源码：**
- `pithtrain/dualpipe/dualpipev.py:__init__` L57-88 — V 形模块初始化
- `pithtrain/dualpipe/layer_partition.py` — edge stage 层数分配
- `pithtrain/dualpipe/dualpipev.py:step()` L478-544 — 8 步调度

---

#### 题目 2：MoE 的 Token Dedup 到底减少了多少通信？代价是什么？

**第一层（概念）：**
EP dispatch 时，一个 token 如果被路由到同一个 EP rank 的多个 experts，不 dedup 需要发送多次。Dedup 后只发送一次独立 token，通过 `expand_idx` 在接收端恢复完整映射。

**第二层（动机+trade-off）：**
```
通信量对比（以 Qwen3-30B-A3B, EP=8, k=1 为例）：
  - 不 dedup：每个 token 发 1 次，但 m 个 token 中可能有重复的 EP rank
  - Dedup：只发独立 EP rank 的 token，发送量 ≈ m × (unique_gpu_fraction)
  - 对于 k=1，每个 token 只去 1 个 expert，dedup 收益有限
  - 对于 k>1（如 DeepSeek-V2 k=6），一个 token 可能去 6 个 experts 分布在多个 ranks
    dedup 可以把发送量从 m×k 降到 m×unique_ranks

代价：
  1. 内存：需要额外的 dedup_local_pos [m×ep_size] 存储去重位置
  2. 计算：3 个 Triton kernel 的开销（bincount, prefix sum, counting sort）
  3. 复杂度：expand_idx 的 all-to-all + adjust_expand_idx 增加了同步点

为什么值得？
  - 对于典型 micro-batch（m=4096, ep=8, k=6），不 dedup 发送 24K tokens
  - Dedup 后约发送 8K tokens（假设均匀分布）→ 减少 67% 通信
  - 通信减少直接转化为 bubble 时间减少（comm 时间 < compute 时间）
```

**第三层（局限性+改进）：**
```
局限性：
  1. k=1 时 dedup 几乎无收益（每个 token 只去 1 个 expert，必然只去 1 个 rank）
  2. Dedup 算法本身有 overhead：3 个 kernel launch + 2 次 all-to-all（meta + expand_idx）
  3. 当 EP 很大（如 EP=32）时，tl.histogram 的 EP_PADDED=next_power_of_2(33)=64，
     内存 overhead 变大

改进方法：
  1. 动态判断：如果 estimated_dedup_ratio < threshold，跳过 dedup 直接发送
  2. 把 meta all-to-all 和 expand_idx all-to-all fuse 成一个（如果有硬件支持）
  3. 对于 k=1 的模型（如 Qwen3），简化 dedup 路径：不需要 dedup，直接复制 tokens
```

**对应源码：**
- `pithtrain/operators/ep_dispatch.py:moe_ep_prepare_dispatch` L578-681 — 入口
- `pithtrain/operators/ep_dispatch.py:fused_dedup_prepare_dispatch` L270-419 — 3 个 kernel
- `pithtrain/operators/ep_dispatch.py:_dedup_bincount_kernel` L36-103 — kernel 1

---

#### 题目 3：FP8 训练为什么用 128-element block scaling？为什么 power-of-2 scale？

**第一层（概念）：**
FP8 (E4M3) 的动态范围远小于 BF16。Block scaling 把 tensor 分成 128-element 的块，每块独立计算 amax，得到 scale factor，然后量化 `x_fp8 = x / scale`。

**第二层（动机+trade-off）：**
```
为什么不用 per-tensor scaling？
  - 一个 tensor 中不同元素的 magnitude 差异可能很大
  - 例如 attention output：少数 token 的 QK 值很大，其余很小
  - Per-tensor scale 由最大值决定 → 小值被压缩到接近 0 → 精度损失大

为什么用 128-element block？
  - DeepGEMM 的 GEMM kernel 要求 K 维 block size = 128（硬件约束）
  - Block 太小（如 32）：scale 数量太多，overhead 大，精度提升有限
  - Block 太大（如 512）：同 block 内数值差异大，精度损失大
  - 128 是 DeepGEMM 支持和精度平衡的最优解

为什么 power-of-2 scale？
  - 量化：x_fp8 = x / 2^e，反量化：x_bf16 = x_fp8 * 2^e
  - 除以/乘以 2^e 在 IEEE-754 中是精确的（ exponent shift，无 mantissa 舍入）
  - 只有量化前的除法有舍入，且误差被 FP8 的 3-bit mantissa 自然 bound
  - Hopper (SM90)：用 FP32 bit manipulation（clear mantissa, increment exponent if not exact）
  - Blackwell (SM100+)：用 PTX `cvt.rp.satfinite.ue8m0x2.f32` 直接得到 E8M0 exponent

Trade-off：
  - Block scaling 的精度损失 < per-tensor，但 > per-channel（如 per-row）
  - 128-element 是硬件约束下的最优，不是理论上最优
  - Power-of-2 scale 是精度和效率的折中（允许 exact dequant，但允许一定 quant error）
```

**第三层（局限性+改进）：**
```
局限性：
  1. 128-element block 对某些形状不友好（如 attention output [B, S, H]，H 维可能不是 128 的倍数）
  2. Power-of-2 scale 的 granularity 太粗：amax 可能在 2^e 和 2^(e+1) 之间任意位置
     导致 scale 偏大（量化后值偏小）或偏小（溢出）
  3. E4M3 的 max=448，对于 outlier 很大的模型（如 MoE gate output），仍然可能溢出

改进方法：
  1. 动态 block size：根据 tensor shape 自动选择 64/128/256
  2. 用 E8M0 scale 的 finer granularity：Blackwell 的 MXFP8 支持 per-block 的 finer scaling
  3. 对于已知 outlier 的维度（如 MoE gate），用 per-expert scaling 替代 per-block
```

**对应源码：**
- `pithtrain/operators/deepgemm_fp8_quantize.py:_compute_fp8_scale` L39-81
- `pithtrain/layers/deepgemm_fp8_linear.py` — FP8Linear 和 FP8GroupLinear

---

#### 题目 4：Zigzag Context Parallelism 为什么能平衡 causal attention 的计算量？

**第一层（概念）：**
标准 CP 把 sequence 切成 `cp_size` 块，rank r 拿第 r 块。Zigzag CP 切成 `2*cp_size` 块，rank r 拿第 r 块和第 `2*cp_size-1-r` 块（front + back）。

**第二层（动机+trade-off）：**
```
问题：标准 CP 的 causal attention 计算量不均衡
  cp_size=4 的例子：
    rank 0 拿 chunk 0：Q  attend 到 chunk 0（本地），K/V 来自 rank 3
      → Q 的 causal mask 只覆盖本地 S/4，计算量 ≈ (S/4)^2
    rank 3 拿 chunk 3：Q  attend 到 chunk 0,1,2,3
      → Q 的 causal mask 覆盖整个 S，计算量 ≈ S^2
    → rank 3 的计算量是 rank 0 的 16 倍！

Zigzag 的解法：
  chunk:  0  1  2  3  4  5  6  7
  rank:   0  1  2  3  3  2  1  0
  
  rank 0 拿 chunk 0（轻）+ chunk 7（重）
    - chunk 0 的 Q attend 到 chunk 0（轻）
    - chunk 7 的 Q attend 到 chunk 0-7（重）
    - 总计算量 ≈ 平衡
  
  rank 3 拿 chunk 3（重）+ chunk 4（轻）
    - chunk 3 的 Q attend 到 chunk 0-3（重）
    - chunk 4 的 Q attend 到 chunk 0-4（轻）
    - 总计算量 ≈ 平衡

Trade-off：
  - 数据加载复杂：需要从两个不连续的 chunk 读取（front_offset 和 back_offset）
  - RoPE 位置编码复杂：需要构建 zigzag position_ids
  - Ring attention 的 step 逻辑复杂：需要分三种情况（step==0, kv<r, kv>r）
```

**第三层（局限性+改进）：**
```
局限性：
  1. sequence_length 必须能被 2*cp_size 整除（PithTrain 有 assert 检查）
  2. 当 cp_size=1 时，zigzag 退化为标准 CP（无收益）
  3. KV 环形传递的复杂度从 O(cp_size) 升到 O(2*cp_size)（每个 rank 持有两个 chunks）

改进方法：
  1. 对于不能被整除的 sequence_length，用 padding + mask 处理（但会浪费计算）
  2. 自适应 zigzag：根据每个 rank 的计算量动态调整 chunk 分配（但需要额外 profiling）
  3. 对于 decode 阶段（1 token），zigzag 不适用（只有一个 token），需要 fallback 到标准 CP
```

**对应源码：**
- `pithtrain/operators/ring_attention.py` L1-49 — Zigzag 设计说明
- `pithtrain/tasks/pretrain_lm.py:get_global_batch` L70-126 — zigzag 数据加载
- `pithtrain/models/qwen3_moe.py:Qwen3MoeModel.forward` L735-825 — zigzag position_ids

---

### 1.4 原理类问题的回答节奏控制

| 面试官追问 | 你应该回答的深度 | 时间建议 |
|-----------|---------------|---------|
| "是什么？" | 一句话定义 + 核心机制 | 30 秒 |
| "为什么需要？" | 问题背景 + 现有方案的不足 | 1 分钟 |
| "解决了什么问题？" | 具体场景 + 量化收益（如 bubble 从 30% 降到 5%） | 1-2 分钟 |
| "有什么代价？" | Trade-off 分析（内存、计算、复杂度） | 1 分钟 |
| "局限性？" | 2-3 个具体场景 + 原因 | 1-2 分钟 |
| "怎么改进？" | 1-2 个可行方向 + 预期效果 | 1-2 分钟 |

**禁忌**：不要一上来就讲第三层（改进）。如果面试官只是礼貌性问问，你讲了 5 分钟 limitations，会显得听不懂空气。观察面试官的反应，如果他在点头或者追问，再深入。

---

## 2. 能力二：实验和方案验证能力

### 2.1 面试官考察方式

```
"你怎么证明这个方案有效？"
  → "对照组是怎么设置的？"
    → "为什么选这个指标？"
      → "有没有排除其他变量？"
        → "实验结果和预期不一致你怎么解释？"
```

**关键**：面试官要看到你有 **科学实验思维**，不是"跑了个实验，loss 降了"就完事。

---

### 2.2 应对策略：实验设计四要素

```
1. 对照组（Baseline）：必须和 proposed method 在完全相同的条件下比较
2. 控制变量：一次只改一个变量，否则不知道哪个因素导致了结果变化
3. 指标选择：指标必须直接回答你要证明的问题
4. 重复性：多次实验取 median/mean，排除随机性
```

---

### 2.3 PithTrain 中的实验验证细节与回答模板

#### 模板 1：训练吞吐验证（论文 §5.1）

**面试官问："你怎么证明 DualPipeV 不比 Megatron 慢？"**

**回答框架：**

```
1. 对照组设置：
   - Baseline：Megatron（commit 3bec9aa），NVIDIA 官方最佳实践
   - Ours：PithTrain（commit 23db182）
   - 相同条件：相同模型（Qwen3-30B-A3B）、相同 PP/DP/CP/EP、相同 sequence length、
               相同 precision（BF16/FP8）、相同 dataset（DCLM）

2. 控制变量：
   - 只改框架，不改模型架构、优化器、超参
   - 硬件完全一致（8×H100 或 8×B200）
   - 每个配置跑 25 steps，取最后 10 steps 的 median step time
   - 排除 warmup 阶段（cudagraph capture、NCCL handshake、allocator priming）

3. 指标选择：
   - 主指标：tokens per second（总吞吐）
   - 为什么不选 MFU？因为 BF16 和 FP8 的 peak FLOPS 不同，MFU 不可比
   - 辅助指标：step time、peak GPU memory

4. 结果：
   - 4/5 配置 PithTrain 超越 Megatron，1/5 差 1.4%（在统计误差内）
   - Qwen3-30B-A3B 8×B200 FP8：PithTrain 134.5K vs Megatron 106.2K（+27%）
   - 证明 DualPipeV 的 compute-comm overlap + torch.compile 的收益是真实的

5. 排除混淆因素：
   - 用 public checkpoint 而非 random init：确保 MoE router 处于 load-balanced 稳态
     如果 router 不平衡，某些 rank 的 compute 会突然增加，干扰 throughput 测量
```

**可深挖的细节：**
- Q：为什么不用 random init？**A**：random init 的 MoE router 可能把所有 tokens 路由到少数 experts，导致 EP 负载不均衡，某些 rank 的计算量突增，吞吐量测量不准确。
- Q：为什么取最后 10 steps 的 median？**A**：前几步有 cudagraph warmup、NCCL 握手、内存分配器 priming。取 median 而不是 mean，是为了排除极端值（如 GC pause、OS 调度抖动）。

---

#### 模板 2：Operator Correctness 验证

**面试官问："你怎么保证自定义 Triton kernel 的输出是对的？"**

**回答框架：**

```
1. 验证方法：每个 operator 必须有一个 PyTorch reference implementation
   - 例如 EP dispatch：先用 PyTorch 实现一遍（scatter + argsort + nonzero + searchsorted）
   - 然后在 test 中对比 Triton kernel 输出和 PyTorch reference 输出

2. 对比指标：
   - 使用 normalized squared error（calc_diff）
   - 阈值：< 1e-3（对于 FP8 量化），< 1e-5（对于 EP dispatch 的索引）

3. 测试覆盖：
   - 不同配置：EP=1/2/4/8, k=1/2/6, m=32/64/128/4096
   - 边界情况：m=0（空 micro-batch）、k=num_experts、EP_SIZE=32
   - 随机 seed 固定：确保失败时 reproducible

4. 具体例子（test_ep_dedup_dispatch.py）：
   - simulate_sender：用 PyTorch 实现 sender 端的 dedup + sort
   - simulate_receiver：用 PyTorch 实现 receiver 端的 expand_idx adjust + gather
   - 对比 Triton 输出的 dedup_sorted_tokens, idxs, expand_idx 是否一致

5. 为什么不用 golden file？
   - Golden file 需要维护，且硬件/软件栈变化时容易失效
   - Reference impl 是代码，随代码一起演化，always up-to-date
```

---

#### 模板 3：Skills Ablation 验证（论文 §5.3）

**面试官问："你怎么证明 agent skills 真的提高了效率？"**

**回答框架：**

```
1. 实验设计：
   - 固定代码库（PithTrain）、固定 agent（Claude Code Opus 4.7）、固定任务
   - 变量：skills 开启 vs 关闭
   - 关闭时：从 working tree 和 git history 中完全删除 skills 文件
   - 每个条件跑 3 次，取 median

2. 任务选择：
   - validate-correctness：agent 需要运行训练、对比 loss curve
   - capture-nsys-profile：agent 需要 profile 训练、分析 kernel

3. 结果：
   - validate-correctness：Agent Turns 114 → 34（-70%），Session Duration 26.0 → 22.9 min
   - capture-nsys-profile：Agent Turns 75 → 36（-52%），Session Duration 9.4 → 6.6 min
   - Active GPU Time 几乎不变（因为 GPU work 是固定的）

4. 解释：
   - Skills 不改变 GPU 工作量，只减少 agent 的推理 overhead
   - Agent 不需要重新推导流程（如 "怎么跑 nsight"），直接按 skill 执行
   - 减少 trial-and-error：skill 的前置条件检查避免了 agent 踩坑

5. 反事实：如果不用 skills，agent 需要：
   - 自己搜索文档（多个 grep/read 调用）
   - 自己尝试命令行参数（可能失败多次）
   - 自己解析 nsight 输出（可能误解）
```

---

### 2.4 实验验证类问题的常见陷阱

| 陷阱 | 错误回答 | 正确回答 |
|------|---------|---------|
| 只讲结果不讲方法 | "我们的方法 loss 更低" | "我们在相同超参、相同数据 split 下，和 baseline 比 loss 低了 X%" |
| 混淆相关性和因果性 | "用了 Muon 优化器所以收敛更快" | "我们控制变量，只改优化器，其他超参不变，跑了 3 次取 median，收敛快了 X%" |
| 忽略 baseline 的工程优化 | "Megatron 没我们快" | "Megatron 我们遵循了 NVIDIA 官方 best practices，包括启用 FP8、调整 micro-batch size 等" |
| 指标选择不当 | "我们的 MFU 更高" | "MFU 在 BF16 和 FP8 之间不可比，我们改用 tokens/sec" |

---

## 3. 能力三：问题定位能力

### 3.1 面试官考察方式

这类问题通常以场景题形式出现：

```
"训练到第 1000 步，loss 突然变成 NaN，你怎么排查？"
"预期吞吐是 150K tokens/sec，实际只有 80K，你怎么定位瓶颈？"
"多节点训练时，某个 rank 卡住了，整个 job 不走了，你怎么处理？"
```

**关键**：面试官不看你会不会"背答案"，而看你 **排查问题的系统化思维**。

---

### 3.2 应对策略：分层排查法

```
第一层：收集现象（symptoms）
  - 什么时候发生的？(step X / 某个 rank / 某种配置)
  - 影响范围？(单 rank / 全局 / 特定模型)
  - 可复现吗？(固定 seed 是否复现)

第二层：缩小范围（isolation）
  - 是代码问题还是配置问题？
  - 是单机问题还是通信问题？
  - 是训练问题还是系统问题？

第三层：根因定位（root cause）
  - 看日志（loss、grad norm、tokens/sec、peak mem）
  - 看 profile（nsys、memory snapshot）
  - 看代码（最近改了什么）

第四层：解决方案与验证
  - 临时 workaround（让训练继续）
  - 根本 fix（改代码/配置）
  - 回归测试（确保不再出现）
```

---

### 3.3 PithTrain 高频故障场景与排查模板

#### 场景 1：训练 loss 突然变成 NaN

**排查流程：**

```
Step 1：收集现象
  - 哪个 step？(如 step 523)
  - 哪些 rank？(全局还是单 rank)
  - 之前 loss 正常吗？(如正常 loss ≈ 2.5)

Step 2：缩小范围
  - 是单步 NaN 还是逐步发散？(单步 → 梯度爆炸；逐步 → 学习率/数值不稳定)
  - 只在特定 rank 上 NaN？(EP/CP 通信问题)
  - FP8 还是 BF16？(FP8 更可能出现 scale 问题)

Step 3：根因定位（PithTrain 特有的排查点）
  1. 检查 gradient norm：
     - 如果 gradient norm 突然飙升 → 梯度爆炸
     - 可能原因：FP8 scale 太小（量化溢出）、learning rate 太大、MoE load imbalance
  
  2. 检查 FP8 scale：
     - 查看 FP8WeightCacheControl 的 scale 值
     - 如果某个 block 的 amax=0 → scale=0 → 量化时除零 → NaN
     - 代码位置：pithtrain/operators/deepgemm_fp8_quantize.py:_compute_fp8_scale
     - 防护：`amax_clamped = tl.maximum(amax.to(tl.float32), 1e-4)`
  
  3. 检查 MoE load balance：
     - 如果某个 expert 的 token 数突然激增 → 该 expert 的权重梯度可能爆炸
     - 查看 MoELoadBalanceLossTracker 的历史值
  
  4. 检查 FSDP 梯度同步：
     - 如果某些 rank 的梯度未同步 → 参数更新不一致 → 后续 step 数值不稳定
     - 查看是否某些 rank 卡在 post_backward

Step 4：解决方案
  - 临时 workaround：
    * 从上一个 checkpoint resume
    * 降低 learning rate
    * 增大 gradient clip norm（当前是 1.0）
  - 根本 fix：
    * 修复 FP8 scale 的数值稳定性（加 clamp）
    * 调整 MoE load balance loss 的系数
    * 检查是否有 invalid input（如全零 token）

Step 5：验证
  - 固定 seed 复现 NaN
  - 修复后跑 100 steps，确保 NaN 不再出现
  - 对比修复前后的 loss curve
```

**对应 PithTrain 源码：**
- `pithtrain/tasks/pretrain_lm.py:train_step` L337-495 — loss、grad norm、peak mem 打印
- `pithtrain/dualpipe/utils.py:FP8WeightCacheControl` — FP8 weight cache
- `pithtrain/modules/load_balance.py:MoELoadBalanceLossTracker` — load balance 监控

---

#### 场景 2：训练吞吐远低于预期

**排查流程：**

```
Step 1：收集现象
  - 预期：150K tokens/sec
  - 实际：80K tokens/sec
  - 是所有 rank 都慢，还是特定 rank？

Step 2：缩小范围
  - 单节点 vs 多节点？（多节点可能网络瓶颈）
  - 相同配置下 Megatron 是多少？（确认是 PithTrain 问题还是硬件问题）
  - 最近改了什么？（代码变更、配置变更、硬件变更）

Step 3：根因定位（PithTrain 特有的排查点）
  1. Profile（Nsight Systems）：
     - 看 top kernels：哪些 CUDA kernel 占了最多时间？
     - 看 pipeline bubble：rank 之间的通信是否同步等待？
     - 看 all-to-all 时间：EP dispatch/combine 是否成为瓶颈？
  
  2. 检查 PP 调度：
     - 如果 bubble 时间很长 → PP 的 num_chunks 不够？
     - 查看 8 步调度中，每个 step 的 compute 和 comm 是否重叠
  
  3. 检查 EP all-to-all：
     - 如果 all-to-all 时间 > compute 时间 → comm 成为瓶颈
     - 可能原因：NVLink 故障、dedup 算法效率低、micro-batch 太小
  
  4. 检查 CP ring attention：
     - 如果 CP > 1，ring attention 的 P2P 是否卡住？
     - 查看 `post_ring_kv` 的 isend/irecv 是否同步等待
  
  5. 检查 FSDP：
     - 如果 FSDP 的 gradient sync 开销大 → 查看 post_backward 是否频繁触发
     - 当前 PithTrain 在 pipeline 循环中 suppress post_backward，只在最后手动调用一次
     - 如果意外触发，会增加 150-250μs × 层数的 overhead

Step 4：解决方案
  - 如果是 PP bubble：
    * 增大 num_chunks（如果内存允许）
    * 或者降低 PP 增大 DP
  - 如果是 EP all-to-all 瓶颈：
    * 检查 NVLink 拓扑（nvidia-smi topo -m）
    * 增大 micro_batch_size（减少 all-to-all 的 fixed overhead）
  - 如果是 CP ring 瓶颈：
    * 检查 ring 的 P2P 是否在 NVLink 域内
    * 降低 CP 增大 PP/DP

Step 5：验证
  - 用 PithTrain 自带的 nsys profile skill：`capture-nsys-profile`
  - 对比优化前后的 step time
  - 确保 throughput 提升的同时，loss curve 不变（ correctness 不受影响）
```

---

#### 场景 3：多节点训练时某个 rank 卡住，整个 job hang

**排查流程：**

```
Step 1：收集现象
  - 哪个 rank 卡住？(rank 5 / 某个 PP stage)
  - 卡在哪个 step？(训练中 / checkpoint 加载时)
  - 其他 rank 在等什么？(NCCL timeout / P2P wait)

Step 2：根因定位（PithTrain 特有的排查点）
  1. NCCL 超时：
     - PithTrain 默认 timeout = 15 min（DistributedCfg.timeout）
     - 如果某个 rank 崩溃，其他 rank 会 hang 直到超时
     - 查看 `setup_failfast_excepthook` 是否生效（应该 os._exit(1)）
  
  2. P2P 通信死锁：
     - DualPipeV 的 isend/irecv 必须配对
     - 如果某个 rank 的 `_send_forward` 没有对应的 `_recv_forward` → deadlock
     - 查看 comm.py 的 TENSOR_SHAPES 和 TENSOR_DTYPE 是否正确设置
  
  3. FSDP 同步问题：
     - 如果某些 rank 的 FSDP state 不一致 → all-reduce 会 hang
     - 查看是否某些 rank 的 gradient 为 None（未参与 backward）

Step 3：解决方案
  - 短期：减小 timeout（如 5 min），让 job 快速失败
  - 长期：
    * 确保所有 rank 的代码/配置一致（diff 对比）
    * 确保 checkpoint 的 rank 文件都存在（rng-rank-*.pt）
    * 用 NCCL 的 `TORCH_NCCL_DUMP_ON_TIMEOUT=1` 查看详细日志

Step 4：验证
  - 在小规模（2 节点 4 卡）上复现 hang
  - 确认 fail-fast 机制能在合理时间内杀掉 job
```

**对应 PithTrain 源码：**
- `pithtrain/modules/distributed.py:setup_failfast_excepthook` L139-162
- `pithtrain/modules/distributed.py:setup_default_process_group` L110-136 — NCCL timeout
- `pithtrain/dualpipe/comm.py` — P2P 通信

---

### 3.4 问题定位类问题的加分项

| 加分项 | 说明 |
|--------|------|
| 提到具体工具 | "用 nsys profile 看 top kernels" 而不是 "我 profiled" |
| 提到具体代码位置 | "pithtrain/dualpipe/dualpipev.py:run_post_backward" |
| 有临时 workaround | "先 resume 训练，再慢慢修" 而不是 "我必须停掉修好" |
| 考虑边界情况 | "m=0 时 dedup 返回空 tensor，不会 crash" |
| 有回归测试 | "修完后跑 `pytest tests/test_ep_dedup_dispatch.py`" |

---

## 4. 能力四：工程落地能力

### 4.1 面试官考察方式

```
"你怎么把这个算法部署到生产环境？"
"系统上线后怎么保证稳定？"
"数据出问题了怎么回滚？"
"如果资源有限，你先优化什么？"
```

**关键**：面试官想知道你 **不是只在实验室跑通，而是考虑过真实生产环境的约束**。

---

### 4.2 应对策略：生产就绪五要素

```
1. 可观测性：监控哪些指标？异常怎么报警？
2. 可靠性：单点故障怎么处理？如何快速恢复？
3. 可回滚：上线后发现 regress，怎么快速回退？
4. 可扩展：模型变大、数据变多时，系统怎么scale？
5. 成本意识：GPU  hours、内存、网络带宽的 trade-off
```

---

### 4.3 PithTrain 工程落地细节与回答模板

#### 模板 1：Checkpoint 机制与数据回滚

**面试官问："训练到第 5000 步发现数据有噪声，要回滚到第 4000 步重训，你怎么做？"**

**回答框架：**

```
1. PithTrain 的 checkpoint 机制：
   - 每 save_interval 步保存一次（如每 1000 步）
   - 格式：torch-dcp/step-00004000/（包含 model + optimizer + scheduler + rng state）
   - Canonical format：PP-independent，可以 reshard 到不同并行配置

2. 回滚步骤：
   a) 停止当前训练（发送 SIGTERM，等待 graceful shutdown）
   b) 修改 script.py：save_location 指向原目录，但不需要改
   c) 重新运行：load_checkpoint 会自动加载最新的 step-00004000
   d) 如果要改超参（如降低 lr），在 script.py 中修改后重启
   e) 如果要跳过有噪声的数据，需要修改 dataset 的起始 index

3. 数据回滚的细节：
   - PithTrain 的 dataset 是 memmap 文件（*.bin）
   - 数据本身不可变（tokenize 后只读）
   - 如果有噪声数据，需要：
     a) 重新 tokenize 干净的语料（examples/tokenize_corpus/launch.sh）
     b) 或者修改 get_global_batch 中的 index 计算，跳过噪声样本

4. 验证回滚成功：
   - 检查 step 号：ctx.training.step 应该从 4000 继续
   - 检查 loss curve：前几步的 loss 应该和 step 4000 时一致
   - 检查 RNG state：rng-rank-*.pt 确保随机性可复现
```

**对应源码：**
- `pithtrain/tasks/pretrain_lm.py:load_checkpoint` L302-334 — checkpoint 加载
- `pithtrain/tasks/pretrain_lm.py:save_checkpoint` L261-299 — checkpoint 保存
- `pithtrain/modules/checkpoint.py` — canonical vs localized 格式转换

---

#### 模板 2：多节点训练的稳定性保障

**面试官问："32 卡（4 节点）训练跑了 3 天，某个节点网络断了，怎么保证不丢进度？"**

**回答框架：**

```
1. PithTrain 的容错机制：
   a) Fail-fast excepthook：
      - 任何 rank 抛出未捕获异常 → os._exit(1)
      - 避免其他 rank hang 在 NCCL drain
      - 代码：pithtrain/modules/distributed.py:setup_failfast_excepthook
   
   b) NCCL heartbeat timeout：
      - 默认 15 分钟（DistributedCfg.timeout）
      - 如果某个 rank 失联，其他 rank 在 15 分钟后自动失败
      - 环境变量：TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
   
   c) Checkpoint 定期保存：
      - 每 save_interval 步保存一次
      - 可以从任意 checkpoint resume
      - 即使 job 失败，最多丢失 save_interval 步的进度

2. 实际部署建议：
   a) 使用 shared filesystem（如 NFS, Lustre）存储 checkpoint
      → 所有节点都可以访问，某个节点挂了不影响 checkpoint 可用性
   
   b) 调整 timeout：
      - 多节点场景下，网络抖动可能导致超时
      - 可以适当增大 timeout（如 30 min）
      - 但不能太大，否则真正的故障会拖很久才被发现
   
   c) 使用 SLURM 的自动重启：
      - srun 的 --restart 参数可以在节点故障时自动重新启动
      - PithTrain 的 launch.sh 已经支持 SLURM（读取 SLURM_* 环境变量）

3. 数据一致性：
   - PithTrain 使用 DCP 保存 checkpoint
   - 每个 rank 写自己的 shard（distributed saving）
   - 如果某个 rank 写入失败，checkpoint 可能不完整
   - DCP 有 metadata 校验，可以检测不完整 checkpoint
```

---

#### 模板 3：内存优化与资源有限时的优先优化

**面试官问："只有 8×H100（80GB），想跑 Qwen3-30B-A3B，但 OOM 了，你怎么优化？"**

**回答框架：**

```
1. 当前默认配置（PithTrain examples）：
   - Qwen3-30B-A3B 需要 8×H200（141GB）或 8×B200（180GB）
   - 8×H100（80GB）默认跑不下

2. 优化选项（按优先级排序）：

   优先级 1：降低 PP（如果当前 PP>1）
     - PP 越大，每卡存的层数越少，但 PP 通信开销大
     - 对于 30B 模型，PP=1 可以用更大的 DP/EP
     - 但 PP=1 时模型需要能放进单卡 → 30B/8 = 3.75B per rank，加上 FSDP shard，可以
   
   优先级 2：开启 FP8
     - FP8 训练：权重和激活都减半
     - 配置：training.fp8_training = "deep-gemm"
     - 要求：Hopper (SM90) 或 Blackwell (SM100) → H100 支持
     - 预期：内存减少 30-40%，吞吐提升 20-30%
   
   优先级 3：调整 EP 和 CP
     - EP 越大，每卡存的 expert 越少 → 内存减少
     - 但 EP 增大 → all-to-all 参与方增多 → 通信 overhead
     - CP 越大，每卡的 sequence 越短 → 激活内存减少
     - Trade-off：EP 和 CP 都受限于 world_size
   
   优先级 4：减小 micro_batch_size
     - micro_batch_size 直接影响激活内存
     - 但太小会导致 GPU 利用率下降（kernel launch overhead）
     - PithTrain 用 accumulate_steps 补偿：global_batch_size = micro_batch_size × dp_size × ep_size × accumulate_steps
   
   优先级 5：Gradient Checkpointing（当前 PithTrain 未实现）
     - 用 compute 换内存：不保存前向激活，反向时重算
     - 可以减少 30-50% 激活内存
     - 但会增加 20-30% 的训练时间

3. 推荐配置（8×H100 跑 Qwen3-30B-A3B）：
   - PP=2, EP=8, DP=1, CP=1（16 卡 → 但只有 8 卡，所以 PP=2 不行）
   - 实际上 8 卡只能 PP=1, EP=8, DP=1
   - 开启 FP8：training.fp8_training = "deep-gemm"
   - micro_batch_size=1, sequence_length=2048（而非 4096）
   - 如果还 OOM，再降 sequence_length 或开 gradient checkpointing

4. 监控内存：
   - PithTrain 内置 memory profiler：
     * training.memory_profile_start / memory_profile_step
     * 输出：snapshot-rank*.pickle（可导入 https://pytorch.org/memory_viz）
   - 峰值内存监控：train_step 中打印 peak_gpu_mem
```

**对应源码：**
- `pithtrain/tasks/pretrain_lm.py:train_step` L416-420 — peak_gpu_mem 监控
- `pithtrain/modules/training.py:TrainingCfg` L134-292 — 超参配置
- `pithtrain/modules/distributed.py:DistributedCfg` L17-66 — 并行配置

---

#### 模板 4：混合精度训练的策略

**面试官问："BF16 参数、FP32 reduce、FP8 compute，这三种精度在 PithTrain 中怎么配合？"**

**回答框架：**

```
1. PithTrain 的混合精度策略：
   - 参数存储：BF16（ MixedPrecisionPolicy param_dtype=torch.bfloat16）
   - 梯度规约：FP32（reduce_dtype=torch.float32）
   - 前向计算：FP8（如果 fp8_training="deep-gemm"）或 BF16

2. 为什么这样设计？
   - BF16 参数：节省内存（2 bytes vs 4 bytes），现代 GPU 的 matmul 对 BF16 有 tensor core 优化
   - FP32 reduce：梯度规约需要高精度，避免多个小梯度累加时舍入误差累积
   - FP8 compute：利用 Hopper/Blackwell 的 FP8 tensor core，吞吐翻倍

3. FSDP 的 MixedPrecisionPolicy：
   ```python
   mp = MixedPrecisionPolicy(
       param_dtype=torch.bfloat16,    # 参数存 BF16
       reduce_dtype=torch.float32,     # 梯度规约用 FP32
       output_dtype=None,              # 前向输出不 cast（保持输入 dtype）
       cast_forward_inputs=True,       # 前向输入 cast 到 param_dtype
   )
   ```

4. Muon 优化器的精度处理：
   - Muon 的 momentum buffer 是 FP32
   - 参数是 BF16，梯度是 BF16
   - Muon 的 Newton-Schulz 正交化在 FP32 下做（数值更稳定）
   - 更新后的参数 cast 回 BF16

5. 工程落地的注意事项：
   - 如果模型有 invalid value（如 inf），BF16 会变成 nan，比 FP32 更难 debug
   - 建议：训练开始时用 BF16 跑 100 步，检查是否有 nan
   - 如果出现 nan，临时切换到 FP32 训练定位问题
```

---

### 4.5 工程落地类问题的通用回答模板

| 问题类型 | 回答结构 |
|---------|---------|
| "怎么部署？" | 环境准备 → 数据准备 → 配置 → 启动 → 监控 → 上线后迭代 |
| "怎么保证稳定？" | 监控指标 + 报警阈值 + 自动 checkpoint + fail-fast 机制 |
| "怎么回滚？" | 定期 checkpoint + 版本化超参 + 数据不可变 + resume 流程 |
| "资源有限怎么优化？" | 优先级排序（PP > EP > CP > DP）+ 内存/计算/通信 trade-off |

---

## 5. 能力五：业务与实际场景理解

### 5.1 面试官考察方式

```
"这个方案适合什么场景？不适合什么？"
"如果公司只有 16 张 H100，你会怎么建议？"
"用户（如产品经理）关心的是模型效果，你关心的是 throughput，怎么对齐？"
"上线后 ROI 怎么衡量？"
```

**关键**：面试官想知道你 **不是技术栈的奴隶，而是能用技术思维解决业务问题的人**。

---

### 5.2 应对策略：场景-技术-成本三角

```
回答任何"适不适合"的问题，都要覆盖三个维度：

1. 场景维度：数据特征、模型规模、 latency 要求、预算约束
2. 技术维度：吞吐、精度、稳定性、可扩展性
3. 成本维度：GPU 数量、训练时间、人力维护、机会成本
```

---

### 5.3 PithTrain 适用场景与回答模板

#### 模板 1：PithTrain 适合什么团队/场景？

**回答框架：**

```
适合的场景：

1. 研究团队 / 大学实验室：
   - 需要快速实验新 MoE 架构（如 DynMoE, MoBA, MoE++）
   - 代码紧凑（11K LoC），agent 友好，新成员上手快
   - 不需要 Megatron 那种 160K LoC 的 broad coverage

2. 中小规模 MoE 预训练：
   - 模型：DeepSeek-V2-Lite, Qwen3-30B-A3B, GPT-OSS-20B/120B
   - 硬件：8-32 张 H100/B200
   - 不需要千卡级别的超大规模训练

3. Agent 辅助开发：
   - 团队用 AI coding agent（Claude Code）开发/维护训练框架
   - PithTrain 的 agent-native 设计让 agent 的 session duration 减少 35-62%

4. 需要快速迭代的场景：
   - 从论文 idea 到可训练原型的时间很重要
   - PithTrain 的 skills（add-new-model, validate-correctness）加速迭代

不适合的场景：

1. 超大规模生产训练（如 GPT-4 级别，万卡）：
   - PithTrain 的 throughput 虽然匹配 Megatron，但缺少一些生产级 feature：
     * 自动并行搜索（auto-parallelism）
     * 弹性训练（fault tolerance  beyond checkpoint）
     * 多平台支持（AMD, Intel GPU）
   - 这些需要长期工程积累，不是 compact framework 能覆盖的

2. Dense 模型训练：
   - PithTrain 的核心优化（EP dispatch, MoE load balance）对 Dense 模型无用
   - 虽然支持 Dense，但不是设计目标，性价比不如专门优化的 Dense 框架

3. 需要 broad model coverage 的场景：
   - PithTrain 只支持 4 个模型家族（DeepSeek-V2, Qwen3, GPT-OSS）
   - 如果需要训练 LLaMA、BERT 等，需要自己 port
```

---

#### 模板 2：如果资源有限，先优化什么？

**面试官问："公司只有 16 张 H100，想训最大的 MoE 模型，你怎么分配 PP/EP/CP/DP？"**

**回答框架：**

```
1. 确定约束：
   - 16 张 H100（80GB each）
   - 目标：训最大的 MoE 模型
   - 约束：PP × CP × EP 必须整除 16

2. 优化目标：
   - 最大化可训练模型参数量
   - 同时保证 reasonable throughput

3. 分配策略：
   a) PP（Pipeline Parallel）：
      - PP 越大，每卡的模型越小，但 bubble 越大
      - 对于 16 卡，PP=2 是 sweet spot（bubble 可控，模型可以很大）
      - PP=4 时 bubble 太大，吞吐下降明显
   
   b) EP（Expert Parallel）：
      - EP 越大，每卡的 expert 越少，激活内存越小
      - 但 EP 越大，all-to-all 通信量越大（参与方多）
      - 对于 MoE，EP 应该尽量大（因为 expert 是 MoE 的主要参数量）
      - 推荐：EP=8（每卡 1/8 的 experts）
   
   c) CP（Context Parallel）：
      - CP 只在 sequence_length 很长时需要（如 32K+）
      - 如果 sequence_length=2048，CP=1 即可
      - 如果需要 32K，CP=2（每卡 16K）
   
   d) DP（Data Parallel）：
      - DP = 16 / (PP × CP × EP)
      - 自动推导，不需要手动设置

4. 推荐配置：
   - PP=2, EP=8, CP=1, DP=1（16 = 2×1×8×1）→ 但 2×1×8=16, DP=1
   - 或者 PP=1, EP=8, CP=2, DP=1（16 = 1×2×8×1）
   - 或者 PP=2, EP=4, CP=2, DP=1（16 = 2×2×4×1）

   对于大模型（如 GPT-OSS-120B）：
     PP=4, EP=8, CP=1, DP=... → 4×1×8=32 > 16，跑不下
     需要 PP=2, EP=8, CP=1, DP=1（16 = 2×1×8×1）

5. 内存估算：
   - 用 PithTrain 自带的 memory_estimator：
     `python -m tools.memory_estimator --help`
   - 输入：模型 config、并行配置
   - 输出：peak memory、是否 OOM

6. 如果还 OOM：
   - 开启 FP8（如果硬件支持）
   - 减小 micro_batch_size
   - 减小 sequence_length
   - 最后手段：gradient checkpointing（PithTrain 未实现，需要自己加）
```

---

#### 模板 3：SFT 和 RLHF 中的 MoE 特殊考虑

**面试官问："预训练的 MoE 模型直接拿来 SFT，router 需要重新训练吗？load balance 怎么处理？"**

**回答框架：**

```
1. Router 在 SFT 中的行为：
   - Pre-training 的 router 已经学会把不同 token 分配到不同 experts
   - SFT 的数据分布不同（instruction-response，token 类型更集中）
   - Router 需要微调（fine-tune），否则可能：
     a) 某些 expert 过载（instruction token 集中在少数 experts）
     b) 某些 expert 闲置（rare token 的 expert 不再被激活）

2. Load Balance 在 SFT 中的调整：
   - Pretraining：用 "global-batch" 或 "sequence" level loss
   - SFT：数据量小（几千到几万条），"micro-batch" loss 的统计不稳定
   - 推荐：用 "global-batch" loss + 增大 lb_coef（如 1e-2 → 1e-1）
   - 原因：SFT 的 batch 多样性差，router 容易 collapse

3. RLHF 中的 MoE 特殊考虑：
   a) Reward hacking：
      - RL 的 reward signal 可能导致模型集中输出某些 token
      - Router 随之 collapse 到少数 experts
      - 需要更强的 load balance loss
   
   b) PP 和 RL 的兼容性：
      - PPO 需要 4 个模型（policy, reference, reward, critic）
      - 每个模型都需要 PP 并行
      - 如果内存紧张，可以把 critic 做成 Dense（共享 policy 的 PP stage）
   
   c) Sampling 和 training 的一致性：
      - MoE 的 top-k 路由是确定性的（给定 hidden_states，topk_idx 固定）
      - Sampling 不影响 routing（因为 routing 在 attention 之后，在 MLP 之前）
      - 所以 SFT/RLHF 的 forward 和 pretraining 一致

4. PithTrain 中的实现：
   - 复用 PretrainLMCfg，只需换 dataset 和 lr
   - 调整 moe_load_balance_coef 和 moe_load_balance_type
   - 如果做 RLHF，需要自己实现 PPO/GRPO loop（当前 PithTrain 只支持 pretrain）
```

---

## 6. 综合模拟：五类能力串联回答

### 6.1 完整案例：从问题定义到上线的全流程

**面试官给了一个开放题："公司想训一个 30B 的 MoE 模型做客服对话，你有 16 张 H100，你怎么设计训练方案？"**

**你应该这样回答（串联五类能力）：**

---

#### 第一层：问题定义与场景分析（能力五）

```
"首先我需要明确业务场景和约束：

业务场景：
  - 任务：客服对话（multi-turn，需要 long context）
  - 模型：30B MoE（激活 3B，推理成本低）
  - 数据：历史客服对话 + 合成数据（预计 10B tokens）

约束：
  - 硬件：16 张 H100（80GB）
  - 时间：希望 2 周内完成 pretraining
  - 预算：不能加卡

关键 trade-off：
  - 模型大小 vs 硬件限制：30B MoE 在 16×H100 上跑不下默认配置
  - 训练时间 vs 效果：数据量 10B tokens，希望至少训 2 epoch
  - 吞吐 vs 精度：FP8 可以提速，但需要验证精度损失"
```

---

#### 第二层：技术方案设计（能力一）

```
"基于 PithTrain 框架，我的设计方案：

1. 并行策略：
   - PP=2, EP=8, CP=1, DP=1（16 = 2×1×8×1）
   - 理由：
     * PP=2：把 30B 模型切成两半，每卡约 15B 参数（加上 FSDP shard，激活内存可控）
     * EP=8：充分利用 MoE 的特性，每卡只存 1/8 的 experts
     * CP=1：如果 sequence_length <= 2048，不需要 CP
     * 如果 sequence_length > 2048，改为 CP=2, EP=4

2. 精度策略：
   - 开启 FP8（deep-gemm）：H100 支持，内存减少 30-40%，吞吐提升 20-30%
   - 理由：客服对话对精度要求不如代码/数学高，FP8 的精度损失可接受

3. 数据策略：
   - Sequence length：2048（多轮对话截断）
   - Global batch size：1024（16 卡 × 每卡 micro_batch_size × accumulate_steps）
   - 如果 OOM，降 micro_batch_size 或 sequence_length

4. 负载均衡：
   - moe_load_balance_type = "global-batch"
   - moe_load_balance_coef = 1e-2
   - 理由：客服对话的 token 分布可能不均匀（如常见问题集中在少数 experts）
"
```

---

#### 第三层：实验验证方案（能力二）

```
"怎么证明这个方案可行？

1. 小规模验证（1-2 天）：
   - 用 2 张 H100，PP=1, EP=2, 跑 100 steps
   - 验证：loss 下降、无 NaN、吞吐达标
   - 如果小规模跑不通，大规模肯定跑不通

2. 内存验证：
   - 用 memory_estimator 估算峰值内存
   - 实际跑 10 steps，监控 peak_gpu_mem
   - 留 10% 内存余量（避免 OOM）

3. 吞吐基准测试：
   - 跑 100 steps，取最后 50 steps 的 median throughput
   - 对比 Megatron 同配置（如果有条件）
   - 目标：>= 100K tokens/sec（16×H100 跑 30B MoE 的合理预期）

4. 精度验证：
   - 前 1000 steps 的 loss curve 应该平滑下降
   - 如果 FP8 导致 loss 不稳定，回退到 BF16

5. 负载均衡验证：
   - 监控 per-expert token count
   - 如果某些 expert 的 token 数 < 10% 或 > 30%，增大 lb_coef
"
```

---

#### 第四层：问题定位预案（能力三）

```
"如果训练中出了问题，我怎么排查？

1. Loss NaN：
   - 先看 gradient norm：如果飙升 → 梯度爆炸
   - 检查 FP8 scale：是否有 amax=0 的 block
   - 检查 MoE load：是否有 expert 过载
   - 临时方案：resume 上一个 checkpoint，降低 lr，增大 grad clip

2. 吞吐突然下降：
   - 用 nsys profile 看 top kernels
   - 检查是否 EP all-to-all 成为瓶颈（comm time > compute time）
   - 检查是否 PP bubble 增大（num_chunks 不够）
   - 检查是否有 rank OOM 导致降级

3. 某些 rank hang：
   - 查看 NCCL 日志（TORCH_NCCL_DUMP_ON_TIMEOUT=1）
   - 检查 fail-fast 是否生效（15 min timeout）
   - 检查 checkpoint 是否完整（DCP metadata）

4. 模型效果不好（上线后）：
   - 看 loss curve：是否收敛？是否过拟合？
   - 看 routing pattern：是否 expert collapse？
   - 看下游 eval：客服对话的BLEU/ROUGE/人工评分
"
```

---

#### 第五层：工程落地与成本（能力四+五）

```
"上线后怎么保证稳定和成本可控？

1. Checkpoint 策略：
   - 每 1000 steps 保存一次
   - 保留最近 5 个 checkpoint（节省存储）
   - 用 canonical format（PP-independent），方便 future resharding

2. 监控指标：
   - 实时：tokens/sec、loss、grad norm、peak mem、per-expert token count
   - 报警：loss NaN、grad norm > 100、throughput < 80K、任何 rank OOM

3. 成本估算：
   - 16×H100 × 2 周 ≈ 16 × 24 × 14 × $2/hour（假设云 GPU $2/hr）
     ≈ $10,752（约 7 万人民币）
   - 如果开启 FP8，训练时间减少 30% → 节省 $3,000

4. 回滚策略：
   - 每 1000 steps 一个 checkpoint
   - 如果发现数据问题，可以回滚到任意 checkpoint
   - 数据不可变（memmap），不需要重新 tokenize

5. 上线后迭代：
   - 先上小流量（10%），监控效果
   - 如果效果好，全量；如果不好，快速回滚
   - 持续收集 bad case，补充到训练数据，定期 retrain
"
```

---

#### 第六层：总结与 trade-off

```
"总结一下，我的方案的核心 trade-off：

1. 精度 vs 速度：FP8 换吞吐，但需要验证精度
2. 并行度 vs bubble：PP=2 让模型能放下，但 bubble 时间增加
3. 负载均衡 vs 模型效果：lb_loss 会轻微影响模型效果，但不加会导致 expert collapse
4. 数据量 vs 时间：10B tokens 在 16×H100 上需要约 2 周，如果时间更紧，可以减数据量

最终建议：
  - 先跑小规模验证（2 卡 100 steps）
  - 确认 loss 下降、无 NaN、吞吐达标
  - 再上 16 卡全量训练
  - 每 1000 steps 保存 checkpoint，监控关键指标
"
```

---

### 6.2 五类能力自检清单

面试结束后，你可以用以下清单自我评估：

| 能力 | 自检问题 | 达标标准 |
|------|---------|---------|
| 底层原理 | 我能讲清每个设计选择的 problem-solution-trade-off 吗？ | 能说出 3 个局限性和 2 个改进方向 |
| 实验验证 | 我能说出实验的对照组、控制变量、指标选择吗？ | 能解释为什么选这个指标而不是其他 |
| 问题定位 | 我有系统化的排查流程，而不是 guess-and-check？ | 能按 分层排查法 说出现象→范围→根因→解决 |
| 工程落地 | 我考虑过监控、回滚、稳定性、成本吗？ | 能说出 3 个监控指标和 2 个容错机制 |
| 业务理解 | 我能说出方案的适用场景和 ROI 吗？ | 能说出什么场景不该用这个方案 |

---

## 附录：五类能力对应 PithTrain 源码速查

| 能力 | 考察点 | 关键源码 |
|------|--------|---------|
| 底层原理 | DualPipeV 为什么 V 形？ | `dualpipev.py` L57-88, L478-544 |
| 底层原理 | MoE dedup 为什么用 counting sort？ | `ep_dispatch.py` L270-419 |
| 底层原理 | FP8 为什么 128-element block？ | `deepgemm_fp8_quantize.py` L39-81 |
| 底层原理 | Zigzag CP 为什么能平衡计算量？ | `ring_attention.py` L1-49 |
| 实验验证 | Operator correctness 怎么保证？ | `tests/test_ep_dedup_dispatch.py` |
| 实验验证 | 吞吐对比的 baseline 怎么选？ | 论文 §5.1, `05_evaluation.tex` |
| 实验验证 | Skills ablation 怎么设计？ | 论文 §5.3, `05_evaluation.tex` L258-329 |
| 问题定位 | Loss NaN 怎么排查？ | `train_step` L337-495（grad norm, peak mem） |
| 问题定位 | 多节点 hang 怎么处理？ | `distributed.py` L139-162（fail-fast） |
| 问题定位 | 吞吐下降怎么 profile？ | `capture-nsys-profile` skill |
| 工程落地 | Checkpoint 怎么回滚？ | `pretrain_lm.py` L302-334 |
| 工程落地 | 多节点稳定性怎么保障？ | `distributed.py` L110-136（NCCL timeout） |
| 工程落地 | 内存不足怎么优化？ | `memory_estimator` tool, `apply_fsdp` L350-414 |
| 业务理解 | 什么场景适合 PithTrain？ | 论文 §3, `README.md` |
| 业务理解 | 资源有限时怎么配置？ | `user-guide.md` L123-148（Scaling） |
