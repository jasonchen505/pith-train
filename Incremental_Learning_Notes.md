# PithTrain 复现增量学习笔记

> 记录在 **8×4090 复现 PithTrain** 过程中，新学到的、前两轮文档未覆盖的细节与 Insight。
> 前两轮文档：《PithTrain_面试深挖准备.md》（知识点地图）、《PithTrain_五类面试能力应对.md》（面试应对策略）

---

## 一、硬件层：从"能跑"到"能稳定跑"的细节

### 1.1 SM 版本与 FP8 的硬约束

**新学到的点**：
- PithTrain 的 `setup_model` 中有隐式的 SM 版本检查：
  ```python
  # pithtrain/layers/deepgemm_fp8_linear.py
  # 如果 fp8_training="deep-gemm"，会 import deep_gemm
  # deep_gemm 内部会检查 torch.cuda.get_device_capability() >= (9, 0)
  ```
- **4090 (SM89) 不支持 FP8 Tensor Core**，这不是软件限制，是硬件限制。
- `deep_gemm` 库在 SM89 上会直接报错或 fallback 到 BF16。

**之前文档的遗漏**：
- 前两轮文档都假设在 Hopper/Blackwell 上运行，没有讨论 **非 FP8 硬件** 的适配。
- 实际工程中，**硬件代际差异** 是第一个要解决的问题。

**实际做法**：
```python
# 在 script.py 中强制禁用 FP8
training.fp8_training = "disabled"  # 4090 必须走这个
```

---

### 1.2 显存分布的精确估算

**新学到的点**：

PithTrain 的显存分为以下几个部分：

```
峰值显存 = max(
    forward 峰值,
    backward 峰值,
    optimizer step 峰值
)

forward 峰值 ≈
    model weights (sharded by FSDP)
    + activations (hidden_states, attention output, MoE dispatched tokens)
    + intermediate buffers (IntermediateTensors, pre-allocated)

backward 峰值 ≈
    forward 峰值
    + gradients (same size as weights, but sharded)
    + optimizer states (Muon: fp32 momentum; AdamW: fp32 momentum + variance)
    + temporary buffers (autograd graph, wgrad)

optimizer step 峰值 ≈
    model weights (sharded)
    + optimizer states (full precision)
    + temp buffer for parameter update
```

**DeepSeek-V2-Lite 在 8×4090 上的估算**：

| 组件 | 单卡估算 | 说明 |
|------|---------|------|
| Model weights (BF16) | ~4 GB | 16B total / 8 EP ranks, 非 MoE 参数 FSDP shard |
| Optimizer states (FP32) | ~8-12 GB | Muon momentum + AdamW states |
| Activations (seq=2048, bs=1) | ~4-6 GB | hidden_states + attention cache + MoE dispatched tokens |
| IntermediateTensors | ~1-2 GB | Pre-allocated, reused |
| **Total** | **~18-22 GB** | 接近 24GB 上限 |

**之前文档的遗漏**：
- 前两轮文档没有给出具体的显存估算公式。
- 实际调参时，**显存是第一个需要精确估算的资源**。

**实用工具**：
```bash
# PithTrain 自带的 memory estimator
python -m tools.memory_estimator --help
# 但这个工具可能还在开发中（README 提到 "still under construction"）
```

**实际做法**：
```python
# 在 train_step 中监控
peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak GPU memory: {peak_gpu_mem:.2f} GB")
# 留 2GB 余量（避免 CUDA OOM）
```

---

## 二、系统层：DualPipeV 的实际约束

### 2.1 `num_chunks >= 2 * pp_size` 的硬性要求

**新学到的点**：
```python
# pithtrain/dualpipe/dualpipev.py:step()
assert num_chunks > 0 and num_chunks >= pp_size * 2, f"{num_chunks=}, {pp_rank=}"
```

- 这个 assert 是 **硬性约束**，不是建议。
- 原因：V 形布局需要 `2*pp_size` 个 chunks（每个 rank 持有两个 chunks）。
- 如果 `num_chunks < 2*pp_size`，调度器没有足够的 micro-batch 来填充 V 形的两个分支。

**之前文档的遗漏**：
- 前两轮文档解释了"为什么需要 num_chunks >= 2*PP"，但没有强调这是 **assert**，不是可调参数。

**实际影响**：
```python
# 如果 PP=2，num_chunks 至少是 4
# 如果 global_batch_size=8, micro_batch_size=1, dp_size=1, ep_size=8
# accumulate_steps = 8 / (1 * 1 * 8) = 1  ← 太小！
# 需要增大 global_batch_size 或减小 ep_size

# 可行的配置：
# PP=2, EP=4, micro_bs=1, global_bs=16
# accumulate_steps = 16 / (1 * 1 * 4) = 4 >= 2*2=4 ✓
```

---

### 2.2 IntermediateTensors 预分配的 in-place 语义

**新学到的点**：

`IntermediateTensors` 的结构是：
```python
@dataclass
class IntermediateTensorsLayer:
    stage1: Stage1Record
    stage2: Stage2Record
    stage3: Stage3Record
    stage4: Stage4Record
    stage5: Stage5Record

@dataclass
class IntermediateTensors:
    prolog: PrologRecord
    layers: List[IntermediateTensorsLayer]
    epilog: EpilogRecord
```

- 每个 record 包含 `args` (输入) 和 `outs` (输出)。
- **关键**：这些 record 是 **pre-allocated**，在 pipeline 循环中被 **in-place 修改**。
- 代码模式：
  ```python
  # 不是创建新的 record，而是修改现有 record 的字段
  intermediate_tensors0.layers[layer_idx0].stage1.args = record.args
  intermediate_tensors0.layers[layer_idx0].stage1.outs = record.outs
  ```

**之前文档的遗漏**：
- 前两轮文档提到了"pre-allocation"，但没有解释 **in-place 修改的语义**。
- 如果没有理解这一点，阅读 `overlap.py` 时会困惑："为什么 sometimes 赋值 args，sometimes 赋值 outs，有时候都赋值为 None？"

**合并 (merge) 的检测逻辑**：
```python
# 检测 stage5 和 stage1 是否被合并
stage1_record = intermediate_tensors1.layers[-l].stage1
use_merged = (
    hasattr(stage1_record, "outs")
    and stage1_record.outs is not None
    and not (hasattr(stage1_record, "args") and stage1_record.args is not None)
)
```
- **合并时**：stage5 只存 `args`（不存 `outs`），stage1 只存 `outs`（不存 `args`）。
- **未合并时**：stage5 和 stage1 都存完整的 `args` + `outs`。
- 这种 **asymmetric None pattern** 是检测合并的关键。

---

### 2.3 Zero Bubble 的实际触发条件

**新学到的点**：

```python
# pithtrain/dualpipe/dualpipev.py:step() Step 6
enable_zb = False
for i in range(step_6):
    if i == step_6 // 2 and pp_rank % 2 == 1:
        enable_zb = True
    if i == step_6 // 2 and pp_rank % 2 == 0:
        enable_zb = True
    self._backward_chunk(1, enable_zb=enable_zb)
    # ...
```

- Zero Bubble 不是全程开启，而是在 Step 6 的 **后半段** 才启用。
- 触发条件：`i == step_6 // 2`（中间点），并且 **odd PP ranks 先启用，even PP ranks 后启用**。
- 为什么分奇偶？因为 V 形布局的对称性，odd 和 even rank 的 backward chunk 可用性不同。

**之前文档的遗漏**：
- 前两轮文档提到"Zero Bubble 把 wgrad 延迟到 idle 时间片"，但没有解释 **具体在哪个 step 启用**。
- 这个细节解释了为什么 PithTrain 的 bubble 比标准 1F1B 小。

---

## 三、算法层：MoE 的实际行为

### 3.1 `k=1` 时 EP Dispatch 的快速路径

**新学到的点**：

```python
# pithtrain/operators/ep_dispatch.py:moe_ep_prepare_dispatch
if ep_size == 1:
    dedup_sorted_tokens = (
        hidden_states.unsqueeze(1).expand(-1, k, -1).reshape(-1, hidden_states.shape[-1])
    )
    return (dedup_sorted_tokens, None, None, None, None, None, None, None)
```

- 当 `ep_size == 1` 时（无 EP），直接 `expand(-1, k, -1)` 复制 k 次，跳过所有 dedup 逻辑。
- 当 `ep_size > 1` 且 `k=1`（如 Qwen3-30B-A3B）：每个 token 只去 1 个 expert，必然只去 1 个 EP rank。
  - **Dedup 理论上没有收益**（不会重复发送到同一个 rank）。
  - 但当前代码 **仍然执行 dedup 逻辑**（`fused_dedup_prepare_dispatch`）。
  - 这是一个可以优化的点！

**实际影响**：
- Qwen3-30B-A3B (k=1, EP=8) 的 EP dispatch 有 overhead 但无收益。
- 如果能在 `k=1` 时跳过 dedup，可以省掉 3 个 Triton kernel + 2 次 all-to-all。

---

### 3.2 Load Balance Loss 的三种粒度的实际效果

**新学到的点**：

| 类型 | 同步范围 | 积累方式 | 适用场景 | PithTrain 默认 |
|------|---------|---------|---------|---------------|
| micro-batch | 无 | 每 micro-batch 独立 | 单机、小 batch | DeepSeek-V2-Lite |
| global-batch | DP×EP all-reduce | 梯度累积步内累加 | 多机、大 batch | Qwen3-30B-A3B |
| sequence | CP all-reduce | 按 sequence 独立计算 | 长 context | 可选 |

**实际观察**：
- DeepSeek-V2-Lite 官方用 `sequence` 类型（因为论文原文如此）。
- Qwen3-30B-A3B 官方用 `global-batch` 类型（因为数据并行更常见）。
- 在单机 8 卡（DP=1）下，`global-batch` 和 `micro-batch` 效果接近（因为没有其他 rank 需要同步）。

**之前的遗漏**：
- 前两轮文档列出了三种类型，但没有解释 **为什么不同模型选择不同类型**。
- 实际选择取决于：是否有 DP（影响 all-reduce 范围）、是否用 CP（影响 sequence 定义）。

---

## 四、工程层：从"能跑"到"能调试"的细节

### 4.1 FSDP + Pipeline 的 post_backward 手动触发

**新学到的点**：

```python
# pithtrain/dualpipe/dualpipev.py:step() 末尾
def run_post_backward(fsdp_module: FSDPModule) -> None:
    fsdp_module.set_is_last_backward(True)
    fsdp_module.set_reshard_after_backward(True)
    fsdp_module.set_requires_gradient_sync(True)
    fsdp_state = fully_shard.state(fsdp_module)
    for state in fsdp_state._state_ctx.all_states:
        if state._fsdp_param_group:
            # 关键：先 accumulate，再 to_accumulated
            for fsdp_param in state._fsdp_param_group.fsdp_params:
                if hasattr(fsdp_param, "_unsharded_param"):
                    fsdp_param.accumulate_unsharded_grad_if_needed()
                    fsdp_param.to_accumulated_grad_if_needed()
            state._fsdp_param_group.post_backward()
    fsdp_state._root_post_backward_final_callback()
```

**关键细节**：
1. `accumulate_unsharded_grad_if_needed()` **必须在** `to_accumulated_grad_if_needed()` **之前调用**。
   - 原因：`to_accumulated_grad_if_needed()` 会把 fp16/bf16 梯度累加到 fp32 accumulator。
   - 如果先 `to_accumulated`，再 `accumulate`，会导致 dtype 不匹配。
2. FSDP 的 `post_backward` 做了 **梯度规约**（all-reduce across DP ranks）。
3. `_root_post_backward_final_callback()` 是真正的 "sync point"，等待所有 rank 的梯度就绪。

**之前文档的遗漏**：
- 前两轮文档提到"手动调用 post_backward"，但没有解释 **wgrad dtype 转换的顺序问题**。
- 这个细节是 PithTrain 能在 Pipeline 下正确使用 FSDP 的关键。

---

### 4.2 Checkpoint 的 Canonical vs Localized 格式的实际转换

**新学到的点**：

```
Localized (运行时):
  module.0.layers.1.mlp.experts.gate_proj.weight  → shape [experts_per_rank, ...]

Canonical (磁盘):
  layers.1.mlp.experts.3.gate_proj.weight          → shape [..., ...] (单个 expert)
```

**转换逻辑的关键细节**：
1. **保存时**（`to_canonical_model`）：
   - Strip `module.{N}.` prefix
   - 对 stacked expert tensor，展开成 `experts.0.weight, experts.1.weight, ...`
   - 每个 rank 只保存自己拥有的 expert（通过 `unwrap_dtensor_experts` 提取 local shard）

2. **加载时**（`to_localized_model`）：
   - Add `module.{N}.` prefix 回来
   - 把单个 expert weight 组合回 stacked tensor
   - 重新映射到当前 PP/EP 配置

**Resharding 的魔法**：
- 因为磁盘格式是 **PP-independent**（用全局 layer ID），所以：
  - 可以在 PP=1 下保存
  - 在 PP=2 下加载
  - 框架自动处理 layer 的分配

**之前文档的遗漏**：
- 前两轮文档描述了 canonical vs localized，但没有解释 **具体是怎么 strip prefix 的**。
- 实际代码中用了正则表达式：`MODULE_PREFIX_RE = re.compile(r"^module\.\d+\.")`

---

### 4.3 Muon + AdamW 混合优化器的 dtype 问题

**新学到的点**：

```python
# pithtrain/modules/optimizer.py
class Muon(Optimizer):
    def step(self, closure=None):
        # ...
        orth = zeropower_via_newtonschulz5(update, steps=5).to(p.dtype)
        # p.dtype 是 bf16，但 orth 是 fp32（Newton-Schulz 在 fp32 下做）
        # 更新时：
        p.data.mul_(1 - lr * wd).add_(orth, alpha=-lr * scale_factor)
        # p.data 是 bf16，但乘法/加法会自动 cast
```

**关键细节**：
- Muon 的 momentum buffer 是 **FP32**，即使参数是 BF16。
- Newton-Schulz 正交化在 FP32 下做（数值更稳定）。
- 更新结果 cast 回 BF16 存入 `p.data`。
- 这意味着：**Muon 优化的参数，其 fp32 momentum 和 bf16 param 之间存在隐式的 dtype 转换**。

**AdamW 的对比**：
- AdamW 的 `exp_avg` 和 `exp_avg_sq` 也是 FP32。
- 但 AdamW 的更新公式中，`param - lr * (exp_avg / (sqrt(exp_avg_sq) + eps) + wd * param)` 也是 FP32 计算后 cast 回 BF16。

**之前文档的遗漏**：
- 前两轮文档提到"Muon 的 momentum buffer 是 FP32"，但没有解释 **为什么这不会导致性能问题**。
- 实际原因是：BF16 参数的更新是通过 `p.data.mul_()` 和 `add_()` 进行的，PyTorch 会自动处理 dtype 转换，但需要确保数值范围在 BF16 可表示范围内。

---

## 五、对比层：PithTrain vs Megatron-LM 的隐性差异

### 5.1 并行策略的命名差异

| 概念 | PithTrain | Megatron-LM | 备注 |
|------|-----------|-------------|------|
| Pipeline Parallel | PP | PP | 相同 |
| Expert Parallel | EP | EP | 相同 |
| Context Parallel | CP | CP | 相同 |
| Data Parallel | DP (via FSDP2) | DP (via DDP/FSDP) | PithTrain 用 FSDP2，Megatron 有多种 |
| Tensor Parallel | ❌ 不支持 | TP (row/column) | PithTrain 没有 TP |

**新学到的点**：
- PithTrain **不支持 Tensor Parallel (TP)**。
- 这意味着在单卡内，如果模型太大，PithTrain 无法像 Megatron 那样用 TP 切分。
- 8×4090 上，如果 PP=1 放不下模型，PithTrain 只能靠 PP/EP/CP，而 Megatron 还可以加 TP。

**影响**：
- PithTrain 的适用场景更窄：需要 **EP 友好的模型**（MoE）。
- Dense 模型在 PithTrain 上无法用 TP，只能靠 PP，内存效率更低。

---

### 5.2 MoE 实现的差异

**Megatron-LM 的 MoE**：
- `megatron/core/extensions/moe/` 下有完整的 MoE 实现
- 支持 `ExpertParallelGroup`、`AllToAll`、`TokenDrop`、`LoadBalanceLoss`
- 有 **capacity factor**（限制每个 expert 最多接收多少 tokens）

**PithTrain 的 MoE**：
- 没有 capacity factor（理论上 expert 可以接收任意多 tokens）
- 用 token dedup 减少通信
- 用 `MoELoadBalanceLossInjector` 做 straight-through estimator

**新学到的点**：
- Megatron 的 MoE 更 "production-ready"（有 capacity factor、 Expert 的 capacity 约束）。
- PithTrain 的 MoE 更 "minimal"（没有 capacity factor，依赖 load balance loss 间接控制）。
- 这意味着：如果 MoE router 非常不平衡，PithTrain 可能出现某个 expert 接收过多 tokens 导致 OOM。

---

### 5.3 优化器的差异

**Megatron-LM**：
- 主要用 Adam + 变体（AdamW, DistributedAdam）
- 支持 Muon（通过 Emerging-Optimizers 库，但非默认）
- 有 **distributed checkpoint** 的 optimizer state 管理

**PithTrain**：
- 默认用 **Muon + AdamW 混合优化器**
- Muon 优化 2D 权重，AdamW 优化其余参数
- 这是一个 **research-oriented** 的选择，不是 production-oriented

**新学到的点**：
- Muon 的 `zeropower_via_newtonschulz5` 是一个 **iterative orthogonalization**，不是 SVD。
- 5 次 NS 迭代的精度：对 LLM 的权重矩阵（奇异值谱相对集中）足够。
- 但 Megatron 没有默认集成 Muon，说明 **Muon 的生产成熟度不如 AdamW**。

---

## 六、复现过程中的"踩坑"记录（待填充）

> 这部分在 actual reproduction 过程中逐步填充。

### 坑 1：torch.compile 在 4090 上的兼容性

**现象**：
```
torch._dynamo.exceptions.Unsupported: call_function <built-in function add> ...
```

**原因**：
- `torch.compile(fullgraph=True)` 在 4090 (SM89) 上可能不如 SM90 稳定。
- 某些 Triton kernel 的 pattern 在 4090 上触发 graph break。

**解决方案**：
- 临时关闭 `fullgraph`（允许 graph break）
- 或者跳过 problematic op 的编译

### 坑 2：NCCL 在 4090 上的配置

**现象**：
```
RuntimeError: NCCL WARN Cuda error: system not initialized
```

**原因**：
- 4090 是 consumer GPU，NCCL 的默认配置可能不最优。
- 需要设置 `NCCL_IB_DISABLE=1`（如果没有 InfiniBand）。

**解决方案**：
```bash
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=2
```

---

## 七、总结：前两轮文档 vs 本轮增量

| 维度 | 前两轮文档 | 本轮增量 |
|------|-----------|---------|
| **硬件层** | 假设 SM90+ | 讨论 4090 (SM89) 的适配 |
| **内存层** | 定性描述 | 定量估算公式 + 24GB 实际规划 |
| **调度层** | 8-step 算法描述 | num_chunks >= 2*PP 的 assert、Zero Bubble 触发条件 |
| **数据流层** | IntermediateTensors 概念 | in-place 修改语义 + merge detection logic |
| **算法层** | MoE 通用原理 | k=1 时 dedup 无收益、三种 lb loss 的实际选择 |
| **工程层** | FSDP 手动触发 | wgrad dtype 转换顺序、canonical/localized 正则转换 |
| **对比层** | 无 | PithTrain vs Megatron 的隐性差异（无 TP、无 capacity factor、Muon vs AdamW） |
| **踩坑层** | 无 | 4090-specific issues（torch.compile、NCCL） |

---

## 八、下一步行动

1. **立即执行**：Phase 0（环境准备 + 10-step smoke test）
2. **记录踩坑**：每遇到一个问题，更新"踩坑记录"部分
3. **迭代优化**：每个 phase 完成后，更新本文件的"增量"部分
4. **对比验证**：Megatron-LM 安装完成后，开始 Phase 4
