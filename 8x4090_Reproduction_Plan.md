# PithTrain 8×4090 完整复现计划

> **硬件 Reality Check**：8×RTX 4090 (24GB, SM89)
> **核心约束**：PithTrain 官方要求 Hopper (SM90) / Blackwell (SM100)，4090 的 SM89 **不支持 FP8 Tensor Core**，且显存仅 24GB/卡。
> **策略**：以 **BF16-only、内存优先、精度可验证** 为原则，在 4090 上最大程度还原 PithTrain 的核心设计与训练流程，不追求吞吐对齐，追求**功能完整性 + 数值正确性**。

---

## 一、硬件约束与可行性评估

### 1.1 4090 的关键限制

| 限制项 | 4090 实际情况 | PithTrain 官方要求 | 影响 |
|--------|--------------|------------------|------|
| **SM 版本** | 8.9 (Ada Lovelace) | 9.0 (Hopper) / 10.0 (Blackwell) | ❌ 不支持 FP8 Tensor Core |
| **显存** | 24 GB × 8 = 192 GB | 80 GB × 8 (H100) | ⚠️ 需要更激进的 sharding |
| **FP8 支持** | ❌ 无 | ✅ deep-gemm 后端必需 | 必须走 BF16 路径 |
| **NVLink** | ✅ 有 (PCIe 5.0 + NVLink 3.0) | - | 通信可用，但带宽 < H100 |
| **torch.compile** | ✅ 支持 | - | 可以验证 fullgraph 行为 |

### 1.2 模型选择：从 Qwen3-30B-A3B 降级到 DeepSeek-V2-Lite

| 模型 | 总参数量 | 激活参数量 | 官方 8×H100 | 预估 8×4090 可行性 |
|------|---------|-----------|------------|------------------|
| Qwen3-30B-A3B | 30B | 3B | ❌ OOM (需 H200/B200) | ❌ **不可行** |
| GPT-OSS-20B | 21B | 3.6B | ✅ 可跑 | ⚠️ 边缘可行 (需 PP=2) |
| GPT-OSS-120B | 117B | 5B | ❌ 需 32 卡 | ❌ 不可行 |
| **DeepSeek-V2-Lite** | **16B** | **2.4B** | ✅ 可跑 (PP=1, EP=8) | ✅ **主攻目标** |

**选择 DeepSeek-V2-Lite 的理由**：
1. 参数量最小（16B / 2.4B 激活），在 24GB 卡上内存压力最小
2. 官方配置明确支持单节点 8 卡
3. 包含完整的 MoE + MLA (Multi-head Latent Attention) 特性，能覆盖 PithTrain 的核心优化点
4. 有公开的 HF checkpoint 可用于验证正确性

### 1.3 并行策略调整（针对 4090）

```
原始 PithTrain 官方配置 (DeepSeek-V2-Lite, 8×H100):
  PP=1, EP=8, CP=1, DP=1

8×4090 调整后配置:
  PP=1, EP=8, CP=1, DP=1  ← 先试这个
  如果 OOM → PP=2, EP=4, CP=1, DP=1
  如果还 OOM → PP=2, EP=4, CP=2, DP=1 (但 2×1×4×2=16 > 8，不行)
  实际可行组合:
    - PP=1, EP=8, CP=1, DP=1  (8 = 1×1×8×1) ← 首选
    - PP=2, EP=4, CP=1, DP=1  (8 = 2×1×4×1) ← 备选
    - PP=1, EP=4, CP=2, DP=1  (8 = 1×2×4×1) ← 如果 sequence 很长
```

**内存估算 (DeepSeek-V2-Lite, BF16)**：
- 模型权重：~16B × 2 bytes = 32 GB
- 分到 8 卡 EP=8：每卡 expert 权重 ≈ 4 GB (16B/8/2，MoE 层参数减半)
- 非 MoE 参数（attention, embedding, norm）FSDP shard 到 (DP, CP, EP) = (1, 1, 8)
- Optimizer states (Muon + AdamW)：FP32 momentum ≈ 参数量的 2-3×
- 激活内存：sequence=2048, batch=1 → 约 4-6 GB/卡

**预估峰值内存**：约 18-22 GB/卡（接近 24GB 上限，需要监控）

---

## 二、复现阶段划分

### Phase 0：环境准备（1-2 天）

**目标**：在 8×4090 上把 PithTrain 跑起来，哪怕只能跑 1 步

**步骤**：

1. **修改 PithTrain 源码以适配 4090**
   - 移除 SM90/SM100 的硬性检查（如果有）
   - 强制 `fp8_training = "disabled"`
   - 确保 `torch.compile(fullgraph=True)` 在 4090 上正常工作

2. **环境搭建**
   ```bash
   # 创建虚拟环境
   uv venv
   uv sync
   
   # 验证 CUDA
   python -c "import torch; print(torch.cuda.get_device_capability())"
   # 应该输出 (8, 9)
   
   # 验证 deep_gemm 不可用（4090 不支持）
   python -c "import deep_gemm" 2>&1 || echo "expected: not available"
   ```

3. **数据准备**
   - 下载 DCLM 数据集的子集（或合成小规模数据用于 smoke test）
   - Tokenize：`bash examples/tokenize_corpus/launch.sh dclm-deepseek-v2`
   - **注意**：如果 DCLM 太大，先用随机 token 的 synthetic data 验证流程

4. **最小化配置验证**
   - 修改 `examples/pretrain_lm/deepseek-v2-lite/script.py`：
     ```python
     training.max_steps = 10        # 只跑 10 步
     training.micro_batch_size = 1
     training.global_batch_size = 8
     training.sequence_length = 1024  # 先从 1024 开始
     training.fp8_training = "disabled"
     distributed.pipeline_parallel_size = 1
     distributed.expert_parallel_size = 8
     distributed.context_parallel_size = 1
     ```
   - 启动训练：
     ```bash
     bash examples/pretrain_lm/launch.sh deepseek-v2-lite
     ```

5. **成功标准**
   - 训练能跑完 10 步，loss 从 ~12 降到 ~10
   - 没有 NaN、没有 OOM
   - 每步能正确保存 checkpoint

---

### Phase 1：正确性验证（2-3 天）

**目标**：证明在 4090 BF16 下，PithTrain 的训练数值是正确的

**步骤**：

1. **Loss Curve 对齐**
   - 用 PithTrain 官方提供的 DeepSeek-V2-Lite checkpoint 初始化
   - 在 4090 上跑 100 steps，记录 loss curve
   - 和论文/官方报告的 loss curve 对比（如果有）
   - 和 HF 原生实现的前 100 steps 对比

2. **Reference Forward 验证**
   - 利用 `ModelImplMode.use_reference_fwd = True`
   - 对比优化路径和 reference 路径的 output 差值
   - 确保在 BF16 下，diff < 1e-3

3. **Checkpoint 转换验证**
   - 训练 256 steps 后保存 checkpoint
   - 用 `convert_checkpoint` 导出到 HF 格式
   - 用 HF 的 `AutoModelForCausalLM` 加载，做一次 forward
   - 对比 logits 是否一致

4. **测试套件运行**
   ```bash
   # 单卡单元测试
   pytest tests/test_silu_mul.py -v
   pytest tests/test_layer_partition.py -v
   pytest tests/test_ep_dedup_dispatch.py -v
   pytest tests/test_grouped_linear_correctness.py -v
   pytest tests/test_fp8_quantize_kernels.py -v  # 应该 skip（FP8 不可用）
   ```

---

### Phase 2：内存优化与配置调优（3-5 天）

**目标**：在 24GB 显存限制下，找到最优的并行/ batch/ seq 配置

**步骤**：

1. **Memory Profiling**
   ```python
   # 在 script.py 中启用
   training.memory_profile_start = 1
   training.memory_profile_stop = 2
   ```
   - 生成 snapshot-rank*.pickle
   - 导入 https://pytorch.org/memory_viz 分析
   - 找出内存瓶颈：是权重？激活？优化器 states？

2. **配置网格搜索**
   | PP | EP | CP | DP | micro_bs | seq_len | 预估可行性 |
   |----|----|----|----|---------|---------|----------|
   | 1 | 8 | 1 | 1 | 1 | 1024 | ✅ 首选 |
   | 1 | 8 | 1 | 1 | 1 | 2048 | ⚠️ 需要验证 |
   | 1 | 8 | 1 | 1 | 2 | 1024 | ⚠️ 需要验证 |
   | 2 | 4 | 1 | 1 | 1 | 1024 | ✅ 备选 |
   | 2 | 4 | 1 | 1 | 1 | 2048 | ⚠️ 需要验证 |
   | 1 | 4 | 2 | 1 | 1 | 4096 | ⚠️ CP=2 需要测试 |

3. **优化策略**
   - 如果 OOM：
     a) 减小 `micro_batch_size`（从 2 降到 1）
     b) 减小 `sequence_length`（从 2048 降到 1024）
     c) 增大 PP（从 1 到 2）
     d) 减小 EP（从 8 到 4，增加 per-rank expert 数量）
   - 如果吞吐太低：
     a) 增大 `micro_batch_size`（如果内存允许）
     b) 增大 `global_batch_size`（增加 accumulate_steps）

4. **Gradient Checkpointing（如果需要）**
   - PithTrain 当前未实现 activation checkpointing
   - 如果需要，手动添加：在 `decoder_layer_forward` 中，选择性不保存中间激活
   - Trade-off：20-30% 额外 compute 换 30-50% 激活内存

---

### Phase 3：核心模块逐个攻破（4-6 天）

**目标**：深入理解并验证 PithTrain 的每个核心模块

**模块清单与验证方法**：

| 模块 | 源码位置 | 验证方法 | 成功标准 |
|------|---------|---------|---------|
| **Model Protocol** | `models/interface.py` | 阅读 + 写一个最小 DecoderLayer | 理解 5-stage split |
| **DualPipeV Scheduler** | `dualpipe/dualpipev.py` | 单步 trace + nsight profile | 能解释 8 步调度 |
| **F/B Overlap Loop** | `dualpipe/overlap.py` | 在 Step 4 加 print 验证 overlap | 理解 compute-comm interleave |
| **5-Stage Execution** | `dualpipe/execution.py` | 对比 stage1_f vs stage5_f 的输出 | 理解 IntermediateTensors |
| **EP Dispatch** | `operators/ep_dispatch.py` | 跑 `test_ep_dedup_dispatch.py` | Triton kernel 和 PyTorch ref 一致 |
| **Ring Attention** | `operators/ring_attention.py` | CP=2 跑 1 step，对比 CP=1 的 loss | zigzag layout 正确 |
| **FP8 Quantize** | `operators/deepgemm_fp8_quantize.py` | 应该 skip（4090 不支持） | 确认 graceful skip |
| **Load Balance** | `modules/load_balance.py` | 打印 per-expert token count | 3 种 lb loss 都能跑 |
| **Muon Optimizer** | `modules/optimizer.py` | 跑 `test_muon.py` | Newton-Schulz 正确 |
| **Checkpointing** | `modules/checkpoint.py` | 保存 → 加载 → 继续训练 | loss 连续不断点 |
| **FSDP Integration** | `modules/training.py:apply_fsdp` | 检查 shard 后的参数名 | MoE/non-MoE 分开 shard |

---

### Phase 4：与 Megatron-LM 对比实验（3-5 天）

**目标**：在相同硬件/模型/数据下，对比 PithTrain 和 Megatron-LM 的行为

**步骤**：

1. **安装 Megatron-LM**
   ```bash
   cd /data/home/yizhou/Megatron-LM
   uv pip install -e .
   # 可能需要额外的 CUDA 扩展编译
   ```

2. **配置 Megatron-LM 跑 DeepSeek-V2-Lite**
   - 找到 Megatron 的 DeepSeek-V2 配置
   - 调整为：PP=1, EP=8, CP=1, DP=1
   - BF16，无 FP8

3. **对比维度**
   | 维度 | PithTrain | Megatron-LM | 对比方法 |
   |------|-----------|-------------|---------|
   | 吞吐 (tok/s) | 待测 | 待测 | 相同 100 steps，取 median |
   | 峰值内存 | 待测 | 待测 | torch.cuda.max_memory_allocated |
   | Loss curve | 待测 | 待测 | 相同 seed，前 1000 steps |
   | 代码行数 | ~11K | ~160K | cloc |
   | 启动时间 | 待测 | 待测 | time python script.py |
   | 调试难度 | 待测 | 待测 | 主观评分 1-10 |

4. **分析差异**
   - 如果 PithTrain 更慢：是 bubble 更大？还是 kernel 效率低？
   - 如果 PithTrain 更耗内存：是 IntermediateTensors 预分配？还是 FSDP shard 策略不同？
   - 如果 loss curve 不一致：是初始化不同？还是 MoE load balance 策略不同？

---

### Phase 5：扩展功能探索（2-3 天，可选）

**目标**：在复现基础上，尝试修改/扩展框架

**可选方向**：

1. **添加新模型：Qwen3-MoE（更小版本）**
   - 用 `qwen3_moe.py` 作为模板
   - 修改 config.json 为更小的模型（如 7B-A2B）
   - 跑通 pretrain

2. **实现 Gradient Checkpointing**
   - 在 `dualpipe/modeling.py` 中，选择性丢弃 intermediate activations
   - 在 backward 时重算
   - 验证内存减少和速度增加的 trade-off

3. **实现 SFT 任务**
   - 复用 `pretrain_lm.py` 的结构
   - 修改 dataset 为 instruction format
   - 调整 lr 和 max_steps

4. **实现简单的 RLHF（GRPO）**
   - 在 PithTrain 上实现 generation loop
   - 用 GRPO 的 group relative advantage
   - 验证 MoE router 在 RL 下的稳定性

---

## 三、关键里程碑与检查点

| 里程碑 | 预计时间 | 检查标准 | 失败回退 |
|--------|---------|---------|---------|
| M0: 10-step smoke test | Day 2 | loss 下降，无 NaN | 检查 CUDA 版本、torch 版本 |
| M1: 100-step 正确性 | Day 5 | loss 平滑下降，和 HF 一致 | 检查 seed、初始化、数据 |
| M2: 256-step 稳定训练 | Day 8 | checkpoint 可保存/加载 | 检查 DCP 格式 |
| M3: 内存优化完成 | Day 13 | peak mem < 22GB | 减小 batch/seq/PP |
| M4: Megatron 对比完成 | Day 18 | 完成 3 维度对比 | 调整 Megatron 配置 |
| M5: 扩展功能完成 | Day 21 | 至少完成 1 个可选方向 | 降低目标 |

---

## 四、风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| **OOM（内存不足）** | 高 | 无法训练 | 1. 减小 seq_len/micro_bs；2. 增大 PP；3. 减小 EP |
| **PithTrain 不支持 4090** | 中 | 代码无法运行 | 1. 修改源码移除 SM 检查；2. 强制 BF16 |
| **Megatron-LM 编译失败** | 中 | 无法对比 | 1. 用 Docker 镜像；2. 降低编译并行度 |
| **数据准备时间过长** | 中 | 无法开始训练 | 1. 用 synthetic data；2. 下载 DCLM 子集 |
| **DeepSeek-V2-Lite 权重不可用** | 低 | 无法验证 | 1. 用 HF 官方权重；2. 从 random init 跑 |
| **torch.compile 在 4090 上有问题** | 中 | 性能下降 | 1. 关闭 fullgraph；2. 用 eager mode |

---

## 五、学习路径（跟着 plan 学）

这个 plan 本身就是一个学习路径。每个 phase 对应的学习重点：

1. **Phase 0**：理解 PithTrain 的启动流程（`launch` → `distributed_context` → `training_context`）
2. **Phase 1**：深入 Model Protocol 和 5-stage decomposition
3. **Phase 2**：理解内存分布（权重、激活、优化器 states）
4. **Phase 3**：逐个攻克核心模块，写 notes
5. **Phase 4**：对比 PithTrain 和 Megatron 的设计哲学差异
6. **Phase 5**：尝试修改框架，理解扩展点

---

## 六、参考资料

- PithTrain 源码：`/data/home/yizhou/pith-train/`
- PithTrain 论文：`/data/home/yizhou/pith-train/paper/`
- PithTrain 面试准备：`/data/home/yizhou/pith-train/PithTrain_面试深挖准备.md`
- Megatron-LM 源码：`/data/home/yizhou/Megatron-LM/`
- DeepSeek-V2-Lite HF：`deepseek-ai/DeepSeek-V2-Lite`
- DCLM 数据集：`https://huggingface.co/datasets/mlfoundations/dclm`
