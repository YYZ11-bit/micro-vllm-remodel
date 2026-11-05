# 为什么当前不接入官方 FlashAttention

- 日期：2026-11-05
- 结论：**保留现有 `triton_paged` 作为正式 fused attention 后端，不引入官方
  `flash-attn` 依赖；本文不反对继续优化 attention kernel，只说明“再接一层
  FlashAttention 库”在当前目标上没有足够收益。**
- 影响：投机解码不需要修改 attention 后端；现有算子已经支持一个请求多行
  query（chunked prefill 与投机验证共用同一条路径）。

---

## 1. 现在 attention 在哪里被调用

### 1.1 选择入口

用户通过 `ModelConfig.attention_operator` 选择后端，默认是 `"reference"`：

[config.py:58](../src/micro_vllm/config.py#L58)

```python
attention_operator: str = "reference"   # "reference" | "sdpa" | "triton_paged"
```

模型构造时把名字传给 `Attention`。四个模型的做法一致：

- Qwen3：[qwen.py:193](../src/micro_vllm/models/qwen.py#L193)
- Qwen2：[qwen2.py:197](../src/micro_vllm/models/qwen2.py#L197)
- Llama：[llama.py:176](../src/micro_vllm/models/llama.py#L176)
- Mistral：[mistral.py:45](../src/micro_vllm/models/mistral.py#L45)

`Attention` 在构造时通过 registry 拿到算子类：

[attention.py:47-54](../src/micro_vllm/attention/attention.py#L47-L54) →
[registry.py:24-51](../src/micro_vllm/attention/operators/registry.py#L24-L51)

registry 对 `triton_paged` 做 lazy import，因此默认的 `reference` 路径不要求
安装 Triton。新增后端只需要在这里注册一个名字。

### 1.2 执行入口

完整调用链如下（以 Qwen3 为例，其余模型相同）：

```text
LLMConfig.model_config.attention_operator
  -> Model.__init__: Attention(spec, name)              # 每层共享同一个实例
  -> ModelRunner.execute(plan)                          # worker/model_runner.py:192
       -> _flatten_model_input(plan)                    # 拼 batch 张量
            -> Attention.build_metadata(...)            # 每个 cache group 一次
       -> model(input_ids, positions, layer_kv_caches, layer_metadata)
            -> DecoderLayer.forward
                 -> AttentionLayer.forward              # q/k/v proj + RoPE
                      -> Attention.__call__              # attention.py:87-102
                           -> AttentionOperator.forward  # 真正算子
```

关键位置：

| 环节 | 文件 | 说明 |
|---|---|---|
| 配置入口 | [config.py:58](../src/micro_vllm/config.py#L58) | `attention_operator` |
| 构造与注册 | [attention.py:47-54](../src/micro_vllm/attention/attention.py#L47-L54)、[registry.py:24-51](../src/micro_vllm/attention/operators/registry.py#L24-L51) | 名字 → 算子类 |
| 元数据构建 | [model_runner.py:580-653](../src/micro_vllm/worker/model_runner.py#L580-L653) | 扁平化 batch、位置、query 起点、context 长度 |
| 元数据注入 | [model_runner.py:629-639](../src/micro_vllm/worker/model_runner.py#L629-L639) | 每个 KV cache group 构建一次 |
| 模型执行 | [model_runner.py:192-238](../src/micro_vllm/worker/model_runner.py#L192-L238) | `ModelRunner.execute` |
| 层内调用 | [qwen2.py:64-94](../src/micro_vllm/models/qwen2.py#L64-L94) | Q/K/V → RoPE → attention → O |
| 算子分派 | [attention.py:87-102](../src/micro_vllm/attention/attention.py#L87-L102) | `Attention.__call__` |
| 算子契约 | [base.py:10-40](../src/micro_vllm/attention/operators/base.py#L10-L40) | `AttentionOperator.forward` |

### 1.3 算子拿到的全部输入

`AttentionMetadata` 是算子与引擎之间唯一的运行时契约：

[attention.py:12-28](../src/micro_vllm/attention/attention.py#L12-L28)

```text
query_positions                 [T]      每个 query token 的绝对位置
query_context_start_positions   [T]      每个 query 可见区间的起点
query_start_locations           [R+1]    每个请求的 query 在扁平 batch 中的区间
context_lengths                 [R]      每个请求的 context 总长度（含本次 token）
block_table                     [R, B]   逻辑块 -> 物理块
slot_mapping                    [T]      当前 K/V 写入的物理 slot
block_size                      int
```

物理 KV cache 的形状是：

```text
[2, num_blocks, block_size, num_key_value_heads, head_dim]
```

分配见 [model_runner.py:428-443](../src/micro_vllm/worker/model_runner.py#L428-L443)，
按层绑定见 [model_runner.py:447-462](../src/micro_vllm/worker/model_runner.py#L447-L462)，
位置到物理 slot 的映射见 [attention.py:31-41](../src/micro_vllm/attention/attention.py#L31-L41)
和 [model_runner.py:655+](../src/micro_vllm/worker/model_runner.py#L655)。

可见性语义由 `AttentionSpec` 决定，不属于 kernel：

- `FullAttentionSpec`：`context_start = 0`
- `SlidingWindowSpec`：`max(0, pos - window + 1)`
- `ChunkedLocalSpec`：按 chunk 对齐的局部前缀

见 [spec.py:18-51](../src/micro_vllm/attention/spec.py#L18-L51)。

### 1.4 一个容易忽略的实现细节

模型只为所有层创建一个 `Attention` 实例：

[qwen.py:193-195](../src/micro_vllm/models/qwen.py#L193-L195)

```python
attention = Attention(FullAttentionSpec(), config.attention_operator)
self.layer_attentions = (attention,) * num_hidden_layers
```

当前三个算子都是无状态的，所以共享实例没有问题。**以后手写算子若需要
per-layer workspace / scratch buffer，必须改成每层一个实例**，否则会互相覆盖。

---

## 2. 现有三个后端与实测

以下数据来自 operator-level spike：RTX 5090、torch 2.8.0+cu128、Triton 3.4.0，
context 512，prefill query length 512 / decode query length 1，16 个 Q head、
8 个 KV head、head_dim 128、block_size 16。测量的是**完整路径**（包含 K/V 写入）：

| 后端 | 实现 | 是否 gather KV | 设备/精度 | prefill | decode | 相对 reference |
|---|---|---|---:|---:|---:|---:|
| `reference` | 逐请求 gather + 显式 softmax | 是 | CPU/CUDA，fp16/bf16/fp32 | 0.3762 ms | 0.4055 ms | 1.00× |
| `sdpa` | gather 后调 `F.scaled_dot_product_attention` | 是 | CPU/CUDA，跟随输入 dtype | 0.2753 ms | 0.3320 ms | 1.37× / 1.22× |
| `triton_paged` | 融合写 KV + 分块 online softmax，直接读 paged cache | **否** | 仅 CUDA fp16 | **0.0773 ms** | **0.0808 ms** | **4.87× / 5.02×** |

数据：[triton-paged-comparison-rtx5090-run-2.json](../benchmarks/results/spikes/triton-paged-comparison-rtx5090-run-2.json)。

两个要点：

1. **`sdpa` 不是 FlashAttention 后端。** 当前 SDPA 路径总是传显式的非空
   mask，PyTorch 会选 memory-efficient CUTLASS kernel；spike 中 flash backend
   明确因 “Flash Attention does not support non-null attn_mask” 不可用。
2. `triton_paged` 的优势主要来自**省掉 gather**。在 decode 的 breakdown 里，
   gather 占 SDPA 完整路径 238 µs 中的约 81 µs（34%），真正的 SDPA kernel 只有
   约 58 µs。triton 路径把写 KV 和 attention 都放在 paged layout 上完成。

---

## 3. 不接入官方 FlashAttention 的理由

### 3.1 它要对比的基线是 `triton_paged`，不是 `reference`

`triton_paged` 已经实现了 FlashAttention 式的分块 online softmax，并且直接读
paged KV。官方 FA2 的 paged 路径做的是同一件事；即使更快，可提升的空间也只是
在 0.08 ms 这个量级上再压缩，而不是从 0.4 ms 降到 0.08 ms。

### 3.2 page size 对齐是硬约束

flash-attn 2.x 的 paged KV kernel 通常要求 page size 是 **256 的倍数**，而当前
`block_size=16`。只有三条路：

- 为 FA 单独用 `block_size=256`：改变 prefix cache 粒度、短请求碎片和 benchmark
  对比；
- 先把 16-token block gather 成 FA 页：多一次拷贝，正好丢掉 FA 相对 gather
  baseline 的优势；
- 在 block table / allocation 层做 16→256 的映射：复杂度最高，且要同时改调度和
  cache。

这三条都不是“接一个库”能解决的，必须先用目标 wheel 做最小验证。若 256 对齐
不成立，FA 只能作为非 paged 的实验路径，无法替代 `triton_paged`。

### 3.3 wheel 与运行环境耦合

`flash-attn` 的 wheel 强依赖 PyTorch / CUDA / GPU 架构：

- 本机 torch 2.8+cu128、RTX 5090（sm120）；
- AutoDL 实例 torch 2.4+cu121、RTX 3080（sm86）；
- 早期 T4（sm75）不支持 FA2。

每换一套环境都要重新找 wheel 或源码编译，和项目“基础依赖尽量小、可选加速
依赖隔离”的原则冲突。它还会进入 CI 的可选 GPU 测试矩阵，维护成本高于收益。

### 3.4 语义覆盖不完整

FA2 对标准 causal / sliding window 支持较好，但：

- `ChunkedLocalSpec` 的可见区间是 per-query 的 chunk 边界，不是标准
  causal/window，必须显式回退；
- `SlidingWindowSpec` 需要按具体版本验证 window 参数；
- 第一版只能对 `FullAttentionSpec` 开放。

结果是同时存在两种 block size、两种语义路径，测试和 benchmark 的对照组变复杂，
而收益只落在 full attention 的一部分 workload 上。

### 3.5 目标 workload 上 attention 占比很低

用上面的算子数据和 1.7B（28 层）做数量级估算：

- decode：28 × 0.0808 ms ≈ **2.3 ms/step**；RTX 3080 上 W4A16 int4pack 的
  step 约 52 ms，attention 约占 **4%**。即使 FA 把 attention 减半，端到端也只有
  **约 2%**。
- prefill：28 × 0.0773 ms ≈ **2.2 ms / 512 token**；int4pack prefill 约
  3100 tok/s（512 token ≈ 165 ms），attention 约占 **1.3%**。

这是跨硬件的数量级估算，不是严格同机对比，但足以说明：当前瓶颈不在 attention
kernel，而在 GEMM、采样、调度和 Python/launch 开销。把精力放在投机解码、CUDA
Graph、TP 上，收益更直接。

### 3.6 与收敛后的计划一致

`MICRO_VLLM_CAPABILITY_REPORT.md` §4.7.1 已经把官方 `flash-attn` 从核心计划中
删除，理由是“现有 Triton kernel 已覆盖；FA2 paged 对 page size 有额外约束”。
本文是该决定的落地说明。

**重新评估的条件**：目标模型变成 7B/14B 以上、上下文 8K+、batch 明显增大，并且
目标 flash-attn wheel 通过 256 对齐验证。到那时再做 time-boxed spike，作为可选
backend 并排对比，不默认替换。

---

## 4. 以后手写 / 替换算子的方向

### 4.1 唯一需要实现的契约

新增后端只需要：

1. 在 `src/micro_vllm/attention/operators/` 下实现一个
   `AttentionOperator` 子类；
2. 在 [registry.py](../src/micro_vllm/attention/operators/registry.py) 注册名字；
3. 通过 `ModelConfig.attention_operator` 选择。

不需要改 Scheduler、KVCacheManager、ModelRunner 或模型层。`Attention` 负责把
`AttentionMetadata` 传进来，算子只需要遵守以下语义：

```text
输入:
  query             [T, H_q,  D]
  key               [T, H_kv, D]
  value             [T, H_kv, D]
  layer_kv_cache    [2, num_blocks, block_size, H_kv, D]
  metadata          见 §1.3

必须完成:
  1. 把当前 key/value 写到 metadata.slot_mapping 指向的物理 slot；
  2. 对每个 query token，只 attend 到
     [query_context_start_positions[t], query_positions[t]] 区间；
  3. 支持一个请求多行 query（chunked prefill、投机验证的 K+1 行）；
  4. 支持 GQA（H_q 是 H_kv 的整数倍）。
```

### 4.2 现有 Triton kernel 的结构

[triton_paged.py](../src/micro_vllm/attention/operators/triton_paged.py) 由两个
kernel 组成：

- `_write_kv_cache_kernel`：grid = `(T, H_kv)`，按 `slot_mapping` 散射写入；
- `_paged_attention_kernel`：grid = `(query_blocks, H_kv)`，按 block table
  直接读 paged KV，fp32 online softmax，`BLOCK_N=16`，GQA 通过
  `HEAD_GROUP_SIZE` 展开。

请求映射用 GPU 侧二分查找 `_find_request_for_query_block`，避免每个请求一次
CPU 切片。这是它比 reference/SDPA 快的主要原因之一。

### 4.3 可优化的方向

如果以后要写更强的 kernel，优先级建议：

1. **去掉 metadata 构建中的 CPU 同步。**
   [attention.py:68-75](../src/micro_vllm/attention/attention.py#L68-L75) 目前用
   `query_positions.tolist()` + Python 循环逐个计算 `context_start`，每步都会
   同步一次 CPU。可以按 `AttentionSpec` 类型向量化，或直接在 kernel 里根据
   spec 参数计算。这也是 CUDA Graph 的前置条件。
2. **支持 bf16 / fp32。** 当前 `triton_paged` 只接受 fp16
   （[triton_paged.py:400-408](../src/micro_vllm/attention/operators/triton_paged.py#L400-L408)）；
   换 dtype 需要重跑数值对拍。
3. **长上下文用 split-K。** decode 时每个 KV head 的 context 很长，单 program
   串行扫完所有 block 会成为瓶颈；split-K + 二次归约可以增加并行度。
4. **调 `BLOCK_N` / `BLOCK_Q` / `num_warps` / `num_stages`。** 当前固定
   `BLOCK_N=16`，适合短上下文；2K/8K 时更大的 tile 可能更好。
5. **减少 block table 读取。** 更大的 page（例如 32/64）能降低间接寻址次数，
   但要和 prefix cache 粒度、碎片一起评估。
6. **CUDA Graph 兼容。** 避免 `.item()`、动态 shape 和每步 CPU 侧 Python 循环；
   metadata 和 buffer 尽量预分配。

### 4.4 测试与验收

任何新算子都必须以 `reference` 为 oracle，至少覆盖：

- Full / Sliding / ChunkedLocal 三种语义；
- prefill、decode、chunked prefill；
- MHA 与 GQA；
- 一个请求多行 query（投机验证的 K+1 行）；
- 跨 block 边界、部分填充的最后一个 block；
- 与 `reference` / 现有 `triton_paged` 的数值对拍（记录 dtype、容差、shape）；
- operator latency 与端到端 TTFT/TPOT，和现有后端同机对比。

现有测试入口：`tests/attention/operators/`；benchmark 入口：
`benchmarks/spike_triton_paged.py`。

---

## 5. 对投机解码的影响

不需要改 attention 后端：

- 投机验证的一次 forward 会给一个请求安排 K+1 行 query；当前
  `AttentionMetadata` 和 `triton_paged` 已经支持 requests 内多行 query，
  chunked prefill 走的是同一条路径；
- KV rollback 只改逻辑 block table 和 `num_computed_tokens`，不涉及 kernel；
- 测试时 CPU/bf16 用 `reference`，GPU fp16 用 `triton_paged`；
- 如果 spec 的 batch token 数超过 W4A16 `auto` 的 32-token 阈值，会退回
  reference GEMM；这与 attention 无关，按 W4A16 报告的收尾项把阈值调到 512
  即可。

---

## 6. 数据与复现

- 对比实验：`benchmarks/results/spikes/triton-paged-comparison-rtx5090-run-2.json`
- SDPA 后端可用性：`benchmarks/results/spikes/sdpa-rtx5090-run-2.json`
- 分项耗时：`benchmarks/results/spikes/attention-breakdown-rtx5090-run-2.json`
- 复现脚本：`benchmarks/spike_triton_paged.py`、`benchmarks/spike_sdpa.py`、
  `benchmarks/spike_attention_breakdown.py`

环境：RTX 5090（sm120）、CUDA 12.8、PyTorch 2.8.0+cu128、Triton 3.4.0；
block_size 16，context 512，GQA 16/8，head_dim 128。
