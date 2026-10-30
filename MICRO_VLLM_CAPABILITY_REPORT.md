# micro-vLLM 源码机制分析报告

- 分析对象：`/home/lenovo/infra/micro-vllm`（commit `28e6d81` 工作区）
- 分析方式：静态源码阅读；未运行 GPU 测试或真实模型推理。第 4 章的显存/选卡
  数字按公开模型 config 估算，需在目标卡上用 benchmark 复核
- 范围：Chunked Prefill、Continuous Batching、Prefix Caching、Paged/FlashAttention、
  量化（W4A16）、投机解码；当前模型支持情况和多模型族扩展改动；以及第 4 章的
  W4A16 / FlashAttention / 投机解码 / CUDA Graph + TP 落地评估与选卡建议

## 0. 结论速览

| 机制 | 是否实现 | 一句话结论 |
|---|---|---|
| Chunked Prefill | ✅ 已实现 | 统一 token-budget scheduler 自动把长 prompt 切成多个 step，并允许 prefill chunk 与 decode 同 batch |
| 连续批处理 Continuous Batching | ✅ 已实现 | 每个 scheduler step 重新组合 running 请求并准入 waiting 请求；但高层 API 是同步离线批量 |
| Prefix Caching | ✅ 已实现（可选，默认关闭） | 按完整 block 做 BLAKE2 链式哈希，支持跨请求复用；不缓存 partial block，无 COW |
| PagedAttention | ✅ 已实现 | Paged KV Cache + reference / SDPA / Triton paged 三个后端 |
| FlashAttention | ❌ 未实现（仅有 FlashAttention 式在线 softmax） | 没有 flash-attn 后端；SDPA 因显式 mask 使用 PyTorch memory-efficient kernel；Triton kernel 自己实现 online softmax |
| 量化 W4A16 | ❌ 未实现 | 只有 fp32/fp16/bf16 dense 权重；没有 QuantConfig/QuantMethod/AWQ/GPTQ/FP8/KV 量化 |
| 投机解码（含 DFlash 一类） | ❌ 未实现 | 没有 Drafter、proposal、verification、acceptance、KV rollback；V5 仅是需求文档 |

上述机制的落地设计、显卡选型与实施顺序见 **第 4 章**；模型适配改动见第 3 章。

---

## 1. 逐项机制分析

### 1.1 Chunked Prefill — ✅ 已实现

**实现方式**：没有独立的 prefill/decode 队列或状态机，而是用统一的
`num_computed_tokens` / `num_pending_tokens` + 全局 token budget 让一个长 prompt
自然跨多个 scheduler step 执行。这正是 vLLM V1 风格的 chunked prefill。

**关键位置**：

- 调度主循环：[src/micro_vllm/core/scheduler.py:72-166](src/micro_vllm/core/scheduler.py#L72-L166)
  - running 请求按剩余 budget 截断：
    [scheduler.py:84-122](src/micro_vllm/core/scheduler.py#L84-L122)
    - `num_scheduled_tokens = min(num_pending_tokens, remaining_token_budget,
      max_model_len - num_computed_tokens)`：
      [scheduler.py:90-94](src/micro_vllm/core/scheduler.py#L90-L94)
  - waiting 请求准入时同样按剩余 budget 截断：
    [scheduler.py:124-158](src/micro_vllm/core/scheduler.py#L124-L158)
    - `num_scheduled_tokens = min(num_known_tokens - num_hit_tokens,
      remaining_token_budget, max_model_len - num_hit_tokens)`：
      [scheduler.py:136-140](src/micro_vllm/core/scheduler.py#L136-L140)
- pending token 的权威定义：
  [src/micro_vllm/core/request.py:93-97](src/micro_vllm/core/request.py#L93-L97)
- 只有计算追上最后一个已知 token 才采样：
  - `requires_sampling = (num_computed_tokens + num_scheduled_tokens ==
    num_known_tokens)`：
    [scheduler.py:242-243](src/micro_vllm/core/scheduler.py#L242-L243)
  - 中间 chunk 不携带 sampling params：
    [scheduler.py:250-255](src/micro_vllm/core/scheduler.py#L250-L255)
- 同一个 plan 可同时包含 decode 和 prefill chunk：
  `running` 先调度，`waiting` 后准入，见
  [scheduler.py:84-158](src/micro_vllm/core/scheduler.py#L84-L158)
- 执行侧把每个请求的 chunk 展平成同一个 batch：
  [src/micro_vllm/worker/model_runner.py:573-642](src/micro_vllm/worker/model_runner.py#L573-L642)
- 测试：
  - `test_long_prompt_is_split_across_multiple_chunks`：
    [tests/core/test_scheduler.py:203](tests/core/test_scheduler.py#L203)
  - `test_intermediate_prompt_chunk_does_not_require_sampled_token`：
    [tests/core/test_scheduler.py:344](tests/core/test_scheduler.py#L344)
  - `test_running_decode_precedes_waiting_prefill_in_same_plan`：
    [tests/core/test_scheduler.py:355](tests/core/test_scheduler.py#L355)

**限制/说明**：

- chunk 大小只由 `max_num_batched_tokens`、`max_model_len` 和 pending token 数共同
  决定；没有单独的 `max_prefill_chunk` 配置项。需求文档中该项仍是未完成：
  [docs/requirements/v1-core-engine.md:94](docs/requirements/v1-core-engine.md#L94)、
  [docs/requirements/v1-core-engine.md:380](docs/requirements/v1-core-engine.md#L380)
- `attention/spec.py` 里的 `ChunkedLocalSpec` **不是** chunked prefill，它是
  “只在当前对齐 chunk 内做局部 attention”的可见性策略，二者不要混淆：
  [src/micro_vllm/attention/spec.py:40-51](src/micro_vllm/attention/spec.py#L40-L51)

---

### 1.2 连续批处理 Continuous Batching — ✅ 已实现（引擎级；高层是同步离线 API）

**实现方式**：`Scheduler` 同时维护 `waiting` 与 `running`，每个 step 先调度
running 请求，再用剩余 token/sequence budget 从 waiting 队列准入新请求；请求完成后
在 `update()` 中移出 running。因此每个 iteration 的 batch 是动态重组的。

**关键位置**：

- `waiting` / `running` 队列：
  [src/micro_vllm/core/scheduler.py:22-24](src/micro_vllm/core/scheduler.py#L22-L24)
- 每个 step 重新调度 running + 准入 waiting：
  [scheduler.py:72-166](src/micro_vllm/core/scheduler.py#L72-L166)
- 完成后从 running 移除并释放：
  [scheduler.py:214-223](src/micro_vllm/core/scheduler.py#L214-L223)
- Engine 的 `schedule -> execute -> update` 单步循环：
  [src/micro_vllm/engine/engine.py:33-56](src/micro_vllm/engine/engine.py#L33-L56)
- Engine 支持运行中动态添加请求：
  [engine.py:17-31](src/micro_vllm/engine/engine.py#L17-L31)
- 高层 `LLM.generate()` 的循环驱动：
  [src/micro_vllm/llm.py:138-145](src/micro_vllm/llm.py#L138-L145)
- 抢占后 victim 回到 waiting 并可重新被准入：
  [scheduler.py:100-121](src/micro_vllm/core/scheduler.py#L100-L121)
- 相关需求项均已勾选：
  [docs/requirements/v1-core-engine.md:72-95](docs/requirements/v1-core-engine.md#L72-L95)

**限制/说明**：

- `LLM.generate()` 是同步离线批量 API：一次性加入所有 prompt，然后阻塞到全部结束；
  不支持 async streaming / OpenAI API server。需求文档中 streaming 仍未实现：
  [docs/requirements/v1-core-engine.md:53-54](docs/requirements/v1-core-engine.md#L53-L54)
- waiting 是 FIFO，没有 priority：
  [scheduler.py:124-130](src/micro_vllm/core/scheduler.py#L124-L130)
- 抢占策略是确定性的“调度 running 列表尾部尚未加入当前 plan 的请求”，没有单独
  policy 抽象：
  [scheduler.py:100-113](src/micro_vllm/core/scheduler.py#L100-L113)

---

### 1.3 Prefix Caching — ✅ 已实现（可选，默认关闭；仅完整 block）

**实现方式**：

1. 对每个完整 token block 做确定性 BLAKE2b 哈希，哈希包含 parent block hash、
   token block、block size、cache namespace、cache group id，从而表达完整前缀链；
2. `BlockPool` 维护 `hash -> physical block` 索引和 refcount；
3. 新请求准入时查找最长可复用前缀，命中 block 直接 acquire（refcount +1）；
4. 提交时把已计算完整 block 的哈希写入 cache，并把重复内容的 block canonical 化；
5. 支持 full / sliding-window / chunked-local 三种 cache group 的不同保留策略。

**关键位置**：

- 开关与配置：
  [src/micro_vllm/config.py:94-97](src/micro_vllm/config.py#L94-L97)
  （`enable_prefix_cache: bool = False`，默认关闭）
- 哈希：
  [src/micro_vllm/core/prefix_cache.py:12-40](src/micro_vllm/core/prefix_cache.py#L12-L40)
- 物理 block 哈希索引、引用计数、缓存 block 的 acquire/allocate/release：
  [src/micro_vllm/core/block.py:20-158](src/micro_vllm/core/block.py#L20-L158)
  - `cache_block` / `get_cached_block` / `acquire_cached_block`：
    [block.py:82-108](src/micro_vllm/core/block.py#L82-L108)
  - `allocate`：
    [block.py:123-150](src/micro_vllm/core/block.py#L123-L150)
- 每个 cache group 的 lookup/commit：
  [src/micro_vllm/core/cache_group_manager.py:98-212](src/micro_vllm/core/cache_group_manager.py#L98-L212)
  - `find_longest_cache_hit`（full）：
    [cache_group_manager.py:224-242](src/micro_vllm/core/cache_group_manager.py#L224-L242)
  - `cache_computed_blocks`：
    [cache_group_manager.py:98-132](src/micro_vllm/core/cache_group_manager.py#L98-L132)
- 多 cache group 的 fixed-point 命中：
  [src/micro_vllm/core/kv_cache_manager.py:113-134](src/micro_vllm/core/kv_cache_manager.py#L113-L134)
- 原子分配缓存 block + 新 block：
  [kv_cache_manager.py:159-222](src/micro_vllm/core/kv_cache_manager.py#L159-L222)
- 每步提交哈希并释放被策略跳过的 block：
  [kv_cache_manager.py:224-274](src/micro_vllm/core/kv_cache_manager.py#L224-L274)
- Scheduler 在 waiting 准入前先查 prefix hit：
  [src/micro_vllm/core/scheduler.py:131-146](src/micro_vllm/core/scheduler.py#L131-L146)
- 测试：
  - `tests/core/test_prefix_cache.py:27-75`
  - `test_waiting_request_uses_prefix_hit_before_admission`：
    [tests/core/test_scheduler.py:511](tests/core/test_scheduler.py#L511)
  - `test_preempted_victim_reenters_from_cached_prefix`：
    [tests/core/test_scheduler.py:671](tests/core/test_scheduler.py#L671)

**限制/说明**：

- 只缓存完整 block，partial block 不共享，也没有 copy-on-write：
  [docs/requirements/v1-core-engine.md:140-141](docs/requirements/v1-core-engine.md#L140-L141)
- prompt 恰好按 block 对齐时，最后一个 block 会保留重新计算以产生 logits：
  [docs/requirements/v1-core-engine.md:146](docs/requirements/v1-core-engine.md#L146)
- 默认 `cache_namespace="default"`，`LLM` 构造时未显式传 namespace：
  [src/micro_vllm/core/kv_cache_manager.py:54-60](src/micro_vllm/core/kv_cache_manager.py#L54-L60)、
  [src/micro_vllm/llm.py:57-59](src/micro_vllm/llm.py#L57-L59)
- 没有 CPU swap/offload；request 释放后零引用 cached block 保留在 free queue 中，
  直到被重新分配或复用：
  [block.py:152-158](src/micro_vllm/core/block.py#L152-L158)

---

### 1.4 PagedAttention / FlashAttention

#### 1.4.1 PagedAttention — ✅ 已实现

**Paged KV Cache 结构**：

- block size 配置：
  [src/micro_vllm/config.py:94](src/micro_vllm/config.py#L94)
- KV cache topology（按 attention spec 分组，多个 cache group 共享一个 BlockPool）：
  [src/micro_vllm/core/kv_cache_spec.py:10-80](src/micro_vllm/core/kv_cache_spec.py#L10-L80)
- 物理 KV tensor 分配：
  - 形状：`[2, num_blocks, block_size, num_key_value_heads, head_dim]`：
    [src/micro_vllm/worker/model_runner.py:418-443](src/micro_vllm/worker/model_runner.py#L418-L443)
  - 分配：
    [model_runner.py:418-426](src/micro_vllm/worker/model_runner.py#L418-L426)
- logical position -> logical block -> physical slot：
  [src/micro_vllm/attention/attention.py:31-41](src/micro_vllm/attention/attention.py#L31-L41)

**三个 attention 后端**（由 `ModelConfig.attention_operator` 选择，
默认 `"reference"`：[config.py:33](src/micro_vllm/config.py#L33)）：

1. **reference**：可读性优先的 PyTorch 实现，写 KV cache + gather 每个请求的可见
   context + 显式 causal mask + softmax：
   [src/micro_vllm/attention/operators/reference.py:11-81](src/micro_vllm/attention/operators/reference.py#L11-L81)；
   KV 写入/gather：[reference.py:84-138](src/micro_vllm/attention/operators/reference.py#L84-L138)
2. **sdpa**：先 page write + per-request gather，再调用 PyTorch
   `F.scaled_dot_product_attention`：
   [src/micro_vllm/attention/operators/sdpa.py:82-114](src/micro_vllm/attention/operators/sdpa.py#L82-L114)
3. **triton_paged**：直接通过 block table 读取 paged KV，不 materialize 完整
   context；融合 page write、GQA/MQA head mapping、causal/context-start mask、
   FP32 online softmax 和 V accumulation：
   - page write kernel：
     [src/micro_vllm/attention/operators/triton_paged.py:11-65](src/micro_vllm/attention/operators/triton_paged.py#L11-L65)
   - fused paged attention kernel：
     [triton_paged.py:94-287](src/micro_vllm/attention/operators/triton_paged.py#L94-L287)
   - online softmax / rescaling：
     [triton_paged.py:189-276](src/micro_vllm/attention/operators/triton_paged.py#L189-L276)
   - launch 与 grid：
     [triton_paged.py:325-386](src/micro_vllm/attention/operators/triton_paged.py#L325-L386)
   - operator 与 CUDA/FP16 限制：
     [triton_paged.py:389-415](src/micro_vllm/attention/operators/triton_paged.py#L389-L415)
- 注册表（`reference`、`sdpa` 静态注册；`triton_paged` lazy import）：
  [src/micro_vllm/attention/operators/registry.py:13-51](src/micro_vllm/attention/operators/registry.py#L13-L51)

#### 1.4.2 FlashAttention — ❌ 未作为后端实现

- 项目没有依赖或封装官方 `flash-attn`，也没有 FlashAttention-2/3 kernel。
- `sdpa` 路径的 benchmark 明确记录：SDPA 实际使用 PyTorch
  **memory-efficient CUTLASS kernel**，不是 FlashAttention：
  [docs/implementation/v1-performance-todo.md:248-253](docs/implementation/v1-performance-todo.md#L248-L253)
- 文档也明确记录：**Flash Attention 因显式 non-null mask 不可用**；cuDNN 只支持
  prefill，不支持 query length 1 的 decode：
  [docs/implementation/v1-performance-todo.md:460-463](docs/implementation/v1-performance-todo.md#L460-L463)
- `triton_paged.py` 的 fused kernel 使用了 **FlashAttention 式的 online softmax**
  （分块遍历 context、running max/sum、增量 rescale），但它是一个自实现的
  correctness-first Triton kernel，不是 FlashAttention 库/kernel。
- 因此准确表述是：**PagedAttention ✅，FlashAttention ❌**；存在一个
  FlashAttention 风格的分块在线 softmax 实现。

**限制/说明**：

- `triton_paged` 只支持 CUDA + `torch.float16`：
  [triton_paged.py:401-408](src/micro_vllm/attention/operators/triton_paged.py#L401-L408)
- `triton` 不在 `pyproject.toml` 的必需依赖中，只有显式选择 `triton_paged` 时才
  lazy import：
  [pyproject.toml:10-17](pyproject.toml#L10-L17)、
  [operators/registry.py:27-38](src/micro_vllm/attention/operators/registry.py#L27-L38)
- `sdpa` 路径仍然有 Python per-request loop 和 dense gather，不是最终 optimized
  paged backend：
  [sdpa.py:88-114](src/micro_vllm/attention/operators/sdpa.py#L88-L114)

---

### 1.5 量化（W4A16） — ❌ 未实现

**证据**：

- `ModelConfig.dtype` 只允许 `float32 / float16 / bfloat16`：
  [src/micro_vllm/config.py:38-42](src/micro_vllm/config.py#L38-L42)
- `Linear` 就是普通 `nn.Linear`，没有量化权重、scale、zero point 或反量化逻辑：
  [src/micro_vllm/layers/linear.py:6-8](src/micro_vllm/layers/linear.py#L6-L8)
- MLP/Embedding/LMHead 也是普通 dense 层：
  [src/micro_vllm/layers/mlp.py:10-21](src/micro_vllm/layers/mlp.py#L10-L21)、
  [src/micro_vllm/layers/embedding.py:8-24](src/micro_vllm/layers/embedding.py#L8-L24)
- KV cache 的 dtype 直接跟随模型 dtype：
  [src/micro_vllm/worker/model_runner.py:418-426](src/micro_vllm/worker/model_runner.py#L418-L426)
  （`torch.empty(..., dtype=self._dtype, ...)`）
- 权重加载是把 checkpoint tensor 直接 `copy_` 到同名 float parameter：
  [src/micro_vllm/model_loader/default.py:44-46](src/micro_vllm/model_loader/default.py#L44-L46)、
  [src/micro_vllm/models/llama.py:222-235](src/micro_vllm/models/llama.py#L222-L235)
- 全仓库没有 `QuantizationConfig`、`QuantMethod`、AWQ/GPTQ/FP8/INT4 kernel；
  `grep` 到的 quantization 只出现在 README 对比表和需求文档中。
- V4 量化需求文档的所有实现项都未勾选：
  [docs/requirements/v4-quantization.md:19-25](docs/requirements/v4-quantization.md#L19-L25)、
  [docs/requirements/v4-quantization.md:62-75](docs/requirements/v4-quantization.md#L62-L75)、
  [docs/requirements/v4-quantization.md:89-100](docs/requirements/v4-quantization.md#L89-L100)
- V1 明确把“权重或 KV 量化”列为不包含：
  [docs/requirements/v1-core-engine.md:563-576](docs/requirements/v1-core-engine.md#L563-L576)

**后果**：

- 不能直接加载 AWQ/GPTQ/Compressed-Tensors 检查点：模型类构造的是 float
  `nn.Linear`，而量化 checkpoint 里是 packed int 权重 + scales/zero points，
  名称和 shape 都对不上，`copy_` 会失败或得到错误结果。
- 没有 W4A16、W8A16、W8A8、FP8 weight/activation、FP8 KV cache 中的任何一种。

---

### 1.6 投机解码（含 DFlash 一类） — ❌ 未实现

**证据**：

- 没有 `Drafter` / `DraftOutput` / `VerificationResult` / acceptance sampler /
  speculative config；全仓库没有 speculative decoding 的实现代码。
- 调度器只支持“每个请求本轮计算一段 pending token，最后一步采一个 token”：
  [src/micro_vllm/core/scheduler.py:225-259](src/micro_vllm/core/scheduler.py#L225-L259)
- `ScheduledRequest` 只有 `requires_sampling` 一个采样标记，没有“哪些 token 是
  proposal”的字段：
  [src/micro_vllm/core/scheduler_output.py:18-33](src/micro_vllm/core/scheduler_output.py#L18-L33)
- `ModelRunnerOutput` 每个请求只返回一个 `SampledToken`，没有多 token 验证结果或
  accepted/rejected 信息：
  [src/micro_vllm/executor/interface.py:9-24](src/micro_vllm/executor/interface.py#L9-L24)
- ModelRunner 只对 sampling rows 计算 logits 并交给 Sampler 采一个 token：
  [src/micro_vllm/worker/model_runner.py:212-238](src/micro_vllm/worker/model_runner.py#L212-L238)
- V5 投机解码需求文档的所有 P0/P1 项均未勾选：
  [docs/requirements/v5-speculative-decoding.md:27-45](docs/requirements/v5-speculative-decoding.md#L27-L45)、
  [docs/requirements/v5-speculative-decoding.md:149-157](docs/requirements/v5-speculative-decoding.md#L149-L157)
- V5 明确把 EAGLE、MTP、Medusa、tree-based speculative decoding 等列为不包含：
  [v5-speculative-decoding.md:149-157](docs/requirements/v5-speculative-decoding.md#L149-L157)

**注**：仓库中没有名为 “DFlash” 的实现或文档。无论指 draft-model、n-gram、
EAGLE、Medusa 还是 DFlash，当前都没有投机解码路径。

---

### 1.7 其他已实现的相关能力（补充）

- 请求抢占与 recompute：
  [scheduler.py:100-121](src/micro_vllm/core/scheduler.py#L100-L121)、
  [request.py:131-141](src/micro_vllm/core/request.py#L131-L141)
- Sliding-window / chunked-local attention 的 cache 保留策略：
  [cache_group_manager.py:248-281](src/micro_vllm/core/cache_group_manager.py#L248-L281)
- 多 KV cache group 共享一个 BlockPool：
  [kv_cache_spec.py:52-80](src/micro_vllm/core/kv_cache_spec.py#L52-L80)、
  [kv_cache_manager.py:54-83](src/micro_vllm/core/kv_cache_manager.py#L54-L83)
- Structured trace（KV layout + 每步 commit）：
  [src/micro_vllm/trace.py](src/micro_vllm/trace.py)
- GPU warmup、显存 profile、物理 block 数推导：
  [model_runner.py:253-416](src/micro_vllm/worker/model_runner.py#L253-L416)

---

## 2. 当前支持的模型

### 2.1 已注册的架构

模型注册表是静态不可变映射，只有三个入口：

[src/micro_vllm/models/registry.py:13-21](src/micro_vllm/models/registry.py#L13-L21)

| HF `architectures` | 实现类 | 文件 | 说明 |
|---|---|---|---|
| `LlamaForCausalLM` | `LlamaModel` | [models/llama.py:143-249](src/micro_vllm/models/llama.py#L143-L249) | dense Llama 风格：RMSNorm + RoPE + SwiGLU + GQA/MQA |
| `MistralForCausalLM` | `MistralModel` | [models/mistral.py:18-127](src/micro_vllm/models/mistral.py#L18-L127) | 复用 `LlamaDecoderLayer`，attention 用 `SlidingWindowSpec` |
| `Qwen3ForCausalLM` | `Qwen3Model` | [models/qwen.py:159-274](src/micro_vllm/models/qwen.py#L159-L274) | dense Qwen3：显式 `head_dim`、Q/K RMSNorm、可选 tied embedding |

选择逻辑：遍历 `config.architectures`，返回第一个能识别的类；否则报错：
[registry.py:24-39](src/micro_vllm/models/registry.py#L24-L39)

### 2.2 这些模型共享的能力

- 单设备、decoder-only、dense 文本生成；一个 `LLM` 实例只加载一个模型：
  [llm.py:50-69](src/micro_vllm/llm.py#L50-L69)
- HF `config.json` + `AutoTokenizer` + safetensors 权重：
  [model_loader/base.py:20-49](src/micro_vllm/model_loader/base.py#L20-L49)、
  [model_loader/default.py:26-46](src/micro_vllm/model_loader/default.py#L26-L46)、
  [model_loader/checkpoint.py:30-55](src/micro_vllm/model_loader/checkpoint.py#L30-L55)
- GQA/MQA（`num_attention_heads % num_key_value_heads == 0`）：
  [llama.py:157-164](src/micro_vllm/models/llama.py#L157-L164)、
  [qwen.py:175-186](src/micro_vllm/models/qwen.py#L175-L186)
- 标准 RoPE（只有 `rope_theta`，没有 `rope_scaling`）：
  [layers/rotary_embedding.py:14-52](src/micro_vllm/layers/rotary_embedding.py#L14-L52)
- 普通 `nn.Linear`、dense SwiGLU、RMSNorm、Embedding/LMHead：
  [layers/linear.py:6-8](src/micro_vllm/layers/linear.py#L6-L8)、
  [layers/mlp.py:10-21](src/micro_vllm/layers/mlp.py#L10-L21)、
  [layers/normalization.py:7-22](src/micro_vllm/layers/normalization.py#L7-L22)
- greedy / temperature / top-k / top-p 采样（但每个请求每步只采一个 token）：
  [sampling/sampler.py](src/micro_vllm/sampling/sampler.py)、
  [sampling/params.py](src/micro_vllm/sampling/params.py)

### 2.3 当前不支持的模型/能力

| 模型/能力 | 状态 | 原因 |
|---|---|---|
| Qwen2 / Qwen2.5（`Qwen2ForCausalLM`） | ❌ | registry 未注册；Qwen2 的 attention bias、无 Q/K norm、可选 sliding window / rope_scaling 与 Qwen3 不完全相同 |
| Qwen3-MoE（`Qwen3MoeForCausalLM`） | ❌ | 没有 MoE layer / router / expert 权重加载 |
| Qwen3-Next 等混合线性注意力 | ❌ | 没有线性注意力/混合层 |
| DeepSeek-V2 / V3 / R1 官方权重（`DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM`） | ❌ | 没有 MLA、MoE、partial RoPE/YaRN、FP8 权重加载 |
| DeepSeek-R1-Distill-Qwen（`Qwen2ForCausalLM`） | ❌ | 实际是 Qwen2 dense 架构；registry 未注册 Qwen2（详见第 3.2 节） |
| Llama 3.1 / 3.2 的 `rope_scaling="llama3"` | ⚠️ 能加载但不保证正确 | `RotaryEmbedding` 只读 `rope_theta`，忽略 `rope_scaling` |
| Llama 4 / MoE Llama | ❌ | 没有 MoE / iRoPE |
| 量化 checkpoint（AWQ/GPTQ/FP8） | ❌ | 没有量化层和反量化加载 |
| Tensor/Pipeline/Expert Parallel | ❌ | V1 明确不包含，见 [v1-core-engine.md:563-576](docs/requirements/v1-core-engine.md#L563-L576) |
| 多模态 | ❌ | V6 仅是需求文档 |

---

## 3. 让 micro-vLLM 同时支持 Qwen / DeepSeek / Llama 需要的改动（按“每个系列一个模型”重写）

### 3.0 目标与结论

目标模型：

- Qwen：`Qwen3-0.6B` / `Qwen3-8B`
- Llama：`Llama-3.2-1B`
- DeepSeek：`DeepSeek-R1-Distill-Qwen-1.5B` / `7B`
- 额外：实现一个**简易 MoE layer**，只验证 forward 能正常工作，不接入任何真实模型

结论：**工作量会显著降低**。主要原因是目标模型的选择避开了 DeepSeek 最复杂的部分：

| 目标模型 | HF `architectures` | 当前状态 | 需要的改动 |
|---|---|---|---|
| Qwen3-0.6B | `Qwen3ForCausalLM` | ✅ 已支持并有测试 | 无 |
| Qwen3-8B | `Qwen3ForCausalLM` | ✅ 代码层面已支持 | 无；只需足够显存 |
| Llama-3.2-1B | `LlamaForCausalLM` | ⚠️ 能加载，但 RoPE 不正确 | 增加 `llama3` rope scaling |
| DeepSeek-R1-Distill-Qwen-1.5B | `Qwen2ForCausalLM` | ❌ 未注册 | 新增 Qwen2 dense 模型 |
| DeepSeek-R1-Distill-Qwen-7B | `Qwen2ForCausalLM` | ❌ 未注册 | 同上；另需足够显存 |
| 简易 MoE | 独立 `layers/moe.py` | ❌ 未实现 | 新增 layer + 单测 |

关键认识：

1. **Qwen3-0.6B/8B 已经支持**。两者都是 dense `Qwen3ForCausalLM`，当前
   `Qwen3Model` 已实现显式 `head_dim`、Q/K RMSNorm、GQA，代码层面无需改动：
   [models/qwen.py:159-274](src/micro_vllm/models/qwen.py#L159-L274)。
2. **Llama-3.2-1B 只差 RoPE scaling**。它是标准 `LlamaForCausalLM`，当前
   `LlamaModel` 已支持 tied embedding、GQA、SwiGLU；但 config 里的
   `rope_scaling={"rope_type": "llama3", ...}` 被忽略：
   [layers/rotary_embedding.py:14-52](src/micro_vllm/layers/rotary_embedding.py#L14-L52)。
3. **DeepSeek-R1-Distill-Qwen-1.5B/7B 不是 DeepSeek-V2/V3 的 MLA+MoE 模型**。
   它们是 Qwen2 dense 架构，`config.json` 中 `architectures` 为
   `Qwen2ForCausalLM`、`model_type` 为 `qwen2`。因此**不需要 MLA、MoE、FP8、
   TP/EP**，只需要新增一个 Qwen2 模型类。
4. **简易 MoE 不需要接入模型**。作为独立 `nn.Module` + 单元测试即可，不触碰
   `ModelRunner`、KV cache、Scheduler。

因此，上一版报告中“DeepSeek-V2/V3 需要 MLA + MoE + FP8 + TP/EP + KV cache
形状重构”的大块工作全部消失。

---

### 3.1 改动 1：Llama-3.2-1B 的 `llama3` RoPE scaling

#### 为什么需要

Llama-3.2-1B 的 `config.json` 使用：

```json
{
  "rope_theta": 500000.0,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  }
}
```

当前 `RotaryEmbedding` 只读取 `rope_theta`，直接用
`1 / (theta ** (arange(0, head_dim, 2) / head_dim))` 计算频率，忽略了
`rope_scaling`。模型虽然能加载和生成，但位置编码数值不对，长上下文和 token
结果都不可靠。

#### 涉及文件

- [layers/rotary_embedding.py:14-52](src/micro_vllm/layers/rotary_embedding.py#L14-L52)
  - 给 `RotaryEmbedding` 增加 `rope_scaling: Mapping[str, object] | None` 参数；
  - 在 `__init__` 中预计算 `inv_freq`（并缓存 `cos/sin` 或在 forward 中复用），
    而不是每次 forward 重新计算；
  - 支持 `rope_type == "llama3"`（兼容旧字段名 `type`）；
  - 其他未知 `rope_type` 直接报错，避免静默使用错误频率。
- [models/llama.py:143-193](src/micro_vllm/models/llama.py#L143-L193)
  - `LlamaModel.__init__` 从 `config.values` 读取 `rope_scaling`：
    [llama.py:165-167](src/micro_vllm/models/llama.py#L165-L167)；
  - 通过 `LlamaDecoderLayer` 传到 `LlamaAttention`，最终传给
    `RotaryEmbedding`：
    - `LlamaAttention` 构造 RoPE：
      [llama.py:49](src/micro_vllm/models/llama.py#L49)
    - `LlamaDecoderLayer` 转发参数：
      [llama.py:88-106](src/micro_vllm/models/llama.py#L88-L106)
    - `LlamaModel` 构造 decoder layer：
      [llama.py:174-182](src/micro_vllm/models/llama.py#L174-L182)
- 可选：[models/qwen.py:64](src/micro_vllm/models/qwen.py#L64) 也把
  `rope_scaling` 透传给 `RotaryEmbedding`。Qwen3-0.6B/8B 的
  `rope_scaling` 为 `null`，所以对目标 Qwen3 模型不是必需，但可以让 Qwen2/Qwen3
  共用同一套 RoPE 实现。

#### `llama3` 频率计算（需与 Hugging Face 对齐）

Hugging Face 的 `_compute_llama3_parameters` 逻辑：

1. 先算标准 `inv_freq`；
2. 计算波长 `wavelen = 2 * pi / inv_freq`；
3. `low_freq_wavelen = original_max_position_embeddings / low_freq_factor`；
4. `high_freq_wavelen = original_max_position_embeddings / high_freq_factor`；
5. 波长大于 `low_freq_wavelen` 的频率除以 `factor`；
6. 波长小于 `high_freq_wavelen` 的频率保持不变；
7. 中间频率做线性平滑：

```text
smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor)
smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / factor)
                    + smooth_factor * inv_freq
```

最终把结果作为 `inv_freq` 使用。实现时建议统一用 `float32`，并保留现有
`rotate_half` 的 half-split 布局。

#### 测试

- 单测：对 Llama-3.2-1B 的 RoPE 参数，比较本实现与 Transformers
  `LlamaRotaryEmbedding` 的 `cos/sin`，覆盖 position `0, 1, 127, 8192, 131071`；
- 回归：`rope_scaling=None` 时，Qwen3-0.6B 和 Llama 3 原始配置的结果与修改前
  完全一致；
- 端到端：tiny Llama + `rope_scaling` 与 Transformers 做 logits/token 对拍；
- 真实模型：Llama-3.2-1B greedy 输出与 Transformers 逐 token 对比。

---

### 3.2 改动 2：新增 Qwen2 支持（同时解锁 DeepSeek-R1-Distill-Qwen）

#### 为什么需要

`DeepSeek-R1-Distill-Qwen-1.5B/7B` 的 HF config 是：

- `architectures: ["Qwen2ForCausalLM"]`
- `model_type: "qwen2"`

它们是 Qwen2 dense 模型，不是 DeepSeek-V2/V3 的 MLA + MoE。当前 registry 只有
`Qwen3ForCausalLM`，没有 `Qwen2ForCausalLM`：
[models/registry.py:15-21](src/micro_vllm/models/registry.py#L15-L21)。
因此加载时会直接报 “unsupported model architectures”。

#### 新增文件

建议新增 `src/micro_vllm/models/qwen2.py`，包含：

- `Qwen2Attention`
- `Qwen2DecoderLayer`
- `Qwen2Model`

可复用现有通用层：

- `SwiGLUMLP`：[layers/mlp.py:10-21](src/micro_vllm/layers/mlp.py#L10-L21)
- `RMSNorm`：[layers/normalization.py:7-22](src/micro_vllm/layers/normalization.py#L7-L22)
- `RotaryEmbedding`：[layers/rotary_embedding.py:14-52](src/micro_vllm/layers/rotary_embedding.py#L14-L52)
- `TokenEmbedding` / `LMHead`：[layers/embedding.py:8-24](src/micro_vllm/layers/embedding.py#L8-L24)
- `Linear`：[layers/linear.py:6-8](src/micro_vllm/layers/linear.py#L6-L8)

#### Qwen2 与 Qwen3 的关键差异

| 项目 | Qwen3 | Qwen2（目标 Distill 模型） |
|---|---|---|
| `q_proj/k_proj/v_proj` bias | 默认无（由 `attention_bias` 控制） | **有 bias**（`attention_bias=True`） |
| `o_proj` bias | 当前 Qwen3 实现跟 `attention_bias` 一致 | **无 bias** |
| Q/K RMSNorm | 有 `q_norm` / `k_norm` | **没有** |
| `head_dim` | 显式配置（0.6B 为 128） | 通常为 `hidden_size // num_attention_heads`，也兼容显式 `head_dim` |
| RoPE | 标准 RoPE | 标准 RoPE；目标 Distill 配置 `rope_scaling=null` |
| sliding window | 无 | 目标 Distill 配置 `use_sliding_window=false`、`sliding_window=null` |
| tied embedding | 可选 | 1.5B 可能是 tied，7B 通常 untied；需两种都支持 |

**最重要的坑**：不能直接把 `Qwen3Attention` 的 `attention_bias=True` 拿来用。
当前 `Qwen3Attention` 对 `q_proj/k_proj/v_proj/o_proj` 四个投影都使用
`bias=attention_bias`：
[models/qwen.py:42-61](src/micro_vllm/models/qwen.py#L42-L61)。
而 Qwen2 的 `o_proj` 没有 bias；如果错误地加了 `o_proj.bias`，checkpoint 中又没有
这个参数，它会保持随机初始化，导致静默错误。因此 `Qwen2Attention` 应显式：

```python
self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)
```

Qwen2 不创建 `q_norm` / `k_norm`。

#### 实现方式选择

有两种做法：

1. **复制并精简 Qwen3 结构（推荐，风险最低）**
   - 新增 `Qwen2Attention`，去掉 q/k norm，按上表设置 bias；
   - 新增 `Qwen2DecoderLayer` / `Qwen2Model`，结构基本复制
     [models/qwen.py:102-240](src/micro_vllm/models/qwen.py#L102-L240)；
   - 代码有约 100–150 行重复，但不触碰已经验证过的 Qwen3 路径。

2. **抽取通用 Qwen decoder layer（更干净，但要多回归 Qwen3）**
   - 让 decoder layer 接收外部构造的 attention module；
   - Qwen2/Qwen3 共用 residual/norm/MLP 结构，只注入不同 attention；
   - 需要额外跑一遍 Qwen3 的现有测试，确认无回归。

考虑到目标是“工作量尽量低”，建议先采用方案 1；后续如果要支持更多 Qwen 变体，
再做方案 2 的重构。

#### 配置读取

`ModelConfig` 是通用只读 mapping，不需要新增配置类：
[config.py:22-81](src/micro_vllm/config.py#L22-L81)。

`Qwen2Model` 读取：

- `hidden_size`、`intermediate_size`、`num_hidden_layers`
- `num_attention_heads`、`num_key_value_heads`
- `head_dim`（可选，缺省 `hidden_size // num_attention_heads`）
- `rms_norm_eps`
- `rope_theta`、`rope_scaling`
- `attention_bias`（Qwen2 为 `True`）
- `tie_word_embeddings`
- `use_sliding_window` / `sliding_window`

对 `use_sliding_window=true` 的配置，建议直接抛出明确的 `ValueError`，而不是静默
忽略；目标 Distill 模型都是 full attention，不需要 sliding window 实现。

#### 权重加载

现有 loader 直接把 checkpoint tensor `copy_` 到同名参数：
[model_loader/default.py:44-46](src/micro_vllm/model_loader/default.py#L44-L46)。

Qwen2 checkpoint 的命名与 Qwen3 基本一致：

```text
model.embed_tokens.weight
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.q_proj.bias
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.k_proj.bias
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.v_proj.bias
model.layers.N.self_attn.o_proj.weight
model.layers.N.mlp.gate_proj.weight
...
lm_head.weight
```

可以复用 Qwen3 的 `_qwen3_parameter_name` 逻辑（去掉 `model.` 前缀、处理 tied
`lm_head.weight`）：
[models/qwen.py:263-274](src/micro_vllm/models/qwen.py#L263-L274)。
建议把该 helper 提成共享函数，或在 `qwen2.py` 中复制一份小函数。

#### 测试

- 单测：tiny Qwen2 config（带 `attention_bias=True`），与 Transformers
  `Qwen2ForCausalLM` 对拍 hidden/logits；
- 测试 `q_proj.bias/k_proj.bias/v_proj.bias` 确实被加载；
- 测试 `o_proj` **没有** bias 参数；
- 测试 tied / untied `lm_head` 两种情况；
- 测试 `use_sliding_window=true` 被明确拒绝；
- 集成：`DeepSeek-R1-Distill-Qwen-1.5B` 端到端 greedy 输出与 Transformers 对比；
- 如果显存允许，再跑 `7B`；否则至少用 tiny config + 1.5B 验证。

#### Tokenizer / 权重文件

- Qwen2 tokenizer 是标准 HF tokenizer，现有 `AutoTokenizer` +
  `trust_remote_code=False` 可以直接加载：
  [tokenizer.py:58-81](src/micro_vllm/tokenizer.py#L58-L81)；
- 权重是标准 safetensors，现有 loader 支持单文件和多 shard：
  [model_loader/checkpoint.py:30-55](src/micro_vllm/model_loader/checkpoint.py#L30-L55)；
- 不需要修改 loader 的下载模式或文件匹配规则。

---

### 3.3 Qwen3-0.6B / 8B：无需代码改动

两者都是 dense `Qwen3ForCausalLM`：

- `Qwen3-0.6B` 在仓库中已有完整测试：
  [tests/integration/test_hub_qwen.py](tests/integration/test_hub_qwen.py)、
  [tests/integration/test_loaded_qwen3.py](tests/integration/test_loaded_qwen3.py)、
  [tests/integration/test_loaded_qwen3_engine.py](tests/integration/test_loaded_qwen3_engine.py)
- `Qwen3-8B` 的架构参数与 0.6B 同类：显式 `head_dim=128`、
  `num_key_value_heads` GQA、Q/K RMSNorm、`rope_theta` 可配置。当前
  `Qwen3Model` 已覆盖：
  [models/qwen.py:159-240](src/micro_vllm/models/qwen.py#L159-L240)。
- 不需要新增模型类、不需要改 attention/cache/loader。

需要注意的只是**资源与配置**，不是代码：

- 8B 模型 FP16 权重约 16 GB，再加上 KV cache 和激活，实际需要 24 GB 级别以上
  显存；T4 16 GB 上无法舒适运行；
- `max_model_len`、`max_num_batched_tokens`、`gpu_memory_utilization` 要按显存
  调整；
- 如果 GPU 不支持 BF16，使用 `torch.float16`。

---

### 3.4 改动 3：简易 MoE layer（独立验证，不接入模型）

#### 目标

实现一个可读、可测试的 MoE 层，验证：

- router 能产生 top-k 路由；
- 每个 token 被分配到 top-k 个 expert；
- expert 输出按 gate 权重加权求和；
- 输出 shape 与输入一致；
- 可选 shared expert 能正确叠加。

**不要求**与 Qwen/Llama/DeepSeek 模型适配，也不要求性能优化。

#### 新增文件

`src/micro_vllm/layers/moe.py`：

```python
class MoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        shared_expert: bool = False,
        normalize_topk: bool = True,
    ) -> None:
        ...
```

实现要点：

- `router = Linear(hidden_size, num_experts, bias=False)`；
- `experts = ModuleList([SwiGLUMLP(hidden_size, intermediate_size)
  for _ in range(num_experts)])`；
- 可选 `shared_expert = SwiGLUMLP(...)`；
- `forward(x)`：
  1. `logits = router(x)`，形状 `[tokens, num_experts]`；
  2. `probs = softmax(logits, dim=-1)`；
  3. `topk_probs, topk_indices = probs.topk(top_k, dim=-1)`；
  4. 如 `normalize_topk`，对 top-k 权重重新归一化；
  5. 对每个 expert，选出 `topk_indices == expert_id` 的 token，执行 expert forward，
     用对应 gate 权重加权，`index_add_` 回输出；
  6. 如果有 shared expert，加上其输出。
- 校验：`num_experts > 0`、`0 < top_k <= num_experts`，否则抛 `ValueError`。

为了保持教学可读性，第一版可以逐 expert 循环，不实现 grouped GEMM / 负载均衡
loss / expert parallel。这个层是 token-wise 的，不需要修改 KV cache 或
`ModelRunner`。

#### 测试

`tests/layers/test_moe.py`：

- 输出 shape 与输入一致；
- `top_k=1`、`top_k=2` 都能工作；
- gate 权重在归一化后 sum 约为 1；
- 用固定输入和固定参数，与手写的 dense reference 计算结果对比；
- 所有 expert 都可被路由到（构造能让不同 expert 胜出的输入）；
- 可选 shared expert 的输出等于 routed output + shared output；
- 非法 `top_k` / `num_experts` 被拒绝；
- 不产生 NaN。

---

### 3.5 明确不需要做的改动（相对上一版报告）

| 上一版中的大改动 | 新范围是否需要 | 原因 |
|---|---|---|
| MLA（Multi-head Latent Attention） | ❌ 不需要 | 目标 DeepSeek 模型是 Qwen2 dense，不是 DeepSeek-V2/V3 |
| `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM` | ❌ 不需要 | 不加载 DeepSeek 官方 MLA+MoE 权重 |
| MoE 接入 Qwen/DeepSeek 模型 | ❌ 不需要 | 只需独立 `MoE` layer + 单测 |
| FP8 权重反量化 | ❌ 不需要 | 目标模型使用 BF16/FP16 safetensors |
| TP / PP / EP | ❌ 不需要 | 目标最大 8B，单卡可覆盖（8B 需足够显存） |
| `KVCacheSpec` / MLA cache 形状重构 | ❌ 不需要 | 所有目标模型仍是标准 K/V paged cache |
| YaRN / 长上下文 rope scaling | ❌ 不需要（目标模型） | Distill-Qwen config `rope_scaling=null`；只需 llama3 scaling |
| 多模型同进程 serving | ❌ 不需要 | 按 `config.json` 选择模型族即可，一次加载一个模型 |
| Qwen2.5-1M / Qwen3-Next | ❌ 不需要 | 不在目标模型列表 |

---

### 3.6 文件级改动清单

| 文件 | 改动 | 必要性 |
|---|---|---|
| [layers/rotary_embedding.py](src/micro_vllm/layers/rotary_embedding.py) | 增加 `rope_scaling` 参数，实现 `llama3` 频率缩放；缓存 `inv_freq` | Llama-3.2-1B 必需 |
| [models/llama.py](src/micro_vllm/models/llama.py) | 读取并透传 `rope_scaling` 到 RoPE | Llama-3.2-1B 必需 |
| `models/qwen2.py`（新增） | Qwen2Attention / Qwen2DecoderLayer / Qwen2Model；q/k/v bias，无 q/k norm，o_proj 无 bias | DeepSeek-R1-Distill-Qwen 必需 |
| [models/registry.py](src/micro_vllm/models/registry.py) | 注册 `Qwen2ForCausalLM -> Qwen2Model` | 必需 |
| `layers/moe.py`（新增） | 独立简易 MoE 层（router + top-k + experts + 可选 shared expert） | 任务要求的验证项 |
| `tests/layers/test_moe.py`（新增） | MoE shape/路由/权重/参考实现对拍 | 必需 |
| `tests/layers/test_rotary_embedding.py` | 增加 llama3 rope scaling 与 HF 对拍 | 必需 |
| `tests/models/test_qwen2_*.py`（新增） | Qwen2 初始化、bias、权重加载、logits 对拍 | 必需 |
| `tests/integration/test_loaded_qwen2*.py`（新增） | DeepSeek-R1-Distill-Qwen 端到端 | 建议 |
| [models/qwen.py](src/micro_vllm/models/qwen.py) | 可选：共享参数名 helper，或透传 `rope_scaling` | 可选 |
| [README.md](README.md) / [README_EN.md](README_EN.md) | 更新支持模型列表 | 建议 |
| `core/`、`attention/`、`worker/model_runner.py`、`model_loader/` | **不需要改动** | 目标范围内无 MLA/MoE/量化/cache 形状变化 |

---

### 3.7 建议实施顺序与验证

1. **Llama3.2 RoPE**
   - 修改 `RotaryEmbedding` + `LlamaModel` 参数透传；
   - 单测与 Transformers `LlamaRotaryEmbedding` 对拍；
   - 跑 Llama-3.2-1B greedy 端到端。
2. **Qwen2 模型**
   - 新增 `Qwen2Attention/Qwen2Model` 和 registry；
   - tiny Qwen2 checkpoint 与 Transformers `Qwen2ForCausalLM` 对拍；
   - 跑 DeepSeek-R1-Distill-Qwen-1.5B；显存允许再跑 7B。
3. **回归**
   - 跑现有 Qwen3-0.6B 测试，确认 RoPE 重构没有改变无 scaling 路径；
   - 跑现有 Llama/Mistral 测试，确认 `rope_scaling=None` 行为不变。
4. **简易 MoE**
   - 新增 layer + 单测；
   - 不需要改任何模型或执行引擎。
5. **可选**
   - 增加 README 支持矩阵；
   - 增加 8B/7B 的 GPU 集成测试标记（如果 CI 有足够显存）。

验证命令：

```bash
cd micro-vllm
python -m pytest tests/layers tests/models tests/core
python -m pytest -m gpu tests/integration
```

其中 7B/8B 的真实模型测试建议标记为 `gpu` 且按显存条件跳过；1.5B/1B/0.6B
可以在较小显存上完成端到端验证。

---

### 3.8 工作量与风险总结

| 工作项 | 相对工作量 | 风险 |
|---|---|---|
| `llama3` RoPE scaling | 小 | 中：必须与 HF 频率公式完全一致，否则文本看似正常但位置错误 |
| Qwen2 模型类 | 中 | 中：`q/k/v` 有 bias、`o_proj` 无 bias、无 Q/K norm；bias 加载和 tied embedding 要测试 |
| 简易 MoE | 小 | 低：独立层，单测即可 |
| 现有 Qwen3/Llama 回归 | 小 | 低 |
| 7B/8B 实机验证 | 资源问题 | 中：需要足够显存；不是代码问题 |

**结论**：可行，且比上一版报告的工作量低很多。真正需要的代码改动只有两块：

1. **Llama-3.2-1B 的 `llama3` RoPE scaling**；
2. **新增 Qwen2 模型**（同时覆盖 DeepSeek-R1-Distill-Qwen-1.5B/7B）。

Qwen3-0.6B/8B 不需要改代码；简易 MoE 作为独立 layer + 单测即可，不需要接入模型、
不需要 MLA/MoE 融合、不需要 FP8/TP/EP，也不需要重构 KV cache 或 attention 抽象。

---

## 4. 量化、FlashAttention、投机解码与多卡推理的可行性

这一章回答四个问题：

1. **W4A16** 能不能在这个代码库里实现，应该做到什么范围；
2. **FlashAttention** 能不能作为正式 backend 接入，和现有 paged KV / Triton 路径是什么关系；
3. **投机解码** 需要改哪些核心契约，最小可行路径是什么；
4. **CUDA Graph + NCCL / Tensor Parallel** 的难点、约束和推荐硬件。

先给结论：

| 能力 | 当前状态 | 可行性 | 主要工作量 | 建议优先级 |
|---|---|---|---|---|
| W4A16 | ❌ 完全未实现 | ✅ 可行 | 量化 layer + 权重加载 + 反量化/GEMM kernel + 精度/显存测试 | 高（收益直接，边界清楚） |
| FlashAttention backend | ❌ 未接入官方 kernel | ✅ 可行，但只建议 Full/Sliding | `flash_attn_with_kvcache` 适配 + 元数据/布局处理 + 对拍 | 中（先测收益，再决定是否替换 Triton） |
| 投机解码 | ❌ 未实现 | ✅ 可行，但改动跨 Scheduler/Request/ModelRunner/KV | n-gram + greedy 先做；draft model 后续 | 中（先做 n-gram 验证回滚） |
| CUDA Graph decode | ❌ 未实现 | ✅ 可行，但要求静态 shape / 去掉 host sync | 静态 buffer + 按 batch bucket 捕获 + padding | 中（decode 收益明确，prefill 先不做） |
| NCCL / TP | ❌ 单卡单进程 | ✅ 可行，但属于大改动 | 并行 layer + 权重切分 + 多进程 Executor + KV 分片 | 高（多卡实验的前提） |

最重要的判断：**这四项都不需要推翻现有 Scheduler / Paged KV 架构**，但投机解码和 TP 会触及核心契约；W4A16 和 FlashAttention 可以主要限制在 `layers/` 和 `attention/` 内。

---

### 4.1 W4A16 量化

#### 4.1.1 当前代码的实际状态

当前代码没有任何量化路径：

- `ModelConfig.dtype` 只允许 `float32 / float16 / bfloat16`：
  [src/micro_vllm/config.py:38-42](src/micro_vllm/config.py#L38-L42)
- `Linear` 就是普通 `nn.Linear`，没有 packed weight、scale、zero point：
  [src/micro_vllm/layers/linear.py:6-8](src/micro_vllm/layers/linear.py#L6-L8)
- MLP / Embedding / LMHead 也都是 dense 层：
  [src/micro_vllm/layers/mlp.py:10-21](src/micro_vllm/layers/mlp.py#L10-L21)、
  [src/micro_vllm/layers/embedding.py:8-24](src/micro_vllm/layers/embedding.py#L8-L24)
- 权重加载是“checkpoint tensor 直接 `copy_` 到同名 float parameter”：
  [src/micro_vllm/model_loader/default.py:44-46](src/micro_vllm/model_loader/default.py#L44-L46)、
  [src/micro_vllm/models/llama.py:222-235](src/micro_vllm/models/llama.py#L222-L235)
- KV cache 的 dtype 直接跟随模型 dtype：
  [src/micro_vllm/worker/model_runner.py:418-426](src/micro_vllm/worker/model_runner.py#L418-L426)
- V4 量化需求文档中的所有实现项都未勾选：
  [docs/requirements/v4-quantization.md:19-25](docs/requirements/v4-quantization.md#L19-L25)、
  [docs/requirements/v4-quantization.md:62-75](docs/requirements/v4-quantization.md#L62-L75)

因此不能直接加载 AWQ/GPTQ 的 packed int4 checkpoint：模型构造出来的是 float
`nn.Linear`，checkpoint 里的 `qweight / scales / qzeros` 名称和 shape 都对不上。

#### 4.1.2 推荐的第一版范围

建议第一版明确限定为：

- **W4A16 weight-only**：权重 4 bit，激活保持 FP16/BF16；
- **对称 per-group 量化**，group size 默认 128，可配置 64；
- **从 FP16/BF16 checkpoint 在加载时量化**，暂不直接加载 AWQ/GPTQ；
- **量化 attention 的 q/k/v/o 和 MLP 的 gate/up/down**；
- **不量化 embedding、RMSNorm、bias、lm_head**（tied embedding 时 lm_head 自然保持
  FP16）；
- 保留一个可读的 **reference dequant + matmul** 路径，作为正确性基线；
- 另加一个 optimized backend，但允许在 kernel 不可用时显式回退 reference，不做
  silent fallback。

这样做的好处是：

- 不依赖外部量化 checkpoint 的格式适配，四类目标模型都能统一处理；
- 量化逻辑主要落在 `layers/`，不侵入 Scheduler / KV cache；
- 精度问题可以通过 group size、保留层、校准数据逐步调；
- 后续要接 AWQ/GPTQ，只需要再加一个 checkpoint 适配层。

第一版**不建议**做：

- AWQ/GPTQ/Marlin/Compressed-Tensors 的完整格式支持；
- 混合精度自动搜索；
- 训练感知量化或在线 calibration pipeline；
- FP8 weight/activation、FP8 KV cache；
- MoE expert 量化（等 MoE 层稳定后再加）。

#### 4.1.3 建议的模块设计

建议新增一个独立的量化子包，不要把 `if quantization == ...` 散落到模型类里：

```text
src/micro_vllm/layers/quantization/
├── base.py          # QuantMethod protocol
├── config.py        # QuantizationConfig
├── w4a16.py         # W4A16Linear / pack / unpack / dequant reference
└── kernels.py       # optional optimized GEMM wrapper
```

`QuantizationConfig` 可以放在 `config.py` 或 `layers/quantization/config.py`：

```python
@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    method: str = "none"          # "none" | "w4a16"
    group_size: int = 128
    symmetric: bool = True
    backend: str = "reference"    # "reference" | "optimized"
    quantize_lm_head: bool = False
```

`QuantMethod` 表达三个阶段（与 V4 文档一致）：

```python
class QuantMethod(Protocol):
    def create_weights(self, module, config) -> None: ...
    def load_weight(self, module, name, tensor) -> None: ...
    def apply(self, module, x: Tensor) -> Tensor: ...
```

`W4A16Linear` 的建议字段：

```python
class W4A16Linear(nn.Module):
    in_features: int
    out_features: int
    group_size: int
    qweight: nn.Parameter      # uint8/int32 packed, [out_features, in_features // 2]
    scales: nn.Parameter       # fp16, [out_features, in_features // group_size]
    bias: nn.Parameter | None  # fp16, [out_features] or None
```

量化/反量化语义必须写死并测试：

```text
scale = max(abs(w_group)) / 7
q = clamp(round(w / scale), -8, 7)
w_hat = q * scale
```

如果采用 asymmetric，则额外存 zero point：

```text
scale = (max(w_group) - min(w_group)) / 15
zero = round(-min(w_group) / scale)
q = clamp(round(w / scale) + zero, 0, 15)
w_hat = (q - zero) * scale
```

第一版建议 symmetric；它对 W4A16 足够常见，metadata 更少，kernel 更简单。

#### 4.1.4 权重加载怎么改

当前模型 `load_weights()` 是：

```python
parameters = dict(self.named_parameters())
parameters[parameter_name].copy_(checkpoint_tensor)
```

见 [models/llama.py:222-235](src/micro_vllm/models/llama.py#L222-L235) 和
[models/qwen.py:247-260](src/micro_vllm/models/qwen.py#L247-L260)。

W4A16 需要把“按参数名 copy”改成“按 module 分发”：

1. checkpoint 名称仍然由模型自己的 `_llama_parameter_name` /
   `_qwen3_parameter_name` 解析；
2. 对于 `...q_proj.weight` / `...gate_proj.weight` 这类名字，找到对应的
   `W4A16Linear` 模块；
3. 调用该模块的 `load_weight(tensor)`，在该函数内部完成 group-wise 量化并只保留
   packed int4 + scales；
4. 对于 bias、norm、embedding、lm_head，仍然走原来的 float `copy_`。

这样做有两个好处：

- **不需要先构造完整的 FP16 模型再量化**，避免 8B 模型在 GPU 上出现
  “16 GB float + 4 GB int4”的峰值；
- 量化后的 `qweight` 是整型 parameter，`model.to(device, dtype=...)` 不会把它
  转成 float；`scales` 会被转成目标 dtype。

如果为了快速验证，也可以先实现“加载 FP16 后调用 `model.quantize_()` 替换
Linear”，但这只建议用于 0.6B/1B/1.5B；7B/8B 应使用流式量化，避免峰值内存。

#### 4.1.5 执行 kernel

建议分两层：

**Reference 路径（必须先做）**

```python
def forward(self, x):
    weight = unpack_int4(self.qweight, self.scales, self.group_size)
    return F.linear(x, weight, self.bias)
```

它不追求性能，只用于：

- 验证 pack/unpack、group 边界、bias、dtype；
- 作为 optimized kernel 的数值基线；
- 在 kernel 不可用时显式回退。

**Optimized 路径（二选一）**

1. **PyTorch `_weight_int4pack_mm` / torchao int4mm**
   - PyTorch 2.4+ 提供 `torch._weight_int4pack_mm` 和
     `_convert_weight_to_int4pack`，但它们是私有 API，packed layout 不是公开稳定
     契约；
   - 需要严格校验 `in_features`、`out_features`、group size、dtype、SM 版本；
   - 如果可用，代码量最小，性能也通常好于手写 dequant + matmul；
   - 必须在目标 PyTorch/CUDA 版本上做一次最小验证，再决定是否依赖。

2. **自己写 Triton W4A16 GEMM**
   - 每个 program 计算一个输出 tile；
   - 从 `qweight` 解包 4 bit，按 group 读取 `scales`，反量化后 `tl.dot`；
   - 优点是完全可控、可读、没有私有 API；
   - 缺点是要自己处理 tile、mask、dtype、边界，性能调优成本高。

对教学型仓库，建议顺序是：

```text
reference dequant + matmul
    -> 验证 PyTorch int4pack 在当前环境是否可用
        -> 可用：封装为 optimized backend
        -> 不可用：写一个 correctness-first Triton kernel
```

不要在 optimized kernel 不可用时静默走一条语义不同的路径；要么显式回退
reference，要么在初始化阶段报错。

#### 4.1.6 显存收益估算

W4A16 的权重位数约为：

```text
4 bit + 16 bit / group_size（scale）
```

group size 128 时约 **4.125 bit/weight**，相对 FP16 的 16 bit 约 **3.88x**
权重压缩。实际还要加上不量化的 embedding、lm_head、norm、bias：

| 模型 | FP16/BF16 权重（约） | W4A16 权重（约，group=128） | 说明 |
|---|---:|---:|---|
| Qwen3-0.6B | 1.2 GB | 0.5–0.6 GB | tied embedding 不量化，占比高 |
| Llama-3.2-1B | 2.5 GB | 1.0–1.1 GB | tied embedding 不量化 |
| DeepSeek-R1-Distill-Qwen-1.5B | 3.0 GB | 1.1–1.2 GB | tied embedding 不量化 |
| DeepSeek-R1-Distill-Qwen-7B | 15.2 GB | 5.4–5.6 GB | lm_head 通常 untied |
| Qwen3-8B | 16.4 GB | 6.0–6.2 GB | lm_head 通常 untied |

注意：

- 以上按“**不量化 embedding / lm_head**”计算；如果量化它们，权重还能再降
  1–2 GB，但小模型掉点风险更高，第一版不做；
- 以上只是**权重**，不含 KV cache 和激活；
- KV cache 默认仍是 FP16，不会因为 W4A16 自动变小；
- 7B/8B 在 24 GB 卡上，W4A16 后权重约 5.4–6.2 GB，剩余约 17–18 GB 可给
  KV/激活，小 batch、4K context 可以跑；长 context 或大 batch 仍建议 40–48 GB；
- 如果以后做 FP8 KV cache，KV 还能再减半，但那是另一个独立特性。

#### 4.1.7 精度与测试

W4A16 对小模型更敏感。建议测试分三层：

1. **算子级**
   - pack/unpack round-trip；
   - group 边界（in_features 不整除 group size 时应拒绝，而不是越界）；
   - reference 与 FP16 的 max/mean error；
   - bias、tied/untied、不同 group size。
2. **模型级**
   - 同一 checkpoint 的 FP16 与 W4A16 logits 对比：cosine similarity、top-1
     agreement、最大绝对误差；
   - greedy 生成的 token 一致率；
   - 在固定小语料上的 perplexity 变化；
   - 如果 0.6B/1B 掉点明显，先试 group size 64，或对第一层/最后一层/lm_head
     保留 FP16。
3. **系统级**
   - 权重字节数、峰值显存、prefill/decode 延迟、吞吐；
   - 与 TP 组合时，确认“先切分后量化”的 scale 语义。

#### 4.1.8 与 TP 的关系

W4A16 和 TP 都会改变 weight layout，建议明确顺序：

```text
checkpoint tensor
    -> 按 TP rank 切分（column/row）
    -> 对本地 shard 做 group-wise 量化
    -> 存本地 packed qweight + scales
```

也就是 **先切分、后量化**。这样可以保证：

- 每个 rank 的 group 边界是本地的、连续的；
- 不需要在 rank 之间同步 scale；
- row parallel 的输入维切分后，group size 仍然整除本地 `in_features`。

如果反过来“先量化后切分”，就要同时切 packed nibble 和 scales，跨 group 边界
会产生额外对齐问题，不建议第一版做。

#### 4.1.9 结论

W4A16 是四项里**边界最清楚、最值得先做**的：

- 不改 Scheduler、KV cache、Engine；
- 主要改 `layers/` + 各模型 `load_weights()`；
- 先 reference、再 optimized；
- 目标模型都能覆盖；
- 显存收益明确（7B/8B 约 15–16 GB -> 5.4–6.2 GB 权重）；
- 精度需要用固定阈值和真实模型验证，不能只看“能生成”。

---

### 4.2 FlashAttention 后端

#### 4.2.1 当前为什么没有 FlashAttention

当前有三个 attention backend：

- `reference`：可读 PyTorch 实现；
- `sdpa`：gather paged KV 后调用 `F.scaled_dot_product_attention`；
- `triton_paged`：直接读 paged KV，自己实现 online softmax。

见 [attention/operators/registry.py:13-51](src/micro_vllm/attention/operators/registry.py#L13-L51)。

文档记录了两个关键事实：

- SDPA 实际使用的是 PyTorch memory-efficient CUTLASS kernel，不是
  FlashAttention：
  [docs/implementation/v1-performance-todo.md:248-253](docs/implementation/v1-performance-todo.md#L248-L253)
- 早期实验里 Flash Attention 因“显式 non-null mask”不可用；cuDNN 只支持
  prefill，不支持 query length 1 的 decode：
  [docs/implementation/v1-performance-todo.md:460-463](docs/implementation/v1-performance-todo.md#L460-L463)

这两个限制在**目标模型**上可以绕开：

- 目标 Qwen3 / Llama-3.2 / Distill-Qwen 都是 **full attention**；
- full attention 可以用 `causal=True`，不需要显式 non-null mask；
- decode 的 query length 是 1，FlashAttention 的
  `flash_attn_with_kvcache` 支持这种情况。

所以“FlashAttention 不可用”是当时实现路径的限制，不是目标模型语义的限制。

#### 4.2.2 建议的接入方式

建议新增可选 backend `"flash_attn"`，**不替换** `triton_paged`，先并排对比：

```text
ModelConfig.attention_operator = "reference" | "sdpa" | "triton_paged" | "flash_attn"
```

实现放在：

```text
src/micro_vllm/attention/operators/flash_attn.py
```

依赖作为 optional extra：

```toml
[project.optional-dependencies]
flash = ["flash-attn>=2.6,<3"]
```

并在 registry 中 lazy import，和 `triton_paged` 的做法一致：
[registry.py:27-38](src/micro_vllm/attention/operators/registry.py#L27-L38)。

#### 4.2.3 与现有 paged KV layout 的对应关系

当前物理 KV cache 形状是：

```text
[2, num_blocks, block_size, num_key_value_heads, head_dim]
```

见 [worker/model_runner.py:428-443](src/micro_vllm/worker/model_runner.py#L428-L443)。

FlashAttention 2 的 `flash_attn_with_kvcache` 期望：

```text
k_cache: [num_blocks, page_block_size, num_kv_heads, head_dim]
v_cache: [num_blocks, page_block_size, num_kv_heads, head_dim]
block_table: [batch, max_num_blocks]  (int32)
cache_seqlens: [batch]                (int32)
```

两者几乎一一对应：

```python
k_cache = layer_kv_cache[0]
v_cache = layer_kv_cache[1]
```

`layer_kv_cache[0]` 是连续 view，可以直接传给 FA。当前 `block_table` 是 int64，
FA 通常要求 int32，需要做一次 `to(torch.int32)`（可以缓存成静态 buffer）。

**一个关键约束是 page size**：flash-attn 2.x 的 paged KV kernel 通常要求
`page_block_size` 是 256 的倍数（以实际安装版本为准），而当前默认
`block_size=16`（[config.py:94](src/micro_vllm/config.py#L94)）。因此 FA backend
可能要：

- 为 FA 路径单独使用 `block_size=256`，这会改变 prefix cache 粒度、短请求的
  KV 内部碎片和 benchmark 对比；
- 或者先把 16-token blocks gather 成 FA 能接受的页，但这会多一次拷贝，失去
  “直接读 paged KV”的优势；
- 或者在 CPU 侧把多个 16-token block 映射成 FA 的 page 语义，但这要求
  block table 和 allocation 同时改，复杂度更高。

这个问题必须在写 kernel 之前先用目标 flash-attn 版本做一次最小验证；如果
256 对齐无法满足，`flash_attn` 只能作为“非 paged”实验路径，不能替代
`triton_paged`。

`cache_seqlens` 应该传“包含当前 scheduled token 在内的总 context 长度”，即
`ScheduledRequest.num_computed_tokens + num_scheduled_tokens`。当前
`AttentionMetadata.context_lengths` 正是这个值：
[attention/attention.py:12-28](src/micro_vllm/attention/attention.py#L12-L28)、
[worker/model_runner.py:589-595](src/micro_vllm/worker/model_runner.py#L589-L595)。

#### 4.2.4 full / sliding / chunked local 的支持矩阵

| AttentionSpec | FlashAttention 支持 | 说明 |
|---|---|---|
| `FullAttentionSpec` | ✅ | `causal=True` 即可，无需显式 mask |
| `SlidingWindowSpec` | ✅（需验证版本） | 用 `window_size=(window_size - 1, 0)`；FA2 的 `flash_attn_with_kvcache` 支持 window |
| `ChunkedLocalSpec` | ❌ | `context_start` 是 per-query 的 aligned chunk 边界，不是标准 causal/window；应显式回退 `triton_paged` |

第一版建议只对 `FullAttentionSpec` 开放 `"flash_attn"`；如果模型配置是
`SlidingWindowSpec`，初始化时给出明确错误或要求显式选择
`"triton_paged"`，不要静默改变语义。

#### 4.2.5 prefill / decode / chunked prefill 的处理

**Decode（query length = 1）**

- 一个 batch 内每个请求 1 个 token；
- 可以把所有请求的 query stack 成 `[batch, 1, num_heads, head_dim]`；
- `cache_seqlens` 是每个请求的 context length；
- `block_table` 是 `[batch, max_blocks]`；
- 一次 `flash_attn_with_kvcache` 完成，不需要 Python per-request loop。

**Prefill / chunked prefill（query length > 1）**

- 当前 `query` 是 flatten 的 `[total_tokens, heads, dim]`，每个请求的 query 段长度
  可能不同；
- 第一版可以按请求循环，调用 `flash_attn_with_kvcache`：
  - `q` 取该请求的 `query_start:query_end`，reshape 成
    `[1, q_len, heads, dim]`；
  - `cache_seqlens=[context_length]`；
  - `block_table` 取该请求一行；
  - `causal=True`；
- 这会保留一个 Python per-request loop，和 `sdpa` 类似，但 FA core 更快；
- 如果 benchmark 显示 CPU loop 成为瓶颈，再按 `q_len` 分组或 pad 成矩形 batch。

**Mixed batch**

- 同一 batch 里可能同时有 decode 和 prefill；
- 第一版按请求循环最稳；
- 后续可以按 `q_len` 分组，decode 组走 batched 路径，prefill 组走 per-request
  路径。

#### 4.2.6 KV 写入

有两种做法：

1. **继续用现有 page-write**：先调用 `write_kv_cache()` / `triton_write_kv_cache()`
   写当前 K/V，然后 FA 只读 cache；
2. **把当前 K/V 直接传给 FA**：`flash_attn_with_kvcache(q, k_cache, v_cache, k=key,
   v=value, ...)`，由 FA 负责 append。

第一版建议用第 1 种，因为：

- 复用已经验证的 `slot_mapping` 语义；
- 不需要让 FA 理解 `slot_mapping`；
- page write 和 attention read 的边界清楚，容易和 `triton_paged` 对拍。

#### 4.2.7 需要显式拒绝的配置

- CPU tensor：直接报错；
- `torch.float32`：FA 不支持，报错或要求显式回退；
- `head_dim` 不是 8 的倍数 / 超过 FA 支持范围：报错；
- `block_size` 不满足当前 flash-attn 版本的 page 对齐要求：报错，并提示改用
  `triton_paged` 或调整 block size；
- `ChunkedLocalSpec`：报错或显式回退；
- 没有安装 `flash-attn`：lazy import 时给出可操作的安装提示。

#### 4.2.8 依赖与硬件

- FlashAttention 2 需要 **Ampere 或更新**（SM80+）；T4（SM75）不支持；
- FlashAttention 3 主要面向 Hopper（SM90）；
- 目标 GPU 建议：A100 / L40S / RTX 4090 / H100；
- `flash-attn` 的 wheel 与 PyTorch/CUDA 版本耦合很强，建议：
  - 固定一套已验证的 torch/CUDA/flash-attn 组合；
  - 在 CI 里把 `flash_attn` 测试标记为可选 GPU 测试；
  - 不在 base dependency 中强制安装。

#### 4.2.9 预期收益与判断标准

当前 `triton_paged` 已经比 SDPA 快很多（文档记录的 RTX 5090 上 prefill
约 3.5x、decode 约 4x）：
[docs/implementation/v1-performance-todo.md:241-253](docs/implementation/v1-performance-todo.md#L241-L253)。

因此 FlashAttention 的收益不一定是“数量级”：

- 对 prefill 长序列，FA 的 kernel 可能更好；
- 对 decode 小 batch，launch 开销和 Python loop 可能抵消收益；
- 对 Qwen3-0.6B 这种小模型，attention 占比不一定高；
- 对 7B/8B 长上下文，收益更可能明显。

建议先做 operator-level benchmark：

```text
prefill: prompt 512 / 2048, batch 1 / 4
decode:  batch 1 / 8, context 512 / 2048
对比:    reference / sdpa / triton_paged / flash_attn
指标:    operator latency, end-to-end TTFT/TPOT, 显存
```

只有在目标 workload 上有稳定收益，才考虑把 `flash_attn` 设为默认 backend；
否则保留为可选实验后端。

#### 4.2.10 结论

FlashAttention 接入是**可行的**，但要注意：

- 目标模型是 full attention，所以可以绕开当年的 non-null mask 限制；
- 现有 paged KV layout 和 FA2 的 `flash_attn_with_kvcache` 基本匹配；
- 第一版按请求循环最稳，batched decode 可以单独优化；
- ChunkedLocal 不支持，必须显式回退；
- T4 不支持 FA2，需要 Ampere+；
- 先和 `triton_paged` 对拍、测收益，不要默认替换。

---

### 4.3 投机解码

#### 4.3.1 当前代码为什么不能直接加

当前整条链路都是“每个请求每步最多采一个 token”：

- `ScheduledRequest` 只有 `requires_sampling` 一个采样标记：
  [core/scheduler_output.py:18-33](src/micro_vllm/core/scheduler_output.py#L18-L33)
- Scheduler 只在 `end_computed == num_known_tokens` 时标记采样：
  [core/scheduler.py:242-258](src/micro_vllm/core/scheduler.py#L242-L258)
- ModelRunner 只对 `sampling_row_indices` 选中的行计算 logits，并交给 Sampler
  采一个 token：
  [worker/model_runner.py:205-238](src/micro_vllm/worker/model_runner.py#L205-L238)
- `ModelRunnerOutput` 只有 `SampledToken`，没有 accepted/rejected：
  [executor/interface.py:9-24](src/micro_vllm/executor/interface.py#L9-L24)
- `Request` 只有 prompt + output，没有 draft token 的状态：
  [core/request.py:75-103](src/micro_vllm/core/request.py#L75-L103)
- KV cache 只有 `allocate` / `commit` / `free`，没有按 token 截断/回滚：
  [core/kv_cache_manager.py:159-287](src/micro_vllm/core/kv_cache_manager.py#L159-L287)

V5 文档已经给出了正确的设计原则：draft token 是“已知但尚未由 target model
计算”的 token，Scheduler 仍然用 `num_known_tokens - num_computed_tokens` 统一
记账，不引入三套 phase 状态机：
[docs/requirements/v5-speculative-decoding.md:15-21](docs/requirements/v5-speculative-decoding.md#L15-L21)。

#### 4.3.2 推荐的最小可行路径

不要一上来做 draft model。建议分四步：

1. **n-gram drafter + greedy acceptance**
   - 不增加模型、不增加显存；
   - 可以先验证 Request/Scheduler/ModelRunner/KV rollback 的完整链路；
   - 输出必须和普通 greedy 完全一致。
2. **draft model + greedy acceptance**
   - 增加一个独立 draft runner 和 draft KV cache；
   - 目标模型验证，接受/拒绝后同步/回滚 draft KV。
3. **stochastic rejection sampling**
   - 需要 proposal probability 和 target distribution；
   - 先在小词表 tiny model 上做分布测试。
4. **variable proposal length batching**
   - 同一 batch 中不同请求的 K 不同；
   - 这是最后的性能优化，不是正确性前提。

#### 4.3.3 数据结构建议

`Request` 增加：

```python
draft_token_ids: list[int]          # 尚未被 target 验证的 proposal
num_known_tokens = prompt + output + draft
all_token_ids = prompt + output + draft
```

增加方法：

```python
def append_draft_tokens(self, token_ids: Sequence[int]) -> None: ...
def accept_draft_tokens(self, num_accepted: int) -> None: ...
def clear_draft_tokens(self) -> None: ...
```

`SchedulerOutput` / `ScheduledRequest` 增加：

```python
num_draft_tokens: int
verification_rows: tuple[int, ...]   # 每个 proposal 对应的 hidden row
```

`ModelRunnerOutput` 增加：

```python
verification_results: tuple[VerificationResult, ...]
```

`VerificationResult` 至少包含：

```python
request_id: str
accepted_token_ids: tuple[int, ...]
replacement_token_id: int | None
extra_token_id: int | None
num_proposed: int
num_accepted: int
```

#### 4.3.4 Scheduler / ModelRunner 的验证语义

假设请求当前已有 `P` 个已知且已计算 token，drafter 追加 `K` 个 proposal：

```text
prompt + accepted output: 长度 P
draft tokens:             长度 K
target forward 输入:      P..P+K-1 这 K 个 token
```

target 每个位置的 logits 用来验证对应 draft token：

```text
position P + i 的 target argmax  vs  draft_token[i]
```

规则（greedy）：

- 从 i=0 开始，连续匹配的 draft token 接受；
- 第一个不匹配的位置 i：接受前 i 个，使用 target argmax 作为 replacement，
  丢弃 i 之后的 proposal；
- 如果 K 个全部接受：再追加 target 在最后一个位置的额外 token；
- EOS / max_tokens / max_model_len 在 accepted / replacement / extra 中都要检查。

因此一次 target forward 最多可以产出 `K+1` 个 token。

`scheduled_request.num_scheduled_tokens` 应该等于 `K`（proposal 数），而不是
`K+1`；额外的 token 是 target 在最后一个位置“顺便”预测出来的，写入
`request.output_token_ids`，但它的 KV 要到下一步才写入 cache。

#### 4.3.5 KV cache 回滚

这是投机解码最容易出错的部分。需要在 `KVCacheManager` 增加：

```python
def truncate_request(self, request_id: str, num_computed_tokens: int) -> None: ...
```

语义：

- 保留 `num_computed_tokens` 之前的完整/部分 block；
- 释放完全在保留位置之后的 block；
- 保留最后一个 partial block，但其中超过 `num_computed_tokens` 的 slot 不再被
  引用；
- 不修改仍被其他 request 共享的 prefix block（只减 refcount）；
- 只有 accepted prefix 的完整 block 才能进入 prefix cache。

当前 `BlockTable` 有 `release_before()` 和 `clear()`，但没有“从某个位置向后
释放”的方法：
[core/block.py:222-247](src/micro_vllm/core/block.py#L222-L247)。
需要新增 `truncate(num_logical_blocks)` 或 `release_from()`。

执行顺序建议：

```text
target forward 写入 K 个 draft token 的 KV
    -> acceptance 计算
    -> request 只保留 accepted prefix
    -> KVCacheManager.truncate_request(accepted_prefix)
    -> commit_step(accepted_prefix)   # 只缓存 accepted 的完整 block
    -> 追加 replacement / extra token
```

如果先 `commit_step` 再回滚，就可能把 rejected token 的完整 block 写进 prefix
cache，后续请求会命中错误内容。因此 **先回滚、后 commit**。

#### 4.3.6 与 prefix cache / preemption 的交互

必须专门测试：

- rejected token 落在 partial block / full block / 跨 block boundary；
- rollback 后继续生成，与从正确 prefix 重跑的结果一致；
- prefix cache on/off 下 greedy 输出一致；
- preemption + rollback 不 double free、不泄漏 block；
- 共享 prefix block 的 refcount 正确；
- 请求 abort 时同时清理 target 和 draft 状态。

现有 `commit_step` 是按 `num_computed_tokens` 缓存完整 block 的：
[core/kv_cache_manager.py:224-274](src/micro_vllm/core/kv_cache_manager.py#L224-L274)。
只要保证传入的是 accepted prefix，语义就能保持一致。

#### 4.3.7 n-gram drafter

第一版建议实现最简单的 suffix match：

```text
在 request 的 token history 中查找当前 suffix 的最长匹配
    -> 取匹配位置之后的最多 K 个 token 作为 proposal
    -> 没有匹配则返回空
```

优点：

- 不占 GPU 显存；
- 不需要 draft model；
- 对重复性强的 prompt / 代码 / 格式化输出可能有效；
- 可以先验证 scheduler + rollback 的正确性。

缺点：

- 对普通自然语言命中率低；
- 无法提供 proposal probability，所以第一版只支持 greedy；
- 没有匹配时就是普通 decode，不能变慢太多。

建议先实现它，作为投机解码的“最小正确闭环”。

#### 4.3.8 draft model drafter

第二版增加：

- 一个独立的 `DraftModelRunner`（可以是现有 `ModelRunner` 的复用）；
- 独立的 draft KV cache；
- `propose(request_tokens, K)` 自回归生成 K 个 token；
- target accepted n 个后，draft KV 截断到 accepted prefix；
- 如果 replacement token 与 draft token 不同，draft KV 需要回滚到 accepted
  prefix，再 forward replacement token；
- draft model 的 tokenizer / vocab 必须与 target 一致，启动时校验，不一致直接
  拒绝。

目标模型配对建议：

| Target | 可用 drafter | 说明 |
|---|---|---|
| Qwen3-8B | Qwen3-0.6B | 同 tokenizer/vocab，最自然 |
| Qwen3-0.6B | n-gram | 没有更小的同族模型；需验证 Qwen2.5-0.5B tokenizer 是否完全兼容 |
| Llama-3.2-1B | n-gram | 没有更小的 Llama 3.2；TinyLlama tokenizer 不兼容 |
| Distill-Qwen-1.5B | Qwen2.5-0.5B | 同 Qwen2 tokenizer/vocab，需实测 |
| Distill-Qwen-7B | Distill-Qwen-1.5B / Qwen2.5-0.5B | 1.5B 接受率更高但更占显存 |

draft model 的显存开销：

- Qwen3-0.6B：约 1.2 GB 权重 + draft KV；
- Qwen2.5-0.5B：约 1 GB 权重 + draft KV；
- 对 8B target，FP16 下 16.4 + 1.2 = 17.6 GB，再加 target KV/激活；24 GB 卡
  只能小 batch，40 GB 更稳；
- W4A16 target 后，8B 权重约 6.1 GB + draft 1.2 GB，24 GB 卡会舒服很多。

#### 4.3.9 stochastic acceptance

greedy 之外，V5 还要求 stochastic rejection sampling。它需要：

- draft 每个 proposal 的概率 `p_draft(x)`；
- target 对应位置的完整分布 `p_target(x)`；
- 接受概率 `min(1, p_target(x) / p_draft(x))`；
- 拒绝时从 residual distribution
  `normalize(max(p_target - p_draft, 0))` 采样 replacement；
- temperature / top-k / top-p 与 speculative sampling 的作用顺序要明确；
- per-request RNG 可复现。

这部分建议放在 greedy 和 draft model 都稳定之后。n-gram 没有概率，第一版只支持
greedy；draft model 可以返回 top-k 概率，再做 stochastic。

#### 4.3.10 指标与测试

必须记录：

- proposed / accepted / rejected token 数；
- acceptance length histogram、acceptance rate；
- target forward 次数、每次验证的 proposal 数；
- draft latency、verification latency、rollback 次数；
- 端到端 throughput、TTFT、TPOT/ITL、显存。

正确性测试：

- 全接受、首 token 拒绝、中间拒绝、空 proposal；
- proposal 中出现 EOS、超过 max_tokens、跨 block boundary；
- greedy speculative 与普通 greedy 输出完全一致；
- rollback 后继续生成与从正确 prefix 重跑一致；
- prefix cache + speculative 不缓存未提交 token；
- preemption + speculative 无 block 泄漏；
- mixed batch 中普通 decode / prefill / verification 互不污染；
- draft model tokenizer/vocab 不匹配时启动失败。

#### 4.3.11 结论

投机解码**可行**，但它不是“加一个 drafter”这么简单：

- Scheduler / Request / ModelRunner / KV cache 都要改；
- KV rollback 是正确性核心；
- 建议先做 n-gram + greedy，验证完整闭环；
- 再做 draft model；
- 最后做 stochastic；
- 目标模型里，Qwen3-8B + Qwen3-0.6B 和 Distill-7B + Distill-1.5B 最有希望；
  Llama-3.2-1B 更适合 n-gram；
- 小模型上投机解码可能因为 draft 开销而变慢，必须用 acceptance rate 和端到端
  数据判断。

---

### 4.4 多卡：CUDA Graph 解码与 NCCL / Tensor Parallel

#### 4.4.1 CUDA Graph 解码

**当前代码的 graph 阻塞点**

`ModelRunner` 已经预分配了一些 scratch buffer：
[worker/model_runner.py:110-120](src/micro_vllm/worker/model_runner.py#L110-L120)，
但还不足以直接 capture：

1. **动态 shape**
   - `_flatten_model_input` 每步从 Python list 构造 `input_ids` /
     `positions` / `sampling_row_indices`：
     [worker/model_runner.py:573-642](src/micro_vllm/worker/model_runner.py#L573-L642)
   - `_build_block_table` 的宽度取决于当前 batch 的最大 block 数：
     [worker/model_runner.py:743-771](src/micro_vllm/worker/model_runner.py#L743-L771)
   - `_build_slot_mapping` 每步动态 `torch.arange`：
     [worker/model_runner.py:682-710](src/micro_vllm/worker/model_runner.py#L682-L710)
2. **host sync**
   - `Attention.build_metadata` 用 `query_positions.tolist()` 在 CPU 上算
     context start：
     [attention/attention.py:44-76](src/micro_vllm/attention/attention.py#L44-L76)
   - `reference` / `sdpa` 里有 `.item()`：
     [attention/operators/sdpa.py:95-108](src/micro_vllm/attention/operators/sdpa.py#L95-L108)
3. **动态分配**
   - `reference` / `sdpa` 的 `gather_kv_cache`、`torch.empty_like`、dense mask
     都会分配新 tensor；
   - `triton_paged` 相对更适合 capture，但仍有 `torch.empty_like(query)` 和
     Python 侧的 grid 计算。
4. **采样**
   - `Sampler._sample_row` 逐行 `.item()`，并创建 `torch.Generator`：
     [sampling/sampler.py:30-71](src/micro_vllm/sampling/sampler.py#L30-L71)
   - 采样不应该放进 graph；graph 只包 model forward。

**推荐设计**

第一版只做 **decode graph**：

- 只支持 `FullAttentionSpec` + `triton_paged`（或 `flash_attn`）；
- 固定 `max_num_seqs` 和 `max_model_len`；
- 按 batch size bucket 捕获，例如 `{1, 2, 4, 8}`；
- 每个 graph 的输入都是静态 buffer：
  - `input_ids`
  - `positions`
  - `slot_mapping`
  - `block_table`（固定宽度 `max_model_len // block_size`）
  - `query_start_locations`
  - `context_lengths`
  - `query_context_start_positions`（full attention 全 0）
- 每步只做 `copy_` 到静态 buffer，然后 `graph.replay()`；
- `compute_logits` 和 sampling 放在 graph 外；
- padding 请求使用 null block / dummy token，输出忽略；
- prefill 先不做 graph（token 数变化太大），后续再按 chunk size bucket。

**需要改的地方**

- `AttentionSpec` 增加向量化的 `context_starts(positions)`，不要在 Python 里
  `tolist()`；
- `ModelRunner` 增加静态 metadata 路径，避免每步创建新 tensor；
- `block_table` 始终使用固定宽度，padding 填 null block；
- `sampling_row_indices` 放进静态 buffer，或在 graph 外做 index select；
- 新增 `CUDAGraphConfig` / `ModelConfig.cuda_graph: bool`；
- graph capture 放在 `initialize_kv_cache()` 之后，warmup 之后；
- 关闭 graph 时完全走原 eager 路径。

**测试**

- graph on/off 的 greedy token 完全一致；
- 不同 batch bucket、不同 context length、跨 block；
- padding 请求不污染真实请求；
- preemption / prefix cache 不回归；
- 显存峰值记录（graph pool 会额外占用）。

**硬件**

CUDA Graph 本身不挑 GPU，T4 也可以；真正的限制是：

- 需要 CUDA；
- 需要 attention backend 没有 host sync / 动态 shape；
- 显存要能容纳 graph pool 和静态 buffer。

#### 4.4.2 NCCL / Tensor Parallel

**当前状态**

- `LLM` 只创建一个 `Worker` / `Scheduler` / `ModelRunner`：
  [llm.py:50-69](src/micro_vllm/llm.py#L50-L69)
- `Worker` 直接构造 `ModelRunner`：
  [worker/worker.py:10-21](src/micro_vllm/worker/worker.py#L10-L21)
- `SingleProcessExecutor` 只转发给一个 worker：
  [executor/single_process.py:8-23](src/micro_vllm/executor/single_process.py#L8-L23)
- 模型层全部是普通 `Linear`：
  [layers/linear.py:6-8](src/micro_vllm/layers/linear.py#L6-L8)
- KV cache 形状按全局 `num_key_value_heads` 分配：
  [worker/model_runner.py:428-443](src/micro_vllm/worker/model_runner.py#L428-L443)
- 权重加载直接 copy 全量 tensor：
  [models/llama.py:222-235](src/micro_vllm/models/llama.py#L222-L235)

**推荐的最小 TP 范围**

第一版建议只做 **TP，不做 PP/DP/EP**：

- 一个进程一张 GPU；
- rank 0 运行 Engine / Scheduler；
- 所有 rank 执行同一个 `SchedulerOutput`；
- 只做 Transformer layer 的 tensor parallel；
- embedding 和 lm_head **复制**（不切 vocab），降低第一版复杂度；
- sampling 只在 rank 0 做；
- KV cache 按 head 切分到各 rank；
- 暂不支持 TP > `num_key_value_heads` 的 GQA 复制策略，直接给出清晰错误。

**Layer 切分**

新增：

```text
src/micro_vllm/parallel/
├── config.py        # ParallelConfig(tp_size)
├── context.py       # ParallelContext(rank, world_size, local_rank, device, group)
├── collectives.py   # all_reduce / broadcast 封装
├── linear.py        # ColumnParallelLinear / RowParallelLinear
└── executor.py      # MultiprocessExecutor
```

切分规则：

| 模块 | 切分方式 |
|---|---|
| `q_proj` | column parallel，按 query head 切 output |
| `k_proj` / `v_proj` | column parallel，按 KV head 切 output |
| `o_proj` | row parallel，按 input 切 |
| `gate_proj` / `up_proj` | column parallel，切 intermediate |
| `down_proj` | row parallel，切 intermediate input |
| `embed_tokens` | 复制 |
| `lm_head` | 复制（第一版） |
| `RMSNorm` | 复制 |
| KV cache | 每个 rank 只存本地 KV heads |

**GQA 约束**

必须满足：

```text
tp_size <= num_key_value_heads
num_attention_heads % tp_size == 0
num_key_value_heads % tp_size == 0
hidden_size % tp_size == 0
intermediate_size % tp_size == 0
```

对目标模型：

| 模型 | q heads | kv heads | 支持的 TP（不复制 KV） |
|---|---:|---:|---|
| Qwen3-0.6B | 16 | 8 | 1/2/4/8 |
| Qwen3-8B | 32 | 8 | 1/2/4/8 |
| Llama-3.2-1B | 32 | 8 | 1/2/4/8 |
| Distill-Qwen-1.5B | 12 | 2 | 1/2 |
| Distill-Qwen-7B | 28 | 4 | 1/2/4 |

所以：

- 8B / 1B / 0.6B 可以做到 TP=8；
- Distill-1.5B 最多 TP=2；
- Distill-7B 最多 TP=4；
- 如果一定要 TP > kv heads，需要复制 KV head，并让多个 rank 计算同一份 KV，
  这会增加显存和通信，第一版不建议。

**权重加载**

模型 `load_weights()` 需要知道当前 rank 的 shard：

- column parallel：`tensor.chunk(tp_size, dim=0)[rank]`；
- row parallel：`tensor.chunk(tp_size, dim=1)[rank]`；
- bias 跟随 column 切分；
- embedding / lm_head / norm：复制；
- q/k/v 的 output 切分必须按 head 对齐，不能简单按任意 dim 切；
- packed QKV 不在第一版目标模型里；如果以后支持，需要按 head 解析。

加载顺序与 W4A16 组合时，遵循“先切分、后量化”。

**KV cache**

`ModelRunner._kv_cache_shape()` 目前直接读全局 config：
[worker/model_runner.py:428-443](src/micro_vllm/worker/model_runner.py#L428-L443)。

TP 下必须改成使用 **本地 KV heads**：

```python
local_kv_heads = global_kv_heads // tp_size
shape = (2, num_blocks, block_size, local_kv_heads, head_dim)
```

同时：

- 所有 rank 的 `num_blocks` 必须一致，建议初始化时
  `all_reduce(MIN)` 取各 rank 可用容量的下界；
- block table / slot mapping 在所有 rank 上完全一致；
- rank 0 的 `KVCacheManager` 分配逻辑不变；
- `physical_kv_bytes` benchmark 也要改成 per-rank local heads。

**多进程执行**

建议新增 `MultiprocessExecutor`，仍然实现现有 `Executor` protocol：
[executor/interface.py:27-38](src/micro_vllm/executor/interface.py#L27-L38)。

初始化流程：

```text
rank 0 创建 ParallelContext / TCP store
    -> spawn ranks 1..N-1
    -> 所有 rank init_process_group("nccl")
    -> 每个 rank 构造 ModelRunner / load shard / init local KV
    -> all_reduce 得到公共 num_blocks
    -> rank 0 构造 Scheduler + KVCacheManager
```

每步：

```text
rank 0 Scheduler.schedule()
    -> broadcast SchedulerOutput 到所有 rank
    -> 每个 rank Worker.execute(plan)
    -> rank 0 得到 ModelRunnerOutput 并 Scheduler.update()
    -> 其他 rank 不需要返回 sampled token
```

需要处理的细节：

- `SchedulerOutput` 必须可 pickle；当前 frozen dataclass 可以；
- 所有 rank 的 request 顺序、block ID、positions 必须一致；
- rank 0 采样后，下一步 plan 会包含新 token，其他 rank 通过 plan 获得；
- worker 异常要能传回 rank 0，并清理所有子进程；
- `shutdown()` 要幂等；
- 单卡 `tp_size=1` 继续走 `SingleProcessExecutor`，不初始化 NCCL。

**采样**

第一版建议复制 lm_head：

- 每个 rank 的 logits 应该一致（模型权重 shard + all_reduce 后）；
- 只在 rank 0 采样；
- 不需要 vocab all-gather；
- 如果以后做 vocab parallel，需要 all-gather logits 或 distributed argmax。

**测试**

- 单层 Column/Row Parallel 与 dense linear 对拍；
- TP=1 vs TP=2/4 的 tiny model logits / greedy token 一致；
- GQA / MQA 的 head 切分；
- KV cache 每 rank local heads，block table 一致；
- prefix cache / preemption / chunked prefill 在 TP 下继续通过；
- 子进程异常能退出，不挂起；
- NCCL 初始化失败有明确错误；
- 多卡 benchmark 记录每 rank 显存和通信时间。

**工作量判断**

TP 是四项里最大的改动：

- 并行 layer + 权重切分：中；
- 多进程 Executor / NCCL 生命周期：大；
- KV cache / ModelRunner local head：中；
- 测试与调试：大；
- 与 W4A16 / CUDA Graph 组合：更大。

建议顺序：

```text
单卡 W4A16 / FA / spec 稳定
    -> TP=1 抽象先落地（ParallelContext, 单卡不初始化 NCCL）
    -> TP=2 最小闭环
    -> TP=4/8 + benchmark
    -> 再考虑 CUDA Graph + TP
```

#### 4.4.3 CUDA Graph + TP 的组合

两者可以组合，但建议分开验证：

- CUDA Graph 捕获 NCCL collective 需要 NCCL 支持 graph capture；
- 所有 rank 必须同时 capture / replay；
- graph 的静态 buffer 在每个 rank 上都要有；
- 任何一个 rank 的 shape / batch bucket 不一致都会死锁；
- 建议先 TP eager 通过，再对 decode graph 做 TP 捕获；
- 不要在同一个 PR 里同时引入 TP、graph、quantization 三个变量。

---

### 4.5 显卡型号与显存建议

以下数字是**近似值**，用于选卡；真实值取决于模型 config、batch、context、
activation、KV block 浪费和框架开销。

#### 4.5.1 目标模型

KV per token 的计算：

```text
2 * num_layers * num_key_value_heads * head_dim * bytes_per_element
```

FP16 下：

| 模型 | 参数量（约） | FP16 权重 | KV/token | 4K context KV | W4A16 权重（约） |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.6B | 1.2 GB | 112 KiB | 448 MiB | 0.5–0.6 GB |
| Llama-3.2-1B | 1.24B | 2.5 GB | 32 KiB | 128 MiB | 1.0–1.1 GB |
| Distill-Qwen-1.5B | 1.5B | 3.0 GB | 28 KiB | 112 MiB | 1.1–1.2 GB |
| Distill-Qwen-7B | 7.6B | 15.2 GB | 56 KiB | 224 MiB | 5.4–5.6 GB |
| Qwen3-8B | 8.2B | 16.4 GB | 144 KiB | 576 MiB | 6.0–6.2 GB |
| Qwen2.5-0.5B（draft） | 0.5B | 1.0 GB | 12 KiB | 48 MiB | — |
| Qwen3-0.6B（draft） | 0.6B | 1.2 GB | 112 KiB | 448 MiB | — |

#### 4.5.2 单卡建议

| 目标 | 最低 | 舒适 | 说明 |
|---|---|---|---|
| Qwen3-0.6B / Llama-1B / Distill-1.5B，FP16 | 8 GB | 16 GB | T4 16GB 可跑，但无 FA2 |
| 同上，W4A16 | 6 GB | 12–16 GB | 6GB 卡只能很小 batch/context |
| Distill-7B / Qwen3-8B，FP16 | 24 GB（紧） | 40–48 GB | 24GB 只适合小 batch、短 context |
| Distill-7B / Qwen3-8B，W4A16 | 16 GB（紧） | 24 GB | 24GB 可覆盖中小 batch；长上下文仍建议 40GB+ |
| 8B + 0.6B draft，FP16 | 24 GB（紧） | 40–48 GB | draft 权重 + target KV |
| 8B + 0.6B draft，W4A16 | 24 GB | 24–40 GB | 权重约 7.3–7.5 GB |

注意：

- 24 GB 跑 8B FP16 时，4K context、batch 1 大约 18–19 GB；batch 8 或 8K
  context 很容易超过 24 GB；
- 7B/8B 的 W4A16 权重虽小，但 KV 和 activation 仍在，长上下文/大 batch 不能
  只看权重；
- 投机解码的 draft model 也有 KV，长 context 时不能忽略。

#### 4.5.3 推荐 GPU 型号

| GPU | 显存 | 架构 | FA2 | W4A16 optimized | 适用 |
|---|---:|---|---|---|---|
| Tesla T4 | 16 GB | Turing SM75 | ❌ | ❌（优化 kernel 通常要求 SM80+） | 现有 baseline、0.6B/1B/1.5B 小 batch |
| RTX 3060 Laptop（本机） | 6 GB | Ampere SM86 | ✅（需装） | ✅（需验证 kernel） | 0.6B/1B/1.5B 小 batch、W4A16 实验 |
| RTX 4090 | 24 GB | Ada SM89 | ✅ | ✅ | 小模型、7B/8B W4A16 小 batch |
| L40S | 48 GB | Ada SM89 | ✅ | ✅ | 7B/8B FP16 单卡、W4A16 舒适 |
| A100 40GB | 40 GB | Ampere SM80 | ✅ | ✅ | 7B/8B 单卡（40GB 偏紧）、TP |
| A100 80GB | 80 GB | Ampere SM80 | ✅ | ✅ | 8B 长上下文、TP 2/4 |
| H100 80GB | 80 GB | Hopper SM90 | ✅ / FA3 | ✅ | 8B、TP、FA3 |
| H200 141GB | 141 GB | Hopper | ✅ / FA3 | ✅ | 大模型/长上下文（本项目偏富余） |

#### 4.5.4 租卡建议

按目标分：

1. **先跑通 baseline + 小模型 + W4A16 + FA2**
   - 1x L40S 48GB 或 1x A100 40GB；
   - 覆盖 Qwen3-0.6B、Llama-1B、Distill-1.5B 的 FP16/W4A16；
   - 7B/8B 可以跑 W4A16 小 batch；
   - L40S 48GB 比 A100 40GB 更宽松，性价比较好。

2. **7B/8B FP16 长上下文**
   - 1x A100 80GB 或 H100 80GB；
   - 24GB 卡只建议 W4A16 + 小 batch。

3. **TP=2/4/8**
   - 优先 A100 80GB / H100 80GB，且同型号、NVLink 互联；
   - 预算有限可以用 2x/4x L40S（PCIe），但 all-reduce 会慢；
   - RTX 4090 多卡没有 NVLink，TP 只适合做正确性实验，不适合作为性能结论。

4. **FA3**
   - 只有 H100/H200 才有意义；A100/L40S/4090 用 FA2。

5. **本地 RTX 3060 6GB**
   - 可以验证 0.6B/1B/1.5B 的 W4A16、FA2、CUDA Graph；
   - 不能验证 7B/8B；
   - 不能做 TP 性能结论。

选择时还要确认：

- 云厂商镜像的 CUDA / PyTorch 版本是否有对应 `flash-attn` wheel；
- 多卡实例是否真的 NVLink，而不是只有 PCIe；
- 显存是整卡还是 MIG 切片；
- 同一 TP group 的 GPU 型号、驱动、CUDA 版本一致。

---

### 4.6 压测基线怎么设计

#### 4.6.1 当前 benchmark 的问题

当前 `benchmarks/benchmark_engine.py` 是一个很好的起点，但它被冻结在：

- 固定模型 `Qwen/Qwen3-0.6B`：
  [benchmarks/benchmark_engine.py:37-44](benchmarks/benchmark_engine.py#L37-L44)
- 固定 `reference` attention；
- 固定 T4 检查：
  [benchmarks/benchmark_engine.py:571-576](benchmarks/benchmark_engine.py#L571-L576)
- 固定 6 个 workload；
- 只测 TTFT / ITL / throughput / peak memory / KV pages：
  [benchmarks/benchmark_engine.py:449-524](benchmarks/benchmark_engine.py#L449-L524)

它不能直接用于 W4A16 / FA / TP / graph 的矩阵对比。

#### 4.6.2 建议先做的 baseline 改造

1. **参数化**
   - `--model` / `--revision` / `--dtype`；
   - `--attention-operator`；
   - `--quantization none|w4a16` + group size/backend；
   - `--tp-size`；
   - `--cuda-graph`；
   - `--speculative none|ngram|draft` + K；
   - `--prefix-cache`；
   - `--batch-size` / `--prompt-len` / `--output-len` / `--num-requests`；
   - `--warmup` / `--repetitions` / `--output`。

2. **workload 矩阵**
   - prefill-only：prompt 128 / 512 / 2048，output 1；
   - decode-only：prompt 128，output 128 / 512；
   - mixed：continuous arrival，不同 prompt/output 混合；
   - prefix cache：共享前缀 512 / 2048，N 个请求；
   - spec：target/draft 组合，K=2/4/6；
   - 并发：1 / 2 / 4 / 8 / 16；
   - TP：1 / 2 / 4 / 8。

3. **指标定义**
   - **TTFT**：从请求被 admit 到第一个 output token 的时间；
   - **TPOT**：第一个 token 之后，每个 output token 的平均时间；当前 benchmark
     只有 ITL，TPOT 可以用 `mean(ITL)` 表示；
   - **ITL**：相邻 token 时间间隔，报 p50/p90/p99；
   - **throughput**：request/s、output token/s、total token/s；
   - **显存**：weights、KV、peak allocated、peak reserved、free/total；
   - **KV utilization**：峰值使用页 / 可用数据页；
   - **spec**：proposal 数、accepted 数、acceptance rate、平均接受长度；
   - **TP**：每 rank 显存、all-reduce 时间、通信占比。

   当前 `_RequestClock.observe` 对同一个 step 内多个 token 使用同一个
   `now_seconds`：
   [benchmarks/benchmark_engine.py:213-224](benchmarks/benchmark_engine.py#L213-L224)。
   普通 decode 没问题；投机解码一次可能产出多个 token，需要额外记录“每个
   target step 的 accepted length”，否则 ITL 会被低估。

4. **测量方法**
   - 模型加载、warmup、KV 初始化不计入测量；
   - 每个配置至少 3–5 次重复，报 median 和 p90；
   - 端到端用 `time.perf_counter` + `torch.cuda.synchronize`；
   - kernel 级用 CUDA events；
   - 固定 seed、greedy、固定 token IDs；
   - 记录 GPU 型号、驱动、CUDA、PyTorch、flash-attn、Triton 版本；
   - 记录是否开 prefix cache、block size、max_num_seqs、max_num_batched_tokens。

5. **baseline 保存**
   - 每个 GPU / 模型 / dtype / backend 组合保存一份 JSON；
   - 目录建议 `benchmarks/results/baseline-<gpu>-<date>/`；
   - 后续每个优化都和对应 baseline 对比，而不是跨 GPU 比较。

#### 4.6.3 建议的验收指标

- W4A16：权重显存下降 ≥ 3x；吞吐不低于 FP16 的 0.9x（目标 1.0x+）；logits /
  token 退化在阈值内；
- FlashAttention：在目标 workload 上 operator latency 或端到端吞吐有稳定收益；
  否则保留为可选；
- 投机解码：acceptance rate 和 target forward 减少量可解释；端到端不劣化；
- CUDA Graph：decode 小 batch 的 CPU overhead 明显下降；token 完全一致；
- TP：TP=2/4/8 的吞吐/延迟 scaling 曲线；与 TP=1 greedy 输出一致；
- 所有功能关闭时，原 reference 路径性能和输出不回归。

---

### 4.7 推荐实施顺序

考虑到“压测基线是所有工作的前提”，建议按下面顺序：

**M0：基线与测量框架（先做）**

- 参数化 benchmark；
- 在目标 GPU 上跑 Qwen3-0.6B / Llama-1B / Distill-1.5B / 7B / 8B 的
  FP16 baseline；
- 保存 TTFT / TPOT / throughput / 显存 / KV 数据；
- 确认模型能加载、tokenizer 正常、greedy 输出稳定。

**M1：模型适配（Part 3）**

- Llama-3.2 `rope_scaling`；
- Qwen2 模型；
- 独立 MoE layer + 单测；
- 这一步是后续所有模型对比的前提。

**M2：W4A16**

- reference 量化 + 反量化；
- 流式权重加载；
- 精度/显存/吞吐测试；
- 再决定 optimized kernel（PyTorch int4pack 或 Triton）。

**M3：FlashAttention**

- optional backend；
- Full/Sliding 支持，ChunkedLocal 回退；
- 与 `triton_paged` 对拍和 benchmark；
- 只有稳定收益才考虑默认。

**M4：投机解码**

- n-gram + greedy + KV rollback；
- 再上 draft model；
- 最后 stochastic；
- 每一步都要有“与普通 greedy 完全一致”的测试。

**M5：CUDA Graph decode**

- 静态 buffer + batch bucket；
- 先只支持 `triton_paged` / full attention；
- graph on/off 对拍；
- 再考虑 prefill graph。

**M6：TP**

- `ParallelContext` + 单卡抽象；
- TP=2 最小闭环；
- TP=4/8；
- NCCL 生命周期、KV 分片、权重切分；
- 最后再做 TP + CUDA Graph。

**M7：组合矩阵**

- W4A16 × FA × spec × graph × TP；
- 按 4.6 的 workload 矩阵跑；
- 输出完整报告和选型建议。

依赖关系：

```text
M0 baseline
  -> M1 model support
  -> M2 W4A16
  -> M3 FlashAttention
  -> M4 speculative
  -> M5 CUDA Graph
  -> M6 TP
  -> M7 combined
```

M2/M3/M4 可以在 M1 之后并行；M5 依赖 M3 或 `triton_paged`；M6 最好在单卡功能
稳定后做，否则调试变量太多。

#### 4.7.1 按 Resume.txt 收敛后的 2–3 个月执行版

这一版按“1 名工程师全职、12 周、尽快拿到可解释的性能结论”重排。它与 4.7 的
M0–M7 不冲突，只是把不适合 2–3 个月的部分移出核心路径。

**先明确一个简历冲突**：`Resume.txt` 第 9 行写了“DFlash 块级投机解码 + Draft
Model”。如果按本次要求去掉 draft model，DFlash 就无法成立。建议二选一：

- 简历改为“实现块级投机解码的验证与 KV 回滚，支持 n-gram drafter”；
- 或把 DFlash 放到 2–3 个月之后，作为需要现成 draft checkpoint 的后续工作。

下面的核心计划**不包含 draft model**；n-gram 投机解码只作为最后的可选加分项。

范围收敛：

| 项 | 处理 | 理由 |
|---|---|---|
| Chunked Prefill / 连续批处理 / Prefix Caching | **保留，只验证和调优** | 代码已实现，不重写；它们是 baseline 的一部分 |
| Triton paged / online softmax | **保留，作为 FlashAttention 实现** | 已实现 prefill/decode 融合；不再引入官方 `flash-attn`，避开 page size 256 对齐问题 |
| Qwen3-0.6B/8B | **保留** | 已支持 |
| Llama-3.2-1B | **保留** | 补 `llama3` RoPE scaling |
| DeepSeek-R1-Distill-Qwen-14B | **保留，替换 1.5B/7B** | `Resume.txt` 明确写的是 14B；它是 dense Qwen2，不是 MoE |
| AWQ W4A16 | **保留** | 加载 AWQ checkpoint + W4A16Linear + 反量化/GEMM + 精度/显存测试 |
| CUDA Graph decode | **保留，只做 decode** | prefill graph 不在简历里，且 shape 变化大 |
| NCCL Tensor Parallel | **保留 TP=2，TP=4 为 stretch** | 简历只要求“多卡”；TP=2 足以证明 NCCL/切分/多进程链路 |
| 端到端 Benchmark | **保留并前置** | 所有结论的前提 |
| DFlash / draft model | **移出核心** | 需要额外 draft 模型、独立 KV 和训练/适配；2–3 个月风险过高 |
| stochastic 投机采样 | **删除** | 简历未提，且依赖 draft 概率分布 |
| MoE | **删除** | 简历未提；14B 是 dense，不需要 MoE |
| 官方 flash-attn backend | **删除** | 现有 Triton kernel 已覆盖；FA2 paged 对 page size 有额外约束 |
| PP / DP / EP | **删除** | 简历未提 |
| FP8 / MLA / FP8 KV | **删除** | 简历未提，收益与工作量不成比例 |
| TP + CUDA Graph 组合 | **stretch，不进核心** | 两个变量同时调，调试成本高 |

**12 周顺序**：

| 周 | 阶段 | 主要工作 | 验收 |
|---|---|---|---|
| W1–W2 | P0 基线 | 参数化 benchmark；接入 `sharegpt-english-10k` 的 100 请求分层样本（50 short / 50 long / 20% 共享前缀）；TTFT/TPOT/E2E/吞吐/显存/KV 指标；跑 Qwen3-0.6B/8B 基线；验证 prefix cache、chunked prefill 开关 | 可复现的 baseline JSON + 曲线；明确硬件/模型/dtype/block size |
| W2–W4 | P1 模型 | Qwen2ForCausalLM（14B）；Llama-3.2 `rope_scaling`；14B 分片权重加载；与 HF logits/greedy 对拍；4K/8K 上下文 | 四个模型都能加载和生成；14B 显存/延迟基线 |
| W4–W7 | P2 AWQ W4A16 | AWQ checkpoint 解析；`W4A16Linear`；reference 反量化；optimized kernel（repack 到 int4pack 或 Triton AWQ GEMM）；8B/14B 精度、显存、吞吐 | 8B 权重显存下降 ≥57%；吞吐 ≥0.9× FP16；14B AWQ 可上 24GB 卡 |
| W7–W9 | P3 CUDA Graph | 静态 metadata/buffer；batch bucket；graph 外采样；单卡 `triton_paged`；AWQ 兼容性 | token 与 eager 完全一致；decode TPOT/吞吐有稳定收益 |
| W9–W12 | P4 TP=2 | `ParallelContext`；column/row parallel linear；权重切分；`MultiprocessExecutor`；NCCL；本地 KV heads；rank-0 scheduler | TP=2 与 TP=1 greedy 一致；吞吐/延迟 scaling；每 rank 显存可解释 |
| W11–W12 | P5 收尾 | 最终矩阵：模型 × FP16/AWQ × graph on/off × TP1/2 × batch/seq/load；整理报告 | 一页结论 + 完整表格；简历数字有对应配置 |
| 可选 | P6 n-gram 投机 | n-gram drafter + greedy 验证 + KV rollback；不做 draft model/stochastic | 输出与 greedy 一致；有 acceptance rate 和端到端数据 |

**两个执行风险**：

- **AWQ checkpoint**：如果找不到 Qwen3-8B / 14B 的预量化 AWQ checkpoint，第一版
  用 RTN W4A16 从 FP16 量化，简历写 “W4A16” 而不是 “AWQ”；或者额外 1 周做
  AWQ 校准，不建议放进核心。
- **TP 预算**：TP=2 需要 2 卡窗口；如果只有单卡，TP 降级为 stretch，简历不要
  写“已完成多卡推理”。

**8 周最低配**：如果时间/预算只够 8 周，按下面砍：

- P0 基线 1 周；
- P1 模型 1.5 周；
- P2 AWQ 2 周（只做 reference + 一个 optimized kernel，不追极致性能）；
- P3 CUDA Graph 1.5 周；
- P4 TP=2 1.5 周；
- P5 收尾 0.5 周。

砍掉 TP=4、n-gram 投机、TP+graph、完整矩阵；只保留三组关键对比：FP16 vs AWQ、
graph on/off、TP1 vs TP2。

**GPU 建议（按 14B 修正）**：

| 用途 | 推荐 |
|---|---|
| 单卡主开发/基线 | **1× A100 80GB** 或 **H100 80GB**；14B FP16、8B、AWQ 都舒适 |
| 预算型单卡 | **1× L40S 48GB**；14B FP16 只适合小 batch/4K，14B AWQ 舒适 |
| TP=2 | **2× A100 80GB NVLink**；预算型 **2× L40S 48GB** |
| 14B AWQ 单卡 | **RTX 4090 24GB** 可跑小 batch/4K；不建议 14B FP16 |
| 本地开发 | RTX 3060 6GB 只用于 0.6B/1B 和单元测试，不用于 14B/TP 结论 |

14B 的关键数字：FP16 权重约 **29–30 GB**，KV 约 **192 KiB/token**（4K 约
768 MiB）；AWQ W4A16 权重约 **8.4–9 GB**。所以 24GB 卡跑 14B 必须 AWQ，48GB
卡跑 14B FP16 只能小 batch，80GB 才是舒适区。

**简历措辞建议**：

- 第 3 行：模型改为 `DeepSeek-R1-Distill-Qwen-14B`，与 `Resume.txt` 一致；
- 第 5 行：TTFT/E2E/吞吐的绝对值依赖 GPU 和样本；换卡后写“相对提升 + 硬件配置”，
  不要直接复用 127.44s/56.35s；
- 第 5 行的 `363.96 → 657.62` 按 100 请求、E2E `268.92 → 148.88s` 反推，是
  **output tokens/s**，不是 req/s（100×约 980 tokens / 268.92s ≈ 364）；简历
  单位必须改，否则数字不自洽；
- 第 7 行：写“Triton fused prefill/decode paged attention（online softmax）”，
  不要写官方 FlashAttention 2/3；
- 第 9 行：去掉 Draft Model 后，把 DFlash 改为“块级投机解码验证 + KV 回滚 +
  n-gram drafter”，或把 DFlash 放到“后续工作”；
- 第 11 行：CUDA Graph 写“decode-only”；TP 写“TP=2（TP=4 扩展）”，不要暗示
  PP/DP/EP。

---

### 4.8 建议的取舍

如果目标是**尽快得到可解释的性能结论**，建议：

1. 先做 baseline + Qwen2/RoPE + W4A16；
2. FlashAttention 作为可选 backend，不急着替换 Triton；
3. 投机解码先做 n-gram + greedy，验证 KV rollback；draft model 不进 2–3 个月核心；
4. CUDA Graph 只做 decode，prefill 暂缓；
5. TP 核心做 TP=2，TP=4 作为 stretch，不追 PP/DP/EP；
6. 每次只引入一个变量，避免“TP + graph + 量化 + 投机”同时调。

如果目标是**完整覆盖四项能力**，则要接受：

- 投机解码和 TP 会改核心契约；
- 需要大量 GPU 测试和多进程调试；
- 需要至少一套 A100/H100 或 L40S 级别的卡；
- 工作量按月计，而不是按天计。

但无论哪种目标，**先有可信的 baseline** 都是前提；否则无法判断这些优化是否真的
带来收益。

---

## 5. 参考文件索引

- 调度与请求生命周期：
  [core/scheduler.py](src/micro_vllm/core/scheduler.py)、
  [core/request.py](src/micro_vllm/core/request.py)、
  [core/scheduler_output.py](src/micro_vllm/core/scheduler_output.py)
- KV cache：
  [core/kv_cache_manager.py](src/micro_vllm/core/kv_cache_manager.py)、
  [core/cache_group_manager.py](src/micro_vllm/core/cache_group_manager.py)、
  [core/block.py](src/micro_vllm/core/block.py)、
  [core/prefix_cache.py](src/micro_vllm/core/prefix_cache.py)、
  [core/kv_cache_spec.py](src/micro_vllm/core/kv_cache_spec.py)
- Attention：
  [attention/spec.py](src/micro_vllm/attention/spec.py)、
  [attention/attention.py](src/micro_vllm/attention/attention.py)、
  [attention/operators/](src/micro_vllm/attention/operators/)
- 执行：
  [worker/model_runner.py](src/micro_vllm/worker/model_runner.py)、
  [worker/worker.py](src/micro_vllm/worker/worker.py)、
  [executor/interface.py](src/micro_vllm/executor/interface.py)、
  [engine/engine.py](src/micro_vllm/engine/engine.py)
- 模型：
  [models/registry.py](src/micro_vllm/models/registry.py)、
  [models/llama.py](src/micro_vllm/models/llama.py)、
  [models/qwen.py](src/micro_vllm/models/qwen.py)、
  [models/mistral.py](src/micro_vllm/models/mistral.py)、
  [layers/](src/micro_vllm/layers/)
- 需求/实现文档：
  [docs/requirements/v1-core-engine.md](docs/requirements/v1-core-engine.md)、
  [docs/requirements/v4-quantization.md](docs/requirements/v4-quantization.md)、
  [docs/requirements/v5-speculative-decoding.md](docs/requirements/v5-speculative-decoding.md)、
  [docs/implementation/v1-performance-todo.md](docs/implementation/v1-performance-todo.md)
