# Qwen2 适配报告：与 Qwen3 的架构差异及改动清单

- 报告日期：2026-10-30
- 代码基线：`28e6d81 feat: introduce scratch buffer for zero-copy tensor allocation` + 未提交工作区
- 目标：在只支持 Llama / Mistral / Qwen3 的 micro-vLLM 上增加 Qwen2 支持，并以
  `Qwen2.5-0.5B-Instruct` 为真机目标，完成 **权重加载 → 前向数值对齐 → 引擎端贪心生成**
  的逐级验证
- 运行环境：Python 3.10 / PyTorch 2.8.0+cu128 / transformers 4.57.6；
  真机 CPU fp32 跑 0.5B，tiny 模型另在 RTX 3060 Laptop（6 GB）上跑 fp16

---

## 0. 结论速览

| 项目 | 结论 |
|---|---|
| 与 Qwen3 的骨架 | 完全相同：pre-norm RMSNorm + SwiGLU MLP + GQA + RoPE + 可选 tied embedding |
| 实质性差异 | 3 处：① Q/K/V 投影**恒有 bias**、`o_proj` 恒无 bias；② **没有 q_norm / k_norm**；③ config **没有 `head_dim` 字段**，需由 `hidden_size // num_attention_heads` 推导 |
| 配置层差异 | Qwen2 的实现带滑窗开关（`use_sliding_window`，0.5B 为关闭）；Qwen3 用显式 `head_dim` 且允许 `hidden_size != heads * head_dim` |
| 新增源码 | 1 个：`src/micro_vllm/models/qwen2.py` |
| 修改源码 | 2 个：模型注册表、`model_runner.py`（scratch buffer 修复，与架构无关） |
| 新增测试 | 6 个文件；修改测试 3 个文件（含 tiny fixture） |
| 真机验证 | Qwen2.5-0.5B：加载通过、与 HF 的 hidden/logits 最大绝对误差 **0.0**、引擎贪心 token 与 HF 完全一致 |
| 全量回归 | `554 passed, 10 skipped`（10 个 skip 均为需要 `MICRO_VLLM_TEST_MODEL=1` 的外部模型测试） |

---

## 1. Qwen2 与 Qwen3 的差异

### 1.1 相同点

两个模型族共享同一套 Transformer 骨架，因此 micro-vLLM 的
`layers/`（`Linear`、`RMSNorm`、`SwiGLUMLP`、`RotaryEmbedding`）、
`attention/`、`core/`、`model_loader/` 全部原样复用，没有为 Qwen2 改动一行：

- Pre-norm RMSNorm（`input_layernorm` / `post_attention_layernorm` / 最终 `norm`）
- SwiGLU MLP：`gate_proj`、`up_proj`、`down_proj`，三者均无 bias
- GQA：`num_key_value_heads < num_attention_heads`
- RoPE，`rope_theta = 1e6`（Qwen2.5 与 Qwen3 这代模型一致）
- `hidden_act = silu`，无 attention dropout
- `tie_word_embeddings` 可选（两个 0.5B/0.6B 模型都为 `true`）
- 权重命名同为 `model.layers.N.self_attn.*`，`lm_head.weight` 在 tied 时与
  `model.embed_tokens.weight` 共享

### 1.2 差异对照表

| 维度 | Qwen2 / Qwen2.5 | Qwen3 | 对 micro-vLLM 的影响 |
|---|---|---|---|
| **Q/K/V bias** | HF `Qwen2Attention` 中硬编码 `bias=True`，**不读 config** | 由 `config.attention_bias` 决定（0.6B 为 `false`） | `Linear(..., bias=...)` 的取值必须按架构固定，不能照抄 Qwen3 |
| **`o_proj` bias** | 硬编码 `bias=False` | `bias=config.attention_bias` | 同上；Qwen2 的 4 个投影必须拆成 `True/True/True/False` |
| **Q/K 归一化** | **没有** | 有 `q_norm` / `k_norm`，作用在 `head_dim` 上 | 少两组参数、少两次前向；checkpoint 中也没有这两个权重名 |
| **`head_dim`** | `Qwen2Config` **没有该字段**，HF 用 `getattr(config, "head_dim", hidden // heads)` 兜底 | 显式字段（0.6B 为 128） | 模型构造和 KV cache 形状都必须走 `hidden_size // num_attention_heads` 兜底 |
| **`hidden_size` 约束** | 恒等于 `heads * head_dim`（0.5B：14 × 64 = 896） | **可以不等于**（0.6B：16 × 128 = 2048 ≠ 1024） | Qwen2 实现里可以安全地加等式校验；Qwen3 不行 |
| **Sliding Window** | 实现真正支持：`use_sliding_window=true` 时按 `sliding_window` 构造因果掩码；0.5B 为 `false` | 实现不使用滑窗（config 里仍保留同名字段，但 `sliding_window=null`） | 未实现该特性 → Qwen2 显式抛 `NotImplementedError`，避免静默错误 |
| **RoPE scaling** | 部分版本有 `rope_scaling`；0.5B 没有 | `null` | 未实现 → 同样显式拒绝 |
| **`attention_bias` 配置项** | 模型不使用该字段（bias 是硬编码的） | 真正生效 | 不能把 Qwen3 的 `values.get("attention_bias", False)` 逻辑套到 Qwen2 |

### 1.3 两个真实模型的 config 对照

`Qwen2.5-0.5B-Instruct` 与 `Qwen3-0.6B`（本机实际下载的 config.json）：

| 字段 | Qwen2.5-0.5B | Qwen3-0.6B |
|---|---|---|
| `architectures` | `Qwen2ForCausalLM` | `Qwen3ForCausalLM` |
| `hidden_size` | 896 | 1024 |
| `num_hidden_layers` | 24 | 28 |
| `num_attention_heads` | 14 | 16 |
| `num_key_value_heads` | 2 | 8 |
| `head_dim` | 无 → 推导为 **64** | 显式 **128** |
| `heads × head_dim` | 896 = hidden ✔ | 2048 ≠ hidden |
| `intermediate_size` | 4864 | 3072 |
| `vocab_size` | 151936 | 151936 |
| `rope_theta` | 1e6 | 1e6 |
| `rms_norm_eps` | 1e-6 | 1e-6 |
| `tie_word_embeddings` | `true` | `true` |
| `use_sliding_window` | `false`（`sliding_window=32768`） | `false`（`sliding_window=null`，字段保留但实现不使用） |
| `attention_bias` | 无该字段 | `false` |
| `rope_scaling` | 无该字段 | `null` |

由此推出 Qwen2.5-0.5B 的实际注意力参数形状（micro-vLLM 中
`q_proj.weight` / `k_proj.weight` / `v_proj.weight` / `o_proj.weight`）：

- `q_proj`: `(896, 896)` + bias `(896,)`
- `k_proj` / `v_proj`: `(128, 896)` + bias `(128,)`（128 = 2 × 64）
- `o_proj`: `(896, 896)`，**无 bias**

### 1.4 代码层面的差异

HF 源码（transformers 4.57.6）可以直接对照：

```python
# modeling_qwen2.py
self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

# modeling_qwen3.py
self.q_proj = nn.Linear(..., bias=config.attention_bias)
self.k_proj = nn.Linear(..., bias=config.attention_bias)
self.v_proj = nn.Linear(..., bias=config.attention_bias)
self.o_proj = nn.Linear(..., bias=config.attention_bias)
self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

micro-vLLM 的处理与之一一对应：
[`Qwen2Attention`](../src/micro_vllm/models/qwen2.py#L20-L93) 把四个投影的 bias 写死为
`True/True/True/False`，并且不创建 `q_norm` / `k_norm`；
[`Qwen3Attention`](../src/micro_vllm/models/qwen.py#L20-L99) 则把 `attention_bias` 传给四个投影并创建两组 RMSNorm。

---

## 2. 改动文件清单（重点）

工作区相较 `28e6d81` 的 Qwen2 相关改动共 **12 个文件：新增 7、修改 5**。
（`bench.py`、`benchmarks/`、`MICRO_VLLM_CAPABILITY_REPORT.md` 等未跟踪文件属于其他
工作，不在本报告范围内。）

### 2.1 总览

| 状态 | 文件 | 作用 |
|---|---|---|
| 新增 | `src/micro_vllm/models/qwen2.py` | Qwen2 模型实现（注意力 / DecoderLayer / 模型 / 权重映射） |
| 修改 | `src/micro_vllm/models/registry.py` | 把 `Qwen2ForCausalLM` 注册到架构表 |
| 修改 | `src/micro_vllm/worker/model_runner.py` | scratch buffer 按 cache group 拆分并支持按需扩容 |
| 修改 | `tests/conftest.py` | 新增 `tiny_hf_qwen2_checkpoint` fixture |
| 修改 | `tests/models/test_registry.py` | 增加 Qwen2 注册解析测试，更新支持列表断言 |
| 修改 | `tests/worker/test_model_runner.py` | 测试辅助函数适配 per-group buffer |
| 新增 | `tests/models/test_qwen2_initialization.py` | Qwen2 结构单元测试（bias / 无 qk norm / 校验） |
| 新增 | `tests/models/test_qwen2_weights.py` | Qwen2 权重加载语义测试（tied / 未知参数） |
| 新增 | `tests/integration/test_loaded_qwen2.py` | tiny checkpoint：结构、加载、CPU/GPU 前向对齐 |
| 新增 | `tests/integration/test_loaded_qwen2_engine.py` | tiny 引擎语义测试 + 0.5B 贪心生成回归 |
| 新增 | `tests/integration/test_real_qwen2.py` | Qwen2.5-0.5B 真模型加载验证 |
| 新增 | `tests/integration/test_real_qwen2_parity.py` | Qwen2.5-0.5B 与 HF 的前向数值对齐 |

### 2.2 新增源码：`src/micro_vllm/models/qwen2.py`

整体结构照搬 `models/qwen.py`（Qwen3），只把 1.2 表中的三处差异落地。文件由四部分组成：

1. **`Qwen2Attention`**（[qwen2.py:20-93](../src/micro_vllm/models/qwen2.py#L20-L93)）
   - `q_proj` / `k_proj` / `v_proj`：`bias=True`；`o_proj`：`bias=False`
   - **不创建** `q_norm` / `k_norm`，前向中 Q/K 直接进 RoPE
   - `RotaryEmbedding(head_dim, base=rope_theta)`

2. **`Qwen2DecoderLayer`**：pre-norm 结构、SwiGLU、共享同一个 `Attention` 实例，
   与 Qwen3 完全一致（未做任何改动）。

3. **`Qwen2Model.__init__`**（[qwen2.py:150-222](../src/micro_vllm/models/qwen2.py#L150-L222)）
   在 Qwen3 骨架上增加了 Qwen2 特有的读取与校验：
   - `head_dim = config.positive_int("head_dim", hidden_size // num_attention_heads)`
     ——兼容“有/没有 `head_dim` 字段”两种 config
   - `if hidden_size != head_dim * num_attention_heads: raise ValueError(...)`
     ——Qwen2 全系列满足该等式；Qwen3-0.6B 不满足，所以这条校验只放在 Qwen2 里
   - `use_sliding_window` 为真 → `NotImplementedError("Sliding window attention is not yet supported.")`
   - `rope_scaling` 非空 → `NotImplementedError("RoPE scaling is not yet supported.")`
   - `tie_word_embeddings` 为真时，`LMHead(..., tied_embedding=self.embed_tokens)` 共享同一个
     `Parameter`，而不是复制一份权重
   - `num_attention_heads % num_key_value_heads != 0` → 直接报错

4. **权重加载**（[qwen2.py:250-284](../src/micro_vllm/models/qwen2.py#L250-L284)）
   - `_qwen2_parameter_name`：去掉 `model.` 前缀；tied 时把 `lm_head.weight` 映射到
     `embed_tokens.weight`，避免共享参数被覆盖或重复拷贝
   - `load_weights` 按参数名逐个 `copy_`，并校验：未知参数 → `KeyError`；形状不一致 →
     `ValueError`；checkpoint 缺参数 → 列出缺失项报错

> 为什么不直接把 `qwen.py` 参数化：三处差异都是**架构固定语义**（bias 硬编码、
> 有无 q/k norm），不是运行时配置。合并会引入按架构分支的构造函数，降低可读性，
> 也与仓库现有的“一族一文件”（`llama.py` / `mistral.py` / `qwen.py`）风格不一致。

### 2.3 修改源码

**`src/micro_vllm/models/registry.py`**

```diff
 from micro_vllm.models.qwen import Qwen3Model
+from micro_vllm.models.qwen2 import Qwen2Model

 _MODEL_CLASSES = MappingProxyType({
     "LlamaForCausalLM": LlamaModel,
     "MistralForCausalLM": MistralModel,
+    "Qwen2ForCausalLM": Qwen2Model,
     "Qwen3ForCausalLM": Qwen3Model,
 })
```

模型发现链路（`model_loader` 读 `config.json` → `ModelConfig.architectures` →
`get_model_class`）本身是所有架构共用的，因此 Qwen2 的接入只需要这一处注册。

**`src/micro_vllm/worker/model_runner.py`**

这部分改动**不是 Qwen2 特有的**，而是在同一条工作线上修复 scratch buffer 引入的回归；
它被 Qwen2 的引擎测试直接覆盖。

- 原来 `slot_mapping` / `block_table` 各只有一块 buffer，被所有 cache group 共用：
  `_build_model_input` 依次为 group 0、group 1 … 构造元数据，后一个 group 会把前一个
  group 写入的内容覆盖掉，而 layer 元数据里保存的是同一块 buffer 的 view。
  结果是 hybrid attention（full + sliding/chunked-local 两组）读到错误 slot，
  数值偏差可达 0.37；单测中也出现 `block_table == [[1]]` 而不是 `[[0]]`。
- 现在：
  - `_allocate_scratch_buffers()`（[model_runner.py:734-756](../src/micro_vllm/worker/model_runner.py#L734-L756)）
    在 `initialize_model()` 建好 KV cache topology **之后**，为每个 cache group 各分配一份
    `slot_mapping`（`max_num_batched_tokens`）和 `block_table`（`max_num_seqs ×
    ceil(max_model_len / block_size)`）
  - `_build_slot_mapping` / `_build_block_table` 按 `cache_group_index` 取各自的 buffer，
    并且**先切片再写入**，只触碰本 plan 的行/列
  - 两个 builder 在 batch 或 block 数超过预分配容量时会一次性扩容（生产路径上
    scheduler 保证不超限，该分支只服务于直接构造 plan 的单元测试/离线调用）
  - `_flatten_model_input` 中四个 1-D buffer（`input_ids` / `positions` /
    `query_start_locations` / `context_lengths`）也通过 `_reserve_1d` 获得同样的兜底

修复后全量测试从 `4 failed, 550 passed` 变为 `554 passed, 10 skipped`。

### 2.4 新增测试

| 文件 | 测试数 | 覆盖内容 |
|---|---|---|
| `tests/models/test_qwen2_initialization.py` | 4 | ① Q/K/V 有 bias、`o_proj` 无 bias、无 `q_norm`/`k_norm`、RoPE base 可配；② 默认值（`head_dim` 显式时、rope theta 1e6、rms eps 1e-6、未 tie）；③ 恒定 bias 规则；④ `heads % kv_heads != 0` 时报错 |
| `tests/models/test_qwen2_weights.py` | 3 | ① 逐参数拷贝（含 3 组 bias，且无 `q_norm`/`k_norm`）；② tied 模型接受 `lm_head.weight` 与 `embed_tokens.weight` 两种 checkpoint 名称，并保持共享同一 `Parameter`；③ 未知参数名 → `KeyError` |
| `tests/integration/test_loaded_qwen2.py` | 4 | ① tiny checkpoint 布局：safetensors 中存在 q/k/v bias、不存在 `o_proj.bias`；② `load_model` 精确还原参数并验证 tied 身份；③ CPU fp32 的 hidden/logits 与 `Qwen2ForCausalLM` 对齐（atol=rtol=1e-5）；④ CUDA fp16 对齐（atol=rtol=4e-3） |
| `tests/integration/test_loaded_qwen2_engine.py` | 7 | 见 2.5 |
| `tests/integration/test_real_qwen2.py` | 1 | 0.5B 真模型：架构类型、tied 身份、Q/K/V bias 存在、`o_proj` 无 bias、`head_dim` 推导正确、各投影形状、切片对比 checkpoint 张量（受本机内存限制只比对前 4096 个元素） |
| `tests/integration/test_real_qwen2_parity.py` | 1 | 0.5B 真模型前向：先跑 micro-vLLM 前向，`del model; gc.collect()` 后再加载 HF 参考模型（避免同时驻留两份 fp32 权重），比对 hidden states 与 logits，并断言 argmax 完全一致 |

### 2.5 引擎级测试（`test_loaded_qwen2_engine.py`）

从 `test_loaded_qwen3_engine.py` 迁移并针对 Qwen2 调整（tiny 模型 `head_dim = 16/4 = 4`，
Qwen3 tiny 是 8）。7 个测试：

1. **full vs chunked prefill**：同一 prompt 在 `token_budget=8` 与 `2` 下生成结果一致，
   且 chunk 切分为 `[2, 2, 1]`
2. **mixed batch + prefix cache**：decode 与新 prefill 同批；第二个请求复用前缀
   （`num_computed_tokens == 4`，前两个 block 相同）
3. **preemption recompute**：块数不足时触发抢占，被抢占请求从头重算，结果与宽裕配置一致
4. **block 复用与终止释放**：请求结束后 block 归还并可被新请求复用；abort 后块数恢复
5. **EOS / MODEL_LENGTH 终止原因**
6. **CUDA fp16 引擎**：KV cache 形状最后一维等于 Qwen2 tiny 的 `head_dim = 4`，
   dtype 为 fp16
7. **真实 0.5B 贪心生成**（`@pytest.mark.model`）：先用 HF 模型逐 token 贪心得到参考序列，
   `del` + `gc.collect()` 释放后再构造 micro-vLLM Engine，逐 token 比对

### 2.6 修改测试

**`tests/conftest.py`**

新增 `tiny_hf_qwen2_checkpoint` fixture（[conftest.py:92-120](../tests/conftest.py#L92-L120)）：

- 用 `Qwen2Config` + `Qwen2ForCausalLM` 生成确定性 checkpoint（seed = 3）
- 参数：vocab 32、hidden 16、intermediate 32、2 层、4 heads、2 kv heads、
  max_position 32、rms eps 1e-5、rope theta 1e6、silu、tied embeddings、
  `use_sliding_window=False`
- **故意不传 `head_dim`**，与真实 Qwen2 config 保持一致，用来验证推导逻辑
- 保存为单文件 `model.safetensors`，与 Qwen3 fixture 的布局一致

**`tests/models/test_registry.py`**

- 新增 `test_registry_resolves_qwen2_architecture`
- 不支持架构的报错信息断言从 3 个架构更新为 4 个

**`tests/worker/test_model_runner.py`**

scratch buffer 改成 per-group 后，两个测试辅助函数需要同步：

- `make_input_runner` 手工设置 `_model` / `_kv_cache_topology`，绕过了
  `initialize_model`，因此显式调用 `_allocate_scratch_buffers()`
- 新增 `make_builder_runner`，用两个不同 spec 的 attention 构造 **2 个 cache group**，
  因为 `build_slot_mapping` / `build_block_table` 现在会按 `cache_group_index` 取各自的 buffer

### 2.7 未改动、直接复用的部分

| 模块 | 为什么不用改 |
|---|---|
| `model_loader/` | 完全按 config.json 的通用字段解析；Qwen2 只是多了几个自己读取的字段；safetensors 迭代与 tied 处理是通用的 |
| `layers/linear.py` | 已支持 `bias=True/False` |
| `layers/normalization.py` | RMSNorm 通用 |
| `layers/mlp.py` | SwiGLU 与 Qwen3 相同 |
| `layers/rotary_embedding.py` | 只依赖 `head_dim` 与 `rope_theta` |
| `attention/`、`core/`、`engine/` | 与模型族无关 |
| `worker/model_runner.py` 的 KV cache 形状 | `_kv_cache_shape` 早已有 `head_dim` 缺省回退（`hidden_size // num_attention_heads`），Qwen2 直接受益 |

---

## 3. 验证链路与结果

按“先小后大、先加载后数值、先单步后引擎”的顺序逐级验证，任何一级失败都不进入下一级：

| 阶段 | 测试文件 | 结果 |
|---|---|---|
| 0. tiny checkpoint 生成 | `tests/conftest.py` fixture | ✅ |
| 1. 结构 + 参数还原 | `tests/models/test_qwen2_*.py`（7 个单测） | ✅ |
| 2. tiny 前向对齐（CPU fp32 / GPU fp16） | `tests/integration/test_loaded_qwen2.py` | ✅ |
| 3. tiny 引擎语义 | `tests/integration/test_loaded_qwen2_engine.py` 前 6 个 | ✅ |
| 4. 0.5B 权重加载 | `tests/integration/test_real_qwen2.py` | ✅ 1 passed，5.67 s |
| 5. 0.5B 前向数值对齐 | `tests/integration/test_real_qwen2_parity.py` | ✅ 1 passed，6.62 s；hidden/logits 最大绝对误差 0.0，argmax 完全一致 |
| 6. 0.5B 引擎贪心生成 | `test_real_qwen2_engine_greedy_matches_hf` | ✅ 1 passed，9.06 s |

实测命令与汇总（三个真模型文件一起跑）：

```text
9 passed in 24.30s
```

全量回归：

```text
554 passed, 10 skipped in 18.99s
```

10 个 skip 全部是 `@pytest.mark.model`（需要 `MICRO_VLLM_TEST_MODEL=1` 才加载外部模型）。

### 3.1 本机内存约束下的两阶段模式

本机可用内存约 7.4 GB，Qwen2.5-0.5B 的 fp32 权重约 2 GB，两份同时驻留再加
Python/PyTorch 开销会触发 OOM（曾被内核 kill，退出码 137）。因此真模型测试统一采用：

1. 先计算 HF 参考结果（或先跑 micro-vLLM 前向并保存输出）
2. `del model` + `gc.collect()` 释放最后一份引用
3. 再加载第二份模型

注意：`del` 必须发生在**测试函数作用域**、且删除的是最后一个引用；
在 helper 里 `del` 函数参数不会释放任何内存，使用 pytest fixture 持有模型时也无法释放。

---

## 4. 已知限制

| 限制 | 说明 |
|---|---|
| Sliding Window Attention | 未实现。`use_sliding_window=True` 的 Qwen2 变体（如部分 Qwen1.5 / Qwen2 长上下文配置）会在构造时直接报 `NotImplementedError`，而不是静默算错 |
| RoPE scaling | 未实现，`rope_scaling` 非空即报错 |
| Qwen2.5-1.5B | 尚未在本机验证（内存不足），计划在 AutoDL GPU 上补做加载 + parity + 生成 |
| 真模型精度 | 真模型测试目前只覆盖 CPU fp32；tiny 模型覆盖了 GPU fp16 |
| 引擎测试的 `model` 标记 | 0.5B 贪心测试默认跳过，需要显式设置环境变量才会加载模型 |

---

## 5. 复现命令

```bash
# 全量（不加载外部模型）
python -m pytest -q

# Qwen2 相关（tiny 模型，默认即可跑）
python -m pytest tests/models/test_qwen2_initialization.py \
                 tests/models/test_qwen2_weights.py \
                 tests/integration/test_loaded_qwen2.py \
                 tests/integration/test_loaded_qwen2_engine.py -q

# Qwen2.5-0.5B 真模型
MICRO_VLLM_TEST_MODEL=1 \
MICRO_VLLM_QWEN2_MODEL_PATH=/home/lenovo/models/Qwen2.5-0.5B-Instruct \
python -m pytest tests/integration/test_real_qwen2.py \
                 tests/integration/test_real_qwen2_parity.py \
                 tests/integration/test_loaded_qwen2_engine.py -q
```

> 本机使用的解释器为 `/home/lenovo/miniconda3/envs/micro-vllm/bin/python`。

---

## 6. 后续工作建议

1. 在 GPU（AutoDL）上完成 Qwen2.5-1.5B 的加载、fp16/bf16 前向对齐与 `LLM.generate()`
   端到端测试，补齐 `tests/integration/test_hub_qwen.py` 的真模型路径
2. 若后续要支持 Qwen2 的 sliding window 或长上下文 `rope_scaling`，应在
   `attention/spec.py` 增加对应的 `AttentionSpec`，而不是在 `Qwen2Model` 里加分支
3. 将 scratch buffer 的 per-group 不变量写成注释/测试：同一逻辑 block 在不同 cache group
   对应不同物理 block ID，因此 block table 与 slot mapping 不能跨 group 复用
