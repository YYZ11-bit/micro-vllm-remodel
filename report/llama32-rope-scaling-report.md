# Llama-3.2 的 `llama3` RoPE scaling 支持报告

- 报告日期：2026-11-02
- 目标：在 micro-vLLM 的 `RotaryEmbedding` 中实现 Llama-3.2 使用的
  `rope_scaling={"rope_type": "llama3", ...}`，并以 `Llama-3.2-1B-Instruct` 为真机目标，
  完成 **权重加载 → RoPE 频率对齐 → 前向 logits 对齐 → 引擎贪心生成** 的逐级验证
- 本地环境：Python 3.10.22 / PyTorch 2.8.0+cu128 / transformers 4.57.6
- 真机环境：AutoDL，RTX 2080 Ti（11 GB，SM 7.5）/ Xeon Platinum 8255C（12 vCPU /
  43 GiB）/ PyTorch 2.4.0+cu121 / transformers 4.57.6

---

## 0. 结论速览

| 项目 | 结论 |
|---|---|
| 频率公式 | 三段式：高频段不动、过渡带线性混合、低频段除以 `factor`；与 HF `_compute_llama3_parameters` 完全一致 |
| 与 HF 的 inv_freq 对齐 | 32 个频率分量最大绝对误差 **0.0**（逐元素完全相等） |
| 与 HF 实现的差异 | 数学一致、结构不同：运行时不依赖 transformers，用扁平 token 布局与 fp32 常驻频率；HF 只在测试中作为 oracle（见第 2 节） |
| Llama-3.2-1B 的频段划分 | 32 个分量中 **15 个不动**（j=0..14）、**3 个平滑过渡**（j=15..17）、**14 个缩小 32 倍**（j=18..31） |
| 修改源码 | 2 个：`layers/rotary_embedding.py`（公式 + 构造期缓存）、`models/llama.py`（从 `ModelConfig.values` 读 `rope_scaling` 并逐层传递） |
| 新增测试 | 4 个层次：layers 单测、models 初始化、tiny checkpoint 集成、真模型 + 引擎 |
| 真机测试 | 4 个真模型测试全部通过，CPU 上 `4 passed in 63.00s` |
| 前向数值对齐 | CPU fp32：hidden / logits 最大绝对误差 **0.0**，argmax 完全一致；GPU fp16：logits 最大绝对误差 2.34e-2，argmax 一致 |
| 引擎贪心 | CPU fp32 / GPU fp16 / GPU bf16 三种配置的 4 个生成 token 均为 `(12366, 13, 578, 6864)`，与 HF 一致 |
| 长上下文 | 8400 token 输入（超过 `original_max_position_embeddings=8192`），分 5 个 prefill chunk，下一 token 与 HF 一致（578） |
| 全量回归 | `576 passed, 14 skipped`（14 个 skip 均为未设置 `MICRO_VLLM_TEST_MODEL=1` 的外部模型测试） |

---

## 1. `llama3` scaling 的频率公式

### 1.1 三段式定义

设

- `θ = rope_theta`（Llama-3.2 为 500000）
- `d = head_dim`（1B 为 64）
- 原始频率 `f_j = θ^(-2j/d)`，`j = 0 .. d/2-1`
- `wavelen_j = 2π / f_j`（该分量的"波长"）

Llama-3 的 scaling 用 config 里的四个参数
`factor=32.0`、`low_freq_factor=1.0`、`high_freq_factor=4.0`、
`original_max_position_embeddings=8192` 先算出两条波长分界线：

```
low_wavelen  = original_max_position_embeddings / low_freq_factor   = 8192
high_wavelen = original_max_position_embeddings / high_freq_factor  = 2048
```

然后按 `wavelen` 落在哪个区间分三段处理：

| 区间 | 含义 | 处理 |
|---|---|---|
| `wavelen > low_wavelen` | 低频（长波长） | `f_j / factor`，即压缩 32 倍 |
| `high_wavelen <= wavelen <= low_wavelen` | 过渡带 | 在"未压缩"和"压缩 32 倍"之间按 `smooth` 线性插值 |
| `wavelen < high_wavelen` | 高频（短波长） | 保持不变 |

过渡带权重与最终结果：

```
smooth_j = (original_max_position_embeddings / wavelen_j - low_freq_factor)
           / (high_freq_factor - low_freq_factor)
scaled_j = f_j / factor                     （低频段）
smoothed_j = (1 - smooth_j) * f_j / factor + smooth_j * f_j
f_j' = smoothed_j  若 high_wavelen <= wavelen_j <= low_wavelen
       scaled_j    若 wavelen_j > low_wavelen
       f_j         否则
```

这种分段保证频率在两条分界线上连续（过渡带两端权重分别为 0 和 1），
避免位置编码在边界处突变。

### 1.2 Llama-3.2-1B 的实际频段

`d = 64`、`θ = 500000`，实测划分：

| 分量下标 | 数量 | 处理 | 代表值 `inv_freq` |
|---|---|---|---|
| j = 0..14 | 15 | 不变 | `j=0` → 1.0；`j=14` → 3.211e-3 |
| j = 15..17 | 3 | 过渡带插值 | 1.291e-3 / 4.296e-4 / 9.708e-5 |
| j = 18..31 | 14 | `f_j / 32` | `j=18` → 1.946e-5；`j=31` → 9.418e-8 |

不变/缩放比值验证：`ratio[0] = 1.0000`、`ratio[31] = 32.0000`，
`ratio == 1` 的分量恰好 15 个。手算三个过渡带分量（j=15/16/17）与实现输出一致。

### 1.3 与 HF 实现的对应关系

HF 在 `transformers/models/llama/modeling_llama.py` 的
`_compute_llama3_parameters` 中做同样计算，区别只是它以 `1 / wavelen` 的形式书写。
本实现在 `layers/rotary_embedding.py::_inverse_frequencies` 中用 PyTorch 张量实现，
全部中间量保持 fp32，避免 fp16 下 `3e-6` 级别的频率下溢。

---

## 2. 为什么必须自己实现（与 HF 实现的差异）

### 2.1 功能上：忽略 scaling 得到的就是另一个模型

Llama-3.2 的权重是在 `llama3` 频率表下训练的，`rope_scaling` 属于训练配置的一部分，
不是推理阶段的可选项。1B 上有 **14 个低频分量被缩小了 32 倍**，其中
`j=18` 的原始频率是 `6.22e-4` rad/token，缩放后是 `1.946e-5` rad/token；
在位置 1024 上两者的累计相位分别是 **0.637 rad** 和 **0.0199 rad**，
到位置 8192 则是约 **5.09 rad** 和 **0.159 rad**。

实测（fp16、CUDA、同一份 micro 实现，只把各层 `_inv_freq` 换成未缩放版本）：

| 输入长度 | logits 最大误差 | top-1 | top-5 |
|---|---|---|---|
| 256 | 0.543 | 相同（279） | 相同 |
| 1024 | 1.501 | 相同（35308） | 第 4 位不同（627 → 503） |

也就是说：上下文越长，偏差从"logits 数值变化"逐渐发展到"候选排序改变"；
在 8192 附近低频分量的相位差达到弧度级，贪心结果不再可信。
所以两条"不实现"的路线都不成立——静默忽略会让输出出错；直接拒绝则 Llama-3.2
根本无法运行。必须实现。

### 2.2 结构上：不能直接调用 HF 的实现

即使公式相同，HF 的 `LlamaRotaryEmbedding` 也不能直接塞进 micro-vLLM 的执行路径：

| 维度 | HF `LlamaRotaryEmbedding` | micro `RotaryEmbedding` |
|---|---|---|
| 运行时依赖 | transformers + `PretrainedConfig`，字段解析、校验都在 HF 内部 | `layers/` 不 import transformers 或 `micro_vllm.config`，只接收 `base` 与 `rope_scaling: Mapping` |
| 张量布局 | `(batch, heads, seq, head_dim)` + 每样本 `position_ids`；cos/sin 形状 `(batch, seq, head_dim)` | `(total_tokens, heads, head_dim)` + 一维 `positions`；同一步可以混合多个请求与 prefill 分块 |
| 位置来源 | 通常由 `arange` / `position_ids` 传入 | 由 scheduler 生成，chunked prefill 时每块只覆盖一段位置 |
| 频率状态 | `inv_freq` 注册为非持久 buffer，forward 内 `.float()` 上采样并关闭 autocast | `_inv_freq` 存为普通属性，不受 `model.to(dtype)` 影响（见 2.2） |
| 后乘系数 | 统一乘 `attention_scaling` | `llama3` 的 `attention_scaling` 恒为 1.0，因此不做后乘 |
| 动态更新 | `dynamic_rope_update` 装饰器在超长序列时重算频率 | 静态缓存；`llama3` 不是 dynamic 类型，引擎复用同一张频率表 |
| 在项目中的角色 | 仅测试里的 oracle | 运行时实现 |

具体原因：

1. **依赖方向**。按仓库的架构约束，`layers/` 是架构无关的通用层，
   不能读模型 config，更不应该把 transformers 变成运行时依赖；
   `models/llama.py` 负责从 `ModelConfig.values` 取出 `rope_scaling` 这个原始 Mapping，
   层只认 `base` 和这个 Mapping。
2. **扁平 token 布局**。paged attention + chunked prefill 下，一步前向里的 token
   可能来自不同请求、不同 chunk，`positions` 是调度器给出的绝对位置向量，
   而不是 `arange(seq_len)`；HF 的 batch-first 接口需要额外的 reshape/拆分，
   与执行路径不匹配。
3. **dtype 策略**。fp16 的最小正规数是 `6.1e-5`，而缩放后的频率最低到 `9.4e-8`，
   属于次正规数。若像 HF 那样把它注册成 buffer，`model.to(dtype)` 会连同 buffer
   一起转成 fp16，精度在构造阶段就丢了；本实现把 `_inv_freq` 保持 fp32，
   forward 中把角度与旋转计算都放在 fp32，最后再把 q/k 转回输入 dtype。
4. **配置解析位置**。`rope_type` 的归一（`rope_type` / 旧版 `type` / 默认）、
   Mapping 校验放在 `LlamaModel` 构造期，失败时抛出带上下文的错误；
   `RotaryEmbedding` 只处理"给定参数字典"这一件事。

### 2.3 复用的边界

- 数学上两者完全一致：32 个 `inv_freq` 分量与 HF 的逐元素最大误差为 **0.0**，
  说明"自己实现"没有引入任何数值偏差。
- HF 在测试中作为 oracle（`test_llama32_rope_scaling_matches_transformers` 等），
  运行时零依赖，这是刻意保持的依赖方向。
- 如果以后要支持 `dynamic` / `yarn` / `linear` 等类型，应把"频率如何随序列长度变化"
  收敛到注意力配置层（`AttentionSpec` 或等价位置），而不是把 HF 的 stateful 实现
  搬进 `layers/`。

---

## 3. 实现落点

### 3.1 `layers/rotary_embedding.py`

- `RotaryEmbedding.__init__(head_dim, base=10_000.0, rope_scaling=None)` 在构造时调用
  `_inverse_frequencies`，把结果缓存为普通属性 `self._inv_freq`。
  **故意不使用 `register_buffer`**：buffer 会跟随 `model.to(dtype)` 被转成 fp16，
  而 `1.4e-7` 这类频率在 fp16 下会下溢为 0、`θ` 相关中间量可能上溢。
- `_inverse_frequencies` 先做类型归一：
  `rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"`，
  兼容旧版 config 的 `"type"` 字段；`None` 或 `"default"` 直接返回原始频率。
- 非 `llama3` 的类型显式拒绝（`NotImplementedError`），不静默回退到默认频率。

### 3.2 `models/llama.py`

- `LlamaModel.__init__` 从 `values.get("rope_scaling")` 读取，并校验
  `isinstance(..., Mapping)`，否则抛 `ValueError("rope_scaling must be a mapping or null")`。
- `LlamaAttention` / `LlamaDecoderLayer` / `LlamaModel` 逐层透传 `rope_scaling`，
  最终传给 `RotaryEmbedding(self.head_dim, base=rope_theta, rope_scaling=rope_scaling)`。
- `layers/` 不 import `micro_vllm.config`，保持"模型读配置、层只收原始值"的边界；
  `rope_scaling` 只作为 `Mapping[str, object]` 向下传递。

### 3.3 与其它后端的边界

`rope_scaling` 目前只在 Llama 模型路径生效；Qwen2 / Qwen3 / Mistral 的模型实现不接受
该字段。如果将来要支持长上下文变体，应在 `attention/spec.py` 层表达，而不是在
`RotaryEmbedding` 里继续堆 `if`。

---

## 4. 测试分层

| 层次 | 文件 | 固定的行为 |
|---|---|---|
| 1. 公式 / 与 HF 对齐 | `tests/layers/test_rotary_embedding.py` | `_inverse_frequencies` 与 HF `LlamaRotaryEmbedding.inv_freq` 逐元素一致；同时断言三个频段的分段数量与缩放比例，防止"公式碰巧对上但频段全错" |
| 2. 配置 → 模型管线 | `tests/models/test_llama_initialization.py` | `ModelConfig.values["rope_scaling"]` 能一路传到 `RotaryEmbedding`；模型内部 `RotaryEmbedding` 的输出与直接构造的 `RotaryEmbedding(16, 500_000.0, LLAMA32_ROPE_SCALING)` 一致；非 Mapping 抛 `ValueError` |
| 3. tiny checkpoint 集成 | `tests/integration/test_loaded_llama.py` | 带 scaling 的 tiny checkpoint 与 HF `LlamaForCausalLM` 在 CPU 上前向对齐；scaled vs unscaled 的位置编码确实不同 |
| 4. 真模型 + 引擎 | `tests/integration/test_real_llama32.py`、`tests/integration/test_loaded_llama_engine.py` | 真实 1B 权重加载、RoPE 对齐、前向对齐、引擎贪心生成与 HF 一致 |

### 4.1 tiny 模型的一个坑

tiny fixture 的 `head_dim=4`、`base=1e4` 时，最长波长只有
`2π / 1e-4 ≈ 628`，全部小于 `high_wavelen = 8192 / 4 = 2048`，
三段式会退化成"全部不动"——表面上测试通过，实际上 scaling 从未生效。

修正方式：把 tiny fixture 的 scaling 参数改成
`original_max_position_embeddings=16`、`factor=8.0`、`low_freq_factor=1.0`、
`high_freq_factor=4.0`，此时分界线为 `low=16`、`high=4`，
tiny 的波长（6.28 和 628.3）分别落入"不动"和"缩放"两段，
修正后实现输出 `[0.576056, 0.00125]` 与 HF 完全一致。
这个坑也说明第 1 层测试必须同时断言频段分布，而不能只比对最终数值。

---

## 5. 真机验证（RTX 2080 Ti 11 GB）

### 5.1 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti，11264 MiB，驱动 595.71.05，compute capability **7.5** |
| CPU / 内存 | Xeon Platinum 8255C，cgroup 限制 12 vCPU / 43 GiB |
| PyTorch / transformers | 2.4.0+cu121 / 4.57.6 |
| 模型 | `LLM-Research/Llama-3.2-1B-Instruct`（ModelScope，bf16 safetensors，2.4 GB） |

### 5.2 真模型测试结果

命令：

```bash
MICRO_VLLM_TEST_MODEL=1 \
MICRO_VLLM_LLAMA32_MODEL_PATH=/root/autodl-tmp/models/Llama-3.2-1B-Instruct \
PYTHONPATH=src python -m pytest \
  tests/integration/test_real_llama32.py \
  tests/integration/test_loaded_llama_engine.py::test_real_llama32_engine_greedy_matches_hf \
  -q --durations=0
```

结果：`4 passed in 63.00s`

| 测试 | 耗时 | 验证内容 |
|---|---|---|
| `test_llama32_load_weight` | 16.23 s | 架构是 `LlamaForCausalLM`、tied embedding、`rope_theta=500000`、`rope_scaling` 字段、q/k/v/o 形状、权重与 safetensors 一致 |
| `test_llama32_rope_scaling_matches_transformers` | 0.01 s | `inv_freq` 最大误差 0.0；在位置 0/1/127/8192/131071 上 q/k 旋转结果与 HF 的 `apply_rotary_pos_emb` 一致（atol=rtol=1e-6） |
| `test_llama32_parity_matches_transformers` | 19.54 s | CPU fp32 下 hidden 与 logits 与 HF eager 完全一致（atol=rtol=1e-5 内，实测 0.0），argmax 全等 |
| `test_real_llama32_engine_greedy_matches_hf` | 26.91 s | 引擎端贪心 4 个 token 与 HF 一致 |

> 注意：上述测试按测试文件中的默认配置在 **CPU fp32** 上运行（`make_loaded_engine`
> 默认 `device="cpu", dtype=float32`）。GPU 数据来自下面单独的验证脚本。

### 5.3 数值对齐（额外验证脚本）

| 配置 | 加载耗时 | micro 前向 | HF 前向 | hidden 最大误差 | logits 最大误差 | argmax 一致 |
|---|---|---|---|---|---|---|
| CPU fp32 | 14.34 s | 0.613 s | 1.465 s | 0.0 | 0.0 | ✅ |
| CUDA fp16 | 17.47 s | 0.231 s | 0.048 s | 1.953e-2 | 2.344e-2 | ✅ |
| CUDA bf16 | 15.79 s | 0.082 s | 0.024 s | 1.250e-1 | 1.562e-1 | ✅ |

CPU fp32 的误差为 0 是因为两边都是同一公式的 fp32 计算、且都走 reference attention；
GPU fp16/bf16 的差异来自累加顺序与低精度舍入，量级与"1B 模型半精度前向"的预期一致，
且 argmax 完全一致。

### 5.4 引擎端贪心生成

prompt：`"The capital of France is"`，`max_tokens=4`，引擎配置
`max_num_batched_tokens=16`、`max_num_seqs=1`、`max_model_len=64`、`block_size=2`。

| 配置 | micro 输出 | HF 输出 | 峰值显存（HF / micro） |
|---|---|---|---|
| CPU fp32 | (12366, 13, 578, 6864) | 同左 | — |
| CUDA fp16 | (12366, 13, 578, 6864) | 同左 | 2.31 GiB / 2.31 GiB |
| CUDA bf16 | (12366, 13, 578, 6864) | 同左 | 2.31 GiB / 2.31 GiB |

（显存为 `torch.cuda.max_memory_allocated()` 峰值；引擎加载含权重与 KV cache。）

### 5.5 长上下文（超过原始 8192 窗口）

prompt 由重复文本 tokenize 后截断为 **8400 token**，`max_num_batched_tokens=2048`、
`block_size=16`，fp16：

| 指标 | HF（sdpa） | micro-vLLM 引擎 |
|---|---|---|
| 下一个 token（argmax） | 578 | 578 ✅ |
| 耗时 | 2.55 s | 加载 17.90 s，总 20.50 s |
| 峰值显存 | 4.35 GiB | 8.70 GiB |
| prefill 分块 | — | 5 步（2048 token/块） |

8400 > `original_max_position_embeddings=8192`，因此这个用例真正走到"低频段被压缩
32 倍"的位置上；下一 token 与 HF 一致，说明 chunked prefill 中跨块的 position 与
RoPE 频率没有错位。

---

## 6. 环境相关注意事项

1. **Turing（SM 7.5）没有原生 bf16**。PyTorch 会在部分算子上做 bf16 模拟，能跑通，
   但数值误差明显大于 fp16（见 5.3），且不适合用来做性能结论。在 2080 Ti 上推荐 fp16。
2. **`load_model` 不负责 dtype / device 迁移**。`load_model(config)` 只在 CPU 上构造并
   加载权重；`worker/model_runner.py` 随后执行
   `model.to(device=self._device, dtype=self._dtype)`。直接调用 `load_model` 的自定义脚本
   必须自己做这一步，否则会出现"模型 fp32、KV cache fp16"的
   `index_copy_(): self and source expected to have the same dtype`。
3. **ModelScope 下载的 `original/` 目录会让整次下载失败**。
   `snapshot_download` 校验 `original/consolidated.00.pth` 时报 `FileIntegrityError`，
   会在下载 tokenizer 文件之前中止。该目录是 Meta 原始格式的冗余副本，可以忽略：
   `snapshot_download(..., ignore_file_pattern=["original/*"])`。
4. **只有 `model.safetensors + config.json` 时无法构建 tokenizer**。
   缺少 `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` 时，
   transformers 回退到慢速 `LlamaTokenizer`，因 `vocab_file=None` 报
   `TypeError: not a string`。这是一个容易误判为"模型下载坏了"的报错。
5. **旧版 config 的字段名是 `"type"`**。部分模型的 `rope_scaling` 没有 `rope_type`，
   实现里用 `get("rope_type") or get("type") or "default"` 归一，避免 `KeyError`。
6. **非 `llama3` 类型目前抛 `NotImplementedError(...)`**，参数是 Ellipsis，
   报错信息不可读；建议改成带 `rope_type` 与支持列表的可操作信息。

---

## 7. 复现命令

```bash
# 全量回归（不加载外部模型）
python -m pytest -q

# Llama-3.2 rope_scaling 相关（tiny 模型，默认即可跑）
python -m pytest tests/layers/test_rotary_embedding.py \
                 tests/models/test_llama_initialization.py \
                 tests/integration/test_loaded_llama.py -q

# Llama-3.2-1B 真模型
MICRO_VLLM_TEST_MODEL=1 \
MICRO_VLLM_LLAMA32_MODEL_PATH=/root/autodl-tmp/models/Llama-3.2-1B-Instruct \
PYTHONPATH=src python -m pytest \
  tests/integration/test_real_llama32.py \
  tests/integration/test_loaded_llama_engine.py::test_real_llama32_engine_greedy_matches_hf \
  -q --durations=0
```

> 本机使用的解释器为 `/home/lenovo/miniconda3/envs/micro-vllm/bin/python`；
> AutoDL 实例使用 `/root/autodl-tmp/venv/bin/python`。

---

## 8. 后续工作建议

1. 在 Ampere 及以上 GPU 上补一次原生 bf16 的 logits 容差对齐；当前 bf16 数据来自
   Turing 的模拟路径，只能证明"能跑且 argmax 一致"。
2. 把 `rope_scaling` 的表达上移到 `attention/spec.py`，让长上下文变体（Qwen2 的
   sliding window / 其它 rope 类型）走同一套 spec，而不是每个模型各自解析 config。
3. 补一个"位置超过 `original_max_position_embeddings` 后确实受益"的行为测试
   （例如 16k 输入下 scaled 与 unscaled 的输出差异），现在只有 8400 token 的
   正确性对齐，没有质量层面的对比。
4. 统一 `_inverse_frequencies` 对缺失字段的报错：目前直接 `rope_scaling["factor"]`
   会抛 `KeyError`，可改为带字段名的 `ValueError`。
