# W4A16 权重量化报告：加载时对称 per-group int4

- 报告日期：2026-11-05
- 代码范围：`layers/quantization/` 子包（reference 路径 + 融合 int4 kernel）、
  `ModelConfig.quantization`、模型加载接线、加载后量化，以及真机验证脚本
- 目标：按 `MICRO_VLLM_CAPABILITY_REPORT.md` 4.1 的第一版范围，实现
  **W4A16 weight-only、对称 per-group、从 FP16/BF16 checkpoint 在加载时量化**，
  并量化其存储、显存、数值精度与吞吐
- 运行环境：
  - 本机：WSL2 / Python 3.10.22 / PyTorch 2.8.0+cu128 / transformers 4.57.6，
    RTX 3060 Laptop **6144 MiB**（SM 8.6）
  - AutoDL：PyTorch 2.4.0+cu121 / transformers 4.57.6 / Python 3.10.8，
    RTX 3080 **10240 MiB**（SM 8.6），12 vCPU Xeon 8255C
  - 3080 的 torch 2.4 与 2.5+ 打包输入格式差异由源码内的版本分支处理，见 §4.6.4
- 验证模型：
  - `Qwen3-0.6B`（本机；28 层，hidden 1024，GQA 16/8，`head_dim=128`，tied embedding）
  - `Llama-3.2-1B-Instruct` 与 `Qwen3-1.7B`（AutoDL；1B：16 层 / hidden 2048 /
    GQA 32/8 / `head_dim=64`；1.7B：28 层 / hidden 2048 / GQA 16/8 / `head_dim=128`）
- 全量回归：**673 passed, 14 skipped**（skip 均为需要 `MICRO_VLLM_TEST_MODEL=1`
  的外部模型测试）

---

## 0. 结论速览

| 项目 | 结论 |
|---|---|
| 实现范围 | 对称 W4A16；量化 q/k/v/o 与 gate/up/down 共 7 个投影；不量化 embedding、RMSNorm、bias、lm_head |
| 量化语义 | `scale = max(|w_group|)/7`、`q = clamp(round(w/scale), -8, 7)`、`w_hat = q*scale`；每元素误差严格 `≤ scale/2` |
| 权重存储 | 投影权重 **840.0 → 216.6 MiB（-74.2%）**；全模型比例：0.6B **0.452**、1B **0.416**、1.7B **0.392**（g128）。融合路径运行时常驻再加一份打包缓存（0.6B +226 MiB） |
| 显存收益 | 相同 70% 显存预算下 KV block：0.6B **+33.0%**、1B **+31.4%**、1.7B **+59.8%**（reference / auto）；`int4pack` 因打包缓存被 profiling 重复计入，0.6B / 1.7B 只有 **+8.6% / +14.0%** |
| 数值正确性 | 单元测试全绿；每元素误差上界实测比值为 `1.0000002`；三个模型上 HF 独立对拍的 W4A16 PPL 与 micro 相差均 **< 0.5%** |
| 模型质量 | ❌ naive RTN 仍不可服务，但**强烈依赖模型规模**：g128 下 PPL 退化 Qwen3-0.6B **+42.7%**、Llama-1B **+28.4%**、Qwen3-1.7B **+10.7%**；g32 下分别为 **+30.0% / +11.0% / +6.0%** |
| 吞吐 | reference 反量化路径 decoding 约为 FP16 的 **0.3–0.5×**；融合 `_weight_int4pack_mm` 后 decode 为 reference 的 **1.87×**，与 FP16 持平或更快 |
| 融合 kernel | 数值与 reference 一致（0.6B PPL 偏差 0.02%，1.7B 0.11%）；3060 与 3080 上都是 M ≤ 512 时比 reference 快、M ≥ 1024 时变慢。阈值 512 已在两块卡上验证，建议采纳 |
| 兼容性 | torch 2.5+ 的 `_convert_weight_to_int4pack` 接受 `uint8 [N, K/2]`，torch 2.4 要求 `int32 [N, K]`；源码已按 `torch.__version__` 分支构造两种输入。3080 全量数据由真实源码直接取得，无需 shim |
| 结论 | 量化流水线本身正确、显存收益明确；瓶颈是**量化算法（RTN）而非实现**。模型越大越能吸收量化误差，但 g128 即使 1.7B 仍有 ~11% PPL 退化；若要可服务的质量，需要 GPTQ/AWQ 或至少 group 32 + 混合精度 |

---

## 1. 实现

### 1.1 量化语义（reference）

实现在 [reference.py](../src/micro_vllm/layers/quantization/reference.py)：

```text
scale = max(abs(w_group)) / 7          # 每组一个 FP32 scale
q     = clamp(round(w / scale), -8, 7) # 每元素 int4
w_hat = q * scale                      # 反量化
```

权重按 `in_features` 方向分组（`weight` 形状为 `[out_features, in_features]`，
group 沿最后一维连续切分）。两个 nibble 打包进一个 `uint8`：高 4 bit 存偶数
列、低 4 bit 存奇数列；解包时用 `int16` 做符号扩展
（`((x ^ 8) - 8)`），避免 `int8` 截断。

由 `round` 的定义可直接得到逐元素误差上界：

```text
|w - w_hat| <= scale / 2
```

对线性层输出的误差上界为：

```text
|ŷ_j - y_j| <= Σ_i |x_i| * scale[j, group(i)] / 2
```

`group_size` 必须是正偶数，且 `in_features % group_size == 0`；不满足时在
`W4A16Linear` 构造阶段抛出 `ValueError`，由 `apply_quantization` 带上层名
转成可操作的错误信息。

### 1.2 `W4A16Linear`

实现在 [w4a16.py](../src/micro_vllm/layers/quantization/w4a16.py)：

- `weight`：`uint8` buffer，形状 `[out_features, in_features // 2]`
- `scale`：FP32 buffer，形状 `[out_features, in_features // group_size]`
- `bias`：普通浮点 `nn.Parameter`（有 bias 的模型，如 Qwen2 的 q/k/v）
- `load_weight(tensor)`：FP32 量化 → 打包 → 原地写入两个 buffer
- `forward(x)`：解包 → 反量化 → `F.linear(x, w_hat.to(x.dtype), bias)`

关键 dtype 行为：`model.to(dtype=floating)` 会转换浮点 buffer（`scale`、`bias`），
但不会改 `uint8` 的 `weight`。`forward` 在反量化后把权重转成激活 dtype，
避免 `float32 weight × float16 activation` 的隐式提升。

`W4A16Linear` 有三条路径，由 `kernel` 选择：

- `reference`：上述 `unpack → dequantize → cast → F.linear`，纯可读实现，
  在任何设备上都能跑；
- `int4pack`：把 `uint8` 的符号 nibble 转成 kernel 要求的无符号 0–15，
  重打包成 `torch._convert_weight_to_int4pack` 的 tiled 布局（int32），
  scale 转成 `[K/group, N, 2]` 的 bfloat16 `(scale, zero)` 后缓存；
  `forward` 调用 `torch.ops.aten._weight_int4pack_mm`；
- `auto`：`num_tokens <= FUSED_DECODE_MAX_TOKENS`（当前 32）时用融合 kernel，
  否则回退 reference。

融合路径的约束来自 kernel 本身：激活和 `scale_and_zero` 必须是 bfloat16，
`group_size ∈ {32, 64, 128, 256}`，`in_features % 128 == 0`、`out_features % 8 == 0`，
且需要 CUDA SM ≥ 8.0。`W4A16Linear` 在构造时校验形状与 group size，
`_ensure_int4pack` 在首次融合前向时校验设备并构建缓存；不满足时抛出
可操作的 `ValueError`/`RuntimeError`，不做静默回退。打包缓存是普通属性而不是
buffer，避免 `model.to(dtype)` 把 bfloat16 的 `scale_and_zero` 转成 fp16。

### 1.3 配置、替换与加载

- [config.py](../src/micro_vllm/config.py)：新增 `QuantizationConfig`
  （`method: "none" | "w4a16"`，`group_size` 默认 128），作为
  `ModelConfig.quantization` 的共享不可变契约；
- [apply.py](../src/micro_vllm/layers/quantization/apply.py)：遍历模型，
  只替换 **项目自己的 `Linear`**（`type(module) is Linear`），因此
  `LMHead`（`Linear` 子类）、`TokenEmbedding`、`RMSNorm` 不会被误量化；
- [model_loader/base.py](../src/micro_vllm/model_loader/base.py)：模型构造后、
  加载权重前调用 `apply_quantization`，保证 checkpoint tensor 走量化加载分支；
- [models/loading.py](../src/micro_vllm/models/loading.py)：按参数名找到
  `W4A16Linear` 后调用 `load_weight`；未知参数名继续返回 `False`，由各模型
  原有的 `KeyError` 兜底；
- `models/{llama,qwen,qwen2,mistral}.py`：四个模型族的 `load_weights` 都先尝试
  量化分支，再回退到普通 `copy_`。

### 1.4 明确不量化的部分

| 组件 | 处理 | 原因 |
|---|---|---|
| `embed_tokens` | FP16 | 词表大、对精度敏感；可行性报告明确排除 |
| `lm_head` | FP16 | tied embedding 时与 `embed_tokens` 共享权重；`type(module) is Linear` 排除 |
| `RMSNorm` | FP16 | 只有逐元素缩放，收益小、风险大 |
| `bias` | FP16 | 数量少；Qwen2 的 q/k/v bias 保持浮点 |
| `q_norm` / `k_norm` | FP16 | Qwen3 特有，作用在 `head_dim` 上 |

---

## 2. 单元与集成测试

| 文件 | 覆盖内容 |
|---|---|
| `tests/layers/quantization/test_reference.py` | 分组量化、打包/解包、负值符号扩展、误差上界 |
| `tests/layers/quantization/test_w4a16.py` | 与 dense 对拍、bias、group size 32/64/128、非法 group size、dtype 转换 |
| `tests/layers/quantization/test_apply.py` | 只替换项目 `Linear`；`LMHead`/`TokenEmbedding`/`RMSNorm` 保留；tied 身份不变；`method="none"` 无操作；group 不整除报错；`kernel` 正确下传 |
| `tests/layers/quantization/test_w4a16_fused.py` | 融合路径与 reference 在 group 32/64/128/256、三种投影形状、M=1/33 上对拍；3D 形状与 dtype 保持；`auto` 阈值；打包缓存与 `load_weight` 失效；CPU/非法形状报错 |
| `tests/models/test_quantized_loading.py` | 四类 tiny checkpoint 的加载；7 个投影全部量化；打包/scale 形状；`.to(fp16)` 后 `uint8` 不变；参数字节 < dense 的 0.5×；Qwen2 bias 保持 FP32 |
| `tests/test_model_config.py` | 默认 `none/128/reference`；接受 `w4a16`；拒绝未知 method、非法 group size 与未知 kernel；`method="none"` 只允许 reference |
| `tests/integration/test_loaded_llama_engine.py` | W4A16 tiny checkpoint 走完整 `ModelRunner + Engine`，2 token 生成 |

全量结果：**673 passed, 14 skipped**。

---

## 3. 真机验证方法

### 3.1 一次性验证脚本

[benchmarks/validate_w4a16.py](../benchmarks/validate_w4a16.py) 在**一次运行**内完成
CPU 存储统计、FP16 与 W4A16 两轮 GPU 测量、数值对拍和吞吐测量，输出一个 JSON：

```bash
python benchmarks/validate_w4a16.py \
    --model /home/lenovo/huggingface/Qwen3-0.6B \
    --group-size 128 --kernel reference \
    --output benchmarks/results/w4a16-validation-qwen3-0.6b-kernel-reference.json
```

`--kernel {reference,int4pack,auto}` 选择量化分支的 matmul 路径；一次运行只测
一种 kernel。脚本的 dense 基线与 CPU 存储统计与 kernel 无关。

测量项：

- **存储**：CPU 上分别加载 dense 与 W4A16 模型，按 tensor 身份去重后统计字节；
  tied `lm_head.weight` 只计一次；
- **显存**：`Worker` 初始化后的 `kv_cache_layout.num_blocks`、KV 字节、
  初始化后 / 全部请求结束后 / 峰值的 `torch.cuda.memory_allocated`
  （后两者能反映融合 kernel 的懒加载打包缓存）；
- **数值**：对 `README_EN.md` 前 1024 token 做一次完整 prefill，比较
  dense 与 W4A16 的 logits（cosine、top-1、top-5、绝对误差）和 PPL；
- **生成**：4 个固定 prompt，greedy、`ignore_eos=True`、32 token，比较前缀一致长度；
- **吞吐**：256 token prompt + 16 new token，batch=1，分别记录 prefill 与 decode 步耗时。

### 3.2 HF 独立对拍

[benchmarks/crosscheck_w4a16_hf.py](../benchmarks/crosscheck_w4a16_hf.py) 用
transformers 自己的 `AutoModelForCausalLM`（Qwen3 / Llama）前向，把 7 个
投影的权重替换成与 micro-vLLM 相同的量化/反量化结果，再算 PPL。它完全
绕开 micro-vLLM 的 `W4A16Linear`、KV cache 和 attention 路径，用来区分
“量化算法误差”与“运行时实现错误”。

---

## 4. 结果

### 4.1 存储

| 组件 | FP16 | W4A16 g128 | 比例 |
|---|---:|---:|---:|
| 7 个投影 | 840.0 MiB | 216.6 MiB | 25.8% |
| `embed_tokens`（tied，含 `lm_head`） | 296.8 MiB | 296.8 MiB | 100% |
| RMSNorm 等 | 0.1 MiB | 0.1 MiB | 100% |
| **合计** | **1136.9 MiB** | **513.4 MiB** | **45.2%** |

投影理论值：4 bit 权重为 FP16 的 1/4；每 128 个权重增加一个 2 B 的 FP16
scale，相当于每权重 `0.5 + 2/128 = 0.5156` B，为 FP16 的 25.8%，与实测一致。

不同 group size 的存储：

| group | 投影字节 | 全模型字节 | 比例 |
|---:|---:|---:|---:|
| 128 | 216.6 MiB | 513.4 MiB | 45.2% |
| 64 | 223.1 MiB | 520.0 MiB | 45.7% |
| 32 | 236.2 MiB | 533.1 MiB | 46.9% |

### 4.2 显存与 KV 容量

`gpu_memory_utilization=0.7`，block size 16：

| 指标 | FP16 | W4A16 g128 | 变化 |
|---|---:|---:|---:|
| KV blocks | 1139 | 1515 | **+33.0%** |
| KV 字节 | 1993.2 MiB | 2651.2 MiB | +33.0% |
| 可缓存 token | 18,224 | 24,240 | +33.0% |
| 初始化后 allocated | 3161.0 MiB | 3175.5 MiB | 基本持平 |
| 峰值 allocated | 3571.8 MiB | 3586.3 MiB | 基本持平 |

解释：总预算固定为显存的 70%，量化省下的约 623 MiB 被 `ModelRunner`
自动换算成更多 KV block，因此“steady allocated”几乎不变，收益体现在
**并发请求/上下文容量 +33%**，而不是峰值下降。若把 `gpu_memory_utilization`
固定为同一个绝对字节数，则量化会直接降低显存占用。

### 4.3 数值质量

PPL 文本：`README_EN.md` 前 1024 token；reference attention；FP16。

| group | PPL dense | PPL W4A16 | ΔPPL | cosine 均值 | top-1 一致率 | top-5 重叠（/5） | 平均绝对误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 19.1419 | 27.3120 | **+42.68%** | 0.9424 | 70.51% | 3.39 | 1.046 |
| 64 | 19.1419 | 26.7123 | +39.55% | 0.9552 | 73.34% | 3.51 | 0.908 |
| 32 | 19.1419 | 24.8783 | +29.97% | 0.9660 | 75.59% | 3.70 | 0.785 |

HF 独立对拍：

| 配置 | HF dense PPL | HF W4A16 PPL | micro W4A16 PPL | micro/HF 偏差 |
|---|---:|---:|---:|---:|
| g128, all | 19.1369 | 27.3773 | 27.3120 | -0.24% |
| g32, all | 19.1369 | 24.9939 | 24.8783 | -0.46% |

dense 路径本身与 HF 的差距仅 0.03%，说明 micro-vLLM 的模型、RoPE、attention、
KV cache 路径没有引入额外误差；量化后端与 HF 的差距 < 0.5%，说明运行时
解包/反量化也正确。

### 4.4 贪心生成

4 个 prompt × 32 token，统计与 FP16 输出相同的**前缀长度**：

| group | The capital of France is | fibonacci | primary colors | self-attention |
|---:|---:|---:|---:|---:|
| 128 | 2 | 0 | 2 | 0 |
| 64 | 2 | 9 | 4 | 0 |
| 32 | 2 | 17 | 4 | 1 |

即使 group 32，多数 prompt 也在前几个 token 内分叉；这不是可接受的服务质量。

### 4.5 吞吐（batch=1，reference attention）

同一台 3060 上，g128 的三种 kernel 各跑一次 `validate_w4a16.py`。
dense 基线在三次运行间有 ±20% 的波动（笔记本 GPU 的功耗/温度策略），
因此更可靠的是同一轮内的比例，以及直接比较 W4A16 的稳态步耗时。

| kernel | prefill tok/s | prefill / 本轮 FP16 | decode tok/s | decode / 本轮 FP16 | decode 稳态步耗时 |
|---|---:|---:|---:|---:|---:|
| reference | 1838.4 | 0.72× | 8.19 | 0.62× | 120.9 ms |
| int4pack | 2425.9 | 0.83× | 15.46 | 1.25× | 64.5 ms |
| auto（阈值 32） | 1750.3 | 0.59× | 13.26 | 0.84× | 65.2 ms（首步 203.8 ms） |

reference 路径每次前向都要重新执行 `unpack → dequantize → cast`，M=1 时这部分
固定开销约为 66 ms/步（28 层），因此 decode 只有 FP16 的 0.6× 左右；
融合 kernel 把这一步换成一次 int4 GEMM，稳态 decode 步耗时从 120.9 ms
降到 64.5 ms（**1.87×**），与同轮 FP16 持平或更快。

`auto` 的 prefill 走 reference（256 > 32），所以 prefill 与 reference 同档；
decode 首步 203.8 ms 包含一次性的打包成本（约 139 ms，28 层的全部投影），
之后与 `int4pack` 完全一致。详见 §4.6。

### 4.6 融合 kernel：`_weight_int4pack_mm`（本机 RTX 3060）

#### 4.6.1 端到端：正确性、吞吐与显存

同一权重、同一 `validate_w4a16.py` 流程，只改 `--kernel`：

| kernel | PPL | logits cosine | top-1 | KV blocks | post-run allocated | peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| reference | 27.3120 | 0.94240 | 70.51% | 1515 | 3176 MiB | 3586 MiB |
| int4pack | 27.3176 | 0.94267 | 70.61% | 1237 | 2939 MiB | 3351 MiB |
| auto | 27.3120 | 0.94240 | 70.51% | 1515 | 3402 MiB | 3586 MiB |

- 融合路径与 reference 的 PPL 差 **0.02%**，cosine/top-1 在测量噪声内，
  4 个 prompt 的贪心前缀一致长度完全相同（2/0/2/0）。说明 bfloat16 激活、
  unsigned nibble 重排和 `[K/group, N, 2]` scale 布局都正确。
- `auto` 的 PPL/logits 与 reference 完全相同，因为 1024 token 的 logits
  prefill 超过阈值 32，走的是 reference。
- decode 稳态步耗时：reference 120.9 ms，int4pack 64.5 ms，auto 65.2 ms
  （**1.87×**）。prefill 256 token：reference 1838 tok/s，int4pack 2426 tok/s，
  auto 1750 tok/s。

#### 4.6.2 层级别扫描：阈值应该定在哪

端到端只测了 M=256（prefill）和 M=1（decode）。为了确定 crossover，
[benchmarks/bench_w4a16_kernel.py](../benchmarks/bench_w4a16_kernel.py) 在
Qwen3-0.6B 的 7 个投影形状上，对每个 M 交替测量 dense / reference / int4pack
的完整层前向（CUDA event 中位数，30 次重复）：

| M | dense (ms) | reference (ms) | int4pack (ms) | fused / ref |
|---:|---:|---:|---:|---:|
| 1 | 0.175 | 2.361 | 0.125 | **18.85×** |
| 2 | 0.196 | 2.374 | 0.128 | 18.59× |
| 4 | 0.193 | 2.379 | 0.128 | 18.55× |
| 8 | 0.195 | 2.379 | 0.150 | 15.82× |
| 16 | 0.183 | 2.371 | 0.188 | 12.62× |
| 32 | 0.211 | 2.395 | 0.281 | 8.51× |
| 64 | 0.234 | 2.424 | 0.457 | 5.31× |
| 128 | 0.325 | 2.514 | 0.805 | 3.12× |
| 256 | 0.508 | 2.713 | 1.522 | 1.78× |
| 512 | 0.915 | 3.108 | 2.874 | 1.08× |
| 1024 | 1.753 | 3.945 | 5.733 | 0.69× |
| 2048 | 2.810 | 4.997 | 11.240 | 0.44× |

表中是“一层 decoder 的 7 个投影之和”；乘以 28 层即为整个模型的投影耗时。
M=1 时 reference 的固定开销 28 × 2.36 ≈ 66 ms/步，正好解释端到端
120.9 − 64.5 ≈ 56 ms 的差距。

结论：

- reference 的成本几乎与 M 无关（M=1 到 512 只从 2.36 涨到 3.11 ms），
  因为它每次前向都要对所有 15.6M 权重做 unpack + dequant + cast；
- int4pack 的耗时随 M 近似线性增长，因此 **M ≤ 512 时融合更快**
  （M=512 仍有 1.08×），到 M=1024 反转（0.69×）；
- 分形状看，`q_proj` 在 M=512 已经略慢（0.91×），其余投影到 512 仍
  ≥ 1.0×。所以单层上限在 512 附近，整体在 512–1024 之间反转。

当前 `FUSED_DECODE_MAX_TOKENS=32` 是上一轮**只测 kernel、且对手是
“已反量化好的 fp16 权重”**时定的；本轮把对手换成真实的 reference 路径后，
32–512 这段 prefill 也被证明是融合路径更快。因此 `auto` 目前把最有利的
区间关掉了：它的 prefill 与 reference 同档（0.59× FP16），而 `int4pack`
是 0.83× FP16。

RTX 3080 上的复扫（§4.6.4）得到同样的结论：M ≤ 512 时融合更快，
M=1024 反转。两块卡、两个模型下 512 都是安全阈值，且引擎的
`max_num_batched_tokens` 正好是 512，因此**建议把
`FUSED_DECODE_MAX_TOKENS` 从 32 改为 512**。这是源码改动，本报告未修改默认值。

#### 4.6.3 打包缓存的显存代价

融合路径除了保留 `uint8` 权重，还缓存一份 int32 tiled 权重和 bfloat16
`scale_and_zero`：

| 指标 | reference | int4pack | auto |
|---|---:|---:|---:|
| 运行后 allocated | 3176 MiB | 2939 MiB | 3402 MiB |
| 相对 reference 的打包缓存 | — | — | **+226 MiB** |
| KV blocks（相对 FP16 1139） | 1515（+33.0%） | 1237（+8.6%） | 1515（+33.0%） |

+226 MiB 约等于 437M 个投影参数各多 0.5 B，再加上每权重 0.0625 B 的
bfloat16 scale/zero；它是 513 MiB 量化 checkpoint 的 44%。稳定运行时
量化权重总占用因此约为 739 MiB（dense 的 65%），而不是 CPU 存储表里的 513 MiB。

`int4pack` 的 KV blocks 比 reference 少 18.4%，除了多出的缓存，还因为
`ModelRunner._profile_model_memory` 把打包缓存**算了两遍**：缓存在 warmup
中构建并持续存活，既计入 `non_kv_cache_memory_bytes`，又计入
`peak_activation_memory_bytes`（peak − baseline）。按单次计入估算，
KV 应约为 1386 blocks（较 FP16 +21.7%）。`auto` 在 profiling 之后才构建缓存，
所以 KV 仍是 1515 blocks，但首次 decode 步要付 139 ms 打包停顿。

两项都可以通过“加载后立即调用 `_ensure_int4pack()`”消除：停顿移到初始化，
profiling 只计一次缓存。这属于后续源码改动，本报告未修改实现。

#### 4.6.4 RTX 3080（Qwen3-1.7B）与 torch 2.4

同一套 `--kernel` 流程在 AutoDL RTX 3080、Qwen3-1.7B、g128 上重跑。
下表数据全部由打过版本分支的源码直接生成，没有使用任何 shim：

| kernel | prefill tok/s | decode tok/s | decode 稳态步 | KV blocks（FP16 1858） | post-run allocated | PPL |
|---|---:|---:|---:|---:|---:|---:|
| reference | 1957.3 | 8.20 | 123.2 ms | 2970（+59.8%） | 6503 MiB | 16.6448 |
| int4pack | 3100.5 | 19.03 | 52.0 ms | 2118（+14.0%） | 5716 MiB | 16.6266 |
| auto（阈值 32） | 1963.0 | 16.93 | 53.7 ms（首步 134.1 ms） | 2970（+59.8%） | 7217 MiB | 16.6448 |

同轮 FP16 基线：prefill 4677–4930 tok/s，decode 19.85–20.56 tok/s。

- 融合 decode 是 reference 的 **2.32×**，与本轮 FP16 基本持平（0.94×）；
  prefill 是 reference 的 **1.58×**，但仍低于 FP16（0.63×）；
- 数值与 reference 一致：PPL 16.6266 vs 16.6448（+0.11%），top-1 相同，
  4 个 prompt 的贪心前缀长度完全相同；
- 打包缓存 **+714 MiB**（7217 − 6503），约为 1.7B 量化 checkpoint 的 55%；
  `int4pack` 的 KV blocks 只有 +14.0%，同样是因为 profiling 重复计入缓存。
  按单次计入估算应约为 2562 blocks（较 FP16 +37.9%）。

层级别扫描在 3080 上与 3060 趋势一致：

| M | reference (ms) | int4pack (ms) | fused / ref |
|---:|---:|---:|---:|
| 1 | 3.670 | 0.645 | **5.69×** |
| 32 | 3.698 | 0.696 | 5.31× |
| 256 | 3.957 | 2.098 | 1.89× |
| 512 | 4.549 | 3.823 | 1.19× |
| 1024 | 5.393 | 7.364 | 0.73× |
| 2048 | 7.520 | 14.443 | 0.52× |

M=1 的 fused/reference 比值从 3060 的 18.85× 降到 5.69×，原因是 3080 的
显存带宽约为 3060 的 2.3×，reference 的 unpack/dequant 便宜得多；但融合
路径仍然全面更快，crossover 也仍在 512–1024 之间（`q_proj`、`o_proj`、
`gate_proj`、`up_proj` 在 512 后反转，k/v 到 1024 才反转，`down_proj` 更早，
在 256 后反转）。

**torch 2.4 兼容性**：torch 2.4 的 `_convert_weight_to_int4pack` 要求
`int32 [N, K]`（每个元素一个无符号 nibble），torch 2.5+ 改为
`uint8 [N, K/2]`（每字节两个 nibble）。`_ensure_int4pack` 现在按
`torch.__version__` 的前两段是否 `>= (2, 5)` 选择输入格式：新版本把两个
nibble 拼成一个 `uint8`，2.4 则把每个无符号 nibble 展开成 `int32 [N, K]`。
两条路径共用同一份 `unpack_int4` + `+8` 的结果，区别只在打包调用的一次性
开销。该分支已在 3080（torch 2.4.0+cu121）上直接跑通，数值与 shim 版本一致
（PPL 16.6266）；`auto` 首步 134.1 ms 是真实的打包停顿，不再是 shim 开销。

### 4.7 更大模型验证：RTX 3080 10 GB

本机 6 GB 只能装下 0.6B 模型。为了验证“量化误差是否随模型规模下降”，
在 AutoDL RTX 3080（10 GB，SM 8.6）上对 `Llama-3.2-1B-Instruct` 与
`Qwen3-1.7B` 重复了同一套流程。测法、文本（`README_EN.md` 前 1024 token）、
dtype（FP16）、attention backend 与 `gpu_memory_utilization=0.7` 均与
0.6B 实验一致。

#### 4.7.1 存储与显存

| 模型 | group | 投影字节 FP16 → W4A16 | 全模型字节 | 比例 | KV blocks | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 128 | 840.0 → 216.6 MiB | 1136.9 → 513.4 MiB | 0.452 | 1139 → 1515 | +33.0% |
| Llama-3.2-1B | 128 | 1856.0 → 478.5 MiB | 2357.1 → 979.6 MiB | 0.416 | 8277 → 10872 | +31.4% |
| Qwen3-1.7B | 128 | 2688.0 → 693.0 MiB | 3281.7 → 1286.7 MiB | 0.392 | 1858 → 2970 | +59.8% |

1.7B 的 KV 增幅远大于 1B，是因为 1.7B 的 FP16 权重占比更高、剩余给 KV 的
预算更少，量化释放的约 2 GB 直接转成了更多 block；1B 的 FP16 权重只有
2.3 GB，基线 KV 就已经很大，量化带来的相对增益被稀释。

#### 4.7.2 质量

| 模型 | group | PPL dense | PPL W4A16 | ΔPPL | cosine | top-1 | top-5（/5） | 贪心前缀（4×32 token） |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | 128 | 19.1419 | 27.3120 | +42.68% | 0.9424 | 70.5% | 3.39 | 2/0/2/0 |
| Qwen3-0.6B | 64 | 19.1419 | 26.7123 | +39.55% | 0.9552 | 73.3% | 3.51 | 2/9/4/0 |
| Qwen3-0.6B | 32 | 19.1419 | 24.8783 | +29.97% | 0.9660 | 75.6% | 3.70 | 2/17/4/1 |
| Llama-3.2-1B | 128 | 18.7771 | 24.1005 | +28.35% | 0.9461 | 71.1% | 3.54 | 5/0/3/0 |
| Llama-3.2-1B | 64 | 18.7771 | 22.3450 | +19.00% | 0.9589 | 77.8% | 3.68 | 5/0/32/2 |
| Llama-3.2-1B | 32 | 18.7771 | 20.8392 | +10.98% | 0.9692 | 80.5% | 3.86 | 32/0/3/3 |
| Qwen3-1.7B | 128 | 15.0377 | 16.6448 | +10.69% | 0.9482 | 79.9% | 3.76 | 1/10/2/8 |
| Qwen3-1.7B | 64 | 15.0377 | 15.9913 | +6.34% | 0.9627 | 82.8% | 3.91 | 1/10/2/2 |
| Qwen3-1.7B | 32 | 15.0377 | 15.9467 | +6.04% | 0.9701 | 83.7% | 4.05 | 1/11/2/2 |

观察：

- **规模效应明显**：同为 g128，0.6B +42.7% → 1B +28.4% → 1.7B +10.7%。
  1.7B 的 top-1 一致率接近 80%，而 0.6B 只有 70.5%。
- **group size 仍然有效**：1.7B 从 g128 的 +10.7% 降到 g32 的 +6.0%，
  1B 从 +28.4% 降到 +11.0%；把 group 从 128 缩到 32 只多占 4.9%（1.7B）
  与 4.4%（1B）的全模型字节。
- **仍不足以服务**：即使 1.7B + g32，PPL 仍 +6%，贪心输出在 1–11 个
  token 内分叉；典型生产 4-bit（GPTQ/AWQ）目标是个位数百分比以下甚至 <1%。
- **家族差异存在**：同为 ~1–2B，Qwen3-1.7B（+10.7%）明显好于
  Llama-3.2-1B（+28.4%）；除规模外，训练分布与权重重尾程度也影响鲁棒性。

#### 4.7.3 HF 独立对拍

| 模型 | HF dense PPL | HF W4A16 PPL | micro W4A16 PPL | micro/HF 偏差 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 19.1369 | 27.3773 | 27.3120 | -0.24% |
| Llama-3.2-1B | 18.7731 | 24.1463 | 24.1005 | -0.19% |
| Qwen3-1.7B | 15.0385 | 16.5893 | 16.6448 | +0.33% |

三个模型、两种架构（Llama/Qwen3）、两种 RoPE 配置下，micro-vLLM 与 HF
用同一组反量化权重得到的 PPL 都在 0.5% 以内，进一步确认运行时路径正确。

#### 4.7.4 吞吐

RTX 3080 上（batch=1，256 token prefill + 16 new token）：

| 模型 | prefill tok/s FP16 → W4A16 | decode tok/s FP16 → W4A16 |
|---|---:|---:|
| Llama-3.2-1B | 7985.8 → 2978.7（0.37×） | 38.64 → 12.54（0.32×） |
| Qwen3-1.7B | 4962.9 → 1978.2（0.40×） | 20.45 → 8.37（0.41×） |

注意 0.6B 的 0.50× 是在 RTX 3060 上测的，与 3080 的绝对值不可直接横比。
3080 的 FP16 GEMM 更快，而 `unpack + dequant` 是逐层执行的固定开销，
因此相对占比反而更大（1B 为 0.32×）。1.7B 回升到 0.41×，是因为 decode
更受权重带宽限制，4-bit 读取的字节更少，部分抵消了反量化开销。共同结论
仍是：未融合的 `unpack + dequant` 让 W4A16 慢于 FP16，瓶颈不在格式本身。

---

## 5. 误差分析：为什么 PPL 退化这么大

### 5.1 量化器本身是精确的

在 `Qwen3-0.6B` 第 0 层真实权重上逐元素检查：

```text
max( |w - w_hat| / (scale/2) ) = 1.0000002
```

误差严格落在理论边界内；打包/解包在 `-8` 等负值极值上也已由单测覆盖。
因此问题不在“算错了”，而在“4-bit RTN 本身的误差”。

### 5.2 为什么 group 128 的权重相对误差约有 12%

第 0 层各投影的实测相对 Frobenius 误差：

| 投影 | std | absmax | g128 | g64 | g32 | g16 |
|---|---:|---:|---:|---:|---:|---:|
| q_proj | 0.0313 | 0.645 | 12.4% | 11.2% | 10.0% | 8.7% |
| k_proj | 0.0306 | 0.434 | 12.0% | 10.9% | 9.8% | 8.6% |
| o_proj | 0.0258 | 0.422 | 11.8% | 10.8% | 9.8% | 8.6% |
| gate_proj | 0.0366 | 0.391 | 12.0% | 10.9% | 9.8% | 8.6% |
| down_proj | 0.0262 | 0.418 | 12.4% | 11.2% | 10.0% | 8.7% |

推导：

```text
scale ≈ max(group) / 7
量化噪声 std ≈ scale / sqrt(12)
```

该模型权重 std≈0.031，group 128 的 group max 约 2.9–3.2 倍 std，
于是 `scale ≈ 0.40–0.46 std`，噪声 std ≈ `0.12 std`，与实测 12% 吻合。
权重分布有重尾（kurtosis 6.76，absmax 达 20.6 倍 std），把 group 缩小到 32
只能把 group max 降到约 2.5 倍 std，因此误差只从 12.4% 降到 10.0%，
收益远小于理想高斯假设下的预期。

另一个被排除的因素：group 内均值只有 `0.17 × scale`（p99 为 0.58），
**非对称量化（zero point）不会带来实质改善**；第一版选择对称是合理的。

跨模型对比（各模型第 0 层 `q_proj`，相对 Frobenius 误差）：

| 模型 | std | absmax | kurtosis | g128 | g64 | g32 | g16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.0313 | 0.645 | 6.76 | 12.4% | 11.2% | 10.0% | 8.7% |
| Llama-3.2-1B | 0.0361 | 0.676 | 7.03 | 12.8% | 11.5% | 10.2% | 8.9% |
| Qwen3-1.7B | 0.0352 | 0.656 | 4.67 | 12.8% | 11.5% | 10.1% | 8.8% |

**权重量化误差几乎与模型规模无关**：三个模型的 g128 误差都在 12.4–12.8%。
但 PPL 退化从 0.6B 的 +42.7% 降到 1.7B 的 +10.7%，说明规模效应来自模型
本身的冗余度和表示能力——同样的权重扰动，在更大的模型里被更多通道和更深
的表示吸收。这也解释了为什么 Llama-1B（+28.4%）虽比 0.6B 好，却仍差于
参数量只多 70% 的 Qwen3-1.7B：Llama-1B 的分布更重尾（kurtosis 7.03 vs
4.67），但两者 g128 的权重误差相同（12.8%），因此差异主要来自 1.7B 的
绝对冗余，而不是量化噪声本身。

### 5.3 不是运行时 bug

| 证据 | 结果 |
|---|---|
| micro dense vs HF dense（Qwen3-0.6B） | 19.1419 vs 19.1369（0.03%） |
| micro W4A16 vs HF 用同一组反量化权重（0.6B） | 27.3120 vs 27.3773（0.24%） |
| group 32 重复对拍（0.6B） | 24.8783 vs 24.9939（0.46%） |
| Llama-3.2-1B，g128 | 24.1005 vs 24.1463（0.19%） |
| Qwen3-1.7B，g128 | 16.6448 vs 16.5893（0.33%） |

micro 与 HF 的差异只来自 FP16 数值细节，量级远小于量化本身的误差；两个
新增模型覆盖了 Llama 与 Qwen3 两种架构，结论不变。

### 5.4 各层类型的贡献

在 HF 上只量化一组投影（group 128，all 为完整量化）：

| 量化范围 | PPL | 相对 dense |
|---|---:|---:|
| 不量化 | 19.14 | — |
| 只量化 attention（q/k/v/o） | 24.19 | +26.4% |
| 只量化 MLP（gate/up/down） | 21.85 | +14.2% |
| 量化全部 | 27.38 | +43.1% |

两类层各自都带来明显退化，叠加后进一步放大；0.6B 模型的冗余度较低，
没有足够能力吸收这种逐层扰动。这与 §4.7 的规模效应一致：1.7B 在相同
权重误差下只退化 +10.7%，说明“模型越大越能吸收量化扰动”是主要原因。

### 5.5 与生产量化方法的差距

naive RTN（round-to-nearest）只用权重本身，不做任何**输出误差补偿**。
GPTQ 用校准集的 Hessian 逐列补偿量化误差，AWQ 用激活分布搜索 per-channel
缩放；两者在同样的 4-bit / group-128 存储下通常能把 PPL 退化压到几个百分点。
本报告的结果应被视为“无校准、无补偿基线”的下界，而不是 W4A16 格式的
理论上限。

---

## 6. 结论与后续建议

### 6.1 已完成

- W4A16 的 reference 量化/打包/反量化路径、`W4A16Linear`、配置、加载时替换、
  四类模型的加载接线与 Engine 集成全部完成；
- 量化误差满足逐元素理论上界，运行时与 HF 独立实现对拍一致（3 个模型，
  Llama/Qwen3 两种架构，偏差均 < 0.5%）；
- 权重存储降低：0.6B 54.8%、1B 58.4%、1.7B 60.8%；
- 在固定显存预算下 KV 容量提升：0.6B +33.0%、1B +31.4%、1.7B +59.8%；
- 质量随规模改善：g128 下 PPL 退化 0.6B +42.7%、1B +28.4%、1.7B +10.7%；
- 融合 `_weight_int4pack_mm` 路径完成：与 reference 数值一致（0.6B PPL 差
  0.02%，1.7B 差 0.11%），decode 稳态步耗时 0.6B 120.9 → 64.5 ms（1.87×）、
  1.7B 123.2 → 52.0 ms（2.37×），M ≤ 512 的 prefill 也比 reference 快
  （0.6B 1.32×、1.7B 1.58×）；torch 2.4/2.5+ 的打包输入差异已由源码分支处理；
- 建立了可复现的 `validate_w4a16.py`（支持 `--kernel`）、
  `bench_w4a16_kernel.py` 与 `crosscheck_w4a16_hf.py`。

### 6.2 当前不适合直接上线

naive RTN 在 1–2B 模型上仍有明显退化：Llama-3.2-1B g128 PPL +28.4%，
Qwen3-1.7B 最好的 g32 也有 +6.0%；32 token 贪心输出多数 prompt 在前
1–11 个 token 内分叉，只有 Llama-1B + g32 的 “France” prompt 完整匹配
32 token。它适合作为**正确性基线和存储/显存实验**，不适合作为默认可用的
量化推理路径。

### 6.3 建议的下一步（按性价比排序）

1. **GPTQ/AWQ 误差补偿**（若要把量化质量做进简历项目）：
   - GPTQ：按层用校准集（128 条左右）估计 Hessian，逐列量化并补偿残差；
   - AWQ：用激活分布搜索 per-channel 缩放，再量化；
   - 预期目标：同尺寸下 PPL 退化 < 5%，贪心输出与 FP16 的一致性显著提升；
   - 需要新增校准数据配置与离线/加载时校准流程，量化格式仍可复用现有
     `qweight + scale`。
2. **混合精度**：若不想引入校准，可只量化 MLP（0.6B 上 +14.2%）或只量化
   attention（+26.4%），用一部分存储收益换取可接受的质量；但两者都不是
   理想解。更大模型上单侧量化的退化应重新测（按 §4.7 的规模效应，1.7B
   上大概率会明显低于 0.6B 的数字）。
3. **融合 kernel 的收尾**（已实现，剩两项）：
   - ✅ torch 2.4 兼容分支已合入：`_ensure_int4pack` 按 `torch.__version__`
     分支，< 2.5 用 `int32 [N, K]`，否则用 `uint8 [N, K/2]`；3080 验证已
     不再需要 shim；
   - 把 `FUSED_DECODE_MAX_TOKENS` 从 32 提到 512：3060（0.6B）与 3080（1.7B）
     上 M=512 仍分别有 1.08× / 1.19×，M=1024 才反转；
   - 在加载完成后立即调用 `_ensure_int4pack()`，消除 `auto` 首次 decode 的
     打包停顿（0.6B 139 ms、1.7B 134 ms），并避免 profiling 把打包缓存
     计算两遍（KV 容量：0.6B 约 1386 blocks，1.7B 约 2562 blocks）。
4. **继续原计划 M3 FlashAttention**：W4A16 的实现与验证已经闭环，质量结论
   明确；FlashAttention 与量化互不阻塞，可按原优先级推进。

### 6.4 本报告的局限

- 最大只测到 Qwen3-1.7B（RTX 3080 10 GB 的容量上限）；7B 及以上未验证，
  按现有趋势应更宽容，但这是推断而非实测；
- 每个模型、每个 group size 只用了 `README_EN.md` 前 1024 token 的 PPL，
  未做下游任务评测（MMLU/GSM8K 等），也没有做多文本方差估计；
- 贪心一致性只用了 4 个 prompt × 32 token，样本量小，适合看量级而非精确
  比较；
- 吞吐使用 reference attention，绝对值受 Python 元数据路径影响，只适合作
  相对比较；RTX 3080 与 RTX 3060 的绝对值不可直接横比；
- 融合 kernel 已在 3060（torch 2.8）与 3080（torch 2.4）上验证；2.4 路径由
  源码内的版本分支生成打包输入，没有用 shim，但 2.4 上的其它模型/形状未做
  额外扫参；
- 层级别扫描的是完整层前向，不含 attention 与 MLP 逐元素算子；它用于确定
  kernel 的相对区间，不用于预测端到端绝对吞吐。

---

## 7. 复现命令

```bash
# 单元/集成测试
python -m pytest -q

# 真机验证：group 128（默认）、64、32
python benchmarks/validate_w4a16.py --model /home/lenovo/huggingface/Qwen3-0.6B \
    --group-size 128 --output benchmarks/results/w4a16-validation-qwen3-0.6b.json
python benchmarks/validate_w4a16.py --model /home/lenovo/huggingface/Qwen3-0.6B \
    --group-size 64 --output benchmarks/results/w4a16-validation-qwen3-0.6b-g64.json
python benchmarks/validate_w4a16.py --model /home/lenovo/huggingface/Qwen3-0.6B \
    --group-size 32 --output benchmarks/results/w4a16-validation-qwen3-0.6b-g32.json

# 三种 kernel（g128）端到端对比
for kernel in reference int4pack auto; do
  python benchmarks/validate_w4a16.py \
      --model /home/lenovo/huggingface/Qwen3-0.6B \
      --group-size 128 --kernel $kernel \
      --output benchmarks/results/w4a16-validation-qwen3-0.6b-kernel-$kernel.json
done

# 层级别 kernel 扫描（确定 fused/reference crossover）
python benchmarks/bench_w4a16_kernel.py \
    --model /home/lenovo/huggingface/Qwen3-0.6B \
    --output benchmarks/results/w4a16-kernel-sweep-qwen3-0.6b.json

# HF 独立对拍（all / attention / mlp）
python benchmarks/crosscheck_w4a16_hf.py --group-size 128 --scope all \
    --output benchmarks/results/w4a16-hf-crosscheck-g128.json
python benchmarks/crosscheck_w4a16_hf.py --group-size 32 --scope all \
    --output benchmarks/results/w4a16-hf-crosscheck-g32.json
python benchmarks/crosscheck_w4a16_hf.py --group-size 128 --scope attention \
    --output benchmarks/results/w4a16-hf-crosscheck-attention-g128.json
python benchmarks/crosscheck_w4a16_hf.py --group-size 128 --scope mlp \
    --output benchmarks/results/w4a16-hf-crosscheck-mlp-g128.json
```

更大模型（AutoDL RTX 3080 10 GB，Python 3.10，torch 2.4.0+cu121）：

```bash
# 同步仓库（模型单独放在实例数据盘 /root/autodl-tmp/models/，
# 通过 HF_ENDPOINT=https://hf-mirror.com 下载）
rsync -az --exclude .git --exclude benchmarks/results \
    ./ root@<autodl-host>:/root/autodl-tmp/w4a16-work/

# 真机验证：两个模型 × 三个 group size
run_model() {
  model=$1; slug=$2
  for gs in 128 64 32; do
    ssh root@<autodl-host> "cd /root/autodl-tmp/w4a16-work && \
      PYTHONPATH=src /root/autodl-tmp/venv/bin/python benchmarks/validate_w4a16.py \
        --model /root/autodl-tmp/models/$model --group-size $gs \
        --output benchmarks/results/rtx3080/$slug-g$gs.json"
  done
}
run_model Llama-3.2-1B-Instruct llama32-1b
run_model Qwen3-1.7B qwen3-1.7b

# 融合 kernel（源码已按 torch 版本选择打包输入格式，2.4 可直接运行）
ssh root@<autodl-host> "cd /root/autodl-tmp/w4a16-work && \
  PYTHONPATH=src /root/autodl-tmp/venv/bin/python benchmarks/bench_w4a16_kernel.py \
    --model /root/autodl-tmp/models/Qwen3-1.7B \
    --output benchmarks/results/rtx3080/qwen3-1.7b-kernel-sweep.json"
for kernel in reference int4pack auto; do
  ssh root@<autodl-host> "cd /root/autodl-tmp/w4a16-work && \
    PYTHONPATH=src /root/autodl-tmp/venv/bin/python benchmarks/validate_w4a16.py \
      --model /root/autodl-tmp/models/Qwen3-1.7B --group-size 128 --kernel $kernel \
      --output benchmarks/results/rtx3080/qwen3-1.7b-kernel-$kernel.json"
done

# HF 独立对拍（g128）
ssh root@<autodl-host> "cd /root/autodl-tmp/w4a16-work && \
  PYTHONPATH=src /root/autodl-tmp/venv/bin/python benchmarks/crosscheck_w4a16_hf.py \
    --model /root/autodl-tmp/models/Qwen3-1.7B --group-size 128 \
    --output benchmarks/results/rtx3080/qwen3-1.7b-hf-crosscheck-g128.json"
```

说明：`validate_w4a16.py` 会按 `gpu_memory_utilization=0.7` 自动换算 KV
block 数；两个脚本都用 `README_EN.md` 前 1024 token 作为 PPL 文本，所以
实例上必须存在该文件（随仓库同步）。

原始数据：

- `benchmarks/results/w4a16-validation-qwen3-0.6b.json`（g128）
- `benchmarks/results/w4a16-validation-qwen3-0.6b-g64.json`
- `benchmarks/results/w4a16-validation-qwen3-0.6b-g32.json`
- `benchmarks/results/w4a16-hf-crosscheck-*.json`
- `benchmarks/results/rtx3080/llama32-1b-{g128,g64,g32}.json`
- `benchmarks/results/rtx3080/qwen3-1.7b-{g128,g64,g32}.json`
- `benchmarks/results/rtx3080/*-hf-crosscheck-g128.json`
- `benchmarks/results/w4a16-validation-qwen3-0.6b-kernel-{reference,int4pack,auto}.json`
- `benchmarks/results/w4a16-kernel-sweep-qwen3-0.6b.json`
- `benchmarks/results/rtx3080/qwen3-1.7b-kernel-{reference,int4pack,auto}.json`
- `benchmarks/results/rtx3080/qwen3-1.7b-kernel-sweep.json`
