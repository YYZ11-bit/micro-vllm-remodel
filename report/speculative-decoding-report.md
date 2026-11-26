# 投机解码 v1 验证报告：n-gram + greedy + adaptive

- 报告日期：2026-11-25
- 代码范围：`speculative/`、`core/request.py`、`core/block.py`、
  `core/kv_cache_manager.py`、`core/scheduler_output.py`、`core/scheduler.py`、
  `executor/interface.py`、`sampling/sampler.py`、`worker/model_runner.py`、`llm.py`
- 验证脚本：
  - [benchmarks/validate_speculative.py](../benchmarks/validate_speculative.py)
  - [benchmarks/sweep_speculative.py](../benchmarks/sweep_speculative.py)
- 原始数据：
  - [1.5B K4 adaptive](../benchmarks/results/speculative-validation-qwen2.5-1.5b-adaptive.json)
  - [1.5B K4 adaptive rerun](../benchmarks/results/speculative-validation-qwen2.5-1.5b-adaptive-rerun.json)
  - [1.5B K6](../benchmarks/results/speculative-validation-qwen2.5-1.5b-k6.json)
  - [1.5B W4A16](../benchmarks/results/speculative-validation-qwen2.5-1.5b-w4a16.json)
  - [1.5B batch](../benchmarks/results/speculative-validation-qwen2.5-1.5b-batch.json)
  - [0.5B sweep summary](../benchmarks/results/sweep-qwen2.5-0.5b/summary.json)
- 本机全量回归：**771 passed, 14 skipped**

---

## 0. 结论速览

| 项目 | 结论 |
|---|---|
| 实现范围 | n-gram drafter + greedy acceptance + per-request adaptive；支持多请求混合 batch |
| 正确性 | 两个模型、5 个 prompt、3 次重复；FP16、W4A16、batch 模式全部逐 token 一致 |
| 重复型文本（K=4） | 1.5B 约 **2.5–2.9×**；W4A16 下约 **4.1–4.6×**；0.5B sweep 中 ngram3/K4 约 **3.3–4.5×** |
| 一般文本（K=4） | adaptive 后约 **0.99–1.08×**，基本与 baseline 持平，没有明显负收益 |
| K=6 | 重复型更强（1.5B 约 **3.6–6.1×**），但 general-explanation 只有 **0.72×**；不建议作为通用默认 |
| 多请求 batch | 5 个 prompt 混合 batch，总吞吐从约 **113 token/s 提升到 169 token/s（1.50×）**，token 全一致 |
| W4A16 + spec | token 全一致，接受率 **95.0%**，聚合加速 **3.97×**；peak 约 **13.1 GB** |
| 自适应 | 窗口 8 个 verified draft、阈值 0.30、初始 cooldown 8、backoff 最大 64；低接受率请求自动关闭，probe 失败会退避 |
| 推荐默认 | **ngram_size=3, max_draft_tokens=4**；K=6 仅用于重复性强的 workload，或先提高 adaptive 阈值 |
| 总体判断 | 投机解码 v1 的正确性、自适应、sweep、W4A16、batch 均已验证；一般文本只做到“不劣化”，没有真正加速 |

---

## 1. 数据流回顾

一次 spec step：

```text
running request (pending == 1, greedy)
        │
        ▼
Scheduler._maybe_propose_draft_tokens
  adaptive 检查：窗口 / 阈值 / cooldown / probe
  k_limit = min(max_draft_tokens, budget - 1, max_model_len - C - 1)
  NGramDrafter.propose(all_token_ids, k_limit) -> (d1..dK)
  request.draft_token_ids = [d1..dK]
        │
        ▼
ScheduledRequest:
  input_token_ids = (pending, d1..dK)
  num_scheduled_tokens = K + 1
  num_draft_tokens = K
        │
        ▼
ModelRunner:
  forward K+1 行
  每行采样一个预测 p0..pK
  RequestVerification(request_id, (p0..pK))
        │
        ▼
Scheduler.update:
  greedy_accept(drafts, predictions)
    ├─ 全部接受 -> accepted + bonus
    └─ 首个分歧 -> accepted 前缀 + replacement
  clear_draft_tokens()
  append accepted / replacement / bonus
  num_computed_tokens += 1 + accepted
  truncate_request(...)
  commit_step(...)
```

关键点：

- drafter 只负责“猜”，不参与模型 forward、采样、接受或 KV 管理；
- `RequestVerification` 是 K+1 个模型预测，不是 K+1 个已提交 token；
- 被拒绝的 draft 在 `commit_step` 前被清除并截断，因此不会污染 prefix cache；
- `replacement` 是模型在第一个分歧位置自己的 token，成为新的 pending token；
- adaptive 状态是 per-request 的，混合 batch 中一个请求关闭不会影响另一个。

---

## 2. 验证方法

### 2.1 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，24,564 MiB |
| 驱动 / CUDA | 580.105.08 / CUDA 13.0（运行时 cu121） |
| PyTorch | 2.4.0+cu121 |
| Python | 3.10.8 |
| transformers / tokenizers | 4.44.2 / 0.19.1（实例预装；Qwen2.5 tokenizer 可用） |
| dtype | FP16 |
| attention operator | `sdpa` |
| block size | 16 |
| `max_model_len` | 1024 |
| `max_new_tokens` | 64 |
| 采样 | greedy（`temperature=0`），`ignore_eos=True` |
| 重复次数 | 每个 prompt 3 次，取中位数 |

模型：

- `Qwen2.5-0.5B`，`gpu_memory_utilization=0.7`（sweep）
- `Qwen2.5-1.5B`，`gpu_memory_utilization=0.65`（FP16 / batch）
- `Qwen2.5-1.5B`，`gpu_memory_utilization=0.6`（W4A16 `int4pack`）

### 2.2 对比方式

每个 prompt（batch 模式为一次提交全部 prompt）分别运行：

1. baseline：`SpeculativeConfig(enabled=False)`；
2. speculative：`SpeculativeConfig(enabled=True)`，默认 `ngram_size=3`、`max_draft_tokens=4`。

要求两者的 `output_token_ids` 完全一致。统计量：

- `engine_steps`：`Engine.step()` 调用次数；
- `draft_tokens_verified`：送进 `greedy_accept` 的 draft 总数；
- `draft_tokens_accepted`：被接受的 draft 总数；
- `acceptance_rate = accepted / verified`；
- 端到端延迟：`LLM.generate()` 的墙钟时间，中位数。

统计通过 monkeypatch `NGramDrafter.propose` 和
`micro_vllm.core.scheduler.greedy_accept` 完成，不修改源码。

### 2.3 Adaptive 策略

- 每个 request 独立统计 `accepted / verified`；
- 每累计 **8 个 verified draft** 评估一次；
- 接受率 `< 0.30`：关闭该请求的 draft，初始 cooldown 8；
- cooldown 结束后只 probe 一次；
- probe 接受率仍 `< 0.30`：backoff 翻倍（最大 64）；
- probe 成功：恢复，backoff 回落；
- probe 返回空 proposal：同样 backoff，避免每步重试；
- finish / abort 时清理状态；preempt 时保留历史统计。

### 2.4 Prompt

重复型：

- `repeated-prose`：重复英文句子；
- `repeated-code`：重复 Python 函数；
- `repeated-numbers`：重复数字序列。

一般型：

- `general-explanation`：解释天空颜色；
- `general-story`：机器人学画画的短段落。

---

## 3. 结果

### 3.1 Qwen2.5-0.5B：ngram3 / K=4（当前 adaptive）

| Prompt | baseline | spec | 加速 | step (base → spec) | drafts | accepted | 接受率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeated-prose | 1.558 s | 0.345 s | **4.52×** | 192 → 42 | 156 | 156 | 100% |
| repeated-code | 1.312 s | 0.393 s | **3.34×** | 192 → 48 | 156 | 156 | 100% |
| repeated-numbers | 1.069 s | 0.256 s | **4.18×** | 192 → 42 | 156 | 156 | 100% |
| general-explanation | 1.060 s | 0.984 s | 1.08× | 192 → 180 | 24 | 12 | 50% |
| general-story | 1.040 s | 1.031 s | 1.01× | 192 → 192 | 12 | 0 | 0% |

聚合：接受率 **95.2%**，中位数加速 **2.72×**。聚合值被 3 个重复型 prompt 拉高。

### 3.2 Qwen2.5-1.5B：ngram3 / K=4（当前 adaptive）

| Prompt | baseline | spec | 加速 | step (base → spec) | drafts | accepted | 接受率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeated-prose | 1.256 s | 0.425 s | **2.95×** | 192 → 42 | 156 | 156 | 100% |
| repeated-code | 1.228 s | 0.486 s | **2.53×** | 192 → 48 | 156 | 156 | 100% |
| repeated-numbers | 1.209 s | 0.422 s | **2.86×** | 192 → 42 | 156 | 156 | 100% |
| general-explanation | 1.214 s | 1.199 s | 1.01× | 192 → 183 | 12 | 9 | 75% |
| general-story | 1.220 s | 1.232 s | 0.99× | 192 → 189 | 24 | 3 | 12% |

聚合：接受率 **95.2%**，中位数加速 **2.51×**。同一配置复跑一次聚合为 **2.84×**，
说明短序列上 GPU 时钟/批次波动会带来约 10% 级别的差异；token 和接受率保持一致。

### 3.3 Qwen2.5-1.5B：ngram3 / K=6

| Prompt | baseline | spec | 加速 | step (base → spec) | drafts | accepted | 接受率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeated-prose | 1.814 s | 0.295 s | **6.15×** | 192 → 30 | 162 | 162 | 100% |
| repeated-code | 1.251 s | 0.351 s | **3.56×** | 192 → 36 | 162 | 162 | 100% |
| repeated-numbers | 1.256 s | 0.294 s | **4.27×** | 192 → 30 | 162 | 162 | 100% |
| general-explanation | 1.247 s | 1.726 s | **0.72×** | 192 → 183 | 18 | 9 | 50% |
| general-story | 1.231 s | 1.241 s | 0.99× | 192 → 189 | 36 | 3 | 8% |

K=6 在重复型上明显更强，但 general-explanation 接受率 50% 高于 adaptive 的 0.30
阈值，所以没有被关闭，K+1 行开销导致 **0.72×**。**K=6 不是通用默认值**；
若要使用，应把 adaptive 阈值提高到约 0.5–0.6，或先做 workload 检测。

### 3.4 W4A16 + spec（1.5B，ngram3/K=4，int4pack）

| Prompt | baseline | spec | 加速 | step (base → spec) | 接受率 |
|---|---:|---:|---:|---:|---:|
| repeated-prose | 1.539 s | 0.338 s | **4.55×** | 192 → 42 | 100% |
| repeated-code | 1.602 s | 0.393 s | **4.08×** | 192 → 48 | 100% |
| repeated-numbers | 1.559 s | 0.362 s | **4.31×** | 192 → 42 | 100% |
| general-explanation | 1.551 s | 1.466 s | 1.06× | 192 → 183 | 38% |
| general-story | 1.559 s | 1.319 s | 1.18× | 192 → 162 | 75% |

聚合：接受率 **95.0%**，中位数加速 **3.97×**，全部 token 一致。
显存 peak 约 **13.1 GB**（W4A16，`gpu_memory_utilization=0.6`）。

### 3.5 多请求混合 batch（1.5B，ngram3/K=4）

一次 `generate` 提交 5 个 prompt（3 重复型 + 2 一般型），共 320 个输出 token：

| 指标 | baseline | spec |
|---|---:|---:|
| 总延迟（中位数） | 2.838 s | 1.895 s |
| 总吞吐 | 约 113 token/s | 约 169 token/s |
| 加速 | — | **1.50×** |
| acceptance rate | — | 95.2% |
| engine steps | 192 | 189 |

5 个 prompt 的 token 输出全部一致。`engine steps` 只减少 3 次，是因为 batch 的
最慢请求（一般型 prompt）仍决定步数；spec 请求更早完成后，后续 batch 行数变少，
所以总延迟仍下降。

### 3.6 Qwen2.5-0.5B sweep

所有 9 个组合 token 全部一致；baseline 均为 960 steps（5 prompt × 3 repeat × 64 token）。

| ngram_size | K | 聚合加速 | 接受率 | spec steps |
|---:|---:|---:|---:|---:|
| 2 | 2 | 1.71× | 86.7% | 573 |
| 2 | 4 | 2.90× | 77.1% | 513 |
| 2 | 6 | 3.82× | 75.5% | 474 |
| 3 | 2 | 2.51× | 96.3% | 579 |
| 3 | 4 | **2.72×** | 95.2% | 504 |
| 3 | 6 | **5.18×** | 92.2% | 468 |
| 4 | 2 | 1.70× | 98.4% | 582 |
| 4 | 4 | 2.53× | 96.3% | 510 |
| 4 | 6 | 3.37× | 94.3% | 477 |

聚合值仍是被重复型 prompt 拉高的。K=6 的 aggregate 最好，但 1.5B 上已验证
其对 general-explanation 有 0.72× 的负收益；因此推荐默认 **K=4**，K=6 留给
重复性明确的 workload。

### 3.7 显存

| 配置 | baseline peak | spec peak |
|---|---:|---:|
| 1.5B FP16，0.65 | 15,008 MiB | 15,008 MiB |
| 1.5B W4A16，0.60 | 13,116 MiB | 13,171 MiB |

Spec 不额外分配模型权重；显存差异主要来自 dtype、量化和
`gpu_memory_utilization`。不同 utilization 下的绝对显存不可直接横向比较。

---

## 4. 解读

### 4.1 重复型文本为什么收益大

n-gram drafter 能找到与后缀匹配的历史片段，并把后续 token 作为 draft。重复型
prompt 的后续文本本身就是重复的，所以 draft 很容易全部命中：

- K=4 时，64 个 decode token 只需要约 14–16 次 target forward（3 次重复合计
  42–48 steps）；
- K=6 时可以降到 10–12 次；
- 1.5B W4A16 下 repeated prompt 达到约 4.1–4.6×。

这是 n-gram 的理想场景，不代表通用对话或代码补全。

### 4.2 一般文本与 adaptive

general prompt 的 n-gram 命中少，接受率通常 0–75%。adaptive 的目标不是加速，
而是避免负收益：

- K=4 下 general prompt 约 0.99–1.08×，基本持平；
- K=6 下 general-explanation 接受率 50%，高于 0.30 阈值，因此没有关闭，出现 0.72×；
- 如果要把 K=6 作为默认，需要提高阈值或增加 workload 检测。

adaptive 目前是写死参数：窗口 8、阈值 0.30、cooldown 8、backoff 最大 64。
这些值对 K=4 有效，但不是对所有 K 都最优。

### 4.3 多请求 batch

batch 中每个请求独立判断 adaptive。总步数由最慢的一般型请求决定，因此
`engine steps` 下降不多；但重复型请求更早完成后，后续 batch 行数减少，
总延迟从 2.838 s 降到 1.895 s（1.50×）。这说明 spec 在多请求下仍有价值，
但吞吐收益取决于 batch 内 workload 的组成。

### 4.4 W4A16 组合

W4A16 + spec 的 token 全部一致，接受率 95%，聚合 3.97×。量化后模型输出变化
会影响 n-gram 命中：general-story 的接受率从 FP16 的 12% 提升到 75%，因此
W4A16 下 general-story 也有 1.18×。这说明 acceptance rate 必须针对量化模型
重新测量，不能直接复用 FP16 的结论。

### 4.5 正确性

两个模型、5 个 prompt、3 次重复；FP16、W4A16、batch 模式全部逐 token 一致。
说明：

- K+1 行的行选择与预测位置正确；
- accepted / replacement / bonus 语义正确；
- 拒绝后的 computed 计数和 KV 截断正确；
- prefix cache 没有被未提交 token 污染；
- 多请求混合 batch 中状态、KV 表和 adaptive 状态互不串扰。

---

## 5. 限制与下一步

### 5.1 当前限制

- 只验证了 n-gram drafter；没有 draft model、EAGLE、Medusa；
- 只支持 greedy；没有 stochastic speculative sampling；
- 只在 Qwen2.5-0.5B / 1.5B、FP16 / W4A16、单卡 SDPA 上验证；
- 没有长上下文、serving 并发、真实对话/代码数据集；
- adaptive 阈值和窗口写死，没有暴露到 `SpeculativeConfig`；
- batch 模式只测 5 个固定 prompt，不是生产流量；
- 没有 CUDA Graph / TP。

### 5.2 建议的下一步

1. 把 adaptive 参数暴露到 `SpeculativeConfig`，并按 K 调整默认阈值；
2. 用真实对话/代码补全数据测 acceptance rate，而不是固定重复型 prompt；
3. 如果继续用 K=6，先做 workload 检测或提高 adaptive 阈值到 0.5–0.6；
4. 在冻结 v1 后开始 CUDA Graph decode 的 design spike，再考虑 TP。

---

## 6. 复现

```bash
cd /root/autodl-tmp/micro-vllm

# 1.5B FP16，K=4
PYTHONPATH=src python benchmarks/validate_speculative.py \
    --model /root/autodl-tmp/models/Qwen2.5-1.5B \
    --gpu-memory-utilization 0.65 \
    --max-new-tokens 64 --repeat 3 \
    --output benchmarks/results/speculative-validation-qwen2.5-1.5b-adaptive.json

# 1.5B W4A16 + spec
PYTHONPATH=src python benchmarks/validate_speculative.py \
    --model /root/autodl-tmp/models/Qwen2.5-1.5B \
    --gpu-memory-utilization 0.6 \
    --max-new-tokens 64 --repeat 3 \
    --quantization w4a16 --group-size 128 --kernel int4pack \
    --output benchmarks/results/speculative-validation-qwen2.5-1.5b-w4a16.json

# 1.5B 多请求混合 batch
PYTHONPATH=src python benchmarks/validate_speculative.py \
    --model /root/autodl-tmp/models/Qwen2.5-1.5B \
    --gpu-memory-utilization 0.65 \
    --max-new-tokens 64 --repeat 3 --batch \
    --output benchmarks/results/speculative-validation-qwen2.5-1.5b-batch.json

# 0.5B sweep
PYTHONPATH=src python benchmarks/sweep_speculative.py \
    --model /root/autodl-tmp/models/Qwen2.5-0.5B \
    --output-dir benchmarks/results/sweep-qwen2.5-0.5b
```

脚本只读取本地 checkpoint，不下载权重；每个 prompt 会先 warmup，再重复 3 次取
中位数。`--max-new-tokens`、`--ngram-size`、`--max-draft-tokens`、`--dtype`、
`--quantization`、`--batch` 均可调整。
