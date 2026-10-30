# Benchmark 结果日志

本文件记录**人工可读**的测试结果，便于横向对比和写报告。原始逐请求数据仍应保存为
JSON（建议放在 `benchmarks/results/` 下），不要把本文件当成原始数据源。

每条记录至少应包含：

- 日期、引擎/版本、模型、模型路径、是否量化；
- GPU 型号/显存、CPU、内存；
- dtype、TP、`max-model-len`、`gpu-memory-utilization`、block size；
- 是否开启 prefix caching / chunked prefill / CUDA Graph；
- 请求数、并发数、输入/输出 token 的实际值；
- 总耗时、吞吐、Avg/P50/P90/P99 E2E、TTFT、TPOT/ITL；
- 测量方式（是否 stream、是否 warmup、重复次数）；
- 已知偏差和未记录项。

---

## B-001：vLLM + Qwen2.5-1.5B-Merged 初步基线

| 字段 | 值 |
|---|---|
| 日期 | 2026-10-28 |
| 引擎 | **vLLM**（`vllm.entrypoints.openai.api_server`），不是 micro-vLLM |
| 模型 | `Qwen2.5-1.5B-Merged` |
| 模型路径 | `/root/autodl-tmp/models/Qwen2.5-1.5B-Merged` |
| served-model-name | `qwen-merged` |
| GPU | **NVIDIA GeForce RTX 4080 SUPER** × 1，32,760 MiB，驱动 595.71.05，compute capability 8.9（SM89） |
| CPU | 16 vCPU，Intel Xeon Platinum 8352V @ 2.10 GHz |
| 内存 | 62 GB |
| 磁盘 | 系统盘 30 GB / 数据盘 50 GB |
| 软件 | PyTorch 2.4.0+cu121，CUDA 12.1，vLLM 0.6.1.post2，Triton 3.0.0 |
| dtype | 未显式指定（vLLM auto，预计 FP16） |
| TP | 未指定（默认 1） |
| 量化 | 未指定（预计无） |
| `max-model-len` | 4096 |
| `gpu-memory-utilization` | 0.65 |
| prefix caching | 未显式指定（取决于 vLLM 版本/默认值） |
| chunked prefill | 未显式指定（取决于 vLLM 版本/默认值） |
| CUDA Graph | 未显式指定（vLLM 默认行为） |
| 请求数 | 100 |
| 并发上限 | 10（`asyncio.Semaphore(10)`，不是 QPS） |
| 输入 | 旧版 `bench.py` 名义 `INPUT_LEN=128`；实际为 `"你好，" * 50`，真实 token 数未记录 |
| 输出 | `max_tokens=128`；实际 completion tokens 未记录 |
| 采样 | `temperature=0.0`（greedy） |
| 流式 | 否（`stream` 未开启） |
| warmup | 无 |
| 重复次数 | 1 |
| 总耗时 | 5.73 s |
| 吞吐 | 17.45 req/s |
| Avg E2E | 569.78 ms |
| P99 E2E | 3601.85 ms |
| TTFT | **未测量** |
| TPOT / ITL | **未测量** |
| 显存峰值 | **未记录** |
| KV cache 使用率 | **未记录** |

### 启动命令

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/Qwen2.5-1.5B-Merged \
    --served-model-name qwen-merged \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.65
```

### 压测脚本

旧版 `micro-vllm/bench.py`（B-001 使用的版本）：

- 100 个请求一次性创建任务，用信号量限制最多 10 个 in-flight；
- 请求 `POST /v1/chat/completions`，非流式；
- `max_tokens=128`，`temperature=0.0`；
- 只记录每个请求的 E2E，未读取响应中的 `usage`；
- 未记录逐请求延迟，只输出平均值和 P99。

该脚本已重写为流式版本，B-002 起使用新版。

### 结果解读

- `100 / 5.73s = 17.45 req/s`，与脚本输出一致。
- 按 Little's Law，`并发 10 / 平均 E2E 0.56978s ≈ 17.55 req/s`，与实测吞吐一致；
  说明这组测试基本是**闭环保和**（concurrency-limited），不是固定 QPS 的开环压测。
- 若每个请求确实生成 128 个 output tokens，则名义 output 吞吐约为
  `17.45 × 128 ≈ 2234 tokens/s`；但脚本没有读取 `usage.completion_tokens`，
  这个数字**不能当作实测值**。
- P99 为 3601.85 ms，远高于 Avg 569.78 ms，说明 100 个请求中存在明显长尾；
  可能是排队、prefill 抢占、个别请求生成更长，或客户端/网络抖动。当前没有逐请求
  数据，无法定位。
- 这套结果**不能**直接和 micro-vLLM 对比：启动的是 vLLM，不是本仓库的
  `micro_vllm.LLM`；模型也不是目标模型列表中的 Qwen3-0.6B/8B、Llama-3.2-1B 或
  Distill-Qwen-14B。

### 已确认的硬件/软件

- GPU：NVIDIA GeForce RTX 4080 SUPER，32,760 MiB，驱动 595.71.05，SM89。
- 软件：PyTorch 2.4.0+cu121，CUDA 12.1，vLLM 0.6.1.post2，Triton 3.0.0。
- 这不是 MIG 切片；SM89 支持 FlashAttention 2，但 4080 SUPER 没有 NVLink，
  多卡 TP 只能走 PCIe。

### 仍需补测的配置

1. **vLLM 实际生效的调度配置**：启动日志里通常会打印 dtype、KV cache 大小、
   `max_num_seqs`、`max_num_batched_tokens`、prefix caching、chunked prefill、
   CUDA Graph 捕获情况。请保存启动日志。
2. **实际 token 数**：新版 `bench.py` 会读取响应里的 `usage.prompt_tokens` 和
   `usage.completion_tokens`；B-001 旧版没有这些数据。
3. **逐请求延迟**：新版 `bench.py` 会保存每个请求的 `e2e_ms`、`ttft_ms`、
   `tpot_ms`、prompt_tokens、completion_tokens 和 ITL。
4. **TTFT / TPOT**：B-001 是非流式，拿不到真实 TTFT；B-002 起使用流式。
5. **warmup 和重复次数**：B-001 没有 warmup、只跑 1 次；B-002 默认 warmup 10、
   重复 3 次。
6. **`gpu-memory-utilization=0.65` 的原因**：32 GB 卡上 0.65 约保留 20.8 GB，
   会限制 KV cache。请确认这是为了避免 OOM，还是有意留显存；后续应同时测 0.65
   和 0.90，观察 KV 容量与吞吐变化。
7. **模型来源**：`Qwen2.5-1.5B-Merged` 是否是 LoRA merge 后的模型？如果是，
   tokenizer/vocab 和基座模型是否一致？这会影响与公开 Qwen2.5-1.5B 的可比性。
8. **arrival pattern**：当前是 100 个请求同时提交、信号量限 10，属于闭环保和；
   如果要测 QPS 曲线，需要固定 arrival rate（例如每 50 ms 发一个请求）。

### 下一步建议

- B-002 已完成，先用上面的“快速核对命令”解决 token 口径问题；
- 然后跑 B-003 并发扫描：1、5、10、20、40，得到吞吐-延迟曲线；
- 再和 micro-vLLM 的离线 `LLM.generate()` 或自建 server 对比，确保模型、dtype、
  token 数、采样参数一致。

### B-002 运行命令

```bash
python bench_v2.py \
  --num-requests 100 \
  --concurrency 10 \
  --warmup 10 \
  --repeats 3 \
  --max-tokens 128 \
  --label vllm-rtx4080s-qwen1.5b-merged \
  --notes "vLLM 0.6.1.post2; max-model-len=4096; gpu-memory-utilization=0.65; TP=1"
```

> 注意：这次运行的是 AutoDL 工作目录里的 `bench_v2.py`，不是仓库里的
> `bench.py`。两者输出字段一致，但 JSON 因此写到了脚本所在目录：
> `/root/autodl-tmp/self-llm/models/Qwen2.5/benchmarks/results/`。

### B-002 结果

| 字段 | 值 |
|---|---|
| 日期 | 2026-10-28 |
| JSON 路径 | `/root/autodl-tmp/self-llm/models/Qwen2.5/benchmarks/results/vllm-rtx4080s-qwen1.5b-merged-20261028-142819.json` |
| 引擎 | vLLM 0.6.1.post2 |
| 模型 | `qwen-merged`（Qwen2.5-1.5B-Merged） |
| GPU | RTX 4080 SUPER 32,760 MiB；启动时已用 20,026 MiB；驱动 595.71.05；SM89 |
| CPU / 内存（脚本可见） | 128 逻辑 CPU / 503.52 GiB（容器可见值；你之前说实例配额是 16 vCPU / 62 GB，二者不一致，见备注） |
| 软件 | Python 3.10.8，PyTorch 2.4.0，vLLM 0.6.1.post2，Triton 3.0.0，aiohttp 3.10.5，numpy 1.26.3 |
| 请求数 | 100，warmup 10，repeats 3 |
| 并发上限 | 10（`asyncio.Semaphore`，闭环保和，不是固定 QPS） |
| 输入 | 每个请求唯一前缀 + `"你好，" * 50`；平均 140 prompt tokens |
| 输出 | `max_tokens=128`，`temperature=0`，流式 |
| 成功/失败 | 300 / 0 |
| 总耗时 | 30.17 s（3 轮合计） |
| req/s | **9.94** |
| output tokens/s | **1272.79**（按实际 128 tokens/请求修正；原始 usage 口径为 1495.48） |
| total tokens/s | **2664.90**（修正；原始 usage 口径为 2887.55） |
| prompt / completion tokens | 42,000 / **38,400 实际**（usage 报 45,120；48 个请求把 `total_tokens=268` 写成了 `completion_tokens`） |
| TTFT mean / p50 / p90 / p99 | **81.05 / 71.56 / 125.49 / 164.68 ms** |
| E2E mean / p50 / p90 / p99 | **1003.65 / 972.94 / 1126.03 / 1166.83 ms** |
| TPOT mean / p50 / p90 / p99 | **6.66（mean 受污染）/ 6.99 / 7.94 / 8.29 ms** |
| ITL mean / p50 / p90 / p99 | 7.24 / 6.78 / 8.54 / 12.49 ms（38,100 个间隔；按 content chunk 计算，不是严格 per-token） |
| client wait（提交→真正发出） mean / p50 / p90 / p99 | **4603.38 / 4613.30 / 8571.49 / 9542.45 ms** |
| TTFT from submit mean / p50 / p90 / p99 | 4684.43 / 4698.80 / 8642.15 / 9622.35 ms |
| E2E from submit mean / p50 / p90 / p99 | 5607.03 / 5583.47 / 9557.57 / 10515.66 ms |
| 显存峰值 | 未记录 |

#### 分轮结果

| 轮次 | 总耗时 | req/s | output tok/s | completion tokens | TTFT mean | E2E mean | TPOT mean | ITL mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.06 s | 9.94 | 1522.99 | 15,320 | 85.23 ms | 1003.91 ms | 6.66 ms | 7.28 ms |
| 2 | 9.59 s | 10.43 | 1509.66 | 14,480 | 70.50 ms | 957.16 ms | 6.66 ms | 7.17 ms |
| 3 | 10.52 s | 9.51 | 1456.25 | 15,320 | 87.43 ms | 1049.89 ms | 6.67 ms | 7.27 ms |

### 结果解读

- **服务端很快**：TTFT 约 81 ms，E2E 约 1.0 s，TPOT p50 约 7.0 ms。
- **用户侧 TTFT 约 4.68 s，不是服务端慢**：100 个请求在 t=0 全部创建，
  但只有 10 个能拿到信号量；其余 90 个在客户端排队。`client_wait_ms`
  平均 4.6 s、p99 9.5 s，正好解释了 `TTFT from submit` 为什么远大于
  `ttft_ms`。写报告时应明确写“服务端 TTFT”还是“端到端用户可见 TTFT”。
- **吞吐不能和 B-001 直接比较**：B-001 是非流式、无 warmup、单次运行，
  且没有读取真实 token 数；B-002 是流式、warmup 10、3 次重复、唯一 prompt。
  9.94 req/s < 17.45 req/s 不代表性能回退，先不要下这个结论。
  按 B-001 的 E2E 570 ms、B-002 的 TPOT 6.7 ms 粗估，B-001 平均只生成了
  约 70 个 token，而 B-002 实际是 128 个；请求数吞吐下降主要来自输出更长。
  这个估算依赖 TPOT 不变，不能替代真实 token 数。
### B-002 的 `completion_tokens` 异常已定位

B-002 每轮 completion tokens 为 15,320 / 14,480 / 15,320，平均 150.4，
超过 `max_tokens=128`。B-003 的逐请求 JSON 显示这是**二值分布**：

- 大部分请求：`completion_tokens = 128`
- 少数请求：`completion_tokens = 268`
- 所有请求的 `content_chunk_count = 128`，`finish_reason = length`

而 `268 = 140 prompt tokens + 128 output tokens`，恰好等于该请求的
`total_tokens`。也就是说，这些请求的 `completion_tokens` 字段被写成了
`total_tokens`，不是模型真的生成了 268 个 token。受影响的请求比例随并发
上升：c=1 时 0%，c=5 时约 2%，c=10 时约 13%，c=20 时约 21%，c=40 时约 25%。
这是 **usage 统计/解析问题**，不是生成长度变化。

因此：

- B-002 的真实输出是 **每个请求 128 tokens**；
- 真实 output 吞吐 = `300 × 128 / 30.17 = 1272.79 tokens/s`，不是 1495.48；
- 报告 TPOT 时不要用 mean（被 268 的请求拉低），用 **p50 = 6.99 ms**；
- B-002 的 45,120 completion 中，多出的 6,720 = 48 × 140，说明约
  **48/300 个请求**受污染。

尚需确认问题出在 vLLM 的 streaming usage 还是 `bench_v2.py` 的解析。
最快检查方式是发一个流式请求，直接看最后的 usage chunk：

```bash
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-merged","messages":[{"role":"user","content":"你好"}],"max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
  | grep '"usage"'
```

如果服务端返回的 `completion_tokens` 正确、而 `bench_v2.py` 记录成 268，
就是客户端解析问题；反之则是 vLLM 0.6.1 的 streaming usage 问题。

## B-003：vLLM + Qwen2.5-1.5B-Merged 并发扫描

### 命令

```bash
for c in 1 5 10 20 40; do
  python bench_v2.py \
    --num-requests 100 \
    --concurrency $c \
    --warmup 10 \
    --repeats 3 \
    --max-tokens 128 \
    --ignore-eos \
    --label vllm-rtx4080s-qwen1.5b-merged-c$c \
    --notes "vLLM 0.6.1.post2; max-model-len=4096; gpu-memory-utilization=0.65; TP=1"
done
```

环境与 B-002 相同：RTX 4080 SUPER 32,760 MiB，vLLM 0.6.1.post2，
PyTorch 2.4.0，Triton 3.0.0；100 请求/轮，warmup 10，3 轮，唯一 prompt，
`max_tokens=128`，`--ignore-eos`。

结果 JSON：

- `benchmarks/results/vllm-rtx4080s-qwen1.5b-merged-c1-20261028-144234.json`
- `benchmarks/results/vllm-rtx4080s-qwen1.5b-merged-c5-20261028-144332.json`
- `benchmarks/results/vllm-rtx4080s-qwen1.5b-merged-c10-20261028-144406.json`
- `benchmarks/results/vllm-rtx4080s-qwen1.5b-merged-c20-20261028-144428.json`
- `benchmarks/results/vllm-rtx4080s-qwen1.5b-merged-c40-20261028-144446.json`

### 总表（300 个 measured 请求）

| 并发 | wall | req/s | output tok/s（服务端口径） | **output tok/s（按 128 修正）** | TTFT mean / p90 / p99 | E2E mean / p90 / p99 | TPOT p50 / p90 / p99（修正） | 268 异常请求 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 243.13 s | 1.23 | 157.94 | **157.94** | 27.49 / 38.60 / 49.17 ms | 809.80 / 878.50 / 949.40 ms | 6.01 / 6.62 / 7.17 ms | 0 / 300 |
| 5 | 53.20 s | 5.64 | 748.12 | **721.80** | 50.48 / 55.92 / 87.28 ms | 885.61 / 893.12 / 925.44 ms | 6.57 / 6.65 / 6.70 ms | 10 / 300 |
| 10 | 29.40 s | 10.20 | 1472.76 | **1306.09** | 70.95 / 105.99 / 127.97 ms | 978.10 / 1099.56 / 1156.90 ms | 6.96 / 7.83 / 8.26 ms | 35 / 300 |
| 20 | 17.65 s | 16.99 | 2682.76 | **2175.21** | 102.28 / 176.48 / 227.74 ms | 1170.60 / 1259.65 / 1287.46 ms | 8.27 / 9.23 / 9.60 ms | 64 / 300 |
| 40 | 13.16 s | 22.80 | 3704.98 | **2917.78** | 217.50 / 500.54 / 675.96 ms | 1538.01 / 1851.57 / 1997.66 ms | 10.03 / 12.61 / 13.04 ms | 74 / 300 |

说明：

- 服务端口径的 output tok/s 用 `usage.completion_tokens` 计算；由于 268 异常，
  它在 c=5 以上被高估，c=40 时高估约 27%。
- 修正列用 `300 × 128 / wall`，因为每个请求实际都是 128 个 content chunk，
  且 `finish_reason=length`。
- 修正 TPOT = `(e2e_ms - ttft_ms) / 127`，逐请求计算后再取分位数；这是本文中
  最可信的 TPOT。
- 所有请求 `content_chunk_count=128`，`finish_reason=length`；c=1 时
  `completion_tokens` 也恰好全是 128，说明 `--ignore-eos` 生效。

### 曲线解读

- **吞吐**：1.23 → 5.64 → 10.20 → 16.99 → 22.80 req/s；
  修正 output 吞吐 158 → 722 → 1306 → 2175 → 2918 tok/s。
  c=1→40 约 **18.5×**。
- **延迟**：TTFT mean 27 → 50 → 71 → 102 → 218 ms；E2E mean
  810 → 886 → 978 → 1171 → 1538 ms；TPOT p50 6.01 → 6.57 → 6.96 → 8.27 →
  10.03 ms。并发越高，单请求越慢，这是正常的吞吐-延迟权衡。
- **拐点**：
  - c=10→20：吞吐 +66.6%，E2E +19.7%；收益很大。
  - c=20→40：吞吐 +34.1%，E2E +31.4%，TPOT p50 +21.3%；开始边际递减。
  - 交互型 SLO（例如 E2E <1 s）适合 c=10；追求最大吞吐可以到 c=40。
- **client wait 不是服务端 TTFT**：c=1 时 100 个请求排队，`client_wait`
  mean 约 40 s，但服务端 TTFT 只有 27 ms；c=40 时 `client_wait` mean 1.37 s、
  p90 3.17 s，仍要在报告里和服务端 TTFT 分开写。
- **c=1 的 100 请求耗时约 80 s** 是并发 1 的预期结果，不是异常；它适合测
  单请求延迟，不适合测吞吐。
- **稳定性**：c=1/5/10/20 的三轮吞吐波动基本在 10% 以内；c=40 三轮分别是
  20.60 / 23.43 / 24.76 req/s，TTFT p90 达到 500 ms，波动明显变大。
  c=40 的结果应写成范围，或增加重复次数/固定 GPU 时钟后再定稿。

### 下一步

- 继续测 c=80、c=100，确认是否在 c=40 之后饱和；同时记录
  `nvidia-smi` 显存峰值和 vLLM 的 preemption 日志（当前只记录了启动时
  20,026 MiB）。
- 单独跑 `--no-unique-prompt`（相同 prompt）对比 prefix caching；
  这会改变 prefill 成本和异常请求比例，必须与唯一 prompt 分开记录。
- 修掉 `completion_tokens=total_tokens` 的解析/统计问题后，再决定是否
  需要重跑 B-002；B-003 的修正列已经可以直接用于报告。
- 后续换 Qwen3-0.6B/8B、Llama-3.2-1B、Distill-Qwen-14B 时，保持
  `--ignore-eos --max-tokens 128` 和相同并发点，才能横向比较。

### 通用备注

- 脚本看到的 128 CPU / 503.52 GiB 是容器内可见的宿主机资源；你之前说的
  16 vCPU / 62 GB 是实例配额。两者都记录，后续报告写“配额 16 vCPU / 62 GB，
  容器可见 128 逻辑核 / 503.52 GiB”，避免读者误以为测试用了 128 核。
- `bench_v2.py` 需要和仓库里的 `bench.py` 合并成同一份文件，否则以后很难
  判断某次结果是哪个版本产生的。仓库里的 `bench.py` 现在会在
  `completion_tokens > max_tokens` 时打印警告，并记录
  `content_chunk_count_sum` 与 `suspect_total_tokens_requests`，方便下次
  直接定位口径问题。
- B-003 的 JSON 已复制回 `benchmarks/results/`；B-002 的 JSON 仍在 AutoDL
  路径下，必要时再补拷。
