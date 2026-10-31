# 独立 MoE 模块报告：稀疏 Top-K 混合专家层

- 报告日期：2026-10-30
- 代码基线：`04fd789` + 本次未提交工作区
- 目标：在 micro-vLLM 中新增一个**独立**的稀疏 Mixture-of-Experts（MoE）前向模块及其单元测试。
  该模块不接入任何现有模型、调度器或权重加载路径，只保证自身接口与测试通过。
- 运行环境：Python 3.10.22 / PyTorch 2.8.0+cu128 / transformers 4.57.6 / CPU

---

## 0. 结论速览

| 项目 | 结论 |
|---|---|
| 新增源码 | 1 个：`src/micro_vllm/layers/moe.py` |
| 修改源码 | 1 个：`src/micro_vllm/layers/__init__.py`（仅新增导出） |
| 新增测试 | 1 个：`tests/layers/test_moe.py`，共 18 个用例 |
| 新增文档 | 1 个：本报告 |
| MoE 核心语义 | 每个 token 由 router 选出 `top_k` 个专家；只对选中的专家做前向，按归一化路由权重加权求和 |
| 数值验证 | 与 HuggingFace `MixtralSparseMoeBlock` 在相同权重下前向一致（`atol=1e-6, rtol=1e-6`） |
| 测试结果 | `tests/layers/test_moe.py`：**18 passed**；全量回归：**572 passed, 10 skipped** |
| 影响范围 | 无。未修改模型、引擎、调度、加载器等任何既有逻辑 |

---

## 1. 改动文件清单

### 1.1 新增文件

| 文件 | 内容 |
|---|---|
| `src/micro_vllm/layers/moe.py` | `TopKRouter` 路由层 + `SparseMoEBlock` 稀疏专家前向层 |
| `tests/layers/test_moe.py` | 18 个 CPU 单元测试，覆盖路由、稀疏分发、数值对齐、梯度与接口约束 |
| `report/moe-layer-report.md` | 本报告 |

### 1.2 修改文件

| 文件 | 改动 |
|---|---|
| `src/micro_vllm/layers/__init__.py` | 从 `moe` 导入并在 `__all__` 中新增 `SparseMoEBlock`、`TopKRouter`，与 `SwiGLUMLP` 等既有层保持一致的导出方式 |

除此之外没有改动任何文件。该模块不修改 `models/`、`worker/`、`model_loader/`、`config` 等任何系统组件，也不参与模型注册或权重加载。

---

## 2. 模块设计

### 2.1 公开接口

```python
from micro_vllm.layers.moe import SparseMoEBlock, TopKRouter

router = TopKRouter(hidden_size=8, num_experts=8, top_k=2)
weights, indices, logits = router(flat_hidden_states)

moe = SparseMoEBlock(
    hidden_size=8,
    intermediate_size=16,
    num_experts=8,
    top_k=2,
)
output = moe(hidden_states)   # 任意前导维度 (..., hidden_size) -> 同形状
```

- `TopKRouter.forward(hidden_states)` 输入 `(tokens, hidden_size)`，返回三元组：
  - `routing_weights`：`(tokens, top_k)`，已按选中专家重新归一化，dtype 与输入一致；
  - `selected_experts`：`(tokens, top_k)`，按概率降序排列的 int64 专家下标；
  - `router_logits`：`(tokens, num_experts)`，原始 gate 输出，便于外部计算负载均衡损失。
- `SparseMoEBlock.forward(hidden_states)` 接受任意前导维度（如 `(batch, seq, hidden)` 或
  `(tokens, hidden)`，也接受一维单 token），输出形状与输入完全一致。
- 专家复用 `layers/mlp.py` 中已有的 `SwiGLUMLP`，每个专家的参数名为
  `gate_proj.weight` / `up_proj.weight` / `down_proj.weight`，没有新增重复的 SwiGLU 实现。

### 2.2 路由算法

对展平后的 token 矩阵 `X ∈ R^{T×H}`：

```
logits = X · W_gate^T                        # W_gate 无 bias
p      = softmax(logits, dim=-1)  (float32)
w, idx = topk(p, k)                          # 每行按概率降序
w      = w / sum(w, dim=-1, keepdim=True)    # 选中专家内重新归一化
```

与 HuggingFace Mixtral 的实现保持一致：softmax 在 float32 下计算，top-k 后再归一化，
权重最后转回输入 dtype。这样每个 token 的输出是其选中专家输出的凸组合，权重和为 1。

### 2.3 稀疏分发

```python
output = zeros_like(flat_hidden_states)
for expert_idx, expert in enumerate(self.experts):
    token_indices, slot_indices = torch.where(selected_experts == expert_idx)
    if token_indices.numel() == 0:
        continue                       # 没有 token 选中的专家完全不执行
    expert_output = expert(flat_hidden_states.index_select(0, token_indices))
    weights = routing_weights[token_indices, slot_indices].unsqueeze(-1)
    output.index_add_(0, token_indices, expert_output * weights)
```

- 计算量只与「被选中的 (token, expert) 对」数量相关；未命中的专家被显式跳过。
- 累加顺序固定为专家下标升序，结果可复现；同一 token 在同一个专家中最多出现一次
  （top-k 下标互不相同），因此不存在同槽位重复累加。
- 没有实现训练期的 router jitter noise 与负载均衡辅助损失：本模块面向推理路径，
  这两项属于训练语义，加入会引入与推理无关的可变行为。

### 2.4 输入约束与错误处理

`TopKRouter` 在构造时校验 `num_experts >= 1` 且 `1 <= top_k <= num_experts`，
不满足时抛出带具体数值的 `ValueError`。这是静态类型无法表达的领域不变量，因此保留运行时校验；
其余内部张量形状交由矩阵乘法和 top-k 自身报错。

---

## 3. 测试清单

文件：`tests/layers/test_moe.py`（纯 CPU，可在默认依赖下运行）。

| # | 用例 | 验证的不变量 |
|---|---|---|
| 1 | `test_router_selects_top_k_and_renormalizes_weights` | 选中的专家就是概率 top-k；权重等于对应概率重新归一化且和为 1 |
| 2–5 | `test_router_rejects_invalid_expert_configuration` | `num_experts=0`、`top_k=0`、`top_k>num_experts`、负数配置均被拒绝 |
| 6–8 | `test_moe_matches_dense_all_expert_reference`（`top_k ∈ {1,2,8}`） | 稀疏分发结果与「所有专家都算一遍再按权重取选中项」的稠密参考实现一致 |
| 9 | `test_moe_matches_transformers_mixtral_block` | 与 HF `MixtralSparseMoeBlock` 在相同权重、相同输入下输出与 router logits 对齐 |
| 10–12 | `test_moe_preserves_input_shape_and_dtype`（fp32/fp16/bf16） | 输出形状不变；权重从 float32 路由概率转回输入 dtype |
| 13 | `test_moe_is_invariant_to_flattened_batch_shape` | `(B,T,H)` 与 `(B*T,H)` 输入逐元素等价 |
| 14 | `test_moe_accepts_empty_and_unbatched_inputs` | 0 token 与一维输入不会崩溃 |
| 15 | `test_moe_skips_experts_that_receive_no_tokens` | 用 forward hook 计数，证实未被选中的专家确实没有执行 |
| 16 | `test_unselected_experts_do_not_contribute_to_a_token` | 扰动某 token 未选中的专家权重，该 token 输出逐位不变 |
| 17 | `test_moe_gradients_reach_only_selected_experts` | 反向传播到达 router 与选中的专家；未选中专家 `.grad is None` |
| 18 | `test_moe_state_dict_names_are_architecture_independent` | 参数名固定为 `router.gate.weight` + `experts.{i}.{gate,up,down}_proj.weight` |

### 数值对齐说明

- 与稠密参考、与 HF Mixtral 的比较均使用 `atol=1e-6, rtol=1e-6`。差异来源仅是
  `index_add_` 在不同实现中的累加顺序（浮点加法不满足结合律），不是算法差异。
- router logits 的比较使用 `atol=0, rtol=0`：两者都是同一次 `Linear` 计算。
- 与 HF 的权重映射：HF `w1 → gate_proj`、`w3 → up_proj`、`w2 → down_proj`，
  `gate.weight → router.gate.weight`；测试中显式拷贝，证明两者计算图等价。

---

## 4. 验证结果

```text
$ python -m pytest tests/layers/test_moe.py -v
...
18 passed in 0.32s

$ python -m pytest -q
572 passed, 10 skipped in 21.14s
```

- 10 个 skip 均为既有约定：需要 `MICRO_VLLM_TEST_MODEL=1` 的外部模型测试，与本次改动无关。
- 全量回归通过说明新增导出没有破坏 `layers` 包或公共 API。

---

## 5. 设计取舍与范围说明

1. **复用 `SwiGLUMLP` 而不是新写专家 MLP**：MoE 专家与稠密 FFN 数学形式相同，
   复用可避免两份等价实现漂移；同时保持参数名与仓库既有层一致。
2. **路由与分发分离**：`TopKRouter` 只负责「选谁、权重多少」，`SparseMoEBlock` 只负责
   「只算被选中的专家并累加」，符合仓库「policy 与 mechanism 分离」的规则，也便于单独测试路由。
3. **不做容量因子 / token drop**：采用与 HF 相同的 full-capacity 语义（不丢 token），
   行为确定、无隐藏超参数。
4. **不接入系统**：按任务要求，本模块不修改模型注册表、不参与权重加载、不感知 KV cache
   或调度；因此不存在 checkpoint 命名适配、专家并行切分、通信等问题。这些属于后续接入工作。
5. **未验证项**：仅在 CPU 上验证；未在 CUDA 上做数值/性能测试，也未实现专家并行或
   grouped GEMM 等优化路径。
