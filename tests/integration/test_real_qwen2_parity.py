import gc
import os
from micro_vllm.attention.attention import map_cache_slots
import pytest
import torch
from micro_vllm.config import LLMConfig, ModelConfig
from micro_vllm.model_loader import load_model
from transformers import AutoTokenizer, Qwen2ForCausalLM

@pytest.mark.model
def test_real_qwen2_parity() -> None:
    path = os.getenv("MICRO_VLLM_QWEN2_MODEL_PATH")
    if path is None:
        pytest.skip("MICRO_VLLM_QWEN2_MODEL_PATH is not set")
    tokenizer = AutoTokenizer.from_pretrained(path)
    input_ids = torch.tensor(
        tokenizer("The capital of France is", add_special_tokens=False)["input_ids"],
        dtype=torch.int64
    )
    positions = torch.arange(input_ids.numel(),dtype=torch.int64)
    model_config = ModelConfig(model=path, device='cpu', dtype=torch.float32)
    model= load_model(LLMConfig(model_config=model_config))

    block_size = 2
    num_logical_blocks = (input_ids.numel() + block_size - 1) // block_size                   # 物理 block 数，必须 >= block_table.max()+1
    physical_blocks = torch.arange(num_logical_blocks - 1, -1, -1)   # 逻辑 block 0 -> 物理 2，逻辑 1 -> 物理 0
    num_blocks = num_logical_blocks
    block_table = physical_blocks.unsqueeze(0)
    slot_mapping = map_cache_slots(positions, physical_blocks, block_size)

    attn = model.layer_attentions[0]
    metadata = attn.build_metadata(
        query_positions=positions,
        query_start_locations=torch.tensor([0, input_ids.numel()], dtype=torch.int64),
        context_lengths=torch.tensor([input_ids.numel()]),
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=block_size,
    )
    # 注意 KV cache 的真实布局是 (2, num_blocks, block_size, kv_heads, head_dim)
    # 第一维是 K/V，不是 block！
    kv_heads = model.config.require_positive_int("num_key_value_heads")
    num_layers = model.config.require_positive_int("num_hidden_layers")
    dtype = model.parameters().__next__().dtype
    head_dim = model.config.require_positive_int("hidden_size") // model.config.require_positive_int("num_attention_heads")
    layer_caches = tuple(
        torch.zeros((2, num_blocks, block_size, kv_heads, head_dim), dtype=dtype)
        for _ in range(num_layers)
    )
    with torch.no_grad():
        hidden = model(input_ids, positions, layer_caches, (metadata,) * num_layers)
        logits = model.compute_logits(hidden)

    del model
    gc.collect()
    reference = Qwen2ForCausalLM.from_pretrained(
        path,
        dtype=torch.float32,          # 4.57 用 dtype=，torch_dtype= 是弃用别名
        attn_implementation="eager",  # 避免 SDPA 的 kernel 差异
    ).eval()

    with torch.no_grad():
        reference_hidden = reference.model(
            input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
        ).last_hidden_state.squeeze(0)
        reference_logits = reference.lm_head(reference_hidden)

    torch.testing.assert_close(hidden, reference_hidden, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits, reference_logits, atol=1e-5, rtol=1e-5)
    assert logits.argmax(-1).equal(reference_logits.argmax(-1))



