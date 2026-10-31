"""Tests for the sparse top-k mixture-of-experts layer."""

import pytest
import torch
from torch.nn import functional as F
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from micro_vllm.layers.moe import SparseMoEBlock, TopKRouter


def _dense_reference(
    moe: SparseMoEBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Compute every expert and combine them with the router weights."""
    hidden_size = hidden_states.shape[-1]
    flat_hidden_states = hidden_states.reshape(-1, hidden_size)
    router_logits = moe.router.gate(flat_hidden_states)
    routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(
        routing_probs,
        moe.router.top_k,
        dim=-1,
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    expert_outputs = torch.stack(
        [expert(flat_hidden_states) for expert in moe.experts],
        dim=1,
    )
    selected_outputs = torch.gather(
        expert_outputs,
        1,
        selected_experts.unsqueeze(-1).expand(-1, -1, hidden_size),
    )
    combined = (selected_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)
    return combined.to(hidden_states.dtype).reshape(hidden_states.shape)


def _copy_mixtral_weights(
    moe: SparseMoEBlock,
    reference: MixtralSparseMoeBlock,
) -> None:
    with torch.no_grad():
        moe.router.gate.weight.copy_(reference.gate.weight)
        for ours, theirs in zip(moe.experts, reference.experts):
            ours.gate_proj.weight.copy_(theirs.w1.weight)
            ours.up_proj.weight.copy_(theirs.w3.weight)
            ours.down_proj.weight.copy_(theirs.w2.weight)


def test_router_selects_top_k_and_renormalizes_weights() -> None:
    router = TopKRouter(hidden_size=6, num_experts=5, top_k=2)
    hidden_states = torch.linspace(-1.0, 1.0, steps=4 * 6).reshape(4, 6)

    routing_weights, selected_experts, router_logits = router(hidden_states)

    probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    expected_weights = torch.gather(probabilities, 1, selected_experts)
    expected_weights = expected_weights / expected_weights.sum(
        dim=-1,
        keepdim=True,
    )
    assert selected_experts.shape == torch.Size((4, 2))
    assert torch.equal(
        selected_experts,
        torch.topk(probabilities, 2, dim=-1).indices,
    )
    torch.testing.assert_close(routing_weights, expected_weights.to(torch.float32))
    torch.testing.assert_close(
        routing_weights.sum(dim=-1),
        torch.ones(4),
    )


@pytest.mark.parametrize(
    ("num_experts", "top_k"),
    [(0, 0), (4, 0), (4, 5), (-1, 1)],
)
def test_router_rejects_invalid_expert_configuration(
    num_experts: int,
    top_k: int,
) -> None:
    with pytest.raises(ValueError):
        TopKRouter(hidden_size=6, num_experts=num_experts, top_k=top_k)


@pytest.mark.parametrize("top_k", [1, 2, 8])
def test_moe_matches_dense_all_expert_reference(top_k: int) -> None:
    torch.manual_seed(0)
    moe = SparseMoEBlock(
        hidden_size=8,
        intermediate_size=16,
        num_experts=8,
        top_k=top_k,
    )
    hidden_states = torch.linspace(-1.0, 1.0, steps=4 * 3 * 8).reshape(4, 3, 8)

    output = moe(hidden_states)
    expected = _dense_reference(moe, hidden_states)

    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)


def test_moe_matches_transformers_mixtral_block() -> None:
    config = MixtralConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_jitter_noise=0.0,
        hidden_act="silu",
    )
    reference = MixtralSparseMoeBlock(config).eval()
    moe = SparseMoEBlock(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
    ).eval()
    _copy_mixtral_weights(moe, reference)
    hidden_states = torch.linspace(-1.0, 1.0, steps=2 * 3 * 8).reshape(2, 3, 8)

    output = moe(hidden_states)
    expected, expected_router_logits = reference(hidden_states)
    _, _, router_logits = moe.router(hidden_states.reshape(-1, 8))

    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(router_logits, expected_router_logits, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_moe_preserves_input_shape_and_dtype(dtype: torch.dtype) -> None:
    moe = SparseMoEBlock(8, 16, 4, 2).to(dtype=dtype)
    hidden_states = torch.randn(2, 3, 8, dtype=dtype)

    output = moe(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype


def test_moe_is_invariant_to_flattened_batch_shape() -> None:
    torch.manual_seed(0)
    moe = SparseMoEBlock(8, 16, 4, 2)
    hidden_states = torch.randn(2, 3, 8)

    batched = moe(hidden_states)
    flattened = moe(hidden_states.reshape(6, 8))

    torch.testing.assert_close(
        batched.reshape(6, 8),
        flattened,
        atol=0,
        rtol=0,
    )


def test_moe_accepts_empty_and_unbatched_inputs() -> None:
    moe = SparseMoEBlock(8, 16, 4, 2)

    assert moe(torch.zeros(0, 8)).shape == torch.Size((0, 8))
    assert moe(torch.zeros(8)).shape == torch.Size((8,))


def test_moe_skips_experts_that_receive_no_tokens() -> None:
    torch.manual_seed(0)
    moe = SparseMoEBlock(8, 16, 8, 1).eval()
    hidden_states = torch.randn(1, 8)
    calls = [0] * 8
    for expert_idx, expert in enumerate(moe.experts):
        expert.register_forward_hook(
            lambda *_, expert_idx=expert_idx: calls.__setitem__(
                expert_idx,
                calls[expert_idx] + 1,
            ),
        )

    moe(hidden_states)
    _, selected_experts, _ = moe.router(hidden_states)

    assert calls[selected_experts.item()] == 1
    assert sum(calls) == 1


def test_unselected_experts_do_not_contribute_to_a_token() -> None:
    torch.manual_seed(0)
    moe = SparseMoEBlock(8, 16, 8, 2).eval()
    hidden_states = torch.randn(3, 8)
    with torch.no_grad():
        before = moe(hidden_states)
    _, selected_experts, _ = moe.router(hidden_states)
    selected_for_first_token = set(selected_experts[0].tolist())
    unselected_expert = next(
        index for index in range(8) if index not in selected_for_first_token
    )

    with torch.no_grad():
        moe.experts[unselected_expert].down_proj.weight.add_(0.5)
        after = moe(hidden_states)

    torch.testing.assert_close(after[0], before[0], atol=0, rtol=0)


def test_moe_gradients_reach_only_selected_experts() -> None:
    torch.manual_seed(0)
    moe = SparseMoEBlock(8, 16, 8, 2).eval()
    hidden_states = torch.randn(3, 8, requires_grad=True)
    _, selected_experts, _ = moe.router(hidden_states)
    selected = set(selected_experts.flatten().tolist())

    moe(hidden_states).sum().backward()

    assert hidden_states.grad is not None
    assert moe.router.gate.weight.grad is not None
    for expert_idx, expert in enumerate(moe.experts):
        grad = expert.gate_proj.weight.grad
        if expert_idx in selected:
            assert grad is not None
            assert torch.count_nonzero(grad) > 0
        else:
            assert grad is None


def test_moe_state_dict_names_are_architecture_independent() -> None:
    moe = SparseMoEBlock(8, 16, 3, 2)

    expected_keys = {"router.gate.weight"}
    for expert_idx in range(3):
        expected_keys.add(f"experts.{expert_idx}.gate_proj.weight")
        expected_keys.add(f"experts.{expert_idx}.up_proj.weight")
        expected_keys.add(f"experts.{expert_idx}.down_proj.weight")

    assert set(moe.state_dict()) == expected_keys
