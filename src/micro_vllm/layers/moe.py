"""Sparse top-k mixture-of-experts feed-forward layers."""

import torch
from torch import nn
from torch.nn import functional as F

from micro_vllm.layers.linear import Linear
from micro_vllm.layers.mlp import SwiGLUMLP


class TopKRouter(nn.Module):
    """Bias-free router that selects the top-k experts for each token.

    Routing probabilities are normalized in float32. The selected weights are
    renormalized to sum to one, so each token's output is a convex combination
    of its selected experts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(
                f"num_experts must be at least 1, got {num_experts}"
            )
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                "top_k must satisfy 1 <= top_k <= num_experts, "
                f"got top_k={top_k}, num_experts={num_experts}"
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Linear(hidden_size, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route flat ``(tokens, hidden_size)`` hidden states.

        Returns:
            routing_weights: ``(tokens, top_k)`` renormalized weights in the
                input dtype.
            selected_experts: ``(tokens, top_k)`` expert indices, descending
                by routing probability.
            router_logits: ``(tokens, num_experts)`` raw gate outputs.
        """
        router_logits = self.gate(hidden_states)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_probs, selected_experts = torch.topk(
            routing_probs,
            self.top_k,
            dim=-1,
        )
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return (
            topk_probs.to(hidden_states.dtype),
            selected_experts,
            router_logits,
        )


class SparseMoEBlock(nn.Module):
    """Sparse mixture of SwiGLU experts with top-k routing.

    Each token is sent to exactly ``top_k`` experts. Experts that receive no
    tokens are skipped entirely, so the compute cost scales with the number of
    selected expert-token pairs rather than with ``num_experts``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            SwiGLUMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply top-k expert dispatch to trailing ``(..., hidden_size)`` dims."""
        leading_shape = hidden_states.shape[:-1]
        hidden_size = hidden_states.shape[-1]
        flat_hidden_states = hidden_states.reshape(-1, hidden_size)
        routing_weights, selected_experts, _ = self.router(flat_hidden_states)

        output = torch.zeros_like(flat_hidden_states)
        for expert_idx, expert in enumerate(self.experts):
            token_indices, slot_indices = torch.where(
                selected_experts == expert_idx
            )
            if token_indices.numel() == 0:
                continue
            expert_output = expert(
                flat_hidden_states.index_select(0, token_indices)
            )
            weights = routing_weights[token_indices, slot_indices].unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output * weights)
        return output.reshape(*leading_shape, hidden_size)
