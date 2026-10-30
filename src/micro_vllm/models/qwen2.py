"""Qwen2 dense architecture and parameter loading."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from micro_vllm.attention.attention import Attention, AttentionMetadata
from micro_vllm.attention.spec import FullAttentionSpec
from micro_vllm.config import ModelConfig
from micro_vllm.layers.embedding import LMHead, TokenEmbedding
from micro_vllm.layers.linear import Linear
from micro_vllm.layers.mlp import SwiGLUMLP
from micro_vllm.layers.normalization import RMSNorm
from micro_vllm.layers.rotary_embedding import RotaryEmbedding


class Qwen2Attention(nn.Module):
    """Qwen2 projections with Q/K/V biases, and paged attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        attention: Attention,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        query_size = num_attention_heads * head_dim
        key_value_size = num_key_value_heads * head_dim
        self.q_proj = Linear(
            hidden_size,
            query_size,
            bias=True,
        )
        self.k_proj = Linear(
            hidden_size,
            key_value_size,
            bias=True,
        )
        self.v_proj = Linear(
            hidden_size,
            key_value_size,
            bias=True,
        )
        self.o_proj = Linear(
            query_size,
            hidden_size,
            bias=False,
        )
        self.rotary_embedding = RotaryEmbedding(head_dim, base=rope_theta)
        self.attention = attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        query = self.q_proj(hidden_states).view(
            -1,
            self.num_attention_heads,
            self.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            -1,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            -1,
            self.num_key_value_heads,
            self.head_dim,
        )
        query, key = self.rotary_embedding(positions, query, key)
        attention_output = self.attention(
            query,
            key,
            value,
            layer_kv_cache,
            metadata,
        )
        return self.o_proj(attention_output.flatten(start_dim=1))


class Qwen2DecoderLayer(nn.Module):
    """Pre-norm Qwen2 decoder layer over one physical KV cache."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention: Attention,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rope_theta=rope_theta,
            attention=attention,
        )
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            positions,
            layer_kv_cache,
            metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen2Model(nn.Module):
    """Single-device dense Qwen2 model for flattened inference execution."""

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        values = config.values
        vocab_size = config.require_positive_int("vocab_size")
        hidden_size = config.require_positive_int("hidden_size")
        intermediate_size = config.require_positive_int("intermediate_size")
        num_hidden_layers = config.require_positive_int("num_hidden_layers")
        num_attention_heads = config.require_positive_int(
            "num_attention_heads"
        )
        num_key_value_heads = config.positive_int(
            "num_key_value_heads",
            num_attention_heads,
        )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        head_dim = config.positive_int(
            "head_dim",
            hidden_size // num_attention_heads,
        )
        if hidden_size != head_dim * num_attention_heads:
            raise ValueError(
                "hidden_size must be equal to head_dim * num_attention_heads"
            )
        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for rotary embeddings"
            )
        use_sliding_window = values.get("use_sliding_window", False)
        if use_sliding_window:
            raise NotImplementedError("Sliding window attention is not yet supported.")
        rope_scaling = values.get("rope_scaling", None)
        if rope_scaling is not None:
            raise NotImplementedError("RoPE scaling is not yet supported.")
        rms_norm_eps = values.get("rms_norm_eps", 1e-6)
        rope_theta = values.get("rope_theta", 1_000_000.0)
        self._tie_word_embeddings = values.get("tie_word_embeddings", False)

        attention = Attention(FullAttentionSpec(), config.attention_operator)
        self.config = config
        self.layer_attentions = (attention,) * num_hidden_layers
        self.embed_tokens = TokenEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                attention=attention,
            )
            for _ in range(num_hidden_layers)
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = LMHead(
            hidden_size,
            vocab_size,
            tied_embedding=(
                self.embed_tokens if self._tie_word_embeddings else None
            ),
        )
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layer_kv_caches: tuple[torch.Tensor, ...],
        layer_attention_metadata: tuple[AttentionMetadata, ...],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer, layer_kv_cache, metadata in zip(
            self.layers,
            layer_kv_caches,
            layer_attention_metadata,
            strict=True,
        ):
            hidden_states = layer(
                hidden_states,
                positions,
                layer_kv_cache,
                metadata,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits only for hidden-state rows selected by the caller."""

        return self.lm_head(hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Copy checkpoint tensors directly into model parameters."""
        parameters = dict(self.named_parameters())
        loaded_name = set()
        with torch.no_grad():
            for checkpoint_name, checkpoint_tensor in weights:
                parameter_name = _qwen2_parameter_name(
                    checkpoint_name,
                    tied_embeddings=self._tie_word_embeddings,
                )
                if parameter_name not in parameters:
                    raise KeyError(
                        f"Checkpoint parameter {checkpoint_name} "
                        f"maps to {parameter_name}, which is not a model parameter"
                    )
                tensor_shape = parameters[parameter_name].shape
                if tensor_shape != checkpoint_tensor.shape:
                    raise ValueError(
                        f"Checkpoint parameter {checkpoint_name} "
                        f"maps to {parameter_name}, which has shape "
                        f"{tensor_shape}, but checkpoint tensor has shape "
                        f"{checkpoint_tensor.shape}"
                    )
                parameters[parameter_name].copy_(checkpoint_tensor)
                loaded_name.add(parameter_name)
            missing_parameters = set(parameters.keys()) - loaded_name
            if missing_parameters:
                missing_parameters = sorted(missing_parameters)
                raise ValueError(
                    f"Checkpoint is missing parameters: {missing_parameters}"
                )


def _qwen2_parameter_name(
    checkpoint_name: str,
    *,
    tied_embeddings: bool,
) -> str:
    if checkpoint_name.startswith("model."):
        parameter_name = checkpoint_name.removeprefix("model.")
    else:
        parameter_name = checkpoint_name
    if tied_embeddings and parameter_name == "lm_head.weight":
        return "embed_tokens.weight"
    return parameter_name
