"""Tests for model configuration and read-only loaded metadata."""

import pytest

from micro_vllm.config import ModelConfig, QuantizationConfig


def test_model_config_accepts_metadata_after_source_configuration() -> None:
    config = ModelConfig(model="org/model", revision="main")

    assert config.metadata is None
    assert config.architectures == ()

    config.set_metadata(
        {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
        }
    )

    assert config.model == "org/model"
    assert config.revision == "main"
    assert config.architectures == ("LlamaForCausalLM",)
    assert config.require_positive_int("hidden_size") == 64


def test_model_config_preserves_generic_metadata() -> None:
    config = ModelConfig(
        model="unused",
        metadata={
            "architectures": ["Unknown", "LlamaForCausalLM"],
            "hidden_size": 64,
            "custom_metadata": {"name": "value"},
        },
    )

    assert config.architectures == ("Unknown", "LlamaForCausalLM")
    assert config.require_positive_int("hidden_size") == 64
    assert config.values.get("custom_metadata") == {"name": "value"}


def test_model_config_copies_and_exposes_read_only_values() -> None:
    values = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
    }
    config = ModelConfig(model="unused", metadata=values)
    values["hidden_size"] = 128

    assert config.require_positive_int("hidden_size") == 64
    with pytest.raises(TypeError):
        config.values["hidden_size"] = 128


@pytest.mark.parametrize(
    "architectures",
    [None, [], "LlamaForCausalLM", [""], ["   "], [True]],
)
def test_model_config_rejects_invalid_architectures(
    architectures: object,
) -> None:
    with pytest.raises(ValueError, match="architectures?|architecture"):
        ModelConfig(model="unused", metadata={"architectures": architectures})


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1"])
def test_model_config_requires_strict_positive_integers(value: object) -> None:
    config = ModelConfig(
        model="unused",
        metadata={
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": value,
        },
    )

    with pytest.raises(ValueError, match="hidden_size must be a positive integer"):
        config.require_positive_int("hidden_size")


def test_model_config_typed_readers_apply_defaults() -> None:
    config = ModelConfig(
        model="unused",
        metadata={"architectures": ["LlamaForCausalLM"]},
    )

    assert config.positive_int("num_key_value_heads", 4) == 4
    assert config.values.get("rope_theta", 10_000.0) == 10_000.0


def test_model_config_defaults_to_unquantized() -> None:
    config = ModelConfig(model="unused")

    assert config.quantization.method == "none"
    assert config.quantization.group_size == 128


def test_model_config_accepts_w4a16_quantization() -> None:
    quantization = QuantizationConfig(method="w4a16", group_size=64)
    config = ModelConfig(model="unused", quantization=quantization)

    assert config.quantization is quantization


@pytest.mark.parametrize("method", ["", "w8a8", "awq", "W4A16", None])
def test_quantization_config_rejects_unknown_methods(method: object) -> None:
    with pytest.raises(ValueError, match="unsupported quantization method"):
        QuantizationConfig(method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("group_size", [0, -8, 1, 3, 127])
def test_quantization_config_rejects_invalid_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="positive even integer"):
        QuantizationConfig(method="w4a16", group_size=group_size)


def test_quantization_config_defaults_to_reference_kernel() -> None:
    assert QuantizationConfig().kernel == "reference"


@pytest.mark.parametrize("kernel", ["reference", "int4pack", "auto"])
def test_quantization_config_accepts_supported_kernels(kernel: str) -> None:
    config = QuantizationConfig(method="w4a16", group_size=32, kernel=kernel)

    assert config.kernel == kernel


@pytest.mark.parametrize("kernel", ["", "int4", "triton", None])
def test_quantization_config_rejects_unknown_kernels(kernel: object) -> None:
    with pytest.raises(ValueError, match="unsupported quantization kernel"):
        QuantizationConfig(method="w4a16", kernel=kernel)  # type: ignore[arg-type]


@pytest.mark.parametrize("kernel", ["int4pack", "auto"])
def test_quantization_config_requires_reference_kernel_when_disabled(
    kernel: str,
) -> None:
    with pytest.raises(ValueError, match="kernel must be 'reference'"):
        QuantizationConfig(method="none", kernel=kernel)


@pytest.mark.parametrize("kernel", ["int4pack", "auto"])
@pytest.mark.parametrize("group_size", [8, 16])
def test_quantization_config_rejects_unsupported_kernel_group_size(
    kernel: str,
    group_size: int,
) -> None:
    with pytest.raises(ValueError, match="group_size must be one of"):
        QuantizationConfig(
            method="w4a16",
            group_size=group_size,
            kernel=kernel,
        )
