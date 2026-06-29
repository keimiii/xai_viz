from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from app.precompute.generate_feature_cache import (
    discover_checkpoints,
    discover_checkpoints_by_strategy,
    extract_finetuned_hidden_states_for_cache,
)
from ssl_attention.evaluation.fine_tuning import FineTunableModel


def test_discover_checkpoints_by_strategy_returns_strategy_specific_paths(tmp_path: Path) -> None:
    (tmp_path / "dinov2_lora_finetuned.pt").touch()
    (tmp_path / "dinov2_full_finetuned.pt").touch()
    (tmp_path / "clip_finetuned.pt").touch()

    result = discover_checkpoints_by_strategy(
        tmp_path,
        model_names=["dinov2", "clip"],
        strategies=["lora", "full", "linear_probe"],
    )

    assert result["dinov2"]["lora"] == tmp_path / "dinov2_lora_finetuned.pt"
    assert result["dinov2"]["full"] == tmp_path / "dinov2_full_finetuned.pt"
    assert result["clip"]["full"] == tmp_path / "clip_finetuned.pt"
    assert "linear_probe" not in result["clip"]


def test_discover_checkpoints_prefers_lora_then_full_then_linear_probe(tmp_path: Path) -> None:
    (tmp_path / "dinov2_full_finetuned.pt").touch()
    (tmp_path / "dinov2_lora_finetuned.pt").touch()
    (tmp_path / "mae_finetuned.pt").touch()

    result = discover_checkpoints(tmp_path, model_names=["dinov2", "mae"])

    assert result["dinov2"] == tmp_path / "dinov2_lora_finetuned.pt"
    assert result["mae"] == tmp_path / "mae_finetuned.pt"


class _CachePathMAEBackbone(nn.Module):
    """MAE-like backbone for feature-cache hidden-state extraction tests."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(patch_size=16)
        self.call_index = 0
        self.noise_inputs: list[torch.Tensor | None] = []
        self.training_states: list[bool] = []

    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        self.call_index += 1
        self.training_states.append(self.training)
        self.noise_inputs.append(None if noise is None else noise.detach().clone())

        batch_size = pixel_values.shape[0]
        seq_length = (pixel_values.shape[-2] // 16) * (pixel_values.shape[-1] // 16)
        patch_values = (
            noise
            if noise is not None
            else torch.full((batch_size, seq_length), float(self.call_index), dtype=torch.float32)
        )

        cls_token = torch.zeros((batch_size, 1, 1), dtype=torch.float32)
        patch_tokens = patch_values.unsqueeze(-1)
        last_hidden_state = torch.cat((cls_token, patch_tokens), dim=1)
        hidden_states = (torch.zeros_like(last_hidden_state), last_hidden_state)

        return SimpleNamespace(
            last_hidden_state=last_hidden_state,
            attentions=(),
            hidden_states=hidden_states if output_hidden_states else None,
        )


def test_extract_finetuned_hidden_states_for_cache_uses_deterministic_mae_analysis() -> None:
    backbone = _CachePathMAEBackbone()
    backbone.train()

    model = FineTunableModel.__new__(FineTunableModel)
    nn.Module.__init__(model)
    model.backbone = backbone
    model.model_name = "mae"
    model._config = cast(Any, SimpleNamespace(num_registers=0))

    pixel_values = torch.zeros((2, 3, 224, 224))
    hidden_states_a = extract_finetuned_hidden_states_for_cache(model, pixel_values)
    hidden_states_b = extract_finetuned_hidden_states_for_cache(model, pixel_values)

    expected = torch.arange(196, dtype=torch.float32).expand(2, -1)
    first_noise = backbone.noise_inputs[0]
    second_noise = backbone.noise_inputs[1]
    assert first_noise is not None
    assert second_noise is not None
    assert torch.equal(first_noise, expected)
    assert torch.equal(second_noise, expected)
    assert backbone.training_states == [False, False]
    assert backbone.training is True
    assert torch.equal(hidden_states_a[0], hidden_states_b[0])
