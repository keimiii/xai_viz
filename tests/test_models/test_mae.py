"""Regression tests for MAE deterministic analysis inference."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from ssl_attention.models.base import (
    BaseVisionModel,
    build_mae_analysis_noise,
    forward_mae_for_analysis,
)
from ssl_attention.models.protocols import ModelOutput


class _TrackingMAEBackbone(nn.Module):
    """Minimal MAE-like backbone that exposes the analysis noise it receives."""

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
        output_attentions: bool = True,
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

        attention = torch.zeros((batch_size, 1, seq_length + 1, seq_length + 1), dtype=torch.float32)
        attention[:, 0, 0, 1:] = patch_values

        hidden_states = None
        if output_hidden_states:
            hidden_states = (torch.zeros_like(last_hidden_state), last_hidden_state)

        return SimpleNamespace(
            last_hidden_state=last_hidden_state,
            attentions=(attention,) if output_attentions else (),
            hidden_states=hidden_states,
        )


class _FrozenMAEWrapper(BaseVisionModel):
    """Minimal frozen MAE wrapper that uses BaseVisionModel.forward()."""

    model_name = "mae"
    model_id = "stub/mae"
    patch_size = 16
    embed_dim = 1
    num_layers = 1
    num_heads = 1
    num_registers = 0

    def _load_processor(self) -> Any:  # pragma: no cover - preprocess is not exercised here
        return None

    def _load_model(self) -> nn.Module:
        return _TrackingMAEBackbone()

    def _extract_output(
        self,
        model_output: Any,
        include_hidden_states: bool = False,
    ) -> ModelOutput:
        hidden_states = None
        if include_hidden_states and model_output.hidden_states is not None:
            hidden_states = list(model_output.hidden_states[1:])

        return ModelOutput(
            cls_token=model_output.last_hidden_state[:, 0, :],
            patch_tokens=model_output.last_hidden_state[:, 1:, :],
            attention_weights=list(model_output.attentions),
            hidden_states=hidden_states,
        )


def test_build_mae_analysis_noise_uses_patch_sequence_length_only() -> None:
    backbone = _TrackingMAEBackbone()
    pixel_values = torch.zeros((2, 3, 224, 224))

    noise = build_mae_analysis_noise(backbone, pixel_values)

    expected = torch.arange(196, dtype=torch.float32).expand(2, -1)
    assert noise.shape == (2, 196)
    assert torch.equal(noise, expected)


def test_forward_mae_for_analysis_passes_canonical_noise() -> None:
    backbone = _TrackingMAEBackbone()
    pixel_values = torch.zeros((2, 3, 224, 224))

    output = forward_mae_for_analysis(
        backbone,
        pixel_values,
        output_attentions=True,
        output_hidden_states=True,
    )

    expected = torch.arange(196, dtype=torch.float32).expand(2, -1)
    first_noise = backbone.noise_inputs[0]
    assert first_noise is not None
    assert torch.equal(first_noise, expected)
    assert output.hidden_states is not None


def test_frozen_mae_forward_is_repeatable_for_attention_and_hidden_states() -> None:
    model = _FrozenMAEWrapper(device=torch.device("cpu"), dtype=torch.float32)
    pixel_values = torch.zeros((2, 3, 224, 224))

    output_a = model.forward(pixel_values, output_hidden_states=True)
    output_b = model.forward(pixel_values, output_hidden_states=True)

    tracking_model = cast(_TrackingMAEBackbone, model.model)
    expected = torch.arange(196, dtype=torch.float32).expand(2, -1)
    first_noise = tracking_model.noise_inputs[0]
    second_noise = tracking_model.noise_inputs[1]
    assert first_noise is not None
    assert second_noise is not None
    assert torch.equal(first_noise, expected)
    assert torch.equal(second_noise, expected)
    assert tracking_model.training_states == [False, False]
    assert output_a.patch_tokens is not None
    assert output_b.patch_tokens is not None
    assert torch.equal(output_a.patch_tokens, output_b.patch_tokens)
    assert torch.equal(output_a.attention_weights[0], output_b.attention_weights[0])
    assert output_a.hidden_states is not None
    assert output_b.hidden_states is not None
    assert torch.equal(output_a.hidden_states[0], output_b.hidden_states[0])
