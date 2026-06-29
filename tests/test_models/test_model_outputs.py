"""Tests for ModelOutput validation and model configuration layouts.

Priority 1: These tests verify:
- ModelOutput.__post_init__() catches batch size mismatches (real validation logic)
- Expected sequence layouts for all ViT models (compact parametrized tests)

Each model has a different sequence layout:
- DINOv2: [CLS] + [4 registers] + [256 patches] = 261 tokens
- DINOv3: [CLS] + [4 registers] + [196 patches] = 201 tokens
- MAE:    [CLS] + [196 patches] = 197 tokens
- CLIP:   [CLS] + [196 patches] = 197 tokens
- SigLIP/SigLIP2: [196 patches] (no CLS, uses pooler) = 196 tokens
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.conftest import MODEL_CONFIGS


class TestModelOutputValidation:
    """Test ModelOutput dataclass validation."""

    def test_batch_size_mismatch_raises(self):
        """Verify ModelOutput raises on batch size mismatch."""
        from ssl_attention.models.protocols import ModelOutput

        cls_token = torch.randn(2, 768)
        patch_tokens = torch.randn(3, 196, 768)  # Different batch size!
        attention_weights = [torch.randn(2, 12, 197, 197)]

        with pytest.raises(ValueError, match="Batch size mismatch"):
            ModelOutput(
                cls_token=cls_token,
                patch_tokens=patch_tokens,
                attention_weights=attention_weights,
            )

    def test_attention_batch_mismatch_raises(self):
        """Verify ModelOutput raises on attention batch mismatch."""
        from ssl_attention.models.protocols import ModelOutput

        cls_token = torch.randn(2, 768)
        patch_tokens = torch.randn(2, 196, 768)
        attention_weights = [torch.randn(3, 12, 197, 197)]  # Different batch!

        with pytest.raises(ValueError, match="Batch size mismatch"):
            ModelOutput(
                cls_token=cls_token,
                patch_tokens=patch_tokens,
                attention_weights=attention_weights,
            )

    def test_valid_output_properties(self):
        """Verify ModelOutput computes properties correctly."""
        from ssl_attention.models.protocols import ModelOutput

        cls_token = torch.randn(2, 768)
        patch_tokens = torch.randn(2, 196, 768)
        attention_weights = [torch.randn(2, 12, 197, 197) for _ in range(12)]

        output = ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            attention_weights=attention_weights,
        )

        assert output.batch_size == 2
        assert output.embed_dim == 768
        assert output.num_patches == 196
        assert output.num_layers == 12


@pytest.mark.parametrize(
    "model_name,expected_seq_len,expected_patches,expected_registers",
    [
        ("dinov2", 261, 256, 4),
        ("dinov3", 201, 196, 4),
        ("mae", 197, 196, 0),
        ("clip", 197, 196, 0),
        ("siglip", 196, 196, 0),
        ("siglip2", 196, 196, 0),
    ],
)
class TestParameterizedModelConfigs:
    """Parametrized tests for all model configurations."""

    def test_sequence_layout(
        self,
        make_mock_output,
        model_name: str,
        expected_seq_len: int,
        expected_patches: int,
        expected_registers: int,
    ):
        """Verify sequence length = 1 + registers + patches (or just patches for SigLIP)."""
        config = MODEL_CONFIGS[model_name]
        output = make_mock_output(seq_len=config["seq_len"], has_pooler=config["has_pooler"])

        actual_seq_len = output.last_hidden_state.shape[1]
        assert actual_seq_len == expected_seq_len

        # Verify math: seq_len = (1 if has CLS else 0) + registers + patches
        has_cls = model_name not in ("siglip", "siglip2")
        computed_seq_len = (1 if has_cls else 0) + expected_registers + expected_patches
        assert actual_seq_len == computed_seq_len

    def test_patch_count_is_square(
        self,
        make_mock_output,
        model_name: str,
        expected_seq_len: int,
        expected_patches: int,
        expected_registers: int,
    ):
        """Verify patch count is a perfect square (for valid 2D grid)."""
        sqrt = math.isqrt(expected_patches)
        assert sqrt * sqrt == expected_patches, f"{model_name} patches must be square"

    def test_patches_per_side(
        self,
        make_mock_output,
        model_name: str,
        expected_seq_len: int,
        expected_patches: int,
        expected_registers: int,
    ):
        """Verify patches_per_side matches expected values."""
        patches_per_side = math.isqrt(expected_patches)

        # DINOv2: 16x16, others: 14x14
        if model_name == "dinov2":
            assert patches_per_side == 16
        else:
            assert patches_per_side == 14
