"""Tests for mean attention extraction (SigLIP's primary attention method).

Code under test: ssl_attention.attention.cls_attention.extract_mean_attention

This function is used by SigLIP (which has no CLS token) to produce per-patch
saliency scores by fusing attention heads and averaging across rows.
"""

from __future__ import annotations

import torch

from ssl_attention.attention.cls_attention import (
    HeadFusion,
    extract_mean_attention,
)


class TestExtractMeanAttention:
    """Test extract_mean_attention for SigLIP-style models."""

    def test_output_shape(self, make_attention_weights):
        """Input (B, H, 196, 196) → output (B, 196)."""
        attention_weights = make_attention_weights(
            batch_size=1, num_heads=12, seq_len=196, num_layers=12
        )

        result = extract_mean_attention(attention_weights, layer=-1)

        assert result.shape == (1, 196)

    def test_layer_selection(self, make_attention_weights):
        """layer=-1 vs layer=0 select different layers."""
        attention_weights = make_attention_weights(
            batch_size=1, num_heads=12, seq_len=196, num_layers=12
        )

        result_first = extract_mean_attention(attention_weights, layer=0)
        result_last = extract_mean_attention(attention_weights, layer=-1)

        assert not torch.allclose(result_first, result_last)

    def test_fusion_strategy(self, make_attention_weights):
        """MEAN vs MAX fusion produce different outputs."""
        attention_weights = make_attention_weights(
            batch_size=1, num_heads=12, seq_len=196, num_layers=12
        )

        result_mean = extract_mean_attention(
            attention_weights, layer=-1, fusion=HeadFusion.MEAN
        )
        result_max = extract_mean_attention(
            attention_weights, layer=-1, fusion=HeadFusion.MAX
        )

        assert not torch.allclose(result_mean, result_max)

    def test_head_indices_selection(self, make_attention_weights):
        """Selecting a subset of heads changes output."""
        attention_weights = make_attention_weights(
            batch_size=1, num_heads=12, seq_len=196, num_layers=12
        )

        result_all = extract_mean_attention(attention_weights, layer=-1)
        result_subset = extract_mean_attention(
            attention_weights, layer=-1, head_indices=[0, 5]
        )

        assert not torch.allclose(result_all, result_subset)

    def test_batch_dimension_preserved(self, make_attention_weights):
        """Batch size 1 and 4 both work correctly."""
        for batch_size in (1, 4):
            attention_weights = make_attention_weights(
                batch_size=batch_size, seq_len=196
            )
            result = extract_mean_attention(attention_weights, layer=-1)
            assert result.shape[0] == batch_size

    def test_values_from_softmax_input(self):
        """With softmax-normalized attention (rows sum to 1), output mean ≈ 1/N."""
        seq_len = 196
        # Create uniform softmax attention: each row sums to 1
        uniform = torch.ones(1, 12, seq_len, seq_len) / seq_len
        attention_weights = [uniform]

        result = extract_mean_attention(attention_weights, layer=0)

        # Each row sums to 1, so mean across rows = 1/seq_len for each column
        expected_value = 1.0 / seq_len
        torch.testing.assert_close(
            result,
            torch.full((1, seq_len), expected_value),
        )

    def test_uniform_attention_gives_uniform_output(self):
        """If all attention values equal, all patches get equal saliency."""
        seq_len = 196
        constant = torch.full((1, 12, seq_len, seq_len), 0.5)
        attention_weights = [constant]

        result = extract_mean_attention(attention_weights, layer=0)

        # All patches should have equal saliency
        assert torch.allclose(result, result[:, :1].expand_as(result))
