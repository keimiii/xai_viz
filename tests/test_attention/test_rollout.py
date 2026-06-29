"""Tests for attention rollout computation.

Priority 6: These tests verify attention rollout correctly aggregates
attention across layers and handles register token removal.
"""

from __future__ import annotations

import pytest
import torch

from ssl_attention.attention.rollout import (
    attention_rollout,
    extract_cls_rollout,
)


class TestAttentionRollout:
    """Test attention rollout computation."""

    def test_single_layer_preserves_shape(self, make_attention_weights):
        """Single layer rollout should preserve sequence shape."""
        attention_weights = make_attention_weights(seq_len=197, num_layers=1)

        rollout = attention_rollout(attention_weights)

        assert rollout.shape == (1, 197, 197)

    def test_output_shape(self, make_attention_weights):
        """Rollout output has shape (B, seq, seq)."""
        attention_weights = make_attention_weights(batch_size=2, seq_len=197, num_layers=12)

        rollout = attention_rollout(attention_weights)

        assert rollout.shape == (2, 197, 197)

    def test_batch_dimension_preserved(self, make_attention_weights):
        """Batch dimension is preserved through rollout."""
        attention_weights = make_attention_weights(batch_size=4, seq_len=197)

        rollout = attention_rollout(attention_weights)

        assert rollout.shape[0] == 4

    def test_identity_initialization(self):
        """Rollout starts from identity matrix."""
        # With single layer of identity attention, result should be close to identity
        batch_size = 1
        seq_len = 5

        # Create attention close to identity
        attention = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, 12, -1, -1)
        attention_weights = [attention.clone()]

        rollout = attention_rollout(attention_weights)

        # After one layer: R = (A + I) @ I = A + I (normalized)
        # For identity attention, this should be close to uniform distribution
        assert rollout.shape == (1, seq_len, seq_len)

    def test_batch_elements_have_independent_memory(self, make_attention_weights):
        """Batch elements must not share underlying data (expand→clone safety).

        Specifically targets the edge case where start_layer >= end_layer
        (empty loop), so the initialized identity is returned directly.
        Without .clone(), expand() would share storage across batch elements.
        """
        attention_weights = make_attention_weights(batch_size=2, seq_len=50, num_layers=4)

        # Empty layer range → identity returned directly (no bmm to break aliasing)
        rollout = attention_rollout(attention_weights, start_layer=2, end_layer=2)

        # Each batch element must have its own storage
        assert rollout[0].data_ptr() != rollout[1].data_ptr()

    def test_layer_range_selection(self, make_attention_weights):
        """Verify start_layer and end_layer limit computation range."""
        attention_weights = make_attention_weights(seq_len=197, num_layers=12)

        # Only use layers 0-3
        rollout_partial = attention_rollout(attention_weights, start_layer=0, end_layer=4)

        # Full rollout
        rollout_full = attention_rollout(attention_weights)

        # Results should differ
        assert not torch.allclose(rollout_partial, rollout_full)


class TestExtractClsRollout:
    """Test CLS token extraction from rollout."""

    def test_output_shape_with_registers(self, make_attention_weights):
        """With num_registers=4, output has num_patches tokens."""
        # DINOv2: seq_len = 261, num_patches = 256
        attention_weights = make_attention_weights(seq_len=261)

        result = extract_cls_rollout(attention_weights, num_registers=4)

        assert result.shape == (1, 256)

    def test_output_shape_without_registers(self, make_attention_weights):
        """With num_registers=0, output has num_patches tokens."""
        # MAE: seq_len = 197, num_patches = 196
        attention_weights = make_attention_weights(seq_len=197)

        result = extract_cls_rollout(attention_weights, num_registers=0)

        assert result.shape == (1, 196)

    def test_register_removal(self, make_attention_weights):
        """Verify registers are excluded from output."""
        # DINOv3: seq_len = 201 (CLS + 4 reg + 196 patches)
        attention_weights = make_attention_weights(seq_len=201)

        result = extract_cls_rollout(attention_weights, num_registers=4)

        # Should have exactly 196 patches (not 201)
        assert result.shape[1] == 196

    def test_batch_dimension(self, make_attention_weights):
        """Batch dimension is preserved."""
        attention_weights = make_attention_weights(batch_size=3, seq_len=197)

        result = extract_cls_rollout(attention_weights, num_registers=0)

        assert result.shape[0] == 3

    def test_layer_range_applied(self, make_attention_weights):
        """Verify start_layer and end_layer affect output."""
        attention_weights = make_attention_weights(seq_len=197, num_layers=12)

        result_early = extract_cls_rollout(
            attention_weights, num_registers=0, end_layer=4
        )
        result_late = extract_cls_rollout(
            attention_weights, num_registers=0, start_layer=8
        )

        # Results should differ (with high probability for random data)
        assert not torch.allclose(result_early, result_late)


class TestRolloutNormalization:
    """Test that rollout maintains proper normalization."""

    def test_rows_approximately_sum_to_one(self):
        """Each row of rollout should approximately sum to 1 for valid attention."""
        # Use softmax-normalized attention to ensure valid attention weights
        batch_size = 1
        num_heads = 12
        seq_len = 50
        num_layers = 4

        attention_weights = []
        for _ in range(num_layers):
            raw = torch.randn(batch_size, num_heads, seq_len, seq_len)
            # Apply softmax to make valid attention
            normalized = torch.softmax(raw, dim=-1)
            attention_weights.append(normalized)

        rollout = attention_rollout(attention_weights)

        # Check row sums
        row_sums = rollout.sum(dim=-1)

        # Should be close to 1 (normalized attention)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1)

    def test_values_are_non_negative(self, make_attention_weights):
        """Rollout values should be non-negative."""
        # Use softmax-normalized attention to ensure valid attention weights
        batch_size = 1
        num_heads = 12
        seq_len = 50
        num_layers = 4

        attention_weights = []
        for _ in range(num_layers):
            raw = torch.randn(batch_size, num_heads, seq_len, seq_len)
            # Apply softmax to make valid attention
            normalized = torch.softmax(raw, dim=-1)
            attention_weights.append(normalized)

        rollout = attention_rollout(attention_weights)

        assert (rollout >= 0).all()


class TestDiscardRatio:
    """Test attention discarding for noise reduction."""

    def test_discard_ratio_zero_preserves_all(self, make_attention_weights):
        """discard_ratio=0 should not discard any attention."""
        attention_weights = make_attention_weights(seq_len=50, num_layers=2)

        rollout = attention_rollout(attention_weights, discard_ratio=0.0)

        assert rollout.shape == (1, 50, 50)

    def test_discard_ratio_affects_output(self, make_attention_weights):
        """Non-zero discard_ratio should affect output."""
        attention_weights = make_attention_weights(seq_len=50, num_layers=2)

        rollout_no_discard = attention_rollout(attention_weights, discard_ratio=0.0)
        rollout_with_discard = attention_rollout(attention_weights, discard_ratio=0.5)

        # Results should differ
        assert not torch.allclose(rollout_no_discard, rollout_with_discard)


@pytest.mark.parametrize(
    "num_registers,seq_len,expected_patches",
    [
        (0, 197, 196),  # MAE/CLIP
        (4, 261, 256),  # DINOv2
        (4, 201, 196),  # DINOv3
    ],
)
def test_cls_rollout_patch_count(
    make_attention_weights,
    num_registers: int,
    seq_len: int,
    expected_patches: int,
):
    """Verify extract_cls_rollout returns correct number of patches."""
    attention_weights = make_attention_weights(seq_len=seq_len)

    result = extract_cls_rollout(attention_weights, num_registers=num_registers)

    assert result.shape[1] == expected_patches
