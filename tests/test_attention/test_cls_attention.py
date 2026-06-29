"""Tests for CLS attention extraction.

Priority 2: These tests verify extract_cls_attention() correctly handles:
- Register offset calculation (patch_start = 1 + num_registers)
- Output shape (B, num_patches) not (B, seq_len)
- Head fusion strategies (MEAN/MAX/MIN)
- Negative layer indexing
"""

from __future__ import annotations

import pytest
import torch

from ssl_attention.attention.cls_attention import (
    HeadFusion,
    attention_to_heatmap,
    extract_cls_attention,
    extract_cls_attention_all_layers,
    fuse_heads,
    get_per_head_attention,
)


class TestFuseHeads:
    """Test head fusion strategies."""

    def test_mean_fusion(self):
        """Verify MEAN fusion averages across heads."""
        # Create attention with known values
        attention = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
        )  # (1, 2, 2, 2)

        result = fuse_heads(attention, HeadFusion.MEAN)

        # Mean of heads: [[3, 4], [5, 6]]
        expected = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
        torch.testing.assert_close(result, expected)

    def test_max_fusion(self):
        """Verify MAX fusion takes maximum across heads."""
        attention = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
        )  # (1, 2, 2, 2)

        result = fuse_heads(attention, HeadFusion.MAX)

        # Max of heads: [[5, 6], [7, 8]]
        expected = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        torch.testing.assert_close(result, expected)

    def test_min_fusion(self):
        """Verify MIN fusion takes minimum across heads."""
        attention = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
        )  # (1, 2, 2, 2)

        result = fuse_heads(attention, HeadFusion.MIN)

        # Min of heads: [[1, 2], [3, 4]]
        expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        torch.testing.assert_close(result, expected)

    def test_head_indices_selection(self):
        """Verify head_indices selects specific heads."""
        attention = torch.randn(2, 12, 197, 197)

        # Select only heads 0 and 5
        result = fuse_heads(attention, HeadFusion.MEAN, head_indices=[0, 5])

        # Should be mean of just those 2 heads
        expected = attention[:, [0, 5]].mean(dim=1)
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        """Verify fuse_heads reduces head dimension."""
        attention = torch.randn(2, 12, 197, 197)
        result = fuse_heads(attention, HeadFusion.MEAN)
        assert result.shape == (2, 197, 197)


class TestExtractClsAttention:
    """Test CLS attention extraction."""

    def test_register_offset_dinov2(self, make_attention_weights):
        """Verify with num_registers=4, patches start at index 5."""
        # DINOv2: seq_len = 261, num_patches = 256
        attention_weights = make_attention_weights(seq_len=261)

        result = extract_cls_attention(attention_weights, layer=-1, num_registers=4)

        # Should return only patch tokens (256)
        assert result.shape == (1, 256)

    def test_register_offset_mae(self, make_attention_weights):
        """Verify with num_registers=0, patches start at index 1."""
        # MAE: seq_len = 197, num_patches = 196
        attention_weights = make_attention_weights(seq_len=197)

        result = extract_cls_attention(attention_weights, layer=-1, num_registers=0)

        # Should return only patch tokens (196)
        assert result.shape == (1, 196)

    def test_output_shape_is_num_patches_not_seq_len(self, make_attention_weights):
        """Verify output shape is (B, num_patches), not (B, seq_len)."""
        # Create attention for DINOv3 (seq=201, patches=196)
        attention_weights = make_attention_weights(seq_len=201)

        result = extract_cls_attention(attention_weights, layer=-1, num_registers=4)

        # Must be 196 (patches), not 201 (full sequence)
        assert result.shape[1] == 196
        assert result.shape[1] != 201

    def test_negative_layer_index(self, make_attention_weights):
        """Verify layer=-1 selects the last layer."""
        attention_weights = make_attention_weights(seq_len=197, num_layers=12)

        result_neg = extract_cls_attention(attention_weights, layer=-1, num_registers=0)
        result_pos = extract_cls_attention(attention_weights, layer=11, num_registers=0)

        # Both should access the same layer
        torch.testing.assert_close(result_neg, result_pos)

    def test_cls_row_extraction(self):
        """Verify we extract CLS row (position 0 attending to others)."""
        # Create attention with known pattern
        batch_size = 1
        num_heads = 2
        seq_len = 5  # CLS + 4 patches

        # Make CLS row (row 0) distinctive
        attention = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        attention[:, :, 0, :] = 1.0  # CLS attends uniformly

        attention_weights = [attention]
        result = extract_cls_attention(attention_weights, layer=0, num_registers=0)

        # CLS to patches (positions 1-4) should all be 1.0
        expected = torch.ones(1, 4)
        torch.testing.assert_close(result, expected)

    def test_batch_dimension_preserved(self, make_attention_weights):
        """Verify batch dimension is preserved in output."""
        attention_weights = make_attention_weights(batch_size=4, seq_len=197)

        result = extract_cls_attention(attention_weights, layer=-1, num_registers=0)

        assert result.shape[0] == 4

    def test_fusion_strategy_applied(self, make_attention_weights):
        """Verify fusion strategy affects output."""
        attention_weights = make_attention_weights(seq_len=197)

        result_mean = extract_cls_attention(
            attention_weights, layer=-1, num_registers=0, fusion=HeadFusion.MEAN
        )
        result_max = extract_cls_attention(
            attention_weights, layer=-1, num_registers=0, fusion=HeadFusion.MAX
        )

        # Results should differ (with high probability for random data)
        assert not torch.allclose(result_mean, result_max)


class TestExtractClsAttentionAllLayers:
    """Test extracting CLS attention from all layers."""

    def test_output_shape(self, make_attention_weights):
        """Verify output has shape (B, L, N)."""
        num_layers = 12
        attention_weights = make_attention_weights(seq_len=197, num_layers=num_layers)

        result = extract_cls_attention_all_layers(attention_weights, num_registers=0)

        assert result.shape == (1, 12, 196)

    def test_layer_ordering(self, make_attention_weights):
        """Verify layers are ordered from first to last."""
        attention_weights = make_attention_weights(seq_len=197, num_layers=12)

        result = extract_cls_attention_all_layers(attention_weights, num_registers=0)

        # First layer attention should match single layer extraction
        layer0 = extract_cls_attention(attention_weights, layer=0, num_registers=0)
        torch.testing.assert_close(result[:, 0, :], layer0)


class TestGetPerHeadAttention:
    """Test per-head attention extraction."""

    def test_output_shape(self, make_attention_weights):
        """Verify output has shape (B, H, N)."""
        attention_weights = make_attention_weights(seq_len=197, num_heads=12)

        result = get_per_head_attention(attention_weights, layer=-1, num_registers=0)

        assert result.shape == (1, 12, 196)

    def test_head_dimension_preserved(self, make_attention_weights):
        """Verify each head's attention is preserved separately."""
        attention_weights = make_attention_weights(seq_len=197, num_heads=12)

        result = get_per_head_attention(attention_weights, layer=-1, num_registers=0)

        # Verify shape: batch, heads, patches
        assert result.shape[1] == 12  # All 12 heads preserved


class TestAttentionToHeatmap:
    """Test attention upsampling to heatmap."""

    def test_output_shape(self):
        """Verify heatmap has correct output dimensions."""
        attention = torch.randn(1, 196)  # 14x14 patches
        heatmap = attention_to_heatmap(attention, image_size=224)

        assert heatmap.shape == (1, 224, 224)

    def test_unbatched_input(self):
        """Verify unbatched input produces unbatched output."""
        attention = torch.randn(196)  # No batch dimension
        heatmap = attention_to_heatmap(attention, image_size=224)

        assert heatmap.shape == (224, 224)

    def test_normalization(self):
        """Verify normalization produces values in [0, 1]."""
        attention = torch.randn(1, 196)
        heatmap = attention_to_heatmap(attention, image_size=224, normalize=True)

        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_no_normalization(self):
        """Verify normalize=False preserves original range."""
        attention = torch.randn(1, 196) * 10 + 5  # Values far from [0, 1]
        heatmap = attention_to_heatmap(attention, image_size=224, normalize=False)

        # Values should not be in [0, 1]
        # (with high probability for this distribution)
        assert heatmap.max() > 1.0 or heatmap.min() < 0.0

    def test_non_square_patches_raises(self):
        """Verify non-square patch count raises error."""
        attention = torch.randn(1, 195)  # Not a perfect square

        with pytest.raises(ValueError, match="square patch grid"):
            attention_to_heatmap(attention, image_size=224)

    def test_16x16_patches(self):
        """Verify 16x16 patch grid (DINOv2) works correctly."""
        attention = torch.randn(1, 256)  # 16x16 patches
        heatmap = attention_to_heatmap(attention, image_size=224)

        assert heatmap.shape == (1, 224, 224)

    def test_14x14_patches(self):
        """Verify 14x14 patch grid works correctly."""
        attention = torch.randn(1, 196)  # 14x14 patches
        heatmap = attention_to_heatmap(attention, image_size=224)

        assert heatmap.shape == (1, 224, 224)


@pytest.mark.parametrize(
    "num_registers,seq_len,expected_patches",
    [
        (0, 197, 196),  # MAE/CLIP: CLS + 196 patches
        (0, 196, 196),  # SigLIP edge case (no CLS, treating as if CLS exists)
        (4, 261, 256),  # DINOv2: CLS + 4 reg + 256 patches
        (4, 201, 196),  # DINOv3: CLS + 4 reg + 196 patches
    ],
)
def test_patch_start_calculation(
    make_attention_weights,
    num_registers: int,
    seq_len: int,
    expected_patches: int,
):
    """Verify patch_start = 1 + num_registers produces correct patch count."""
    attention_weights = make_attention_weights(seq_len=seq_len)

    result = extract_cls_attention(attention_weights, layer=-1, num_registers=num_registers)

    # Output should have exactly expected_patches tokens
    # Note: For SigLIP (seq_len=196), this test treats it as if it had a CLS token
    # In practice, SigLIP uses extract_mean_attention instead
    if seq_len == 196 and num_registers == 0:
        # SigLIP special case: all tokens are patches
        # patch_start = 1 means we get 195 tokens, not 196
        # This is intentional - real SigLIP code doesn't use extract_cls_attention
        assert result.shape[1] == 195
    else:
        assert result.shape[1] == expected_patches
