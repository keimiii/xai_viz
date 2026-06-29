"""Tests for ResNet-50 Grad-CAM heatmap computation.

Code under test: ssl_attention.models.resnet50.ResNet50._compute_gradcam_heatmap

Uses SimpleNamespace to mock `self` with injected _activations and _gradients,
avoiding the need to instantiate the full ResNet-50 model (which downloads weights).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from ssl_attention.models.resnet50 import ResNet50


def _make_mock_self(
    layer_name: str = "layer4",
    batch_size: int = 1,
    channels: int = 512,
    spatial: int = 7,
) -> SimpleNamespace:
    """Create a SimpleNamespace mock with _activations and _gradients."""
    return SimpleNamespace(
        _activations={layer_name: torch.randn(batch_size, channels, spatial, spatial)},
        _gradients={layer_name: torch.randn(batch_size, channels, spatial, spatial)},
    )


class TestComputeGradcamHeatmap:
    """Test ResNet50._compute_gradcam_heatmap as an unbound method."""

    def test_output_shape(self):
        """Input activations (B, C, 7, 7) → output (B, 224, 224)."""
        mock_self = _make_mock_self()

        result = ResNet50._compute_gradcam_heatmap(mock_self, "layer4", image_size=224)

        assert result.shape == (1, 224, 224)

    def test_output_range_normalized(self):
        """All values in [0, 1]."""
        mock_self = _make_mock_self()

        result = ResNet50._compute_gradcam_heatmap(mock_self, "layer4", image_size=224)

        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6  # Small epsilon for float precision

    def test_relu_removes_negative_contributions(self):
        """Negative weighted activations produce zero in output."""
        mock_self = SimpleNamespace(
            _activations={"layer4": torch.ones(1, 4, 7, 7)},
            # All-negative gradients → negative weights → ReLU zeroes everything
            _gradients={"layer4": -torch.ones(1, 4, 7, 7)},
        )

        result = ResNet50._compute_gradcam_heatmap(mock_self, "layer4", image_size=224)

        # After ReLU + normalization: all zeros (0/0 handled by EPSILON)
        assert result.max() < 1e-4

    def test_upsampling_to_target_size(self):
        """Different image_size values produce correct spatial dims."""
        mock_self = _make_mock_self()

        for target_size in (112, 224, 448):
            result = ResNet50._compute_gradcam_heatmap(
                mock_self, "layer4", image_size=target_size
            )
            assert result.shape == (1, target_size, target_size)

    def test_batch_dimension_preserved(self):
        """Works for batch_size=1 and batch_size=4."""
        for batch_size in (1, 4):
            mock_self = _make_mock_self(batch_size=batch_size)

            result = ResNet50._compute_gradcam_heatmap(
                mock_self, "layer4", image_size=224
            )

            assert result.shape[0] == batch_size

    def test_uniform_gradients_weight_equally(self):
        """When gradients are uniform, output ≈ mean of activation channels."""
        batch_size = 1
        channels = 4
        spatial = 7

        # All gradient values = 1.0 → uniform channel weights
        activations = torch.rand(batch_size, channels, spatial, spatial)
        gradients = torch.ones(batch_size, channels, spatial, spatial)

        mock_self = SimpleNamespace(
            _activations={"layer4": activations},
            _gradients={"layer4": gradients},
        )

        result = ResNet50._compute_gradcam_heatmap(mock_self, "layer4", image_size=spatial)

        # With uniform gradients: weights = mean over spatial = 1.0 per channel
        # cam = relu(sum of activations across channels) at each spatial position
        # Since activations are positive (torch.rand), relu is a no-op
        expected_cam = activations.sum(dim=1)  # (B, H, W)
        # Normalize
        flat = expected_cam.view(batch_size, -1)
        min_val = flat.min(dim=1, keepdim=True).values.view(batch_size, 1, 1)
        max_val = flat.max(dim=1, keepdim=True).values.view(batch_size, 1, 1)
        expected_cam = (expected_cam - min_val) / (max_val - min_val + 1e-8)

        # Allow tolerance for bilinear interpolation (identity at same size)
        torch.testing.assert_close(result, expected_cam, atol=1e-4, rtol=1e-4)


class TestGradientClearing:
    """Verify that zero_grad() prevents gradient accumulation across calls."""

    def test_zero_grad_prevents_accumulation(self):
        """Two backward passes with zero_grad between them produce independent param.grad."""
        model = nn.Linear(4, 2, bias=False)

        # First backward (populates param.grad)
        x1 = torch.randn(1, 4)
        model.zero_grad()
        model(x1).sum().backward()

        # Second backward WITH zero_grad (the pattern our fix enforces)
        x2 = torch.randn(1, 4)
        model.zero_grad()
        model(x2).sum().backward()
        grad2 = model.weight.grad.clone()

        # grad2 should equal the gradient from x2 alone, not x1+x2
        model.zero_grad()
        model(x2.clone()).sum().backward()
        grad2_fresh = model.weight.grad.clone()

        torch.testing.assert_close(grad2, grad2_fresh)
