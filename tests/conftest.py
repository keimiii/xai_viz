"""Shared pytest fixtures for SSL Attention tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

# Model configurations: (seq_len, num_patches, num_registers, has_pooler)
MODEL_CONFIGS = {
    "dinov2": {"seq_len": 261, "num_patches": 256, "num_registers": 4, "has_pooler": False},
    "dinov3": {"seq_len": 201, "num_patches": 196, "num_registers": 4, "has_pooler": False},
    "mae": {"seq_len": 197, "num_patches": 196, "num_registers": 0, "has_pooler": False},
    "clip": {"seq_len": 197, "num_patches": 196, "num_registers": 0, "has_pooler": False},
    "siglip": {"seq_len": 196, "num_patches": 196, "num_registers": 0, "has_pooler": True},
    "siglip2": {"seq_len": 196, "num_patches": 196, "num_registers": 0, "has_pooler": True},
}


@dataclass
class MockHuggingFaceOutput:
    """Mock HuggingFace model output for testing _extract_output methods."""

    last_hidden_state: Tensor
    attentions: tuple[Tensor, ...]
    pooler_output: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


@pytest.fixture
def device() -> torch.device:
    """Get the test device (CPU for consistent testing)."""
    return torch.device("cpu")


@pytest.fixture
def make_mock_output() -> Callable[..., MockHuggingFaceOutput]:
    """Factory fixture for creating mock HuggingFace model outputs.

    Returns a factory function that creates MockHuggingFaceOutput instances
    with configurable sequence length, batch size, and optional pooler output.
    """

    def _make(
        seq_len: int,
        batch_size: int = 2,
        num_heads: int = 12,
        num_layers: int = 12,
        embed_dim: int = 768,
        has_pooler: bool = False,
        include_hidden_states: bool = False,
    ) -> MockHuggingFaceOutput:
        """Create a mock HuggingFace model output.

        Args:
            seq_len: Sequence length (CLS + registers + patches).
            batch_size: Number of samples in batch.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            embed_dim: Embedding dimension.
            has_pooler: Whether to include pooler_output.
            include_hidden_states: Whether to include hidden_states.

        Returns:
            MockHuggingFaceOutput with correct tensor shapes.
        """
        last_hidden_state = torch.randn(batch_size, seq_len, embed_dim)
        attentions = tuple(
            torch.randn(batch_size, num_heads, seq_len, seq_len) for _ in range(num_layers)
        )
        pooler_output = torch.randn(batch_size, embed_dim) if has_pooler else None

        hidden_states = None
        if include_hidden_states:
            # L+1 tensors: embedding layer + L transformer layers
            hidden_states = tuple(
                torch.randn(batch_size, seq_len, embed_dim) for _ in range(num_layers + 1)
            )

        return MockHuggingFaceOutput(
            last_hidden_state=last_hidden_state,
            attentions=attentions,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
        )

    return _make


@pytest.fixture
def sample_annotation():
    """Create a simple ImageAnnotation for testing.

    Returns a factory that imports and creates ImageAnnotation on demand
    to avoid import issues if the module isn't available.
    """
    from ssl_attention.data.annotations import BoundingBox, ImageAnnotation

    return ImageAnnotation(
        image_id="test_image.jpg",
        styles=("Q46261",),
        bboxes=(
            BoundingBox(left=0.1, top=0.2, width=0.3, height=0.4, label=0, group_label=0),
        ),
    )


@pytest.fixture
def make_annotation():
    """Factory for creating ImageAnnotation with custom bboxes."""
    from ssl_attention.data.annotations import BoundingBox, ImageAnnotation

    def _make(
        image_id: str = "test.jpg",
        bboxes: list[tuple[float, float, float, float]] | None = None,
    ) -> ImageAnnotation:
        """Create an ImageAnnotation.

        Args:
            image_id: Image filename.
            bboxes: List of (left, top, width, height) tuples.

        Returns:
            ImageAnnotation with the specified bboxes.
        """
        if bboxes is None:
            bboxes = [(0.1, 0.2, 0.3, 0.4)]

        bbox_objects = tuple(
            BoundingBox(left=left, top=t, width=w, height=h, label=i, group_label=0)
            for i, (left, t, w, h) in enumerate(bboxes)
        )

        return ImageAnnotation(
            image_id=image_id,
            styles=("Q46261",),
            bboxes=bbox_objects,
        )

    return _make


@pytest.fixture
def make_attention_weights():
    """Factory for creating mock attention weight tensors."""

    def _make(
        batch_size: int = 1,
        num_heads: int = 12,
        seq_len: int = 197,
        num_layers: int = 12,
    ) -> list[Tensor]:
        """Create mock attention weights.

        Args:
            batch_size: Number of samples.
            num_heads: Attention heads per layer.
            seq_len: Sequence length.
            num_layers: Number of layers.

        Returns:
            List of attention tensors, one per layer.
        """
        return [
            torch.randn(batch_size, num_heads, seq_len, seq_len) for _ in range(num_layers)
        ]

    return _make
