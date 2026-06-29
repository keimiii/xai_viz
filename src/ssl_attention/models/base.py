"""Base class for vision model wrappers."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoConfig, AutoImageProcessor, BatchFeature

from ssl_attention.models.protocols import ModelOutput
from ssl_attention.utils.device import get_device, get_dtype_for_device


def _resolve_mae_patch_size(model: nn.Module) -> tuple[int, int]:
    """Resolve MAE patch size from the wrapped HF model or adapter wrapper."""
    config_sources = (
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
    )

    for config in config_sources:
        patch_size = getattr(config, "patch_size", None)
        if patch_size is None:
            continue
        if isinstance(patch_size, int):
            return patch_size, patch_size
        if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            return int(patch_size[0]), int(patch_size[1])

    raise ValueError("Could not determine MAE patch size for deterministic analysis forward.")


def build_mae_analysis_noise(model: nn.Module, pixel_values: Tensor) -> Tensor:
    """Build canonical MAE noise so patch order stays stable during analysis.

    HuggingFace MAE still calls `random_masking()` when `mask_ratio=0.0`; the
    per-patch noise controls the patch ordering used by that path.
    """
    batch_size, _, height, width = pixel_values.shape
    patch_height, patch_width = _resolve_mae_patch_size(model)

    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(
            "MAE analysis inputs must align with the model patch size. "
            f"Got image size {height}x{width} for patches {patch_height}x{patch_width}."
        )

    seq_length = (height // patch_height) * (width // patch_width)
    canonical_noise = torch.arange(
        seq_length,
        device=pixel_values.device,
        dtype=torch.float32,
    )
    return canonical_noise.unsqueeze(0).expand(batch_size, -1)


def forward_mae_for_analysis(
    model: nn.Module,
    pixel_values: Tensor,
    *,
    output_attentions: bool = True,
    output_hidden_states: bool = False,
) -> Any:
    """Run a MAE analysis forward with deterministic patch ordering."""
    noise = build_mae_analysis_noise(model, pixel_values)
    return model(
        pixel_values=pixel_values,
        noise=noise,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )


class BaseVisionModel(ABC, nn.Module):
    """Abstract base class for SSL vision model wrappers.

    Provides common functionality for device handling, preprocessing,
    and inference context management. Subclasses must implement
    _load_model() and _extract_output().

    Attributes:
        model_name: Short identifier (e.g., 'dinov2').
        model_id: HuggingFace model identifier.
        patch_size: Pixels per patch (14 or 16).
        embed_dim: Token embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Attention heads per layer.
        num_registers: Register token count (0 if none).
        device: Compute device.
        dtype: Tensor dtype for inference.
    """

    # Subclasses must define these
    model_name: str
    model_id: str
    patch_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    num_registers: int = 0

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the model wrapper.

        Args:
            device: Target device. Auto-detects if None.
            dtype: Tensor dtype. Uses optimal for device if None.
        """
        super().__init__()

        self.device = device or get_device()
        self.dtype = dtype or get_dtype_for_device(self.device)

        # Load processor and model
        self.processor = self._load_processor()
        self.model = self._load_model()

        # Move to device and set eval mode
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def _load_processor(self) -> Any:
        """Load the image processor for this model.

        Returns:
            HuggingFace image processor.
        """
        return AutoImageProcessor.from_pretrained(self.model_id)

    def _load_config(self) -> Any:
        """Load model config with attention output enabled.

        Returns:
            HuggingFace model config with output_attentions=True.
        """
        config = AutoConfig.from_pretrained(self.model_id)
        config.output_attentions = True
        return config

    @abstractmethod
    def _load_model(self) -> nn.Module:
        """Load the underlying model.

        Subclasses implement this to load their specific architecture.
        Model should NOT be moved to device here (done in __init__).

        Returns:
            The loaded PyTorch model.
        """
        ...

    @abstractmethod
    def _extract_output(
        self, model_output: Any, include_hidden_states: bool = False
    ) -> ModelOutput:
        """Extract standardized output from model-specific output.

        Args:
            model_output: Raw output from the model's forward pass.
            include_hidden_states: Whether to include per-layer hidden states.

        Returns:
            ModelOutput with cls_token, patch_tokens, attention_weights,
            and optionally hidden_states.
        """
        ...

    def preprocess(self, images: list[Image.Image]) -> Tensor:
        """Preprocess PIL images for model input.

        Args:
            images: List of PIL Images.

        Returns:
            Tensor of shape (B, C, H, W) on the model's device.
        """
        # Most HuggingFace processors return BatchFeature with 'pixel_values'
        processed: BatchFeature = self.processor(images=images, return_tensors="pt")
        pixel_values: Tensor = processed["pixel_values"]
        return pixel_values.to(device=self.device, dtype=self.dtype)

    @contextmanager
    def inference_context(self) -> Generator[None, None, None]:
        """Context manager for inference (no gradients, eval mode).

        Usage:
            with model.inference_context():
                output = model(images)
        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            if was_training:
                self.model.train()

    def forward(
        self, images: Tensor, output_hidden_states: bool = False
    ) -> ModelOutput:
        """Process preprocessed images through the model.

        Args:
            images: Preprocessed images from preprocess(), shape (B, C, H, W).
            output_hidden_states: If True, include per-layer hidden states in output.

        Returns:
            ModelOutput with embeddings, attention weights, and optionally hidden_states.
        """
        with self.inference_context():
            # HuggingFace MAE still permutes patches unless analysis forwards
            # supply deterministic noise alongside mask_ratio=0.0.
            if self.model_name == "mae":
                model_output = forward_mae_for_analysis(
                    self.model,
                    images,
                    output_attentions=True,
                    output_hidden_states=output_hidden_states,
                )
            else:
                model_output = self.model(
                    pixel_values=images,
                    output_attentions=True,
                    output_hidden_states=output_hidden_states,
                )
            return self._extract_output(model_output, output_hidden_states)

    @property
    def image_size(self) -> int:
        """Expected input image size (assumes square images)."""
        # Most ViT models use 224x224
        size = getattr(self.processor, "size", {})
        if isinstance(size, dict):
            val = size.get("height", size.get("shortest_edge", 224))
            return int(val) if val is not None else 224
        return 224

    @property
    def num_patches_per_side(self) -> int:
        """Number of patches along one dimension."""
        return self.image_size // self.patch_size

    @property
    def total_patches(self) -> int:
        """Total number of image patches."""
        return self.num_patches_per_side ** 2

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"device={self.device}, "
            f"dtype={self.dtype}, "
            f"patches={self.total_patches})"
        )
