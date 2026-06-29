"""Heatmap rendering utilities for attention visualization.

This module converts attention tensors into colored heatmap images
using matplotlib colormaps.

Key functions:
- `apply_colormap()`: Apply colormap to normalized tensor
- `render_heatmap()`: Full pipeline from attention to colored image
- `render_heatmap_batch()`: Batch version for efficiency
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from PIL import Image

# Use non-interactive backend for headless rendering
matplotlib.use("Agg")


# Pre-load common colormaps for efficiency
_COLORMAPS: dict[str, matplotlib.colors.Colormap] = {}


def _get_colormap(name: str) -> matplotlib.colors.Colormap:
    """Get colormap by name with caching."""
    if name not in _COLORMAPS:
        _COLORMAPS[name] = matplotlib.colormaps.get_cmap(name)
    return _COLORMAPS[name]


def apply_colormap(
    attention: Tensor | np.ndarray,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Apply a matplotlib colormap to a 2D attention map.

    Args:
        attention: 2D attention map of shape (H, W) with values in [0, 1].
        colormap: Name of matplotlib colormap. Options include:
            - "viridis": Yellow-green-blue (default, perceptually uniform)
            - "plasma": Purple-pink-yellow (high contrast)
            - "inferno": Black-red-yellow (dramatic)
            - "magma": Black-purple-white (subtle)
            - "jet": Rainbow (not recommended, but common)
            - "hot": Black-red-yellow-white
        vmin: Minimum value for normalization. If None, uses attention.min().
        vmax: Maximum value for normalization. If None, uses attention.max().

    Returns:
        RGBA image as numpy array of shape (H, W, 4) with uint8 values [0, 255].

    Example:
        >>> attn = torch.rand(224, 224)
        >>> rgba = apply_colormap(attn, colormap="viridis")
        >>> rgba.shape
        (224, 224, 4)
    """
    # Convert to numpy if needed
    if isinstance(attention, Tensor):
        attention = attention.detach().cpu().numpy()

    # Normalize to [0, 1]
    if vmin is None:
        vmin = float(attention.min())
    if vmax is None:
        vmax = float(attention.max())

    if vmax - vmin > 1e-8:
        normalized = (attention - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(attention)

    # Clip to valid range
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    cmap = _get_colormap(colormap)
    colored = cmap(normalized)  # Returns (H, W, 4) float in [0, 1]

    # Convert to uint8
    return (colored * 255).astype(np.uint8)


def render_heatmap(
    attention: Tensor,
    colormap: str = "viridis",
    output_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Render attention tensor as a colored heatmap PIL Image.

    Args:
        attention: 2D attention map of shape (H, W).
        colormap: Matplotlib colormap name.
        output_size: Optional (width, height) to resize output.

    Returns:
        PIL Image in RGBA mode.

    Example:
        >>> attn = torch.rand(224, 224)
        >>> img = render_heatmap(attn, colormap="plasma")
        >>> img.save("heatmap.png")
    """
    from PIL import Image as PILImage

    # Ensure 2D
    if attention.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {attention.shape}")

    # Apply colormap
    rgba = apply_colormap(attention, colormap=colormap)

    # Convert to PIL
    img = PILImage.fromarray(rgba, mode="RGBA")

    # Resize if requested
    if output_size is not None:
        img = img.resize(output_size, resample=PILImage.Resampling.BILINEAR)

    return img


def render_heatmap_batch(
    attention_batch: Tensor,
    colormap: str = "viridis",
    output_size: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """Render a batch of attention maps as heatmap images.

    Args:
        attention_batch: 3D tensor of shape (B, H, W).
        colormap: Matplotlib colormap name.
        output_size: Optional (width, height) to resize outputs.

    Returns:
        List of PIL Images, one per batch element.

    Example:
        >>> batch = torch.rand(10, 224, 224)
        >>> images = render_heatmap_batch(batch, colormap="inferno")
        >>> len(images)
        10
    """
    if attention_batch.dim() != 3:
        raise ValueError(f"Expected 3D tensor (B, H, W), got shape {attention_batch.shape}")

    return [
        render_heatmap(attention_batch[i], colormap=colormap, output_size=output_size)
        for i in range(attention_batch.shape[0])
    ]


def create_colorbar(
    colormap: str = "viridis",
    width: int = 40,
    height: int = 256,
    orientation: str = "vertical",
    label_min: str = "Low",
    label_max: str = "High",
) -> Image.Image:
    """Create a standalone colorbar image for legends.

    Args:
        colormap: Matplotlib colormap name.
        width: Width of colorbar in pixels.
        height: Height of colorbar in pixels.
        orientation: "vertical" or "horizontal".
        label_min: Label for minimum value.
        label_max: Label for maximum value.

    Returns:
        PIL Image of the colorbar.
    """
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    # Generate gradient
    if orientation == "vertical":
        gradient = np.linspace(1, 0, height).reshape(height, 1)
        gradient = np.tile(gradient, (1, width))
    else:
        gradient = np.linspace(0, 1, width).reshape(1, width)
        gradient = np.tile(gradient, (height, 1))

    # Apply colormap
    gradient_tensor = torch.from_numpy(gradient).float()
    rgba = apply_colormap(gradient_tensor, colormap=colormap, vmin=0, vmax=1)
    img = PILImage.fromarray(rgba, mode="RGBA")

    # Add labels (optional - requires font)
    try:
        # Create larger canvas for labels
        padding = 30 if orientation == "vertical" else 20
        if orientation == "vertical":
            canvas = PILImage.new("RGBA", (width + padding, height), (255, 255, 255, 0))
            canvas.paste(img, (0, 0))
        else:
            canvas = PILImage.new("RGBA", (width, height + padding), (255, 255, 255, 0))
            canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        # Use default font (PIL's built-in)
        font = ImageFont.load_default()

        if orientation == "vertical":
            draw.text((width + 5, 0), label_max, fill=(0, 0, 0, 255), font=font)
            draw.text((width + 5, height - 15), label_min, fill=(0, 0, 0, 255), font=font)
        else:
            draw.text((0, height + 2), label_min, fill=(0, 0, 0, 255), font=font)
            draw.text((width - 30, height + 2), label_max, fill=(0, 0, 0, 255), font=font)

        return canvas
    except Exception:
        # If font fails, just return the gradient
        return img
