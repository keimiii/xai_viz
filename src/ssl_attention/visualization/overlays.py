"""Overlay utilities for combining images, heatmaps, and annotations.

This module provides functions for creating composite visualizations
that show attention patterns overlaid on original images with bounding boxes.

Key functions:
- `create_overlay()`: Blend heatmap with original image
- `draw_bboxes()`: Draw bounding box annotations
- `create_attention_overlay()`: Full pipeline for attention visualization
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

if TYPE_CHECKING:
    from ssl_attention.data.annotations import BoundingBox, ImageAnnotation


# Default colors for bounding boxes (RGBA)
DEFAULT_BBOX_COLOR = (0, 255, 0, 255)  # Green
DEFAULT_BBOX_WIDTH = 2


def create_overlay(
    background: PILImage.Image,
    foreground: PILImage.Image,
    alpha: float = 0.5,
) -> PILImage.Image:
    """Blend foreground image over background with transparency.

    Args:
        background: Base image (RGB or RGBA).
        foreground: Overlay image (must have alpha channel or will be converted).
        alpha: Blending factor [0, 1]. 0 = only background, 1 = only foreground.

    Returns:
        Blended PIL Image in RGBA mode.

    Example:
        >>> original = Image.open("church.jpg")
        >>> heatmap = render_heatmap(attention)
        >>> overlay = create_overlay(original, heatmap, alpha=0.5)
    """
    # Ensure same size
    if background.size != foreground.size:
        foreground = foreground.resize(background.size, resample=PILImage.Resampling.BILINEAR)

    # Convert to RGBA
    if background.mode != "RGBA":
        background = background.convert("RGBA")
    if foreground.mode != "RGBA":
        foreground = foreground.convert("RGBA")

    # Convert to numpy for blending
    bg_arr = np.array(background, dtype=np.float32)
    fg_arr = np.array(foreground, dtype=np.float32)

    # Use foreground alpha channel, scaled by our alpha parameter
    fg_alpha = (fg_arr[:, :, 3:4] / 255.0) * alpha
    bg_alpha = 1.0 - fg_alpha

    # Blend RGB channels
    blended = bg_arr.copy()
    blended[:, :, :3] = bg_arr[:, :, :3] * bg_alpha + fg_arr[:, :, :3] * fg_alpha

    # Keep full opacity
    blended[:, :, 3] = 255

    return PILImage.fromarray(blended.astype(np.uint8), mode="RGBA")


def draw_bboxes(
    image: PILImage.Image,
    bboxes: list[BoundingBox] | tuple[BoundingBox, ...],
    color: tuple[int, int, int, int] = DEFAULT_BBOX_COLOR,
    width: int = DEFAULT_BBOX_WIDTH,
    labels: list[str] | None = None,
) -> PILImage.Image:
    """Draw bounding boxes on an image.

    Args:
        image: Input image (will be converted to RGBA).
        bboxes: List of BoundingBox objects with normalized coordinates.
        color: RGBA color for boxes.
        width: Line width in pixels.
        labels: Optional text labels for each box.

    Returns:
        New PIL Image with bounding boxes drawn.

    Example:
        >>> from ssl_attention.data.annotations import load_annotations
        >>> annotations = load_annotations(ANNOTATIONS_PATH)
        >>> img = Image.open("church.jpg")
        >>> img_with_boxes = draw_bboxes(img, annotations["Q123.jpg"].bboxes)
    """
    # Copy and convert to RGBA
    image = image.convert("RGBA") if image.mode != "RGBA" else image.copy()

    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    for i, bbox in enumerate(bboxes):
        # Convert normalized coords to pixels
        x1 = int(bbox.left * img_width)
        y1 = int(bbox.top * img_height)
        x2 = int(bbox.right * img_width)
        y2 = int(bbox.bottom * img_height)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # Draw label if provided
        if labels is not None and i < len(labels):
            draw.text((x1, y1 - 12), labels[i], fill=color)

    return image


def draw_bboxes_by_label(
    image: PILImage.Image,
    bboxes: list[BoundingBox] | tuple[BoundingBox, ...],
    label_colors: dict[int, tuple[int, int, int, int]] | None = None,
    width: int = DEFAULT_BBOX_WIDTH,
) -> PILImage.Image:
    """Draw bounding boxes with colors based on feature type label.

    Args:
        image: Input image.
        bboxes: List of BoundingBox objects.
        label_colors: Dict mapping label index to RGBA color.
            If None, uses a default palette.
        width: Line width in pixels.

    Returns:
        New PIL Image with colored bounding boxes.
    """
    # Default color palette (12 distinct colors)
    DEFAULT_PALETTE = [
        (255, 0, 0, 255),      # Red
        (0, 255, 0, 255),      # Green
        (0, 0, 255, 255),      # Blue
        (255, 255, 0, 255),    # Yellow
        (255, 0, 255, 255),    # Magenta
        (0, 255, 255, 255),    # Cyan
        (255, 128, 0, 255),    # Orange
        (128, 0, 255, 255),    # Purple
        (0, 255, 128, 255),    # Spring green
        (255, 0, 128, 255),    # Pink
        (128, 255, 0, 255),    # Lime
        (0, 128, 255, 255),    # Sky blue
    ]

    image = image.convert("RGBA") if image.mode != "RGBA" else image.copy()

    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    for bbox in bboxes:
        # Get color for this label
        if label_colors and bbox.label in label_colors:
            color = label_colors[bbox.label]
        else:
            color = DEFAULT_PALETTE[bbox.label % len(DEFAULT_PALETTE)]

        # Convert normalized coords to pixels
        x1 = int(bbox.left * img_width)
        y1 = int(bbox.top * img_height)
        x2 = int(bbox.right * img_width)
        y2 = int(bbox.bottom * img_height)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    return image


def create_attention_overlay(
    image: PILImage.Image,
    attention_heatmap: PILImage.Image,
    annotation: ImageAnnotation | None = None,
    alpha: float = 0.5,
    show_bboxes: bool = True,
    bbox_color: tuple[int, int, int, int] = DEFAULT_BBOX_COLOR,
    bbox_width: int = DEFAULT_BBOX_WIDTH,
) -> PILImage.Image:
    """Create a complete attention visualization overlay.

    Combines:
    1. Original image as background
    2. Attention heatmap with transparency
    3. Optional bounding box annotations

    Args:
        image: Original image.
        attention_heatmap: Pre-rendered heatmap image (from render_heatmap).
        annotation: Optional ImageAnnotation for bbox overlay.
        alpha: Heatmap transparency [0, 1].
        show_bboxes: Whether to draw bounding boxes.
        bbox_color: Color for bounding boxes.
        bbox_width: Line width for bounding boxes.

    Returns:
        Composite PIL Image showing attention over the image.

    Example:
        >>> from ssl_attention.visualization import render_heatmap, create_attention_overlay
        >>> heatmap_img = render_heatmap(attention)
        >>> overlay = create_attention_overlay(
        ...     original_img, heatmap_img, annotation, alpha=0.5
        ... )
    """
    # Start with heatmap overlay
    result = create_overlay(image, attention_heatmap, alpha=alpha)

    # Add bounding boxes if requested
    if show_bboxes and annotation is not None and annotation.bboxes:
        result = draw_bboxes(result, annotation.bboxes, color=bbox_color, width=bbox_width)

    return result


def create_side_by_side(
    images: list[PILImage.Image],
    titles: list[str] | None = None,
    padding: int = 10,
    background_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> PILImage.Image:
    """Create a side-by-side comparison of multiple images.

    Args:
        images: List of PIL Images to combine.
        titles: Optional titles for each image.
        padding: Space between images in pixels.
        background_color: RGBA background color.

    Returns:
        Combined PIL Image with all inputs side by side.
    """
    if not images:
        raise ValueError("At least one image required")

    # Ensure all images are RGBA
    images = [img.convert("RGBA") if img.mode != "RGBA" else img for img in images]

    # Find max height and total width
    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images) + padding * (len(images) - 1)

    # Add space for titles if provided
    title_height = 20 if titles else 0
    canvas_height = max_height + title_height

    # Create canvas
    canvas = PILImage.new("RGBA", (total_width, canvas_height), background_color)

    # Paste images
    x_offset = 0
    for i, img in enumerate(images):
        # Center vertically
        y_offset = title_height + (max_height - img.height) // 2
        canvas.paste(img, (x_offset, y_offset))

        # Draw title if provided
        if titles and i < len(titles):
            draw = ImageDraw.Draw(canvas)
            draw.text((x_offset + 5, 2), titles[i], fill=(0, 0, 0, 255))

        x_offset += img.width + padding

    return canvas


def create_grid(
    images: list[PILImage.Image],
    cols: int = 3,
    padding: int = 5,
    background_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> PILImage.Image:
    """Arrange images in a grid layout.

    Args:
        images: List of PIL Images.
        cols: Number of columns.
        padding: Space between images.
        background_color: RGBA background color.

    Returns:
        Grid layout PIL Image.
    """
    if not images:
        raise ValueError("At least one image required")

    # Ensure all images are RGBA
    images = [img.convert("RGBA") if img.mode != "RGBA" else img for img in images]

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create canvas
    canvas_width = cols * max_width + (cols - 1) * padding
    canvas_height = rows * max_height + (rows - 1) * padding
    canvas = PILImage.new("RGBA", (canvas_width, canvas_height), background_color)

    # Paste images
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x = col * (max_width + padding) + (max_width - img.width) // 2
        y = row * (max_height + padding) + (max_height - img.height) // 2

        canvas.paste(img, (x, y))

    return canvas
