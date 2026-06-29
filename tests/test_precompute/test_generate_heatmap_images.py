"""Tests for generate_heatmap_images.py fine-tuned cache key support."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from PIL import Image as PILImage

from app.precompute import generate_heatmap_images as gh
from ssl_attention.config import AttentionMethod


def test_generate_heatmaps_uses_finetuned_cache_and_output_keys(tmp_path, monkeypatch) -> None:
    """Fine-tuned mode should read/write using '{model}_finetuned' keys."""
    image_id = "Q1234_wd0.jpg"
    image_path = tmp_path / image_id
    PILImage.new("RGB", (224, 224), color=(120, 120, 120)).save(image_path)

    # Patch global paths/render helpers for an isolated unit test
    monkeypatch.setattr(gh, "IMAGES_PATH", tmp_path)
    monkeypatch.setattr(
        gh,
        "render_heatmap",
        lambda *_args, **_kwargs: PILImage.new("RGB", (224, 224)),
    )
    monkeypatch.setattr(
        gh,
        "create_attention_overlay",
        lambda *_args, **_kwargs: PILImage.new("RGB", (224, 224)),
    )

    dataset = [
        {
            "image_id": image_id,
            "annotation": SimpleNamespace(bboxes=[]),
        }
    ]

    cache = MagicMock()
    cache.load.return_value = torch.ones(14, 14)

    output_dir = tmp_path / "heatmaps"
    stats = gh.generate_heatmaps_for_model(
        model_name="dinov2",
        dataset=dataset,  # type: ignore[arg-type]
        attention_cache=cache,
        output_dir=output_dir,
        layers=[0],
        methods=[AttentionMethod.CLS],
        skip_existing=False,
        cache_model_key="dinov2_finetuned",
        output_model_key="dinov2_finetuned",
    )

    cache.load.assert_called_once_with(
        "dinov2_finetuned",
        "layer0",
        image_id,
        variant="cls",
    )
    assert stats["processed"] == 3
    assert stats["errors"] == 0

    overlay_path = output_dir / "dinov2_finetuned" / "layer0" / "cls" / "overlay" / f"{image_id}.png"
    assert overlay_path.exists()
