"""Tests for generate_attention_cache.py fine-tuned model support."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.precompute import generate_attention_cache as generate_attention_cache_module
from app.precompute.generate_attention_cache import (
    FINETUNE_MODELS,
    discover_checkpoints,
    discover_checkpoints_by_strategy,
    generate_attention_for_model,
)
from ssl_attention.config import MODELS, AttentionMethod


class TestDiscoverCheckpoints:
    """Tests for discover_checkpoints() helper."""

    def test_finds_existing_checkpoints(self, tmp_path: Path) -> None:
        """Finds .pt files in temp dir and returns correct dict."""
        # Create fake checkpoint files
        (tmp_path / "dinov2_finetuned.pt").touch()
        (tmp_path / "siglip_finetuned.pt").touch()

        result = discover_checkpoints(tmp_path)

        assert "dinov2" in result
        assert "siglip" in result
        assert result["dinov2"] == tmp_path / "dinov2_finetuned.pt"
        assert result["siglip"] == tmp_path / "siglip_finetuned.pt"

    def test_empty_dir_returns_empty_dict(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Empty dir returns empty dict with warnings printed."""
        result = discover_checkpoints(tmp_path)

        assert result == {}
        captured = capsys.readouterr()
        assert "Warning: No checkpoint for" in captured.out

    def test_filters_to_requested_models(self, tmp_path: Path) -> None:
        """Only checks models in model_names list."""
        (tmp_path / "dinov2_finetuned.pt").touch()
        (tmp_path / "siglip_finetuned.pt").touch()

        result = discover_checkpoints(tmp_path, model_names=["dinov2"])

        assert "dinov2" in result
        assert "siglip" not in result

    def test_skips_resnet(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """resnet50 is excluded with warning."""
        (tmp_path / "resnet50_finetuned.pt").touch()

        result = discover_checkpoints(tmp_path, model_names=["resnet50"])

        assert result == {}
        captured = capsys.readouterr()
        assert "not fine-tunable" in captured.out

    def test_mixed_found_and_missing(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Returns only models with checkpoints, warns about missing."""
        (tmp_path / "dinov2_finetuned.pt").touch()
        # mae has no checkpoint

        result = discover_checkpoints(tmp_path, model_names=["dinov2", "mae"])

        assert "dinov2" in result
        assert "mae" not in result
        captured = capsys.readouterr()
        assert "mae" in captured.out


class TestDiscoverCheckpointsByStrategy:
    """Tests for strategy-aware checkpoint discovery."""

    def test_discovers_multiple_strategies_per_model(self, tmp_path: Path) -> None:
        (tmp_path / "dinov2_lora_finetuned.pt").touch()
        (tmp_path / "dinov2_full_finetuned.pt").touch()

        result = discover_checkpoints_by_strategy(tmp_path, model_names=["dinov2"])

        assert "dinov2" in result
        assert set(result["dinov2"]) == {"lora", "full"}

    def test_uses_legacy_checkpoint_for_full_only(self, tmp_path: Path) -> None:
        (tmp_path / "clip_finetuned.pt").touch()

        result = discover_checkpoints_by_strategy(
            tmp_path,
            model_names=["clip"],
            strategies=["full", "linear_probe"],
        )

        assert result["clip"]["full"] == tmp_path / "clip_finetuned.pt"
        assert "linear_probe" not in result["clip"]


class TestCacheModelKey:
    """Verify the cache key naming convention for fine-tuned models."""

    def test_finetuned_key_format(self) -> None:
        """Fine-tuned cache keys follow '{model}_finetuned' pattern."""
        for model_name in FINETUNE_MODELS:
            expected = f"{model_name}_finetuned"
            assert expected.endswith("_finetuned")
            # Verify it doesn't collide with any frozen model key
            assert expected not in MODELS


class TestNumRegistersFromConfig:
    """Verify num_registers is accessible from MODELS config for all models."""

    @pytest.mark.parametrize("model_name", list(MODELS.keys()))
    def test_num_registers_accessible(self, model_name: str) -> None:
        """MODELS[name].num_registers is accessible for all models."""
        config = MODELS[model_name]
        assert isinstance(config.num_registers, int)
        assert config.num_registers >= 0


class TestFineTuneModelsConstant:
    """Verify FINETUNE_MODELS aligns with project expectations."""

    def test_excludes_resnet50(self) -> None:
        """ResNet-50 should not be in FINETUNE_MODELS."""
        assert "resnet50" not in FINETUNE_MODELS

    def test_includes_all_vits(self) -> None:
        """All ViT models should be in FINETUNE_MODELS."""
        expected_vits = {"dinov2", "dinov3", "mae", "clip", "siglip", "siglip2"}
        assert expected_vits == FINETUNE_MODELS

    def test_all_finetune_models_exist_in_config(self) -> None:
        """Every model in FINETUNE_MODELS must exist in MODELS config."""
        for name in FINETUNE_MODELS:
            assert name in MODELS, f"{name} in FINETUNE_MODELS but not in MODELS config"


class FakeCache:
    """Minimal cache test double for per-head generation tests."""

    def __init__(self) -> None:
        self.stored_variants: list[str] = []

    def exists(self, model: str, layer: str, image_id: str, variant: str) -> bool:
        return variant == "cls"

    def store(self, model: str, layer: str, image_id: str, tensor: torch.Tensor, variant: str) -> None:
        self.stored_variants.append(variant)


class FakeModel:
    """Minimal model test double for attention cache generation."""

    def to(self, device: str) -> FakeModel:
        return self

    def eval(self) -> FakeModel:
        return self

    def preprocess(self, images: list[object]) -> torch.Tensor:
        return torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(attention_weights=[torch.zeros(1)] * 12)


class TestPerHeadAttentionGeneration:
    """Verify --per-head generation handles partially cached attention safely."""

    def test_backfills_missing_heads_when_fused_attention_is_already_cached(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = [{"image_id": "img1.jpg", "image": object()}]
        cache = FakeCache()

        monkeypatch.setattr(generate_attention_cache_module, "create_model", lambda _model_name: FakeModel())
        monkeypatch.setattr(
            generate_attention_cache_module,
            "extract_cls_attention",
            lambda *_args, **_kwargs: torch.ones((1, 14, 14), dtype=torch.float32),
        )
        monkeypatch.setattr(
            generate_attention_cache_module,
            "attention_to_heatmap",
            lambda attention, image_size: attention,
        )

        stats = generate_attention_for_model(
            "dinov2",
            dataset=dataset,  # type: ignore[arg-type]
            cache=cache,  # type: ignore[arg-type]
            layers=[11],
            methods=[AttentionMethod.CLS],
            device="cpu",
            skip_existing=True,
            per_head=True,
        )

        assert stats["processed"] == MODELS["dinov2"].num_heads
        assert stats["skipped"] == 1
        assert cache.stored_variants == [
            f"cls_head{head_idx}" for head_idx in range(MODELS["dinov2"].num_heads)
        ]
