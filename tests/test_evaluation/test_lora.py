"""Tests for LoRA fine-tuning configuration and constants."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from ssl_attention.config import FINETUNE_MODELS
from ssl_attention.evaluation.fine_tuning import (
    LORA_TARGET_MODULES,
    FineTuningConfig,
)


class TestLoRATargetModules:
    """Tests for the LORA_TARGET_MODULES constant."""

    def test_covers_all_vit_models(self) -> None:
        """All ViT model names should have LoRA target modules defined."""
        expected = {"dinov2", "dinov3", "mae", "clip", "siglip", "siglip2"}
        assert set(LORA_TARGET_MODULES.keys()) == expected

    def test_each_model_has_two_modules(self) -> None:
        """Each model should target exactly 2 attention modules (Q and V)."""
        for model_name, modules in LORA_TARGET_MODULES.items():
            assert len(modules) == 2, f"{model_name} has {len(modules)} modules, expected 2"

    def test_dinov2_mae_use_query_value(self) -> None:
        """DINOv2 and MAE use 'query'/'value' naming convention."""
        for model_name in ("dinov2", "mae"):
            assert LORA_TARGET_MODULES[model_name] == ["query", "value"]

    def test_dinov3_clip_siglip_use_proj(self) -> None:
        """DINOv3, CLIP, SigLIP, and SigLIP2 use 'q_proj'/'v_proj' naming convention."""
        for model_name in ("dinov3", "clip", "siglip", "siglip2"):
            assert LORA_TARGET_MODULES[model_name] == ["q_proj", "v_proj"]


class TestFineTuningConfigLoRA:
    """Tests for LoRA-related fields in FineTuningConfig."""

    def test_lora_and_freeze_raises(self) -> None:
        """use_lora=True + freeze_backbone=True should raise ValueError."""
        with pytest.raises(ValueError, match="conflicting strategies"):
            FineTuningConfig(
                model_name="dinov2",
                use_lora=True,
                freeze_backbone=True,
            )

    @pytest.mark.parametrize(
        "use_lora,explicit_lr,expected_lr",
        [
            (True, None, 1e-4),      # LoRA auto-adjusts default
            (True, 5e-6, 5e-6),      # Explicit LR preserved
            (False, None, 1e-5),     # Non-LoRA keeps default
        ],
    )
    def test_backbone_lr_adjustment(self, use_lora, explicit_lr, expected_lr) -> None:
        """Verify backbone LR is set correctly based on LoRA and explicit override."""
        kwargs: dict = {"model_name": "dinov2", "use_lora": use_lora}
        if explicit_lr is not None:
            kwargs["learning_rate_backbone"] = explicit_lr
        config = FineTuningConfig(**kwargs)
        assert config.learning_rate_backbone == expected_lr

    def test_asdict_roundtrip_includes_lora_fields(self) -> None:
        """asdict() should include all LoRA fields for checkpoint serialization."""
        config = FineTuningConfig(
            model_name="dinov2",
            use_lora=True,
            lora_rank=16,
            lora_alpha=64,
            lora_dropout=0.05,
            lora_target_modules=["query", "value"],
        )
        d = asdict(config)
        assert d["use_lora"] is True
        assert d["lora_rank"] == 16
        assert d["lora_alpha"] == 64
        assert d["lora_dropout"] == 0.05
        assert d["lora_target_modules"] == ["query", "value"]

    def test_default_lora_fields(self) -> None:
        """Default LoRA field values should match expected defaults."""
        config = FineTuningConfig(model_name="dinov2")
        assert config.use_lora is False
        assert config.lora_rank == 8
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.lora_target_modules is None

    def test_non_finetunable_model_raises(self) -> None:
        """Models outside FINETUNE_MODELS should be rejected."""
        assert "resnet50" not in FINETUNE_MODELS
        with pytest.raises(ValueError, match="not fine-tunable"):
            FineTuningConfig(model_name="resnet50")
