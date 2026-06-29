"""Tests for app.backend.validators."""

import pytest
from fastapi import HTTPException

from app.backend.validators import (
    resolve_default_method,
    validate_layer_for_model,
    validate_method,
    validate_model,
)


def test_resolve_default_method_cls_models():
    """Models using CLS attention should resolve to 'cls'."""
    for model in ("dinov2", "dinov3", "mae", "clip"):
        assert resolve_default_method(model) == "cls", f"{model} should default to cls"


def test_resolve_default_method_siglip_models():
    """SigLIP and SigLIP2 should resolve to 'mean'."""
    assert resolve_default_method("siglip") == "mean"
    assert resolve_default_method("siglip2") == "mean"


def test_resolve_default_method_resnet():
    """ResNet-50 should resolve to 'gradcam'."""
    assert resolve_default_method("resnet50") == "gradcam"


def test_resolve_default_method_finetuned_variant():
    """Fine-tuned variants should inherit the base model default method."""
    assert resolve_default_method("dinov2_finetuned") == "cls"
    assert resolve_default_method("siglip2_finetuned") == "mean"
    assert resolve_default_method("dinov2_finetuned_lora") == "cls"


# --- validate_method tests ---


class TestValidateMethod:
    """Tests for validate_method."""

    def test_none_returns_default_cls(self):
        """method=None should return the model's default method."""
        assert validate_method("dinov2", None) == "cls"
        assert validate_method("dinov3", None) == "cls"
        assert validate_method("mae", None) == "cls"
        assert validate_method("clip", None) == "cls"

    def test_none_returns_default_mean(self):
        """SigLIP and SigLIP2 should default to 'mean'."""
        assert validate_method("siglip", None) == "mean"
        assert validate_method("siglip2", None) == "mean"

    def test_none_returns_default_gradcam(self):
        """ResNet should default to 'gradcam'."""
        assert validate_method("resnet50", None) == "gradcam"

    def test_explicit_cls_accepted(self):
        """Explicit 'cls' should be accepted for ViT models."""
        assert validate_method("dinov2", "cls") == "cls"

    def test_explicit_rollout_accepted(self):
        """Explicit 'rollout' should be accepted for ViT models."""
        assert validate_method("dinov2", "rollout") == "rollout"
        assert validate_method("mae", "rollout") == "rollout"

    def test_invalid_method_string(self):
        """Invalid method string should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_method("dinov2", "invalid_method")
        assert exc_info.value.status_code == 400
        assert "Invalid method" in exc_info.value.detail

    def test_method_not_available_for_model(self):
        """Valid method but not supported by the model should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_method("resnet50", "cls")
        assert exc_info.value.status_code == 400
        assert "not available" in exc_info.value.detail

    def test_siglip_rejects_cls(self):
        """SigLIP and SigLIP2 do not support CLS method."""
        with pytest.raises(HTTPException) as exc_info:
            validate_method("siglip2", "cls")
        assert exc_info.value.status_code == 400
        with pytest.raises(HTTPException) as exc_info:
            validate_method("siglip", "cls")
        assert exc_info.value.status_code == 400

    def test_siglip_accepts_mean(self):
        """SigLIP and SigLIP2 accept 'mean' method."""
        assert validate_method("siglip", "mean") == "mean"
        assert validate_method("siglip2", "mean") == "mean"

    def test_finetuned_variant_uses_base_method_compatibility(self):
        """Fine-tuned variants should validate methods like their base model."""
        assert validate_method("dinov2_finetuned", "rollout") == "rollout"
        assert validate_method("dinov2_finetuned_lora", "rollout") == "rollout"
        with pytest.raises(HTTPException) as exc_info:
            validate_method("siglip2_finetuned", "cls")
        assert exc_info.value.status_code == 400


# --- validate_model tests ---


class TestValidateModel:
    """Tests for validate_model return values and error handling."""

    def test_returns_resolved_name(self):
        """validate_model should return the resolved canonical model name."""
        assert validate_model("dinov2") == "dinov2"
        assert validate_model("siglip") == "siglip"
        assert validate_model("siglip2") == "siglip2"
        assert validate_model("dinov2_finetuned") == "dinov2_finetuned"
        assert validate_model("dinov2_finetuned_lora") == "dinov2_finetuned_lora"

    def test_invalid_model_raises(self):
        """Unknown model should raise HTTPException 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_model("nonexistent_model")
        assert exc_info.value.status_code == 400
        assert "Invalid model" in exc_info.value.detail

    def test_non_finetunable_model_variant_rejected(self):
        """Fine-tuned variant should be rejected for non-fine-tunable models."""
        with pytest.raises(HTTPException) as exc_info:
            validate_model("resnet50_finetuned")
        assert exc_info.value.status_code == 400
        assert "Fine-tuned variant not supported" in exc_info.value.detail

    def test_invalid_finetune_strategy_rejected(self):
        """Unknown strategy suffix should be rejected."""
        with pytest.raises(HTTPException) as exc_info:
            validate_model("dinov2_finetuned_unknown")
        assert exc_info.value.status_code == 400
        assert "Invalid fine-tuning strategy" in exc_info.value.detail


# --- validate_layer_for_model tests ---


class TestValidateLayerForModel:
    """Tests for validate_layer_for_model return values and error handling."""

    def test_returns_layer_key(self):
        """validate_layer_for_model should return the layer key string."""
        assert validate_layer_for_model(5, "dinov2") == "layer5"
        assert validate_layer_for_model(0, "dinov2") == "layer0"

    def test_out_of_bounds_raises(self):
        """Layer index >= num_layers should raise HTTPException 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_layer_for_model(999, "dinov2")
        assert exc_info.value.status_code == 400
        assert "Invalid layer" in exc_info.value.detail

    def test_negative_layer_raises(self):
        """Negative layer index should raise HTTPException 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_layer_for_model(-1, "dinov2")
        assert exc_info.value.status_code == 400

    def test_finetuned_variant_uses_base_layer_count(self):
        """Fine-tuned variants should validate layers against the base model config."""
        assert validate_layer_for_model(11, "dinov2_finetuned") == "layer11"
        assert validate_layer_for_model(11, "dinov2_finetuned_lora") == "layer11"
