"""Evaluation utilities for SSL attention analysis.

This module provides tools for validating SSL model features before
attention analysis:

- Linear Probe: Validates that frozen features are discriminative for
  architectural style classification
- Fine-tuning: Enables training SSL models on style classification to
  compare attention patterns before/after alignment

Example:
    >>> from ssl_attention.evaluation import train_linear_probe_sklearn
    >>> from ssl_attention.config import STYLE_NAMES
    >>>
    >>> # features: (N, D) tensor, labels: (N,) tensor
    >>> result = train_linear_probe_sklearn(features, labels, class_names=STYLE_NAMES)
    >>> print(f"CV Accuracy: {result.cv_mean:.1%} ± {result.cv_std:.1%}")
    >>>
    >>> # Expected: >50% accuracy (random = 25%)
    >>> # If accuracy is near random, attention analysis may not be meaningful

Fine-tuning example:
    >>> from ssl_attention.evaluation import FineTuningConfig, FineTuner, FineTunableModel
    >>> from ssl_attention.data import FullDataset
    >>>
    >>> config = FineTuningConfig(model_name="dinov2", num_epochs=10)
    >>> model = FineTunableModel("dinov2", freeze_backbone=False)
    >>> tuner = FineTuner(config)
    >>> result = tuner.train(model, dataset)
    >>> print(f"Best val accuracy: {result.best_val_acc:.1%}")
"""

from ssl_attention.evaluation.fine_tuning import (
    LORA_TARGET_MODULES,
    ClassificationHead,
    FineTunableModel,
    FineTuner,
    FineTuningConfig,
    FineTuningResult,
    get_checkpoint_candidates,
    get_finetuned_cache_key,
    infer_strategy_id,
    load_finetuned_model,
    load_run_manifest,
    save_training_results,
)
from ssl_attention.evaluation.linear_probe import (
    ProbeResult,
    compare_model_features,
    extract_features_for_probe,
    print_probe_summary,
    train_linear_probe_sklearn,
)

__all__ = [
    # Linear probe
    "ProbeResult",
    "train_linear_probe_sklearn",
    "extract_features_for_probe",
    "compare_model_features",
    "print_probe_summary",
    # Fine-tuning
    "LORA_TARGET_MODULES",
    "FineTuningConfig",
    "FineTuningResult",
    "ClassificationHead",
    "FineTunableModel",
    "FineTuner",
    "infer_strategy_id",
    "get_checkpoint_candidates",
    "get_finetuned_cache_key",
    "load_run_manifest",
    "load_finetuned_model",
    "save_training_results",
]
