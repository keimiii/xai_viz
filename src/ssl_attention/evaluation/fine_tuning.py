"""Fine-tuning module for SSL models on WikiChurches style classification.

This module enables fine-tuning SSL ViT models (DINOv2, DINOv3, MAE, CLIP, SigLIP, SigLIP2)
on the 4-class architectural style classification task. After fine-tuning, attention
patterns can be compared before/after to measure alignment shift.

Key classes:
- FineTuningConfig: Hyperparameters and settings
- FineTuningResult: Training metrics and checkpoint path
- ClassificationHead: Linear classifier on CLS token
- FineTunableModel: Wraps SSL backbone + classification head
- FineTuner: Training orchestrator

Example:
    >>> from ssl_attention.evaluation.fine_tuning import (
    ...     FineTuningConfig, FineTuner, FineTunableModel
    ... )
    >>> from ssl_attention.data import FullDataset
    >>> from ssl_attention.config import DATASET_PATH
    >>>
    >>> config = FineTuningConfig(model_name="dinov2", num_epochs=10)
    >>> model = FineTunableModel(config.model_name, freeze_backbone=False)
    >>> dataset = FullDataset(DATASET_PATH, filter_labeled=True)
    >>>
    >>> tuner = FineTuner(config)
    >>> result = tuner.train(model, dataset)
    >>> print(f"Best validation accuracy: {result.best_val_acc:.1%}")
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from transformers import AutoImageProcessor, get_cosine_schedule_with_warmup

from ssl_attention.config import (
    ANNOTATIONS_PATH,
    DATASET_PATH,
    FINETUNE_MODELS,
    FINETUNE_STRATEGIES,
    MODELS,
    NUM_STYLES,
    STYLE_NAMES,
    FineTuningStrategy,
)
from ssl_attention.evaluation.fine_tuning_artifacts import (
    CHECKPOINTS_ROOT,
    RESULTS_ROOT,
    TRAINING_GIT_COMMIT_SHA_FIELD,
    ExperimentPaths,
    build_dataset_version_hint,
    build_run_id,
    ensure_experiment_layout,
    get_experiment_paths,
    get_git_commit_sha,
    load_active_experiment,
    load_json,
    load_run_matrix,
    make_experiment_id,
    normalize_fine_tuning_results_payload,
    normalize_run_manifest_payload,
    repo_relative_path,
    save_json,
    save_run_matrix,
    write_active_experiment,
)
from ssl_attention.models.base import forward_mae_for_analysis
from ssl_attention.models.protocols import ModelOutput
from ssl_attention.utils.device import clear_memory, get_device

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ssl_attention.data.wikichurches import FullDataset


# =============================================================================
# Output Directories
# =============================================================================

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CHECKPOINTS_PATH = CHECKPOINTS_ROOT
RESULTS_PATH = RESULTS_ROOT
LEGACY_MANIFESTS_PATH = RESULTS_PATH / "fine_tuning_manifests"

# LoRA target modules per model (HF attention projection names differ)
LORA_TARGET_MODULES: dict[str, list[str]] = {
    "dinov2": ["query", "value"],
    "dinov3": ["q_proj", "v_proj"],
    "mae": ["query", "value"],
    "clip": ["q_proj", "v_proj"],
    "siglip": ["q_proj", "v_proj"],
    "siglip2": ["q_proj", "v_proj"],
}

_LEGACY_FINETUNED_SUFFIX = "_finetuned.pt"

PRIMARY_SPLIT_POLICY = "random_stratified_excluding_annotated_eval"
EXPLORATORY_SPLIT_POLICY = "annotated_eval_validation"
PRIMARY_SELECTION_METRIC = "classification_validation_accuracy"


@dataclass
class FineTuningSplitArtifact:
    """Shared train/validation split provenance for an experiment batch."""

    split_id: str
    experiment_id: str
    seed: int
    dataset_root: str
    dataset_version_hint: dict[str, Any]
    policy: str
    exclude_annotated_from_train: bool
    exclude_annotated_from_val: bool
    annotated_eval_image_ids: list[str]
    train_image_ids: list[str]
    val_image_ids: list[str]
    train_class_counts: dict[str, int]
    val_class_counts: dict[str, int]
    created_at: str


def build_split_id(experiment_id: str, *, seed: int, exploratory: bool) -> str:
    """Build a stable split identifier inside one experiment batch."""
    scope = "exploratory" if exploratory else "primary"
    return f"{experiment_id}__{scope}__seed{seed}"


def _style_count_dict() -> dict[str, int]:
    """Create an empty style-count dictionary keyed by style name."""
    return {style_name: 0 for style_name in STYLE_NAMES}


def _labels_to_style_counts(labels: list[int]) -> dict[str, int]:
    """Convert integer style labels into a readable count mapping."""
    counts = _style_count_dict()
    for label in labels:
        counts[STYLE_NAMES[label]] += 1
    return counts


def infer_strategy_id(*, freeze_backbone: bool, use_lora: bool) -> str:
    """Infer strategy ID from training flags."""
    if use_lora:
        return FineTuningStrategy.LORA.value
    if freeze_backbone:
        return FineTuningStrategy.LINEAR_PROBE.value
    return FineTuningStrategy.FULL.value


def get_checkpoint_filename(model_name: str, strategy_id: str) -> str:
    """Get strategy-aware checkpoint filename."""
    return f"{model_name}_{strategy_id}_finetuned.pt"


def get_finetuned_cache_key(model_name: str, strategy_id: str | None = None) -> str:
    """Get cache key for fine-tuned model attention/heatmaps."""
    if strategy_id is None:
        return f"{model_name}_finetuned"
    return f"{model_name}_finetuned_{strategy_id}"


def strategy_uses_legacy_checkpoint_fallback(strategy_id: str | None) -> bool:
    """Return whether an explicit strategy may reuse a legacy checkpoint name."""
    return strategy_id == FineTuningStrategy.FULL.value


def get_checkpoint_candidates(
    model_name: str,
    strategy_id: str | None = None,
    experiment_id: str | None = None,
) -> list[Path]:
    """Return checkpoint candidates in preference order.

    Prefers experiment-scoped checkpoint directories when an explicit
    `experiment_id` is provided or an active experiment is configured.
    Falls back to the legacy top-level checkpoint layout for compatibility.
    """
    candidates: list[Path] = []
    checkpoint_dirs: list[Path] = []
    resolved_experiment_id = experiment_id
    if resolved_experiment_id is None:
        active = load_active_experiment()
        if active is not None:
            resolved_experiment_id = active.get("experiment_id")
    if resolved_experiment_id:
        checkpoint_dirs.append(get_experiment_paths(resolved_experiment_id).checkpoints_dir)
    checkpoint_dirs.append(CHECKPOINTS_PATH)

    if strategy_id is None:
        for checkpoint_dir in checkpoint_dirs:
            for strategy in (
                FineTuningStrategy.LORA.value,
                FineTuningStrategy.FULL.value,
                FineTuningStrategy.LINEAR_PROBE.value,
            ):
                candidates.append(checkpoint_dir / get_checkpoint_filename(model_name, strategy))
            candidates.append(checkpoint_dir / f"{model_name}{_LEGACY_FINETUNED_SUFFIX}")
        return candidates

    for checkpoint_dir in checkpoint_dirs:
        candidates.append(checkpoint_dir / get_checkpoint_filename(model_name, strategy_id))
        if strategy_uses_legacy_checkpoint_fallback(strategy_id):
            candidates.append(checkpoint_dir / f"{model_name}{_LEGACY_FINETUNED_SUFFIX}")
    return candidates


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning an SSL model.

    Attributes:
        model_name: Name of the SSL model to fine-tune (dinov2, dinov3, mae, clip, siglip, siglip2).
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate_backbone: Learning rate for backbone parameters.
        learning_rate_head: Learning rate for classification head.
        weight_decay: Weight decay for AdamW optimizer.
        freeze_backbone: If True, only train classification head (linear probe).
        val_split: Fraction of data to use for validation.
        exclude_annotated_eval: If True, exclude bbox-annotated images from
            training/validation splits to avoid data leakage in Q2 evaluation.
        val_on_annotated_eval: If True, use bbox-annotated images as the
            validation set and train on remaining labeled images. This is an
            exploratory mode because it reuses the annotated evaluation pool for
            checkpoint selection.
        seed: Random seed for reproducibility.
        experiment_id: Experiment-batch identifier for coordinated runs.
        split_id: Shared split artifact identifier reused across runs in the
            same experiment batch.
        split_artifact_path: Optional repo-relative split artifact path.
        run_scope: "primary" for the main experiment path and "exploratory" for
            non-primary modes such as annotated-eval validation.
        checkpoint_selection_metric: Criterion used for checkpoint selection.
        checkpoint_selection_split: Human-readable identifier for the selection
            split used during training.
        warmup_ratio: Fraction of total training steps for linear LR warmup.
        max_grad_norm: Maximum gradient norm for clipping (0 disables clipping).
        use_augmentation: Whether to apply training data augmentations.
        use_lora: If True, apply LoRA adapters to backbone attention layers.
        lora_rank: Rank of LoRA decomposition matrices.
        lora_alpha: Scaling factor for LoRA (effective scale = alpha / rank).
        lora_dropout: Dropout probability for LoRA layers.
        lora_target_modules: Attention modules to adapt. Auto-resolved per model if None.
    """

    model_name: str
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate_backbone: float = 1e-5
    learning_rate_head: float = 1e-3
    weight_decay: float = 0.01
    freeze_backbone: bool = False
    val_split: float = 0.2
    exclude_annotated_eval: bool = True
    val_on_annotated_eval: bool = False
    seed: int = 42
    experiment_id: str = field(default_factory=make_experiment_id)
    split_id: str | None = None
    split_artifact_path: str | None = None
    run_scope: Literal["primary", "exploratory"] = "primary"
    checkpoint_selection_metric: str = PRIMARY_SELECTION_METRIC
    checkpoint_selection_split: str | None = None
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_augmentation: bool = True
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.model_name not in FINETUNE_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' is not fine-tunable. "
                f"Available fine-tunable models: {sorted(FINETUNE_MODELS)}."
            )
        if self.use_lora and self.freeze_backbone:
            raise ValueError(
                "use_lora=True and freeze_backbone=True are conflicting strategies. "
                "LoRA fine-tunes backbone adapters; freeze_backbone disables all backbone training."
            )
        # Auto-adjust backbone LR for LoRA when user hasn't changed the default
        if self.use_lora and self.learning_rate_backbone == 1e-5:
            self.learning_rate_backbone = 1e-4

        valid_strategies = {s.value for s in FINETUNE_STRATEGIES}
        if self.strategy_id not in valid_strategies:
            raise ValueError(f"Invalid strategy_id: {self.strategy_id}. Valid: {sorted(valid_strategies)}")
        if self.val_on_annotated_eval and not self.exclude_annotated_eval:
            raise ValueError(
                "val_on_annotated_eval=True requires exclude_annotated_eval=True "
                "so annotated eval images are not used in training."
            )
        if self.val_on_annotated_eval:
            self.run_scope = "exploratory"
        if self.split_id is None:
            self.split_id = build_split_id(
                self.experiment_id,
                seed=self.seed,
                exploratory=self.val_on_annotated_eval,
            )
        if self.checkpoint_selection_split is None:
            self.checkpoint_selection_split = (
                EXPLORATORY_SPLIT_POLICY if self.val_on_annotated_eval else PRIMARY_SPLIT_POLICY
            )

    @property
    def strategy_id(self) -> str:
        """Fine-tuning strategy identifier inferred from config flags."""
        return infer_strategy_id(
            freeze_backbone=self.freeze_backbone,
            use_lora=self.use_lora,
        )

    @property
    def run_id(self) -> str:
        """Stable experiment-scoped run identifier."""
        return build_run_id(self.experiment_id, self.model_name, self.strategy_id)


@dataclass
class FineTuningResult:
    """Result of fine-tuning an SSL model.

    Attributes:
        model_name: Name of the fine-tuned model.
        best_val_acc: Best validation accuracy achieved.
        best_epoch: Epoch at which best validation accuracy occurred.
        train_history: List of dicts with per-epoch metrics.
        checkpoint_path: Path to saved checkpoint.
        config: The configuration used for training.
    """

    model_name: str
    strategy_id: str
    best_val_acc: float
    best_epoch: int
    train_history: list[dict[str, float]]
    checkpoint_path: Path
    manifest_path: Path
    experiment_id: str
    run_id: str
    split_id: str
    split_artifact_path: Path
    run_scope: Literal["primary", "exploratory"]
    training_git_commit_sha: str | None = None
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_excluded_eval_samples: int = 0
    config: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Model Components
# =============================================================================


class ClassificationHead(nn.Module):
    """Linear classification head for SSL features.

    Takes CLS token (or pooled features) and produces class logits.

    Args:
        embed_dim: Dimension of input features (typically 768).
        num_classes: Number of output classes.
        dropout: Dropout probability before classifier.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = NUM_STYLES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass through classification head.

        Args:
            features: CLS token features of shape (B, embed_dim).

        Returns:
            Logits of shape (B, num_classes).
        """
        result: Tensor = self.head(features)
        return result


class FineTunableModel(nn.Module):
    """Wraps an SSL backbone with a classification head for fine-tuning.

    Supports all 6 ViT SSL models with unified interface:
    - DINOv2: Uses CLS token from dinov2-with-registers
    - DINOv3: Uses CLS token from dinov3 with RoPE
    - MAE: Uses CLS token with all patches visible and deterministic analysis ordering
    - CLIP: Uses CLS token from vision encoder
    - SigLIP: Uses pooler_output (no CLS token in sequence)

    Args:
        model_name: Name of the SSL backbone.
        num_classes: Number of classification classes.
        freeze_backbone: If True, freeze backbone weights.
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses float32 for MPS compatibility.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = NUM_STYLES,
        freeze_backbone: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.device = device or get_device()
        # MPS compatibility: use float32
        self.dtype = dtype or torch.float32
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

        # Load model configuration
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
        if model_name not in FINETUNE_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported by FineTunableModel. "
                f"Available fine-tunable models: {sorted(FINETUNE_MODELS)}."
            )
        self._config = MODELS[model_name]

        # Load backbone and processor
        self.backbone = self._load_backbone()
        self.processor = AutoImageProcessor.from_pretrained(self._config.model_id)

        # Apply LoRA adapters before moving to device
        if use_lora:
            self._apply_lora()

        # Create classification head
        self.classifier = ClassificationHead(
            embed_dim=self._config.embed_dim,
            num_classes=num_classes,
        )

        # Move to device
        self.backbone = self.backbone.to(device=self.device, dtype=self.dtype)
        self.classifier = self.classifier.to(device=self.device, dtype=self.dtype)

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _load_backbone(self) -> nn.Module:
        """Load the SSL backbone model.

        Returns:
            The backbone model with output_attentions enabled.
        """
        from transformers import (
            AutoConfig,
            AutoModel,
            CLIPVisionConfig,
            CLIPVisionModel,
            Siglip2VisionConfig,
            Siglip2VisionModel,
            SiglipVisionConfig,
            SiglipVisionModel,
            ViTMAEConfig,
            ViTMAEModel,
        )

        model_id = self._config.model_id

        if self.model_name == "mae":
            mae_config = ViTMAEConfig.from_pretrained(model_id)
            mae_config.output_attentions = True
            mae_config.mask_ratio = 0.0  # Keep all patches visible; analysis adds deterministic noise
            return ViTMAEModel.from_pretrained(model_id, config=mae_config)

        elif self.model_name == "clip":
            clip_config = CLIPVisionConfig.from_pretrained(model_id)
            clip_config.output_attentions = True
            return CLIPVisionModel.from_pretrained(model_id, config=clip_config)

        elif self.model_name == "siglip":
            siglip_config = SiglipVisionConfig.from_pretrained(model_id)
            siglip_config.output_attentions = True
            return SiglipVisionModel.from_pretrained(model_id, config=siglip_config)

        elif self.model_name == "siglip2":
            auto_config = AutoConfig.from_pretrained(model_id)
            model_type = getattr(auto_config, "model_type", None)
            if model_type == "siglip2":
                siglip2_config = Siglip2VisionConfig.from_pretrained(model_id)
                siglip2_config.output_attentions = True
                return Siglip2VisionModel.from_pretrained(model_id, config=siglip2_config)
            if model_type == "siglip":
                siglip_config = SiglipVisionConfig.from_pretrained(model_id)
                siglip_config.output_attentions = True
                return SiglipVisionModel.from_pretrained(model_id, config=siglip_config)
            raise ValueError(
                f"Unsupported SigLIP2 checkpoint type '{model_type}' for {model_id}"
            )

        else:  # dinov2, dinov3
            auto_config = AutoConfig.from_pretrained(model_id)
            auto_config.output_attentions = True
            model: nn.Module = AutoModel.from_pretrained(
                model_id, config=auto_config, trust_remote_code=True
            )
            return model

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to backbone attention layers.

        Uses HuggingFace PEFT to wrap the backbone with low-rank adapters.
        PEFT auto-freezes non-LoRA backbone params, so get_optimizer_param_groups()
        (which filters by requires_grad) picks up only LoRA + head params.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "LoRA requires the 'peft' package. Install it with: pip install peft>=0.7.0"
            ) from None

        # Resolve target modules
        target_modules = self.lora_target_modules
        if target_modules is None:
            if self.model_name not in LORA_TARGET_MODULES:
                raise ValueError(
                    f"No default LoRA target modules for '{self.model_name}'. "
                    f"Available: {list(LORA_TARGET_MODULES.keys())}. "
                    f"Provide lora_target_modules explicitly."
                )
            target_modules = LORA_TARGET_MODULES[self.model_name]

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.backbone = get_peft_model(self.backbone, lora_config)  # type: ignore[arg-type]

        # Log trainable parameter count
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(
            f"LoRA applied: {trainable:,} trainable params / {total:,} total "
            f"({trainable / total:.2%})"
        )

    def _extract_features(self, model_output: Any) -> tuple[Tensor, list[Tensor]]:
        """Extract CLS features and attention weights from backbone output.

        Args:
            model_output: Raw output from backbone forward pass.

        Returns:
            Tuple of (cls_features, attention_weights).
        """
        attentions = list(model_output.attentions)

        if self.model_name in ("siglip", "siglip2"):
            # SigLIP/SigLIP2 use pooler_output (no CLS in sequence)
            cls_features = model_output.pooler_output
        else:
            # DINOv2, DINOv3, MAE, CLIP all have CLS at position 0
            cls_features = model_output.last_hidden_state[:, 0, :]

        return cls_features, attentions

    def preprocess(self, images: list) -> Tensor:
        """Preprocess PIL images for model input.

        Args:
            images: List of PIL Images.

        Returns:
            Tensor of shape (B, C, H, W) on the model's device.
        """
        processed = self.processor(images=images, return_tensors="pt")
        pixel_values: Tensor = processed["pixel_values"]
        return pixel_values.to(device=self.device, dtype=self.dtype)

    def _forward_backbone_for_analysis(
        self,
        pixel_values: Tensor,
        *,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Any:
        """Run an analysis forward pass with deterministic MAE behavior.

        HuggingFace MAE still permutes patches when analysis forwards omit
        explicit `noise`, even in eval mode with `mask_ratio=0.0`.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.inference_mode():
                if self.model_name == "mae":
                    return forward_mae_for_analysis(
                        self.backbone,
                        pixel_values,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                forward_kwargs: dict[str, bool] = {}
                if output_attentions:
                    forward_kwargs["output_attentions"] = True
                if output_hidden_states:
                    forward_kwargs["output_hidden_states"] = True
                return self.backbone(
                    pixel_values=pixel_values,
                    **forward_kwargs,
                )
        finally:
            if was_training:
                self.backbone.train()

    def forward(
        self, pixel_values: Tensor
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Forward pass through backbone and classifier.

        Args:
            pixel_values: Preprocessed images of shape (B, C, H, W).

        Returns:
            Tuple of (logits, cls_features, attention_weights).
            - logits: Classification logits of shape (B, num_classes).
            - cls_features: CLS token features of shape (B, embed_dim).
            - attention_weights: List of attention tensors per layer.
        """
        # Forward through backbone
        model_output = self.backbone(
            pixel_values=pixel_values,
            output_attentions=True,
        )

        # Extract features and attention
        cls_features, attention_weights = self._extract_features(model_output)

        # Classify
        logits = self.classifier(cls_features)

        return logits, cls_features, attention_weights

    def extract_attention(self, pixel_values: Tensor) -> ModelOutput:
        """Extract attention in the same format as base models.

        This allows fine-tuned models to be used with existing
        attention analysis metrics (IoU, pointing accuracy).

        Args:
            pixel_values: Preprocessed images of shape (B, C, H, W).

        Returns:
            ModelOutput compatible with attention analysis.
        """
        model_output = self._forward_backbone_for_analysis(
            pixel_values,
            output_attentions=True,
        )

        hidden_states = model_output.last_hidden_state
        attentions = list(model_output.attentions)

        if self.model_name in ("siglip", "siglip2"):
            cls_token = model_output.pooler_output
            patch_tokens = hidden_states  # All positions are patches
        else:
            cls_token = hidden_states[:, 0, :]
            # Skip CLS + registers for patch tokens
            patch_start = 1 + self._config.num_registers
            patch_tokens = hidden_states[:, patch_start:, :]

        return ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            attention_weights=attentions,
        )

    def get_optimizer_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Get parameter groups with differential learning rates.

        Args:
            lr_backbone: Learning rate for backbone (smaller to preserve features).
            lr_head: Learning rate for classification head (larger for fast learning).
            weight_decay: Weight decay coefficient.

        Returns:
            List of parameter group dicts for optimizer.
        """
        param_groups = []

        # Backbone parameters (if not frozen)
        if not self.freeze_backbone:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
            if backbone_params:
                param_groups.append({
                    "params": backbone_params,
                    "lr": lr_backbone,
                    "weight_decay": weight_decay,
                })

        # Head parameters
        head_params = list(self.classifier.parameters())
        param_groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
        })

        return param_groups


# =============================================================================
# Training
# =============================================================================


class FineTuner:
    """Training orchestrator for fine-tuning SSL models.

    Handles the full training loop including:
    - Train/validation split (stratified)
    - Class-weighted loss for imbalanced data
    - Cosine LR schedule with linear warmup
    - Gradient clipping for training stability
    - Data augmentation (crop, flip, color jitter)
    - Checkpoint saving (model, optimizer, scheduler)
    - Training history logging

    Args:
        config: Fine-tuning configuration.

    Example:
        >>> config = FineTuningConfig(model_name="dinov2", num_epochs=10)
        >>> tuner = FineTuner(config)
        >>> model = FineTunableModel("dinov2")
        >>> result = tuner.train(model, dataset)
    """

    def __init__(self, config: FineTuningConfig) -> None:
        self.config = config
        self.device = get_device()

        # Set seeds for reproducibility
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _compute_class_weights(self, labels: list[int]) -> Tensor:
        """Compute inverse frequency class weights for imbalanced data.

        Args:
            labels: List of class labels.

        Returns:
            Tensor of class weights normalized to sum to num_classes.
        """
        label_counts = np.bincount(labels, minlength=NUM_STYLES)
        # Inverse frequency weighting
        weights = 1.0 / (label_counts + 1e-6)
        # Normalize to sum to num_classes
        weights = weights / weights.sum() * NUM_STYLES
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _stratified_split(
        self,
        dataset: FullDataset,
        val_split: float,
        exclude_image_ids: set[str] | None = None,
    ) -> tuple[list[int], list[int], int]:
        """Split dataset into train/val with stratification.

        Args:
            dataset: Full dataset with style_label in samples.
            val_split: Fraction for validation.
            exclude_image_ids: Image IDs to exclude from both splits (e.g.,
                bbox-annotated evaluation images to prevent data leakage).

        Returns:
            Tuple of (train_indices, val_indices, n_excluded).
        """
        # Get all labels, skipping excluded images
        all_labels = []
        n_excluded = 0
        for i in range(len(dataset)):
            sample = dataset.get_metadata(i)
            if exclude_image_ids and sample["image_id"] in exclude_image_ids:
                n_excluded += 1
                continue
            label = sample["style_label"]
            if label is not None:
                all_labels.append((i, label))

        # Group by label
        label_to_indices: dict[int, list[int]] = {}
        for idx, label in all_labels:
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        # Split each class
        train_indices = []
        val_indices = []
        rng = random.Random(self.config.seed)

        for _label, indices in label_to_indices.items():
            rng.shuffle(indices)
            n_val = max(1, int(len(indices) * val_split))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        return train_indices, val_indices, n_excluded

    def _collect_labels_for_indices(
        self,
        dataset: FullDataset,
        indices: Sequence[int],
    ) -> list[int]:
        """Collect labeled subset targets without decoding images."""
        labels: list[int] = []
        for idx in indices:
            label = dataset.get_metadata(idx)["style_label"]
            if label is not None:
                labels.append(label)
        return labels

    def _split_with_annotated_eval_validation(
        self,
        dataset: FullDataset,
        eval_image_ids: set[str],
    ) -> tuple[list[int], list[int], int]:
        """Use annotated images as fixed validation set; others for training.

        Args:
            dataset: Full dataset with style labels.
            eval_image_ids: Image IDs in annotated subset.

        Returns:
            Tuple of (train_indices, val_indices, n_val_annotated).
        """
        train_indices: list[int] = []
        val_indices: list[int] = []

        for i in range(len(dataset)):
            sample = dataset.get_metadata(i)
            if sample["style_label"] is None:
                continue
            if sample["image_id"] in eval_image_ids:
                val_indices.append(i)
            else:
                train_indices.append(i)

        return train_indices, val_indices, len(val_indices)

    def _build_image_index_map(self, dataset: FullDataset) -> dict[str, int]:
        """Map image IDs to dataset indices without loading image bytes."""
        return {
            dataset.get_metadata(index)["image_id"]: index
            for index in range(len(dataset))
        }

    def _create_split_artifact(
        self,
        dataset: FullDataset,
        eval_image_ids: set[str],
        split_artifact_path: Path,
    ) -> FineTuningSplitArtifact:
        """Create and persist a shared split artifact for this experiment batch."""
        if self.config.val_on_annotated_eval:
            train_indices, val_indices, _ = self._split_with_annotated_eval_validation(
                dataset,
                eval_image_ids,
            )
            policy = EXPLORATORY_SPLIT_POLICY
            exclude_annotated_from_val = False
        else:
            train_indices, val_indices, _ = self._stratified_split(
                dataset,
                self.config.val_split,
                exclude_image_ids=eval_image_ids,
            )
            policy = PRIMARY_SPLIT_POLICY
            exclude_annotated_from_val = True

        train_image_ids = [dataset.get_metadata(index)["image_id"] for index in train_indices]
        val_image_ids = [dataset.get_metadata(index)["image_id"] for index in val_indices]
        train_labels = self._collect_labels_for_indices(dataset, train_indices)
        val_labels = self._collect_labels_for_indices(dataset, val_indices)

        artifact = FineTuningSplitArtifact(
            split_id=self.config.split_id or build_split_id(
                self.config.experiment_id,
                seed=self.config.seed,
                exploratory=self.config.val_on_annotated_eval,
            ),
            experiment_id=self.config.experiment_id,
            seed=self.config.seed,
            dataset_root=repo_relative_path(DATASET_PATH),
            dataset_version_hint=build_dataset_version_hint(DATASET_PATH),
            policy=policy,
            exclude_annotated_from_train=self.config.exclude_annotated_eval,
            exclude_annotated_from_val=exclude_annotated_from_val,
            annotated_eval_image_ids=sorted(eval_image_ids),
            train_image_ids=train_image_ids,
            val_image_ids=val_image_ids,
            train_class_counts=_labels_to_style_counts(train_labels),
            val_class_counts=_labels_to_style_counts(val_labels),
            created_at=datetime.now(UTC).isoformat(),
        )
        save_json(split_artifact_path, asdict(artifact))
        return artifact

    def _load_or_create_split_artifact(
        self,
        dataset: FullDataset,
        eval_image_ids: set[str],
        split_artifact_path: Path,
    ) -> FineTuningSplitArtifact:
        """Load an existing shared split artifact or create it once."""
        if split_artifact_path.exists():
            with open(split_artifact_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            return FineTuningSplitArtifact(**payload)
        return self._create_split_artifact(dataset, eval_image_ids, split_artifact_path)

    def _subset_from_image_ids(
        self,
        dataset: FullDataset,
        image_ids: list[str],
        *,
        image_index_map: dict[str, int],
    ) -> Subset:
        """Build a subset from split-artifact image IDs."""
        indices = [image_index_map[image_id] for image_id in image_ids]
        return Subset(dataset, indices)

    def _collate_fn(self, batch: list[dict]) -> dict[str, Any]:
        """Collate function for DataLoader.

        Filters out samples with None labels and returns images as list.
        """
        # Filter samples with valid labels
        valid_batch = [s for s in batch if s.get("style_label") is not None]
        if not valid_batch:
            # Return empty batch markers
            return {"images": [], "labels": torch.tensor([], dtype=torch.long)}

        return {
            "images": [s["image"] for s in valid_batch],
            "labels": torch.tensor(
                [s["style_label"] for s in valid_batch], dtype=torch.long
            ),
        }

    def _build_train_transform(self) -> T.Compose:
        """Build data augmentation transform for training images.

        Returns a transform that applies random crop, flip, and color jitter.
        Outputs PIL images — model.preprocess() handles ToTensor/Normalize.
        """
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    @staticmethod
    def _augment_train_images(
        images: list, transform: T.Compose
    ) -> list:
        """Apply augmentation transform to a list of PIL images."""
        return [transform(img) for img in images]

    def train(
        self,
        model: FineTunableModel,
        dataset: FullDataset,
    ) -> FineTuningResult:
        """Train the model on the dataset.

        Uses cosine LR schedule with linear warmup, gradient clipping,
        and optional data augmentation for training stability.

        Args:
            model: FineTunableModel to train.
            dataset: FullDataset with labeled images.

        Returns:
            FineTuningResult with training metrics and checkpoint path.
        """
        # Create experiment-scoped output directories
        experiment_paths = ensure_experiment_layout(self.config.experiment_id)

        # Optionally exclude bbox-annotated image IDs to avoid data leakage.
        eval_image_ids: set[str] = set()
        if self.config.exclude_annotated_eval or self.config.val_on_annotated_eval:
            from ssl_attention.data import load_annotations

            annotations = load_annotations(ANNOTATIONS_PATH)
            eval_image_ids = set(annotations.keys())

        split_artifact_path = experiment_paths.split_artifacts_dir / f"{self.config.split_id}.json"
        split_artifact = self._load_or_create_split_artifact(
            dataset,
            eval_image_ids,
            split_artifact_path,
        )
        self.config.split_artifact_path = repo_relative_path(split_artifact_path)

        image_index_map = self._build_image_index_map(dataset)
        train_subset = self._subset_from_image_ids(
            dataset,
            split_artifact.train_image_ids,
            image_index_map=image_index_map,
        )
        val_subset = self._subset_from_image_ids(
            dataset,
            split_artifact.val_image_ids,
            image_index_map=image_index_map,
        )
        n_excluded = len(split_artifact.annotated_eval_image_ids)

        if split_artifact.policy == EXPLORATORY_SPLIT_POLICY:
            print("Validation set source: annotated eval images (exploratory mode)")
            print(
                f"Exploratory run: reusing {len(split_artifact.val_image_ids)} annotated "
                "images for checkpoint selection."
            )
        else:
            print(
                f"Primary run: excluded {len(split_artifact.annotated_eval_image_ids)} "
                "bbox-annotated eval images from train/validation."
            )
            print(
                f"Reusing shared validation split {split_artifact.split_id} "
                f"({len(split_artifact.val_image_ids)} val images)."
            )
        print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}")

        if len(train_subset) == 0 or len(val_subset) == 0:
            raise ValueError(
                "Train/validation split produced an empty subset. "
                "For full-dataset fine-tuning, ensure dataset/images includes non-annotated labeled images. "
                "For annotated-only runs, use exclude_annotated_eval=False."
            )

        # Build augmentation transform
        train_transform = self._build_train_transform() if self.config.use_augmentation else None

        # Create data loaders (num_workers=0 for MPS compatibility)
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
        )

        # Compute class weights from training data
        train_labels = self._collect_labels_for_indices(dataset, train_subset.indices)
        class_weights = self._compute_class_weights(train_labels)
        print(f"Class weights: {class_weights.tolist()}")

        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer with differential learning rates
        param_groups = model.get_optimizer_param_groups(
            lr_backbone=self.config.learning_rate_backbone,
            lr_head=self.config.learning_rate_head,
            weight_decay=self.config.weight_decay,
        )
        optimizer = AdamW(param_groups)

        # LR scheduler: cosine decay with linear warmup
        num_training_steps = self.config.num_epochs * len(train_loader)
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training loop
        train_history: list[dict[str, float]] = []
        best_val_acc = 0.0
        best_epoch = 0
        strategy_id = self.config.strategy_id
        checkpoint_path = experiment_paths.checkpoints_dir / get_checkpoint_filename(
            self.config.model_name,
            strategy_id,
        )

        for epoch in range(self.config.num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if len(batch["images"]) == 0:
                    continue

                # Apply augmentation (training only)
                images = batch["images"]
                if train_transform is not None:
                    images = self._augment_train_images(images, train_transform)

                # Preprocess and forward
                pixel_values = model.preprocess(images)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits, _, _ = model(pixel_values)
                loss = criterion(logits, labels)
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.config.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += len(labels)

                # MPS memory management
                if self.device.type == "mps":
                    torch.mps.empty_cache()

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch["images"]) == 0:
                        continue

                    pixel_values = model.preprocess(batch["images"])
                    labels = batch["labels"].to(self.device)

                    logits, _, _ = model(pixel_values)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * len(labels)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += len(labels)

            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            # Log epoch
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
            train_history.append(epoch_metrics)

            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.1%}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.1%}, "
                f"lr={current_lr:.2e}"
            )

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, val_acc, checkpoint_path
                )
                print(f"  -> New best! Saved checkpoint to {checkpoint_path}")

        # Final cleanup
        clear_memory()

        manifest_path = self._save_run_manifest(
            experiment_paths=experiment_paths,
            checkpoint_path=checkpoint_path,
            split_artifact=split_artifact,
            strategy_id=strategy_id,
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
            num_train_samples=len(train_subset),
            num_val_samples=len(val_subset),
            num_excluded_eval_samples=n_excluded,
        )

        return FineTuningResult(
            model_name=self.config.model_name,
            strategy_id=strategy_id,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            train_history=train_history,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            experiment_id=self.config.experiment_id,
            run_id=self.config.run_id,
            split_id=split_artifact.split_id,
            split_artifact_path=split_artifact_path,
            run_scope=self.config.run_scope,
            training_git_commit_sha=get_git_commit_sha(),
            num_train_samples=len(train_subset),
            num_val_samples=len(val_subset),
            num_excluded_eval_samples=n_excluded,
            config=asdict(self.config),
        )

    def _save_checkpoint(
        self,
        model: FineTunableModel,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        val_acc: float,
        path: Path,
    ) -> None:
        """Save model checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            scheduler: LR scheduler state to save.
            epoch: Current epoch number.
            val_acc: Validation accuracy at this checkpoint.
            path: Path to save checkpoint.
        """
        checkpoint = {
            "model_name": self.config.model_name,
            "strategy_id": self.config.strategy_id,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_accuracy": val_acc,
            "config": asdict(self.config),
        }
        torch.save(checkpoint, path)

    def _save_run_manifest(
        self,
        experiment_paths: ExperimentPaths,
        checkpoint_path: Path,
        split_artifact: FineTuningSplitArtifact,
        strategy_id: str,
        best_epoch: int,
        best_val_acc: float,
        num_train_samples: int,
        num_val_samples: int,
        num_excluded_eval_samples: int,
    ) -> Path:
        """Persist run manifest for reproducibility and downstream analysis."""
        manifest_path = experiment_paths.manifests_dir / f"{self.config.run_id}_manifest.json"
        training_git_commit_sha = get_git_commit_sha()
        manifest = {
            "run_id": self.config.run_id,
            "experiment_id": self.config.experiment_id,
            "run_scope": self.config.run_scope,
            "model": self.config.model_name,
            "strategy": strategy_id,
            "split_id": split_artifact.split_id,
            "seed": self.config.seed,
            "epochs": self.config.num_epochs,
            "checkpoint_path": repo_relative_path(checkpoint_path),
            "manifest_path": repo_relative_path(manifest_path),
            "split_artifact_path": repo_relative_path(
                experiment_paths.split_artifacts_dir / f"{split_artifact.split_id}.json"
            ),
            TRAINING_GIT_COMMIT_SHA_FIELD: training_git_commit_sha,
            "checkpoint_selection_metric": self.config.checkpoint_selection_metric,
            "checkpoint_selection_split": self.config.checkpoint_selection_split,
            "selected_epoch": best_epoch,
            "best_val_score": best_val_acc,
            "split": {
                "train_samples": num_train_samples,
                "val_samples": num_val_samples,
                "excluded_eval_samples": num_excluded_eval_samples,
                "val_split": self.config.val_split,
                "val_source": split_artifact.policy,
            },
        }
        save_json(manifest_path, normalize_run_manifest_payload(manifest, drop_legacy=True))
        return manifest_path


# =============================================================================
# Loading Fine-tuned Models
# =============================================================================


def load_finetuned_model(
    model_name: str,
    checkpoint_path: Path | None = None,
    strategy_id: str | None = None,
    experiment_id: str | None = None,
    device: torch.device | None = None,
) -> FineTunableModel:
    """Load a fine-tuned model from checkpoint.

    Args:
        model_name: Name of the model (dinov2, dinov3, mae, clip, siglip, siglip2).
        checkpoint_path: Path to checkpoint. If None, uses default path.
        strategy_id: Optional strategy selector ("linear_probe", "lora", "full").
            If omitted, uses first available checkpoint in default priority order.
        experiment_id: Optional experiment batch identifier used for
            experiment-scoped checkpoint discovery.
        device: Target device. Auto-detects if None.

    Returns:
        FineTunableModel loaded with fine-tuned weights.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
    """
    if checkpoint_path is None:
        candidates = get_checkpoint_candidates(
            model_name,
            strategy_id=strategy_id,
            experiment_id=experiment_id,
        )
        checkpoint_path = next((path for path in candidates if path.exists()), None)
        if checkpoint_path is None:
            if strategy_id is None:
                raise FileNotFoundError(
                    f"No checkpoint found for {model_name} in {CHECKPOINTS_PATH}"
                )
            raise FileNotFoundError(
                f"No checkpoint found for {model_name}/{strategy_id} in {CHECKPOINTS_PATH}"
            )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct config from checkpoint
    config = checkpoint.get("config", {})
    freeze_backbone = config.get("freeze_backbone", False)
    use_lora = config.get("use_lora", False)

    # Create model (LoRA wrapper must be applied before load_state_dict)
    model = FineTunableModel(
        model_name=model_name,
        num_classes=NUM_STYLES,
        freeze_backbone=freeze_backbone,
        device=device,
        use_lora=use_lora,
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        lora_target_modules=config.get("lora_target_modules"),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def save_training_results(
    results: list[FineTuningResult],
    output_path: Path | None = None,
) -> None:
    """Save training results to JSON.

    Args:
        results: List of FineTuningResult objects.
        output_path: Path to save JSON. If None, uses default.
    """
    if not results:
        raise ValueError("save_training_results requires at least one result")

    experiment_id = results[0].experiment_id
    experiment_paths = ensure_experiment_layout(experiment_id)
    if output_path is None:
        output_path = experiment_paths.fine_tuning_results_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_payload = normalize_fine_tuning_results_payload(load_json(output_path) or {}, drop_legacy=True)
    existing_runs = {
        run["run_id"]: run
        for run in existing_payload.get("runs", [])
        if isinstance(run, dict) and run.get("run_id")
    }

    # Convert to serializable format and merge with any previous runs in the batch.
    for result in results:
        existing_runs[result.run_id] = {
            "experiment_id": result.experiment_id,
            "run_id": result.run_id,
            "model_name": result.model_name,
            "strategy_id": result.strategy_id,
            "best_val_acc": result.best_val_acc,
            "best_epoch": result.best_epoch,
            "train_history": result.train_history,
            "checkpoint_path": repo_relative_path(result.checkpoint_path),
            "manifest_path": repo_relative_path(result.manifest_path),
            "split_id": result.split_id,
            "split_artifact_path": repo_relative_path(result.split_artifact_path),
            "run_scope": result.run_scope,
            TRAINING_GIT_COMMIT_SHA_FIELD: result.training_git_commit_sha,
            "num_train_samples": result.num_train_samples,
            "num_val_samples": result.num_val_samples,
            "num_excluded_eval_samples": result.num_excluded_eval_samples,
            "config": result.config,
        }

    ordered_runs = [
        existing_runs[run_id]
        for run_id in sorted(existing_runs)
    ]

    save_json(
        output_path,
        normalize_fine_tuning_results_payload(
            {"experiment_id": experiment_id, "runs": ordered_runs},
            drop_legacy=True,
        ),
    )

    run_matrix = load_run_matrix(experiment_id)
    runs = dict(run_matrix.get("runs", {}))
    for result in results:
        runs[result.run_id] = {
            "run_id": result.run_id,
            "experiment_id": result.experiment_id,
            "model": result.model_name,
            "strategy": result.strategy_id,
            "split_id": result.split_id,
            "checkpoint_path": repo_relative_path(result.checkpoint_path),
            "selected_epoch": result.best_epoch,
            "selection_metric": result.config.get("checkpoint_selection_metric", PRIMARY_SELECTION_METRIC),
            "checkpoint_selection_split": result.config.get("checkpoint_selection_split"),
            "best_val_score": result.best_val_acc,
            "manifest_path": repo_relative_path(result.manifest_path),
            "analysis_artifact_paths": {},
            "run_scope": result.run_scope,
            TRAINING_GIT_COMMIT_SHA_FIELD: result.training_git_commit_sha,
        }
    save_run_matrix(
        experiment_id,
        {
            **run_matrix,
            "split_id": results[0].split_id,
            "runs": runs,
        },
    )
    write_active_experiment(
        experiment_id,
        split_id=results[0].split_id,
        run_matrix_path=experiment_paths.run_matrix_path,
        fine_tuning_results_path=output_path,
    )

    print(f"Saved training results to {output_path}")


def load_run_manifest(
    model_name: str,
    strategy_id: str,
    manifests_dir: Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Load strategy-specific fine-tuning run manifest."""
    if manifests_dir is not None:
        candidate_paths = [manifests_dir / f"{model_name}_{strategy_id}_manifest.json"]
    else:
        resolved_experiment_id = experiment_id
        if resolved_experiment_id is None:
            active = load_active_experiment()
            if active is not None:
                resolved_experiment_id = active.get("experiment_id")
        manifest_candidates: list[Path] = []
        if resolved_experiment_id:
            run_id = build_run_id(resolved_experiment_id, model_name, strategy_id)
            manifest_candidates.append(
                get_experiment_paths(resolved_experiment_id).manifests_dir / f"{run_id}_manifest.json"
            )
        manifest_candidates.append(LEGACY_MANIFESTS_PATH / f"{model_name}_{strategy_id}_manifest.json")
        candidate_paths = manifest_candidates

    manifest_path = next((path for path in candidate_paths if path.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(
            "Run manifest not found in any candidate path: "
            + ", ".join(str(path) for path in candidate_paths)
        )
    with open(manifest_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return normalize_run_manifest_payload(data, drop_legacy=True)
