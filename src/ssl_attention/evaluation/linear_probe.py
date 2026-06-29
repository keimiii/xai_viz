"""Linear probe evaluation for feature quality validation.

This module validates that SSL model features contain discriminative information
about architectural styles BEFORE analyzing attention patterns. If features
can't distinguish Gothic from Baroque, attention analysis is questionable.

Uses sklearn for simplicity:
- LogisticRegression with L2 regularization
- StratifiedKFold cross-validation for small dataset (139 images)
- Standard metrics: accuracy, F1, per-class accuracy

Expected baselines:
- Random: 25% accuracy (4 classes)
- Good features: >50% accuracy
- Excellent features: >70% accuracy

Example:
    from ssl_attention.evaluation import train_linear_probe_sklearn

    # Extract features from model (implementation depends on model)
    features = model.extract_features(images)  # (N, D)
    labels = [sample["style_label"] for sample in dataset]

    result = train_linear_probe_sklearn(features, labels)
    print(f"CV accuracy: {result.cv_mean:.1%} ± {result.cv_std:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ssl_attention.models.protocols import VisionBackbone


@dataclass
class ProbeResult:
    """Result of linear probe evaluation.

    Attributes:
        train_accuracy: Accuracy computed on full training data (not a
            generalization estimate); use ``cv_mean`` for generalization.
        train_per_class_accuracy: Per-class accuracy computed on full training
            data (not a generalization estimate); use ``cv_mean`` for
            generalization.
        confusion_matrix: Confusion matrix (true x predicted) from full refit.
        train_f1_macro: Macro-averaged F1 computed on full training data (not a
            generalization estimate); use ``cv_mean`` for generalization.
        cv_scores: Accuracy scores for each CV fold.
        cv_mean: Mean accuracy across CV folds.
        cv_std: Std deviation of accuracy across CV folds.
        class_names: Names of the classes.
    """

    train_accuracy: float
    train_per_class_accuracy: dict[int, float]
    confusion_matrix: np.ndarray
    train_f1_macro: float
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    class_names: tuple[str, ...] = field(default_factory=tuple)


def train_linear_probe_sklearn(
    features: np.ndarray | Tensor,
    labels: np.ndarray | list[int] | Tensor,
    n_splits: int = 5,
    class_names: tuple[str, ...] | None = None,
    random_state: int = 42,
    max_iter: int = 1000,
) -> ProbeResult:
    """Train and evaluate a linear probe using sklearn.

    Uses logistic regression with L2 regularization and stratified
    k-fold cross-validation.

    Args:
        features: Feature array of shape (N, D).
        labels: Class labels of shape (N,). Must be integers 0 to C-1.
        n_splits: Number of cross-validation folds.
        class_names: Optional names for each class.
        random_state: Random seed for reproducibility.
        max_iter: Maximum iterations for logistic regression.

    Returns:
        ProbeResult with accuracy metrics and confusion matrix.

    Raises:
        ValueError: If features and labels have mismatched lengths.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Convert to numpy
    if isinstance(features, Tensor):
        features = features.numpy()
    if isinstance(labels, Tensor):
        labels = labels.numpy()
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if len(features) != len(labels):
        raise ValueError(
            f"Feature/label length mismatch: {len(features)} != {len(labels)}"
        )

    # Filter out samples with None/invalid labels
    valid_mask = labels >= 0
    features = features[valid_mask]
    labels = labels[valid_mask]

    n_classes = len(np.unique(labels))
    if class_names is None:
        class_names = tuple(str(i) for i in range(n_classes))

    # Pipeline ensures scaling is fit only on each CV train fold (no data leakage)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, features, labels, cv=cv, scoring="accuracy")

    # Train on full data for final metrics
    pipeline.fit(features, labels)
    predictions = pipeline.predict(features)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    cm = confusion_matrix(labels, predictions)

    # Per-class accuracy
    per_class_acc: dict[int, float] = {}
    for i in range(n_classes):
        class_mask = labels == i
        if class_mask.sum() > 0:
            per_class_acc[i] = accuracy_score(labels[class_mask], predictions[class_mask])
        else:
            per_class_acc[i] = 0.0

    return ProbeResult(
        train_accuracy=float(accuracy),
        train_per_class_accuracy=per_class_acc,
        confusion_matrix=cm,
        train_f1_macro=float(f1),
        cv_scores=cv_scores.tolist(),
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        class_names=class_names,
    )


def extract_features_for_probe(
    model: VisionBackbone,
    dataset: Any,
    batch_size: int = 16,
    device: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Extract features and labels from a dataset using an SSL model.

    Args:
        model: SSL model with forward() returning ModelOutput with features.
        dataset: Dataset yielding dicts with "image" and "style_label" keys.
        batch_size: Batch size for feature extraction.
        device: Device to use. If None, auto-detect.

    Returns:
        Tuple of (features, labels) tensors.
        - features: (N, D) tensor of CLS token features
        - labels: (N,) tensor of style labels (-1 for unlabeled)
    """
    from ssl_attention.utils.device import get_device

    if device is None:
        device = str(get_device())

    # Simple collate that keeps images as list
    def collate(batch: list[dict]) -> dict:
        return {
            "images": [s["image"] for s in batch],
            "style_labels": [
                s["style_label"] if s["style_label"] is not None else -1
                for s in batch
            ],
        }

    loader: DataLoader[Any] = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate
    )

    all_features: list[Tensor] = []
    all_labels: list[int] = []

    # Set model to eval mode (model should be an nn.Module wrapper)
    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for batch in loader:
            # Preprocess and forward
            pixel_values = model.preprocess(batch["images"])
            pixel_values = pixel_values.to(device)
            output = model.forward(pixel_values)

            # Get CLS token features (or pooled features)
            features = output.cls_token  # (B, D)
            all_features.append(features.cpu())
            all_labels.extend(batch["style_labels"])

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    return features_tensor, labels_tensor


def compare_model_features(
    models: dict[str, VisionBackbone],
    dataset: Any,
    batch_size: int = 16,
    n_splits: int = 5,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, ProbeResult]:
    """Compare feature quality across multiple models.

    Args:
        models: Dict mapping model name to SSLModel instance.
        dataset: Dataset yielding dicts with "image" and "style_label" keys.
        batch_size: Batch size for feature extraction.
        n_splits: Number of CV folds.
        class_names: Optional class names for results.

    Returns:
        Dict mapping model name to ProbeResult.

    Example:
        >>> from ssl_attention.models import load_model
        >>> models = {
        ...     "dinov2": load_model("dinov2"),
        ...     "clip": load_model("clip"),
        ... }
        >>> results = compare_model_features(models, dataset)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.cv_mean:.1%}")
    """
    results: dict[str, ProbeResult] = {}

    for model_name, model in models.items():
        print(f"Extracting features for {model_name}...")
        features, labels = extract_features_for_probe(
            model, dataset, batch_size=batch_size
        )

        print(f"Training probe for {model_name}...")
        result = train_linear_probe_sklearn(
            features,
            labels,
            n_splits=n_splits,
            class_names=class_names,
        )
        results[model_name] = result

        print(f"  {model_name}: {result.cv_mean:.1%} ± {result.cv_std:.1%}")

    return results


def print_probe_summary(result: ProbeResult, model_name: str = "") -> None:
    """Print a formatted summary of probe results.

    Args:
        result: ProbeResult to summarize.
        model_name: Optional model name for header.
    """
    header = f"Linear Probe Results{f' - {model_name}' if model_name else ''}"
    print(f"\n{header}")
    print("=" * len(header))

    print(f"Cross-validation: {result.cv_mean:.1%} ± {result.cv_std:.1%}")
    print(f"  Fold scores: {[f'{s:.1%}' for s in result.cv_scores]}")

    print(f"\nTrain accuracy (full refit): {result.train_accuracy:.1%}")
    print(f"Train F1 (full refit): {result.train_f1_macro:.3f}")

    print("\nTrain per-class accuracy (full refit):")
    for class_idx, acc in sorted(result.train_per_class_accuracy.items()):
        name = (
            result.class_names[class_idx]
            if class_idx < len(result.class_names)
            else str(class_idx)
        )
        print(f"  {name}: {acc:.1%}")

    print("\nConfusion Matrix:")
    print(result.confusion_matrix)
