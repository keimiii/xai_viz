#!/usr/bin/env python3
"""Per-feature Δ IoU analysis for fine-tuned models.

Computes IoU between the attention heatmap and each individual feature's
bounding boxes (rather than the union mask used by analyze_q2_metrics.py),
then takes the delta against the frozen baseline from metrics.db.

Primary use case: investigating the MAE Renaissance spike by identifying
which specific architectural features drive per-image IoU gains.

Outputs:
  - Console table: per-feature Δ IoU sorted by delta
  - JSON artifact: full per-feature statistics
  - PNG figure: bar chart of per-feature Δ IoU for selected model

Usage:
  uv run python experiments/scripts/analyze_feature_delta_iou.py
  uv run python experiments/scripts/analyze_feature_delta_iou.py --model mae --strategy full
  uv run python experiments/scripts/analyze_feature_delta_iou.py --model mae --strategy full --style Renaissance
  uv run python experiments/scripts/analyze_feature_delta_iou.py --model clip --strategy full --min-boxes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.attention.cls_attention import (  # noqa: E402
    HeadFusion,
    attention_to_heatmap,
    extract_cls_attention,
    extract_mean_attention,
)
from ssl_attention.config import (  # noqa: E402
    ANNOTATIONS_PATH,
    CACHE_PATH,
    DATASET_PATH,
    DEFAULT_METHOD,
    STYLE_MAPPING,
    STYLE_NAMES,
    AttentionMethod,
)
from ssl_attention.data.annotations import ImageAnnotation, load_annotations  # noqa: E402
from ssl_attention.data.wikichurches import AnnotatedSubset  # noqa: E402
from ssl_attention.evaluation.fine_tuning import (  # noqa: E402
    FineTunableModel,
    load_finetuned_model,
)
from ssl_attention.evaluation.fine_tuning_artifacts import (  # noqa: E402
    get_experiment_paths,
    load_active_experiment,
)
from ssl_attention.metrics.iou import compute_iou  # noqa: E402
from ssl_attention.utils.device import clear_memory  # noqa: E402

METRICS_DB_PATH = CACHE_PATH / "metrics.db"

STYLE_BY_ID: dict[str, str] = {qid: STYLE_NAMES[idx] for qid, idx in STYLE_MAPPING.items()}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureDeltaRow:
    feature_label: int
    feature_name: str
    bbox_count: int          # number of bbox elements across all images
    image_count: int         # number of images containing this feature
    frozen_mean_iou: float
    finetuned_mean_iou: float
    delta_iou: float


# ---------------------------------------------------------------------------
# Frozen IoU from metrics.db
# ---------------------------------------------------------------------------

def load_frozen_feature_iou(
    model: str,
    layer: int,
    method: str,
    percentile: int,
    feature_labels: set[int],
) -> dict[int, float]:
    """Return {feature_label: mean_iou} from the frozen metrics.db."""
    conn = sqlite3.connect(METRICS_DB_PATH)
    layer_str = f"layer{layer}"
    placeholders = ",".join("?" * len(feature_labels))
    rows = conn.execute(
        f"""
        SELECT feature_label, mean_iou
        FROM feature_metrics
        WHERE model=? AND layer=? AND method=? AND percentile=?
          AND feature_label IN ({placeholders})
        """,
        [model, layer_str, method, percentile, *sorted(feature_labels)],
    ).fetchall()
    conn.close()
    return {label: iou for label, iou in rows}


# ---------------------------------------------------------------------------
# Attention extraction (mirrors analyze_q2_metrics.py)
# ---------------------------------------------------------------------------

def extract_attention_heatmap(
    model: FineTunableModel,
    pixel_values: torch.Tensor,
    model_name: str,
    layer: int,
) -> torch.Tensor:
    method = DEFAULT_METHOD.get(model_name, AttentionMethod.CLS)

    with torch.no_grad():
        output = (
            model.extract_attention(pixel_values)
            if isinstance(model, FineTunableModel)
            else model(pixel_values)
        )
        attention_weights = output.attention_weights

    if method == AttentionMethod.CLS:
        attn = extract_cls_attention(attention_weights, layer=layer, fusion=HeadFusion.MEAN)
    elif method == AttentionMethod.MEAN:
        attn = extract_mean_attention(attention_weights, layer=layer, fusion=HeadFusion.MEAN)
    else:
        raise ValueError(f"Unsupported attention method for this script: {method}")

    return attention_to_heatmap(attn, image_size=224, normalize=True).squeeze(0)


# ---------------------------------------------------------------------------
# Per-feature IoU computation
# ---------------------------------------------------------------------------

def compute_per_feature_iou(
    heatmap: torch.Tensor,
    annotation: ImageAnnotation,
    percentile: int,
) -> dict[int, float]:
    """Return {group_label: iou} for each distinct feature group in the annotation."""
    h, w = heatmap.shape[-2:]

    # Group bboxes by group_label
    groups: dict[int, list] = defaultdict(list)
    for bbox in annotation.bboxes:
        groups[bbox.group_label].append(bbox)

    result: dict[int, float] = {}
    for group_label, bboxes in groups.items():
        # Build union mask for this feature group only
        mask = torch.zeros(h, w, dtype=torch.bool, device=heatmap.device)
        for bbox in bboxes:
            mask |= bbox.to_mask(h, w).to(heatmap.device)

        iou, _, _ = compute_iou(heatmap, mask, percentile)
        result[group_label] = float(iou)

    return result


# ---------------------------------------------------------------------------
# Style filtering helpers
# ---------------------------------------------------------------------------

def get_style_image_ids(style_name: str | None, annotations: dict) -> set[str] | None:
    """Return image IDs for the given style, or None (= all images)."""
    if style_name is None:
        return None
    filtered = {
        img_id
        for img_id, ann in annotations.items()
        if any(STYLE_BY_ID.get(s) == style_name for s in ann.styles)
    }
    return filtered


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    *,
    model_name: str,
    strategy_id: str,
    experiment_id: str,
    layer: int,
    percentile: int,
    style_filter: str | None,
    min_boxes: int,
) -> list[FeatureDeltaRow]:
    # --- Load feature type names from building_parts.json ---
    with open(ANNOTATIONS_PATH) as f:
        raw = json.load(f)
    label_to_name: dict[int, str] = {i: m["name"] for i, m in enumerate(raw["meta"])}

    # --- Load dataset ---
    annotations = load_annotations(ANNOTATIONS_PATH)
    dataset = AnnotatedSubset(DATASET_PATH)

    style_image_ids = get_style_image_ids(style_filter, annotations)
    if style_image_ids is not None:
        print(f"Style filter '{style_filter}': {len(style_image_ids)} images")

    # --- Load fine-tuned checkpoint ---
    paths = get_experiment_paths(experiment_id)
    manifest_path = (
        paths.results_dir
        / "manifests"
        / f"{experiment_id}__{model_name}__{strategy_id}_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    checkpoint_path = project_root / manifest["checkpoint_path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Download the checkpoint from the other laptop first."
        )

    print(f"Loading fine-tuned {model_name} ({strategy_id}) from {checkpoint_path.name}...")
    ft_model = load_finetuned_model(model_name, checkpoint_path)
    ft_model.eval()

    method = DEFAULT_METHOD.get(model_name, AttentionMethod.CLS)

    # --- Iterate images, compute per-feature IoU ---
    # accumulators: {feature_label: [iou, ...]}
    ft_iou_acc: dict[int, list[float]] = defaultdict(list)
    bbox_count_acc: dict[int, int] = defaultdict(int)

    for sample in tqdm(dataset, desc=f"{model_name}/{strategy_id}"):
        image_id = sample["image_id"]
        annotation = sample["annotation"]

        if style_image_ids is not None and image_id not in style_image_ids:
            continue
        if not annotation.bboxes:
            continue

        processed = ft_model.processor(images=[sample["image"]], return_tensors="pt")
        pixel_values = processed["pixel_values"].to(
            device=ft_model.device, dtype=ft_model.dtype
        )

        heatmap = extract_attention_heatmap(ft_model, pixel_values, model_name, layer)

        per_feature = compute_per_feature_iou(heatmap, annotation, percentile)
        for group_label, iou_val in per_feature.items():
            ft_iou_acc[group_label].append(iou_val)

        # Count bbox elements per group
        groups: dict[int, int] = defaultdict(int)
        for bbox in annotation.bboxes:
            groups[bbox.group_label] += 1
        for gl, cnt in groups.items():
            bbox_count_acc[gl] += cnt

    clear_memory()

    # --- Load frozen IoU from metrics.db ---
    all_labels = set(ft_iou_acc.keys())
    frozen_iou_map = load_frozen_feature_iou(
        model=model_name,
        layer=layer,
        method=method.value if hasattr(method, "value") else str(method),
        percentile=percentile,
        feature_labels=all_labels,
    )

    # --- Build result rows ---
    rows: list[FeatureDeltaRow] = []
    for label, ft_iou_list in ft_iou_acc.items():
        if bbox_count_acc[label] < min_boxes:
            continue
        frozen_iou = frozen_iou_map.get(label)
        if frozen_iou is None:
            continue  # feature not in frozen DB (e.g. not seen at that layer)

        ft_mean = float(np.mean(ft_iou_list))
        rows.append(
            FeatureDeltaRow(
                feature_label=label,
                feature_name=label_to_name.get(label, f"label_{label}"),
                bbox_count=bbox_count_acc[label],
                image_count=len(ft_iou_list),
                frozen_mean_iou=frozen_iou,
                finetuned_mean_iou=ft_mean,
                delta_iou=ft_mean - frozen_iou,
            )
        )

    rows.sort(key=lambda r: r.delta_iou, reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(rows: list[FeatureDeltaRow], model: str, strategy: str, style: str | None) -> None:
    style_label = f" [{style}]" if style else ""
    print(f"\n{'=' * 90}")
    print(f"Per-Feature Δ IoU (p90)  —  {model} / {strategy}{style_label}")
    print(f"{'=' * 90}")
    print(f"{'Feature':<40}  {'Frozen':>8}  {'FT':>8}  {'Δ':>8}  {'n_imgs':>6}  {'n_boxes':>7}")
    print("-" * 90)
    for r in rows:
        sign = "▲" if r.delta_iou > 0.01 else ("▼" if r.delta_iou < -0.01 else " ")
        print(
            f"{r.feature_name:<40}  {r.frozen_mean_iou:>8.4f}  {r.finetuned_mean_iou:>8.4f}"
            f"  {r.delta_iou:>+8.4f}{sign}  {r.image_count:>6}  {r.bbox_count:>7}"
        )
    print()


def save_json(rows: list[FeatureDeltaRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"rows": [asdict(r) for r in rows]}, f, indent=2)
    print(f"JSON saved to: {output_path}")


def save_figure(
    rows: list[FeatureDeltaRow],
    output_path: Path,
    model: str,
    strategy: str,
    style: str | None,
    top_n: int = 20,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    # Show top_n by absolute Δ
    display = sorted(rows, key=lambda r: abs(r.delta_iou), reverse=True)[:top_n]
    display = sorted(display, key=lambda r: r.delta_iou, reverse=True)

    names = [r.feature_name for r in display]
    deltas = [r.delta_iou for r in display]
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, max(5, len(display) * 0.4)))
    y = np.arange(len(display))
    ax.barh(y, deltas, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%+.3f"))
    ax.set_xlabel("Δ IoU (fine-tuned − frozen, p90)", fontsize=10)
    style_label = f" [{style}]" if style else ""
    ax.set_title(
        f"Per-Feature Δ IoU — {model} / {strategy}{style_label}\n"
        f"Top {top_n} features by |Δ IoU|",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-feature Δ IoU for fine-tuned vs. frozen models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="mae", help="Model name (default: mae)")
    parser.add_argument(
        "--strategy",
        default="full",
        choices=["full", "lora", "linear_probe"],
        help="Fine-tuning strategy (default: full)",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment ID. Defaults to active experiment.",
    )
    parser.add_argument("--layer", type=int, default=11, help="Attention layer (default: 11)")
    parser.add_argument(
        "--percentile", type=int, default=90, help="IoU percentile threshold (default: 90)"
    )
    parser.add_argument(
        "--style",
        default=None,
        choices=[*STYLE_NAMES, None],
        help="Restrict analysis to one architectural style.",
    )
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=2,
        help="Minimum total bbox count for a feature to appear in output (default: 2)",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-figure", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    active = load_active_experiment()
    experiment_id = args.experiment_id or (
        str(active["experiment_id"]) if active else None
    )
    if experiment_id is None:
        raise RuntimeError("No active experiment. Pass --experiment-id explicitly.")

    print(f"Experiment: {experiment_id}")
    print(f"Model: {args.model}  Strategy: {args.strategy}  Layer: {args.layer}  P{args.percentile}")

    rows = run_analysis(
        model_name=args.model,
        strategy_id=args.strategy,
        experiment_id=experiment_id,
        layer=args.layer,
        percentile=args.percentile,
        style_filter=args.style,
        min_boxes=args.min_boxes,
    )

    if not rows:
        print("No features met the minimum box count threshold.")
        return

    print_table(rows, args.model, args.strategy, args.style)

    exp_dir = get_experiment_paths(experiment_id).results_dir
    suffix = f"_{args.style.lower()}" if args.style else ""
    json_path = args.output_json or exp_dir / f"feature_delta_iou_{args.model}_{args.strategy}{suffix}.json"
    fig_path = args.output_figure or exp_dir / f"feature_delta_iou_{args.model}_{args.strategy}{suffix}.png"

    save_json(rows, json_path)
    save_figure(rows, fig_path, args.model, args.strategy, args.style)
    print("Done.")


if __name__ == "__main__":
    main()
