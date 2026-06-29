"""Generate image assets for the mid-project progress presentation.

Reads from pre-computed heatmap cache and metrics data — no model inference.

Output: PNG images in outputs/slides/ for embedding in the PPTX.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
resolve_active_artifact_path = importlib.import_module(
    "ssl_attention.evaluation.fine_tuning_artifacts"
).resolve_active_artifact_path

CACHE = ROOT / "outputs" / "cache" / "heatmaps"
ORIGINALS_CLEAN = CACHE / "originals" / "clean"
ORIGINALS_BBOX = CACHE / "originals" / "bbox"
FIGURES = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"
OUT = ROOT / "outputs" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
Q2_RESULTS = resolve_active_artifact_path(
    "q2_metrics_path",
    RESULTS / "q2_metrics_analysis.json",
)
METRICS_SUMMARY = ROOT / "outputs" / "cache" / "metrics_summary.json"

# Selected images per style (highest bbox count)
STYLE_IMAGES = {
    "Romanesque": "Q526047_wd0.jpg",
    "Gothic": "Q2981_wd0.jpg",
    "Renaissance": "Q1165020_wd0.jpg",
    "Baroque": "Q1502706_wd0.jpg",
}

# Hero image for title slide + motivation comparison
HERO_IMAGE = "Q206823_wd0.jpg"  # Gothic, 10 bboxes

# Palette (matches project)
STEEL_BLUE = "#4E79A7"
TEAL = "#93B7BE"
TERRACOTTA = "#D4764E"
CHARCOAL = "#2D3436"
WARM_GRAY = "#8A817C"
SUCCESS = "#3A7D44"
FAILURE = "#C04E4E"
MODEL_LABELS = {
    "clip": "CLIP",
    "dinov2": "DINOv2",
    "dinov3": "DINOv3",
    "mae": "MAE",
    "siglip": "SigLIP",
    "siglip2": "SigLIP 2",
}


class ModelPoint(TypedDict):
    frozen: float
    delta: float
    strategy: str
    sig: bool


def _load_cached_image(model: str, layer: str, method: str, variant: str, image_id: str) -> Image.Image:
    """Load a pre-rendered heatmap image from cache."""
    path = CACHE / model / layer / method / variant / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Cached image not found: {path}")
    return Image.open(path).convert("RGBA")


def _add_label(img: Image.Image, label: str, position: str = "bottom") -> Image.Image:
    """Add a text label bar to the top or bottom of an image."""
    bar_h = 32
    w, h = img.size
    new_h = h + bar_h
    canvas = Image.new("RGBA", (w, new_h), (45, 52, 54, 255))  # charcoal
    y_offset = bar_h if position == "top" else 0
    canvas.paste(img, (0, y_offset))
    draw = ImageDraw.Draw(canvas)
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    text_y = new_h - bar_h + 6 if position == "bottom" else 6
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) // 2, text_y), label, fill=(255, 255, 255, 255), font=font)
    return canvas


# ---------------------------------------------------------------------------
# Slide 1: Title hero image (DINOv3 overlay + bboxes on Gothic church)
# ---------------------------------------------------------------------------
def generate_slide1_title() -> None:
    print("Generating Slide 1: Title hero image...")
    # Use overlay WITHOUT bboxes for a cleaner hero visual
    img = _load_cached_image("dinov3", "layer11", "cls", "overlay", HERO_IMAGE)
    img.save(OUT / "slide01_title_hero.png")
    print(f"  -> {OUT / 'slide01_title_hero.png'}")


# ---------------------------------------------------------------------------
# Slide 2: Good vs bad attention comparison (DINOv3 vs MAE, same image)
# ---------------------------------------------------------------------------
def generate_slide2_motivation() -> None:
    print("Generating Slide 2: Good vs bad attention comparison...")
    # Also get the overlays (image + heatmap, no bboxes) for context
    good_overlay = _load_cached_image("dinov3", "layer11", "cls", "overlay", HERO_IMAGE)
    bad_overlay = _load_cached_image("mae", "layer4", "cls", "overlay", HERO_IMAGE)

    # Create a 2x2 grid: top row = overlays (context), bottom row = pure heatmaps (contrast)
    # But actually, just the overlays without bboxes already show the difference well
    good_labelled = _add_label(good_overlay, "DINOv3 — Focused on portal & tracery")
    bad_labelled = _add_label(bad_overlay, "MAE — Diffuse, unfocused attention")

    # Also add the original image for reference
    original_path = ORIGINALS_CLEAN / f"{HERO_IMAGE}.png"
    original = Image.open(original_path).convert("RGBA")
    original_labelled = _add_label(original, "Original Image")

    # Three-panel: original | DINOv3 | MAE
    padding = 12
    panels = [original_labelled, good_labelled, bad_labelled]
    w = sum(p.width for p in panels) + padding * (len(panels) - 1)
    h = max(p.height for p in panels)
    canvas = Image.new("RGBA", (w, h), (248, 249, 250, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + padding

    canvas.save(OUT / "slide02_good_vs_bad.png")
    print(f"  -> {OUT / 'slide02_good_vs_bad.png'}")


# ---------------------------------------------------------------------------
# Slide 4: 2x2 grid of churches by style with bounding boxes
# ---------------------------------------------------------------------------
def generate_slide4_dataset() -> None:
    print("Generating Slide 4: Style grid...")
    images = []
    for style, img_id in STYLE_IMAGES.items():
        path = ORIGINALS_BBOX / f"{img_id}.png"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {style}")
            continue
        img = Image.open(path).convert("RGBA")
        img = _add_label(img, style, position="bottom")
        images.append(img)

    if len(images) != 4:
        print("  WARNING: Expected 4 images, got", len(images))
        return

    # Resize all to same dimensions
    target_w = max(img.width for img in images)
    target_h = max(img.height for img in images)
    resized = []
    for img in images:
        canvas = Image.new("RGBA", (target_w, target_h), (248, 249, 250, 255))
        x_off = (target_w - img.width) // 2
        y_off = (target_h - img.height) // 2
        canvas.paste(img, (x_off, y_off))
        resized.append(canvas)

    # 2x2 grid
    gap = 12
    grid_w = target_w * 2 + gap
    grid_h = target_h * 2 + gap
    canvas = Image.new("RGBA", (grid_w, grid_h), (248, 249, 250, 255))
    canvas.paste(resized[0], (0, 0))
    canvas.paste(resized[1], (target_w + gap, 0))
    canvas.paste(resized[2], (0, target_h + gap))
    canvas.paste(resized[3], (target_w + gap, target_h + gap))

    canvas.save(OUT / "slide04_style_grid.png")
    print(f"  -> {OUT / 'slide04_style_grid.png'}")


# ---------------------------------------------------------------------------
# Slide 6: Methodology pipeline (4 panels)
# ---------------------------------------------------------------------------
def generate_slide6_pipeline() -> None:
    """Slide 6 pipeline images are now captured via Playwright screenshots."""
    print("Slide 6: Pipeline images will be captured via Playwright (skipping Python generation)")


# ---------------------------------------------------------------------------
# Slide 13: Frozen IoU vs Delta IoU scatter
# ---------------------------------------------------------------------------
def generate_slide13_scatter() -> None:
    print("Generating Slide 13: Frozen IoU vs Delta IoU scatter...")

    metrics_summary = json.loads(METRICS_SUMMARY.read_text(encoding="utf-8"))
    q2_results = json.loads(Q2_RESULTS.read_text(encoding="utf-8"))

    q2_rows = [
        row for row in q2_results.get("rows", [])
        if row.get("metric") == "iou" and row.get("percentile") == 90
    ]

    models_data: dict[str, ModelPoint] = {}
    for model_key, summary in metrics_summary.get("models", {}).items():
        candidate_rows = [row for row in q2_rows if row.get("model_name") == model_key]
        if not candidate_rows:
            continue
        best_row = max(candidate_rows, key=lambda row: row.get("mean_delta", float("-inf")))
        models_data[MODEL_LABELS.get(model_key, model_key)] = {
            "frozen": float(summary.get("best_iou", 0.0)),
            "delta": float(best_row.get("mean_delta", 0.0)),
            "strategy": str(best_row.get("strategy_id", "full")).replace("_", " ").title(),
            "sig": bool(best_row.get("significant", False)),
        }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, d in models_data.items():
        color = SUCCESS if d["sig"] else WARM_GRAY
        marker = "D" if d["strategy"] == "LoRA" else "o"
        ax.scatter(d["frozen"], d["delta"], c=color, s=120, marker=marker,
                   edgecolors=CHARCOAL, linewidths=0.8, zorder=5)
        # Label offset to avoid overlap
        x_off, y_off = 0.003, 0.002
        if name == "SigLIP 2":
            y_off = -0.005
        ax.annotate(name, (d["frozen"] + x_off, d["delta"] + y_off),
                    fontsize=11, fontweight="bold", color=CHARCOAL)

    # Quadrant annotations
    ax.axhline(y=0.01, color="#cccccc", linestyle="--", linewidth=0.8)
    ax.text(0.11, 0.055, "Already\naligned", fontsize=9, color="#999999",
            ha="center", style="italic")
    ax.text(0.045, 0.055, "Improvable\nvia fine-tuning", fontsize=9,
            color=SUCCESS, ha="center", style="italic", fontweight="bold")
    ax.text(0.045, -0.002, "Low alignment,\nno improvement", fontsize=9,
            color=FAILURE, ha="center", style="italic")

    ax.set_xlabel("Frozen IoU @ 90th Percentile (Best Layer)", fontsize=12)
    ax.set_ylabel("Best \u0394 IoU (Fine-tuned \u2212 Frozen)", fontsize=12)
    ax.set_title("Pre-training Objective Determines Alignment & Plasticity",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0.02, 0.16)
    ax.set_ylim(-0.008, 0.075)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=SUCCESS,
               markersize=10, markeredgecolor=CHARCOAL, label="Significant (LoRA)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SUCCESS,
               markersize=10, markeredgecolor=CHARCOAL, label="Significant (Full)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=WARM_GRAY,
               markersize=10, markeredgecolor=CHARCOAL, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT / "slide13_scatter.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> {OUT / 'slide13_scatter.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Output directory: {OUT}\n")
    generate_slide1_title()
    generate_slide2_motivation()
    generate_slide4_dataset()
    generate_slide6_pipeline()
    generate_slide13_scatter()
    print(f"\nDone! {len(list(OUT.glob('*.png')))} images generated in {OUT}")


if __name__ == "__main__":
    main()
