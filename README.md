# Where Do Vision Foundation Models Look? A Comparative Study of Attention Alignment in Architectural Image Recognition

This repository accompanies the paper *"Where Do Vision Foundation Models Look? A Comparative Study of Attention Alignment in Architectural Image Recognition"*.

It provides the code and interactive visualization app used to evaluate whether vision foundation models attend to the same architectural features that domain experts identify as diagnostically important, using the WikiChurches benchmark dataset.

## Overview

The study examines attention alignment across three axes:

1. **Frozen alignment** — how well frozen models align with expert-annotated bounding boxes across five metrics (IoU, Coverage, MSE, KL, EMD).
2. **Adaptation effects** — how Linear Probe, LoRA, and Full fine-tuning change attention alignment.
3. **Head specialization** — which attention heads align best with specific architectural features.

**Models evaluated:** DINOv2, DINOv3, MAE, CLIP, SigLIP, SigLIP 2, ResNet-50

**Attention methods:** CLS token attention, Attention Rollout, Grad-CAM

## Requirements

- Python 3.12+, [uv](https://github.com/astral-sh/uv)
- Node.js 18+

## Setup

```bash
uv sync
```

Download precomputed artifacts (dataset, cache, checkpoints) from the [Google Drive artifact folder](https://drive.google.com/drive/folders/1pT8VrK6d9h-sZzAr6qhPxvNrVrRi-8Cd?usp=sharing), then copy into the repo root:

```bash
rsync -av dataset outputs /path/to/xai_viz/
```

Expected structure after copying:

```
xai_viz/
├── dataset/
└── outputs/
    ├── cache/
    │   ├── attention_viz.h5
    │   ├── features.h5
    │   ├── metrics.db
    │   ├── metrics_summary.json
    │   └── heatmaps/
    └── checkpoints/
        └── fine_tuning_primary_20260327/
```

## Running the App

```bash
./dev.sh
```

Backend at `http://127.0.0.1:8000`, frontend at `http://127.0.0.1:5173`.

## App Routes

| Route | Purpose |
|-------|---------|
| `/` | Gallery of annotated WikiChurches images |
| `/image/:imageId` | Single-image attention overlays, annotations, and metrics |
| `/compare` | Frozen model and fine-tuned variant comparisons |
| `/dashboard` | Q1 overview and Q3 head specialization surface |
| `/q2` | Fine-tuning adaptation summary |
| `/q3-report` | Head ranking, feature matrix, and frozen-to-adapted delta views |

## Reproducing Results From Scratch

Skip this section if using the precomputed Drive artifacts.

### Dataset

```bash
uv run python scripts/download_wikichurches.py --files churches.json image_meta.json building_parts.json
```

### Precompute caches

```bash
uv run python -m app.precompute.generate_attention_cache --models all
uv run python -m app.precompute.generate_feature_cache --models all
uv run python -m app.precompute.generate_heatmap_images --colormap viridis
uv run python -m app.precompute.generate_metrics_cache
```

Per-head caches for Q3:

```bash
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_metrics_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
```

### Fine-tuning (Q2)

```bash
EXPERIMENT_ID=fine_tuning_primary_20260327

uv run python experiments/scripts/fine_tune_models.py --all --freeze-backbone --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --lora --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --epochs 3 --experiment-id "$EXPERIMENT_ID"

uv run python experiments/scripts/analyze_q2_metrics.py \
  --experiment-id "$EXPERIMENT_ID" \
  --models clip dinov2 dinov3 mae siglip siglip2 \
  --strategies linear_probe lora full
```

## Repository Layout

```
xai_viz/
├── app/                 # FastAPI backend, React frontend, precompute scripts
├── dataset/             # WikiChurches data (not tracked, download separately)
├── experiments/         # Fine-tuning and analysis scripts
├── outputs/             # Caches, checkpoints, results (not tracked)
├── scripts/             # Dataset download utilities
├── src/ssl_attention/   # Core library: models, attention extraction, metrics
└── tests/               # Pytest suite
```

## Developer Checks

```bash
uv run ruff check .
uv run mypy
uv run pytest
cd app/frontend && npm run lint && npm run build
```

## Citation

If you use this code or dataset, please cite the WikiChurches dataset:

Barz, B., & Denzler, J. (2021). WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges. *NeurIPS Datasets and Benchmarks Track*. [arXiv:2108.06959](https://arxiv.org/abs/2108.06959)
# xai_viz
