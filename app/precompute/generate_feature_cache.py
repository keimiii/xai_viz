"""Generate feature cache for all models and layers.

Extracts CLS token and patch token features for all 139 annotated images across
5 models and 12 layers, storing results in HDF5 format. These features are used
for computing cosine similarity between bounding boxes and image patches.

Usage:
    python -m app.precompute.generate_feature_cache --models all
    python -m app.precompute.generate_feature_cache --models dinov2 clip
    python -m app.precompute.generate_feature_cache --models dinov2 --layers 10 11
    python -m app.precompute.generate_feature_cache --finetuned --models dinov2 --strategies lora
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.cache import FeatureCache
from ssl_attention.config import (
    CACHE_PATH,
    DATASET_PATH,
    FINETUNE_MODELS,
    FINETUNE_STRATEGIES,
    MODELS,
)
from ssl_attention.data import AnnotatedSubset
from ssl_attention.evaluation.fine_tuning import (
    CHECKPOINTS_PATH,
    get_finetuned_cache_key,
    load_finetuned_model,
    strategy_uses_legacy_checkpoint_fallback,
)
from ssl_attention.models import create_model
from ssl_attention.utils import get_device


def discover_checkpoints_by_strategy(
    checkpoint_dir: Path,
    model_names: list[str] | None = None,
    strategies: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Discover available fine-tuned checkpoints grouped by strategy."""
    candidates = model_names if model_names else sorted(FINETUNE_MODELS)
    valid_strategies = {s.value for s in FINETUNE_STRATEGIES}
    requested_strategies = (
        sorted(valid_strategies) if not strategies else [s for s in strategies if s in valid_strategies]
    )
    found: dict[str, dict[str, Path]] = {}

    for name in candidates:
        if name not in FINETUNE_MODELS:
            print(f"Warning: {name} is not fine-tunable (skipped)")
            continue

        model_checkpoints: dict[str, Path] = {}
        for strategy in requested_strategies:
            strategy_path = checkpoint_dir / f"{name}_{strategy}_finetuned.pt"
            if strategy_path.exists():
                model_checkpoints[strategy] = strategy_path
                continue

            legacy_path = checkpoint_dir / f"{name}_finetuned.pt"
            if strategy_uses_legacy_checkpoint_fallback(strategy) and legacy_path.exists():
                model_checkpoints[strategy] = legacy_path

        if model_checkpoints:
            found[name] = model_checkpoints
        else:
            print(
                f"Warning: No checkpoint for {name} with strategies {requested_strategies} "
                f"in {checkpoint_dir}"
            )

    return found


def discover_checkpoints(
    checkpoint_dir: Path,
    model_names: list[str] | None = None,
) -> dict[str, Path]:
    """Backward-compatible checkpoint discovery (single best per model)."""
    by_strategy = discover_checkpoints_by_strategy(
        checkpoint_dir,
        model_names=model_names,
        strategies=["lora", "full", "linear_probe"],
    )
    selected: dict[str, Path] = {}
    for model_name, strategy_paths in by_strategy.items():
        for strategy in ("lora", "full", "linear_probe"):
            if strategy in strategy_paths:
                selected[model_name] = strategy_paths[strategy]
                break
    return selected


def extract_finetuned_hidden_states_for_cache(
    model: Any,
    pixel_values: torch.Tensor,
) -> list[torch.Tensor]:
    """Extract per-layer hidden states for fine-tuned cache generation.

    Fine-tuned MAE must use the deterministic analysis path so feature-cache
    patch tokens stay aligned with the attention cache and heatmaps.
    """
    backbone_output = model._forward_backbone_for_analysis(  # noqa: SLF001
        pixel_values,
        output_hidden_states=True,
    )
    if backbone_output.hidden_states is None:
        raise ValueError("Fine-tuned backbone did not return hidden states.")
    return list(backbone_output.hidden_states[1:])


def generate_features_for_model(
    model_name: str,
    dataset: AnnotatedSubset,
    cache: FeatureCache,
    layers: list[int] | None = None,
    device: torch.device | str = "cpu",
    skip_existing: bool = True,
    finetuned: bool = False,
    checkpoint_path: Path | None = None,
    strategy_id: str | None = None,
) -> dict[str, int]:
    """Generate patch features for a single model.

    Args:
        model_name: Name of model (e.g., "dinov2").
        dataset: Annotated dataset.
        cache: FeatureCache instance.
        layers: Specific layers to process. None = all layers.
        device: Compute device.
        skip_existing: Skip if already cached.

    Returns:
        Dict with statistics: {"processed": N, "skipped": M, "errors": E}
    """
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    model_config = MODELS[model_name]
    num_layers = model_config.num_layers
    layers_to_process = layers if layers else list(range(num_layers))

    cache_model_key = get_finetuned_cache_key(model_name, strategy_id) if finetuned else model_name
    mode_label = "fine-tuned" if finetuned else "frozen"

    print(f"\n{'='*60}")
    print(f"Processing {cache_model_key} [{mode_label}] ({len(layers_to_process)} layers)")
    print(f"Patch size: {model_config.patch_size}x{model_config.patch_size}")
    print(f"Embed dim: {model_config.embed_dim}")
    print(f"{'='*60}")

    model: Any
    load_device = torch.device(device) if isinstance(device, str) else device

    # Load model
    if finetuned:
        print(f"Loading fine-tuned {model_name} from {checkpoint_path}...")
        model = load_finetuned_model(
            model_name,
            checkpoint_path=checkpoint_path,
            strategy_id=strategy_id,
            device=load_device,
        )
    else:
        print(f"Loading {model_name}...")
        model = create_model(model_name)
        model.to(load_device)
        model.eval()

    try:
        # Process each image
        for sample in tqdm(dataset, desc=f"{cache_model_key}"):
            image_id = sample["image_id"]
            image = sample["image"]

            try:
                # Check if all layers already cached for this image
                if skip_existing:
                    all_cached = all(
                        cache.exists(cache_model_key, f"layer{layer}", image_id)
                        for layer in layers_to_process
                    )
                    if all_cached:
                        stats["skipped"] += len(layers_to_process)
                        continue

                # Run inference once with hidden states to get all layer outputs
                if finetuned:
                    pixel_values = model.preprocess([image])
                    hidden_states = extract_finetuned_hidden_states_for_cache(
                        model,
                        pixel_values,
                    )
                else:
                    with torch.no_grad():
                        preprocessed = model.preprocess([image]).to(device)
                        output = model.forward(preprocessed, output_hidden_states=True)
                    if output.hidden_states is None:
                        raise ValueError("Model did not return hidden states.")
                    hidden_states = output.hidden_states

                # Extract features for each requested layer
                for layer in layers_to_process:
                    layer_key = f"layer{layer}"

                    if skip_existing and cache.exists(cache_model_key, layer_key, image_id):
                        stats["skipped"] += 1
                        continue

                    # Get hidden states for this specific layer
                    # hidden_states[layer] is the output after transformer layer `layer`
                    layer_hidden = hidden_states[layer]  # (B, seq_len, D)

                    # Extract CLS token and patch tokens from this layer's hidden state
                    # Structure depends on model type:
                    # - DINOv2/v3: [CLS] + [registers] + [patches]
                    # - CLIP/MAE: [CLS] + [patches]
                    # - SigLIP/SigLIP2: [patches] only (no CLS in sequence)
                    if model_name in ("siglip", "siglip2"):
                        # SigLIP/SigLIP2 have no CLS token in sequence - compute mean pooling
                        cls_token = layer_hidden.mean(dim=1)  # (B, D)
                        patch_tokens = layer_hidden  # (B, 196, D)
                    else:
                        # All other models: CLS at position 0, patches after (+ registers)
                        cls_token = layer_hidden[:, 0, :]  # (B, D)
                        patch_start = 1 + model_config.num_registers
                        patch_tokens = layer_hidden[:, patch_start:, :]  # (B, N, D)

                    cache.store(
                        model=cache_model_key,
                        layer=layer_key,
                        image_id=image_id,
                        cls_token=cls_token,
                        patch_tokens=patch_tokens,
                    )
                    stats["processed"] += 1

            except Exception as e:
                print(f"\nError processing {image_id}: {e}")
                stats["errors"] += 1

    finally:
        # Free GPU memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate feature cache")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to process (or 'all')",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Specific layers to process (default: all 12)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=CACHE_PATH / "features.h5",
        help="Path to feature cache HDF5 file",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing cached items",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Generate cache for fine-tuned models (requires checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINTS_PATH,
        help=f"Directory containing fine-tuned checkpoints (default: {CHECKPOINTS_PATH})",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["auto"],
        choices=["auto", "all", "linear_probe", "lora", "full"],
        help=(
            "Fine-tuning strategies to cache in --finetuned mode. "
            "'auto' keeps legacy behavior (one best checkpoint per model)."
        ),
    )
    args = parser.parse_args()

    # Determine models to process
    if "all" in args.models:
        models_to_process = sorted(FINETUNE_MODELS) if args.finetuned else list(MODELS.keys())
    else:
        valid_models = set(FINETUNE_MODELS) if args.finetuned else set(MODELS.keys())
        models_to_process = [m for m in args.models if m in valid_models]
        invalid = [m for m in args.models if m not in valid_models]
        if invalid:
            print(f"Warning: Unknown models ignored: {invalid}")
            print(f"Available: {sorted(valid_models)}")

    if not models_to_process:
        print("No valid models specified")
        return 1

    checkpoints: dict[str, dict[str, Path]] = {}
    model_strategy_pairs: list[tuple[str, str | None]] = []
    if args.finetuned:
        strategies: list[str] | None
        if "auto" in args.strategies:
            discovered = discover_checkpoints(args.checkpoint_dir, models_to_process)
            if not discovered:
                print("No fine-tuned checkpoints found. Nothing to do.")
                return 1
            for model_name, checkpoint in discovered.items():
                checkpoints[model_name] = {"auto": checkpoint}
                model_strategy_pairs.append((model_name, None))
            models_to_process = list(discovered.keys())
        else:
            if "all" in args.strategies:
                strategies = [s.value for s in FINETUNE_STRATEGIES]
            else:
                strategies = args.strategies
            checkpoints = discover_checkpoints_by_strategy(
                args.checkpoint_dir,
                model_names=models_to_process,
                strategies=strategies,
            )
            if not checkpoints:
                print("No fine-tuned checkpoints found. Nothing to do.")
                return 1
            for model_name, strategy_paths in checkpoints.items():
                for discovered_strategy in sorted(strategy_paths):
                    model_strategy_pairs.append((model_name, discovered_strategy))
            models_to_process = sorted(checkpoints.keys())

    # Setup
    device = args.device or get_device()
    mode_label = "FINE-TUNED" if args.finetuned else "FROZEN"
    print(f"Mode: {mode_label}")
    print(f"Device: {device}")

    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"Dataset: {len(dataset)} annotated images")

    cache = FeatureCache(args.cache_path)
    print(f"Cache: {args.cache_path}")
    if args.finetuned:
        print(f"Checkpoints: {args.checkpoint_dir}")
        if model_strategy_pairs:
            print(f"Model/strategy pairs: {model_strategy_pairs}")

    # Process models one at a time to conserve memory
    total_stats = {"processed": 0, "skipped": 0, "errors": 0}

    targets: list[tuple[str, str | None]] = (
        model_strategy_pairs if args.finetuned else [(m, None) for m in models_to_process]
    )

    for model_name, strategy_id in targets:
        checkpoint_path = None
        if args.finetuned:
            if strategy_id is None:
                checkpoint_path = checkpoints[model_name]["auto"]
            else:
                checkpoint_path = checkpoints[model_name][strategy_id]
        stats = generate_features_for_model(
            model_name=model_name,
            dataset=dataset,
            cache=cache,
            layers=args.layers,
            device=device,
            skip_existing=not args.no_skip,
            finetuned=args.finetuned,
            checkpoint_path=checkpoint_path,
            strategy_id=strategy_id,
        )

        for key in total_stats:
            total_stats[key] += stats[key]

        cache_key = get_finetuned_cache_key(model_name, strategy_id) if args.finetuned else model_name
        print(f"\n{cache_key} complete: {stats}")

    print(f"\n{'='*60}")
    print(f"SUMMARY ({mode_label})")
    print(f"{'='*60}")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total errors: {total_stats['errors']}")

    return 0 if total_stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
