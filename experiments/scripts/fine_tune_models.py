#!/usr/bin/env python3
"""Fine-tune SSL models on WikiChurches style classification.

Note: On MPS devices (Apple Silicon), we set PYTORCH_ENABLE_MPS_FALLBACK=1
to handle unsupported operations by falling back to CPU.

This script trains one or all SSL models on the 4-class architectural
style classification task. After training, models can be used to compare
attention patterns before vs after fine-tuning.

Usage:
    # Single model
    uv run python experiments/scripts/fine_tune_models.py --model dinov2

    # All models
    uv run python experiments/scripts/fine_tune_models.py --all

    # Head-only training (freeze backbone)
    uv run python experiments/scripts/fine_tune_models.py --model dinov2 --freeze-backbone

    # LoRA fine-tuning
    uv run python experiments/scripts/fine_tune_models.py --model dinov2 --lora

    # LoRA with custom rank
    uv run python experiments/scripts/fine_tune_models.py --model dinov2 --lora --lora-rank 16

    # Quick smoke test
    uv run python experiments/scripts/fine_tune_models.py --model dinov2 --epochs 1

Examples:
    # Full training run (10 epochs default)
    uv run python experiments/scripts/fine_tune_models.py --all

    # Custom hyperparameters
    uv run python experiments/scripts/fine_tune_models.py --model clip \
        --epochs 20 --batch-size 8 --lr-backbone 5e-6
"""

import argparse
import os
import sys
from pathlib import Path

# Enable MPS fallback for unsupported ops (bicubic interpolation backward)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.config import (  # noqa: E402
    DATASET_PATH,
    FINETUNE_MODELS,
    STYLE_NAMES,
)
from ssl_attention.data.wikichurches import FullDataset  # noqa: E402
from ssl_attention.evaluation.fine_tuning import (  # noqa: E402
    FineTunableModel,
    FineTuner,
    FineTuningConfig,
    FineTuningResult,
    save_training_results,
)
from ssl_attention.evaluation.fine_tuning_artifacts import make_experiment_id  # noqa: E402
from ssl_attention.utils.device import clear_memory  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SSL models on WikiChurches style classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        choices=sorted(FINETUNE_MODELS),
        help="Single model to fine-tune",
    )
    model_group.add_argument(
        "--all",
        action="store_true",
        help="Fine-tune all models",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-5,
        help="Learning rate for backbone (default: 1e-5)",
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="Learning rate for classification head (default: 1e-3)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone weights (head-only training)",
    )

    # LoRA settings
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA adapters on backbone attention layers",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout probability (default: 0.1)",
    )

    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    eval_mode_group = parser.add_mutually_exclusive_group()
    eval_mode_group.add_argument(
        "--include-annotated-eval",
        action="store_true",
        help=(
            "Include bbox-annotated images in training/validation split. "
            "Use this when dataset/images only contains the 139 annotated images."
        ),
    )
    eval_mode_group.add_argument(
        "--val-on-annotated-eval",
        action="store_true",
        help=(
            "Exploratory mode: choose checkpoints on the bbox-annotated pool. "
            "This is not the primary methodology because it reuses the annotated "
            "evaluation images for model selection."
        ),
    )
    eval_mode_group.add_argument(
        "--val-on-random-split",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Random stratified validation is now "
            "the default primary path."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help=(
            "Experiment batch identifier. Defaults to a timestamped id and is "
            "shared across all runs launched by this command."
        ),
    )

    return parser.parse_args()


def train_single_model(
    model_name: str,
    args: argparse.Namespace,
    dataset: FullDataset,
) -> FineTuningResult:
    """Train a single model.

    Args:
        model_name: Name of the model to train.
        args: Command line arguments.
        dataset: The dataset to train on.

    Returns:
        FineTuningResult with training metrics.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")

    # Create configuration
    config = FineTuningConfig(
        model_name=model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate_backbone=args.lr_backbone,
        learning_rate_head=args.lr_head,
        freeze_backbone=args.freeze_backbone,
        val_split=args.val_split,
        exclude_annotated_eval=not args.include_annotated_eval,
        val_on_annotated_eval=args.val_on_annotated_eval,
        seed=args.seed,
        experiment_id=args.experiment_id,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    print(f"Config: {config}")

    # Create model
    print(f"Loading {model_name} backbone...")
    model = FineTunableModel(
        model_name=model_name,
        freeze_backbone=args.freeze_backbone,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Create trainer and train
    tuner = FineTuner(config)
    result = tuner.train(model, dataset)

    print(
        f"\nBest validation accuracy: {result.best_val_acc:.1%} "
        f"(epoch {result.best_epoch}, strategy={result.strategy_id})"
    )
    print(f"Checkpoint saved to: {result.checkpoint_path}")
    print(f"Run manifest saved to: {result.manifest_path}")

    # Clean up memory before next model
    del model
    clear_memory()

    return result


def main() -> None:
    """Main entry point."""
    args = parse_args()
    if args.experiment_id is None:
        args.experiment_id = make_experiment_id()
    if args.val_on_random_split:
        print(
            "--val-on-random-split is now redundant because the shared non-annotated "
            "validation split is the default primary path."
        )

    # Load dataset
    print("Loading WikiChurches dataset...")
    dataset = FullDataset(DATASET_PATH, filter_labeled=True)
    print(f"Dataset size: {len(dataset)} labeled images")
    print(f"Style classes: {STYLE_NAMES}")

    # Determine which models to train
    model_names = sorted(FINETUNE_MODELS) if args.all else [args.model]

    print(f"\nModels to train: {model_names}")
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Epochs: {args.epochs}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Exclude annotated eval images: {not args.include_annotated_eval}")
    if args.val_on_annotated_eval:
        print("Validation: annotated eval holdout (exploratory only)")
    else:
        print("Validation: shared non-annotated stratified split (primary)")

    # Train models
    results: list[FineTuningResult] = []

    for model_name in model_names:
        result = train_single_model(model_name, args, dataset)
        results.append(result)

    # Save all results
    save_training_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Model':<10} {'Strategy':<14} {'Best Val Acc':>12} {'Best Epoch':>12}")
    print("-" * 52)
    for result in results:
        print(
            f"{result.model_name:<10} {result.strategy_id:<14} "
            f"{result.best_val_acc:>11.1%} {result.best_epoch:>12}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
