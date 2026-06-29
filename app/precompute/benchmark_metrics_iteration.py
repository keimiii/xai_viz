"""Benchmark metadata-only versus image-loading iteration for metrics precompute."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.config import DATASET_PATH
from ssl_attention.data import AnnotatedSubset


def _legacy_image_loading_iteration(dataset: AnnotatedSubset) -> dict[str, int]:
    """Traverse the dataset through __getitem__, forcing image decode."""
    total_bboxes = 0
    total_width = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"]
        annotation = sample["annotation"]
        total_bboxes += annotation.num_bboxes
        total_width += image.size[0]
        image.close()

    return {
        "num_images": len(dataset),
        "total_bboxes": total_bboxes,
        "width_checksum": total_width,
    }


def _metadata_only_iteration(dataset: AnnotatedSubset) -> dict[str, int]:
    """Traverse image IDs + annotations without decoding any image bytes."""
    total_bboxes = 0
    total_width = 0

    for image_id in dataset.image_ids:
        annotation = dataset.annotations[image_id]
        total_bboxes += annotation.num_bboxes
        total_width += 1

    return {
        "num_images": len(dataset.image_ids),
        "total_bboxes": total_bboxes,
        "width_checksum": total_width,
    }


def _time_iteration(dataset: AnnotatedSubset, runs: int) -> dict[str, Any]:
    """Time both iteration strategies and validate they traverse the same set."""
    legacy_timings: list[float] = []
    metadata_timings: list[float] = []
    legacy_checksums: list[dict[str, int]] = []
    metadata_checksums: list[dict[str, int]] = []

    for _ in range(runs):
        start = time.perf_counter()
        legacy_checksums.append(_legacy_image_loading_iteration(dataset))
        legacy_timings.append(time.perf_counter() - start)

        start = time.perf_counter()
        metadata_checksums.append(_metadata_only_iteration(dataset))
        metadata_timings.append(time.perf_counter() - start)

    if any(checksum != legacy_checksums[0] for checksum in legacy_checksums[1:]):
        raise RuntimeError("Legacy iteration checksums diverged between runs")
    if any(checksum["num_images"] != metadata_checksums[0]["num_images"] for checksum in metadata_checksums[1:]):
        raise RuntimeError("Metadata-only iteration image counts diverged between runs")
    if legacy_checksums[0]["num_images"] != metadata_checksums[0]["num_images"]:
        raise RuntimeError("Traversal modes visited different image counts")
    if legacy_checksums[0]["total_bboxes"] != metadata_checksums[0]["total_bboxes"]:
        raise RuntimeError("Traversal modes visited different annotation totals")

    legacy_median = statistics.median(legacy_timings)
    metadata_median = statistics.median(metadata_timings)

    return {
        "num_images": legacy_checksums[0]["num_images"],
        "total_bboxes": legacy_checksums[0]["total_bboxes"],
        "runs": runs,
        "legacy_image_loading": {
            "timings_seconds": legacy_timings,
            "median_seconds": legacy_median,
        },
        "metadata_only": {
            "timings_seconds": metadata_timings,
            "median_seconds": metadata_median,
        },
        "speedup_ratio": legacy_median / metadata_median if metadata_median > 0 else None,
    }


def _print_human_report(report: dict[str, Any], dataset_root: Path) -> None:
    print(f"Dataset root: {dataset_root}")
    print(f"Annotated images: {report['num_images']}")
    print(f"Total bboxes: {report['total_bboxes']}")
    print(f"Runs: {report['runs']}")
    print("")
    print("| Mode | Timings (s) | Median (s) |")
    print("| --- | --- | --- |")
    for label, key in (
        ("Legacy image loading", "legacy_image_loading"),
        ("Metadata only", "metadata_only"),
    ):
        timings = ", ".join(f"{value:.6f}" for value in report[key]["timings_seconds"])
        print(f"| {label} | {timings} | {report[key]['median_seconds']:.6f} |")

    speedup = report["speedup_ratio"]
    if speedup is not None:
        print("")
        print(f"Speedup (legacy / metadata): {speedup:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark legacy image-loading iteration versus metadata-only traversal.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_PATH,
        help="Path to the WikiChurches dataset root.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timing runs per traversal mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    args = parser.parse_args()

    dataset = AnnotatedSubset(args.dataset_root)
    report = _time_iteration(dataset, args.runs)

    if args.json:
        print(json.dumps({"dataset_root": str(args.dataset_root), **report}, indent=2))
    else:
        _print_human_report(report, args.dataset_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
