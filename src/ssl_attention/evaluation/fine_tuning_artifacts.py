"""Helpers for experiment-scoped fine-tuning artifacts and provenance.

This module keeps the experiment-batch bookkeeping light enough to import from
training scripts, analysis scripts, and the backend config without pulling in
heavy ML dependencies.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "results"
EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
ACTIVE_EXPERIMENT_PATH = RESULTS_ROOT / "active_experiment.json"

LEGACY_GIT_COMMIT_SHA_FIELD = "git_commit_sha"
TRAINING_GIT_COMMIT_SHA_FIELD = "training_git_commit_sha"
ANALYSIS_GIT_COMMIT_SHA_FIELD = "analysis_git_commit_sha"


@dataclass(frozen=True)
class ExperimentPaths:
    """Canonical filesystem layout for one fine-tuning experiment batch."""

    experiment_id: str
    results_dir: Path
    manifests_dir: Path
    split_artifacts_dir: Path
    checkpoints_dir: Path
    run_matrix_path: Path
    fine_tuning_results_path: Path
    q2_metrics_path: Path
    q2_delta_iou_path: Path


def make_experiment_id(*, prefix: str = "fine_tuning") -> str:
    """Create a timestamped default experiment identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def build_run_id(experiment_id: str, model_name: str, strategy_id: str) -> str:
    """Build a stable run identifier for one model/strategy inside an experiment."""
    return f"{experiment_id}__{model_name}__{strategy_id}"


def get_experiment_paths(experiment_id: str) -> ExperimentPaths:
    """Return the canonical artifact paths for an experiment batch."""
    results_dir = EXPERIMENTS_ROOT / experiment_id
    return ExperimentPaths(
        experiment_id=experiment_id,
        results_dir=results_dir,
        manifests_dir=results_dir / "manifests",
        split_artifacts_dir=results_dir / "splits",
        checkpoints_dir=CHECKPOINTS_ROOT / experiment_id,
        run_matrix_path=results_dir / "run_matrix.json",
        fine_tuning_results_path=results_dir / "fine_tuning_results.json",
        q2_metrics_path=results_dir / "q2_metrics_analysis.json",
        q2_delta_iou_path=results_dir / "q2_delta_iou_analysis.json",
    )


def ensure_experiment_layout(experiment_id: str) -> ExperimentPaths:
    """Create the canonical directory layout for an experiment batch."""
    paths = get_experiment_paths(experiment_id)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    paths.split_artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return paths


def repo_relative_path(path: Path) -> str:
    """Serialize a project-local path relative to the repository root."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def project_path(path_str: str | None) -> Path | None:
    """Resolve a repo-relative path stored in JSON back to an absolute path."""
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_dataset_version_hint(dataset_root: Path) -> dict[str, Any]:
    """Create a lightweight dataset fingerprint for provenance records."""
    churches_path = dataset_root / "churches.json"
    annotations_path = dataset_root / "building_parts.json"
    images_path = dataset_root / "images"

    def _mtime_iso(path: Path) -> str | None:
        if not path.exists():
            return None
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()

    return {
        "churches_json_mtime": _mtime_iso(churches_path),
        "annotations_mtime": _mtime_iso(annotations_path),
        "image_dir_mtime": _mtime_iso(images_path),
    }


def get_git_commit_sha() -> str | None:
    """Return the current git commit SHA if the repo is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _normalize_git_commit_field(
    payload: dict[str, Any],
    *,
    target_field: str,
    drop_legacy: bool,
) -> dict[str, Any]:
    """Map legacy commit provenance onto an explicit field name."""
    normalized = dict(payload)
    legacy_value = normalized.get(LEGACY_GIT_COMMIT_SHA_FIELD)
    if normalized.get(target_field) is None and legacy_value is not None:
        normalized[target_field] = legacy_value
    if drop_legacy:
        normalized.pop(LEGACY_GIT_COMMIT_SHA_FIELD, None)
    return normalized


def normalize_training_provenance(
    payload: dict[str, Any],
    *,
    drop_legacy: bool = False,
) -> dict[str, Any]:
    """Normalize legacy training provenance to the explicit training field."""
    return _normalize_git_commit_field(
        payload,
        target_field=TRAINING_GIT_COMMIT_SHA_FIELD,
        drop_legacy=drop_legacy,
    )


def normalize_analysis_provenance(
    payload: dict[str, Any],
    *,
    drop_legacy: bool = False,
) -> dict[str, Any]:
    """Normalize legacy analysis provenance to the explicit analysis field."""
    return _normalize_git_commit_field(
        payload,
        target_field=ANALYSIS_GIT_COMMIT_SHA_FIELD,
        drop_legacy=drop_legacy,
    )


def normalize_run_manifest_payload(
    payload: dict[str, Any],
    *,
    drop_legacy: bool = False,
) -> dict[str, Any]:
    """Normalize one run manifest to the explicit training provenance field."""
    return normalize_training_provenance(payload, drop_legacy=drop_legacy)


def normalize_fine_tuning_results_payload(
    payload: dict[str, Any],
    *,
    drop_legacy: bool = False,
) -> dict[str, Any]:
    """Normalize one training-results ledger payload."""
    normalized = dict(payload)
    normalized["runs"] = [
        normalize_training_provenance(run, drop_legacy=drop_legacy)
        if isinstance(run, dict)
        else run
        for run in payload.get("runs", [])
    ]
    return normalized


def normalize_run_matrix_payload(
    payload: dict[str, Any],
    *,
    drop_legacy: bool = False,
) -> dict[str, Any]:
    """Normalize one run-matrix payload across training and analysis provenance."""
    normalized = dict(payload)
    runs: dict[str, Any] = {}
    for run_id, run in payload.get("runs", {}).items():
        if not isinstance(run, dict):
            runs[run_id] = run
            continue
        normalized_run = normalize_training_provenance(run, drop_legacy=drop_legacy)
        if ANALYSIS_GIT_COMMIT_SHA_FIELD in run:
            normalized_run[ANALYSIS_GIT_COMMIT_SHA_FIELD] = run.get(ANALYSIS_GIT_COMMIT_SHA_FIELD)
        runs[run_id] = normalized_run
    normalized["runs"] = runs
    if drop_legacy:
        normalized.pop(LEGACY_GIT_COMMIT_SHA_FIELD, None)
    return normalized


def normalize_q2_analysis_payload(
    payload: dict[str, Any],
    *,
    drop_legacy: bool = False,
) -> dict[str, Any]:
    """Normalize one Q2 analysis payload to the explicit analysis field."""
    return normalize_analysis_provenance(payload, drop_legacy=drop_legacy)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON if it exists."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_active_experiment() -> dict[str, Any] | None:
    """Load the active-experiment pointer if it exists."""
    return load_json(ACTIVE_EXPERIMENT_PATH)


def write_active_experiment(
    experiment_id: str,
    *,
    split_id: str | None = None,
    run_matrix_path: Path | None = None,
    fine_tuning_results_path: Path | None = None,
    q2_metrics_path: Path | None = None,
    q2_delta_iou_path: Path | None = None,
) -> dict[str, Any]:
    """Persist the active-experiment pointer for the app/docs/figure pipeline."""
    existing = load_active_experiment() or {}
    payload = {
        **existing,
        "experiment_id": experiment_id,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if split_id is not None:
        payload["split_id"] = split_id
    if run_matrix_path is not None:
        payload["run_matrix_path"] = repo_relative_path(run_matrix_path)
    if fine_tuning_results_path is not None:
        payload["fine_tuning_results_path"] = repo_relative_path(fine_tuning_results_path)
    if q2_metrics_path is not None:
        payload["q2_metrics_path"] = repo_relative_path(q2_metrics_path)
    if q2_delta_iou_path is not None:
        payload["q2_delta_iou_path"] = repo_relative_path(q2_delta_iou_path)
    save_json(ACTIVE_EXPERIMENT_PATH, payload)
    return payload


def resolve_active_artifact_path(artifact_key: str, legacy_path: Path) -> Path:
    """Resolve an artifact path through the active-experiment pointer.

    Falls back to the legacy repository-level path when no pointer or keyed path
    is available yet.
    """
    active = load_active_experiment()
    if not active:
        return legacy_path
    resolved = project_path(active.get(artifact_key))
    if resolved is None:
        return legacy_path
    return resolved


def load_run_matrix(experiment_id: str) -> dict[str, Any]:
    """Load one experiment's run matrix, returning an empty skeleton if needed."""
    paths = get_experiment_paths(experiment_id)
    existing = load_json(paths.run_matrix_path)
    if existing is not None:
        return normalize_run_matrix_payload(existing, drop_legacy=True)
    return {
        "experiment_id": experiment_id,
        "selection_rule": "best classification validation accuracy on shared non-annotated validation split",
        "runs": {},
    }


def save_run_matrix(experiment_id: str, payload: dict[str, Any]) -> None:
    """Persist a run matrix payload for one experiment."""
    paths = ensure_experiment_layout(experiment_id)
    payload = normalize_run_matrix_payload(
        {
        **payload,
        "experiment_id": experiment_id,
        "updated_at": datetime.now(UTC).isoformat(),
        },
        drop_legacy=True,
    )
    save_json(paths.run_matrix_path, payload)


def refresh_experiment_training_provenance(experiment_id: str) -> dict[str, int]:
    """Rewrite legacy training provenance fields for one experiment batch."""
    paths = get_experiment_paths(experiment_id)
    refreshed = {
        "manifests": 0,
        "fine_tuning_results": 0,
        "run_matrix": 0,
    }

    for manifest_path in sorted(paths.manifests_dir.glob("*_manifest.json")):
        payload = load_json(manifest_path)
        if payload is None:
            continue
        save_json(manifest_path, normalize_run_manifest_payload(payload, drop_legacy=True))
        refreshed["manifests"] += 1

    fine_tuning_results = load_json(paths.fine_tuning_results_path)
    if fine_tuning_results is not None:
        save_json(
            paths.fine_tuning_results_path,
            normalize_fine_tuning_results_payload(fine_tuning_results, drop_legacy=True),
        )
        refreshed["fine_tuning_results"] = 1

    run_matrix = load_json(paths.run_matrix_path)
    if run_matrix is not None:
        save_json(paths.run_matrix_path, normalize_run_matrix_payload(run_matrix, drop_legacy=True))
        refreshed["run_matrix"] = 1

    return refreshed
