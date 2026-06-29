"""Tests for strategy-aware fine-tuning artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssl_attention.evaluation import fine_tuning_artifacts as artifacts_module
from ssl_attention.evaluation.fine_tuning import (
    get_checkpoint_candidates,
    get_checkpoint_filename,
    get_finetuned_cache_key,
    infer_strategy_id,
    load_run_manifest,
)


def test_infer_strategy_id() -> None:
    assert infer_strategy_id(freeze_backbone=True, use_lora=False) == "linear_probe"
    assert infer_strategy_id(freeze_backbone=False, use_lora=True) == "lora"
    assert infer_strategy_id(freeze_backbone=False, use_lora=False) == "full"


def test_checkpoint_filename_strategy_aware() -> None:
    assert get_checkpoint_filename("dinov2", "lora") == "dinov2_lora_finetuned.pt"
    assert get_checkpoint_filename("clip", "linear_probe") == "clip_linear_probe_finetuned.pt"


def test_finetuned_cache_key_strategy_aware() -> None:
    assert get_finetuned_cache_key("mae") == "mae_finetuned"
    assert get_finetuned_cache_key("mae", "full") == "mae_finetuned_full"


def test_checkpoint_candidates_include_legacy_for_full_only() -> None:
    full_candidates = get_checkpoint_candidates("dinov2", strategy_id="full")
    lp_candidates = get_checkpoint_candidates("dinov2", strategy_id="linear_probe")

    assert full_candidates[0].name == "dinov2_full_finetuned.pt"
    assert full_candidates[1].name == "dinov2_finetuned.pt"
    assert lp_candidates[0].name == "dinov2_linear_probe_finetuned.pt"
    assert {candidate.name for candidate in lp_candidates} == {"dinov2_linear_probe_finetuned.pt"}


def test_load_run_manifest(tmp_path: Path) -> None:
    manifest = {
        "model": "dinov2",
        "strategy": "lora",
        "seed": 42,
        "epochs": 10,
        "checkpoint_path": "outputs/checkpoints/dinov2_lora_finetuned.pt",
        "git_commit_sha": "abc123",
        "split": {"train_samples": 10, "val_samples": 2, "excluded_eval_samples": 1, "val_split": 0.2},
    }
    manifest_path = tmp_path / "dinov2_lora_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_run_manifest("dinov2", "lora", manifests_dir=tmp_path)

    assert loaded["model"] == "dinov2"
    assert loaded["strategy"] == "lora"
    assert loaded["training_git_commit_sha"] == "abc123"
    assert "git_commit_sha" not in loaded


def test_refresh_experiment_training_provenance_rewrites_legacy_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(artifacts_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(artifacts_module, "CHECKPOINTS_ROOT", tmp_path / "outputs" / "checkpoints")
    monkeypatch.setattr(artifacts_module, "RESULTS_ROOT", tmp_path / "outputs" / "results")
    monkeypatch.setattr(
        artifacts_module,
        "EXPERIMENTS_ROOT",
        artifacts_module.RESULTS_ROOT / "experiments",
    )
    monkeypatch.setattr(
        artifacts_module,
        "ACTIVE_EXPERIMENT_PATH",
        artifacts_module.RESULTS_ROOT / "active_experiment.json",
    )

    experiment_id = "exp_refresh"
    paths = artifacts_module.ensure_experiment_layout(experiment_id)
    manifest_path = paths.manifests_dir / f"{experiment_id}__dinov2__lora_manifest.json"
    artifacts_module.save_json(
        manifest_path,
        {
            "run_id": f"{experiment_id}__dinov2__lora",
            "git_commit_sha": "trainsha",
        },
    )
    artifacts_module.save_json(
        paths.fine_tuning_results_path,
        {
            "experiment_id": experiment_id,
            "runs": [
                {
                    "run_id": f"{experiment_id}__dinov2__lora",
                    "git_commit_sha": "trainsha",
                }
            ],
        },
    )
    artifacts_module.save_json(
        paths.run_matrix_path,
        {
            "experiment_id": experiment_id,
            "runs": {
                f"{experiment_id}__dinov2__lora": {
                    "run_id": f"{experiment_id}__dinov2__lora",
                    "git_commit_sha": "trainsha",
                    "analysis_artifact_paths": {},
                }
            },
        },
    )

    refreshed = artifacts_module.refresh_experiment_training_provenance(experiment_id)

    assert refreshed == {
        "manifests": 1,
        "fine_tuning_results": 1,
        "run_matrix": 1,
    }

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["training_git_commit_sha"] == "trainsha"
    assert "git_commit_sha" not in manifest_payload

    fine_tuning_results_payload = json.loads(paths.fine_tuning_results_path.read_text(encoding="utf-8"))
    assert fine_tuning_results_payload["runs"][0]["training_git_commit_sha"] == "trainsha"
    assert "git_commit_sha" not in fine_tuning_results_payload["runs"][0]

    run_matrix_payload = json.loads(paths.run_matrix_path.read_text(encoding="utf-8"))
    run_entry = run_matrix_payload["runs"][f"{experiment_id}__dinov2__lora"]
    assert run_entry["training_git_commit_sha"] == "trainsha"
    assert "analysis_git_commit_sha" not in run_entry
    assert "git_commit_sha" not in run_entry


def test_load_run_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run_manifest("dinov2", "full", manifests_dir=tmp_path)
