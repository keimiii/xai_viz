"""Tests for fine-tuning metadata-only split and label bookkeeping."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from PIL import Image
from torch import nn

from ssl_attention.config import STYLE_MAPPING
from ssl_attention.data.wikichurches import FullDataset
from ssl_attention.evaluation import fine_tuning as fine_tuning_module
from ssl_attention.evaluation import fine_tuning_artifacts as artifacts_module
from ssl_attention.evaluation.fine_tuning import (
    FineTunableModel,
    FineTuner,
    FineTuningConfig,
    FineTuningResult,
    FineTuningSplitArtifact,
    save_training_results,
)


class _NoGetItemFullDataset(FullDataset):
    """Dataset variant that fails if training setup touches image bytes."""

    def __getitem__(self, idx: int) -> Any:
        raise AssertionError(f"fine-tuning setup should not index dataset[{idx}]")


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    """Create a small classification dataset with two labeled classes."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    style_qids = list(STYLE_MAPPING.keys())
    assert len(style_qids) >= 2

    churches = {
        "Q100": {"styles": [style_qids[0]]},
        "Q101": {"styles": [style_qids[0]]},
        "Q102": {"styles": [style_qids[0]]},
        "Q200": {"styles": [style_qids[1]]},
        "Q201": {"styles": [style_qids[1]]},
        "Q202": {"styles": [style_qids[1]]},
        "Q999": {"styles": []},
    }

    for qid in churches:
        Image.new("RGB", (1, 1), "white").save(images_dir / f"{qid}_wd0.jpg")

    with open(tmp_path / "churches.json", "w", encoding="utf-8") as f:
        json.dump(churches, f)

    return tmp_path


@pytest.fixture()
def full_dataset(dataset_root: Path) -> FullDataset:
    return FullDataset(dataset_root)


@pytest.fixture()
def metadata_only_dataset(dataset_root: Path) -> _NoGetItemFullDataset:
    return _NoGetItemFullDataset(dataset_root)


def _label_counter(dataset: FullDataset, indices: list[int]) -> Counter[int]:
    labels = [
        dataset.get_metadata(idx)["style_label"]
        for idx in indices
        if dataset.get_metadata(idx)["style_label"] is not None
    ]
    return Counter(label for label in labels if label is not None)


class TestFineTunerMetadataSetup:
    """Training setup should stay on metadata-only paths."""

    def test_stratified_split_is_deterministic_for_fixed_seed(self, full_dataset: FullDataset) -> None:
        config = FineTuningConfig(model_name="dinov2", seed=123, val_split=0.5)

        tuner_a = FineTuner(config)
        train_a, val_a, excluded_a = tuner_a._stratified_split(full_dataset, config.val_split)

        tuner_b = FineTuner(config)
        train_b, val_b, excluded_b = tuner_b._stratified_split(full_dataset, config.val_split)

        assert train_a == train_b
        assert val_a == val_b
        assert excluded_a == excluded_b == 0

    def test_stratified_split_excludes_eval_ids_and_preserves_label_distribution(
        self,
        full_dataset: FullDataset,
    ) -> None:
        config = FineTuningConfig(model_name="dinov2", seed=7, val_split=0.5)
        tuner = FineTuner(config)
        exclude_image_ids = {"Q100_wd0.jpg", "Q200_wd0.jpg"}

        train_subset, val_subset, n_excluded = tuner._stratified_split(
            full_dataset,
            config.val_split,
            exclude_image_ids=exclude_image_ids,
        )

        train_ids = {full_dataset.get_metadata(idx)["image_id"] for idx in train_subset}
        val_ids = {full_dataset.get_metadata(idx)["image_id"] for idx in val_subset}

        included_indices = [
            idx
            for idx in range(len(full_dataset))
            if full_dataset.get_metadata(idx)["image_id"] not in exclude_image_ids
            and full_dataset.get_metadata(idx)["style_label"] is not None
        ]
        included_counts = _label_counter(full_dataset, included_indices)
        train_counts = _label_counter(full_dataset, train_subset)
        val_counts = _label_counter(full_dataset, val_subset)

        assert n_excluded == 2
        assert exclude_image_ids.isdisjoint(train_ids)
        assert exclude_image_ids.isdisjoint(val_ids)
        assert train_counts + val_counts == included_counts

        for label, count in included_counts.items():
            expected_val = max(1, int(count * config.val_split))
            assert val_counts[label] == expected_val
            assert train_counts[label] == count - expected_val

    def test_collect_labels_for_indices_uses_metadata_only(
        self,
        metadata_only_dataset: _NoGetItemFullDataset,
    ) -> None:
        config = FineTuningConfig(model_name="dinov2", seed=11, val_split=0.5)
        tuner = FineTuner(config)

        labels = tuner._collect_labels_for_indices(
            metadata_only_dataset,
            list(range(len(metadata_only_dataset))),
        )

        assert labels == [0, 0, 0, 1, 1, 1]

    def test_shared_split_artifact_is_reused_across_runs(
        self,
        full_dataset: FullDataset,
        dataset_root: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        experiment_id = "exp_shared_split"
        split_path = tmp_path / "shared_split.json"
        eval_image_ids = {"Q100_wd0.jpg", "Q200_wd0.jpg"}

        monkeypatch.setattr(fine_tuning_module, "DATASET_PATH", dataset_root)

        config_a = FineTuningConfig(
            model_name="dinov2",
            experiment_id=experiment_id,
            seed=7,
            val_split=0.5,
        )
        config_b = FineTuningConfig(
            model_name="clip",
            experiment_id=experiment_id,
            seed=7,
            val_split=0.5,
            use_lora=True,
        )

        artifact_a = FineTuner(config_a)._load_or_create_split_artifact(
            full_dataset,
            eval_image_ids,
            split_path,
        )
        artifact_b = FineTuner(config_b)._load_or_create_split_artifact(
            full_dataset,
            eval_image_ids,
            split_path,
        )

        assert split_path.exists()
        assert artifact_a.split_id == artifact_b.split_id == config_a.split_id == config_b.split_id
        assert artifact_a.policy == artifact_b.policy == fine_tuning_module.PRIMARY_SPLIT_POLICY
        assert artifact_a.annotated_eval_image_ids == sorted(eval_image_ids)
        assert artifact_a.train_image_ids == artifact_b.train_image_ids
        assert artifact_a.val_image_ids == artifact_b.val_image_ids
        assert set(artifact_a.annotated_eval_image_ids).isdisjoint(artifact_a.train_image_ids)
        assert set(artifact_a.annotated_eval_image_ids).isdisjoint(artifact_a.val_image_ids)

    def test_save_run_manifest_records_primary_provenance(
        self,
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
        monkeypatch.setattr(fine_tuning_module, "get_git_commit_sha", lambda: "abc123")

        config = FineTuningConfig(model_name="dinov2", experiment_id="exp_manifest", seed=99)
        tuner = FineTuner(config)
        experiment_paths = artifacts_module.ensure_experiment_layout(config.experiment_id)
        checkpoint_path = experiment_paths.checkpoints_dir / "dinov2_full_finetuned.pt"
        checkpoint_path.write_text("checkpoint", encoding="utf-8")

        split_artifact = FineTuningSplitArtifact(
            split_id=config.split_id or "missing",
            experiment_id=config.experiment_id,
            seed=config.seed,
            dataset_root="data/wikichurches",
            dataset_version_hint={"churches_json_mtime": None},
            policy=fine_tuning_module.PRIMARY_SPLIT_POLICY,
            exclude_annotated_from_train=True,
            exclude_annotated_from_val=True,
            annotated_eval_image_ids=["Q100_wd0.jpg"],
            train_image_ids=["Q101_wd0.jpg", "Q201_wd0.jpg"],
            val_image_ids=["Q102_wd0.jpg", "Q202_wd0.jpg"],
            train_class_counts={"Romanesque": 1, "Gothic": 1},
            val_class_counts={"Romanesque": 1, "Gothic": 1},
            created_at="2026-03-27T00:00:00+00:00",
        )

        manifest_path = tuner._save_run_manifest(
            experiment_paths,
            checkpoint_path,
            split_artifact,
            config.strategy_id,
            best_epoch=4,
            best_val_acc=0.87,
            num_train_samples=2,
            num_val_samples=2,
            num_excluded_eval_samples=1,
        )

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == config.run_id
        assert payload["experiment_id"] == config.experiment_id
        assert payload["run_scope"] == "primary"
        assert payload["split_id"] == split_artifact.split_id
        assert payload["checkpoint_path"] == "outputs/checkpoints/exp_manifest/dinov2_full_finetuned.pt"
        assert payload["manifest_path"] == (
            "outputs/results/experiments/exp_manifest/manifests/"
            f"{config.run_id}_manifest.json"
        )
        assert payload["split_artifact_path"] == (
            "outputs/results/experiments/exp_manifest/splits/"
            f"{split_artifact.split_id}.json"
        )
        assert payload["training_git_commit_sha"] == "abc123"
        assert "git_commit_sha" not in payload
        assert payload["checkpoint_selection_metric"] == fine_tuning_module.PRIMARY_SELECTION_METRIC
        assert payload["checkpoint_selection_split"] == fine_tuning_module.PRIMARY_SPLIT_POLICY
        assert payload["selected_epoch"] == 4
        assert payload["best_val_score"] == pytest.approx(0.87)


class _DummyBackbone(nn.Module):
    """Minimal backbone stub for attention-extraction tests."""

    def __init__(self) -> None:
        super().__init__()
        self.training_states: list[bool] = []

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = True) -> SimpleNamespace:
        self.training_states.append(self.training)
        return SimpleNamespace(
            last_hidden_state=torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
            attentions=[torch.ones((1, 1, 5, 5), dtype=torch.float32)],
        )


def test_extract_attention_uses_eval_mode_and_restores_training_state() -> None:
    backbone = _DummyBackbone()
    backbone.train()

    model = FineTunableModel.__new__(FineTunableModel)
    nn.Module.__init__(model)
    model.backbone = backbone
    model.model_name = "dinov2"
    model._config = cast(Any, SimpleNamespace(num_registers=1))

    output = FineTunableModel.extract_attention(model, torch.zeros((1, 3, 224, 224)))

    assert backbone.training_states == [False]
    assert backbone.training is True
    assert output.cls_token.shape == (1, 3)
    assert output.patch_tokens is not None
    assert output.patch_tokens.shape == (1, 3, 3)


class _UnstableMAEBackbone(nn.Module):
    """MAE-like backbone that is stable only when deterministic noise is supplied."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(patch_size=16)
        self.call_index = 0
        self.noise_inputs: list[torch.Tensor | None] = []
        self.training_states: list[bool] = []

    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        self.call_index += 1
        self.training_states.append(self.training)
        self.noise_inputs.append(None if noise is None else noise.detach().clone())

        batch_size = pixel_values.shape[0]
        seq_length = (pixel_values.shape[-2] // 16) * (pixel_values.shape[-1] // 16)
        patch_values = (
            noise
            if noise is not None
            else torch.full((batch_size, seq_length), float(self.call_index), dtype=torch.float32)
        )

        cls_token = torch.zeros((batch_size, 1, 1), dtype=torch.float32)
        patch_tokens = patch_values.unsqueeze(-1)
        last_hidden_state = torch.cat((cls_token, patch_tokens), dim=1)

        attention = torch.zeros((batch_size, 1, seq_length + 1, seq_length + 1), dtype=torch.float32)
        attention[:, 0, 0, 1:] = patch_values

        hidden_states = None
        if output_hidden_states:
            hidden_states = (torch.zeros_like(last_hidden_state), last_hidden_state)

        return SimpleNamespace(
            last_hidden_state=last_hidden_state,
            attentions=(attention,) if output_attentions else (),
            hidden_states=hidden_states,
        )


def test_extract_attention_mae_is_repeatable_and_restores_training_state() -> None:
    backbone = _UnstableMAEBackbone()
    backbone.train()

    model = FineTunableModel.__new__(FineTunableModel)
    nn.Module.__init__(model)
    model.backbone = backbone
    model.model_name = "mae"
    model._config = cast(Any, SimpleNamespace(num_registers=0))

    pixel_values = torch.zeros((2, 3, 224, 224))
    output_a = FineTunableModel.extract_attention(model, pixel_values)
    output_b = FineTunableModel.extract_attention(model, pixel_values)

    expected = torch.arange(196, dtype=torch.float32).expand(2, -1)
    first_noise = backbone.noise_inputs[0]
    second_noise = backbone.noise_inputs[1]
    assert first_noise is not None
    assert second_noise is not None
    assert torch.equal(first_noise, expected)
    assert torch.equal(second_noise, expected)
    assert backbone.training_states == [False, False]
    assert backbone.training is True
    assert output_a.patch_tokens is not None
    assert output_b.patch_tokens is not None
    assert torch.equal(output_a.patch_tokens, output_b.patch_tokens)
    assert torch.equal(output_a.attention_weights[0], output_b.attention_weights[0])


def test_save_training_results_merges_existing_experiment_runs(
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

    result_a = FineTuningResult(
        model_name="clip",
        strategy_id="linear_probe",
        best_val_acc=0.8,
        best_epoch=1,
        train_history=[],
        checkpoint_path=tmp_path / "outputs" / "checkpoints" / "exp_batch" / "clip_linear_probe_finetuned.pt",
        manifest_path=tmp_path / "outputs" / "results" / "experiments" / "exp_batch" / "manifests" / "a.json",
        experiment_id="exp_batch",
        run_id="exp_batch__clip__linear_probe",
        split_id="exp_batch__primary__seed42",
        split_artifact_path=tmp_path / "outputs" / "results" / "experiments" / "exp_batch" / "splits" / "exp_batch__primary__seed42.json",
        run_scope="primary",
        training_git_commit_sha="abc123",
        config={"checkpoint_selection_metric": fine_tuning_module.PRIMARY_SELECTION_METRIC},
    )
    result_b = FineTuningResult(
        model_name="clip",
        strategy_id="full",
        best_val_acc=0.9,
        best_epoch=2,
        train_history=[],
        checkpoint_path=tmp_path / "outputs" / "checkpoints" / "exp_batch" / "clip_full_finetuned.pt",
        manifest_path=tmp_path / "outputs" / "results" / "experiments" / "exp_batch" / "manifests" / "b.json",
        experiment_id="exp_batch",
        run_id="exp_batch__clip__full",
        split_id="exp_batch__primary__seed42",
        split_artifact_path=tmp_path / "outputs" / "results" / "experiments" / "exp_batch" / "splits" / "exp_batch__primary__seed42.json",
        run_scope="primary",
        training_git_commit_sha="abc123",
        config={"checkpoint_selection_metric": fine_tuning_module.PRIMARY_SELECTION_METRIC},
    )

    save_training_results([result_a])
    save_training_results([result_b])

    payload = json.loads(
        (
            tmp_path
            / "outputs"
            / "results"
            / "experiments"
            / "exp_batch"
            / "fine_tuning_results.json"
        ).read_text(encoding="utf-8")
    )

    assert [run["run_id"] for run in payload["runs"]] == [
        "exp_batch__clip__full",
        "exp_batch__clip__linear_probe",
    ]
    assert {run["training_git_commit_sha"] for run in payload["runs"]} == {"abc123"}
    assert all("git_commit_sha" not in run for run in payload["runs"])

    run_matrix_payload = json.loads(
        (
            tmp_path
            / "outputs"
            / "results"
            / "experiments"
            / "exp_batch"
            / "run_matrix.json"
        ).read_text(encoding="utf-8")
    )
    run_entries = list(run_matrix_payload["runs"].values())
    assert {run["training_git_commit_sha"] for run in run_entries} == {"abc123"}
    assert all("git_commit_sha" not in run for run in run_entries)
