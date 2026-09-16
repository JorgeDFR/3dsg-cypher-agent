from __future__ import annotations

import sys
from datetime import UTC, datetime

import pytest

from cypher_3dsg_agent_distillation.core.config import teacher_run_paths
from cypher_3dsg_agent_distillation.core.versioning import (
    atomic_write_yaml,
    confirm_dataset_lineage,
    create_version_dir,
    redact_secrets,
    resolve_version_dir,
    validate_version,
    version_id,
)


def test_readable_utc_version_format():
    assert version_id(datetime(2026, 9, 8, 12, 5, 47, tzinfo=UTC)) == (
        "2026-09-08-T-12-05-47-Z"
    )
    assert validate_version("2026-09-08-T-12-05-47-Z") == (
        "2026-09-08-T-12-05-47-Z"
    )


def test_invalid_or_path_like_versions_are_rejected():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_version("../latest")


def test_metadata_redacts_credentials_but_keeps_generation_token_limits():
    assert redact_secrets(
        {"api_key": "private", "password": "private", "max_tokens": 2048}
    ) == {"api_key": "<redacted>", "password": "<redacted>", "max_tokens": 2048}


def test_latest_resolution_does_not_depend_on_contiguous_versions(tmp_path):
    root = tmp_path / "versions"
    first, _ = create_version_dir(root, "2026-09-01-T-10-00-00-Z")
    _, middle_dir = create_version_dir(root, "2026-09-02-T-10-00-00-Z")
    last, _ = create_version_dir(root, "2026-09-03-T-10-00-00-Z")
    middle_dir.rmdir()

    selected, _ = resolve_version_dir(root)

    assert first != last
    assert selected == last


def test_latest_completed_model_ignores_interrupted_run(tmp_path):
    root = tmp_path / "models"
    complete, complete_dir = create_version_dir(root, "2026-09-01-T-10-00-00-Z")
    _, running_dir = create_version_dir(root, "2026-09-02-T-10-00-00-Z")
    atomic_write_yaml({"status": "complete"}, complete_dir / "metadata.yaml")
    atomic_write_yaml({"status": "running"}, running_dir / "metadata.yaml")

    selected, _ = resolve_version_dir(root, completed_only=True)

    assert selected == complete


def test_teacher_paths_are_grouped_by_teacher_then_version(tmp_path):
    data_config = {"dataset_dir": str(tmp_path / "dataset")}

    paths = teacher_run_paths(
        data_config,
        teacher_name="openai/gpt-oss-120b",
        teacher_run="2026-09-08-T-12-05-47-Z",
        create_version=True,
    )

    assert paths.run_dir == (
        tmp_path
        / "dataset"
        / "traces"
        / "openai-gpt-oss-120b"
        / "2026-09-08-T-12-05-47-Z"
    )


def test_noninteractive_lineage_mismatch_requires_explicit_override(tmp_path):
    split = tmp_path / "test.jsonl"
    split.write_text("{}\n", encoding="utf-8")
    data_config = {
        "scene_id": "scene-b",
        "dataset_version": "2026-09-02-T-10-00-00-Z",
        "dataset_dir": str(tmp_path),
        "splits": {"test": str(split)},
    }
    metadata = {
        "dataset": {
            "scene_id": "scene-a",
            "version": "2026-09-01-T-10-00-00-Z",
        }
    }

    with pytest.raises(RuntimeError, match="allow-dataset-mismatch"):
        confirm_dataset_lineage(model_metadata=metadata, data_config=data_config)

    assert confirm_dataset_lineage(
        model_metadata=metadata,
        data_config=data_config,
        allow_mismatch=True,
    ) is True


@pytest.mark.parametrize(
    ("response", "expected"),
    [("y", True), ("yes", True), ("n", False), ("", False)],
)
def test_interactive_lineage_confirmation_returns_user_choice(
    monkeypatch, capsys, response, expected
):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: response)

    confirmed = confirm_dataset_lineage(
        model_metadata={
            "dataset": {"dataset_id": "trained", "version": "trained-version"}
        },
        data_config={
            "dataset_id": "selected",
            "dataset_version": "selected-version",
        },
    )

    assert confirmed is expected
    stderr = capsys.readouterr().err
    assert "Model/dataset lineage mismatch" in stderr
    assert ("Cancelled:" in stderr) is (not expected)
