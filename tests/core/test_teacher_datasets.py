from pathlib import Path

import pytest

from cypher_3dsg_agent_distillation.core import config
from cypher_3dsg_agent_distillation.core.versioning import atomic_write_yaml, read_metadata


@pytest.fixture
def prepared_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GENERATED_DATASETS_ROOT", tmp_path / "datasets")
    config_path = tmp_path / "data.yaml"
    atomic_write_yaml(
        {
            "dataset_id": "demo",
            "question_sets": [
                {
                    "name": "qa",
                    "question_templates_path": "configs/question_templates/cypher_qa_3dsg.yaml",
                    "experiment_profile_path": (
                        "configs/experiment_profiles/cypher_qa_template.yaml"
                    ),
                }
            ],
            "scenes": [{"scene_id": "office", "scene_path": "/source/office.json"}],
        },
        config_path,
    )
    data = config.load_data_config(
        config_path,
        dataset_version="2026-01-01-T-00-00-00-Z",
        create_version=True,
    )
    for key in ("raw_questions_path", "seed_records_jsonl_path"):
        path = Path(data[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"input: {key}\n")
    atomic_write_yaml(
        {
            "version": data["dataset_version"],
            "status": "complete",
            "split_fingerprint": "old-hash",
            "stages": {"prepare-seeds": {"output": data["seed_records_jsonl_path"]}, "split": {}},
        },
        data["metadata_path"],
    )
    return data


@pytest.mark.parametrize("existing", ["complete", "partial", "legacy", "splits"])
def test_teacher_forks_inputs_and_preserves_old_artifacts(prepared_dataset, existing):
    data = prepared_dataset
    source = Path(data["dataset_dir"])
    if existing == "legacy":
        trace = source / "traces" / "old-teacher" / "2026-01-01-T-00-00-01-Z" / "accepted.json"
    elif existing == "splits":
        trace = Path(data["splits"]["train"])
    else:
        trace = source / "traces" / ("metadata.yaml" if existing == "partial" else "accepted.json")
    trace.parent.mkdir(parents=True)
    trace.write_text("original traces")
    report = source / "reports" / "split.report.json"
    report.parent.mkdir()
    report.write_text("original report")
    metadata_before = Path(data["metadata_path"]).read_bytes()

    result = config.prepare_teacher_dataset(data, teacher_name="new-teacher")
    target = Path(result["dataset_dir"])

    assert target != source
    assert trace.read_text() == "original traces"
    assert Path(data["metadata_path"]).read_bytes() == metadata_before
    assert not (target / "splits").exists()
    assert not (target / "reports").exists()
    for key in ("raw_questions_path", "seed_records_jsonl_path"):
        assert Path(result[key]).read_bytes() == Path(data[key]).read_bytes()
        assert Path(result[key]).is_relative_to(target)
    assert Path(result["teacher_traces_jsonl_path"]) == target / "traces" / "teacher_traces.jsonl"
    metadata = read_metadata(target)
    assert metadata["parent_dataset"]["version"] == data["dataset_version"]
    assert metadata["status"] == "building"
    assert "split_fingerprint" not in metadata
    assert "split" not in metadata["stages"]
    assert metadata["stages"]["prepare-seeds"]["output"] == result["seed_records_jsonl_path"]
    assert (
        config.load_data_config(data["data_config_path"])["dataset_version"]
        == result["dataset_version"]
    )


def test_first_teacher_uses_existing_dataset_and_second_always_forks(prepared_dataset):
    first = config.prepare_teacher_dataset(prepared_dataset, teacher_name="synthetic")
    second = config.prepare_teacher_dataset(first, teacher_name="synthetic")
    third = config.prepare_teacher_dataset(first, teacher_name="synthetic")
    assert first["dataset_version"] == prepared_dataset["dataset_version"]
    assert len({first["dataset_version"], second["dataset_version"], third["dataset_version"]}) == 3


def test_fork_refuses_missing_prepared_inputs(prepared_dataset):
    first = config.prepare_teacher_dataset(prepared_dataset, teacher_name="synthetic")
    Path(first["seed_records_jsonl_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="Prepared dataset input"):
        config.prepare_teacher_dataset(first, teacher_name="synthetic")


def test_loads_only_the_current_flat_teacher_layout(prepared_dataset):
    data = prepared_dataset
    root = Path(data["dataset_dir"])
    atomic_write_yaml({"teacher": {"teacher_name": "synthetic"}}, root / "metadata.yaml")
    loaded = config.load_data_config(data["data_config_path"])
    assert loaded["teacher_name"] == "synthetic"
    assert loaded["teacher_traces_jsonl_path"] == str(root / "traces" / "teacher_traces.jsonl")
    assert loaded["teacher_metadata_path"] == str(root / "traces/metadata.yaml")
