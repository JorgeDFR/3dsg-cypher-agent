from pathlib import Path

import pytest

from cypher_3dsg_agent_distillation.core import config as config_module
from cypher_3dsg_agent_distillation.core.config import (
    PROJECT_ROOT,
    data_paths_from_config,
    load_data_config,
    model_reports_dir,
    ollama_export_metadata,
    teacher_name_from_config,
)


def test_data_paths_keep_scene_path_explicit_and_derive_generated_scene_dir():
    version = "2026-09-08-T-12-05-47-Z"
    paths = data_paths_from_config(
        {
            "scene_id": "office_scene",
            "scene_path": "/tmp/hydra/scene_graph.json",
            "schema_path": "schema_3dsg.yaml",
            "experiment_profile_path": "cypher_qa_template.yaml",
            "question_templates_path": "cypher_qa_3dsg.yaml",
        },
        dataset_version=version,
    )

    assert paths.scene_path == Path("/tmp/hydra/scene_graph.json")
    assert paths.schema_path == PROJECT_ROOT / "configs" / "schema_3dsg.yaml"
    assert paths.experiment_profile_path == (
        PROJECT_ROOT / "configs" / "experiment_profiles" / "cypher_qa_template.yaml"
    )
    assert paths.question_templates_path == (
        PROJECT_ROOT / "configs" / "question_templates" / "cypher_qa_3dsg.yaml"
    )
    assert paths.raw_questions_path == (
        PROJECT_ROOT
        / "data"
        / "datasets"
        / "office_scene"
        / version
        / "generated_questions.yaml"
    )
    assert paths.seed_records_jsonl_path == (
        PROJECT_ROOT
        / "data"
        / "datasets"
        / "office_scene"
        / version
        / "cypher_seed_records.jsonl"
    )
    assert paths.teacher_traces_jsonl_path == (
        PROJECT_ROOT
        / "data"
        / "datasets"
        / "office_scene"
        / version
        / "traces"
        / "teacher_traces.jsonl"
    )
    assert paths.accepted_traces_path == (
        PROJECT_ROOT
        / "data"
        / "datasets"
        / "office_scene"
        / version
        / "traces"
        / "accepted.json"
    )
    assert paths.splits["test"] == (
        PROJECT_ROOT
        / "data"
        / "datasets"
        / "office_scene"
        / version
        / "splits"
        / "test.jsonl"
    )


def test_load_data_config_expands_derived_paths(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated"
    version = "2026-09-08-T-12-05-47-Z"
    (generated_root / "uhumans2" / version).mkdir(parents=True)
    monkeypatch.setattr(config_module, "GENERATED_DATASETS_ROOT", generated_root)
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "scene_id: uhumans2",
                "scene_path: data/scenes/uhumans2_office.json",
                "schema_path: schema_3dsg.yaml",
            ]
        ),
        encoding="utf-8",
    )

    config = load_data_config(config_path)

    assert config["scene_path"] == str(PROJECT_ROOT / "data" / "scenes" / "uhumans2_office.json")
    assert config["schema_path"] == str(PROJECT_ROOT / "configs" / "schema_3dsg.yaml")
    assert config["teacher_traces_jsonl_path"] == str(
        generated_root / "uhumans2" / version / "traces" / "teacher_traces.jsonl"
    )
    assert config["accepted_traces_path"] == str(
        generated_root / "uhumans2" / version / "traces" / "accepted.json"
    )
    assert config["splits"]["train"] == str(
        generated_root / "uhumans2" / version / "splits" / "train.jsonl"
    )


def test_scene_id_must_not_be_a_path():
    with pytest.raises(ValueError, match="single folder name"):
        data_paths_from_config(
            {
                "scene_id": "../uhumans2",
                "scene_path": "/tmp/scene.json",
            }
        )


def test_multi_scene_dataset_paths_share_outputs_and_isolate_scene_values():
    version = "2026-09-14-T-15-34-47-Z"
    paths = data_paths_from_config(
        {
            "dataset_id": "office_and_spot",
            "scenes": [
                {"scene_id": "office", "scene_path": "/tmp/office.json"},
                {"scene_id": "spot", "scene_path": "/tmp/spot.json"},
            ],
        },
        dataset_version=version,
    )

    assert paths.dataset_id == "office_and_spot"
    assert [scene.scene_id for scene in paths.scenes] == ["office", "spot"]
    assert paths.raw_questions_path.parent == paths.dataset_dir
    assert paths.scenes[0].scene_template_values_path == (
        paths.dataset_dir / "scenes" / "office" / "scene_template_values.yaml"
    )
    assert paths.scenes[1].scene_template_values_path == (
        paths.dataset_dir / "scenes" / "spot" / "scene_template_values.yaml"
    )


def test_multi_scene_dataset_rejects_duplicate_scene_ids():
    with pytest.raises(ValueError, match="Duplicate scene_id"):
        data_paths_from_config(
            {
                "dataset_id": "duplicates",
                "scenes": [
                    {"scene_id": "same", "scene_path": "/tmp/a.json"},
                    {"scene_id": "same", "scene_path": "/tmp/b.json"},
                ],
            }
        )


def test_teacher_folder_uses_model_name_or_synthetic_provider():
    assert teacher_name_from_config(
        {"provider": "openrouter", "model": "openai/gpt-oss-120b"}
    ) == "gpt-oss-120b"
    assert teacher_name_from_config({"provider": "synthetic"}) == "synthetic"


def test_data_paths_resolve_multiple_question_sets():
    paths = data_paths_from_config(
        {
            "scene_id": "office_scene",
            "scene_path": "/tmp/scene.json",
            "question_sets": [
                {
                    "name": "qa",
                    "question_templates_path": "cypher_qa_3dsg.yaml",
                    "experiment_profile_path": "cypher_qa_template.yaml",
                },
                {
                    "name": "pddl",
                    "question_templates_path": "cypher_pddl_3dsg.yaml",
                    "experiment_profile_path": "cypher_pddl_template.yaml",
                },
            ],
        }
    )

    assert [question_set.name for question_set in paths.question_sets] == ["qa", "pddl"]
    assert paths.question_sets[1].question_templates_path.name == "cypher_pddl_3dsg.yaml"


def test_duplicate_question_set_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate question set"):
        data_paths_from_config(
            {
                "scene_id": "office_scene",
                "scene_path": "/tmp/scene.json",
                "question_sets": [
                    {
                        "name": "same",
                        "question_templates_path": "cypher_qa_3dsg.yaml",
                        "experiment_profile_path": "cypher_qa_template.yaml",
                    },
                    {
                        "name": "same",
                        "question_templates_path": "cypher_pddl_3dsg.yaml",
                        "experiment_profile_path": "cypher_pddl_template.yaml",
                    },
                ],
            }
        )


def test_ollama_metadata_identifies_source_model_version_and_report_directory(tmp_path):
    export_dir = tmp_path / "ollama" / "qwen-agent"
    export_dir.mkdir(parents=True)
    (export_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "model_name: qwen-agent",
                "source_model:",
                "  family: qwen-family",
                "  version: 2026-09-08-T-15-05-21-Z",
            ]
        ),
        encoding="utf-8",
    )
    student = {
        "model_family": "qwen-family",
        "model_version": "2026-09-08-T-15-05-21-Z",
        "ollama_export": {"export_root": str(tmp_path / "ollama")},
    }
    data = {"dataset_dir": str(tmp_path / "dataset")}

    metadata = ollama_export_metadata(student, "qwen-agent-q4_k_m")

    assert metadata["source_model"]["version"] == "2026-09-08-T-15-05-21-Z"
    assert model_reports_dir(data, student) == (
        tmp_path
        / "dataset"
        / "reports"
        / "qwen-family"
        / "2026-09-08-T-15-05-21-Z"
    )
