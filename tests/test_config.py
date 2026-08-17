from pathlib import Path

import pytest

from cypher_3dsg_agent_distillation.config import (
    PROJECT_ROOT,
    data_paths_from_config,
    load_data_config,
)


def test_data_paths_keep_scene_path_explicit_and_derive_generated_scene_dir():
    paths = data_paths_from_config(
        {
            "scene_id": "office_scene",
            "scene_path": "/tmp/hydra/scene_graph.json",
            "schema_path": "schema_3dsg.yaml",
            "experiment_profile_path": "cypher_qa_template.yaml",
            "question_templates_path": "cypher_qa_3dsg.yaml",
        }
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
        PROJECT_ROOT / "data" / "3dsg" / "office_scene" / "generated_questions.yaml"
    )
    assert paths.seed_records_jsonl_path == (
        PROJECT_ROOT
        / "data"
        / "3dsg"
        / "office_scene"
        / "cypher_seed_records.jsonl"
    )
    assert paths.teacher_traces_jsonl_path == (
        PROJECT_ROOT
        / "data"
        / "3dsg"
        / "office_scene"
        / "traces"
        / "teacher_traces.jsonl"
    )
    assert paths.accepted_traces_path == (
        PROJECT_ROOT / "data" / "3dsg" / "office_scene" / "traces" / "accepted.json"
    )
    assert paths.splits["test"] == (
        PROJECT_ROOT / "data" / "3dsg" / "office_scene" / "splits" / "test.jsonl"
    )


def test_load_data_config_expands_derived_paths(tmp_path):
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "scene_id: uhumans2",
                "scene_path: data/scenes/3dsg_uhumans2.json",
                "schema_path: schema_3dsg.yaml",
            ]
        ),
        encoding="utf-8",
    )

    config = load_data_config(config_path)

    assert config["scene_path"] == str(PROJECT_ROOT / "data" / "scenes" / "3dsg_uhumans2.json")
    assert config["schema_path"] == str(PROJECT_ROOT / "configs" / "schema_3dsg.yaml")
    assert config["teacher_traces_jsonl_path"] == str(
        PROJECT_ROOT / "data" / "3dsg" / "uhumans2" / "traces" / "teacher_traces.jsonl"
    )
    assert config["accepted_traces_path"] == str(
        PROJECT_ROOT / "data" / "3dsg" / "uhumans2" / "traces" / "accepted.json"
    )
    assert config["splits"]["train"] == str(
        PROJECT_ROOT / "data" / "3dsg" / "uhumans2" / "splits" / "train.jsonl"
    )


def test_scene_id_must_not_be_a_path():
    with pytest.raises(ValueError, match="single folder name"):
        data_paths_from_config(
            {
                "scene_id": "../uhumans2",
                "scene_path": "/tmp/scene.json",
            }
        )
