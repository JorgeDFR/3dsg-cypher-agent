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
from cypher_3dsg_agent_distillation.core.versioning import atomic_write_yaml


def current_config(**overrides):
    return {
        "dataset_id": "office_scene",
        "scenes": [{"scene_id": "office", "scene_path": "/tmp/hydra/scene_graph.json"}],
        "question_sets": [
            {
                "name": "qa",
                "question_templates_path": "cypher_qa_3dsg.yaml",
                "experiment_profile_path": "cypher_qa_template.yaml",
            }
        ],
        **overrides,
    }


def test_current_paths_isolate_scene_inputs_and_share_versioned_outputs():
    version = "2026-09-08-T-12-05-47-Z"
    paths = data_paths_from_config(current_config(), dataset_version=version)
    root = PROJECT_ROOT / "data/datasets/office_scene" / version
    assert paths.scenes[0].scene_path == Path("/tmp/hydra/scene_graph.json")
    assert (
        paths.scenes[0].scene_template_values_path
        == root / "scenes/office/scene_template_values.yaml"
    )
    assert paths.schema_path == PROJECT_ROOT / "configs/schema_3dsg.yaml"
    assert (
        paths.question_sets[0].question_templates_path
        == PROJECT_ROOT / "configs/question_templates/cypher_qa_3dsg.yaml"
    )
    assert paths.raw_questions_path == root / "generated_questions.yaml"
    assert paths.seed_records_jsonl_path == root / "cypher_seed_records.jsonl"
    assert paths.teacher_traces_jsonl_path == root / "traces/teacher_traces.jsonl"
    assert paths.splits["test"] == root / "splits/test.jsonl"
    assert not hasattr(paths, "scene_id")
    assert not hasattr(paths, "question_templates_path")


def test_load_data_config_expands_current_paths_without_first_item_aliases(tmp_path, monkeypatch):
    generated_root = tmp_path / "datasets"
    version = "2026-09-08-T-12-05-47-Z"
    (generated_root / "office_scene" / version).mkdir(parents=True)
    monkeypatch.setattr(config_module, "GENERATED_DATASETS_ROOT", generated_root)
    path = tmp_path / "data.yaml"
    atomic_write_yaml(current_config(), path)
    config = load_data_config(path)
    assert config["scenes"][0]["scene_path"] == "/tmp/hydra/scene_graph.json"
    assert config["teacher_traces_jsonl_path"] == str(
        generated_root / "office_scene" / version / "traces/teacher_traces.jsonl"
    )
    assert (
        not {"scene_path", "scene_id", "experiment_profile_path", "question_templates_path"}
        & config.keys()
    )


@pytest.mark.parametrize("missing", ["dataset_id", "scenes", "question_sets"])
def test_current_config_requires_explicit_dataset_scenes_and_question_sets(missing):
    config = current_config()
    config.pop(missing)
    with pytest.raises((KeyError, ValueError), match=missing):
        data_paths_from_config(config)


@pytest.mark.parametrize(
    "retired", ["scene_id", "scene_path", "experiment_profile_path", "question_templates_path"]
)
def test_retired_top_level_fields_are_rejected(retired):
    with pytest.raises(ValueError, match="Unsupported data configuration"):
        data_paths_from_config(current_config(**{retired: "old"}))


def test_scene_id_must_not_be_a_path():
    with pytest.raises(ValueError, match="single folder name"):
        data_paths_from_config(
            current_config(scenes=[{"scene_id": "../office", "scene_path": "x"}])
        )


def test_multi_scene_paths_and_question_sets_are_explicit():
    config = current_config(
        scenes=[
            {"scene_id": "office", "scene_path": "/tmp/office.json"},
            {"scene_id": "spot", "scene_path": "/tmp/spot.json"},
        ]
    )
    config["question_sets"].append(
        {
            "name": "pddl",
            "question_templates_path": "cypher_pddl_3dsg.yaml",
            "experiment_profile_path": "cypher_pddl_template.yaml",
        }
    )
    paths = data_paths_from_config(config)
    assert [s.scene_id for s in paths.scenes] == ["office", "spot"]
    assert [q.name for q in paths.question_sets] == ["qa", "pddl"]
    assert paths.scenes[0].scene_dir != paths.scenes[1].scene_dir
    assert paths.question_sets[1].question_templates_path.name == "cypher_pddl_3dsg.yaml"


def test_duplicate_scene_and_question_set_ids_are_rejected():
    config = current_config()
    config["scenes"] *= 2
    with pytest.raises(ValueError, match="Duplicate scene_id"):
        data_paths_from_config(config)
    config = current_config()
    config["question_sets"] *= 2
    with pytest.raises(ValueError, match="Duplicate question set"):
        data_paths_from_config(config)


def test_teacher_name_uses_model_name_or_synthetic_provider():
    assert (
        teacher_name_from_config({"provider": "openrouter", "model": "openai/gpt-oss-120b"})
        == "gpt-oss-120b"
    )
    assert teacher_name_from_config({"provider": "synthetic"}) == "synthetic"


def test_ollama_metadata_and_reports_identify_current_model_version(tmp_path):
    export_dir = tmp_path / "ollama/qwen-agent"
    export_dir.mkdir(parents=True)
    atomic_write_yaml(
        {
            "model_name": "qwen-agent",
            "source_model": {"family": "qwen-family", "version": "2026-09-08-T-15-05-21-Z"},
        },
        export_dir / "metadata.yaml",
    )
    student = {
        "model_family": "qwen-family",
        "model_version": "2026-09-08-T-15-05-21-Z",
        "ollama_export": {"export_root": str(tmp_path / "ollama")},
    }
    metadata = ollama_export_metadata(student, "qwen-agent-q4_k_m")
    assert metadata["source_model"]["version"] == student["model_version"]
    assert (
        model_reports_dir({"dataset_dir": str(tmp_path / "dataset")}, student)
        == tmp_path / "dataset/reports/qwen-family" / student["model_version"]
    )
