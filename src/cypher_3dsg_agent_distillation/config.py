from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
GENERATED_3DSG_ROOT = PROJECT_ROOT / "data" / "3dsg"


@dataclass(frozen=True)
class DataPaths:
    scene_id: str
    scene_path: Path
    schema_path: Path
    experiment_profile_path: Path | None
    question_templates_path: Path
    scene_dir: Path
    scene_template_values_path: Path
    raw_questions_path: Path
    seed_records_jsonl_path: Path
    teacher_traces_jsonl_path: Path
    accepted_traces_path: Path
    rejected_traces_path: Path
    splits: dict[str, Path]
    test_report_path: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"Expected mapping at {path}, got {type(data).__name__}"
        raise TypeError(msg)
    return data


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def expanded_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def resolve_project_path(path: str | Path) -> Path:
    resolved = expanded_path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def resolve_config_path(path: str | Path, subdir: str | Path = "") -> Path:
    resolved = expanded_path(path)
    if resolved.is_absolute():
        return resolved
    if resolved.parent == Path("."):
        return CONFIG_ROOT / subdir / resolved
    return PROJECT_ROOT / resolved


def scene_id_from_config(config: dict[str, Any]) -> str:
    if "scene_id" not in config:
        raise KeyError("data config requires `scene_id`.")
    scene_id = str(config["scene_id"]).strip()
    if not scene_id:
        raise ValueError("scene_id must not be empty.")
    if scene_id in {".", ".."} or "/" in scene_id or "\\" in scene_id:
        raise ValueError("scene_id must be a single folder name, not a path.")
    return scene_id


def data_paths_from_config(config: dict[str, Any]) -> DataPaths:
    scene_id = scene_id_from_config(config)
    if "scene_path" not in config:
        raise KeyError("data config requires `scene_path` for the user-provided 3DSG file.")

    scene_dir = GENERATED_3DSG_ROOT / scene_id
    experiment_profile_path = config.get("experiment_profile_path")
    return DataPaths(
        scene_id=scene_id,
        scene_path=resolve_project_path(config["scene_path"]),
        schema_path=resolve_config_path(config.get("schema_path", "schema_3dsg.yaml")),
        experiment_profile_path=(
            resolve_config_path(experiment_profile_path, "experiment_profiles")
            if experiment_profile_path
            else None
        ),
        question_templates_path=resolve_config_path(
            config.get("question_templates_path", "cypher_qa_3dsg.yaml"),
            "question_templates",
        ),
        scene_dir=scene_dir,
        scene_template_values_path=scene_dir / "scene_template_values.yaml",
        raw_questions_path=scene_dir / "generated_questions.yaml",
        seed_records_jsonl_path=scene_dir / "cypher_seed_records.jsonl",
        teacher_traces_jsonl_path=scene_dir / "traces" / "teacher_traces.jsonl",
        accepted_traces_path=scene_dir / "traces" / "accepted.json",
        rejected_traces_path=scene_dir / "traces" / "rejected.json",
        splits={
            "train": scene_dir / "splits" / "train.jsonl",
            "eval": scene_dir / "splits" / "eval.jsonl",
            "test": scene_dir / "splits" / "test.jsonl",
            "challenge": scene_dir / "splits" / "challenge.jsonl",
        },
        test_report_path=scene_dir / "reports" / "test.agent.report.json",
    )


def load_data_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(resolve_project_path(path))
    paths = data_paths_from_config(config)
    resolved = dict(config)
    resolved.update(
        {
            "scene_path": str(paths.scene_path),
            "schema_path": str(paths.schema_path),
            "question_templates_path": str(paths.question_templates_path),
            "scene_template_values_path": str(paths.scene_template_values_path),
            "raw_questions_path": str(paths.raw_questions_path),
            "seed_records_jsonl_path": str(paths.seed_records_jsonl_path),
            "teacher_traces_jsonl_path": str(paths.teacher_traces_jsonl_path),
            "accepted_traces_path": str(paths.accepted_traces_path),
            "rejected_traces_path": str(paths.rejected_traces_path),
            "splits": {name: str(split_path) for name, split_path in paths.splits.items()},
            "test_report_path": str(paths.test_report_path),
        }
    )
    if paths.experiment_profile_path is not None:
        resolved["experiment_profile_path"] = str(paths.experiment_profile_path)
    return resolved
