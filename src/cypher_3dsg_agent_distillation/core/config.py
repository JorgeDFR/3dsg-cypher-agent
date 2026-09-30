from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

from cypher_3dsg_agent_distillation.core.versioning import (
    atomic_write_yaml,
    create_version_dir,
    iso_now,
    read_metadata,
    resolve_version_dir,
    slugify,
    utc_now,
    validate_version,
    version_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"
GENERATED_DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
# Kept as a compatibility alias for callers which imported the old constant.
GENERATED_3DSG_ROOT = GENERATED_DATASETS_ROOT


@dataclass(frozen=True)
class QuestionSetPaths:
    name: str
    question_templates_path: Path
    experiment_profile_path: Path


@dataclass(frozen=True)
class ScenePaths:
    scene_id: str
    scene_path: Path
    scene_dir: Path
    scene_template_values_path: Path


@dataclass(frozen=True)
class DataPaths:
    dataset_id: str
    scenes: tuple[ScenePaths, ...]
    # Compatibility aliases for legacy single-scene callers. New dataset code
    # should iterate ``scenes`` instead.
    scene_id: str
    dataset_version: str
    scene_path: Path
    schema_path: Path
    experiment_profile_path: Path | None
    question_templates_path: Path
    question_sets: tuple[QuestionSetPaths, ...]
    dataset_root: Path
    dataset_dir: Path
    metadata_path: Path
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


def _safe_config_id(value: Any, *, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError(f"{field} must be a single folder name, not a path.")
    return normalized


def dataset_id_from_config(config: dict[str, Any]) -> str:
    value = config.get("dataset_id", config.get("scene_id"))
    if value is None:
        raise KeyError("data config requires `dataset_id`.")
    return _safe_config_id(value, field="dataset_id")


def scene_id_from_config(config: dict[str, Any]) -> str:
    """Return the sole scene id for compatibility with legacy callers."""
    scenes = scenes_from_config(config)
    if len(scenes) != 1:
        raise ValueError("This operation requires a single configured scene.")
    return scenes[0]["scene_id"]


def scenes_from_config(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_scenes = config.get("scenes")
    if raw_scenes is None:
        if "scene_id" not in config or "scene_path" not in config:
            raise KeyError(
                "data config requires a non-empty `scenes` list containing "
                "scene_id and scene_path."
            )
        raw_scenes = [
            {"scene_id": config["scene_id"], "scene_path": config["scene_path"]}
        ]
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("scenes must be a non-empty list.")

    scenes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_scenes):
        if not isinstance(item, dict):
            raise TypeError(f"scenes[{index}] must be a mapping.")
        if "scene_id" not in item or "scene_path" not in item:
            raise KeyError(f"scenes[{index}] requires scene_id and scene_path.")
        scene_id = _safe_config_id(item["scene_id"], field=f"scenes[{index}].scene_id")
        if scene_id in seen_ids:
            raise ValueError(f"Duplicate scene_id `{scene_id}`.")
        seen_ids.add(scene_id)
        scenes.append(
            {
                **item,
                "scene_id": scene_id,
                "scene_path": resolve_project_path(item["scene_path"]),
            }
        )
    return tuple(scenes)


def data_paths_from_config(
    config: dict[str, Any],
    *,
    dataset_version: str | None = None,
    create_version: bool = False,
    require_existing: bool = False,
) -> DataPaths:
    dataset_id = dataset_id_from_config(config)
    configured_scenes = scenes_from_config(config)
    legacy_single_scene = "scenes" not in config

    dataset_root = GENERATED_DATASETS_ROOT / dataset_id
    requested_version = dataset_version or config.get("dataset_version")
    if create_version:
        resolved_version, scene_dir = create_version_dir(
            dataset_root,
            None if requested_version in (None, "latest") else str(requested_version),
        )
    elif require_existing:
        resolved_version, scene_dir = resolve_version_dir(
            dataset_root,
            None if requested_version is None else str(requested_version),
            artifact_label="dataset version",
        )
    else:
        resolved_version = str(requested_version or "latest")
        if resolved_version != "latest":
            resolved_version = validate_version(resolved_version)
        scene_dir = dataset_root / resolved_version
    resolved_scenes = tuple(
        ScenePaths(
            scene_id=scene["scene_id"],
            scene_path=Path(scene["scene_path"]),
            scene_dir=(
                scene_dir
                if legacy_single_scene
                else scene_dir / "scenes" / scene["scene_id"]
            ),
            scene_template_values_path=(
                scene_dir / "scene_template_values.yaml"
                if legacy_single_scene
                else scene_dir / "scenes" / scene["scene_id"] / "scene_template_values.yaml"
            ),
        )
        for scene in configured_scenes
    )

    raw_question_sets = config.get("question_sets")
    if raw_question_sets is not None:
        if not isinstance(raw_question_sets, list) or not raw_question_sets:
            raise ValueError("question_sets must be a non-empty list when provided.")
        question_sets = []
        seen_names: set[str] = set()
        for index, item in enumerate(raw_question_sets):
            if not isinstance(item, dict):
                raise TypeError(f"question_sets[{index}] must be a mapping.")
            name = str(item.get("name") or "").strip()
            if not name:
                raise ValueError(f"question_sets[{index}] requires a non-empty name.")
            if name in seen_names:
                raise ValueError(f"Duplicate question set name `{name}`.")
            seen_names.add(name)
            try:
                templates_path = item["question_templates_path"]
                profile_path = item["experiment_profile_path"]
            except KeyError as exc:
                raise KeyError(
                    f"question_sets[{index}] requires question_templates_path and "
                    "experiment_profile_path."
                ) from exc
            question_sets.append(
                QuestionSetPaths(
                    name=name,
                    question_templates_path=resolve_config_path(
                        templates_path, "question_templates"
                    ),
                    experiment_profile_path=resolve_config_path(
                        profile_path, "experiment_profiles"
                    ),
                )
            )
        resolved_question_sets = tuple(question_sets)
    else:
        profile_value = config.get("experiment_profile_path") or "cypher_qa_template.yaml"
        resolved_question_sets = (
            QuestionSetPaths(
                name=str(config.get("question_set_name") or "cypher_qa"),
                question_templates_path=resolve_config_path(
                    config.get("question_templates_path", "cypher_qa_3dsg.yaml"),
                    "question_templates",
                ),
                experiment_profile_path=resolve_config_path(
                    profile_value, "experiment_profiles"
                ),
            ),
        )

    first_question_set = resolved_question_sets[0]
    first_scene = resolved_scenes[0]
    return DataPaths(
        dataset_id=dataset_id,
        scenes=resolved_scenes,
        scene_id=first_scene.scene_id,
        dataset_version=resolved_version,
        scene_path=first_scene.scene_path,
        schema_path=resolve_config_path(config.get("schema_path", "schema_3dsg.yaml")),
        experiment_profile_path=first_question_set.experiment_profile_path,
        question_templates_path=first_question_set.question_templates_path,
        question_sets=resolved_question_sets,
        dataset_root=dataset_root,
        dataset_dir=scene_dir,
        metadata_path=scene_dir / "metadata.yaml",
        scene_dir=scene_dir,
        scene_template_values_path=first_scene.scene_template_values_path,
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


def load_data_config(
    path: str | Path,
    *,
    dataset_version: str | None = None,
    create_version: bool = False,
) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    config = load_yaml(config_path)
    paths = data_paths_from_config(
        config,
        dataset_version=dataset_version,
        create_version=create_version,
        require_existing=not create_version,
    )
    resolved = dict(config)
    resolved.update(
        {
            "dataset_id": paths.dataset_id,
            "data_config_path": str(config_path),
            "dataset_version": paths.dataset_version,
            "dataset_root": str(paths.dataset_root),
            "dataset_dir": str(paths.dataset_dir),
            "metadata_path": str(paths.metadata_path),
            "scenes": [
                {
                    "scene_id": scene.scene_id,
                    "scene_path": str(scene.scene_path),
                    "scene_dir": str(scene.scene_dir),
                    "scene_template_values_path": str(scene.scene_template_values_path),
                }
                for scene in paths.scenes
            ],
            # Compatibility aliases for commands that explicitly operate on a
            # single scene (for example load_3dsg_neo4j without --scene).
            "scene_id": paths.scene_id,
            "scene_path": str(paths.scene_path),
            "schema_path": str(paths.schema_path),
            "question_templates_path": str(paths.question_templates_path),
            "question_sets": [
                {
                    "name": question_set.name,
                    "question_templates_path": str(question_set.question_templates_path),
                    "experiment_profile_path": str(question_set.experiment_profile_path),
                }
                for question_set in paths.question_sets
            ],
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
    try:
        dataset_metadata = read_metadata(paths.metadata_path)
    except (FileNotFoundError, TypeError, yaml.YAMLError):
        dataset_metadata = {}
    teacher = dataset_metadata.get("teacher") or dataset_metadata.get("active_teacher_run")
    if isinstance(teacher, dict):
        resolved["teacher_name"] = teacher.get("teacher_name")
    traces_dir = paths.dataset_dir / "traces"
    # Read existing versioned datasets without moving or rewriting their artifacts.
    if not (traces_dir / "metadata.yaml").exists() and isinstance(teacher, dict):
        if teacher.get("version") and teacher.get("teacher_name"):
            traces_dir = traces_dir / teacher["teacher_name"] / validate_version(teacher["version"])
    resolved.update({
        "teacher_metadata_path": str(traces_dir / "metadata.yaml"),
        "teacher_traces_jsonl_path": str(traces_dir / "teacher_traces.jsonl"),
        "accepted_traces_path": str(traces_dir / "accepted.json"),
        "rejected_traces_path": str(traces_dir / "rejected.json"),
    })
    return resolved


def teacher_name_from_config(config: dict[str, Any]) -> str:
    provider = str(config.get("provider") or "openrouter").strip().lower()
    if provider == "synthetic":
        return "synthetic"
    model = str(config.get("model") or provider).strip().rstrip("/")
    return slugify(model.rsplit("/", 1)[-1], fallback=provider)


def fork_dataset_for_teacher(data_config: dict[str, Any]) -> dict[str, Any]:
    """Copy prepared inputs into a new dataset, leaving traces/splits/reports behind."""
    source = Path(data_config["dataset_dir"])
    for key in ("raw_questions_path", "seed_records_jsonl_path"):
        if not Path(data_config[key]).is_file():
            raise FileNotFoundError(f"Prepared dataset input is missing: {data_config[key]}")
    metadata = read_metadata(source)
    moment = utc_now()
    while True:
        try:
            version, target = create_version_dir(source.parent, version_id(moment))
            break
        except FileExistsError:
            moment += timedelta(seconds=1)

    def rebase(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, str):
            value = value.replace(str(source) + "/", str(target) + "/")
            try:
                old = str(source.relative_to(PROJECT_ROOT))
                new = str(target.relative_to(PROJECT_ROOT))
                value = value.replace(old + "/", new + "/")
            except ValueError:
                pass
            return str(target) if value == str(source) else value
        return value

    try:
        for item in source.iterdir():
            if item.name in {"metadata.yaml", "traces", "splits", "reports"}:
                continue
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        metadata = rebase(metadata)
        for key in ("teacher", "active_teacher_run", "split_files", "split_fingerprint"):
            metadata.pop(key, None)
        metadata["stages"] = {
            key: value for key, value in metadata.get("stages", {}).items()
            if key not in {"split", "teacher", "generate_traces"}
        }
        metadata.update({
            "version": version,
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "status": "building",
            "parent_dataset": {
                "dataset_id": data_config["dataset_id"],
                "version": data_config["dataset_version"],
            },
        })
        atomic_write_yaml(metadata, target / "metadata.yaml")
    except BaseException:
        shutil.rmtree(target)
        raise
    resolved = rebase(data_config)
    resolved["dataset_version"] = version
    resolved.pop("teacher_name", None)
    return resolved


def prepare_teacher_dataset(
    data_config: dict[str, Any], *, teacher_name: str
) -> dict[str, Any]:
    """Reserve the dataset's sole trace set; existing work always causes a fork."""
    traces = Path(data_config["dataset_dir"]) / "traces"
    splits = Path(data_config["dataset_dir"]) / "splits"
    if (traces.exists() and any(traces.iterdir())) or (
        splits.exists() and any(splits.iterdir())
    ):
        data_config = fork_dataset_for_teacher(data_config)
        traces = Path(data_config["dataset_dir"]) / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent teachers from sharing a trace set.
    try:
        with (traces / "metadata.yaml").open("x", encoding="utf-8") as handle:
            yaml.safe_dump({"status": "running", "teacher_name": teacher_name}, handle)
    except FileExistsError:
        return prepare_teacher_dataset(data_config, teacher_name=teacher_name)
    return {
        **data_config,
        "teacher_name": slugify(teacher_name, fallback="teacher"),
        "teacher_metadata_path": str(traces / "metadata.yaml"),
        "teacher_traces_jsonl_path": str(traces / "teacher_traces.jsonl"),
        "accepted_traces_path": str(traces / "accepted.json"),
        "rejected_traces_path": str(traces / "rejected.json"),
    }


def load_student_config(
    path: str | Path,
    *,
    model_version: str | None = None,
    create_version: bool = False,
) -> dict[str, Any]:
    config = load_student_family_config(path)
    config_path = Path(config["student_config_path"])
    training = dict(config.get("training") or {})
    family_root = Path(config["model_family_root"])
    if create_version:
        version, output_dir = create_version_dir(
            family_root,
            None if model_version in (None, "latest") else model_version,
        )
    else:
        version, output_dir = resolve_version_dir(
            family_root,
            model_version,
            completed_only=True,
            artifact_label="LoRA model version",
        )
    training["output_dir"] = str(output_dir)
    resolved = dict(config)
    resolved["training"] = training
    resolved.update(
        {
            "student_config_path": str(config_path),
            "model_family": family_root.name,
            "model_family_root": str(family_root),
            "model_version": version,
            "model_dir": str(output_dir),
            "model_metadata_path": str(output_dir / "metadata.yaml"),
        }
    )
    return resolved


def load_student_family_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    config = load_yaml(config_path)
    training = dict(config.get("training") or {})
    if "output_dir" not in training:
        raise KeyError("student config requires training.output_dir.")
    family_root = resolve_project_path(training["output_dir"])
    resolved = dict(config)
    resolved.update(
        {
            "student_config_path": str(config_path),
            "model_family": family_root.name,
            "model_family_root": str(family_root),
        }
    )
    return resolved


def model_reports_dir(
    data_config: dict[str, Any],
    student_config: dict[str, Any],
    *,
    baseline: bool = False,
    model_version: str | None = None,
) -> Path:
    family = slugify(str(student_config.get("model_family") or "student"), fallback="student")
    version = (
        "baseline"
        if baseline
        else slugify(
            str(model_version or student_config.get("model_version") or "unknown"),
            fallback="unknown",
        )
    )
    dataset_dir_value = data_config.get("dataset_dir")
    if dataset_dir_value:
        reports_root = Path(dataset_dir_value) / "reports"
    else:
        legacy_reports = Path(data_config["test_report_path"]).parent
        reports_root = (
            legacy_reports.parent if legacy_reports.name == "dataset" else legacy_reports
        )
    return reports_root / family / version


def existing_report_path(path: str | Path) -> Path:
    """Read historical reports while always writing to the current flat layout."""
    selected = Path(path)
    if selected.exists():
        return selected
    for root in selected.parents:
        if root.name != "reports":
            continue
        relative = selected.relative_to(root)
        legacy = root / ("dataset" if len(relative.parts) == 1 else "models") / relative
        return legacy if legacy.exists() else selected
    return selected


def ollama_export_metadata(
    student_config: dict[str, Any], ollama_model: str
) -> dict[str, Any]:
    export_config = student_config.get("ollama_export") or {}
    export_root = resolve_project_path(export_config.get("export_root", "outputs/ollama"))
    matches: list[tuple[int, dict[str, Any]]] = []
    if export_root.is_dir():
        for path in export_root.glob("*/metadata.yaml"):
            try:
                metadata = load_yaml(path)
            except (OSError, TypeError, yaml.YAMLError):
                continue
            exported_name = str(metadata.get("model_name") or path.parent.name)
            if ollama_model == exported_name or ollama_model.startswith(f"{exported_name}-"):
                matches.append((len(exported_name), metadata))
    if not matches:
        raise FileNotFoundError(
            f"No outputs/ollama metadata.yaml matches registered model `{ollama_model}`. "
            "Export the model with this repository before evaluating it."
        )
    return max(matches, key=lambda item: item[0])[1]
