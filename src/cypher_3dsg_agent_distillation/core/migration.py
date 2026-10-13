from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.core.config import (
    GENERATED_DATASETS_ROOT,
    PROJECT_ROOT,
    load_yaml,
    resolve_project_path,
)
from cypher_3dsg_agent_distillation.core.versioning import (
    atomic_write_yaml,
    config_record,
    file_fingerprint,
    git_provenance,
    split_fingerprints,
    version_id,
)

LEGACY_DATA_ROOT = PROJECT_ROOT / "data" / "3dsg"
LORA_ROOT = PROJECT_ROOT / "outputs" / "lora"
OLLAMA_ROOT = PROJECT_ROOT / "outputs" / "ollama"
LEGACY_SUFFIX = re.compile(r"-v\d+$", flags=re.IGNORECASE)


def artifact_timestamp(path: Path) -> tuple[str, str]:
    moment = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return version_id(moment), moment.isoformat()


def move(source: Path, target: Path, *, apply: bool) -> None:
    print(f"MOVE {source.relative_to(PROJECT_ROOT)} -> {target.relative_to(PROJECT_ROOT)}")
    if not apply:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Migration target already exists: {target}")
    shutil.move(str(source), str(target))


def teacher_identity(traces_file: Path) -> str:
    try:
        first_line = traces_file.open("r", encoding="utf-8").readline()
        record = json.loads(first_line)
        metadata = record.get("metadata") or {}
        provider = str(metadata.get("teacher_provider") or "teacher")
        if provider == "synthetic":
            return "synthetic"
        model = str(metadata.get("teacher_model") or provider).rstrip("/")
        return re.sub(r"[^A-Za-z0-9._-]+", "-", model.rsplit("/", 1)[-1])
    except (OSError, json.JSONDecodeError, AttributeError):
        return "legacy-teacher"


def migrate_dataset(data_config_path: Path, *, apply: bool) -> dict[str, Any] | None:
    data_config = load_yaml(data_config_path)
    scene_id = str(data_config["scene_id"])
    legacy_scene = LEGACY_DATA_ROOT / scene_id
    if not legacy_scene.is_dir():
        existing = sorted((GENERATED_DATASETS_ROOT / scene_id).glob("*/metadata.yaml"))
        if not existing:
            return None
        return load_yaml(existing[-1])

    marker = legacy_scene / "scene_template_values.yaml"
    version, created_at = artifact_timestamp(marker if marker.exists() else legacy_scene)
    target = GENERATED_DATASETS_ROOT / scene_id / version
    move(legacy_scene, target, apply=apply)
    if not apply:
        return {"scene_id": scene_id, "version": version}

    traces_root = target / "traces"
    traces_file = traces_root / "teacher_traces.jsonl"
    active_teacher = None
    if traces_file.is_file():
        teacher_name = teacher_identity(traces_file)
        _, teacher_created_at = artifact_timestamp(traces_file)
        teacher_dir = traces_root
        atomic_write_yaml(
            {
                "schema_version": 1,
                "artifact_type": "teacher_traces",
                "teacher_name": teacher_name,
                "created_at": teacher_created_at,
                "updated_at": teacher_created_at,
                "status": "complete",
                "legacy_import": True,
                "dataset": {"scene_id": scene_id, "version": version},
                "outputs": {
                    name: file_fingerprint(teacher_dir / filename, relative_to=PROJECT_ROOT)
                    for name, filename in (
                        ("teacher_traces", "teacher_traces.jsonl"),
                        ("accepted", "accepted.json"),
                        ("rejected", "rejected.json"),
                    )
                },
            },
            teacher_dir / "metadata.yaml",
        )
        active_teacher = {"teacher_name": teacher_name}

    splits = {
        name: target / "splits" / f"{name}.jsonl"
        for name in ("train", "eval", "test", "challenge")
    }
    split_files, split_fingerprint = split_fingerprints(
        splits, project_root=PROJECT_ROOT
    )
    schema_path = resolve_project_path(
        Path("configs") / str(data_config.get("schema_path", "schema_3dsg.yaml"))
    )
    if not schema_path.exists():
        schema_path = PROJECT_ROOT / "configs" / str(
            data_config.get("schema_path", "schema_3dsg.yaml")
        )
    metadata = {
        "schema_version": 1,
        "artifact_type": "dataset",
        "scene_id": scene_id,
        "version": version,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "complete",
        "legacy_import": True,
        "source_scene": file_fingerprint(
            resolve_project_path(data_config["scene_path"]), relative_to=PROJECT_ROOT
        ),
        "configs": {
            "data_config": config_record(data_config_path, project_root=PROJECT_ROOT),
            "schema": config_record(schema_path, project_root=PROJECT_ROOT),
        },
        "code": git_provenance(PROJECT_ROOT),
        "teacher": active_teacher,
        "split_files": split_files,
        "split_fingerprint": split_fingerprint,
        "stages": {"legacy_import": {"completed_at": created_at}},
    }
    atomic_write_yaml(metadata, target / "metadata.yaml")
    try:
        LEGACY_DATA_ROOT.rmdir()
    except OSError:
        pass
    return metadata


def lora_family(name: str) -> str:
    return LEGACY_SUFFIX.sub("", name)


def migrate_lora_models(
    student_config_path: Path,
    dataset_metadata: dict[str, Any] | None,
    *,
    apply: bool,
) -> dict[str, dict[str, Any]]:
    student_config = load_yaml(student_config_path)
    configured_root = resolve_project_path(student_config["training"]["output_dir"])
    configured_family = configured_root.name
    entries = [
        path
        for path in LORA_ROOT.iterdir()
        if path.is_dir() and (path / "adapter_config.json").is_file()
    ]
    grouped: dict[str, list[Path]] = {}
    for entry in entries:
        grouped.setdefault(lora_family(entry.name), []).append(entry)

    imported: dict[str, dict[str, Any]] = {}
    for family, family_entries in grouped.items():
        family_root = LORA_ROOT / family
        canonical = next((entry for entry in family_entries if entry.name == family), None)
        temporary = LORA_ROOT / f".{family}.legacy-import"
        if canonical is not None:
            move(canonical, temporary, apply=apply)
            if apply:
                canonical = temporary
        if apply:
            family_root.mkdir(parents=True, exist_ok=True)

        ordered = [entry for entry in family_entries if entry.name != family]
        if canonical is not None:
            ordered.append(canonical)
        for entry in ordered:
            marker = entry / "adapter_model.safetensors"
            if not marker.exists():
                marker = entry / "adapter_model.bin"
            version, created_at = artifact_timestamp(marker)
            target = family_root / version
            move(entry, target, apply=apply)
            linked = family == configured_family and entry == canonical and dataset_metadata
            metadata = {
                "schema_version": 1,
                "artifact_type": "lora_model",
                "model_family": family,
                "version": version,
                "created_at": created_at,
                "updated_at": created_at,
                "status": "complete",
                "legacy_import": True,
                "lineage_status": "known" if linked else "orphaned",
                "dataset": (
                    {
                        "scene_id": dataset_metadata["scene_id"],
                        "version": dataset_metadata["version"],
                        "split_fingerprint": dataset_metadata.get("split_fingerprint"),
                        "split_files": dataset_metadata.get("split_files"),
                        "teacher_name": (
                            dataset_metadata.get("teacher") or {}
                        ).get("teacher_name"),
                    }
                    if linked
                    else None
                ),
                "student_config": config_record(
                    student_config_path, project_root=PROJECT_ROOT
                ),
                "code": git_provenance(PROJECT_ROOT),
            }
            if apply:
                atomic_write_yaml(metadata, target / "metadata.yaml")
            imported[f"{family}:{entry.name}"] = metadata
    return imported


def migrate_reports_and_ollama(
    dataset_metadata: dict[str, Any] | None,
    imported_models: dict[str, dict[str, Any]],
    *,
    apply: bool,
) -> None:
    if not dataset_metadata:
        return
    scene_id = dataset_metadata["scene_id"]
    dataset_version = dataset_metadata["version"]
    dataset_dir = GENERATED_DATASETS_ROOT / scene_id / dataset_version
    reports = dataset_dir / "reports"
    current = next(
        (
            metadata
            for metadata in imported_models.values()
            if metadata.get("lineage_status") == "known"
        ),
        None,
    )
    if current and reports.is_dir():
        family = current["model_family"]
        model_version = current["version"]
        for source in list(reports.iterdir()):
            if not source.is_file():
                continue
            if source.name == "split.report.json" or ".agent.report.json" in source.name:
                continue
            elif ".baseline." in source.name:
                destination = reports / family / "baseline" / source.name
            else:
                destination = reports / family / model_version / source.name
            move(source, destination, apply=apply)

        export_dir = OLLAMA_ROOT / family
        if export_dir.is_dir():
            source_model_path = LORA_ROOT / family / model_version
            metadata = {
                "schema_version": 1,
                "artifact_type": "ollama_export",
                "model_name": family,
                "updated_at": datetime.now(UTC).isoformat(),
                "model_variant": "fine-tuned",
                "base_model": load_yaml(source_model_path / "adapter_config.json").get(
                    "base_model_name_or_path"
                ),
                "source_model": {
                    "family": family,
                    "version": model_version,
                    "path": str(source_model_path.relative_to(PROJECT_ROOT)),
                },
                "dataset": current.get("dataset"),
                "legacy_import": True,
            }
            if apply:
                atomic_write_yaml(metadata, export_dir / "metadata.yaml")


def command(args: argparse.Namespace) -> None:
    data_config = resolve_project_path(args.data_config)
    student_config = resolve_project_path(args.student_config)
    dataset_metadata = migrate_dataset(data_config, apply=args.apply)
    imported_models = migrate_lora_models(
        student_config, dataset_metadata, apply=args.apply
    )
    migrate_reports_and_ollama(
        dataset_metadata, imported_models, apply=args.apply
    )
    print("Migration applied." if args.apply else "Dry run only; pass --apply to migrate.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy dataset and LoRA directories.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--student-config", default="configs/students/student_qwen3.5_4b.yaml")
    parser.add_argument("--apply", action="store_true")
    command(parser.parse_args())
