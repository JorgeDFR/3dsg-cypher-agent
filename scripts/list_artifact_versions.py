from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from cypher_3dsg_agent_distillation.core.config import (
    GENERATED_DATASETS_ROOT,
    dataset_id_from_config,
    load_data_config,
    load_student_family_config,
    load_yaml,
)
from cypher_3dsg_agent_distillation.core.versioning import (
    read_metadata,
    version_directories,
)


def status(path: Path) -> str:
    try:
        return str(read_metadata(path).get("status") or "unknown")
    except (FileNotFoundError, TypeError):
        return "missing-metadata"


def print_versions(title: str, root: Path, *, completed_only: bool = False) -> None:
    versions = version_directories(root, completed_only=completed_only)
    print(f"{title}: {root}")
    if not versions:
        print("  none")
        return
    for index, path in enumerate(reversed(versions)):
        marker = " (latest)" if index == 0 else ""
        print(f"  {path.name} [{status(path)}]{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List dataset, teacher, and LoRA versions.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--student-config", default="configs/student_qwen3.5_4b.yaml")
    parser.add_argument("--dataset-version", default=None)
    args = parser.parse_args()

    raw_data = load_yaml(args.data_config)
    dataset_root = GENERATED_DATASETS_ROOT / dataset_id_from_config(raw_data)
    print_versions("Dataset versions", dataset_root)
    try:
        data_config = load_data_config(
            args.data_config, dataset_version=args.dataset_version
        )
    except FileNotFoundError:
        data_config = None
    if data_config:
        traces_root = Path(data_config["dataset_dir"]) / "traces"
        for teacher_root in sorted(path for path in traces_root.glob("*") if path.is_dir()):
            print_versions(
                f"Teacher runs ({teacher_root.name})",
                teacher_root,
                completed_only=True,
            )

    student = load_student_family_config(args.student_config)
    print_versions(
        f"LoRA versions ({student['model_family']})",
        Path(student["model_family_root"]),
        completed_only=True,
    )


if __name__ == "__main__":
    main()
