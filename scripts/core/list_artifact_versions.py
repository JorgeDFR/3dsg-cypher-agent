# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

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
    parser = argparse.ArgumentParser(
        description="List dataset and LoRA versions, including each dataset's teacher."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--student-config", default="configs/students/student_qwen3.5_4b.yaml")
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
        print(f"Selected dataset: {data_config['dataset_version']}")
        print(f"Teacher: {data_config.get('teacher_name') or 'not generated'}")
        print(f"Traces: {data_config['teacher_traces_jsonl_path']}")

    student = load_student_family_config(args.student_config)
    print_versions(
        f"LoRA versions ({student['model_family']})",
        Path(student["model_family_root"]),
        completed_only=True,
    )


if __name__ == "__main__":
    main()
