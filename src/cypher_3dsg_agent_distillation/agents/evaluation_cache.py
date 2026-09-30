"""Content-based identities for completed student evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.core.versioning import (
    file_fingerprint,
    mapping_fingerprint,
    serializable_arguments,
)


def model_files(source: str | None) -> dict[str, Any]:
    if not source or not Path(source).is_dir():
        return {}
    root = Path(source)
    return {
        str(path.relative_to(root)): file_fingerprint(path)
        for path in sorted(root.iterdir())
        if path.is_file()
        and path.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja"}
    }


def evaluation_identity(
    args: argparse.Namespace,
    data_config: dict[str, Any],
    student_config: dict[str, Any],
    adapter_dir: str,
    *,
    runtime_identity: dict[str, Any] | None = None,
) -> str:
    splits = data_config["splits"]
    selected = splits if args.split == "all" else {args.split: splits[args.split]}
    arguments = serializable_arguments(args, excluded={
        "force", "output", "no_progress", "show_db_notifications",
        "data_config", "student_config", "dataset_version", "model_version",
        "allow_dataset_mismatch", "adapter_dir", "username", "uri", "database",
        "ollama_host", "ollama_timeout", "ollama_keep_alive",
    })
    model = student_config["model"]
    identity = {
        "schema_version": 1,
        "dataset": [data_config.get("dataset_id"), data_config.get("dataset_version")],
        "splits": {name: file_fingerprint(path) for name, path in selected.items()},
        "scenes": [
            file_fingerprint(scene["scene_path"]) for scene in data_config.get("scenes", [])
        ],
        "schema": (
            file_fingerprint(data_config["schema_path"])
            if data_config.get("schema_path") else None
        ),
        "model": model,
        "model_files": model_files(model.get("base_model")),
        "tokenizer_files": model_files(args.tokenizer or model.get("tokenizer")),
        "model_version": None if args.baseline else student_config.get("model_version"),
        "adapter_files": {} if args.baseline else model_files(adapter_dir),
        "runtime_identity": runtime_identity,
        "arguments": arguments,
    }
    if args.mode == "eval-loss":
        training = student_config.get("training") or {}
        identity["loss_defaults"] = {
            "max_length": args.max_length or training.get("max_length", 4096),
            "loss_mode": args.loss_mode or training.get("loss_mode", "assistant-only"),
        }
    return mapping_fingerprint(identity)


def completed_evaluation(path: Path, identity: str) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(report, dict):
        return False
    metadata = report.get("metadata")
    return (
        isinstance(metadata, dict)
        and metadata.get("evaluation_status") == "complete"
        and metadata.get("evaluation_identity") == identity
        and isinstance(report.get("summary"), dict)
    )


def evaluation_completed(report: dict[str, Any]) -> bool:
    """Model mistakes are results; unavailable services are incomplete evaluations."""
    raw = report["metadata"].get("raw_generation")
    if raw is None:
        return True
    if not (raw.get("neo4j") or {}).get("connected"):
        return False
    for record in report.get("records", []):
        for section in record.values():
            if isinstance(section, dict) and section.get("status") in {
                "generation_error", "evaluation_error",
            }:
                return False
    return True
