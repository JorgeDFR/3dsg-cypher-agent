#!/usr/bin/env python3
"""Validate benchmark model artifacts and emit provisioning records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_enabled_models(models_path: Path, artifacts_root: Path) -> list[dict[str, Any]]:
    with models_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Model manifest must contain a 'models' list.")

    enabled: list[dict[str, Any]] = []
    aliases: set[str] = set()
    model_names: set[str] = set()
    for index, entry in enumerate(payload["models"]):
        if not isinstance(entry, dict):
            raise ValueError(f"Model entry {index} must be a mapping.")
        if entry.get("enabled", True) is False:
            continue

        alias = required_text(entry, "alias", index)
        model = required_text(entry, "model", index)
        artifact_dir_name = required_text(entry, "artifact_dir", index)
        if alias in aliases:
            raise ValueError(f"Duplicate enabled model alias: {alias}")
        if model in model_names:
            raise ValueError(f"Duplicate enabled Ollama model name: {model}")
        aliases.add(alias)
        model_names.add(model)

        artifact_dir = (artifacts_root / artifact_dir_name).resolve()
        try:
            artifact_dir.relative_to(artifacts_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Model {model!r} artifact_dir escapes the artifacts root: {artifact_dir_name}"
            ) from exc
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"Model artifact directory not found: {artifact_dir}")

        modelfile = artifact_dir / "Modelfile"
        metadata_path = artifact_dir / "Modelfile.json"
        if not modelfile.is_file():
            raise FileNotFoundError(f"Modelfile not found for {model}: {modelfile}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Modelfile metadata not found for {model}: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gguf_path = Path(str(metadata.get("gguf_path", "")))
        if not gguf_path.is_absolute():
            gguf_path = artifact_dir / gguf_path
        elif not gguf_path.is_file():
            # Metadata written in Docker contains a container path. The artifact
            # filename remains stable when mounted into the benchmark container.
            gguf_path = artifact_dir / gguf_path.name
        if not gguf_path.is_file():
            raise FileNotFoundError(f"Selected GGUF not found for {model}: {gguf_path}")

        enabled.append(
            {
                **entry,
                "alias": alias,
                "model": model,
                "artifact_dir": str(artifact_dir),
                "modelfile": str(modelfile),
                "gguf_path": str(gguf_path),
            }
        )

    if not enabled:
        raise ValueError("Model manifest has no enabled models.")
    return enabled


def required_text(entry: dict[str, Any], key: str, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model entry {index} requires non-empty string field {key!r}.")
    return value.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-path", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--format", choices=["json", "tsv"], default="json")
    args = parser.parse_args()

    models = load_enabled_models(args.models_path, args.artifacts_root)
    if args.format == "tsv":
        for model in models:
            print(f"{model['model']}\t{model['modelfile']}")
    else:
        print(json.dumps(models, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
