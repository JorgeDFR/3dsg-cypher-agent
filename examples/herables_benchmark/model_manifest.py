#!/usr/bin/env python3
"""Bridge project-local Ollama exports into a Heracles Benchmark manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

MODEL_RUNTIME_FIELDS = ("alias", "model", "enabled", "capabilities", "parameters")
CAPABILITY_FIELDS = {"reasoning", "reasoning_efforts", "temperature", "seed"}


def required_text(entry: dict[str, Any], key: str, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model entry {index} requires non-empty string field {key!r}.")
    return value.strip()


def load_model_entries(models_path: Path) -> list[dict[str, Any]]:
    """Load and validate the repository-owned model sweep definition."""
    with models_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Model manifest must contain a 'models' list.")

    entries: list[dict[str, Any]] = []
    aliases: set[str] = set()
    model_names: set[str] = set()
    for index, raw_entry in enumerate(payload["models"]):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Model entry {index} must be a mapping.")
        entry = deepcopy(raw_entry)
        alias = required_text(entry, "alias", index)
        model = required_text(entry, "model", index)
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"Model entry {index} field 'enabled' must be boolean.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
            raise ValueError(f"Model alias contains unsupported characters: {alias}")
        if alias in aliases:
            raise ValueError(f"Duplicate model alias: {alias}")
        if model in model_names:
            raise ValueError(f"Duplicate Ollama model name: {model}")
        aliases.add(alias)
        model_names.add(model)

        if enabled:
            artifact_dir = required_text(entry, "artifact_dir", index)
            capabilities = entry.get("capabilities")
            if not isinstance(capabilities, dict):
                raise ValueError(f"Enabled model {model!r} requires a capabilities mapping.")
            missing = CAPABILITY_FIELDS - set(capabilities)
            unknown = set(capabilities) - CAPABILITY_FIELDS
            if missing or unknown:
                details = []
                if missing:
                    details.append("missing " + ", ".join(sorted(missing)))
                if unknown:
                    details.append("unknown " + ", ".join(sorted(unknown)))
                raise ValueError(f"Invalid capabilities for {model!r}: {'; '.join(details)}")
            entry["artifact_dir"] = artifact_dir

        entry.update(alias=alias, model=model, enabled=enabled)
        entries.append(entry)

    if not any(entry["enabled"] for entry in entries):
        raise ValueError("Model manifest has no enabled models.")
    return entries


def load_enabled_models(models_path: Path, artifacts_root: Path) -> list[dict[str, Any]]:
    """Resolve and validate the local files needed by ``ollama create``."""
    enabled: list[dict[str, Any]] = []
    artifacts_root = artifacts_root.resolve()
    for entry in load_model_entries(models_path):
        if not entry["enabled"]:
            continue
        artifact_dir_name = entry["artifact_dir"]
        artifact_dir = (artifacts_root / artifact_dir_name).resolve()
        try:
            artifact_dir.relative_to(artifacts_root)
        except ValueError as exc:
            raise ValueError(
                f"Model {entry['model']!r} artifact_dir escapes the artifacts root: "
                f"{artifact_dir_name}"
            ) from exc
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"Model artifact directory not found: {artifact_dir}")

        modelfile = artifact_dir / "Modelfile"
        metadata_path = artifact_dir / "Modelfile.json"
        if not modelfile.is_file():
            raise FileNotFoundError(f"Modelfile not found for {entry['model']}: {modelfile}")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Modelfile metadata not found for {entry['model']}: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gguf_path = Path(str(metadata.get("gguf_path", "")))
        if not gguf_path.is_absolute():
            gguf_path = artifact_dir / gguf_path
        elif not gguf_path.is_file():
            # Export metadata may contain the path used inside the training
            # container. The mounted artifact retains the same filename.
            gguf_path = artifact_dir / gguf_path.name
        if not gguf_path.is_file():
            raise FileNotFoundError(f"Selected GGUF not found for {entry['model']}: {gguf_path}")

        enabled.append(
            {
                **entry,
                "artifact_dir": str(artifact_dir),
                "modelfile": str(modelfile),
                "gguf_path": str(gguf_path),
            }
        )

    return enabled


def ollama_runtime_name(model: str) -> str:
    """Use the explicit tag emitted by Ollama's ``/api/tags`` endpoint."""
    final_component = model.rsplit("/", 1)[-1]
    return model if ":" in final_component else f"{model}:latest"


def materialize_benchmark_config(
    benchmark_config: Path,
    models_path: Path,
    output_path: Path,
    *,
    ollama_container_name: str,
) -> dict[str, Any]:
    """Combine an upstream question config with this repository's model list."""
    with benchmark_config.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Heracles benchmark config must use schema_version 1.")
    providers = manifest.get("providers")
    if not isinstance(providers, dict) or not isinstance(providers.get("ollama"), dict):
        raise ValueError("Heracles benchmark config must define providers.ollama.")

    runtime_models = []
    for entry in load_model_entries(models_path):
        runtime_entry = {key: deepcopy(entry[key]) for key in MODEL_RUNTIME_FIELDS if key in entry}
        runtime_entry["model"] = ollama_runtime_name(entry["model"])
        runtime_models.append(runtime_entry)

    ollama = providers["ollama"]
    ollama["enabled"] = True
    ollama["models"] = runtime_models
    local_metrics = ollama.get("local_metrics")
    if isinstance(local_metrics, dict):
        local_metrics["ollama_container_name"] = ollama_container_name

    # This integration intentionally benchmarks project-local Ollama exports.
    # Keeping the other provider structurally valid lets the upstream validator
    # remain the single authority for the complete manifest schema.
    openrouter = providers.get("openrouter")
    if not isinstance(openrouter, dict):
        openrouter = {}
        providers["openrouter"] = openrouter
    openrouter.update(enabled=False, models=[])
    openrouter.setdefault("require_parameters", True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-path", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--ollama-container-name", default="3dsg-heracles-benchmark-ollama")
    parser.add_argument("--format", choices=["json", "tsv"], default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.benchmark_config or args.output_config:
            if not args.benchmark_config or not args.output_config:
                raise ValueError(
                    "--benchmark-config and --output-config must be provided together."
                )
            manifest = materialize_benchmark_config(
                args.benchmark_config,
                args.models_path,
                args.output_config,
                ollama_container_name=args.ollama_container_name,
            )
            print(
                json.dumps(manifest, indent=2, sort_keys=True)
                if args.format == "json"
                else str(args.output_config)
            )
            return 0

        if args.artifacts_root is None:
            raise ValueError("--artifacts-root is required when provisioning models.")
        models = load_enabled_models(args.models_path, args.artifacts_root)
        if args.format == "tsv":
            for model in models:
                print(f"{ollama_runtime_name(model['model'])}\t{model['modelfile']}")
        else:
            print(json.dumps(models, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
