"""Configuration and preflight helpers for the Docker deployment workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

from cypher_3dsg_agent_distillation.core.cli_output import print_summary
from cypher_3dsg_agent_distillation.core.config import (
    load_student_config,
    load_student_family_config,
)
from cypher_3dsg_agent_distillation.training.ollama_export import (
    build_export_settings,
    settings_for_modelfile_variant,
)


def deployment_config(student_config: str) -> list[str]:
    """Return the model name followed by canonical Modelfile variant names."""
    config = load_student_family_config(student_config)
    name = (config.get("ollama_export") or {}).get("model_name")
    if not isinstance(name, str) or not name.strip() or "\n" in name or "\r" in name:
        raise ValueError("ollama_export.model_name must be a non-empty single-line string")
    settings = build_export_settings(config)
    variants = dict.fromkeys(
        settings_for_modelfile_variant(settings, selector).modelfile_quantization
        for selector in [settings.gguf_outtype, *settings.quantized_gguf_paths]
    )
    return [settings.model_name, *variants]


def require_trained_student(student_config: str) -> Path:
    config = load_student_config(student_config)
    adapter_dir = Path(config["training"]["output_dir"])
    has_weights = any(
        (adapter_dir / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if not (adapter_dir / "adapter_config.json").is_file() or not has_weights:
        raise FileNotFoundError(f"Trained student model not found in {adapter_dir}")
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("config", "check"):
        commands.add_parser(command).add_argument("--student-config", required=True)
    summary = commands.add_parser("registration-summary")
    summary.add_argument("--model-name", required=True)
    summary.add_argument("--variants", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "config":
        print("\n".join(deployment_config(args.student_config)))
    elif args.command == "check":
        adapter_dir = require_trained_student(args.student_config)
        print(f"[INFO] Found trained student model in {adapter_dir}.")
    else:
        print_summary(
            "Ollama registration complete",
            [(variant, f"{args.model_name}-{variant.lower()}") for variant in args.variants],
        )


if __name__ == "__main__":
    main()
