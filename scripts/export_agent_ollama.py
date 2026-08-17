from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from cypher_3dsg_agent_distillation.config import ensure_parent, load_data_config


def write_modelfile(
    *,
    model_name: str,
    gguf_path: str,
    output_path: str,
    system_prompt: str,
) -> Path:
    path = ensure_parent(output_path)
    path.write_text(
        "\n".join(
            [
                f"FROM {gguf_path}",
                "",
                f'SYSTEM """{system_prompt.strip()}"""',
                "",
                'PARAMETER stop "<|im_end|>"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "gguf_path": gguf_path,
                "modelfile": str(path),
                "recommended_quantization": "Q4_K_M",
                "secondary_quantization": "Q5_K_M",
                "target_hardware": "Jetson Orin NX 16GB",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def command(args: argparse.Namespace) -> None:
    config = load_data_config(args.data_config)
    export_config = config.get("ollama_export", {})
    write_modelfile(
        model_name=export_config.get("model_name", "3dsg-cypher-agent-distilled"),
        gguf_path=export_config.get("gguf_path", "outputs/model.Q4_K_M.gguf"),
        output_path=export_config.get("modelfile_path", "outputs/ollama/Modelfile"),
        system_prompt=export_config.get(
            "system_prompt",
            "You are a 3D scene graph agent. Use tools for read-only Cypher queries.",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create project-local Ollama export files.")
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
