from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.config import (
    ensure_parent,
    expanded_path,
    load_data_config,
    load_yaml,
    resolve_project_path,
)
from cypher_3dsg_agent_distillation.train import dtype_from_name

DEFAULT_MODEL_NAME = "3dsg-cypher-agent-distilled"
DEFAULT_QUANTIZATION = "Q4_K_M"
DEFAULT_GGUF_OUTTYPE = "bf16"
DEFAULT_STOP_TOKENS = ["<|im_end|>"]

EXPORT_STAGES = ("merge", "convert", "quantize", "modelfile")
ALL_STAGES = ("merge", "convert", "quantize", "modelfile")
GGUF_OUTTYPE_TO_TORCH_DTYPE = {
    "auto": "auto",
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
}


@dataclass(frozen=True)
class OllamaExportSettings:
    model_name: str
    base_model: str
    base_only: bool
    tokenizer: str | None
    trust_remote_code: bool
    adapter_dir: Path | None
    merged_model_dir: Path | None
    base_gguf_path: Path
    quantized_gguf_paths: dict[str, Path]
    selected_quantization: str
    modelfile_gguf_path: Path
    modelfile_path: Path
    stop_tokens: list[str]
    gguf_outtype: str
    convert_args: list[str]
    quantize_args: list[str]

    @property
    def selected_gguf_path(self) -> Path:
        return self.quantized_gguf_paths[self.selected_quantization]


def require_export_dependencies():
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        msg = "Install export dependencies with `pip install -e '.[export]'` before merging."
        raise RuntimeError(msg) from exc
    return AutoModelForCausalLM, AutoTokenizer, PeftModel


def require_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install export dependencies with `pip install -e '.[export]'` "
            "before exporting a base model."
        ) from exc
    return snapshot_download


def as_string_list(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise TypeError(f"{name} must be a string or list of strings.")


def normalized_quantizations(value: Any) -> list[str]:
    quantizations = as_string_list(value, name="ollama_export.quantizations")
    if not quantizations:
        quantizations = [DEFAULT_QUANTIZATION]
    normalized = []
    for quantization in quantizations:
        normalized_quantization = quantization.strip().upper()
        if not normalized_quantization:
            raise ValueError("Quantization names must not be empty.")
        if normalized_quantization not in normalized:
            normalized.append(normalized_quantization)
    return normalized


def merge_dtype_from_gguf_outtype(outtype: str) -> str:
    normalized = outtype.strip().lower()
    if normalized not in GGUF_OUTTYPE_TO_TORCH_DTYPE:
        return "auto"
    return GGUF_OUTTYPE_TO_TORCH_DTYPE[normalized]


def resolve_optional_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return resolve_project_path(value)


def export_root_from_config(export_config: dict[str, Any]) -> Path:
    if export_config.get("export_root"):
        return resolve_project_path(export_config["export_root"])
    if export_config.get("modelfile_path"):
        return resolve_project_path(export_config["modelfile_path"]).parent
    return resolve_project_path("outputs/ollama")


def configured_primary_quantization(export_config: dict[str, Any]) -> str:
    if export_config.get("quantization"):
        return str(export_config["quantization"]).strip().upper()
    return DEFAULT_QUANTIZATION


def derived_quantized_path(
    *,
    model_name: str,
    quantization: str,
    export_dir: Path,
    configured_gguf_path: Path | None,
    configured_quantization: str,
) -> Path:
    if configured_gguf_path is None:
        return export_dir / f"{model_name}.{quantization}.gguf"
    if quantization == configured_quantization:
        return configured_gguf_path

    suffix = f".{configured_quantization}"
    stem = configured_gguf_path.stem
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return configured_gguf_path.with_name(f"{stem}.{quantization}.gguf")


def build_export_settings(
    data_config: dict[str, Any],
    student_config: dict[str, Any],
    *,
    model_name: str | None = None,
    adapter_dir: str | Path | None = None,
    merged_model_dir: str | Path | None = None,
    base_gguf_path: str | Path | None = None,
    gguf_path: str | Path | None = None,
    modelfile_gguf: str | Path | None = None,
    modelfile_path: str | Path | None = None,
    base_only: bool = False,
    quantizations: Sequence[str] | None = None,
    selected_quantization: str | None = None,
    gguf_outtype: str | None = None,
) -> OllamaExportSettings:
    export_config = dict(data_config.get("ollama_export") or {})
    model_config = student_config["model"]
    training_config = student_config.get("training") or {}
    configured_adapter_dir = adapter_dir or training_config.get("output_dir")
    if not base_only and configured_adapter_dir is None:
        raise ValueError("Fine-tuned exports require --adapter-dir or training.output_dir.")

    resolved_model_name = model_name or export_config.get("model_name") or DEFAULT_MODEL_NAME
    export_root = export_root_from_config(export_config)
    export_dir = export_root / str(resolved_model_name)
    resolved_gguf_outtype = (
        gguf_outtype or export_config.get("gguf_outtype") or DEFAULT_GGUF_OUTTYPE
    )
    configured_gguf_path = resolve_optional_project_path(
        gguf_path or export_config.get("gguf_path")
    )
    configured_quantization = configured_primary_quantization(export_config)
    resolved_quantizations = normalized_quantizations(
        quantizations or export_config.get("quantizations") or configured_quantization
    )
    resolved_selected_quantization = (
        (
            selected_quantization
            or export_config.get("selected_quantization")
            or resolved_quantizations[0]
        )
        .strip()
        .upper()
    )
    if resolved_selected_quantization not in resolved_quantizations:
        msg = (
            f"selected quantization `{resolved_selected_quantization}` is not in "
            f"configured quantizations: {', '.join(resolved_quantizations)}"
        )
        raise ValueError(msg)

    resolved_base_gguf_path = resolve_project_path(
        base_gguf_path
        or export_config.get("base_gguf_path")
        or export_dir / f"{resolved_model_name}.{resolved_gguf_outtype}.gguf"
    )
    resolved_quantized_gguf_paths = {
        quantization: derived_quantized_path(
            model_name=str(resolved_model_name),
            quantization=quantization,
            export_dir=export_dir,
            configured_gguf_path=configured_gguf_path,
            configured_quantization=configured_quantization,
        )
        for quantization in resolved_quantizations
    }
    resolved_modelfile_gguf_path = resolve_project_path(
        modelfile_gguf
        or export_config.get("modelfile_gguf")
        or resolved_quantized_gguf_paths[resolved_selected_quantization]
    )

    return OllamaExportSettings(
        model_name=str(resolved_model_name),
        base_model=str(model_config["base_model"]),
        base_only=base_only,
        tokenizer=model_config.get("tokenizer"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        adapter_dir=(None if base_only else resolve_project_path(configured_adapter_dir)),
        merged_model_dir=(
            None
            if base_only
            else resolve_project_path(
                merged_model_dir
                or export_config.get("merged_model_dir")
                or export_dir / "merged-hf"
            )
        ),
        base_gguf_path=resolved_base_gguf_path,
        quantized_gguf_paths=resolved_quantized_gguf_paths,
        selected_quantization=resolved_selected_quantization,
        modelfile_gguf_path=resolved_modelfile_gguf_path,
        modelfile_path=resolve_project_path(
            modelfile_path or export_config.get("modelfile_path") or export_dir / "Modelfile"
        ),
        stop_tokens=as_string_list(
            export_config.get("stop_tokens", DEFAULT_STOP_TOKENS),
            name="ollama_export.stop_tokens",
        ),
        gguf_outtype=str(resolved_gguf_outtype),
        convert_args=as_string_list(
            export_config.get("convert_args"),
            name="ollama_export.convert_args",
        ),
        quantize_args=as_string_list(
            export_config.get("quantize_args"),
            name="ollama_export.quantize_args",
        ),
    )


def tokenizer_source(settings: OllamaExportSettings) -> str:
    if settings.tokenizer:
        return str(settings.tokenizer)
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    if settings.adapter_dir and any(
        (settings.adapter_dir / filename).exists() for filename in tokenizer_files
    ):
        return str(settings.adapter_dir)
    return settings.base_model


def merge_lora_adapter(
    settings: OllamaExportSettings,
    *,
    device_map: str | None = "auto",
    dtype: str | None = None,
    max_shard_size: str = "4GB",
) -> Path:
    if settings.base_only:
        raise ValueError("Cannot merge a LoRA adapter for a base-only export.")
    if settings.adapter_dir is None or settings.merged_model_dir is None:
        raise ValueError("Fine-tuned exports require adapter and merged-model paths.")
    if not settings.adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {settings.adapter_dir}")

    AutoModelForCausalLM, AutoTokenizer, PeftModel = require_export_dependencies()
    model_dtype = dtype_from_name(dtype or merge_dtype_from_gguf_outtype(settings.gguf_outtype))
    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "trust_remote_code": settings.trust_remote_code,
    }
    if device_map:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(settings.base_model, **model_kwargs)
    peft_model = PeftModel.from_pretrained(model, str(settings.adapter_dir))
    merged_model = peft_model.merge_and_unload()
    output_dir = settings.merged_model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source(settings),
        trust_remote_code=settings.trust_remote_code,
    )
    tokenizer.save_pretrained(output_dir)
    return output_dir


def resolve_base_model_snapshot(settings: OllamaExportSettings) -> Path:
    """Resolve the exact Hugging Face base checkpoint without copying it to outputs."""

    candidate = Path(settings.base_model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    snapshot_download = require_huggingface_hub()
    return Path(snapshot_download(repo_id=settings.base_model)).resolve()


def conversion_source_dir(settings: OllamaExportSettings) -> Path:
    if settings.base_only:
        return resolve_base_model_snapshot(settings)
    if settings.merged_model_dir is None:
        raise ValueError("Fine-tuned exports require a merged Hugging Face directory.")
    return settings.merged_model_dir


def require_existing_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def require_existing_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def resolve_llama_cpp_dir(value: str | Path | None) -> Path:
    raw_value = value or os.environ.get("LLAMA_CPP_DIR")
    if raw_value is None:
        raise RuntimeError("Set --llama-cpp-dir or LLAMA_CPP_DIR to a llama.cpp checkout.")
    path = expanded_path(raw_value)
    if not path.is_absolute():
        path = resolve_project_path(path)
    return require_existing_dir(path, "llama.cpp directory")


def optional_llama_cpp_dir(value: str | Path | None) -> Path | None:
    raw_value = value or os.environ.get("LLAMA_CPP_DIR")
    if raw_value is None:
        return None
    path = expanded_path(raw_value)
    if not path.is_absolute():
        path = resolve_project_path(path)
    return path


def resolve_converter_python_bin(
    *,
    python_bin: str | Path | None,
    llama_cpp_dir: str | Path | None,
) -> str:
    if python_bin is not None:
        return str(python_bin)

    resolved_llama_cpp_dir = optional_llama_cpp_dir(llama_cpp_dir)
    if resolved_llama_cpp_dir is not None:
        candidates = [
            resolved_llama_cpp_dir / ".venv" / "bin" / "python",
            resolved_llama_cpp_dir / ".venv" / "Scripts" / "python.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

    return sys.executable


def resolve_convert_script(
    *,
    llama_cpp_dir: str | Path | None,
    convert_script: str | Path | None,
) -> Path:
    if convert_script is not None:
        path = expanded_path(convert_script)
        if not path.is_absolute():
            path = resolve_project_path(path)
        return require_existing_file(path, "llama.cpp convert_hf_to_gguf.py")
    resolved_llama_cpp_dir = resolve_llama_cpp_dir(llama_cpp_dir)
    return require_existing_file(
        resolved_llama_cpp_dir / "convert_hf_to_gguf.py",
        "llama.cpp convert_hf_to_gguf.py",
    )


def resolve_quantize_binary(
    *,
    llama_cpp_dir: str | Path | None,
    quantize_bin: str | Path | None,
) -> Path:
    if quantize_bin is not None:
        path = expanded_path(quantize_bin)
        if not path.is_absolute():
            path = resolve_project_path(path)
        return require_existing_file(path, "llama.cpp llama-quantize binary")

    resolved_llama_cpp_dir = resolve_llama_cpp_dir(llama_cpp_dir)
    candidates = [
        resolved_llama_cpp_dir / "build" / "bin" / "llama-quantize",
        resolved_llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize",
        resolved_llama_cpp_dir / "llama-quantize",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find llama-quantize. Build llama.cpp with CMake or pass --quantize-bin."
    )


def run_checked(command: list[str], *, dry_run: bool = False) -> None:
    print(shlex.join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def convert_hf_to_gguf(
    settings: OllamaExportSettings,
    *,
    llama_cpp_dir: str | Path | None = None,
    convert_script: str | Path | None = None,
    python_bin: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    source_dir = (
        Path(settings.base_model)
        if dry_run and settings.base_only
        else conversion_source_dir(settings)
    )
    if not dry_run:
        description = (
            "base Hugging Face model directory"
            if settings.base_only
            else "merged Hugging Face model directory"
        )
        require_existing_dir(source_dir, description)
    script = resolve_convert_script(
        llama_cpp_dir=llama_cpp_dir,
        convert_script=convert_script,
    )
    output_path = ensure_parent(settings.base_gguf_path)
    command = [
        resolve_converter_python_bin(
            python_bin=python_bin,
            llama_cpp_dir=llama_cpp_dir,
        ),
        str(script),
        str(source_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        settings.gguf_outtype,
        "--model-name",
        settings.model_name,
        *settings.convert_args,
    ]
    run_checked(command, dry_run=dry_run)
    return output_path


def quantize_gguf(
    settings: OllamaExportSettings,
    *,
    llama_cpp_dir: str | Path | None = None,
    quantize_bin: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Path]:
    if not dry_run:
        require_existing_file(settings.base_gguf_path, "base GGUF file")
    binary = resolve_quantize_binary(
        llama_cpp_dir=llama_cpp_dir,
        quantize_bin=quantize_bin,
    )
    outputs = {}
    for quantization, output_path in settings.quantized_gguf_paths.items():
        ensured_output_path = ensure_parent(output_path)
        command = [
            str(binary),
            *settings.quantize_args,
            str(settings.base_gguf_path),
            str(ensured_output_path),
            quantization,
        ]
        run_checked(command, dry_run=dry_run)
        outputs[quantization] = ensured_output_path
    return outputs


def write_modelfile(
    *,
    model_name: str,
    gguf_path: str | Path,
    output_path: str | Path,
    stop_tokens: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = ensure_parent(output_path)
    gguf_reference = modelfile_gguf_reference(gguf_path, path)
    lines = [
        f"FROM {gguf_reference}",
        "",
    ]
    for stop_token in stop_tokens or DEFAULT_STOP_TOKENS:
        lines.append(f'PARAMETER stop "{stop_token}"')
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_payload = {
        "model_name": model_name,
        "gguf_path": str(gguf_path),
        "modelfile": str(path),
        "recommended_quantization": DEFAULT_QUANTIZATION,
    }
    if metadata:
        metadata_payload.update(metadata)
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def modelfile_gguf_reference(gguf_path: str | Path, modelfile_path: Path) -> str:
    gguf = Path(gguf_path)
    if gguf.is_absolute():
        try:
            if gguf.parent.resolve() == modelfile_path.parent.resolve():
                return f"./{gguf.name}"
        except FileNotFoundError:
            pass
        return str(gguf)
    if gguf.parent == Path("."):
        return f"./{gguf.name}"
    return str(gguf)


def export_metadata(settings: OllamaExportSettings) -> dict[str, Any]:
    metadata = {
        "base_model": settings.base_model,
        "model_variant": "base" if settings.base_only else "fine-tuned",
        "base_gguf_path": str(settings.base_gguf_path),
        "gguf_path": str(settings.modelfile_gguf_path),
        "modelfile_gguf_path": str(settings.modelfile_gguf_path),
        "quantized_gguf_paths": {
            quantization: str(path) for quantization, path in settings.quantized_gguf_paths.items()
        },
        "selected_quantization": settings.selected_quantization,
        "gguf_outtype": settings.gguf_outtype,
        "recommended_quantization": settings.selected_quantization,
    }
    if settings.adapter_dir is not None:
        metadata["adapter_dir"] = str(settings.adapter_dir)
    if settings.merged_model_dir is not None:
        metadata["merged_model_dir"] = str(settings.merged_model_dir)
    return metadata


def write_settings_modelfile(settings: OllamaExportSettings) -> Path:
    return write_modelfile(
        model_name=settings.model_name,
        gguf_path=settings.modelfile_gguf_path,
        output_path=settings.modelfile_path,
        stop_tokens=settings.stop_tokens,
        metadata=export_metadata(settings),
    )


def expand_stages(stages: Sequence[str] | None) -> list[str]:
    if not stages:
        return ["modelfile"]
    selected = set()
    for stage in stages:
        if stage == "all":
            selected.update(ALL_STAGES)
            continue
        if stage not in EXPORT_STAGES:
            expected = ", ".join(("all", *EXPORT_STAGES))
            raise ValueError(f"Unsupported export stage `{stage}`. Expected one of: {expected}.")
        selected.add(stage)
    return [stage for stage in EXPORT_STAGES if stage in selected]


def run_export(
    settings: OllamaExportSettings,
    *,
    stages: Sequence[str],
    llama_cpp_dir: str | Path | None = None,
    convert_script: str | Path | None = None,
    quantize_bin: str | Path | None = None,
    python_bin: str | Path | None = None,
    device_map: str | None = "auto",
    merge_dtype: str | None = None,
    max_shard_size: str = "4GB",
    dry_run: bool = False,
) -> None:
    for stage in expand_stages(stages):
        if stage == "merge":
            if settings.base_only:
                if dry_run:
                    print(f"base {settings.base_model} (no LoRA merge)")
                continue
            if dry_run:
                print(
                    "merge "
                    f"{settings.base_model} + {settings.adapter_dir} -> "
                    f"{settings.merged_model_dir}"
                )
                continue
            merge_lora_adapter(
                settings,
                device_map=device_map,
                dtype=merge_dtype,
                max_shard_size=max_shard_size,
            )
        elif stage == "convert":
            convert_hf_to_gguf(
                settings,
                llama_cpp_dir=llama_cpp_dir,
                convert_script=convert_script,
                python_bin=python_bin,
                dry_run=dry_run,
            )
        elif stage == "quantize":
            quantize_gguf(
                settings,
                llama_cpp_dir=llama_cpp_dir,
                quantize_bin=quantize_bin,
                dry_run=dry_run,
            )
        elif stage == "modelfile":
            if dry_run:
                print(f"modelfile {settings.modelfile_gguf_path} -> {settings.modelfile_path}")
                continue
            write_settings_modelfile(settings)


def command(args: argparse.Namespace) -> None:
    data_config = load_data_config(args.data_config)
    student_config = load_yaml(resolve_project_path(args.student_config))
    settings = build_export_settings(
        data_config,
        student_config,
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        merged_model_dir=args.merged_model_dir,
        base_gguf_path=args.base_gguf_path,
        gguf_path=args.gguf_path,
        modelfile_gguf=args.modelfile_gguf,
        modelfile_path=args.modelfile_path,
        base_only=args.base_only,
        quantizations=args.quantizations,
        selected_quantization=args.selected_quantization,
        gguf_outtype=args.outtype,
    )
    run_export(
        settings,
        stages=args.stages,
        llama_cpp_dir=args.llama_cpp_dir,
        convert_script=args.convert_script,
        quantize_bin=args.quantize_bin,
        python_bin=args.python_bin,
        device_map=args.device_map,
        merge_dtype=args.merge_dtype,
        max_shard_size=args.max_shard_size,
        dry_run=args.dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a base or LoRA-fine-tuned student to Ollama/GGUF artifacts."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--student-config",
        default="configs/student_qwen3.5_4b.yaml",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["modelfile"],
        choices=("all", *EXPORT_STAGES),
        help=(
            "Export stages to run. `all` expands to merge, convert, quantize, and modelfile. "
            "Default keeps the previous Modelfile-only behavior."
        ),
    )
    parser.add_argument("--model-name")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--merged-model-dir")
    parser.add_argument("--base-gguf-path")
    parser.add_argument("--gguf-path")
    parser.add_argument(
        "--modelfile-gguf",
        help=(
            "GGUF path to write in the Modelfile. Defaults to the selected "
            "quantized GGUF; pass the base GGUF path to use BF16/F16 directly."
        ),
    )
    parser.add_argument("--modelfile-path")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help=(
            "Export model.base_model directly without loading or merging the LoRA "
            "adapter. Use --model-name to select its Ollama directory/name."
        ),
    )
    parser.add_argument("--quantizations", nargs="+")
    parser.add_argument("--selected-quantization")
    parser.add_argument(
        "--outtype",
        default=None,
        help="GGUF conversion outtype, e.g. bf16 or f16.",
    )
    parser.add_argument("--llama-cpp-dir", help="llama.cpp checkout. Defaults to LLAMA_CPP_DIR.")
    parser.add_argument("--convert-script", help="Path to llama.cpp convert_hf_to_gguf.py.")
    parser.add_argument("--quantize-bin", help="Path to built llama-quantize binary.")
    parser.add_argument(
        "--python-bin",
        default=None,
        help=(
            "Python interpreter for llama.cpp conversion. Defaults to "
            "<llama-cpp-dir>/.venv/bin/python when present, else this interpreter."
        ),
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--merge-dtype",
        help="Torch dtype used when loading the base model to merge.",
    )
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
