from __future__ import annotations

import json

import pytest

from cypher_3dsg_agent_distillation.core.config import load_yaml
from cypher_3dsg_agent_distillation.training import ollama_export
from cypher_3dsg_agent_distillation.training.ollama_export import (
    build_export_settings,
    convert_hf_to_gguf,
    expand_stages,
    merge_dtype_from_gguf_outtype,
    quantize_gguf,
    resolve_converter_python_bin,
    run_export,
    write_settings_modelfile,
)


def student_config(tmp_path):
    return {
        "model": {
            "base_model": "vendor/base-model",
            "tokenizer": None,
            "trust_remote_code": True,
        },
        "training": {
            "output_dir": str(tmp_path / "adapter"),
        },
        "ollama_export": {
            "model_name": "agent-distilled",
            "export_root": str(tmp_path),
            "stop_tokens": ["<stop>"],
            "quantizations": ["Q4_K_M", "Q8_0"],
        },
    }


def test_student_profiles_own_model_specific_export_settings():
    data = load_yaml("configs/data.yaml")
    qwen = load_yaml("configs/student_qwen3.5_4b.yaml")
    granite = load_yaml("configs/student_granite4.1_3b.yaml")

    assert "ollama_export" not in data
    assert qwen["ollama_export"]["model_name"] == "qwen3.5-4b-finetune"
    assert qwen["ollama_export"]["convert_args"] == ["--no-mtp"]
    assert granite["ollama_export"]["model_name"] == "granite4.1-3b-finetune"
    assert "convert_args" not in granite["ollama_export"]


def test_build_export_settings_derives_all_quantized_paths(tmp_path):
    settings = build_export_settings(student_config(tmp_path))
    export_dir = tmp_path / "agent-distilled"

    assert settings.model_name == "agent-distilled"
    assert settings.base_model == "vendor/base-model"
    assert settings.trust_remote_code is True
    assert settings.modelfile_quantization == "Q4_K_M"
    assert settings.merged_model_dir == export_dir / "merged-hf"
    assert settings.base_gguf_path == export_dir / "agent-distilled.bf16.gguf"
    assert settings.quantized_gguf_paths["Q4_K_M"] == export_dir / "agent-distilled.Q4_K_M.gguf"
    assert settings.quantized_gguf_paths["Q8_0"] == export_dir / "agent-distilled.Q8_0.gguf"
    assert settings.modelfile_gguf_path == export_dir / "agent-distilled.Q4_K_M.gguf"
    assert settings.modelfile_path == export_dir / "Modelfile"


def test_build_export_settings_separates_models_by_name(tmp_path):
    first = build_export_settings(student_config(tmp_path))
    second = build_export_settings(
        student_config(tmp_path),
        model_name="another-agent",
    )

    assert first.modelfile_path.parent == tmp_path / "agent-distilled"
    assert second.modelfile_path.parent == tmp_path / "another-agent"
    assert first.modelfile_gguf_path.parent != second.modelfile_gguf_path.parent


def test_write_settings_modelfile_writes_metadata(tmp_path):
    settings = build_export_settings(student_config(tmp_path))
    export_dir = tmp_path / "agent-distilled"
    settings.modelfile_gguf_path.parent.mkdir(parents=True)
    settings.modelfile_gguf_path.touch()

    modelfile_path = write_settings_modelfile(settings)

    assert modelfile_path.read_text(encoding="utf-8") == (
        'FROM ./agent-distilled.Q4_K_M.gguf\n\nPARAMETER stop "<stop>"\n'
    )
    metadata = json.loads((export_dir / "Modelfile.json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == "agent-distilled"
    assert metadata["model_variant"] == "fine-tuned"
    assert metadata["adapter_dir"] == str(tmp_path / "adapter")
    assert metadata["merged_model_dir"] == str(export_dir / "merged-hf")
    assert metadata["modelfile_quantization"] == "Q4_K_M"
    assert metadata["modelfile_gguf_path"] == str(export_dir / "agent-distilled.Q4_K_M.gguf")
    assert metadata["quantized_gguf_paths"]["Q8_0"] == str(
        export_dir / "agent-distilled.Q8_0.gguf"
    )


def test_base_only_settings_do_not_create_adapter_or_merged_paths(tmp_path):
    settings = build_export_settings(
        student_config(tmp_path),
        base_only=True,
        model_name="base-model",
    )

    assert settings.base_only is True
    assert settings.adapter_dir is None
    assert settings.merged_model_dir is None
    assert settings.base_gguf_path == tmp_path / "base-model" / "base-model.bf16.gguf"
    assert settings.modelfile_gguf_path == (
        tmp_path / "base-model" / "base-model.Q4_K_M.gguf"
    )


def test_base_only_settings_do_not_require_training_config(tmp_path):
    config = student_config(tmp_path)
    del config["training"]

    settings = build_export_settings(
        config,
        base_only=True,
        model_name="base-model",
    )

    assert settings.adapter_dir is None
    assert settings.merged_model_dir is None


def test_modelfile_gguf_override_selects_unquantized_artifact(tmp_path):
    unquantized = tmp_path / "agent-distilled" / "agent-distilled.bf16.gguf"
    unquantized.parent.mkdir(parents=True)
    unquantized.touch()
    settings = build_export_settings(
        student_config(tmp_path),
        modelfile_gguf="bf16",
    )

    write_settings_modelfile(settings)

    export_dir = tmp_path / "agent-distilled"
    assert settings.modelfile_gguf_path == unquantized
    assert settings.modelfile_quantization == "BF16"
    assert (
        (export_dir / "Modelfile")
        .read_text(encoding="utf-8")
        .startswith("FROM ./agent-distilled.bf16.gguf\n")
    )
    metadata = json.loads((export_dir / "Modelfile.json").read_text(encoding="utf-8"))
    assert metadata["gguf_path"] == str(unquantized)
    assert metadata["modelfile_gguf_path"] == str(unquantized)
    assert metadata["modelfile_quantization"] == "BF16"
    assert metadata["recommended_quantization"] == "BF16"
    assert metadata["quantized_gguf_paths"]["Q4_K_M"] == str(
        export_dir / "agent-distilled.Q4_K_M.gguf"
    )


def test_modelfile_gguf_short_quantization_resolves_artifact(tmp_path):
    settings = build_export_settings(
        student_config(tmp_path),
        modelfile_gguf="q8",
    )

    assert settings.modelfile_quantization == "Q8_0"
    assert settings.modelfile_gguf_path == (
        tmp_path / "agent-distilled" / "agent-distilled.Q8_0.gguf"
    )


def test_modelfile_gguf_rejects_paths(tmp_path):
    with pytest.raises(ValueError, match="expects a quantization"):
        build_export_settings(
            student_config(tmp_path),
            modelfile_gguf="outputs/model.bf16.gguf",
        )


def test_write_settings_modelfile_rejects_missing_quantization_file(tmp_path):
    settings = build_export_settings(
        student_config(tmp_path),
        modelfile_gguf="q8",
    )

    with pytest.raises(FileNotFoundError, match="quantization `Q8_0`"):
        write_settings_modelfile(settings)


def test_base_only_modelfile_metadata_omits_merge_fields(tmp_path):
    settings = build_export_settings(
        student_config(tmp_path),
        base_only=True,
        model_name="base-model",
    )
    settings.modelfile_gguf_path.parent.mkdir(parents=True)
    settings.modelfile_gguf_path.touch()

    write_settings_modelfile(settings)

    metadata = json.loads((tmp_path / "base-model" / "Modelfile.json").read_text(encoding="utf-8"))
    assert metadata["model_variant"] == "base"
    assert "adapter_dir" not in metadata
    assert "merged_model_dir" not in metadata
    assert not (tmp_path / "base-model" / "merged-hf").exists()


def test_convert_hf_to_gguf_invokes_llama_cpp_converter(tmp_path, monkeypatch):
    settings = build_export_settings(student_config(tmp_path))
    export_dir = tmp_path / "agent-distilled"
    settings.merged_model_dir.mkdir(parents=True)
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    convert_script.touch()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        output_path = command[command.index("--outfile") + 1]
        ollama_export.Path(output_path).touch()

    monkeypatch.setattr(ollama_export.subprocess, "run", fake_run)

    output_path = convert_hf_to_gguf(
        settings,
        llama_cpp_dir=llama_cpp_dir,
        python_bin="python",
    )

    assert output_path == export_dir / "agent-distilled.bf16.gguf"
    assert not settings.merged_model_dir.exists()
    assert calls == [
        (
            [
                "python",
                str(convert_script),
                str(export_dir / "merged-hf"),
                "--outfile",
                str(export_dir / "agent-distilled.bf16.gguf"),
                "--outtype",
                "bf16",
                "--model-name",
                "agent-distilled",
            ],
            True,
        )
    ]


def test_convert_hf_to_gguf_uses_llama_cpp_venv_python_by_default(tmp_path, monkeypatch):
    settings = build_export_settings(student_config(tmp_path))
    settings.merged_model_dir.mkdir(parents=True)
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    convert_script.touch()
    venv_python = llama_cpp_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        output_path = command[command.index("--outfile") + 1]
        ollama_export.Path(output_path).touch()

    monkeypatch.setattr(ollama_export.subprocess, "run", fake_run)

    convert_hf_to_gguf(settings, llama_cpp_dir=llama_cpp_dir)

    assert calls[0][0][0] == str(venv_python)


def test_convert_failure_retains_merged_model(tmp_path, monkeypatch):
    settings = build_export_settings(student_config(tmp_path))
    settings.merged_model_dir.mkdir(parents=True)
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    (llama_cpp_dir / "convert_hf_to_gguf.py").touch()

    def fail_conversion(_command, check):
        raise ollama_export.subprocess.CalledProcessError(1, "convert")

    monkeypatch.setattr(ollama_export.subprocess, "run", fail_conversion)

    with pytest.raises(ollama_export.subprocess.CalledProcessError):
        convert_hf_to_gguf(settings, llama_cpp_dir=llama_cpp_dir)

    assert settings.merged_model_dir.exists()


def test_convert_rejects_gguf_output_inside_cleanup_directory(tmp_path):
    merged_dir = tmp_path / "agent-distilled" / "merged-hf"
    settings = build_export_settings(
        student_config(tmp_path),
        merged_model_dir=merged_dir,
        base_gguf_path=merged_dir / "agent-distilled.bf16.gguf",
    )

    with pytest.raises(ValueError, match="must be outside"):
        convert_hf_to_gguf(settings, llama_cpp_dir=tmp_path / "llama.cpp")


def test_convert_base_model_uses_resolved_hugging_face_snapshot(tmp_path, monkeypatch):
    settings = build_export_settings(
        student_config(tmp_path),
        base_only=True,
        model_name="base-model",
    )
    snapshot = tmp_path / "huggingface-cache" / "snapshot"
    snapshot.mkdir(parents=True)
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    convert_script.touch()
    calls = []

    monkeypatch.setattr(
        ollama_export,
        "resolve_base_model_snapshot",
        lambda _settings: snapshot,
    )
    monkeypatch.setattr(
        ollama_export.subprocess,
        "run",
        lambda command, check: (
            calls.append((command, check)),
            ollama_export.Path(command[command.index("--outfile") + 1]).touch(),
        ),
    )

    convert_hf_to_gguf(
        settings,
        llama_cpp_dir=llama_cpp_dir,
        python_bin="python",
    )

    assert calls[0][0][2] == str(snapshot)
    assert not (tmp_path / "base-model" / "merged-hf").exists()


def test_quantize_gguf_invokes_llama_quantize_for_each_quantization(tmp_path, monkeypatch):
    settings = build_export_settings(student_config(tmp_path))
    export_dir = tmp_path / "agent-distilled"
    settings.base_gguf_path.parent.mkdir(parents=True)
    settings.base_gguf_path.touch()
    quantize_bin = tmp_path / "llama-quantize"
    quantize_bin.touch()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(ollama_export.subprocess, "run", fake_run)

    outputs = quantize_gguf(settings, quantize_bin=quantize_bin)

    assert outputs == {
        "Q4_K_M": export_dir / "agent-distilled.Q4_K_M.gguf",
        "Q8_0": export_dir / "agent-distilled.Q8_0.gguf",
    }
    assert calls == [
        (
            [
                str(quantize_bin),
                str(export_dir / "agent-distilled.bf16.gguf"),
                str(export_dir / "agent-distilled.Q4_K_M.gguf"),
                "Q4_K_M",
            ],
            True,
        ),
        (
            [
                str(quantize_bin),
                str(export_dir / "agent-distilled.bf16.gguf"),
                str(export_dir / "agent-distilled.Q8_0.gguf"),
                "Q8_0",
            ],
            True,
        ),
    ]


def test_expand_stages_keeps_deployment_order():
    assert expand_stages(["all"]) == [
        "merge",
        "convert",
        "quantize",
        "modelfile",
    ]


def test_expand_stages_rejects_ollama_create_stage():
    try:
        expand_stages(["create"])
    except ValueError as exc:
        assert "Unsupported export stage `create`" in str(exc)
    else:
        raise AssertionError("create stage should not be supported")


def test_merge_dtype_from_gguf_outtype_maps_llama_cpp_names_to_torch_names():
    assert merge_dtype_from_gguf_outtype("bf16") == "bfloat16"
    assert merge_dtype_from_gguf_outtype("f16") == "float16"
    assert merge_dtype_from_gguf_outtype("q8_0") == "auto"


def test_resolve_converter_python_bin_keeps_explicit_interpreter(tmp_path):
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    venv_python = llama_cpp_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()

    assert (
        resolve_converter_python_bin(
            python_bin="python",
            llama_cpp_dir=llama_cpp_dir,
        )
        == "python"
    )


def test_run_export_dry_run_does_not_write_modelfile(tmp_path):
    settings = build_export_settings(student_config(tmp_path))

    run_export(settings, stages=["modelfile"], dry_run=True)

    assert not (tmp_path / "agent-distilled" / "Modelfile").exists()
    assert not (tmp_path / "agent-distilled" / "Modelfile.json").exists()


def test_run_export_base_only_skips_merge_stage(tmp_path, monkeypatch):
    settings = build_export_settings(
        student_config(tmp_path),
        base_only=True,
        model_name="base-model",
    )
    merges = []
    monkeypatch.setattr(
        ollama_export,
        "merge_lora_adapter",
        lambda *_args, **_kwargs: merges.append(True),
    )

    run_export(settings, stages=["merge"], dry_run=False)

    assert merges == []
