from __future__ import annotations

import json

from cypher_3dsg_agent_distillation import ollama_export
from cypher_3dsg_agent_distillation.ollama_export import (
    build_export_settings,
    convert_hf_to_gguf,
    expand_stages,
    merge_dtype_from_gguf_outtype,
    quantize_gguf,
    resolve_converter_python_bin,
    run_export,
    write_settings_modelfile,
)


def data_config(tmp_path):
    return {
        "ollama_export": {
            "model_name": "agent-distilled",
            "export_root": str(tmp_path),
            "system_prompt": "Use the run_cypher_query tool.",
            "stop_tokens": ["<stop>"],
            "quantizations": ["Q4_K_M", "Q5_K_M"],
        }
    }


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
    }


def test_build_export_settings_derives_all_quantized_paths(tmp_path):
    settings = build_export_settings(data_config(tmp_path), student_config(tmp_path))
    export_dir = tmp_path / "agent-distilled"

    assert settings.model_name == "agent-distilled"
    assert settings.base_model == "vendor/base-model"
    assert settings.trust_remote_code is True
    assert settings.selected_quantization == "Q4_K_M"
    assert settings.merged_model_dir == export_dir / "merged-hf"
    assert settings.base_gguf_path == export_dir / "agent-distilled.bf16.gguf"
    assert settings.quantized_gguf_paths["Q4_K_M"] == export_dir / "agent-distilled.Q4_K_M.gguf"
    assert settings.quantized_gguf_paths["Q5_K_M"] == export_dir / "agent-distilled.Q5_K_M.gguf"
    assert settings.modelfile_path == export_dir / "Modelfile"


def test_build_export_settings_separates_models_by_name(tmp_path):
    first = build_export_settings(data_config(tmp_path), student_config(tmp_path))
    second = build_export_settings(
        data_config(tmp_path),
        student_config(tmp_path),
        model_name="another-agent",
    )

    assert first.modelfile_path.parent == tmp_path / "agent-distilled"
    assert second.modelfile_path.parent == tmp_path / "another-agent"
    assert first.selected_gguf_path.parent != second.selected_gguf_path.parent


def test_write_settings_modelfile_writes_metadata(tmp_path):
    settings = build_export_settings(data_config(tmp_path), student_config(tmp_path))
    export_dir = tmp_path / "agent-distilled"

    modelfile_path = write_settings_modelfile(settings)

    assert modelfile_path.read_text(encoding="utf-8") == (
        "FROM ./agent-distilled.Q4_K_M.gguf\n"
        "\n"
        'SYSTEM """Use the run_cypher_query tool."""\n'
        "\n"
        'PARAMETER stop "<stop>"\n'
    )
    metadata = json.loads((export_dir / "Modelfile.json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == "agent-distilled"
    assert metadata["adapter_dir"] == str(tmp_path / "adapter")
    assert metadata["merged_model_dir"] == str(export_dir / "merged-hf")
    assert metadata["selected_quantization"] == "Q4_K_M"
    assert metadata["quantized_gguf_paths"]["Q5_K_M"] == str(
        export_dir / "agent-distilled.Q5_K_M.gguf"
    )


def test_convert_hf_to_gguf_invokes_llama_cpp_converter(tmp_path, monkeypatch):
    settings = build_export_settings(data_config(tmp_path), student_config(tmp_path))
    export_dir = tmp_path / "agent-distilled"
    settings.merged_model_dir.mkdir(parents=True)
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    convert_script.touch()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(ollama_export.subprocess, "run", fake_run)

    output_path = convert_hf_to_gguf(
        settings,
        llama_cpp_dir=llama_cpp_dir,
        python_bin="python",
    )

    assert output_path == export_dir / "agent-distilled.bf16.gguf"
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
    settings = build_export_settings(data_config(tmp_path), student_config(tmp_path))
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

    monkeypatch.setattr(ollama_export.subprocess, "run", fake_run)

    convert_hf_to_gguf(settings, llama_cpp_dir=llama_cpp_dir)

    assert calls[0][0][0] == str(venv_python)


def test_quantize_gguf_invokes_llama_quantize_for_each_quantization(tmp_path, monkeypatch):
    settings = build_export_settings(data_config(tmp_path), student_config(tmp_path))
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
        "Q5_K_M": export_dir / "agent-distilled.Q5_K_M.gguf",
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
                str(export_dir / "agent-distilled.Q5_K_M.gguf"),
                "Q5_K_M",
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
    settings = build_export_settings(data_config(tmp_path), student_config(tmp_path))

    run_export(settings, stages=["modelfile"], dry_run=True)

    assert not (tmp_path / "agent-distilled" / "Modelfile").exists()
    assert not (tmp_path / "agent-distilled" / "Modelfile.json").exists()
