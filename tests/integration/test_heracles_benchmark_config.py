from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT / "examples" / "herables_benchmark"
CAPABILITIES = {
    "reasoning": "optional",
    "reasoning_efforts": None,
    "temperature": True,
    "seed": True,
}


def load_manifest_module():
    module_path = BENCHMARK_DIR / "model_manifest.py"
    spec = importlib.util.spec_from_file_location("benchmark_model_manifest", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_artifact(root: Path, name: str, *, metadata_gguf: str | None = None) -> None:
    artifact_dir = root / name
    artifact_dir.mkdir(parents=True)
    gguf = artifact_dir / f"{name}.Q4_K_M.gguf"
    gguf.touch()
    (artifact_dir / "Modelfile").write_text(
        f"FROM ./{gguf.name}\n",
        encoding="utf-8",
    )
    (artifact_dir / "Modelfile.json").write_text(
        json.dumps({"gguf_path": metadata_gguf or str(gguf)}),
        encoding="utf-8",
    )


def test_manifest_validates_enabled_artifacts_and_container_paths(tmp_path):
    module = load_manifest_module()
    artifacts = tmp_path / "models"
    write_artifact(
        artifacts,
        "base",
        metadata_gguf="/home/user/workspace/project/outputs/ollama/base/base.Q4_K_M.gguf",
    )
    write_artifact(artifacts, "fine-tuned")
    manifest = tmp_path / "students.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "alias": "base",
                        "model": "base",
                        "artifact_dir": "base",
                        "enabled": True,
                        "capabilities": CAPABILITIES,
                    },
                    {
                        "alias": "fine-tuned",
                        "model": "fine-tuned",
                        "artifact_dir": "fine-tuned",
                        "enabled": True,
                        "capabilities": CAPABILITIES,
                    },
                    {
                        "alias": "disabled",
                        "model": "disabled",
                        "artifact_dir": "missing",
                        "enabled": False,
                        "capabilities": CAPABILITIES,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    models = module.load_enabled_models(manifest, artifacts)

    assert [model["model"] for model in models] == ["base", "fine-tuned"]
    assert models[0]["gguf_path"] == str(artifacts / "base" / "base.Q4_K_M.gguf")


def test_manifest_rejects_artifact_directory_traversal(tmp_path):
    module = load_manifest_module()
    manifest = tmp_path / "students.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "alias": "outside",
                        "model": "outside",
                        "artifact_dir": "../outside",
                        "capabilities": CAPABILITIES,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the artifacts root"):
        module.load_enabled_models(manifest, tmp_path / "models")


def test_benchmark_compose_requires_gpu_and_owns_ollama():
    compose = yaml.safe_load((BENCHMARK_DIR / "docker-compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert services["ollama"]["gpus"] == "all"
    assert services["main"]["gpus"] == "all"
    assert not (BENCHMARK_DIR / "docker-compose.gpu.yaml").exists()
    assert compose["x-benchmark-common"]["image"].endswith("3dsg-heracles-benchmark-main:latest}")
    assert services["ollama"]["container_name"] == "3dsg-heracles-benchmark-ollama"
    assert "ports" not in services["ollama"]
    assert "ports" not in services["neo4j"]
    assert services["docker-socket-proxy"]["environment"]["CONTAINERS"] == 1


def test_benchmark_paths_resolve_from_relocated_directory():
    compose = yaml.safe_load((BENCHMARK_DIR / "docker-compose.yaml").read_text(encoding="utf-8"))
    common = compose["x-benchmark-common"]
    build_root = (BENCHMARK_DIR / common["build"]["context"]).resolve()
    volumes = common["volumes"]

    assert build_root == PROJECT_ROOT / "external" / "heracles_benchmark"
    assert common["build"]["dockerfile"] == "docker/benchmark/Dockerfile.benchmark"
    assert not any("HERACLES_BENCHMARK_ROOT" in volume for volume in volumes)
    assert not any("external/heracles" in volume for volume in volumes)
    assert any("BENCHMARK_CONFIG_HOST" in volume for volume in volumes)
    assert any("MODEL_MANIFEST_HOST" in volume for volume in volumes)
    assert any(
        "model_manifest.py:/home/benchmark/workspace/model_manifest.py:ro" in volume
        for volume in volumes
    )
    assert any("../../outputs/ollama:/models:ro" == volume for volume in volumes)
    assert compose["x-benchmark-environment"]["HERACLES_BENCHMARK_PATH"].endswith("/benchmark")


def test_benchmark_image_uses_only_packages_pinned_by_benchmark_submodule():
    dockerfile = (
        PROJECT_ROOT
        / "external"
        / "heracles_benchmark"
        / "docker"
        / "benchmark"
        / "Dockerfile.benchmark"
    ).read_text(encoding="utf-8")

    assert not (BENCHMARK_DIR / "Dockerfile.benchmark").exists()
    assert "COPY external/heracles/heracles" in dockerfile
    assert "COPY external/heracles_agents" in dockerfile
    assert "COPY configs" in dockerfile
    assert "COPY data" in dockerfile
    assert "/home/benchmark/workspace/benchmark_runner.sh" in dockerfile


def test_heracles_benchmark_is_a_declared_submodule():
    gitmodules = (PROJECT_ROOT / ".gitmodules").read_text(encoding="utf-8")

    assert gitmodules.count("[submodule ") == 1
    assert "path = external/heracles_benchmark" in gitmodules
    assert "url = https://github.com/JorgeDFR/heracles-benchmark.git" in gitmodules


def test_main_compose_has_profiled_ollama_for_project_split_validation():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "docker-compose.yaml").read_text(encoding="utf-8")
    )

    ollama = compose["services"]["ollama"]
    assert ollama["profiles"] == ["ollama"]
    assert ollama["gpus"] == "all"
    assert ollama["container_name"] == "3dsg-cypher-agent-ollama"
    assert "ports" not in ollama
    assert "../outputs/ollama:/models:ro" in ollama["volumes"]
    assert "ollama-cache" in compose["volumes"]
    assert compose["x-main-environment"]["OLLAMA_HOST"] == "http://ollama:11434"


def test_main_stack_exposes_explicit_ollama_lifecycle_and_cli_commands():
    launcher = (PROJECT_ROOT / "docker" / "stack.sh").read_text(encoding="utf-8")

    assert "ollama-up" in launcher
    assert "compose --profile ollama up -d --wait ollama" in launcher
    assert 'compose exec -T ollama ollama "$@"' in launcher


def test_main_stack_exposes_demo_train_and_demo_deploy_workflows():
    launcher = (PROJECT_ROOT / "docker" / "stack.sh").read_text(encoding="utf-8")
    train_workflow = launcher.split("run_demo_train()", 1)[1].split(
        "require_trained_student()", 1
    )[0]
    deploy_workflow = launcher.split("run_demo_deploy()", 1)[1].split(
        'if [[ $# -lt 1 ]]', 1
    )[0]

    assert "run_demo_train()" in launcher
    assert "demo-train)" in launcher
    assert 'run_demo_train "$@"' in launcher
    assert "run_demo_deploy()" in launcher
    assert "require_trained_student" in launcher
    assert "scripts/training/deploy_helpers.py check" in launcher
    assert "python -c" not in launcher
    assert "--stages merge convert quantize" in deploy_workflow
    assert '--modelfile-variants "${variants[@]}"' in launcher
    assert "Generate Ollama Modelfiles" in deploy_workflow
    assert "Start Ollama and register GGUF variants" in deploy_workflow
    assert 'total_steps=$((4 + ${#variants[@]}))' in deploy_workflow
    assert 'ollama create "${model_name}-${variant,,}"' in launcher
    assert 'for variant in "${variants[@]}"' in deploy_workflow
    assert "--runtime ollama" in deploy_workflow
    assert launcher.count("--split challenge") >= 3
    assert "demo-deploy)" in launcher
    assert 'run_demo_deploy "$@"' in launcher
    assert "load_3dsg_neo4j.py" not in train_workflow
    assert "load_3dsg_neo4j.py" not in deploy_workflow
    assert "DEMO_SCENE" not in launcher
    assert "DEPLOY_SCENE" not in launcher


def test_materialized_config_keeps_questions_and_overlays_local_models(tmp_path):
    module = load_manifest_module()
    source = tmp_path / "benchmark.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmark": {
                    "scene_id": "scene",
                    "questions": {
                        "qa": "data/questions/scene/qa.yaml",
                        "pddl": "data/questions/scene/pddl.yaml",
                        "metadata": "data/questions/scene/metadata.yaml",
                    },
                },
                "providers": {
                    "ollama": {
                        "enabled": True,
                        "models": [],
                        "local_metrics": {"ollama_container_name": "ollama"},
                    },
                    "openrouter": {"enabled": True, "models": []},
                },
            }
        ),
        encoding="utf-8",
    )
    models = tmp_path / "models.yaml"
    models.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "alias": "student",
                        "model": "student",
                        "artifact_dir": "student",
                        "enabled": True,
                        "capabilities": CAPABILITIES,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = module.materialize_benchmark_config(
        source,
        models,
        tmp_path / "resolved.yaml",
        ollama_container_name="project-ollama",
    )

    assert (
        resolved["benchmark"]["questions"]
        == yaml.safe_load(source.read_text(encoding="utf-8"))["benchmark"]["questions"]
    )
    assert resolved["providers"]["ollama"]["models"][0]["model"] == "student:latest"
    assert "artifact_dir" not in resolved["providers"]["ollama"]["models"][0]
    assert resolved["providers"]["ollama"]["local_metrics"]["ollama_container_name"] == (
        "project-ollama"
    )
    assert resolved["providers"]["openrouter"]["enabled"] is False


def test_student_manifest_uses_stable_base_and_fine_tuned_directories():
    manifest = yaml.safe_load(
        (BENCHMARK_DIR / "config" / "model_lists" / "students.yaml").read_text(encoding="utf-8")
    )
    qwen = manifest["models"][:2]

    assert qwen[0]["model"] == "qwen3.5-4b-baseline"
    assert qwen[0]["artifact_dir"] == "qwen3.5-4b-baseline"
    assert qwen[0]["variant"] == "baseline"
    assert qwen[1]["model"] == "qwen3.5-4b-finetune"
    assert qwen[1]["artifact_dir"] == "qwen3.5-4b-finetune"
    assert qwen[1]["variant"] == "finetune"
    assert all(entry["enabled"] is True for entry in qwen)
    assert all("q4km" not in entry["artifact_dir"].lower() for entry in qwen)
    assert all(set(entry["capabilities"]) == set(CAPABILITIES) for entry in qwen)


def test_launcher_uses_upstream_question_config_and_creates_local_models():
    launcher = (BENCHMARK_DIR / "run_benchmark.sh").read_text(encoding="utf-8")

    assert 'HERACLES_BENCHMARK_ROOT="${PROJECT_ROOT}/external/heracles_benchmark"' in launcher
    assert "--config PATH" in launcher
    assert "--models PATH" in launcher
    assert "model_manifest.py" in launcher
    assert "ollama create" in launcher
    assert "mapfile -t provisioning_records" in launcher
    assert 'ollama create "${model}" -f "${modelfile}" </dev/null' in launcher
    assert "Provisioned and verified" in launcher
    assert "/home/benchmark/workspace/benchmark_runner.sh" in launcher
    assert "check_required_gpu" in launcher
    assert "--gpus all" in launcher
    assert "NVIDIA Container Toolkit" in launcher
    assert "CPU" not in launcher


def test_integration_does_not_keep_a_forked_benchmark_runner_or_experiments():
    assert not (BENCHMARK_DIR / "benchmark_runner.sh").exists()

    assert not (BENCHMARK_DIR / "config" / "cypher_qa_student_model_sweep.yaml").exists()
    assert not (BENCHMARK_DIR / "config" / "cypher_pddl_student_model_sweep.yaml").exists()
