from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT / "examples" / "herables_benchmark"


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
                    },
                    {
                        "alias": "fine-tuned",
                        "model": "fine-tuned",
                        "artifact_dir": "fine-tuned",
                        "enabled": True,
                    },
                    {
                        "alias": "disabled",
                        "model": "disabled",
                        "artifact_dir": "missing",
                        "enabled": False,
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
    assert compose["x-benchmark-common"]["image"].endswith(
        "3dsg-heracles-benchmark-main:latest}"
    )
    assert services["ollama"]["container_name"] == "3dsg-heracles-benchmark-ollama"
    assert "ports" not in services["ollama"]
    assert "ports" not in services["neo4j"]
    assert services["docker-socket-proxy"]["environment"]["CONTAINERS"] == 1


def test_benchmark_paths_resolve_from_relocated_directory():
    compose = yaml.safe_load((BENCHMARK_DIR / "docker-compose.yaml").read_text(encoding="utf-8"))
    common = compose["x-benchmark-common"]
    build_root = (BENCHMARK_DIR / common["build"]["context"]).resolve()
    volume_sources = [
        (BENCHMARK_DIR / volume.split(":", 1)[0]).resolve() for volume in common["volumes"]
    ]

    assert build_root == PROJECT_ROOT
    assert common["build"]["dockerfile"] == ("examples/herables_benchmark/Dockerfile.benchmark")
    assert PROJECT_ROOT / "outputs" / "heracles_benchmark" in volume_sources
    assert PROJECT_ROOT / "outputs" / "ollama" in volume_sources
    assert not (PROJECT_ROOT / "examples" / "heracles_agents").exists()


def test_main_compose_no_longer_contains_ollama():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker" / "docker-compose.yaml").read_text(encoding="utf-8")
    )

    assert "ollama" not in compose["services"]
    assert "ollama-cache" not in compose["volumes"]
    assert "OLLAMA_HOST" not in compose["x-main-environment"]


@pytest.mark.parametrize(
    ("filename", "task", "questions"),
    [
        ("cypher_qa_student_model_sweep.yaml", "cypher", "qa_questions.yaml"),
        ("cypher_pddl_student_model_sweep.yaml", "pddl", "pddl_questions.yaml"),
    ],
)
def test_experiment_configs_share_models_and_collect_local_metrics(filename, task, questions):
    experiment = yaml.safe_load((BENCHMARK_DIR / "config" / filename).read_text(encoding="utf-8"))
    sweep = next(iter(experiment["model_sweeps"].values()))

    assert experiment["metadata"]["task"] == task
    assert experiment["metadata"]["local_metrics"]["enabled"] is True
    assert (
        experiment["metadata"]["local_metrics"]["ollama_container_name"]
        == "3dsg-heracles-benchmark-ollama"
    )
    assert sweep["provider"] == "ollama"
    assert sweep["models_path"].endswith("model_lists/students.yaml")
    assert sweep["template"]["questions"].endswith(questions)


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


def test_launcher_defaults_to_both_benchmarks_and_creates_models_in_benchmark_ollama():
    launcher = (BENCHMARK_DIR / "run_benchmark.sh").read_text(encoding="utf-8")

    assert "--benchmarks cypher-qa" in launcher
    assert "--benchmarks cypher-pddl" in launcher
    assert "--benchmarks both" in launcher
    assert "RUN_CYPHER_QA=1\nRUN_CYPHER_PDDL=1" in launcher
    assert "select_benchmarks" not in launcher
    assert "compose exec -T ollama ollama create" in launcher
    assert "mapfile -t provisioning_records" in launcher
    assert 'ollama create "${model}" -f "${modelfile}" </dev/null' in launcher
    assert "Provisioned and verified" in launcher
    assert 'docker image inspect "${BENCHMARK_IMAGE}"' in launcher
    assert "Reusing existing benchmark image" in launcher
    assert "force_build_benchmark_image" in launcher
    assert "CPU mode" not in launcher


def test_report_uses_only_results_written_by_current_run():
    runner = (BENCHMARK_DIR / "benchmark_runner.sh").read_text(encoding="utf-8")

    assert 'RUN_CYPHER_QA="${1:-1}"' in runner
    assert 'RUN_CYPHER_PDDL="${2:-1}"' in runner
    assert 'RUN_MARKER="$(mktemp)"' in runner
    assert '-newer "${RUN_MARKER}"' in runner
