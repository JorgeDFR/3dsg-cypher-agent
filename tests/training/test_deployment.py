import pytest
import yaml

from cypher_3dsg_agent_distillation.training.deployment import (
    deployment_config,
    require_trained_student,
)


def test_deployment_rejects_missing_model_name(tmp_path):
    config = tmp_path / "student.yaml"
    config.write_text(yaml.safe_dump({
        "model": {"base_model": "vendor/model"},
        "training": {"output_dir": str(tmp_path / "models")},
    }))
    with pytest.raises(ValueError, match="ollama_export.model_name"):
        deployment_config(str(config))


def test_preflight_selects_latest_complete_model_and_requires_weights(tmp_path):
    root = tmp_path / "models"
    config = tmp_path / "student.yaml"
    config.write_text(yaml.safe_dump({"training": {"output_dir": str(root)}}))
    for version, status in [("2026-09-30-T-12-12-17-Z", "complete"),
                            ("2026-10-01-T-12-12-17-Z", "running")]:
        directory = root / version
        directory.mkdir(parents=True)
        (directory / "metadata.yaml").write_text(yaml.safe_dump({"status": status}))
    completed = root / "2026-09-30-T-12-12-17-Z"
    (completed / "adapter_config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Trained student model not found"):
        require_trained_student(str(config))
    (completed / "adapter_model.safetensors").touch()
    assert require_trained_student(str(config)) == completed
