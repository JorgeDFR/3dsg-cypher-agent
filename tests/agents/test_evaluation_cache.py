import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cypher_3dsg_agent_distillation.agents import student_evaluation
from cypher_3dsg_agent_distillation.agents.evaluation_cache import (
    completed_evaluation,
    evaluation_identity,
)


@pytest.fixture
def evaluation(tmp_path):
    split = tmp_path / "test.jsonl"
    split.write_text("{}\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    data = {
        "dataset_dir": str(tmp_path),
        "dataset_id": "demo",
        "question_sets": [
            {
                "name": "qa",
                "question_templates_path": "configs/question_templates/cypher_qa_3dsg.yaml",
                "experiment_profile_path": "configs/experiment_profiles/cypher_qa_template.yaml",
            }
        ],
        "dataset_version": "one",
        "splits": {"test": str(split)},
    }
    student = {
        "model": {"base_model": "base"},
        "model_family": "student",
        "model_version": "one",
        "model_family_root": str(tmp_path),
        "training": {"output_dir": str(adapter)},
    }
    args = argparse.Namespace(
        split="test",
        mode="raw-generation",
        baseline=True,
        tokenizer=None,
        max_new_tokens=1024,
        no_progress=False,
        force=False,
        output=None,
    )
    return args, data, student, str(adapter)


def test_baseline_reused_after_new_training(evaluation):
    args, data, student, adapter = evaluation
    initial = evaluation_identity(*evaluation)
    student["model_version"] = "new-training"
    student["training"].update({"learning_rate": 0.1, "max_length": 1024})
    assert evaluation_identity(args, data, student, adapter) == initial
    args.no_progress = True
    args.force = True
    args.output = "/other/report.json"
    assert evaluation_identity(args, data, student, adapter) == initial


@pytest.mark.parametrize("change", ["dataset", "split", "model", "tokens", "runtime", "weights"])
def test_changed_inputs_invalidate_evaluation(evaluation, change):
    args, data, student, adapter = evaluation
    args.baseline = False
    initial = evaluation_identity(*evaluation)
    runtime = None
    if change == "dataset":
        data["dataset_version"] = "two"
    elif change == "split":
        Path(data["splits"]["test"]).write_text('{"changed": true}\n')
    elif change == "model":
        student["model_version"] = "two"
    elif change == "tokens":
        args.max_new_tokens = 512
    elif change == "runtime":
        runtime = {"digest": "new-digest"}
    else:
        (Path(adapter) / "adapter_model.safetensors").write_bytes(b"changed")
    assert evaluation_identity(*evaluation, runtime_identity=runtime) != initial


@pytest.mark.parametrize(
    "content",
    ["{", "[]", "{}", '{"metadata": null}', '{"metadata": {"evaluation_status": "running"}}'],
)
def test_incomplete_or_corrupt_reports_are_not_reused(tmp_path, content):
    path = tmp_path / "report.json"
    path.write_text(content)
    assert not completed_evaluation(path, "identity")


def test_command_skips_before_loading_baseline_and_force_bypasses_cache(
    evaluation,
    monkeypatch,
    tmp_path,
):
    _, data, student, adapter = evaluation
    captured = []
    with monkeypatch.context() as parser_patch:
        parser_patch.setattr(
            sys,
            "argv",
            [
                "evaluate_student",
                "--baseline",
                "--mode",
                "raw-generation",
                "--split",
                "test",
            ],
        )
        parser_patch.setattr(student_evaluation, "command", captured.append)
        student_evaluation.main()
    args = captured[0]
    monkeypatch.setattr(student_evaluation, "load_data_config", lambda *a, **k: data)
    monkeypatch.setattr(student_evaluation, "load_student_family_config", lambda *a: student)
    report = (
        tmp_path / "reports" / "student" / "baseline" / "test.baseline.raw-generation.report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "summary": {},
                "metadata": {
                    "evaluation_status": "complete",
                    "evaluation_identity": evaluation_identity(args, data, student, adapter),
                },
            }
        )
    )

    def model_load(*args, **kwargs):
        raise RuntimeError("model load reached")

    monkeypatch.setattr(student_evaluation, "build_raw_generation_runtime", model_load)
    before = report.read_bytes()
    student_evaluation.command(args)
    assert report.read_bytes() == before
    forced = deepcopy(args)
    forced.force = True
    with pytest.raises(RuntimeError, match="model load reached"):
        student_evaluation.command(forced)


def test_validation_config_changes_invalidate_cache_but_test_fixtures_do_not(evaluation, tmp_path):
    args, data, student, adapter = evaluation
    template = tmp_path / "questions.yaml"
    template.write_text("templates: []\n")
    sidecar = tmp_path / "validation"
    sidecar.mkdir()
    catalog = sidecar / template.name
    catalog.write_text("version: 1\ntemplates: {}\n")
    defaults = sidecar / "defaults.yaml"
    defaults.write_text("version: 1\ndefaults: {}\n")
    data["question_sets"] = [{"question_templates_path": str(template)}]
    initial = evaluation_identity(*evaluation)
    defaults.write_text('version: 1\ndefaults: {answer: {empty: "<>"}}\n')
    changed = evaluation_identity(*evaluation)
    assert changed != initial
    catalog.write_text("version: 1\ntemplates: {changed: {}}\n")
    changed_catalog = evaluation_identity(*evaluation)
    assert changed_catalog != changed
    fixtures = sidecar / "fixtures"
    fixtures.mkdir()
    (fixtures / "cases.yaml").write_text("version: 1\ncases: {}\n")
    assert evaluation_identity(*evaluation) == changed_catalog
