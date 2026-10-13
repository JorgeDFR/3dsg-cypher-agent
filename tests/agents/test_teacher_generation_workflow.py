import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cypher_3dsg_agent_distillation.agents import openrouter_teacher
from cypher_3dsg_agent_distillation.core.config import PROJECT_ROOT, load_yaml
from cypher_3dsg_agent_distillation.core.io import write_jsonl
from cypher_3dsg_agent_distillation.core.versioning import atomic_write_yaml


class Tokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class Validator:
    def __init__(self, **kwargs):
        pass

    def validate(self, cypher, execute=True):
        return {"executable": True, "rows": [{"count": 3}], "error": None}

    def close(self):
        pass


@pytest.mark.parametrize("provider,stop_after_first_request", [
    ("synthetic", False), ("openrouter", False), ("openrouter", True),
])
def test_generation_checkpoints_cost_coverage_and_resumes_without_duplicates(
    tmp_path,
    monkeypatch,
    provider,
    stop_after_first_request,
):
    import cypher_3dsg_agent_distillation.training.train as training

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")

    traces = tmp_path / "traces"
    seed_path = tmp_path / "seeds.jsonl"
    seeds = [
        {
            "uid": f"seed-{i}",
            "scene_id": "scene",
            "question": "How many objects?",
            "query_type": "object_total",
            "task_family": "cypher_qa",
            "output_type": "SLDP",
            "cypher": "MATCH (o:Object) RETURN count(o) AS count",
            "solution": 3,
            "solution_formats": ["int"],
            "metadata": {"question_set": "cypher_qa"},
        }
        for i in range(3)
    ]
    write_jsonl(seeds, seed_path)
    teacher_config = tmp_path / "teacher.yaml"
    atomic_write_yaml(
        {
            "provider": provider,
            "model": "teacher",
            "traces_per_seed": 1,
            "reasoning": {"enabled": True},
        },
        teacher_config,
    )
    atomic_write_yaml({}, tmp_path / "metadata.yaml")
    data = {
        "dataset_dir": str(tmp_path),
        "dataset_id": "test",
        "dataset_version": "2026-10-13-T-12-00-00-Z",
        "metadata_path": str(tmp_path / "metadata.yaml"),
        "seed_records_jsonl_path": str(seed_path),
        "teacher_metadata_path": str(traces / "metadata.yaml"),
        "teacher_traces_jsonl_path": str(traces / "teacher_traces.jsonl"),
        "accepted_traces_path": str(traces / "accepted.json"),
        "rejected_traces_path": str(traces / "rejected.json"),
        "data_config_path": str(PROJECT_ROOT / "configs/data.yaml"),
        "schema_path": str(PROJECT_ROOT / "configs/schema_3dsg.yaml"),
        "scenes": [{"scene_id": "scene", "scene_path": "unused"}],
        "question_sets": [
            {
                "name": "cypher_qa",
                "experiment_profile_path": str(
                    PROJECT_ROOT / "configs/experiment_profiles/cypher_qa_template.yaml"
                ),
                "question_templates_path": str(
                    PROJECT_ROOT / "configs/question_templates/cypher_qa_3dsg.yaml"
                ),
            }
        ],
    }
    monkeypatch.setattr(openrouter_teacher, "load_data_config", lambda *a, **k: dict(data))
    student = load_yaml("configs/students/student_qwen3.5_4b.yaml")
    student["student_config_path"] = str(PROJECT_ROOT / "configs/students/student_qwen3.5_4b.yaml")
    monkeypatch.setattr(openrouter_teacher, "load_student_family_config", lambda *a: student)
    monkeypatch.setattr(training, "load_tokenizer", lambda *a: Tokenizer())
    monkeypatch.setattr(openrouter_teacher, "Neo4jValidator", Validator)
    monkeypatch.setattr(openrouter_teacher, "load_scene", lambda **kwargs: None)
    calls = []

    def send(request):
        calls.append(request)
        after_tool = request["messages"][-1]["role"] == "tool"
        assistant = (
            {"role": "assistant", "reasoning": "Return a number."}
            if after_tool
            else {
                "role": "assistant",
                "reasoning": "Count Object nodes.",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": json.dumps(
                                {
                                    "cypher_string": "MATCH (o:Object) RETURN count(o) AS count",
                                }
                            ),
                        },
                    }
                ],
            }
        )
        if after_tool:
            assistant["content"] = "<answer>3</answer>"
        return {
            "choices": [{"message": assistant, "finish_reason": "stop"}],
            "usage": {"cost": 0.001},
        }

    monkeypatch.setattr(
        openrouter_teacher.OpenRouterTeacherClient,
        "send_chat_request",
        lambda self, request: send(request),
    )
    args = SimpleNamespace(
        teacher_config=str(teacher_config),
        student_config="student",
        data_config="data",
        dataset_version=None,
        teacher=None,
        teacher_model=None,
        traces_per_seed=None,
        static_only=False,
        max_tool_calls=None,
        preflight=False,
        uri=None,
        username="neo4j",
        password="password",
        database="neo4j",
        neo4j_timeout=10,
        pilot=True,
        pilot_per_task_type=2,
        pilot_extra_per_tag=0,
        sample_seed=42,
        max_cost_usd=None,
        resume=False,
    )
    original_reserve = openrouter_teacher.OpenRouterTeacherClient.reserve_request_budget
    if stop_after_first_request:

        def stop(self, request):
            if self.requests:
                raise openrouter_teacher.TeacherBudgetExceeded("test budget limit")

        monkeypatch.setattr(
            openrouter_teacher.OpenRouterTeacherClient, "reserve_request_budget", stop
        )
        openrouter_teacher.generate_command(args)
        assert load_yaml(traces / "metadata.yaml")["status"] == "budget_limited"
        assert len(calls) == 1
        assert load_yaml(traces / "pilot_report.yaml")["coverage"]["attempts"] == 0
        monkeypatch.setattr(
            openrouter_teacher.OpenRouterTeacherClient, "reserve_request_budget", original_reserve
        )
        args.resume = True
        args.dataset_version = data["dataset_version"]
    openrouter_teacher.generate_command(args)
    report = load_yaml(traces / "pilot_report.yaml")
    assert report["coverage"]["accepted"] == 2
    assert report["coverage"]["accepted_with_both_stage_reasoning"] == 2
    assert load_yaml(traces / "metadata.yaml")["status"] == "pilot_complete"
    records = (traces / "teacher_traces.jsonl").read_bytes()
    if provider == "openrouter":
        assert report["billing"]["total_cost_usd"] == pytest.approx(0.004)
        assert len(calls) == 4
    args.dataset_version = data["dataset_version"]
    args.resume = True
    openrouter_teacher.generate_command(args)
    assert (traces / "teacher_traces.jsonl").read_bytes() == records
    assert len((traces / "attempts.jsonl").read_text().splitlines()) == 2
    assert len(calls) == (4 if provider == "openrouter" else 0)
    assert Path(data["metadata_path"]).exists()
