import argparse
import json
from copy import deepcopy

import pytest

from cypher_3dsg_agent_distillation.agents import teacher_check
from cypher_3dsg_agent_distillation.agents.openrouter_teacher import OpenRouterTeacherClient
from cypher_3dsg_agent_distillation.core.config import PROJECT_ROOT
from cypher_3dsg_agent_distillation.core.io import write_jsonl
from cypher_3dsg_agent_distillation.core.versioning import atomic_write_yaml


def response(*, tool=False, reasoning=True, cost=0.001, finish=None):
    assistant = {"role": "assistant"}
    if reasoning:
        assistant["reasoning"] = "Count Object nodes." if tool else "Return a number."
    if tool:
        assistant["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "run_cypher_query",
                    "arguments": json.dumps(
                        {"cypher_string": "MATCH (o:Object) RETURN count(o) AS count"}
                    ),
                },
            }
        ]
    else:
        assistant["content"] = "<answer>3</answer>"
    return {
        "id": "gen-test",
        "model": "teacher",
        "provider": "test-provider",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 30},
            "cost": cost,
        },
        "choices": [
            {
                "finish_reason": finish or ("tool_calls" if tool else "stop"),
                "native_finish_reason": "provider-stop",
                "message": assistant,
            }
        ],
        "provider_extension": {"latency": 123},
    }


def setup_check(tmp_path, monkeypatch, *, cost=0.001):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    config = tmp_path / "teacher.yaml"
    atomic_write_yaml(
        {
            "provider": "openrouter",
            "model": "teacher",
            # Legacy routing settings must not make their way into requests.
            "routing": {"require_parameters": True, "max_price": {"prompt": 0.5}},
        },
        config,
    )
    parser = argparse.ArgumentParser()
    teacher_check.add_check_parser(parser.add_subparsers(required=True))
    args = parser.parse_args(
        [
            "check",
            "--teacher-config",
            str(config),
            "--output",
            str(tmp_path / "check.json"),
        ]
    )

    def preflight(self):
        self.model_info = {
            "id": "teacher",
            "supported_parameters": ["reasoning", "tools"],
            "reasoning": {"default_effort": "medium"},
            "pricing": {"prompt": "0.0000005", "completion": "0.0000015"},
        }

    calls = []

    def send(self, request):
        calls.append(deepcopy(request))
        return response(tool=len(calls) == 1, cost=cost)

    monkeypatch.setattr(OpenRouterTeacherClient, "preflight", preflight)
    monkeypatch.setattr(OpenRouterTeacherClient, "send_chat_request", send)
    setup_dataset(tmp_path, monkeypatch)
    return args, calls


def test_check_uses_dataset_and_preserves_response_metrics(tmp_path, monkeypatch):
    args, calls = setup_check(tmp_path, monkeypatch)
    teacher_check.check_command(args)
    report = json.loads((tmp_path / "check.json").read_text())
    assert len(calls) == 2
    assert calls[0]["reasoning"] == {"enabled": True, "exclude": False}
    assert "effort" not in calls[0]["reasoning"]
    assert report["capability"]["readable_reasoning_status"] == "observed"
    assert report["metrics"]["billed_cost_usd"] == 0.002
    assert report["metrics"]["completion_tokens"] == 100  # Includes 60 reasoning tokens.
    assert report["requests"][0]["raw_response"]["provider_extension"] == {"latency": 123}
    assert report["projection"]["full_dataset_cost_usd"] == pytest.approx(0.006)
    assert "routing" not in report["effective"]
    assert all("provider" not in call for call in calls)
    assert report["example"]["question"] == "How many objects?"
    assert report["example"]["expected_answer"] == 3
    assert report["capability"]["reasoning_stages"]["tool_call"]["formats"] == ["text"]
    assert report["capability"]["reasoning_stages"]["final_answer"]["formats"] == ["text"]


class Validator:
    closed = False

    def __init__(self, **kwargs):
        pass

    def validate(self, cypher, execute=True):
        return {"executable": True, "rows": [{"count": 3}], "error": None}

    def close(self):
        Validator.closed = True


def setup_dataset(tmp_path, monkeypatch):
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
    write_jsonl(seeds, tmp_path / "seeds.jsonl")
    data = {
        "dataset_id": "check-test",
        "seed_records_jsonl_path": str(tmp_path / "seeds.jsonl"),
        "schema_path": str(PROJECT_ROOT / "configs/schema_3dsg.yaml"),
        "scenes": [{"scene_id": "scene", "scene_path": "unused"}],
        "question_sets": [
            {
                "name": "cypher_qa",
                "question_templates_path": str(
                    PROJECT_ROOT / "configs/question_templates/cypher_qa_3dsg.yaml"
                ),
                "experiment_profile_path": str(
                    PROJECT_ROOT / "configs/experiment_profiles/cypher_qa_template.yaml"
                ),
            }
        ],
    }
    monkeypatch.setattr(teacher_check, "load_data_config", lambda *a, **k: data)
    monkeypatch.setattr(teacher_check, "load_scene", lambda **k: None)
    monkeypatch.setattr(teacher_check, "Neo4jValidator", Validator)


def test_dataset_check_runs_one_conversation_and_projects_all_seed_attempts(tmp_path, monkeypatch):
    args, calls = setup_check(tmp_path, monkeypatch)
    setup_dataset(tmp_path, monkeypatch)
    args.traces_per_seed = 2
    teacher_check.check_command(args)
    report = json.loads((tmp_path / "check.json").read_text())
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "tool"
    assert report["outcome"]["accepted"]
    assert report["capability"]["both_stage_reasoning_and_accepted"]
    assert report["metrics"]["billed_cost_usd"] == 0.002
    assert report["projection"]["target_traces"] == 6
    assert report["projection"]["full_dataset_cost_usd"] == pytest.approx(0.012)
    assert report["projection"]["sequential_api_seconds"] > 0
    assert Validator.closed
    assert not (tmp_path / "teacher_traces.jsonl").exists()


@pytest.mark.parametrize(
    "cost,budget,max_requests",
    [
        (None, 0.1, 2),
        (0.001, 0.003, 2),
        (0.001, 0.1, 1),
    ],
)
def test_interrupted_check_keeps_metrics_and_never_projects_partial_trace(
    tmp_path,
    monkeypatch,
    cost,
    budget,
    max_requests,
):
    args, calls = setup_check(tmp_path, monkeypatch, cost=cost)
    setup_dataset(tmp_path, monkeypatch)
    args.max_cost_usd, args.max_requests = budget, max_requests

    # Give the request a predictable conservative reservation independent of profile size.
    def reserve(self, request):
        if self.requests:
            if cost is None:
                raise teacher_check.TeacherBudgetExceeded("Previous cost is unknown.")
            if budget < 0.01:
                raise teacher_check.TeacherBudgetExceeded("Next reservation exceeds budget.")

    monkeypatch.setattr(OpenRouterTeacherClient, "reserve_request_budget", reserve)
    teacher_check.check_command(args)
    report = json.loads((tmp_path / "check.json").read_text())
    assert len(calls) == 1
    assert report["outcome"]["status"] == "incomplete"
    assert report["projection"]["full_dataset_cost_usd"] is None
    assert report["metrics"]["requests"] == 1
    assert report["requests"][0]["raw_response"]["choices"][0]["message"]["tool_calls"]


def report_for(*, finish="stop", cost=0.001, readable=True, status="completed"):
    raw = response(reasoning=readable, cost=cost, finish=finish)
    return teacher_check.check_report(
        model_info={"pricing": {"prompt": "0.0000005", "completion": "0.0000015"}},
        requests=[
            {
                "usage": raw["usage"],
                "cost_usd": cost,
                "latency_seconds": 2.0,
                "finish_reason": finish,
            }
        ],
        responses=[raw],
        messages=[raw["choices"][0]["message"]],
        wall_seconds=3.0,
        outcome={"status": status, "accepted": True},
        target_traces=100,
    )


def test_projections_use_billing_and_timing_without_double_counting_reasoning():
    report = report_for()
    assert report["metrics"]["completion_tokens_per_second"] == 25
    assert report["projection"]["full_dataset_cost_usd"] == pytest.approx(0.1)
    assert report["projection"]["sequential_api_seconds"] == 200
    assert report["projection"]["sequential_conversation_seconds"] == 300
    assert report["metrics"]["reasoning_tokens"] == 30


def test_reasoning_token_count_is_not_evidence_of_readable_text():
    report = report_for(readable=False)
    assert report["metrics"]["reasoning_tokens"] == 30
    assert report["capability"]["readable_reasoning_status"] == "not_observed"


def test_truncated_reasoning_is_inconclusive_and_does_not_project_dataset_cost():
    report = report_for(finish="length", readable=False)
    assert report["capability"]["readable_reasoning_status"] == "inconclusive_truncated"
    assert report["projection"]["full_dataset_cost_usd"] is None


def test_missing_billing_is_explicit_and_estimate_uses_catalog_prices():
    report = report_for(cost=None)
    assert report["metrics"]["billed_cost_usd"] is None
    assert report["metrics"]["unknown_cost_requests"] == 1
    assert report["projection"]["cost_basis"] == "catalog_estimate"
    assert report["metrics"]["catalog_estimated_cost_usd"] == pytest.approx(0.000125)
    assert report["projection"]["full_dataset_cost_usd"] == pytest.approx(0.0125)


def test_check_does_not_overwrite_report_before_calling_api(tmp_path, monkeypatch):
    args, calls = setup_check(tmp_path, monkeypatch)
    (tmp_path / "check.json").write_text("previous report")
    with pytest.raises(FileExistsError):
        teacher_check.check_command(args)
    assert not calls


def test_default_effort_override_is_omitted_on_wire(tmp_path, monkeypatch):
    args, calls = setup_check(tmp_path, monkeypatch)
    atomic_write_yaml(
        {
            "provider": "openrouter",
            "model": "teacher",
            "reasoning": {"enabled": True, "effort": "low"},
        },
        args.teacher_config,
    )
    args.reasoning_effort = "default"
    teacher_check.check_command(args)
    assert "effort" not in calls[0]["reasoning"]


def test_disabled_check_filters_dataset_messages_but_keeps_raw_response(
    tmp_path,
    monkeypatch,
):
    args, calls = setup_check(tmp_path, monkeypatch)
    args.include_reasoning = False
    teacher_check.check_command(args)
    report = json.loads((tmp_path / "check.json").read_text())
    assert calls[0]["reasoning"] == {"exclude": True}
    assert "reasoning" not in report["messages"][-1]
    assert report["requests"][0]["raw_response"]["choices"][0]["message"]["reasoning"]
    assert report["capability"]["reasoning_stages"]["tool_call"]["readable_turns"] == 1
    assert report["capability"]["reasoning_stages"]["tool_call"]["saved_readable_turns"] == 0


@pytest.mark.parametrize(
    "removed_option", [["--mode", "prompt"], ["--mode", "dataset"], ["--prompt", "custom prompt"]]
)
def test_check_cli_has_no_prompt_or_mode_option(removed_option):
    parser = argparse.ArgumentParser()
    teacher_check.add_check_parser(parser.add_subparsers(required=True))
    with pytest.raises(SystemExit):
        parser.parse_args(["check", *removed_option])
    args = parser.parse_args(["check"])
    assert not hasattr(args, "mode")
    assert args.max_requests == 2
    assert args.max_tokens == 2048
