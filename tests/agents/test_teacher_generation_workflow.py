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
        return {
            "executable": True,
            "rows": [{"class": "chair"}] if "o.class AS class" in cypher else [{"count": 3}],
            "error": None,
        }

    def close(self):
        pass


@pytest.mark.parametrize(
    "provider,stop_after_first_request,answer_mode,repair_enabled,crash_at,recover_enabled",
    [
        ("synthetic", False, "valid", False, None, False),
        ("openrouter", False, "valid", True, None, False),
        ("openrouter", True, "valid", True, None, False),
        ("openrouter", False, "invalid_format", True, None, False),
        ("openrouter", False, "invalid_format", False, None, False),
        ("openrouter", False, "mismatch", True, None, False),
        ("openrouter", True, "invalid_format", True, None, False),
        ("openrouter", False, "invalid_format", True, "before_commit", False),
        ("openrouter", False, "invalid_format", True, "after_commit", False),
        ("openrouter", False, "diagnostic", False, None, True),
        ("openrouter", False, "diagnostic", True, None, False),
        ("openrouter", False, "diagnostic", True, "before_commit", True),
        ("openrouter", False, "diagnostic", True, "after_commit", True),
    ],
)
def test_generation_checkpoints_cost_coverage_and_resumes_without_duplicates(
    tmp_path,
    monkeypatch,
    provider,
    stop_after_first_request,
    answer_mode,
    repair_enabled,
    crash_at,
    recover_enabled,
    capsys,
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
            "validation_contract": {
                "result": {"kind": "scalar", "type": "integer"},
                "answer": {"formats": ["int"], "empty": "0"},
            },
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
            "routing": {"require_parameters": True, "max_price": {"prompt": 0.5}},
            "optimizations": {
                "repair_invalid_final_answers": repair_enabled,
                "recover_earlier_retrievals": recover_enabled,
                "tuning_prompt": "Use one complete retrieval." if provider == "openrouter" else "",
            },
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
    total_requests = 6 if answer_mode == "diagnostic" else 4

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
        tool_count = sum(m["role"] == "tool" for m in request["messages"])
        if after_tool and answer_mode == "diagnostic" and tool_count == 1:
            assistant = {
                "role": "assistant",
                "reasoning": "Inspect Object classes.",
                "tool_calls": [
                    {
                        "id": "diagnostic",
                        "type": "function",
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": json.dumps(
                                {
                                    "cypher_string": (
                                    "MATCH (o:Object) RETURN DISTINCT o.class AS class"
                                ),
                                }
                            ),
                        },
                    }
                ],
            }
        elif after_tool:
            answer = {
                "valid": "3",
                "diagnostic": "3",
                "invalid_format": "{3}",
                "mismatch": "999",
            }[answer_mode]
            assistant["content"] = f"<answer>{answer}</answer>"
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
    if crash_at:
        replace = openrouter_teacher.TeacherRunJournal.replace_attempts

        def crash(self, attempts):
            if crash_at == "after_commit":
                replace(self, attempts)
            raise RuntimeError("test optimization interruption")

        monkeypatch.setattr(openrouter_teacher.TeacherRunJournal, "replace_attempts", crash)
        with pytest.raises(RuntimeError, match="test optimization interruption"):
            openrouter_teacher.generate_command(args)
        assert len(calls) == total_requests
        monkeypatch.setattr(openrouter_teacher.TeacherRunJournal, "replace_attempts", replace)
        args.resume = True
        args.dataset_version = data["dataset_version"]
    openrouter_teacher.generate_command(args)
    report = load_yaml(traces / "pilot_report.yaml")
    repaired = 2 if answer_mode == "invalid_format" and repair_enabled else 0
    recovered = 2 if answer_mode == "diagnostic" and recover_enabled else 0
    accepted = 2 if answer_mode == "valid" or repaired or recovered else 0
    assert report["coverage"]["accepted"] == accepted
    assert report["coverage"]["rejected"] == 2 - accepted
    assert report["optimizations"]["repaired"] == repaired
    assert report["optimizations"]["recovered"] == recovered
    assert report["optimizations"]["accepted_before"] == accepted - repaired - recovered
    assert report["optimizations"]["rejected_after"] == 2 - accepted
    expected_both_stages = 0 if provider == "synthetic" else accepted
    assert report["coverage"]["accepted_with_both_stage_reasoning"] == expected_both_stages
    metadata = load_yaml(traces / "metadata.yaml")
    assert metadata["status"] == "pilot_complete"
    assert metadata["effective"]["tool_feedback_policy"] == "execution_only_v1"
    assert metadata["counts"]["accepted"] == accepted
    assert metadata["counts"]["rejected"] == 2 - accepted
    assert metadata["counts"]["repaired"] == repaired
    assert metadata["counts"]["recovered"] == recovered
    assert metadata["optimizations"] == report["optimizations"]
    saved_config = metadata["effective"]["teacher_config"]
    assert saved_config["optimizations"]["repair_invalid_final_answers"] is repair_enabled
    assert saved_config["optimizations"]["recover_earlier_retrievals"] is recover_enabled
    records = (traces / "teacher_traces.jsonl").read_bytes()
    assert b"Use one complete retrieval." not in records
    assert "Use one complete retrieval." not in (traces / "accepted.json").read_text()
    assert "Use one complete retrieval." not in (traces / "attempts.jsonl").read_text()
    accepted_debug = json.loads((traces / "accepted.json").read_text())
    rejected_debug = json.loads((traces / "rejected.json").read_text())
    assert len(accepted_debug) == accepted
    assert len(rejected_debug) == 2 - accepted
    journal = list(map(json.loads, (traces / "attempts.jsonl").read_text().splitlines()))
    assert sum(item["accepted"] for item in journal) == accepted
    if repaired:
        assert "teacher optimizations repaired=2" in capsys.readouterr().out
        for item in journal:
            assert item["optimization"]["raw_rejection"]["final_answer"] == "{3}"
            assert item["record"]["messages"][-1]["reasoning"] == "Return a number."
            assert item["record"]["teacher_model"] == "teacher"
            assert item["record"]["validation"]["answer_grounded"] is True
            assert item["debug_record"]["final_answer"] == "3"
            assert item["debug_record"]["details"]["optimization"]["original_reason"] == (
                "invalid_answer_format"
            )
            assert item["token_lengths"]["with_reasoning"] > 0
            assert item["record"]["uid"] in report["matched_retained_uids"] or (
                not item["matched_trainable"]
            )
    if recovered:
        from cypher_3dsg_agent_distillation.agents.agent_evaluation import (
            recompute_trace_validation,
        )
        from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord

        assert "recovered=2" in capsys.readouterr().out
        assert report["optimizations"]["by_kind"] == {"recover_earlier_retrievals": 2}
        for item in journal:
            raw = item["optimization"]["raw_rejection"]
            record = AgentTraceRecord.model_validate(item["record"])
            assert record.model_dump(exclude_none=True)["messages"] == raw["messages"]
            assert record.metadata["grounding_tool_call_index"] == 1
            assert record.metadata["diagnostic_tool_call_indices"] == [2]
            assert recompute_trace_validation(record)["answer_grounded"] is True
            assert item["debug_record"]["validation"]["rows"] == [{"count": 3}]
            from cypher_3dsg_agent_distillation.cypher.validation_audit import saved_result

            assert saved_result(item["debug_record"]) == (
                record.gold_cypher, [{"count": 3}], None,
            )
    if provider == "synthetic":
        for record in map(json.loads, records.decode().splitlines()):
            tool_turn = next(message for message in record["messages"] if message.get("tool_calls"))
            assert not tool_turn.get("reasoning")
            assert tool_turn["reasoning_selection"]["selected_rules"] == []
            assert record["messages"][-1]["reasoning"] == "SLDP integer."
    if provider == "openrouter":
        for call in calls:
            assert "provider" not in call
            user = next(m for m in reversed(call["messages"]) if m["role"] == "user")
            assert user["content"].endswith("Use one complete retrieval.")
        assert report["billing"]["total_cost_usd"] == pytest.approx(total_requests * 0.001)
        assert len(calls) == total_requests
    args.dataset_version = data["dataset_version"]
    args.resume = True
    openrouter_teacher.generate_command(args)
    assert (traces / "teacher_traces.jsonl").read_bytes() == records
    assert len((traces / "attempts.jsonl").read_text().splitlines()) == 2
    assert len(calls) == (total_requests if provider == "openrouter" else 0)
    assert load_yaml(traces / "pilot_report.yaml")["optimizations"]["repaired"] == repaired
    assert load_yaml(traces / "pilot_report.yaml")["optimizations"]["recovered"] == recovered
    assert list(map(json.loads, (traces / "attempts.jsonl").read_text().splitlines())) == journal
    assert Path(data["metadata_path"]).exists()
    if provider == "openrouter" and answer_mode == "valid" and not stop_after_first_request:
        old_metadata = load_yaml(traces / "metadata.yaml")
        old_metadata["effective"].pop("tool_feedback_policy")
        atomic_write_yaml(old_metadata, traces / "metadata.yaml")
        with pytest.raises(ValueError, match="different tool-feedback policy"):
            openrouter_teacher.generate_command(args)
        assert len(calls) == total_requests
