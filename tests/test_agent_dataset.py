import json
from collections import Counter

import pytest
from agent_helpers import answerable_trace

from cypher_3dsg_agent_distillation.agent_dataset import (
    assert_no_split_leakage,
    grouped_split_records,
    optimize_student_trace,
    optimize_student_traces,
    optimize_student_traces_with_counts,
    read_student_trace_records,
    read_student_trace_records_with_counts,
    repair_rejected_invalid_answer_trace,
    split_report_lines,
    split_report_payload,
)
from cypher_3dsg_agent_distillation.agent_records import AgentTraceRecord
from cypher_3dsg_agent_distillation.io import write_jsonl


def make_record(uid: str, task_type: str, cypher: str) -> AgentTraceRecord:
    return AgentTraceRecord.model_validate(
        answerable_trace(
            uid=uid,
            task_type=task_type,
            gold_cypher=cypher,
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {"cypher_string": cypher},
                            }
                        }
                    ],
                },
                {"role": "tool", "tool_name": "run_cypher_query", "content": "[]"},
                {"role": "assistant", "content": "<answer>3</answer>"},
            ],
        )
    )


def rejected_invalid_answer_trace(*, two_tool_calls: bool = False, **overrides):
    cypher = "MATCH (o:Object) RETURN o.nodeSymbol AS object"
    messages = [{"role": "user", "content": "question"}]
    if two_tool_calls:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {"cypher_string": "MATCH broken"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "tool_name": "run_cypher_query",
                    "content": '{"ok": false}',
                },
            ]
        )
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": cypher},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "tool_name": "run_cypher_query",
                "content": '{"ok": true, "rows": [{"object": "O1"}]}',
            },
            {"role": "assistant", "content": "<answer>{O1}</answer>"},
        ]
    )
    payload = {
        "uid": "rejected_teacher",
        "seed_uid": "seed_rejected",
        "scene_id": "scene",
        "query_type": "objects_in_place_symbol",
        "question": "Which object is in place p1?",
        "cypher": cypher,
        "real_solution": "O1",
        "cypher_solution": "O1",
        "final_answer": "{O1}",
        "validation": {
            "executable": True,
            "solution_match": False,
            "cypher_solution_match": True,
            "final_answer_match": None,
            "final_answer_cypher_match": False,
        },
        "messages": messages,
        "reason": "invalid_answer_format",
    }
    payload.update(overrides)
    return payload


def test_assert_no_split_leakage_rejects_repeated_cypher():
    record = make_record("one", "count", "MATCH (n) RETURN count(n) AS count")
    leaked = make_record("two", "count", "MATCH (n) RETURN count(n) AS count")

    with pytest.raises(ValueError, match="Split leakage"):
        assert_no_split_leakage({"train": [record], "test": [leaked]})


def test_grouped_split_keeps_challenge_task_out_of_main_splits():
    records = [
        make_record("one", "count", "MATCH (n) RETURN 1 AS count"),
        make_record("two", "lookup", "MATCH (n) RETURN 2 AS count"),
        make_record("three", "place_hops", "MATCH (n) RETURN 3 AS count"),
    ]

    train, eval_records, test, challenge = grouped_split_records(
        records,
        train_size=0.7,
        eval_size=0.15,
        test_size=0.15,
        seed=1,
        challenge_task_types={"place_hops"},
    )

    assert [record.uid for record in challenge] == ["three"]
    assert "three" not in {record.uid for record in train + eval_records + test}


def test_grouped_split_rejects_split_sizes_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        grouped_split_records(
            [],
            train_size=0.5,
            eval_size=0.1,
            test_size=0.1,
            seed=1,
        )


def test_optimize_student_trace_leaves_one_tool_trace_unchanged():
    record = make_record("one", "count", "MATCH (n) RETURN count(n) AS count")

    optimized = optimize_student_trace(record)

    assert optimized is record


def test_optimize_student_trace_prunes_failed_tool_attempts():
    record = AgentTraceRecord.model_validate(
        answerable_trace(
            uid="multi",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {"cypher_string": "MATCH (o:Object) DELETE o"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "tool_name": "run_cypher_query",
                    "content": '{"ok": false}',
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {
                                    "cypher_string": (
                                        "MATCH (o:Object) RETURN count(o) AS count"
                                    )
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "tool_name": "run_cypher_query",
                    "content": '{"ok": true, "rows": [{"count": 3}]}',
                },
                {"role": "assistant", "content": "<answer>3</answer>"},
            ],
            metadata={
                "tool_call_count": 2,
                "successful_tool_call_index": 2,
            },
            gold_cypher="MATCH (o:Object) RETURN count(o) AS count",
        )
    )

    optimized = optimize_student_trace(record)

    assert optimized is not record
    assert len(optimized.messages) == 5
    optimized_cypher = optimized.tool_calls[0].function.arguments["cypher_string"]
    assert optimized_cypher.startswith("MATCH (o:Object)")
    assert optimized.metadata["student_optimized"]
    assert optimized.metadata["raw_tool_call_count"] == 2
    assert optimized.metadata["dropped_tool_call_count"] == 1
    assert optimized.split_group == record.split_group


def test_optimize_student_traces_with_counts_reports_pruned_attempts():
    record = AgentTraceRecord.model_validate(
        answerable_trace(
            uid="multi",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {"cypher_string": "MATCH broken"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "tool_name": "run_cypher_query",
                    "content": '{"ok": false}',
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {
                                    "cypher_string": (
                                        "MATCH (o:Object) RETURN count(o) AS count"
                                    )
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "tool_name": "run_cypher_query",
                    "content": '{"ok": true, "rows": [{"count": 3}]}',
                },
                {"role": "assistant", "content": "<answer>3</answer>"},
            ],
            metadata={
                "tool_call_count": 2,
                "successful_tool_call_index": 2,
            },
            gold_cypher="MATCH (o:Object) RETURN count(o) AS count",
        )
    )

    optimized, counts = optimize_student_traces_with_counts([record])

    assert optimized[0].metadata["student_optimized"]
    assert counts["pruned_failed_tool_attempts"] == 1
    assert counts["dropped_tool_calls"] == 1


def test_optimize_student_trace_keeps_incomplete_trace_unpruned():
    record = AgentTraceRecord.model_validate(
        answerable_trace(
            metadata={
                "tool_call_count": 2,
                "successful_tool_call_index": 2,
                "incomplete_conversation": True,
            },
        )
    )

    assert optimize_student_trace(record) is record


def test_repair_rejected_invalid_answer_trace_replaces_only_final_answer():
    repaired = repair_rejected_invalid_answer_trace(rejected_invalid_answer_trace())

    assert repaired is not None
    assert repaired.uid == "rejected_teacher_repaired"
    assert repaired.real_solution == "O1"
    assert repaired.cypher_solution == "O1"
    assert repaired.final_answer == "O1"
    assert repaired.final_answer_text == "<answer>O1</answer>"
    assert repaired.messages[-1].content == "<answer>O1</answer>"
    assert repaired.validation.solution_match
    assert repaired.validation.final_answer_match
    assert repaired.metadata["final_answer_repaired"]
    assert repaired.metadata["original_final_answer"] == "{O1}"


def test_repair_rejected_invalid_answer_trace_requires_verified_cypher_solution():
    assert repair_rejected_invalid_answer_trace(
        rejected_invalid_answer_trace(reason="solution_mismatch")
    ) is None
    assert repair_rejected_invalid_answer_trace(
        rejected_invalid_answer_trace(cypher_solution="O2")
    ) is None


def test_student_trace_reader_can_include_repaired_invalid_answer_rejections(tmp_path):
    accepted_path = tmp_path / "teacher_traces.jsonl"
    rejected_path = tmp_path / "rejected.json"
    write_jsonl([answerable_trace(uid="accepted")], accepted_path)
    rejected_path.write_text(
        json.dumps([rejected_invalid_answer_trace(two_tool_calls=True)]),
        encoding="utf-8",
    )
    config = {
        "teacher_traces_jsonl_path": str(accepted_path),
        "rejected_traces_path": str(rejected_path),
    }

    accepted_only = read_student_trace_records(config, include_repaired_rejected=False)
    with_repaired, source_counts = read_student_trace_records_with_counts(
        config,
        include_repaired_rejected=True,
    )
    optimized = optimize_student_traces(with_repaired)

    assert [record.uid for record in accepted_only] == ["accepted"]
    assert source_counts["accepted_teacher_traces"] == 1
    assert source_counts["rejected_total"] == 1
    assert source_counts["rejected_reason:invalid_answer_format"] == 1
    assert source_counts["repaired_invalid_answer_format"] == 1
    assert {record.uid for record in with_repaired} == {
        "accepted",
        "rejected_teacher_repaired",
    }
    repaired = next(record for record in optimized if record.uid == "rejected_teacher_repaired")
    assert len(repaired.messages) == 4
    assert repaired.tool_calls[0].function.arguments["cypher_string"].startswith("MATCH (o:Object)")
    assert repaired.final_answer_text == "<answer>O1</answer>"
    assert repaired.metadata["student_optimized"]


def test_split_report_lines_include_split_sizes_and_optimization_counts():
    train = [make_record("train-one", "count", "MATCH (n) RETURN 1 AS count")]
    eval_records = [make_record("eval-one", "lookup", "MATCH (n) RETURN 2 AS count")]
    test = []
    challenge = [make_record("challenge-one", "place_hops", "MATCH (n) RETURN 3 AS count")]

    lines = split_report_lines(
        source_counts=Counter(
            {
                "accepted_teacher_traces": 2,
                "rejected_total": 3,
                "rejected_reason:invalid_answer_format": 2,
                "repaired_invalid_answer_format": 1,
            }
        ),
        optimization_counts=Counter(
            {
                "pruned_failed_tool_attempts": 2,
                "dropped_tool_calls": 3,
            }
        ),
        cap_dropped_counts=Counter({"count": 1}),
        split_records={
            "train": train,
            "eval": eval_records,
            "test": test,
            "challenge": challenge,
        },
        split_paths={
            "train": "train.jsonl",
            "eval": "eval.jsonl",
            "test": "test.jsonl",
            "challenge": "challenge.jsonl",
        },
        optimize_enabled=True,
    )
    text = "\n".join(lines)

    assert "accepted teacher traces loaded: 2" in text
    assert "rejected traces scanned: 3 (invalid_answer_format=2)" in text
    assert "final answer repairs from invalid_answer_format rejects: 1" in text
    assert "pruned failed tool attempts: 2 traces, 3 tool calls dropped" in text
    assert "dropped by max_per_task_type: count=1" in text
    assert "train: 1 questions (count=1) -> train.jsonl" in text
    assert "test: 0 questions (none) -> test.jsonl" in text


def test_split_report_payload_lists_optimized_questions():
    record = AgentTraceRecord.model_validate(
        answerable_trace(
            uid="optimized",
            task_type="object_count",
            metadata={
                "student_optimized": True,
                "raw_tool_call_count": 2,
                "dropped_tool_call_count": 1,
                "source_trace_uid": "teacher-one",
            },
        )
    )

    report = split_report_payload(
        data_config={
            "scene_id": "scene",
            "teacher_traces_jsonl_path": "teacher.jsonl",
            "rejected_traces_path": "rejected.json",
            "max_per_task_type": None,
            "split": {"seed": 1},
        },
        render_view="ollama_native",
        source_counts=Counter({"accepted_teacher_traces": 1}),
        optimization_counts=Counter({"pruned_failed_tool_attempts": 1}),
        cap_dropped_counts=Counter(),
        split_records={
            "train": [record],
            "eval": [],
            "test": [],
            "challenge": [],
        },
        split_paths={
            "train": "train.jsonl",
            "eval": "eval.jsonl",
            "test": "test.jsonl",
            "challenge": "challenge.jsonl",
        },
        optimize_enabled=True,
    )

    assert report["metadata"]["optimize_student_traces"] is True
    assert report["splits"]["train"]["optimized_questions"] == 1
    assert report["optimized_questions"] == [
        {
            "uid": "optimized",
            "scene_id": "scene",
            "split": "train",
            "task_type": "object_count",
            "question": "How many chairs?",
            "split_group": record.split_group,
            "gold_cypher": "MATCH (o:Object) RETURN count(o) AS count",
            "expected_solution": 3,
            "final_answer": None,
            "optimization_kinds": ["pruned_failed_tool_attempts"],
            "student_optimized": True,
            "final_answer_repaired": False,
            "raw_tool_call_count": 2,
            "dropped_tool_call_count": 1,
            "successful_tool_call_index": None,
            "seed_uid": None,
            "source_trace_uid": "teacher-one",
            "source_trace_status": None,
            "source_rejection_reason": None,
            "repair_type": None,
            "original_final_answer": None,
        }
    ]
