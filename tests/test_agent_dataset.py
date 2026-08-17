import pytest
from agent_helpers import answerable_trace

from cypher_3dsg_agent_distillation.agent_dataset import (
    assert_no_split_leakage,
    grouped_split_records,
    optimize_student_trace,
)
from cypher_3dsg_agent_distillation.agent_records import AgentTraceRecord


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
