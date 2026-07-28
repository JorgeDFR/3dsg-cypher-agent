import pytest
from agent_helpers import answerable_trace

from cypher_3dsg_agent_distillation.agent_dataset import (
    assert_no_split_leakage,
    grouped_split_records,
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
