import pytest

from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord
from tests.helpers import answerable_trace


def test_agent_trace_record_accepts_task_type_and_builds_split_group():
    record = AgentTraceRecord.model_validate(answerable_trace())

    assert record.task_type == "object_class_count"
    assert record.first_cypher_argument.startswith("MATCH")
    assert record.split_group.startswith("scene:object_class_count:")
    assert record.model_dump()["task_type"] == "object_class_count"


def test_agent_trace_record_rejects_historical_query_type_input():
    payload = answerable_trace()
    payload["query_type"] = payload.pop("task_type")
    with pytest.raises(ValueError, match="Unsupported trace fields"):
        AgentTraceRecord.model_validate(payload)


def test_agent_trace_record_rejects_answerable_trace_without_tool_call():
    payload = answerable_trace(messages=[{"role": "assistant", "content": "<answer>3</answer>"}])

    with pytest.raises(ValueError, match="tool call"):
        AgentTraceRecord.model_validate(payload)


def test_agent_trace_record_rejects_refusal_with_tool_call():
    payload = answerable_trace(gold_cypher="// WRITE_QUERY_NOT_ALLOWED")

    with pytest.raises(ValueError, match="Refusal traces"):
        AgentTraceRecord.model_validate(payload)


def test_invalid_input_empty_answer_remains_a_tool_using_trace():
    payload = answerable_trace()
    payload.update(
        {
            "expected_solution": "",
            "tags": ["invalid_input"],
        }
    )
    payload["messages"][-1]["content"] = "<answer></answer>"

    record = AgentTraceRecord.model_validate(payload)

    assert not record.is_refusal
    assert record.first_cypher_argument is not None
