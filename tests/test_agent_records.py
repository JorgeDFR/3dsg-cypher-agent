import pytest
from agent_helpers import answerable_trace

from cypher_3dsg_agent_distillation.agent_records import AgentTraceRecord


def test_agent_trace_record_accepts_task_type_and_builds_split_group():
    record = AgentTraceRecord.model_validate(answerable_trace())

    assert record.task_type == "object_class_count"
    assert record.first_cypher_argument.startswith("MATCH")
    assert record.split_group.startswith("scene:object_class_count:")
    assert record.model_dump()["task_type"] == "object_class_count"


def test_agent_trace_record_accepts_legacy_query_type_input():
    payload = answerable_trace()
    payload["query_type"] = payload.pop("task_type")

    record = AgentTraceRecord.model_validate(payload)

    assert record.task_type == "object_class_count"


def test_agent_trace_record_rejects_answerable_trace_without_tool_call():
    payload = answerable_trace(messages=[{"role": "assistant", "content": "<answer>3</answer>"}])

    with pytest.raises(ValueError, match="tool call"):
        AgentTraceRecord.model_validate(payload)


def test_agent_trace_record_rejects_refusal_with_tool_call():
    payload = answerable_trace(gold_cypher="// WRITE_QUERY_NOT_ALLOWED")

    with pytest.raises(ValueError, match="Refusal traces"):
        AgentTraceRecord.model_validate(payload)
