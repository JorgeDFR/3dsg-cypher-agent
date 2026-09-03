from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord
from cypher_3dsg_agent_distillation.agents.agent_rendering import RenderView, render_trace
from tests.helpers import answerable_trace


def test_ollama_native_render_preserves_tool_calls():
    record = AgentTraceRecord.model_validate(answerable_trace())

    rendered = render_trace(record, RenderView.OLLAMA_NATIVE)

    assert rendered["render_view"] == "ollama_native"
    assert rendered["messages"][2]["tool_calls"][0]["function"]["name"] == "run_cypher_query"


def test_custom_xml_render_turns_tool_call_into_text():
    record = AgentTraceRecord.model_validate(answerable_trace())

    rendered = render_trace(record, RenderView.CUSTOM_XML)

    assert rendered["render_view"] == "custom_xml"
    assert rendered["messages"][2]["content"].startswith("<tool>run_cypher_query")
    assert "cypher_string=" in rendered["messages"][2]["content"]
    assert rendered["messages"][3]["role"] == "user"
    assert rendered["messages"][3]["content"].startswith("Output of tool call:")
