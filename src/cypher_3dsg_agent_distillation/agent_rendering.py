from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from cypher_3dsg_agent_distillation.agent_records import (
    DEFAULT_AGENT_TOOL,
    AgentMessage,
    AgentTraceRecord,
)


class RenderView(StrEnum):
    OLLAMA_NATIVE = "ollama_native"
    CUSTOM_XML = "custom_xml"


def shell_quote_single(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def custom_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    rendered_args = ",".join(
        f"{name}={shell_quote_single(str(value)) if isinstance(value, str) else value}"
        for name, value in arguments.items()
    )
    return f"<tool>{tool_name}({rendered_args})</tool>"


def render_custom_message(message: AgentMessage) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_calls:
        tool_call = message.tool_calls[0]
        return {
            "role": "assistant",
            "content": custom_tool_call(tool_call.function.name, tool_call.function.arguments),
        }
    if message.role == "tool":
        return {"role": "user", "content": f"Output of tool call: {message.content or ''}"}
    payload = message.model_dump(exclude_none=True)
    payload.pop("tool_calls", None)
    payload.pop("tool_name", None)
    return payload


def render_ollama_native(record: AgentTraceRecord) -> dict[str, Any]:
    return {
        **record.model_dump(by_alias=True, exclude_none=True),
        "render_view": RenderView.OLLAMA_NATIVE.value,
        "tools": [DEFAULT_AGENT_TOOL],
        "messages": [
            message.model_dump(exclude_none=True, by_alias=True) for message in record.messages
        ],
    }


def render_custom_xml(record: AgentTraceRecord) -> dict[str, Any]:
    return {
        **record.model_dump(by_alias=True, exclude_none=True),
        "render_view": RenderView.CUSTOM_XML.value,
        "messages": [render_custom_message(message) for message in record.messages],
    }


def render_trace(record: AgentTraceRecord, view: str | RenderView) -> dict[str, Any]:
    view = RenderView(view)
    if view == RenderView.OLLAMA_NATIVE:
        return render_ollama_native(record)
    if view == RenderView.CUSTOM_XML:
        return render_custom_xml(record)
    raise ValueError(f"Unsupported render view: {view}")


def render_trace_jsonl(records: list[AgentTraceRecord], view: str | RenderView) -> list[str]:
    return [json.dumps(render_trace(record, view), ensure_ascii=False) for record in records]
