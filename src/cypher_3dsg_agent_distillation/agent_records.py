from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_AGENT_TOOL = "run_cypher_query"
DEFAULT_TEACHER_MODEL = "openai/gpt-oss-120b"
WRITE_REFUSAL_ANSWER = "WRITE_QUERY_NOT_ALLOWED"


def normalize_cypher(cypher: str) -> str:
    return re.sub(r"\s+", " ", cypher.strip())


def split_group_for_record(scene_id: str | None, task_type: str, cypher: str) -> str:
    digest = hashlib.sha1(normalize_cypher(cypher).encode("utf-8")).hexdigest()[:16]
    return f"{scene_id or 'unknown'}:{task_type}:{digest}"


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: ToolFunction


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = None

    @field_validator("tool_calls", mode="before")
    @classmethod
    def normalize_tool_calls(cls, value: Any) -> Any:
        if value is None:
            return None
        normalized = []
        for item in value:
            if "function" in item:
                normalized.append(item)
            else:
                normalized.append({"type": "function", "function": item})
        return normalized


class TraceValidation(BaseModel):
    model_config = ConfigDict(extra="allow")

    static_valid: bool = False
    executable: bool = False
    solution_match: bool = False
    static_issues: list[str] = Field(default_factory=list)
    error: str | None = None


class AgentTraceRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    uid: str
    scene_id: str | None = None
    question: str
    task_type: str = Field(validation_alias=AliasChoices("task_type", "query_type"))
    tools: list[str] = Field(default_factory=lambda: [DEFAULT_AGENT_TOOL])
    messages: list[AgentMessage]
    gold_cypher: str
    expected_solution: Any | None = None
    split_group: str | None = None
    teacher_model: str = DEFAULT_TEACHER_MODEL
    validation: TraceValidation = Field(default_factory=TraceValidation)
    read_only: bool = True
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trace(self) -> AgentTraceRecord:
        if not self.messages:
            raise ValueError("Agent trace must include at least one message.")
        if self.split_group is None:
            self.split_group = split_group_for_record(
                self.scene_id,
                self.task_type,
                self.gold_cypher,
            )
        if self.is_refusal:
            if self.tool_calls:
                raise ValueError("Refusal traces must not include tool calls.")
        elif DEFAULT_AGENT_TOOL in self.tools:
            if not self.tool_calls:
                raise ValueError("Answerable traces must include a run_cypher_query tool call.")
        return self

    @property
    def is_refusal(self) -> bool:
        if normalize_cypher(self.gold_cypher) in {
            WRITE_REFUSAL_ANSWER,
            f"// {WRITE_REFUSAL_ANSWER}",
        }:
            return True
        final = self.final_answer_text
        return final is not None and WRITE_REFUSAL_ANSWER in final

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for message in self.messages:
            calls.extend(message.tool_calls or [])
        return calls

    @property
    def first_cypher_argument(self) -> str | None:
        for tool_call in self.tool_calls:
            if tool_call.function.name == DEFAULT_AGENT_TOOL:
                value = tool_call.function.arguments.get("cypher_string")
                return str(value) if value is not None else None
        return None

    @property
    def final_answer_text(self) -> str | None:
        for message in reversed(self.messages):
            if message.role == "assistant" and message.content:
                return message.content
        return None


def trace_to_json(record: AgentTraceRecord) -> dict[str, Any]:
    return record.model_dump(by_alias=True, exclude_none=True)


def trace_to_json_line(record: AgentTraceRecord) -> str:
    return json.dumps(trace_to_json(record), ensure_ascii=False)
