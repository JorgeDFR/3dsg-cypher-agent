from __future__ import annotations

import json
from typing import Any

from cypher_3dsg_agent_distillation.agent_records import DEFAULT_AGENT_TOOL

AGENT_SYSTEM_PROMPT = """You are a 3D scene graph agent.
Use the run_cypher_query tool when a read-only Cypher query is needed.
Return the final answer as compact answer notation wrapped in <answer>...</answer>.
Do not call tools for write requests; answer WRITE_QUERY_NOT_ALLOWED instead."""


def run_cypher_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": DEFAULT_AGENT_TOOL,
            "description": (
                "Run a read-only Cypher query against a Neo4j database containing "
                "a 3D Scene Graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cypher_string": {
                        "type": "string",
                        "description": "The read-only Cypher query to execute.",
                    }
                },
                "required": ["cypher_string"],
                "additionalProperties": False,
            },
        },
    }


def build_agent_user_prompt(schema: str, question: str) -> str:
    return (
        "Use the available tools to answer this 3D Scene Graph question.\n\n"
        f"{schema.strip()}\n\n"
        f"Question: {question.strip()}"
    )


def build_teacher_prompt(
    schema: str,
    question: str,
    expected_solution: Any,
    read_only: bool,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate one complete agent trace as JSON only. The JSON must contain "
                "a `messages` array. If the request is answerable and read-only, the "
                "assistant must call run_cypher_query with a cypher_string argument, "
                "then include a tool result placeholder, then a final assistant answer. "
                "For write requests, do not call tools and answer WRITE_QUERY_NOT_ALLOWED.\n\n"
                f"Tool schema:\n{json.dumps(run_cypher_tool_schema(), indent=2)}\n\n"
                f"Read only: {read_only}\n"
                f"Expected answer notation: {expected_solution}\n\n"
                f"{build_agent_user_prompt(schema, question)}"
            ),
        },
    ]


def build_trace_messages(
    schema: str,
    question: str,
    cypher: str,
    tool_result: str,
    final_answer: Any,
    *,
    refusal: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": build_agent_user_prompt(schema, question)},
    ]
    if refusal:
        messages.append({"role": "assistant", "content": f"<answer>{final_answer}</answer>"})
        return messages
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": DEFAULT_AGENT_TOOL,
                            "arguments": {"cypher_string": cypher.strip()},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_name": DEFAULT_AGENT_TOOL, "content": str(tool_result)},
            {"role": "assistant", "content": f"<answer>{final_answer}</answer>"},
        ]
    )
    return messages
