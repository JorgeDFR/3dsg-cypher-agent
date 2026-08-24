from __future__ import annotations

from typing import Any

from cypher_3dsg_agent_distillation.agent_records import DEFAULT_AGENT_TOOL


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
