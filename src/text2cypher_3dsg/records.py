from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Text2CypherRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    uid: str
    question: str
    cypher: str = Field(alias="gold_cypher")
    query_type: str = "unknown"
    read_only: bool = True
    expected_result_hash: str | None = None
    solution: Any | None = None
    tags: list[str] = Field(default_factory=list)
    scene_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreparedRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    uid: str
    question: str
    cypher: str
    query_type: str
    read_only: bool
    expected_result_hash: str | None = None
    solution: Any | None = None
    tags: list[str]
    scene_id: str | None = None
    schema_name: str
    schema_text: str = Field(alias="schema")
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = Field(default_factory=dict)
