from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Text2CypherRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    uid: str
    question: str
    cypher: str = Field(alias="gold_cypher")
    task_type: str = "unknown"
    query_type: str | None = None
    database: str = "heracles_scene_graph"
    read_only: bool = True
    expected_result_hash: str | None = None
    name: str | None = None
    solution: str | None = None
    tags: list[str] = Field(default_factory=list)
    scene_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.query_type is None:
            self.query_type = self.task_type
        elif self.task_type == "unknown":
            self.task_type = self.query_type


class PreparedRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    uid: str
    question: str
    cypher: str
    task_type: str
    query_type: str
    database: str
    read_only: bool
    expected_result_hash: str | None = None
    tags: list[str]
    scene_id: str | None = None
    schema_name: str
    schema_text: str = Field(alias="schema")
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = Field(default_factory=dict)
