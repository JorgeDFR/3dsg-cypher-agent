from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cypher_3dsg_agent_distillation.core.config import data_paths_from_config, load_yaml


class CypherSeedRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    uid: str
    question: str
    cypher: str = Field(alias="gold_cypher")
    query_type: str = "unknown"
    task_family: str | None = None
    output_type: str | None = None
    experiment_profile: str | None = None
    read_only: bool = True
    expected_result_hash: str | None = None
    solution: Any | None = None
    solution_formats: list[str] = Field(default_factory=list)
    answer_validation: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    scene_id: str | None = None
    labelspace_description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreparedCypherSeedRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    uid: str
    question: str
    cypher: str
    query_type: str
    task_family: str = "cypher_qa"
    output_type: str = "SLDP"
    experiment_profile: str | None = None
    read_only: bool
    expected_result_hash: str | None = None
    solution: Any | None = None
    solution_formats: list[str] = Field(default_factory=list)
    answer_validation: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None
    tags: list[str]
    scene_id: str | None = None
    labelspace_description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_raw_seed_records(path: str | Path) -> list[CypherSeedRecord]:
    data = load_yaml(path)
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        raise TypeError("Expected `questions` to be a list.")
    return [CypherSeedRecord.model_validate(question) for question in questions]


def load_raw_seed_records_from_config(data_config: dict[str, Any]) -> list[CypherSeedRecord]:
    path = data_paths_from_config(data_config).raw_questions_path
    records: list[CypherSeedRecord] = []
    seen_uids: set[str] = set()
    for record in load_raw_seed_records(path):
        if record.uid in seen_uids:
            raise ValueError(f"Duplicate uid `{record.uid}` while loading {path}.")
        seen_uids.add(record.uid)
        records.append(record)
    return records


def prepare_seed_records(
    raw_records: Iterable[CypherSeedRecord],
    *,
    task_family: str = "cypher_qa",
    output_type: str = "SLDP",
    metadata_defaults: dict[str, Any] | None = None,
) -> list[PreparedCypherSeedRecord]:
    prepared: list[PreparedCypherSeedRecord] = []
    for record in raw_records:
        metadata = dict(metadata_defaults or {})
        metadata.update(record.metadata)
        prepared.append(
            PreparedCypherSeedRecord(
                uid=record.uid,
                question=record.question,
                cypher=record.cypher.strip(),
                query_type=record.query_type,
                task_family=record.task_family or task_family,
                output_type=record.output_type or output_type,
                experiment_profile=(
                    record.experiment_profile or metadata.get("experiment_profile")
                ),
                read_only=record.read_only,
                expected_result_hash=record.expected_result_hash,
                solution=record.solution,
                solution_formats=record.solution_formats,
                answer_validation=record.answer_validation,
                grounding=record.grounding,
                tags=record.tags,
                scene_id=record.scene_id,
                labelspace_description=record.labelspace_description,
                metadata=metadata,
            )
        )
    return prepared
