from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cypher_3dsg_agent_distillation.core.config import data_paths_from_config, load_yaml


class SeedRecord(BaseModel):
    validation_contract: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("validation_contract")
    @classmethod
    def valid_contract(cls, value):
        from cypher_3dsg_agent_distillation.cypher.validation_config import contract_for_record

        return contract_for_record({"validation_contract": value})

    @field_validator("metadata")
    @classmethod
    def require_template_provenance(cls, value):
        if not isinstance(value.get("template_bindings"), dict) or not value.get(
            "template_identity"
        ):
            raise ValueError(
                "Current template identity and explicit bindings are required; regenerate seeds."
            )
        return value


class CypherSeedRecord(SeedRecord):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    uid: str
    question: str
    cypher: str = Field(alias="gold_cypher")
    query_type: str
    task_family: str | None = None
    output_type: str | None = None
    experiment_profile: str | None = None
    read_only: bool = True
    expected_result_hash: str | None = None
    solution: Any | None = None
    solution_formats: list[str] = Field(default_factory=list)
    grounding: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    scene_id: str | None = None
    labelspace_description: str | None = None


class PreparedCypherSeedRecord(SeedRecord):
    model_config = ConfigDict(populate_by_name=True)

    uid: str
    question: str
    cypher: str
    query_type: str
    task_family: str
    output_type: str
    experiment_profile: str | None = None
    read_only: bool = True
    expected_result_hash: str | None = None
    solution: Any | None = None
    solution_formats: list[str] = Field(default_factory=list)
    grounding: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    scene_id: str | None = None
    labelspace_description: str | None = None


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
                experiment_profile=record.experiment_profile,
                read_only=record.read_only,
                expected_result_hash=record.expected_result_hash,
                solution=record.solution,
                solution_formats=record.solution_formats,
                grounding=record.grounding,
                validation_contract=record.validation_contract,
                tags=record.tags,
                scene_id=record.scene_id,
                labelspace_description=record.labelspace_description,
                metadata=metadata,
            )
        )
    return prepared


def load_prepared_seed_records(path):
    from cypher_3dsg_agent_distillation.core.io import read_jsonl

    return [
        PreparedCypherSeedRecord.model_validate(record).model_dump(exclude_none=True)
        for record in read_jsonl(path)
    ]
