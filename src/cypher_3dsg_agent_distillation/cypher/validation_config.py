"""Versioned, provider-independent question validation contracts."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = 1
VALIDATOR_VERSION = 1
INLINE_FIELDS = {
    "solution_format",
    "solution_formats",
    "grounding",
    "goal",
    "answer_validation",
    "read_only",
    "validation_contract",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScoreSource(StrictModel):
    kind: Literal["property_counts"] = "property_counts"
    layer: Literal["Object", "Place", "MeshPlace", "Room", "Building"]
    property: Literal["class", "nodeSymbol"] = "class"


class Comparison(StrictModel):
    mode: Literal["exact", "sldp", "pddl_equivalence", "top_k_with_ties"] = "exact"
    numeric_abs_tolerance: float = Field(default=0, ge=0, allow_inf_nan=False)
    point_abs_tolerance: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    limit: int | dict[str, str] | None = None
    scores: dict[str, float] | None = None
    score_source: ScoreSource | None = None

    @field_validator("limit", mode="before")
    @classmethod
    def valid_limit(cls, value):
        if value is None:
            return value
        if (
            isinstance(value, dict)
            and set(value) == {"binding"}
            and isinstance(value["binding"], str)
        ):
            return value
        if type(value) is not int or value < 0:
            raise ValueError("limit must be a nonnegative integer or controlled binding")
        return value

    @model_validator(mode="after")
    def valid_scores(self):
        if self.score_source and self.mode != "top_k_with_ties":
            raise ValueError("score_source requires top_k_with_ties")
        if self.scores and any(not math.isfinite(value) for value in self.scores.values()):
            raise ValueError("Scores must be finite")
        return self


class ValueSpec(StrictModel):
    type: Literal["integer", "number", "string", "node_symbol", "point"] = "string"
    layers: list[str | dict[str, str]] = Field(default_factory=list)
    property: str | None = None
    oracle_source: str = "value"
    origin: dict[str, Any] = Field(default_factory=dict)
    cardinality: Literal["set", "list", "multiset"] = "set"
    nulls: Literal["reject", "empty_set"] = "reject"
    dimensions: Literal[3] = 3
    column: str | None = None
    position: int | None = Field(default=None, ge=0)
    path: list[str | int] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_selector(self):
        if self.column is not None and self.position is not None:
            raise ValueError("Declare column or position, not both")
        return self


class RowCount(StrictModel):
    min: int = Field(default=0, ge=0)
    max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.max is not None and self.max < self.min:
            raise ValueError("row_count.max must be at least min")
        return self


class ResultSpec(StrictModel):
    kind: Literal["scalar", "set", "multiset", "list", "tuples", "role_sets"]
    type: Literal["integer", "number", "string", "node_symbol", "point"] = "string"
    layers: list[str | dict[str, str]] = Field(default_factory=list)
    property: str | None = None
    oracle_source: str = "value"
    origin: dict[str, Any] = Field(default_factory=dict)
    element: ValueSpec | None = None
    roles: dict[str, ValueSpec] = Field(default_factory=dict)
    accepted_forms: list[Literal["scalar_columns", "collected_columns", "nodes", "maps"]] = Field(
        default_factory=lambda: ["scalar_columns", "collected_columns", "nodes"]
    )
    mapping: Literal["provenance_or_unique_type", "explicit"] = "provenance_or_unique_type"
    ambiguity: Literal["unresolved"] = "unresolved"
    extra_columns: Literal["reject", "declared", "inferred"] = "inferred"
    auxiliary: dict[str, ValueSpec] = Field(default_factory=dict)
    symbol_case_sensitive: Literal[True] = True
    row_count: RowCount = Field(default_factory=RowCount)
    comparison: Comparison = Field(default_factory=Comparison)
    unordered_tuple: bool = False

    @model_validator(mode="after")
    def shape(self):
        if self.kind in {"role_sets", "tuples"} and not self.roles:
            raise ValueError("roles are required for role_sets/tuples")
        if self.unordered_tuple and self.kind != "tuples":
            raise ValueError("unordered_tuple only applies to tuples")
        if self.auxiliary and self.extra_columns not in {"declared", "inferred"}:
            raise ValueError("auxiliary columns require extra_columns: declared or inferred")
        if (
            self.mapping == "explicit"
            and not self.roles
            and (
                self.element is None
                or self.element.column is None
                and self.element.position is None
            )
        ):
            raise ValueError(
                "Explicit scalar/collection mapping requires element.column or position"
            )
        if self.mapping == "explicit" and any(
            role.column is None and role.position is None for role in self.roles.values()
        ):
            raise ValueError("Explicit mapping requires a column or position for every role")
        return self


def symbol_type(layers):
    if any(not isinstance(layer, str) for layer in layers):
        return ""  # validated again after controlled bindings resolve
    return (
        "PlaceOrMeshPlace" if set(layers) == {"Place", "MeshPlace"} else layers[0] if layers else ""
    )


class AnswerSpec(StrictModel):
    formats: list[str] = Field(default_factory=list)
    empty: str = ""
    answer_tags: Literal["exactly_one"] = "exactly_one"
    comparison: Comparison = Field(default_factory=Comparison)

    @field_validator("formats")
    @classmethod
    def valid_formats(cls, values):
        from cypher_3dsg_agent_distillation.graph.heracles_validation import _canonical_sldp_format

        def check(value):
            if value.startswith(("set[", "list[")) and value.endswith("]"):
                return check(value[value.index("[") + 1 : -1])
            return value in {"str", "int", "float", "point", "bool"}

        if any(not check(_canonical_sldp_format(value)) for value in values):
            raise ValueError("Unknown answer format")
        return values


class QueryPolicy(StrictModel):
    read_only: Literal[True] = True
    refusal: str | None = None


class ValidationContract(StrictModel):
    version: Literal[1] = 1
    query_policy: QueryPolicy = Field(default_factory=QueryPolicy)
    result: ResultSpec
    answer: AnswerSpec = Field(default_factory=AnswerSpec)
    goal: dict[str, Any] | None = None

    @field_validator("version", mode="before")
    @classmethod
    def strict_version(cls, value):
        if type(value) is not int or value != CONTRACT_VERSION:
            raise ValueError("Unsupported contract version")
        return value

    @model_validator(mode="after")
    def goal_references(self):
        if self.goal is not None:
            if self.result.kind != "role_sets":
                raise ValueError("Goal rendering requires a role_sets result")

            def goal_keys(value):
                if not isinstance(value, dict):
                    raise ValueError("Goal clauses must be mappings")
                allowed = (
                    {"fact", "arguments", "multiple", "negate_each"}
                    if "fact" in value
                    else {"operator", "clauses"}
                )
                if value.keys() - allowed:
                    raise ValueError("Unknown goal configuration fields")
                for argument in value.get("arguments", []):
                    if isinstance(argument, dict):
                        if set(argument) not in ({"result"}, {"literal"}):
                            raise ValueError("Goal arguments require exactly result or literal")
                        if "result" in argument and argument["result"] not in self.result.roles:
                            raise ValueError("Goal refers to undeclared grounding role")
                for child in value.get("clauses", []):
                    goal_keys(child)

            goal_keys(self.goal)
            from cypher_3dsg_agent_distillation.cypher.task_semantics import validate_goal_semantics

            fields = {
                k: {"symbol_type": symbol_type(v.layers)} for k, v in self.result.roles.items()
            }
            validate_goal_semantics(self.goal, fields)
        elif self.answer.formats:
            from cypher_3dsg_agent_distillation.graph.heracles_validation import (
                sldp_value_matches_solution_formats,
            )

            if not sldp_value_matches_solution_formats(self.answer.empty, self.answer.formats):
                raise ValueError("Configured empty answer violates answer.formats")
        values = [*self.result.roles.values(), *self.result.auxiliary.values()]
        if self.result.element:
            values.append(self.result.element)
        values.append(self.result)
        for value in values:
            for layer in value.layers:
                if isinstance(layer, dict):
                    if set(layer) != {"binding"}:
                        raise ValueError("Layer adapter requires a binding")
                elif layer not in {"Object", "Place", "MeshPlace", "Room", "Building"}:
                    raise ValueError("Unknown node-symbol layer")
        return self


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently taking the last declaration."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r}, line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def strict_yaml(path):
    value = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if (isinstance(value, dict) and isinstance(result.get(key), dict))
            else deepcopy(value)
        )
    return result


def default_validation_path(template_path):
    path = Path(template_path)
    return path.parent / "validation" / path.name


def validation_inputs(template_path, validation_config_path=None, validation_defaults_path=None):
    path = Path(validation_config_path or default_validation_path(template_path))
    defaults = Path(validation_defaults_path or path.parent / "defaults.yaml")
    return path, defaults


def load_question_catalog(
    template_path, validation_config_path=None, validation_defaults_path=None
):
    """Compose private generation/validation inputs; source YAML stays separated."""
    path, defaults_path = validation_inputs(
        template_path, validation_config_path, validation_defaults_path
    )
    catalog = strict_yaml(template_path)
    declarations = strict_yaml(path)
    defaults = strict_yaml(defaults_path)
    for config, source in ((declarations, path), (defaults, defaults_path)):
        if type(config.get("version")) is not int or config["version"] != CONTRACT_VERSION:
            raise ValueError(f"Unsupported validation version: {source}")
    if set(declarations) != {"version", "templates"} or set(defaults) != {"version", "defaults"}:
        raise ValueError("Validation files require version and templates/defaults only")
    if INLINE_FIELDS & catalog.keys():
        raise ValueError("Move validation fields out of the main question catalog")
    if not isinstance(defaults["defaults"], dict) or "version" in defaults["defaults"]:
        raise ValueError("Defaults must be a mapping with version declared only at file level")
    templates = catalog.get("templates", [])
    names = [t["query_type"] for t in templates]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate query_type in {template_path}")
    if not isinstance(declarations["templates"], dict) or set(names) != set(
        declarations["templates"]
    ):
        raise ValueError(f"Validation coverage differs from {template_path}")
    provenance = {str(p.resolve()): fingerprint(strict_yaml(p)) for p in (path, defaults_path)}
    for template in templates:
        inline = INLINE_FIELDS & template.keys()
        if inline:
            raise ValueError(
                f"{template['query_type']}: move inline fields {sorted(inline)} to {path}"
            )
        entry = declarations["templates"][template["query_type"]]
        if not isinstance(entry, dict) or "version" in entry:
            raise ValueError("Declare validation version once at the file level")
        contract = ValidationContract.model_validate(
            merge(defaults["defaults"], declarations["templates"][template["query_type"]])
        )
        raw = contract.model_dump(mode="json")
        # Check every controlled binding against template variable roots.
        roots = set(template.get("variables", {}))

        def check(value, roots=roots, query_type=template["query_type"]):
            if isinstance(value, dict):
                if set(value) == {"binding"} and value["binding"].split(".")[0] not in roots:
                    raise ValueError(f"{query_type}: unknown binding {value['binding']}")
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        check(raw)
        template.update(validation_contract=raw, validation_provenance=provenance)
    return catalog


def resolve_contract(contract, variables):
    from cypher_3dsg_agent_distillation.cypher.task_semantics import flattened_bindings

    bindings = flattened_bindings(variables)

    def visit(value):
        if isinstance(value, dict):
            if set(value) == {"binding"}:
                return deepcopy(bindings[value["binding"]])
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value

    return ValidationContract.model_validate(visit(contract)).model_dump(mode="json")


def contract_for_record(record):
    contract = record.get("validation_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            "A top-level validation_contract is required; regenerate the dataset "
            "using the current question catalogs."
        )
    ValidationContract.model_validate(contract)
    return contract


def implementation_fingerprint():
    files = [
        "validation_config.py",
        "result_validation.py",
        "query_provenance.py",
        "validation_audit.py",
        "validation_refresh.py",
        "task_semantics.py",
        "pddl_goals.py",
    ]
    hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in files
        if Path(__file__).with_name(name).exists()
    }
    from cypher_3dsg_agent_distillation.graph import heracles_validation, neo4j_validation

    for module in (heracles_validation, neo4j_validation):
        hashes[module.__name__] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    for relative in [
        "agents/student_evaluation.py",
        "agents/openrouter_teacher.py",
        "agents/trace_evidence.py",
        "agents/teacher_optimizations.py",
        "agents/agent_evaluation.py",
        "agents/agent_records.py",
        "cypher/cypher_seed_records.py",
        "cypher/evaluation.py",
    ]:
        source = Path(__file__).parents[1] / relative
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return fingerprint({"engine": VALIDATOR_VERSION, "files": hashes})


def validation_identity(data_config):
    """Content inputs shared by dataset, teacher-resume and evaluation identities."""
    from cypher_3dsg_agent_distillation.core.config import resolve_project_path

    files = {}
    question_sets = data_config["question_sets"]
    for item in question_sets:
        path, defaults = validation_inputs(
            resolve_project_path(item["question_templates_path"]),
            item.get("validation_config_path"),
            item.get("validation_defaults_path"),
        )
        for candidate in (path, defaults):
            files[str(candidate)] = (
                hashlib.sha256(candidate.read_bytes()).hexdigest()
                if candidate.exists()
                else "missing"
            )
    return {
        "contract_version": CONTRACT_VERSION,
        "implementation": implementation_fingerprint(),
        "files": files,
    }


def template_identity(template):
    return fingerprint({key: template.get(key) for key in ("query_type", "cypher", "variables")})


def template_bindings(record, template):
    """Validate explicit generation provenance; never reconstruct historical bindings."""
    from cypher_3dsg_agent_distillation.cypher.task_semantics import (
        parse_query,
        resolve_semantics,
        validate_query_semantics,
    )

    metadata = record.get("metadata") or {}
    bindings = metadata.get("template_bindings")
    if not isinstance(bindings, dict) or metadata.get("template_identity") != template_identity(
        template
    ):
        raise ValueError("Missing or incompatible template provenance; regenerate this dataset.")
    query = metadata.get("seed_gold_cypher") or record.get("cypher") or record.get("gold_cypher")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Missing trusted source query; regenerate this dataset.")
    validate_query_semantics(resolve_semantics(parse_query(template["cypher"]), bindings), query)
    return bindings
