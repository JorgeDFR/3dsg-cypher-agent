"""Conservative, answer-independent selection of saved retrieval evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cypher_3dsg_agent_distillation.agents.agent_records import DEFAULT_AGENT_TOOL
from cypher_3dsg_agent_distillation.cypher.query_provenance import projections
from cypher_3dsg_agent_distillation.cypher.result_validation import (
    _compatible_type,
    _lineage_atom,
    _projection_spec,
    decode_result,
)
from cypher_3dsg_agent_distillation.cypher.task_semantics import parse_query
from cypher_3dsg_agent_distillation.cypher.validation_config import ValidationContract, ValueSpec

EARLIER_RETRIEVAL_POLICY = "earlier_retrievals_v1"


@dataclass
class ToolExecution:
    call_index: int  # One-based, counting every tool call, including failed calls.
    query: str | None
    call_id: str | None = None
    rows: list[dict[str, Any]] | None = None
    columns: list[str] | None = None
    complete: bool = False

    def saved_result(self):
        return self.query, self.rows, self.columns


@dataclass
class EarlierRetrieval:
    execution: ToolExecution
    diagnostics: dict[int, str]


def tool_executions(messages):
    """Pair individual calls with their responses, never by matching result values."""
    executions, pending = [], None
    for message in messages:
        calls = message.get("tool_calls") or []
        if calls:
            pending = None
            for call in calls:
                function = call.get("function") or {}
                args = function.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                query = args.get("cypher_string") if isinstance(args, dict) else None
                entry = ToolExecution(
                    len(executions) + 1,
                    query
                    if function.get("name") == DEFAULT_AGENT_TOOL
                    and isinstance(query, str)
                    and query.strip()
                    else None,
                    call.get("id"),
                )
                executions.append(entry)
            if len(calls) == 1 and message.get("role") == "assistant":
                pending = executions[-1]
        elif message.get("role") == "tool":
            if pending is None:
                # Orphaned feedback makes the stream unsafe to reinterpret.
                return []
            response_id = message.get("tool_call_id")
            if (response_id is not None and response_id != pending.call_id) or message.get(
                "tool_name"
            ) not in (None, DEFAULT_AGENT_TOOL):
                return []
            try:
                payload = json.loads(message.get("content") or "")
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict) and payload.get("ok") is True:
                rows = payload.get("rows")
                if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                    pending.rows = rows
                    pending.columns = payload.get("columns")
                    valid_columns = pending.columns is None or (
                        isinstance(pending.columns, list)
                        and all(isinstance(column, str) for column in pending.columns)
                    )
                    pending.complete = (
                        bool(pending.query)
                        and valid_columns
                        and not any(payload.get(key) for key in ("rows_truncated", "rows_omitted"))
                    )
            pending = None
        elif message.get("role") in {"assistant", "user", "system"}:
            pending = None
    return executions


def _roles(contract):
    spec = contract.result
    return spec.roles or {
        "value": spec.element
        or ValueSpec(
            type=spec.type,
            layers=spec.layers,
            property=spec.property,
            origin=spec.origin,
        )
    }


def _simple_inventory(operations, lineage):
    matches = [op for op in operations if op["kind"] == "match"]
    return (
        len(matches) == 1
        and not matches[0].get("optional")
        and not matches[0].get("properties")
        and all(op["kind"] in {"match", "project", "limit", "order"} for op in operations)
        and all(info.get("kind") in {"property", "labels"} for info in lineage)
        and len({info.get("source") for info in lineage}) == 1
    )


def _reference_inputs(query):
    """Public input IDs from the task query, excluding IDs that are its output roles."""
    operations = parse_query(query)["operations"]
    output = {
        _lineage_atom(info).get("origin", {}).get("nodeSymbol") for info in projections(query)
    }
    inputs = set()
    for op in operations:
        if op["kind"] == "match":
            expr = op.get("properties", {}).get("nodeSymbol", {})
            if expr.get("kind") == "literal" and isinstance(expr["value"], str):
                inputs.add(expr["value"])
    return inputs - output


def diagnostic_reason(query, contract, reference_query=None):
    """Prove an auxiliary query from schema/public task constraints, never gold values.

    Deliberately unsupported: arbitrary malformed/ambiguous result shapes, wrong
    task filters, unparsed Cypher, and attempts to infer intent from final answers.
    """
    try:
        lineage = projections(query)
        operations = parse_query(query)["operations"]
        roles = _roles(contract)
        if not lineage or any(info.get("kind") in {"unknown", "union"} for info in lineage):
            return None
        # An explicitly named task output cannot be reclassified as a diagnostic.
        reserved = {name for name in roles if name != "value"}
        reserved.update(role.column for role in roles.values() if role.column is not None)
        if any(info["column"] in reserved for info in lineage):
            return None
        known = [_projection_spec(info) for info in lineage]
        if (
            all(known)
            and all(
                not any(_compatible_type(role, info) for role in roles.values()) for info in lineage
            )
            and all(info.get("kind") in {"property", "labels", "computed"} for info in lineage)
        ):
            return "projection_types_do_not_represent_task_outputs"
        if not reference_query:
            return None
        reference = parse_query(reference_query)["operations"]
        atoms = [_lineage_atom(info) for info in lineage]
        if (
            all(info.get("kind") in {"property", "labels"} for info in lineage)
            and len({info.get("source") for info in atoms}) == 1
            and all(
                op["kind"] in {"match", "project", "filter", "order", "limit"} for op in operations
            )
            and len([op for op in operations if op["kind"] == "match"]) == 1
            and atoms[0].get("origin", {}).get("nodeSymbol") in _reference_inputs(reference_query)
        ):
            return "inspect_public_input_node"
        if (
            _simple_inventory(operations, lineage)
            and any(op["kind"] == "limit" for op in operations)
            and not _simple_inventory(reference, projections(reference_query))
        ):
            return "bounded_unfiltered_inventory_for_a_constrained_task"
    except (ValueError, KeyError, TypeError):
        pass
    return None


def select_earlier_retrieval(messages, raw_contract, reference_query=None):
    """Strip only proven diagnostic suffixes, then stop at the first task query.

    Selection receives neither expected answers nor expected grounding. A newer
    task retrieval is never bypassed because it fails correctness comparison.
    """
    if not raw_contract:
        return None
    contract = ValidationContract.model_validate(raw_contract)
    executions = tool_executions(messages)
    if len(executions) < 2:
        return None
    diagnostics = {}
    for execution in reversed(executions):
        if not execution.complete:
            return None
        reason = diagnostic_reason(execution.query, contract, reference_query)
        if reason:
            diagnostics[execution.call_index] = reason
            continue
        if not diagnostics:
            return None
        try:
            lineage = projections(execution.query)
        except (ValueError, KeyError, TypeError):
            return None
        if not lineage or any(info.get("kind") == "unknown" for info in lineage):
            return None
        decoded = decode_result(execution.query, execution.rows, raw_contract, execution.columns)
        return EarlierRetrieval(execution, diagnostics) if decoded.status == "pass" else None
    return None


def uses_earlier_retrieval(record):
    metadata = record.get("metadata") or {}
    return "retrieval_evidence_policy" in metadata or "grounding_tool_call_index" in metadata


def marked_earlier_execution(record):
    """Recompute selection; saved indices cannot override the policy."""
    metadata = record.get("metadata") or {}
    if metadata.get("retrieval_evidence_policy") != EARLIER_RETRIEVAL_POLICY:
        return None
    selected = select_earlier_retrieval(
        record.get("messages") or [],
        record.get("validation_contract"),
        metadata.get("seed_gold_cypher"),
    )
    if selected is None:
        return None
    if (
        metadata.get("grounding_tool_call_index") != selected.execution.call_index
        or type(metadata.get("grounding_tool_call_index")) is not int
        or metadata.get("diagnostic_tool_call_indices") != sorted(selected.diagnostics)
        or (record.get("gold_cypher") or record.get("cypher")) != selected.execution.query
    ):
        return None
    return selected.execution.saved_result()
