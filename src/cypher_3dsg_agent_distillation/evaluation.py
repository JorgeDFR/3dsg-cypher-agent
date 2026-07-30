from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from cypher_3dsg_agent_distillation.config import ensure_parent
from cypher_3dsg_agent_distillation.heracles_validation import answer_matches_with_heracles
from cypher_3dsg_agent_distillation.neo4j_validation import (
    Neo4jValidator,
    is_safe_refusal,
    result_hash,
    static_checks,
)
from cypher_3dsg_agent_distillation.templates import format_solution_field


class ResultShapeError(ValueError):
    pass


def point_to_solution(value: Any) -> str:
    if isinstance(value, str) and value.startswith("POINT("):
        return value
    try:
        x = value.x
        y = value.y
        z = value.z
    except AttributeError:
        x = value[0]
        y = value[1]
        z = value[2]
    return f"POINT({float(x):.3f} {float(y):.3f} {float(z):.3f})"


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(format_solution_field(value))
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([<>\[\]{},:])\s*", r"\1", text)
    return text


def _strip_outer_collection(text: str, opener: str, closer: str) -> str | None:
    if text.startswith(opener) and text.endswith(closer):
        return text[1:-1]
    return None


def _split_top_level_items(text: str) -> list[str] | None:
    items = []
    start = 0
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closers = set(pairs.values())
    for index, character in enumerate(text):
        if character in pairs:
            stack.append(pairs[character])
        elif character in closers:
            if not stack or stack[-1] != character:
                return None
            stack.pop()
        elif character == "," and not stack:
            item = text[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    if stack:
        return None
    item = text[start:].strip()
    if item:
        items.append(item)
    return items


def _flat_set_items(value: Any, *, expected_set: bool = False) -> set[str] | None:
    normalized = normalize_answer(value)
    if not normalized:
        return set()

    body = _strip_outer_collection(normalized, "<", ">")
    if body is None:
        if expected_set:
            return None
        body = _strip_outer_collection(normalized, "[", "]")
    if body is None:
        if "," not in normalized:
            return None
        body = normalized

    items = _split_top_level_items(body)
    if items is None:
        return None
    return {normalize_answer(item) for item in items}


def answer_matches(actual: Any, expected: Any, output_type: str = "SLDP") -> bool:
    valid_format, correct = answer_matches_with_heracles(
        str(actual),
        str(expected),
        output_type,
    )
    return valid_format and correct


def rendered_solution(value: Any, list_style: str = "list") -> str:
    return str(format_solution_field(value, list_style=list_style)).strip()


def scalar_or_set(values: list[Any]) -> Any:
    if not values:
        return set()
    if len(values) == 1:
        return values[0]
    return set(values)


def expected_uses_informative_solution(expected_solution: Any) -> bool:
    return "{" in str(expected_solution or "")


def node_property(value: Any, property_name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(property_name)
    if hasattr(value, "get"):
        try:
            return value.get(property_name)
        except TypeError:
            pass
    try:
        return value[property_name]
    except (KeyError, TypeError):
        return None


def row_property(
    row: dict[str, Any],
    property_name: str,
    preferred_keys: list[str] | None = None,
    node_keys: list[str] | None = None,
) -> Any:
    for key in preferred_keys or []:
        value = row.get(key)
        if value is not None:
            return value

    value = row.get(property_name)
    if value is not None:
        return value

    suffix = f".{property_name}"
    for key, value in row.items():
        if key.endswith(suffix) and value is not None:
            return value

    for key in node_keys or []:
        if key in row:
            value = node_property(row[key], property_name)
            if value is not None:
                return value

    if len(row) == 1:
        value = next(iter(row.values()))
        node_value = node_property(value, property_name)
        if node_value is not None:
            return node_value
    return None


def row_values(
    rows: list[dict[str, Any]],
    property_name: str,
    preferred_keys: list[str] | None = None,
    node_keys: list[str] | None = None,
) -> list[Any]:
    values = []
    for row in rows:
        value = row_property(row, property_name, preferred_keys, node_keys)
        if value is None:
            columns = ", ".join(row) or "<none>"
            raise ResultShapeError(
                f"missing `{property_name}` in returned columns: {columns}"
            )
        values.append(value)
    return values


def row_count_value(rows: list[dict[str, Any]]) -> int | float:
    if not rows:
        return 0
    row = rows[0]
    value = row.get("count")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    for key, value in row.items():
        if key.lower().startswith("count(") and isinstance(value, int | float):
            return value
    if len(row) == 1:
        value = next(iter(row.values()))
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
    columns = ", ".join(row) or "<none>"
    raise ResultShapeError(f"missing numeric `count` in returned columns: {columns}")


def row_dict(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    output = {}
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if key in {"center", "distance"}:
            if key == "center":
                value = point_to_solution(value)
            elif isinstance(value, int | float):
                value = round(float(value), 6)
        output[key] = value
    return output


def rows_to_solution_value(
    query_type: str,
    rows: list[dict[str, Any]],
    informative: bool,
) -> Any:
    if query_type == "object_class_inventory":
        return scalar_or_set(
            sorted(
                set(
                    row_values(
                        rows,
                        "class",
                        preferred_keys=["class"],
                        node_keys=["o", "object"],
                    )
                )
            )
        )
    if query_type == "object_class_centers":
        if informative:
            return [
                row_dict(row, ["nodeSymbol", "center", "class"])
                for row in rows
            ]
        return scalar_or_set(
            [
                point_to_solution(center)
                for center in row_values(
                    rows,
                    "center",
                    preferred_keys=["center"],
                    node_keys=["o", "object"],
                )
            ]
        )
    if query_type == "object_class_count":
        return row_count_value(rows)
    if query_type == "top_object_classes_by_count":
        if informative:
            return [
                {"class": row.get("class"), "count": row.get("count")}
                for row in rows
            ]
        return [row.get("class") for row in rows]
    if query_type in {
        "objects_in_room_symbol",
        "objects_in_room_class",
        "objects_in_place_symbol",
        "objects_in_bounding_box",
    }:
        if informative:
            return [row_dict(row, ["object", "center", "class"]) for row in rows]
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["object", "nodeSymbol"],
                node_keys=["o", "object"],
            )
        )
    if query_type == "object_place_lookup":
        if informative:
            return [row_dict(row, ["place", "center"]) for row in rows]
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["place", "nodeSymbol"],
                node_keys=["p", "place"],
            )
        )
    if query_type in {"nearest_object_to_mesh_place", "nearest_object_to_object"}:
        if not rows:
            return set()
        if informative:
            row = dict(rows[0])
            if "distance" in row and isinstance(row["distance"], int | float):
                row["distance"] = round(float(row["distance"]), 6)
            return row
        return row_property(
            rows[0],
            "nodeSymbol",
            preferred_keys=["object", "nodeSymbol"],
            node_keys=["o", "object"],
        )
    if query_type == "object_radius_from_point":
        if informative:
            return [
                row_dict(row, ["object", "class", "center", "distance"])
                for row in rows
            ]
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["object", "nodeSymbol"],
                node_keys=["o", "object"],
            )
        )
    if query_type == "mesh_place_hops":
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["meshPlace", "nodeSymbol"],
                node_keys=["m", "meshPlace"],
            )
        )
    if query_type == "room_adjacency_count":
        if not rows:
            return 0
        if informative:
            return {
                "room": rows[0].get("room"),
                "adjacent_rooms": rows[0].get("adjacent_rooms"),
            }
        return rows[0].get("adjacent_rooms", 0)
    if query_type == "place_hops":
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["place", "nodeSymbol"],
                node_keys=["p", "place"],
            )
        )
    return rows


def rows_to_solution_text(record: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    informative = expected_uses_informative_solution(record.get("solution"))
    value = rows_to_solution_value(record.get("query_type", ""), rows, informative)
    list_style = "set" if informative else "list"
    return rendered_solution(value, list_style=list_style)


def evaluate_records(
    records: list[dict[str, Any]],
    cypher_field: str,
    allow_writes: bool,
    validator: Neo4jValidator,
    policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    evaluated = []
    for record in records:
        cypher = record.get(cypher_field, "")
        expected_solution = record.get("solution")
        record_allows_writes = allow_writes or not bool(record.get("read_only", True))
        static_issues = static_checks(cypher, record_allows_writes, policy=policy)
        is_refusal = is_safe_refusal(cypher)

        execution = {
            "executed": False,
            "executable": False,
            "syntax_error": False,
            "timeout": False,
            "elapsed_ms": None,
            "row_count": 0,
            "result_hash": None,
            "result_shape_error": None,
            "error": None,
        }
        actual_solution = None

        if not static_issues and is_refusal:
            actual_solution = "WRITE_QUERY_NOT_ALLOWED"
            execution["executable"] = True
        elif not static_issues:
            validation = validator.validate(cypher, execute=True)
            rows = validation.get("rows", [])
            execution = {
                "executed": True,
                "executable": bool(validation["executable"]),
                "syntax_error": bool(validation["syntax_error"]),
                "timeout": bool(validation["timeout"]),
                "elapsed_ms": validation["elapsed_ms"],
                "row_count": len(rows),
                "result_hash": result_hash(rows) if validation["executable"] else None,
                "result_shape_error": None,
                "error": validation["error"],
            }
            if validation["executable"]:
                try:
                    actual_solution = rows_to_solution_text(record, rows)
                except ResultShapeError as exc:
                    execution["result_shape_error"] = str(exc)

        solution_match = (
            answer_matches(
                actual_solution,
                expected_solution,
                str(record.get("output_type") or "SLDP"),
            )
            if actual_solution is not None and expected_solution is not None
            else False
        )
        status = "pass"
        if static_issues:
            status = "static_error"
        elif execution["error"]:
            status = "execution_error"
        elif not solution_match:
            status = "solution_mismatch"

        evaluated.append(
            {
                "uid": record.get("uid"),
                "query_type": record.get("query_type"),
                "question": record.get("question"),
                "cypher": cypher,
                "expected_solution": expected_solution,
                "actual_solution": actual_solution,
                "solution_match": solution_match,
                "status": status,
                "static_valid": not static_issues,
                "static_issues": static_issues,
                "refusal": is_refusal,
                **execution,
            }
        )
    return evaluated


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    counters = Counter()
    by_query_type: dict[str, Counter] = {}
    for record in records:
        query_type = record.get("query_type") or "unknown"
        by_query_type.setdefault(query_type, Counter())
        for key in [
            "static_valid",
            "executed",
            "executable",
            "syntax_error",
            "timeout",
            "solution_match",
            "refusal",
        ]:
            if record.get(key):
                counters[key] += 1
                by_query_type[query_type][key] += 1
        if record.get("error"):
            counters["execution_error"] += 1
            by_query_type[query_type]["execution_error"] += 1
        if record.get("status") == "solution_mismatch":
            counters["solution_mismatch"] += 1
            by_query_type[query_type]["solution_mismatch"] += 1
        if record.get("status") == "static_error":
            counters["static_error"] += 1
            by_query_type[query_type]["static_error"] += 1

    return {
        "total": total,
        "metrics": dict(counters),
        "rates": {
            key: (counters[key] / total if total else 0.0)
            for key in [
                "static_valid",
                "executable",
                "solution_match",
                "execution_error",
                "solution_mismatch",
                "static_error",
            ]
        },
        "by_query_type": {
            query_type: {
                "total": sum(1 for record in records if record.get("query_type") == query_type),
                "metrics": dict(counter),
            }
            for query_type, counter in sorted(by_query_type.items())
        },
    }


def build_report(
    records: list[dict[str, Any]],
    *,
    input_path: str,
    scene_path: str,
    cypher_field: str,
) -> dict[str, Any]:
    return {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "input": input_path,
            "scene": scene_path,
            "cypher_field": cypher_field,
        },
        "summary": build_summary(records),
        "records": records,
    }


def write_report(report: dict[str, Any], output: str | Path) -> None:
    output_path = ensure_parent(output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_report_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    total = summary["total"]
    metrics = Counter(summary.get("metrics", {}))

    console = Console()
    table = Table(title="Cypher Validation Report")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")
    for key, label in [
        ("static_valid", "Static valid"),
        ("executable", "Executed/refusal OK"),
        ("solution_match", "Solution match"),
        ("execution_error", "Execution errors"),
        ("solution_mismatch", "Solution mismatches"),
        ("static_error", "Static errors"),
        ("timeout", "Timeouts"),
        ("syntax_error", "Syntax errors"),
    ]:
        count = metrics[key]
        rate = count / total if total else 0.0
        table.add_row(label, str(count), f"{rate:.2%}")
    console.print(table)

    query_table = Table(title="By Query Type")
    query_table.add_column("Query Type")
    query_table.add_column("N", justify="right")
    query_table.add_column("Exec OK", justify="right")
    query_table.add_column("Match", justify="right")
    query_table.add_column("Errors", justify="right")
    for query_type, payload in summary.get("by_query_type", {}).items():
        n = payload["total"]
        query_metrics = Counter(payload.get("metrics", {}))
        errors = (
            query_metrics["execution_error"]
            + query_metrics["solution_mismatch"]
            + query_metrics["static_error"]
        )
        query_table.add_row(
            query_type,
            str(n),
            f"{query_metrics['executable'] / n:.2%}" if n else "0.00%",
            f"{query_metrics['solution_match'] / n:.2%}" if n else "0.00%",
            str(errors),
        )
    console.print(query_table)
