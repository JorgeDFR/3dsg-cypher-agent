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

from cypher_3dsg_agent_distillation.core.config import ensure_parent
from cypher_3dsg_agent_distillation.cypher.pddl_goals import validate_grounding_rows
from cypher_3dsg_agent_distillation.cypher.templates import format_solution_field
from cypher_3dsg_agent_distillation.graph.heracles_validation import answer_matches_with_heracles
from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
    Neo4jValidator,
    is_safe_refusal,
    result_hash,
    static_checks,
)

NODE_SYMBOL_RE = re.compile(r"^[A-Za-z]\d+$")


class ResultShapeError(ValueError):
    pass


def is_number_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def is_coordinate_sequence(value: Any) -> bool:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return False
    return all(is_number_like(value[index]) for index in range(3))


def point_coordinates(value: Any) -> tuple[Any, Any, Any]:
    if isinstance(value, Mapping):
        if all(axis in value for axis in ("x", "y", "z")):
            return value["x"], value["y"], value["z"]
        raise ResultShapeError(f"expected point with x/y/z coordinates, got {value!r}")

    try:
        return value.x, value.y, value.z
    except AttributeError:
        pass

    if is_coordinate_sequence(value):
        return value[0], value[1], value[2]

    raise ResultShapeError(f"expected point value, got {type(value).__name__}")


def is_point_like(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("POINT(")
    try:
        x, y, z = point_coordinates(value)
    except ResultShapeError:
        return False
    return is_number_like(x) and is_number_like(y) and is_number_like(z)


def is_aggregate_result_value(value: Any) -> bool:
    if isinstance(value, str):
        return False
    if not isinstance(value, list | tuple | set):
        return False
    return not is_coordinate_sequence(value)


def point_to_solution(value: Any) -> str:
    if isinstance(value, str) and value.startswith("POINT("):
        return value
    x, y, z = point_coordinates(value)
    if not all(is_number_like(axis) for axis in (x, y, z)):
        raise ResultShapeError(f"expected numeric point coordinates, got {value!r}")
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


def answer_matches(
    actual: Any,
    expected: Any,
    output_type: str = "SLDP",
    *,
    answer_validation: dict[str, Any] | None = None,
) -> bool:
    valid_format, correct = answer_matches_with_heracles(
        str(actual),
        str(expected),
        output_type,
        answer_validation=answer_validation,
    )
    return valid_format and correct


def rendered_solution(value: Any, solution_formats: Any = None) -> str:
    return str(format_solution_field(value, solution_formats=solution_formats)).strip()


def solution_formats_for_record(record: dict[str, Any]) -> Any:
    declared = record.get("solution_formats") or record.get("solution_format")
    if declared:
        return declared

    expected = next(
        (
            record[key]
            for key in ("expected_solution", "real_solution", "solution")
            if record.get(key) is not None
        ),
        None,
    )
    text = str(expected).strip() if expected is not None else ""
    if text.startswith("<") and text.endswith(">"):
        return ["set[str]"]
    if text.startswith("[") and text.endswith("]"):
        return ["list[str]"]
    return None


def scalar_or_set(values: list[Any]) -> Any:
    if not values:
        return set()
    if len(values) == 1:
        return values[0]
    return set(values)


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


def plural_column_key(key: str) -> str:
    if key.endswith("s"):
        return key
    if key.endswith(("ch", "sh", "x", "z")) or key.endswith("class"):
        return f"{key}es"
    return f"{key}s"


def collection_items(value: Any) -> list[Any]:
    return list(value) if is_aggregate_result_value(value) else [value]


def symbol_values(value: Any) -> list[str] | None:
    items = collection_items(value)
    if all(isinstance(item, str) and NODE_SYMBOL_RE.fullmatch(item) for item in items):
        return items
    return None


def class_values(value: Any) -> list[str] | None:
    items = collection_items(value)
    if all(isinstance(item, str) and not NODE_SYMBOL_RE.fullmatch(item) for item in items):
        return items
    return None


def point_values(value: Any) -> list[Any] | None:
    items = collection_items(value)
    if all(is_point_like(item) for item in items):
        return items
    return None


def inferred_row_property(row: dict[str, Any], property_name: str) -> Any:
    if property_name == "nodeSymbol":
        collection_matches = []
        for value in row.values():
            if isinstance(value, str) and NODE_SYMBOL_RE.fullmatch(value):
                return value
            values = symbol_values(value)
            if values is not None:
                collection_matches.append(values)
            nested_value = node_property(value, property_name)
            if isinstance(nested_value, str) and NODE_SYMBOL_RE.fullmatch(nested_value):
                return nested_value
        if len(collection_matches) == 1:
            return collection_matches[0]

    if property_name == "class":
        matches = []
        for value in row.values():
            values = class_values(value)
            if values is not None:
                matches.append(values)
        if len(matches) == 1:
            return matches[0] if len(matches[0]) != 1 else matches[0][0]

    if property_name == "center":
        matches = []
        for value in row.values():
            values = point_values(value)
            if values is not None:
                matches.append(values)
                continue
            nested_value = node_property(value, property_name)
            if nested_value is not None:
                nested_values = point_values(nested_value)
                if nested_values is not None:
                    matches.append(nested_values)
        if len(matches) == 1:
            return matches[0] if len(matches[0]) != 1 else matches[0][0]

    return None


def row_property(
    row: dict[str, Any],
    property_name: str,
    preferred_keys: list[str] | None = None,
    node_keys: list[str] | None = None,
) -> Any:
    for key in preferred_keys or []:
        for column_key in (key, plural_column_key(key)):
            value = row.get(column_key)
            if value is not None:
                return value

    for column_key in (property_name, plural_column_key(property_name)):
        value = row.get(column_key)
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

    inferred = inferred_row_property(row, property_name)
    if inferred is not None:
        return inferred

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
    *,
    flatten_collections: bool = False,
) -> list[Any]:
    values = []
    for row in rows:
        value = row_property(row, property_name, preferred_keys, node_keys)
        if value is None:
            columns = ", ".join(row) or "<none>"
            raise ResultShapeError(
                f"missing `{property_name}` in returned columns: {columns}"
            )
        if flatten_collections and is_aggregate_result_value(value):
            values.extend(value)
        else:
            values.append(value)
    return values


def row_count_value(
    rows: list[dict[str, Any]],
    preferred_keys: list[str] | None = None,
) -> int | float:
    if not rows:
        return 0
    row = rows[0]
    for key in preferred_keys or []:
        value = row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
    value = row.get("count")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    for key, value in row.items():
        if key.lower().startswith("count(") and isinstance(value, int | float):
            return value
    for key, value in row.items():
        normalized_key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if (
            (normalized_key == "count" or normalized_key.endswith("_count"))
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        ):
            return value
    if len(row) == 1:
        value = next(iter(row.values()))
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
    columns = ", ".join(row) or "<none>"
    raise ResultShapeError(f"missing numeric `count` in returned columns: {columns}")


def rows_indicate_no_match(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return True

    count_values = [
        value
        for row in rows
        for key, value in row.items()
        if (
            key.lower().startswith("count(")
            or re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_").endswith("count")
        )
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ]
    if count_values and all(value == 0 for value in count_values):
        return True

    def empty_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, bool):
            return False
        if isinstance(value, int | float):
            return value == 0
        if isinstance(value, list | tuple | set | dict):
            return len(value) == 0
        return False

    return all(empty_value(value) for row in rows for value in row.values())


def rows_to_solution_value(
    query_type: str,
    rows: list[dict[str, Any]],
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
                        flatten_collections=True,
                    )
                )
            )
        )
    if query_type in {"object_class_centers", "object_center_by_symbol"}:
        return scalar_or_set(
            [
                point_to_solution(center)
                for center in row_values(
                    rows,
                    "center",
                    preferred_keys=["center"],
                    node_keys=["o", "object"],
                    flatten_collections=True,
                )
            ]
        )
    if query_type in {
        "object_class_count",
        "object_total",
        "room_total",
        "combined_place_total",
        "room_object_count",
        "place_object_count",
    }:
        return row_count_value(rows)
    if query_type == "object_class_by_symbol":
        if not rows:
            return ""
        return row_property(rows[0], "class", preferred_keys=["class"])
    if query_type == "top_object_classes_by_count":
        return row_values(
            rows,
            "class",
            preferred_keys=["class"],
            node_keys=["o", "object"],
            flatten_collections=True,
        )
    if query_type in {
        "objects_in_room_symbol",
        "objects_in_room_class",
        "objects_in_place_symbol",
        "objects_in_bounding_box",
        "objects_within_radius_of_object",
    }:
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["object", "nodeSymbol"],
                node_keys=["o", "object"],
                flatten_collections=True,
            )
        )
    if query_type == "object_place_lookup":
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["place", "nodeSymbol"],
                node_keys=["p", "place"],
                flatten_collections=True,
            )
        )
    if query_type in {
        "nearest_object_to_mesh_place",
        "nearest_object_to_object",
        "nearest_object_any_class",
        "farthest_object_from_object",
        "nearest_object_different_class",
    }:
        if not rows:
            return set()
        return row_property(
            rows[0],
            "nodeSymbol",
            preferred_keys=["object", "nodeSymbol"],
            node_keys=["o", "object"],
        )
    if query_type == "object_radius_from_point":
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["object", "nodeSymbol"],
                node_keys=["o", "object"],
                flatten_collections=True,
            )
        )
    if query_type == "mesh_place_hops":
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["meshPlace", "nodeSymbol"],
                node_keys=["m", "meshPlace"],
                flatten_collections=True,
            )
        )
    if query_type == "room_adjacency_count":
        return row_count_value(
            rows,
            preferred_keys=[
                "adjacent_rooms",
                "neighbor_count",
                "neighbour_count",
                "adjacency_count",
                "room_count",
            ],
        )
    if query_type in {
        "object_classes_in_room_symbol",
        "object_classes_in_place_symbol",
    }:
        return scalar_or_set(
            row_values(
                rows,
                "class",
                preferred_keys=["class"],
                flatten_collections=True,
            )
        )
    if query_type in {
        "closest_cross_class_object_pairs",
        "closest_room_pairs",
    }:
        return scalar_or_set(
            row_values(
                rows,
                "pair",
                preferred_keys=["pair"],
                flatten_collections=True,
            )
        )
    if query_type in {
        "adjacent_rooms",
        "most_connected_rooms",
        "least_connected_rooms",
    }:
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["room", "nodeSymbol"],
                flatten_collections=True,
            )
        )
    if query_type in {
        "adjacent_places",
        "most_connected_places",
        "least_connected_places",
    }:
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["place", "nodeSymbol"],
                flatten_collections=True,
            )
        )
    if query_type == "nearest_place_to_object":
        if not rows:
            return ""
        return row_property(
            rows[0],
            "nodeSymbol",
            preferred_keys=["place", "nodeSymbol"],
        )
    if query_type == "nearest_room_to_object":
        if not rows:
            return ""
        return row_property(
            rows[0],
            "nodeSymbol",
            preferred_keys=["room", "nodeSymbol"],
        )
    if query_type == "place_hops":
        return scalar_or_set(
            row_values(
                rows,
                "nodeSymbol",
                preferred_keys=["place", "nodeSymbol"],
                node_keys=["p", "place"],
                flatten_collections=True,
            )
        )
    return rows


def rows_to_solution_text(record: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if "invalid_input" in (record.get("tags") or []) and rows_indicate_no_match(rows):
        expected = next(
            (
                record[key]
                for key in ("expected_solution", "real_solution", "solution")
                if key in record and record[key] is not None
            ),
            "",
        )
        return str(expected).strip()
    value = rows_to_solution_value(record.get("query_type", ""), rows)
    solution_formats = solution_formats_for_record(record)
    return rendered_solution(value, solution_formats=solution_formats)


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
                if str(record.get("task_family") or "cypher_qa") == "cypher_pddl":
                    grounding_match, actual_solution, _actual_grounding = (
                        validate_grounding_rows(rows, record.get("grounding"))
                    )
                    execution["grounding_match"] = grounding_match
                else:
                    try:
                        actual_solution = rows_to_solution_text(record, rows)
                    except ResultShapeError as exc:
                        execution["result_shape_error"] = str(exc)

        if str(record.get("task_family") or "cypher_qa") == "cypher_pddl":
            solution_match = bool(execution.get("grounding_match"))
        else:
            solution_match = (
                answer_matches(
                    actual_solution,
                    expected_solution,
                    str(record.get("output_type") or "SLDP"),
                    answer_validation=record.get("answer_validation"),
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
