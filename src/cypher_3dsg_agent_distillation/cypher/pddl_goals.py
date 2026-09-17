from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterable
from typing import Any


def _ordered_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, set):
        values = sorted(value, key=str)
    elif isinstance(value, list | tuple):
        values = list(value)
    else:
        values = [value]
    return [str(item) for item in values if item is not None and str(item).strip()]


def build_grounding(
    value: Any,
    grounding_spec: dict[str, Any],
    variables: dict[str, Any],
) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}
    for field_name, field_spec in (grounding_spec.get("fields") or {}).items():
        source = str(field_spec.get("source", "value"))
        if source == "value":
            field_value = value
        elif source.startswith("variable."):
            field_value = variables.get(source.removeprefix("variable."))
        elif isinstance(value, dict):
            field_value = value.get(source)
        else:
            field_value = None
        values = _ordered_values(field_value)
        if str(field_spec.get("cardinality") or "set") == "set":
            values = sorted(set(values))
        expected[str(field_name)] = values
    return expected


def grounding_contract(
    value: Any,
    grounding_spec: dict[str, Any],
    variables: dict[str, Any],
) -> dict[str, Any]:
    fields = {}
    for field_name, field_spec in (grounding_spec.get("fields") or {}).items():
        fields[str(field_name)] = {
            key: field_spec[key]
            for key in ("aliases", "symbol_type", "cardinality")
            if key in field_spec
        }
    return {
        "expected": build_grounding(value, grounding_spec, variables),
        "fields": fields,
        "match": str(grounding_spec.get("match", "exact")),
    }


def grounding_is_non_empty(grounding: dict[str, Any]) -> bool:
    expected = grounding.get("expected") or {}
    return bool(expected) and all(bool(values) for values in expected.values())


def _argument_values(argument: Any, grounding: dict[str, list[str]]) -> list[str]:
    if isinstance(argument, str):
        return [argument]
    if not isinstance(argument, dict):
        raise TypeError("PDDL goal arguments must be strings or mappings.")
    if "literal" in argument:
        return [str(argument["literal"])]
    if "result" in argument:
        return list(grounding.get(str(argument["result"]), []))
    raise KeyError("PDDL goal argument requires `literal` or `result`.")


def render_goal_spec(spec: dict[str, Any], grounding: dict[str, list[str]]) -> str | None:
    if "fact" in spec:
        predicate = str(spec["fact"])
        argument_values = [
            _argument_values(argument, grounding) for argument in spec.get("arguments", [])
        ]
        if any(not values for values in argument_values):
            return None
        combinations = list(itertools.product(*argument_values)) if argument_values else [()]
        facts = [
            f"({predicate}{' ' if values else ''}{' '.join(values)})" for values in combinations
        ]
        if bool(spec.get("negate_each")):
            facts = [f"(not {fact})" for fact in facts]
        if len(facts) == 1:
            return facts[0]
        operator = str(spec.get("multiple") or "or")
        if operator not in {"and", "or"}:
            raise ValueError("PDDL `multiple` must be `and` or `or`.")
        return f"({operator} {' '.join(facts)})"

    operator = str(spec.get("operator") or "")
    if operator not in {"and", "or", "not"}:
        raise ValueError("PDDL goal specification requires `fact` or a supported `operator`.")
    rendered_clauses = [render_goal_spec(clause, grounding) for clause in spec.get("clauses", [])]
    if operator == "not" and len(rendered_clauses) != 1:
        raise ValueError("PDDL `not` requires exactly one clause.")
    if operator in {"and", "or"} and not rendered_clauses:
        raise ValueError(f"PDDL `{operator}` requires at least one clause.")
    if operator in {"and", "not"} and any(clause is None for clause in rendered_clauses):
        return None

    clauses = [clause for clause in rendered_clauses if clause is not None]
    if not clauses:
        return None
    if operator == "or" and len(clauses) == 1:
        return clauses[0]
    return f"({operator} {' '.join(clauses)})"


def render_pddl_solution(
    goal_spec: dict[str, Any],
    grounding: dict[str, Any],
) -> str | None:
    expected = grounding.get("expected") or {}
    return render_goal_spec(goal_spec, expected)


SYMBOL_PATTERNS = {
    "Object": re.compile(r"^O\d+$"),
    "Place": re.compile(r"^p\d+$"),
    "MeshPlace": re.compile(r"^P\d+$"),
    "PlaceOrMeshPlace": re.compile(r"^[pP]\d+$"),
    "Room": re.compile(r"^R\d+$"),
    "Building": re.compile(r"^B\d+$"),
}


def _symbols_in_value(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _symbols_in_value(child)
    elif isinstance(value, list | tuple | set):
        for child in value:
            yield from _symbols_in_value(child)


def _grounding_values_in_value(value: Any) -> Iterable[str]:
    if isinstance(value, str | int | float):
        yield str(value)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _grounding_values_in_value(child)
    elif isinstance(value, list | tuple | set):
        for child in value:
            yield from _grounding_values_in_value(child)


def _inferred_symbols_for_field(
    row: dict[str, Any],
    field_name: str,
    fields: dict[str, Any],
) -> list[str]:
    """Infer an unaliased grounding field only from an unambiguous symbol pattern.

    A row such as ``{"waypoint": "O1", "destination": "p2"}`` carries enough
    type information to populate distinct Object and Place fields. Numeric values
    such as counts have no symbol pattern and are never inferred. Likewise, when
    two contract fields accept the same symbol (for example, two Object fields),
    aliases remain required so that a symbol is not assigned to both fields.
    """
    inferred = []
    for value in row.values():
        for symbol in _symbols_in_value(value):
            matching_fields = []
            for candidate_name, candidate_spec in fields.items():
                pattern = SYMBOL_PATTERNS.get(str(candidate_spec.get("symbol_type") or ""))
                if pattern is not None and pattern.fullmatch(symbol):
                    matching_fields.append(str(candidate_name))
            if matching_fields == [field_name]:
                inferred.append(symbol)
    return inferred


def grounding_from_rows(
    rows: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, list[str]]:
    extracted: dict[str, list[str]] = {}
    fields = contract.get("fields") or {}
    for field_name, field_spec in fields.items():
        field_name = str(field_name)
        aliases = [str(alias) for alias in field_spec.get("aliases") or [field_name]]
        pattern = SYMBOL_PATTERNS.get(str(field_spec.get("symbol_type") or ""))
        values = []
        for row in rows:
            matched = False
            for alias in aliases:
                if alias not in row or row[alias] is None:
                    continue
                values.extend(_grounding_values_in_value(row[alias]))
                matched = True
                break
            if not matched:
                values.extend(_inferred_symbols_for_field(row, field_name, fields))
        if pattern is not None:
            values = [value for value in values if pattern.fullmatch(value)]
        cardinality = str(field_spec.get("cardinality") or "set")
        if cardinality == "set":
            values = sorted(set(values))
        extracted[str(field_name)] = values
    return extracted


def _semantic_grounding_from_rows(
    rows: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, list[str]] | None:
    """Map result columns to grounding roles by their values, not their aliases."""
    fields = contract.get("fields") or {}
    expected = {
        str(key): _ordered_values(value) for key, value in (contract.get("expected") or {}).items()
    }
    match_mode = str(contract.get("match") or "exact")
    column_values: dict[str, list[str]] = {}
    for row in rows:
        for column, value in row.items():
            column_values.setdefault(str(column), []).extend(_grounding_values_in_value(value))

    candidates: dict[str, list[str]] = {}
    for field_name, field_spec in fields.items():
        field_name = str(field_name)
        expected_values = set(expected.get(field_name, []))
        if not expected_values:
            candidates[field_name] = []
            continue
        pattern = SYMBOL_PATTERNS.get(str(field_spec.get("symbol_type") or ""))
        matching_columns = []
        for column, raw_values in column_values.items():
            values = {value for value in raw_values if pattern is None or pattern.fullmatch(value)}
            matches = (
                expected_values.issubset(values)
                if match_mode == "subset"
                else values == expected_values
            )
            if matches:
                matching_columns.append(column)
        if not matching_columns:
            return None
        candidates[field_name] = matching_columns

    assignments: dict[str, str] = {}

    def assign(field_names: list[str], used: dict[str, str]) -> bool:
        if not field_names:
            return True
        field_name = min(field_names, key=lambda name: len(candidates[name]))
        remaining = [name for name in field_names if name != field_name]
        for column in candidates[field_name]:
            previous_field = used.get(column)
            if previous_field is not None and set(expected[previous_field]) != set(
                expected[field_name]
            ):
                continue
            assignments[field_name] = column
            used[column] = field_name
            if assign(remaining, used):
                return True
            assignments.pop(field_name, None)
            if previous_field is None:
                used.pop(column, None)
            else:
                used[column] = previous_field
        return False

    non_empty_fields = [name for name in candidates if expected.get(name)]
    if not assign(non_empty_fields, {}):
        return None

    return {
        field_name: (
            sorted(
                {
                    value
                    for value in column_values[assignments[field_name]]
                    if (
                        SYMBOL_PATTERNS.get(str(fields[field_name].get("symbol_type") or ""))
                        is None
                        or SYMBOL_PATTERNS[
                            str(fields[field_name].get("symbol_type") or "")
                        ].fullmatch(value)
                    )
                }
            )
            if field_name in assignments
            else []
        )
        for field_name in fields
    }


def validate_grounding_rows(
    rows: list[dict[str, Any]],
    contract: dict[str, Any] | None,
) -> tuple[bool, str | None, dict[str, list[str]] | None]:
    if not contract:
        return True, None, None
    expected = {
        str(key): _ordered_values(value) for key, value in (contract.get("expected") or {}).items()
    }
    actual = grounding_from_rows(rows, contract)
    match_mode = str(contract.get("match") or "exact")
    if match_mode == "subset":
        matches = all(
            set(values).issubset(set(actual.get(key, []))) for key, values in expected.items()
        )
    elif match_mode == "exact":
        matches = all(set(actual.get(key, [])) == set(values) for key, values in expected.items())
    else:
        raise ValueError(f"Unsupported grounding match mode `{match_mode}`.")
    expected_is_empty = bool(expected) and all(not values for values in expected.values())
    if matches and expected_is_empty:
        patterns = {
            pattern
            for field_spec in (contract.get("fields") or {}).values()
            if (pattern := SYMBOL_PATTERNS.get(str(field_spec.get("symbol_type") or "")))
            is not None
        }
        matches = not any(
            pattern.fullmatch(symbol)
            for row in rows
            for symbol in _symbols_in_value(row)
            for pattern in patterns
        )
    if not matches and not expected_is_empty:
        semantic_actual = _semantic_grounding_from_rows(rows, contract)
        if semantic_actual is not None:
            actual = semantic_actual
            if match_mode == "subset":
                matches = all(
                    set(values).issubset(set(actual.get(key, [])))
                    for key, values in expected.items()
                )
            else:
                matches = all(
                    set(actual.get(key, [])) == set(values) for key, values in expected.items()
                )

    actual_is_empty = bool(actual) and all(not values for values in actual.values())
    rendered = (
        ""
        if expected_is_empty and actual_is_empty
        else json.dumps(actual, ensure_ascii=False, sort_keys=True)
    )
    return matches, rendered, actual
