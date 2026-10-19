from __future__ import annotations

import itertools
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


def build_grounding(value, contract, variables):
    expected = {}
    for name, role in contract["result"]["roles"].items():
        source = role.get("oracle_source", "value")
        if source == "value":
            field_value = value
        elif source.startswith("variable."):
            field_value = variables.get(source.removeprefix("variable."))
        elif isinstance(value, dict):
            field_value = value.get(source)
        else:
            field_value = None
        values = _ordered_values(field_value)
        expected[name] = sorted(set(values)) if role.get("cardinality", "set") == "set" else values
    return expected


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
