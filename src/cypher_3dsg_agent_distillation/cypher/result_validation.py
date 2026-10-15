"""Decode actual results before looking at gold values; compare with typed contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any

from cypher_3dsg_agent_distillation.cypher.query_provenance import (
    canonical_key,
    projections,
    semantic_evidence,
)
from cypher_3dsg_agent_distillation.cypher.validation_config import (
    ValidationContract,
    ValueSpec,
    contract_for_record,
)

SYMBOLS = {
    "Object": r"O\d+",
    "Place": r"p\d+",
    "MeshPlace": r"P\d+",
    "Room": r"R\d+",
    "Building": r"B\d+",
}


class DecodeError(ValueError):
    pass


class MappingUnresolved(DecodeError):
    pass


@dataclass
class DecodedResult:
    status: str
    actual: Any = None
    bindings: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    candidates: dict[str, list[str]] = field(default_factory=dict)


def _atom(value, spec: ValueSpec, forms):
    if isinstance(value, Mapping):
        native = value.get("__neo4j_type__")
        if native == "node":
            if "nodes" not in forms:
                raise DecodeError("Full nodes are not accepted")
            if spec.layers and not set(spec.layers).intersection(value.get("labels", [])):
                raise DecodeError("Returned node has incompatible labels")
            value = value["properties"]
        if "nodes" not in forms and "maps" not in forms:
            raise DecodeError("Structured values are not accepted by this contract")
        if spec.type == "point" and all(k in value for k in ("x", "y", "z")):
            value = [value[k] for k in ("x", "y", "z")]
        elif spec.property and spec.property in value:
            value = value[spec.property]
        else:
            raise DecodeError(f"Missing configured property {spec.property}")
    if spec.type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise DecodeError("Expected an integer")
    elif spec.type == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise DecodeError("Expected a finite number")
    elif spec.type == "point":
        if hasattr(value, "x"):
            value = [value.x, value.y, value.z]
        if isinstance(value, str) and value.startswith("POINT("):
            try:
                value = [float(v) for v in value[6:-1].split()]
            except ValueError as exc:
                raise DecodeError("Invalid point") from exc
        if (
            not isinstance(value, list | tuple)
            or len(value) != 3
            or any(
                isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
                for v in value
            )
        ):
            raise DecodeError("Expected a finite 3D point")
        value = tuple(float(v) for v in value)
    elif spec.type == "node_symbol":
        patterns = [SYMBOLS[layer] for layer in spec.layers] if spec.layers else SYMBOLS.values()
        if not isinstance(value, str) or not any(re.fullmatch(p, value) for p in patterns):
            raise DecodeError("Expected a case-sensitive node symbol of the declared layer")
    elif not isinstance(value, str):
        raise DecodeError("Expected a string")
    return value


def _values(value, spec, forms):
    for key in spec.path:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise DecodeError("Missing explicitly configured map/list path") from exc
    if value is None:
        if spec.nulls == "empty_set":
            return []
        raise DecodeError("Null in a required result role")
    if spec.type == "point":
        try:
            return [_atom(value, spec, forms)]
        except DecodeError:
            pass
    if isinstance(value, list | tuple | set):
        if "collected_columns" not in forms:
            raise DecodeError("Collected values are not accepted")
        # Do not filter invalid members: every member must satisfy the contract.
        return [_atom(item, spec, forms) for item in value]
    if not isinstance(value, Mapping) and "scalar_columns" not in forms:
        raise DecodeError("Scalar columns are not accepted")
    return [_atom(value, spec, forms)]


def _lineage_atom(lineage):
    return lineage.get("item", {}) if lineage.get("kind") == "collection" else lineage


def _origin_matches(spec, lineage):
    lineage = _lineage_atom(lineage)
    for part in spec.path:
        lineage = (
            lineage.get("entries", {}).get(part, {"kind": "unknown"})
            if isinstance(part, str)
            else lineage.get("item", {"kind": "unknown"})
        )
    if lineage.get("kind") == "map" and spec.property:
        lineage = lineage.get("entries", {}).get(spec.property, {"kind": "unknown"})
    if lineage.get("kind") == "unknown":
        return None
    if (
        spec.property
        and lineage.get("kind") == "property"
        and (lineage.get("property") != spec.property)
    ):
        return False
    if spec.origin and any(lineage.get("origin", {}).get(k) != v for k, v in spec.origin.items()):
        return False
    if (
        spec.layers
        and lineage.get("layers")
        and not set(spec.layers).intersection(lineage["layers"])
    ):
        return False
    return True


def _unique(values):
    unique = {canonical_key(v): v for v in values}
    return [unique[key] for key in sorted(unique)]


def decode_result(query, rows, contract, columns=None):
    """This API intentionally has no expected_solution/expected_grounding argument."""
    c = ValidationContract.model_validate(contract)
    spec = c.result
    candidates = {}
    try:
        if not isinstance(rows, list | tuple) or any(not isinstance(row, Mapping) for row in rows):
            raise DecodeError("Result rows must be mappings")
        if len(rows) < spec.row_count.min or (
            spec.row_count.max is not None and len(rows) > spec.row_count.max
        ):
            raise DecodeError("Returned row count violates contract")
        keys = list(columns or dict.fromkeys(k for row in rows for k in row))
        try:
            lineage = projections(query)
        except (ValueError, KeyError, TypeError):
            lineage = []
        if not keys and lineage:
            keys = [p["column"] for p in lineage]
        if len(keys) != len(set(keys)) or len(lineage) != len({p["column"] for p in lineage}):
            raise DecodeError("Duplicate returned column names lose projection information")
        by_column = {item["column"]: item for item in lineage}
        roles = spec.roles or {
            "value": ValueSpec(
                type=spec.type, layers=spec.layers, property=spec.property, origin=spec.origin
            )
            if spec.element is None
            else spec.element
        }
        candidates, data = {}, {}
        for name, role in roles.items():
            if any(not isinstance(layer, str) for layer in role.layers):
                raise DecodeError("Resolve controlled layer bindings before decoding")
            if role.position is not None and role.position >= len(keys) and rows:
                raise DecodeError("Configured column position is outside the returned columns")
            selected = role.column or (
                keys[role.position]
                if role.position is not None and role.position < len(keys)
                else None
            )
            # Configured semantic keys are authoritative, not hints to be swapped.
            if selected is None and name != "value" and name in keys:
                selected = name
            possible = []
            for key in keys:
                if selected is not None and key != selected:
                    continue
                try:
                    values = [_values(row[key], role, spec.accepted_forms) for row in rows]
                except KeyError:
                    continue
                except DecodeError:
                    if selected == key:
                        raise
                    continue
                origin = _origin_matches(role, by_column.get(key, {"kind": "unknown"}))
                if origin is False:
                    if selected == key:
                        raise DecodeError(f"Projection origin conflicts with role {name}")
                    continue
                if role.origin and origin is None and selected is None and rows:
                    continue
                possible.append(key)
                data[name, key] = values
            # Empty rows carry no answer values; roles still cannot invent nonempty results.
            if not rows and not keys:
                possible = [f"empty:{name}"]
                data[name, possible[0]] = []
            if not possible:
                if role.origin and not lineage and selected is None and rows:
                    raise MappingUnresolved(f"Query provenance needed for role {name}")
                raise DecodeError(f"No compatible projection for role {name}")
            candidates[name] = possible
        # All admissible assignments must agree; never choose by gold-value overlap.
        assignments = []
        names = list(roles)
        if math.prod(len(candidates[n]) for n in names) > 256:
            raise MappingUnresolved("Too many admissible role assignments")
        for keys_for_roles in product(*(candidates[n] for n in names)):
            assignment = dict(zip(names, keys_for_roles, strict=True))
            valid = True
            for i, name in enumerate(names):
                for other in names[:i]:
                    independent_paths = (
                        roles[name].path
                        and roles[other].path
                        and roles[name].path != roles[other].path
                    )
                    if (
                        assignment[name] == assignment[other]
                        and not independent_paths
                        and (roles[name].origin != roles[other].origin or not roles[name].origin)
                    ):
                        valid = False
            if not valid:
                continue
            unused = set(keys) - set(keys_for_roles)
            if unused - spec.auxiliary.keys():
                continue
            for key in unused:
                for row in rows:
                    _values(row[key], spec.auxiliary[key], spec.accepted_forms)
            assignments.append(assignment)
        if not assignments:
            raise DecodeError("Missing roles or undeclared extra result columns")

        def materialize(assignment):
            values = {
                n: [v for row_values in data[n, key] for v in row_values]
                for n, key in assignment.items()
            }
            if spec.kind == "role_sets":
                return {
                    n: _unique(v) if roles[n].cardinality == "set" else v for n, v in values.items()
                }
            if spec.kind == "tuples":
                pairs = []
                for index in range(len(rows)):
                    parts = [data[n, assignment[n]][index] for n in names]
                    if any(len(part) != 1 for part in parts):
                        raise DecodeError("Tuple fields require one value per row")
                    pair = [part[0] for part in parts]
                    pairs.append(sorted(pair, key=canonical_key) if spec.unordered_tuple else pair)
                return _unique(pairs)
            flat = values["value"]
            if spec.kind == "scalar":
                if not flat:
                    return None
                if len(flat) != 1:
                    raise DecodeError("Scalar result requires exactly one value")
                return flat[0]
            if spec.kind == "set":
                return _unique(flat)
            if spec.kind == "flexible":
                return flat[0] if len(flat) == 1 else _unique(flat)
            return flat

        interpretations = [(materialize(a), a) for a in assignments]
        if len({canonical_key(v) for v, _ in interpretations}) != 1:
            raise MappingUnresolved("Multiple materially different result interpretations")
        actual, bindings = interpretations[0]
        return DecodedResult("pass", actual, bindings, candidates=candidates)
    except MappingUnresolved as exc:
        return DecodedResult("mapping_unresolved", issues=[str(exc)], candidates=candidates)
    except DecodeError as exc:
        return DecodedResult("invalid_result_shape", issues=[str(exc)], candidates=candidates)


def _matched_pairs(a, b, equal):
    """Maximum one-to-one matching, also used for missing/extra diagnostics."""
    matched = {}

    def augment(i, seen):
        for j, value in enumerate(b):
            if j in seen or not equal(a[i], value):
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(a)):
        augment(i, set())
    return matched


def _matching(a, b, equal):
    return len(a) == len(b) and len(_matched_pairs(a, b, equal)) == len(a)


def _top_k_matches(parsed, rules):
    if not isinstance(parsed, tuple) or parsed[0] != "set":
        return False
    values = parsed[1:]
    scores, limit = rules.get("scores"), rules.get("limit")
    if not isinstance(scores, dict) or type(limit) is not int or limit < 0:
        return False
    if any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
        return False
    size = min(limit, len(scores))
    if len(values) != size or any(v not in scores for v in values):
        return False
    if not size:
        return True
    cutoff = sorted(scores.values(), reverse=True)[size - 1]
    required = {v for v, score in scores.items() if score > cutoff}
    allowed = {v for v, score in scores.items() if score >= cutoff}
    return required <= set(values) <= allowed


def sldp_equal(a, b, comparison):
    if isinstance(a, tuple) and isinstance(b, tuple):
        if a[0] != b[0]:
            return False
        if a[0] == "point":
            return len(a) == len(b) and all(
                abs(x - y) <= comparison.point_abs_tolerance
                for x, y in zip(a[1:], b[1:], strict=True)
            )
        if a[0] == "set":
            return _matching(
                _unique(a[1:]), _unique(b[1:]), lambda x, y: sldp_equal(x, y, comparison)
            )
        return len(a) == len(b) and all(
            sldp_equal(x, y, comparison) for x, y in zip(a[1:], b[1:], strict=True)
        )
    if isinstance(a, float) and isinstance(b, float):
        return (
            math.isfinite(a) and math.isfinite(b) and abs(a - b) <= comparison.numeric_abs_tolerance
        )
    return type(a) is type(b) and a == b


def _answer_value(parsed, c):
    """Validate parsed gold and generated values against result types, not just notation."""
    if parsed is None:
        if c.answer.empty != "":
            raise DecodeError("Empty text is not the configured empty answer")
        return

    def atom(value, spec):
        if spec.type == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if spec.type == "point" and isinstance(value, tuple) and value[0] == "point":
            value = value[1:]
        return _atom(value, spec, ["scalar_columns", "maps", "nodes"])

    spec = c.result.element or ValueSpec(type=c.result.type, layers=c.result.layers)
    if c.result.kind == "tuples":
        for pair in parsed[1:]:
            if (
                not isinstance(pair, tuple)
                or pair[0] != "list"
                or len(pair) - 1 != len(c.result.roles)
            ):
                raise DecodeError("Expected a tuple relation")
            for value, role in zip(pair[1:], c.result.roles.values(), strict=True):
                atom(value, role)
    elif isinstance(parsed, tuple) and parsed[0] in {"set", "list"}:
        for value in parsed[1:]:
            atom(value, spec)
    else:
        atom(parsed, spec)


def compare_answers(actual, expected, contract, answer_validation=None):
    from cypher_3dsg_agent_distillation.graph.heracles_validation import (
        answer_matches_with_heracles,
        lark_parse_sldp,
        sldp_value_matches_solution_formats,
        validate_pddl_goal,
    )

    c = ValidationContract.model_validate(contract)
    rules = answer_validation or c.answer.comparison.model_dump()
    if c.query_policy.refusal and str(expected).strip() == c.query_policy.refusal:
        return {
            "status": "pass" if str(actual).strip() == c.query_policy.refusal else "mismatch",
            "format_valid": True,
            "match": str(actual).strip() == c.query_policy.refusal,
        }
    if c.goal is not None:
        if str(expected).strip() and not validate_pddl_goal(str(expected))[0]:
            return {"status": "invalid_expected_solution", "format_valid": None, "match": None}
        valid, match = answer_matches_with_heracles(str(actual), str(expected), "PDDL")
        return {"status": "pass" if match else "mismatch", "format_valid": valid, "match": match}
    try:
        if not sldp_value_matches_solution_formats(str(expected), c.answer.formats):
            raise ValueError("Expected solution violates declared formats")
        if str(expected).strip():
            gold = lark_parse_sldp(str(expected))
        else:
            gold = None
        _answer_value(gold, c)
        if c.answer.comparison.mode == "top_k_with_ties" and not _top_k_matches(gold, rules):
            raise ValueError("Expected ranking or score metadata is inconsistent")
    except Exception:
        return {"status": "invalid_expected_solution", "format_valid": None, "match": None}
    if not sldp_value_matches_solution_formats(str(actual), c.answer.formats):
        return {"status": "invalid_answer_format", "format_valid": False, "match": False}
    try:
        parsed = lark_parse_sldp(str(actual)) if str(actual).strip() else None
        _answer_value(parsed, c)
    except Exception:
        return {"status": "invalid_answer_format", "format_valid": False, "match": False}
    if c.result.type == "integer" and isinstance(parsed, float) and not parsed.is_integer():
        return {"status": "invalid_answer_format", "format_valid": False, "match": False}
    if c.result.unordered_tuple:

        def unordered_pairs(value):
            if isinstance(value, tuple) and value[0] == "set":
                return (
                    "set",
                    *[("list", *sorted(pair[1:], key=canonical_key)) for pair in value[1:]],
                )
            return value

        parsed, gold = unordered_pairs(parsed), unordered_pairs(gold)
    match = (
        parsed is gold
        if parsed is None or gold is None
        else sldp_equal(parsed, gold, c.answer.comparison)
    )
    if c.result.kind == "multiset" and isinstance(parsed, tuple) and isinstance(gold, tuple):
        match = _matching(parsed[1:], gold[1:], lambda a, b: sldp_equal(a, b, c.answer.comparison))
    if c.answer.comparison.mode == "top_k_with_ties":
        match = _top_k_matches(parsed, rules)
    return {"status": "pass" if match else "mismatch", "format_valid": True, "match": match}


def render_decoded(actual, contract):
    c = ValidationContract.model_validate(contract)
    if c.result.kind == "role_sets":
        return json.dumps(actual, sort_keys=True)

    def render(value):
        if isinstance(value, tuple):
            return "POINT(" + " ".join(repr(v) for v in value) + ")"
        if isinstance(value, list):
            return "[" + ", ".join(render(v) for v in value) + "]"
        return str(value)

    if actual is None or isinstance(actual, list) and not actual:
        return c.answer.empty
    if (
        c.result.kind in {"set", "tuples"}
        or c.result.kind == "flexible"
        and isinstance(actual, list)
    ):
        return "<" + ", ".join(render(v) for v in actual) + ">"
    return render(actual)


def validate_record_result(record, rows, query, columns=None):
    contract = contract_for_record(record)
    if not contract:
        return {"status": "not_checked", "match": None, "issues": ["No saved validation contract"]}
    decoded = decode_result(query, rows, contract, columns)
    result = {
        **asdict(decoded),
        "match": None,
        "solution": None,
        "semantic_check": (
            {"status": "semantics_unverified", "reason": "disabled by contract"}
            if contract.get("semantic_checks", {}).get("mode") == "unverified"
            else semantic_evidence(
                query,
                (record.get("metadata") or {}).get("seed_gold_cypher")
                or record.get("gold_cypher")
                or record.get("cypher"),
            )
        ),
    }
    if decoded.status != "pass":
        return result
    c = ValidationContract.model_validate(contract)
    result["solution"] = render_decoded(decoded.actual, contract)
    if c.result.kind == "role_sets":
        expected = (record.get("grounding") or {}).get("expected")
        if expected is None:
            result.update(status="not_checked", issues=["Missing expected grounding"])
            return result
        if set(expected) != set(c.result.roles):
            result.update(
                status="invalid_expected_solution", issues=["Expected grounding roles differ"]
            )
            return result
        try:
            expected = {
                name: [
                    _atom(value, c.result.roles[name], c.result.accepted_forms) for value in values
                ]
                for name, values in expected.items()
            }
        except (DecodeError, TypeError):
            result.update(
                status="invalid_expected_solution", issues=["Invalid expected grounding types"]
            )
            return result
        expected = {
            name: _unique(values) if c.result.roles[name].cardinality == "set" else values
            for name, values in expected.items()
        }

        def equal_value(a, b, role):
            if role.type == "point":
                return all(
                    abs(x - y) <= c.result.comparison.point_abs_tolerance
                    for x, y in zip(a, b, strict=True)
                )
            if role.type in {"integer", "number"}:
                return abs(a - b) <= c.result.comparison.numeric_abs_tolerance
            return a == b

        pairs = {
            name: _matched_pairs(
                decoded.actual[name],
                values,
                lambda a, b, role=c.result.roles[name]: equal_value(a, b, role),
            )
            for name, values in expected.items()
        }
        result["missing"] = {
            name: [v for i, v in enumerate(values) if i not in pairs[name]]
            for name, values in expected.items()
        }
        result["extra"] = {
            name: [v for i, v in enumerate(decoded.actual[name]) if i not in pairs[name].values()]
            for name in expected
        }
        result["match"] = all(
            _matching(
                decoded.actual[name],
                values,
                lambda a, b, role=c.result.roles[name]: equal_value(a, b, role),
            )
            if c.result.roles[name].cardinality != "list"
            else len(decoded.actual[name]) == len(values)
            and all(
                equal_value(a, b, c.result.roles[name])
                for a, b in zip(decoded.actual[name], values, strict=True)
            )
            for name, values in expected.items()
        )
    else:
        expected = next(
            (
                record[k]
                for k in ("expected_solution", "real_solution", "solution")
                if record.get(k) is not None
            ),
            None,
        )
        if expected is None:
            result.update(status="not_checked", issues=["Missing expected solution"])
            return result
        comparison_contract = c.model_dump(mode="json")
        comparison_contract["answer"]["comparison"] = c.result.comparison.model_dump(mode="json")
        comparison = compare_answers(
            result["solution"], str(expected), comparison_contract, record.get("answer_validation")
        )
        result["match"] = comparison["match"]
        if comparison["status"] == "invalid_expected_solution":
            result.update(status="invalid_expected_solution")
            return result
    result["status"] = "pass" if result["match"] else "mismatch"
    return result


def actual_final_solution(result, contract):
    from cypher_3dsg_agent_distillation.cypher.pddl_goals import render_goal_spec

    c = ValidationContract.model_validate(contract)
    if c.goal is None:
        return result.get("solution")
    if result.get("actual") is None:
        return None
    return render_goal_spec(c.goal, result["actual"]) or ""


def compare_record_answer(record, actual, expected):
    """Compatibility tuple for callers migrating to contracts."""
    from cypher_3dsg_agent_distillation.graph.heracles_validation import (
        answer_matches_with_heracles,
    )

    contract = contract_for_record(record)
    if contract:
        outcome = compare_answers(actual, expected, contract, record.get("answer_validation"))
        return outcome["format_valid"], outcome["match"]
    return answer_matches_with_heracles(
        str(actual),
        str(expected),
        str(record.get("output_type") or "SLDP"),
        answer_validation=record.get("answer_validation"),
        solution_formats=record.get("solution_formats") or record.get("solution_format"),
    )


def final_grounding_check(record, answer, result):
    contract = contract_for_record(record)
    if not contract or not result or result.get("solution") is None or answer is None:
        return {"status": "not_checked", "match": None}
    expected = actual_final_solution(result, contract)
    # Tie alternatives belong to answer-vs-oracle evaluation. Grounding must match
    # what this execution actually returned.
    grounded_contract = ValidationContract.model_validate(contract).model_dump(mode="json")
    comparison = grounded_contract["answer"]["comparison"]
    grounded_contract["answer"]["comparison"] = {
        "mode": "pddl_equivalence" if contract.get("goal") else "exact",
        "numeric_abs_tolerance": comparison["numeric_abs_tolerance"],
        "point_abs_tolerance": comparison["point_abs_tolerance"],
    }
    return compare_answers(answer, expected, grounded_contract)
