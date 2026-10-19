"""Decode actual results before looking at gold values; compare with typed contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any

from cypher_3dsg_agent_distillation.cypher.query_provenance import (
    canonical_key,
    predicate_covers,
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
    auxiliary_columns: list[str] = field(default_factory=list)
    role_rows: dict[str, list[int]] = field(default_factory=dict)


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
    if lineage.get("kind") == "union":
        matches = [_origin_matches(spec, item) for item in lineage["alternatives"]]
        return matches[0] if all(item is matches[0] for item in matches) else None
    lineage = _lineage_atom(lineage)
    for part in spec.path:
        if isinstance(part, str):
            lineage = lineage.get("entries", {}).get(part, {"kind": "unknown"})
        elif lineage.get("kind") == "list":
            try:
                lineage = lineage["items"][part]
            except IndexError:
                return None
        else:
            lineage = lineage.get("item", {"kind": "unknown"})
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
    uncertain = False
    for key, value in spec.origin.items():
        if key in lineage.get("origin", {}):
            if lineage["origin"][key] != value:
                return False
        elif key in lineage.get("domains", {}):
            if value not in lineage["domains"][key]:
                return False
            uncertain = True
        else:
            uncertain = True
    if (
        spec.layers
        and lineage.get("layers")
        and not set(spec.layers).intersection(lineage["layers"])
    ):
        return False
    return None if uncertain else True


def _projection_spec(info):
    """Known expression types, independent of aliases and expected values."""
    if info.get("kind") == "union":
        specs = [_projection_spec(item) for item in info["alternatives"]]
        return specs[0] if specs and all(item == specs[0] for item in specs) else None
    info = _lineage_atom(info)
    kind = info.get("value_type")
    if kind is None:
        return None
    return ValueSpec(type=kind, layers=info.get("layers", []) if kind == "node_symbol" else [])


def _compatible_type(role, info):
    known = _projection_spec(info)
    return (
        known is None
        or known.type == role.type
        or (known.type, role.type)
        in {
            ("integer", "number"),
            ("node_symbol", "string"),
        }
    )


def _row_provenance(lineage, row, forms):
    """Attach same-node observations; identify UNION branches only by actual evidence."""
    if not lineage:
        return {}
    branches = (
        list(zip(*(item["alternatives"] for item in lineage), strict=True))
        if lineage[0].get("kind") == "union"
        else [lineage]
    )
    compatible = []
    for branch in branches:
        entries = deepcopy(list(branch))
        sources = {}
        try:
            for info in entries:
                if info["column"] not in row:
                    raise DecodeError("Missing returned column")
                value = row[info["column"]]
                known = _projection_spec(info)
                if value is not None and known:
                    _values(value, known, forms)
                source = info.get("source")
                if not source or info.get("kind") == "collection":
                    continue
                evidence = sources.setdefault(source, {"origin": {}, "domains": {}, "layers": []})
                evidence["domains"].update(info.get("domains", {}))
                evidence["layers"] = sorted(set(evidence["layers"] + info.get("layers", [])))
                observed = dict(info.get("origin", {}))
                labels = None
                if info.get("kind") == "property" and not isinstance(value, Mapping | list | tuple):
                    if value is not None:
                        prop = info["property"]
                        if prop in observed and observed[prop] != value:
                            raise DecodeError("Returned property conflicts with query provenance")
                        observed[prop] = value
                elif info.get("kind") == "labels":
                    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                        raise DecodeError("Expected a list of node labels")
                    labels = value
                elif isinstance(value, Mapping) and value.get("__neo4j_type__") == "node":
                    for prop, actual in value["properties"].items():
                        if prop in observed and observed[prop] != actual:
                            raise DecodeError("Returned node conflicts with query provenance")
                        observed[prop] = actual
                    labels = value["labels"]
                if labels is not None:
                    if evidence["layers"] and not set(evidence["layers"]) <= set(labels):
                        raise DecodeError("Returned labels conflict with query provenance")
                    evidence["layers"] = labels
                for prop, actual in observed.items():
                    if prop in evidence["origin"] and evidence["origin"][prop] != actual:
                        raise DecodeError("Same-node projections disagree")
                    if prop in evidence["domains"] and actual not in evidence["domains"][prop]:
                        raise DecodeError("Returned property is outside its query constraint")
                    evidence["origin"][prop] = actual
            for info in entries:
                if info.get("source") in sources:
                    info.update(sources[info["source"]])
            compatible.append({info["column"]: info for info in entries})
        except DecodeError:
            continue
    if not compatible:
        raise DecodeError("Returned values conflict with projection provenance")
    if len(compatible) == 1:
        return compatible[0]
    # Preserve every possible branch. A role can use the row only if those
    # branches agree about its provenance; a simple typed answer may not need
    # branch identity at all.
    return {
        column: {"kind": "union", "alternatives": [item[column] for item in compatible]}
        for column in compatible[0]
    }


def _covers_role(role, info, *, empty_rows=False):
    """A complete, constrained source must cover even a role with no returned rows."""
    alternatives = info.get("alternatives", [info])
    for item in alternatives:
        if item.get("kind") not in {"property", "node"} or not (
            item.get("complete") or empty_rows and item.get("empty_complete")
        ):
            continue
        if not _compatible_type(role, item) or _origin_matches(role, item) is False:
            continue
        if all(
            item.get("origin", {}).get(key) == value
            or value in item.get("domains", {}).get(key, [])
            or any(
                key in predicate["properties"] and predicate_covers(predicate, role.origin)
                for predicate in item.get("predicates", [])
            )
            for key, value in role.origin.items()
        ):
            return True
    return False


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
        if lineage and keys and set(keys) != set(by_column):
            raise DecodeError("Saved result columns disagree with the executed query projection")
        observed_lineage = lineage
        row_provenance = [
            _row_provenance(observed_lineage, row, spec.accepted_forms) for row in rows
        ]
        roles = spec.roles or {
            "value": ValueSpec(
                type=spec.type, layers=spec.layers, property=spec.property, origin=spec.origin
            )
            if spec.element is None
            else spec.element
        }
        # A literal list projection preserves tuple positions without guessing
        # from aliases or expected values. Explicit contract selectors win.
        if spec.kind == "tuples" and not any(
            role.column is not None or role.position is not None or role.path
            for role in roles.values()
        ):
            packed = [
                info
                for info in lineage
                if info.get("kind") == "list"
                and len(info["items"]) == len(roles)
                and all(
                    _compatible_type(role, item)
                    for role, item in zip(
                        roles.values(),
                        info["items"],
                        strict=True,
                    )
                )
            ]
            if len(packed) == 1:
                roles = {
                    name: role.model_copy(update={"column": packed[0]["column"], "path": [index]})
                    for index, (name, role) in enumerate(roles.items())
                }
        candidates, data, partitions = {}, {}, {}
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
            pooled_union = (
                spec.kind == "role_sets"
                and name in by_column
                and by_column[name].get("kind") == "union"
                and _origin_matches(role, by_column[name]) is None
            )
            if selected is None and name != "value" and name in keys and not pooled_union:
                selected = name
            possible = []
            for key in keys:
                if selected is not None and key != selected:
                    continue
                info = by_column.get(key, {"kind": "unknown"})
                if not _compatible_type(role, info):
                    continue
                origin = _origin_matches(role, info)
                row_matches = [
                    _origin_matches(role, evidence.get(key, {"kind": "unknown"}))
                    for evidence in row_provenance
                ]
                partition = (
                    spec.kind == "role_sets"
                    and bool(role.origin)
                    and selected is None
                    and not role.path
                    and _covers_role(role, info, empty_rows=not rows)
                    and (origin is not True or any(match is not True for match in row_matches))
                )
                if partition:
                    if any(match is None for match in row_matches):
                        continue
                    mask = [match is True for match in row_matches]
                else:
                    mask = [True] * len(rows)
                    if origin is False or any(match is False for match in row_matches):
                        if selected == key:
                            raise DecodeError(f"Projection origin conflicts with role {name}")
                        continue
                    if (
                        role.origin
                        and selected is None
                        and (
                            any(match is None for match in row_matches)
                            or not rows
                            and origin is not True
                        )
                    ):
                        continue
                try:
                    values = [
                        _values(row[key], role, spec.accepted_forms) if included else []
                        for row, included in zip(rows, mask, strict=True)
                    ]
                except KeyError:
                    continue
                except DecodeError:
                    if selected == key:
                        raise
                    continue
                possible.append(key)
                data[name, key] = values
                if partition:
                    partitions[name, key] = mask
            # Empty rows carry no answer values; roles still cannot invent nonempty results.
            if not rows and not keys:
                possible = [f"empty:{name}"]
                data[name, possible[0]] = []
            if not possible:
                if (
                    role.origin
                    and selected is None
                    and (
                        not lineage or any(_origin_matches(role, item) is None for item in lineage)
                    )
                ):
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
                        and not (
                            (name, assignment[name]) in partitions
                            and (other, assignment[other]) in partitions
                        )
                        and (roles[name].origin != roles[other].origin or not roles[name].origin)
                    ):
                        valid = False
            if not valid:
                continue
            for key in set(keys_for_roles):
                users = [n for n in names if assignment[n] == key]
                if any((n, key) in partitions for n in users) and any(
                    sum(partitions.get((n, key), [True] * len(rows))[i] for n in users) != 1
                    for i in range(len(rows))
                ):
                    valid = False
            if not valid:
                continue
            unused = set(keys) - set(keys_for_roles)
            can_infer_auxiliary = all(
                key in by_column
                or roles[name].column is not None
                or roles[name].position is not None
                for name, key in assignment.items()
            )
            for key in unused:
                auxiliary = spec.auxiliary.get(key)
                if auxiliary is None and spec.extra_columns == "inferred" and can_infer_auxiliary:
                    auxiliary = _projection_spec(by_column.get(key, {}))
                if auxiliary is None:
                    valid = False
                    break
                for row in rows:
                    _values(row[key], auxiliary, spec.accepted_forms)
            if valid:
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
            return flat

        interpretations = [(materialize(a), a) for a in assignments]
        if len({canonical_key(v) for v, _ in interpretations}) != 1:
            raise MappingUnresolved("Multiple materially different result interpretations")
        actual, bindings = interpretations[0]
        return DecodedResult(
            "pass",
            actual,
            bindings,
            candidates=candidates,
            auxiliary_columns=sorted(set(keys) - set(bindings.values())),
            role_rows={
                name: [i for i, included in enumerate(partitions[name, key]) if included]
                for name, key in bindings.items()
                if (name, key) in partitions
            },
        )
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


def compare_answers(actual, expected, contract):
    from cypher_3dsg_agent_distillation.graph.heracles_validation import (
        answer_matches_with_heracles,
        lark_parse_sldp,
        sldp_value_matches_solution_formats,
        validate_pddl_goal,
    )

    c = ValidationContract.model_validate(contract)
    rules = c.answer.comparison.model_dump()
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
    if c.result.kind in {"set", "tuples"}:
        return "<" + ", ".join(render(v) for v in actual) + ">"
    return render(actual)


def validate_record_result(record, rows, query, columns=None):
    contract = contract_for_record(record)
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
            (record[k] for k in ("expected_solution", "solution") if record.get(k) is not None),
            None,
        )
        if expected is None:
            result.update(status="not_checked", issues=["Missing expected solution"])
            return result
        comparison_contract = c.model_dump(mode="json")
        comparison_contract["answer"]["comparison"] = c.result.comparison.model_dump(mode="json")
        comparison = compare_answers(result["solution"], str(expected), comparison_contract)
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
