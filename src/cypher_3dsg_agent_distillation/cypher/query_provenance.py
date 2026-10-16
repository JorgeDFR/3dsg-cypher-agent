"""Conservative projection lineage for the supported read-only Cypher fragment."""

from __future__ import annotations

import json
import re
from copy import deepcopy

from cypher_3dsg_agent_distillation.cypher.task_semantics import (
    _canonical,
    _parser,
    parse_query,
    walk_expression,
)


def _union_branches(query):
    # Mask literals/comments before looking for top-level UNION. A keyword inside
    # a string, comment, or nested expression is not a branch boundary.
    masked = re.sub(
        r"'(?:(?:\\.)|[^'\\])*'|\"(?:(?:\\.)|[^\"\\])*\"|`[^`]*`"
        r"|//[^\n]*|/\*(?s:.*?)\*/",
        lambda match: " " * len(match.group()),
        query,
    )
    depth, start, branches = 0, 0, []
    for token in re.finditer(r"[(){}\[\]]|\bUNION(?:\s+ALL)?\b", masked, re.I):
        value = token.group().upper()
        if value in {"(", "{", "["}:
            depth += 1
        elif value in {")", "}", "]"}:
            depth -= 1
        elif depth == 0:
            branches.append(query[start : token.start()])
            start = token.end()
    return [*branches, query[start:]]


def projections(query):
    branches = [
        _branch_projections(branch, index) for index, branch in enumerate(_union_branches(query))
    ]
    if len(branches) == 1:
        return branches[0]
    names = [item["column"] for item in branches[0]]
    if any([item["column"] for item in branch] != names for branch in branches):
        raise ValueError("UNION branches require matching projection names and order")
    # LIMIT around a UNION may truncate the merged stream. Never use another
    # branch's local metadata as proof that an absent role was fully retrieved.
    if any(not item["complete"] for branch in branches for item in branch):
        empty_complete = all(item["empty_complete"] for branch in branches for item in branch)
        for branch in branches:
            for item in branch:
                item["complete"] = False
                item["empty_complete"] = empty_complete
    return [
        {"column": name, "kind": "union", "alternatives": [branch[i] for branch in branches]}
        for i, name in enumerate(names)
    ]


def _branch_projections(query, branch):
    operations = parse_query(query)["operations"]
    env = {}
    returned = []
    source_count = 0
    complete = not any(op["kind"] == "limit" for op in operations)
    empty_complete = all(
        type(op["value"]) is int and op["value"] > 0
        for op in operations if op["kind"] == "limit"
    )

    def resolve(expr):
        kind = expr.get("kind")
        if kind == "ref":
            name, *tail = expr["name"].split(".")
            value = deepcopy(env.get(name, {"kind": "unknown"}))
            for key in tail:
                if value.get("kind") == "map":
                    value = value["entries"].get(key, {"kind": "unknown"})
                elif value.get("kind") == "node":
                    value = {
                        **value,
                        "kind": "property",
                        "property": key,
                        "value_type": {
                            "nodeSymbol": "node_symbol",
                            "class": "string",
                            "name": "string",
                            "center": "point",
                        }.get(key),
                    }
                else:
                    return {"kind": "unknown"}
            return value
        if kind == "call" and expr["name"] == "collect":
            return {"kind": "collection", "item": resolve(expr["arguments"][0])}
        if kind == "list":
            return {"kind": "list", "items": [resolve(item) for item in expr["items"]]}
        if kind == "index":
            container = resolve(expr["value"])
            index = expr["index"]
            if index.get("kind") != "literal" or type(index.get("value")) is not int:
                return {"kind": "unknown"}
            if container.get("kind") == "collection":
                return {**container["item"], "complete": False}
            if container.get("kind") == "list":
                try:
                    return container["items"][index["value"]]
                except IndexError:
                    pass
            return {"kind": "unknown"}
        if kind == "map":
            return {"kind": "map", "entries": {k: resolve(v) for k, v in expr["entries"].items()}}
        if kind == "call":
            args = [resolve(arg) for arg in expr["arguments"]]
            if expr["name"] == "labels" and args[0].get("kind") == "node":
                return {**args[0], "kind": "labels", "value_type": "string"}
            value_type = {
                "count": "integer",
                "length": "integer",
                "tointeger": "integer",
                "point.distance": "number",
                "point": "point",
                "substring": "string",
                "tolower": "string",
            }.get(expr["name"])
            if expr["name"] in {"min", "max"}:
                value_type = args[0].get("value_type")
            return {"kind": "computed", "function": expr["name"], "value_type": value_type}
        if kind == "literal":
            value = expr["value"]
            value_type = (
                "integer"
                if type(value) is int
                else "number"
                if type(value) is float
                else "string"
                if isinstance(value, str)
                else None
            )
            return {"kind": "literal", "value_type": value_type}
        return {"kind": "unknown"}

    def domains(expr):
        if expr.get("kind") == "logical" and expr["operator"] in {"and", "or"}:
            branches = [domains(arg) for arg in expr["arguments"]]
            if expr["operator"] == "or":
                common = set.intersection(*(set(branch) for branch in branches))
                return {
                    key: list(dict.fromkeys(v for branch in branches for v in branch[key]))
                    for key in common
                }
            result = {}
            for branch in branches:
                for key, values in branch.items():
                    result[key] = [v for v in values if key not in result or v in result[key]]
            return result
        if expr.get("kind") == "compare":
            lhs, rhs = expr["left"], expr["right"]
            if expr["operator"] == "=" and lhs.get("kind") == "literal":
                lhs, rhs = rhs, lhs
            if lhs.get("kind") == "ref":
                if expr["operator"] == "=" and rhs.get("kind") == "literal":
                    return {lhs["name"]: [rhs["value"]]}
                if expr["operator"] == "IN" and rhs.get("kind") == "list" and all(
                    item.get("kind") == "literal" for item in rhs["items"]
                ):
                    return {lhs["name"]: [item["value"] for item in rhs["items"]]}
        return {}

    def filters(expr):
        if expr.get("kind") == "logical" and expr.get("operator") == "and":
            for arg in expr["arguments"]:
                filters(arg)
        if expr.get("kind") == "compare":
            pairs = [(expr["left"], expr["right"])]
            if expr["operator"] == "=":
                pairs.append((expr["right"], expr["left"]))
            for lhs, rhs in pairs:
                if lhs.get("kind") != "ref":
                    continue
                parts = lhs["name"].split(".")
                node = env.get(parts[0], {})
                if len(parts) != 2 or node.get("kind") != "node":
                    continue
                if expr["operator"] == "=" and rhs.get("kind") in {"literal", "binding"}:
                    node["origin"][parts[1]] = (
                        rhs["value"] if rhs["kind"] == "literal" else {"binding": rhs["name"]}
                    )
                elif (
                    expr["operator"] == "IN"
                    and rhs.get("kind") == "list"
                    and all(item.get("kind") == "literal" for item in rhs["items"])
                ):
                    node["domains"][parts[1]] = [item["value"] for item in rhs["items"]]

    for op in operations:
        if op["kind"] == "match":
            previous = env.get(op["role"], {})
            source = previous.get("source")
            if source is None:
                source = f"{branch}:{source_count}"
                source_count += 1
            env[op["role"]] = {
                "kind": "node",
                "source": source,
                "domains": deepcopy(previous.get("domains", {})),
                "optional": previous.get("optional", True) and op.get("optional", False),
                "layers": sorted(set(previous.get("layers", []) + op.get("layers", []))),
                "origin": {
                    **previous.get("origin", {}),
                    **{
                        k: (v["value"] if v["kind"] == "literal" else {"binding": v["name"]})
                        for k, v in op.get("properties", {}).items()
                        if v.get("kind") in {"literal", "binding"}
                    },
                },
            }
        elif op["kind"] == "filter":
            filters(op["expression"])
            for name, values in domains(op["expression"]).items():
                role, _, prop = name.partition(".")
                if env.get(role, {}).get("kind") == "node" and prop:
                    env[role]["domains"][prop] = values
            for role, node in env.items():
                if node.get("kind") != "node":
                    continue
                properties = [
                    expr["name"][len(role) + 1:]
                    for expr in walk_expression(op["expression"])
                    if expr.get("kind") == "ref" and expr["name"].startswith(role + ".")
                ]
                if properties:
                    node.setdefault("predicates", []).append({
                        "role": role, "properties": properties, "expression": op["expression"],
                    })
        elif op["kind"] in {"with", "project"}:
            values = [
                (
                    item.get("alias") or item["expression"].get("name", ""),
                    resolve(item["expression"]),
                )
                for item in op["items"]
            ]
            if op["kind"] == "with":
                env = dict(values)
            else:
                returned = [
                    {"column": name, **value, "branch": branch,
                     "complete": complete and value.get("complete", True),
                     "empty_complete": empty_complete}
                    for name, value in values
                ]
        elif op["kind"] == "unwind":
            value = resolve(op["expression"])
            env[op["alias"]] = value.get("item", {"kind": "unknown"})
    return returned


def predicate_covers(predicate, origin):
    """Prove a search includes a public role constraint, including empty searches."""
    unknown = object()

    def evaluate(expr):
        kind = expr.get("kind")
        if kind == "literal":
            return expr["value"]
        if kind == "ref":
            prefix = predicate["role"] + "."
            return origin.get(expr["name"][len(prefix):], unknown) if expr["name"].startswith(
                prefix
            ) else unknown
        if kind == "call" and expr["name"] == "tolower":
            value = evaluate(expr["arguments"][0])
            return value.lower() if isinstance(value, str) else unknown
        if kind == "list":
            return [evaluate(item) for item in expr["items"]]
        if kind == "logical":
            values = [evaluate(arg) for arg in expr["arguments"]]
            if expr["operator"] == "or":
                return True if any(v is True for v in values) else unknown
            if expr["operator"] == "and":
                return True if all(v is True for v in values) else unknown
            return unknown
        if kind == "compare":
            left, right = evaluate(expr["left"]), evaluate(expr["right"])
            if left is unknown or right is unknown:
                return unknown
            if expr["operator"] == "=":
                return left == right
            if expr["operator"] == "CONTAINS" and isinstance(left, str) and isinstance(right, str):
                return right in left
            if expr["operator"] == "IN" and isinstance(right, list):
                return left in right
        return unknown

    return evaluate(predicate["expression"]) is True


def reference_roles(query):
    """Build configuration candidates from trusted gold projections, never answers."""
    return projections(query)


def semantic_evidence(actual, reference):
    if not actual or not reference:
        return {"status": "semantics_unverified", "reason": "query unavailable"}
    try:

        def signature(query):
            spec = parse_query(query)
            # Return names are cosmetic here; positional correspondence is retained.
            aliases = {}
            for op in spec["operations"]:
                if op["kind"] == "project":
                    for index, item in enumerate(op["items"]):
                        old = item.get("alias")
                        if old:
                            aliases[old] = f"output_{index}"
                            item["alias"] = aliases[old]
                elif op["kind"] == "order":
                    for item in op["items"]:
                        expr = item["expression"]
                        if expr.get("kind") == "ref" and expr["name"] in aliases:
                            expr["name"] = aliases[expr["name"]]
            # Clause boundaries affect relationship uniqueness and OPTIONAL scope.
            clauses = _parser().parse(query.rstrip().rstrip(";"))
            boundaries = [
                (clause["optional"], len(clause["operations"]))
                if clause["kind"] == "match_clause"
                else clause["kind"]
                for clause in clauses
            ]
            return _canonical(spec), boundaries

        if signature(actual) == signature(reference):
            return {
                "status": "equivalent_in_supported_fragment",
                "method": "alpha-renaming and cosmetic projection aliases",
            }
    except (ValueError, KeyError, TypeError):
        pass
    return {
        "status": "semantics_unverified",
        "reason": "No supported equivalence proof; AST differences are not a counterexample.",
    }


def canonical_key(value):
    return json.dumps(value, sort_keys=True, default=str)
