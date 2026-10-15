"""Conservative projection lineage for the supported read-only Cypher fragment."""

from __future__ import annotations

import json
from copy import deepcopy

from cypher_3dsg_agent_distillation.cypher.task_semantics import _canonical, _parser, parse_query


def projections(query):
    operations = parse_query(query)["operations"]
    env = {}
    returned = []

    def resolve(expr):
        kind = expr.get("kind")
        if kind == "ref":
            name, *tail = expr["name"].split(".")
            value = deepcopy(env.get(name, {"kind": "unknown"}))
            for key in tail:
                if value.get("kind") == "map":
                    value = value["entries"].get(key, {"kind": "unknown"})
                elif value.get("kind") == "node":
                    value = {**value, "kind": "property", "property": key}
                else:
                    return {"kind": "unknown"}
            return value
        if kind == "call" and expr["name"] == "collect":
            return {"kind": "collection", "item": resolve(expr["arguments"][0])}
        if kind == "map":
            return {"kind": "map", "entries": {k: resolve(v) for k, v in expr["entries"].items()}}
        if kind == "call":
            return {"kind": "computed", "function": expr["name"]}
        return {"kind": "unknown"}

    def filters(expr):
        if expr.get("kind") == "logical" and expr.get("operator") == "and":
            for arg in expr["arguments"]:
                filters(arg)
        if expr.get("kind") == "compare" and expr.get("operator") == "=":
            for lhs, rhs in ((expr["left"], expr["right"]), (expr["right"], expr["left"])):
                if lhs.get("kind") == "ref" and rhs.get("kind") in {"literal", "binding"}:
                    parts = lhs["name"].split(".")
                    if len(parts) == 2 and env.get(parts[0], {}).get("kind") == "node":
                        env[parts[0]]["origin"][parts[1]] = (
                            rhs["value"] if rhs["kind"] == "literal" else {"binding": rhs["name"]}
                        )

    for op in operations:
        if op["kind"] == "match":
            previous = env.get(op["role"], {})
            env[op["role"]] = {
                "kind": "node",
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
                returned = [{"column": name, **value} for name, value in values]
        elif op["kind"] == "unwind":
            value = resolve(op["expression"])
            env[op["alias"]] = value.get("item", {"kind": "unknown"})
    return returned


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
