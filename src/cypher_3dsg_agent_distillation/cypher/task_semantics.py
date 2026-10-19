"""Validated retrieval operations shared by templates and reasoning.

The parser deliberately supports a read-only Cypher subset, not arbitrary Cypher.
Unsupported syntax raises a diagnostic rather than yielding a partial explanation.
Question names, tags, wording, solutions and expected grounding are never inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from functools import lru_cache
from typing import Any, Literal

from lark import Lark, Transformer, UnexpectedInput
from pydantic import BaseModel, ConfigDict, Field, model_validator

SEMANTICS_VERSION = 1
REASONING_CONTEXT_VERSION = 2
REASONING_RENDERER_VERSION = 3

GRAMMAR = r"""
start: clause+
?clause: match_clause | optional_match | with_clause | where_clause | return_clause
       | order_clause | limit_clause | unwind_clause
match_clause: "MATCH"i pattern ("," pattern)*
optional_match: "OPTIONAL"i "MATCH"i pattern ("," pattern)*
pattern: node (edge node)* -> plain_pattern
node: "(" NAME? labels? props? ")"
labels: (":" layer)+
?layer: NAME | BINDING
props: "{" entries? "}"
edge: "-" "[" edge_spec "]" "->" -> outgoing
    | "<-" "[" edge_spec "]" "-" -> incoming
    | "-" "[" edge_spec "]" "-" -> undirected
edge_spec: NAME? ":" layer bounds?
bounds: "*" bound ".." bound
      | "*" bound -> exact_bounds
?bound: INT | BINDING
with_clause: "WITH"i distinct? projections
return_clause: "RETURN"i distinct? projections
distinct: "DISTINCT"i
projections: projection ("," projection)*
projection: expr ("AS"i NAME)?
where_clause: "WHERE"i expr
order_clause: "ORDER"i "BY"i sort ("," sort)*
sort: expr direction?
direction: "ASC"i -> ascending
         | "DESC"i -> descending
limit_clause: "LIMIT"i bound
unwind_clause: "UNWIND"i expr "AS"i NAME
?expr: or_expr
?or_expr: and_expr ("OR"i and_expr)* -> disjunction
?and_expr: not_expr ("AND"i not_expr)* -> conjunction
?not_expr: "NOT"i not_expr -> negation
         | comparison
?comparison: atom COMP atom -> comparison
           | atom "IN"i atom -> membership
           | atom "NOT"i "IN"i atom -> exclusion
           | atom "CONTAINS"i atom -> contains
           | atom "IS"i "NULL"i -> is_null
           | atom "IS"i "NOT"i "NULL"i -> is_not_null
           | path ":" layer -> label_test
           | atom
?atom: primary ("[" expr "]")* -> indexed
?primary: path "(" call_args? ")" -> call
     | path -> reference
     | STRING -> string
     | SIGNED_NUMBER -> number
     | BINDING -> binding
     | "null"i -> null
     | "true"i -> true
     | "false"i -> false
     | "*" -> wildcard
     | "[" arguments? "]" -> sequence
     | "{" entries? "}" -> mapping
     | "(" expr ")"
path: NAME ("." NAME)*
call_args: distinct? arguments
arguments: expr ("," expr)*
entries: entry ("," entry)*
entry: NAME ":" expr
COMP: "<>" | "<=" | ">=" | "=" | "<" | ">"
BINDING: /\{[A-Za-z_][A-Za-z0-9_.]*\}/
NAME: /[A-Za-z_][A-Za-z0-9_]*/
STRING: /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/
%import common.INT
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
%ignore /\/\/[^\n]*/
%ignore /\/\*(?s:.*?)\*\//
"""


def literal(value: Any) -> dict[str, Any]:
    return {"kind": "literal", "value": value}


class _AST(Transformer):
    def start(self, items):
        anonymous = {}
        for clause in items:
            for op in clause.get("operations", []):
                for field in ("role", "source", "target"):
                    value = op.get(field)
                    if value is not None and not isinstance(value, str):
                        op[field] = anonymous.setdefault(value, f"$anonymous_{len(anonymous)}")
        return items

    def indexed(self, items):
        value = items[0]
        for index in items[1:]:
            value = {"kind": "index", "value": value, "index": index}
        return value

    def path(self, items):
        return ".".join(str(item) for item in items)

    def reference(self, items):
        return {"kind": "ref", "name": items[0]}

    def string(self, items):
        raw = str(items[0])
        value = json.loads(raw) if raw.startswith('"') else raw[1:-1].replace("\\'", "'")
        if re.fullmatch(r"\{[A-Za-z_][A-Za-z0-9_.]*\}", value):
            return {"kind": "binding", "name": value[1:-1]}
        return literal(value)

    def number(self, items):
        value = str(items[0])
        return literal(float(value) if any(c in value.lower() for c in ".e") else int(value))

    def binding(self, items):
        return {"kind": "binding", "name": str(items[0])[1:-1]}

    def null(self, _items):
        return literal(None)

    def true(self, _items):
        return literal(True)

    def false(self, _items):
        return literal(False)

    def wildcard(self, _items):
        return {"kind": "ref", "name": "*"}

    def distinct(self, _items):
        return True

    def arguments(self, items):
        return items

    def call_args(self, items):
        return {"arguments": items[-1], "distinct": items[0] is True}

    def call(self, items):
        return {
            "kind": "call",
            "name": items[0].lower(),
            **(items[1] if len(items) > 1 else {"arguments": [], "distinct": False}),
        }

    def entry(self, items):
        return str(items[0]), items[1]

    def entries(self, items):
        return dict(items)

    def mapping(self, items):
        return {"kind": "map", "entries": items[0] if items else {}}

    def sequence(self, items):
        return {"kind": "list", "items": items[0] if items else []}

    def props(self, items):
        return items[0] if items else {}

    def labels(self, items):
        return [str(item) for item in items]

    def node(self, items):
        role = next((str(item) for item in items if isinstance(item, str)), object())
        labels = next((item for item in items if isinstance(item, list)), [])
        props = next((item for item in items if isinstance(item, dict)), {})
        return {"role": role, "layers": labels, "properties": props}

    def bounds(self, items):
        return [str(item) if str(item).startswith("{") else int(item) for item in items]

    def exact_bounds(self, items):
        value = int(items[0])
        return [value, value]

    def edge_spec(self, items):
        variable = str(items[0]) if len(items) > 1 and isinstance(items[1], str) else None
        if variable is not None:
            items = items[1:]
        return {
            "relationship": str(items[0]),
            "bounds": items[1] if len(items) > 1 else [1, 1],
            **({"relationship_variable": variable} if variable else {}),
        }

    def outgoing(self, items):
        return {**items[0], "direction": "outgoing"}

    def incoming(self, items):
        return {**items[0], "direction": "incoming"}

    def undirected(self, items):
        return {**items[0], "direction": "undirected"}

    def plain_pattern(self, items):
        operations = [{"kind": "match", **items[0]}]
        for index in range(1, len(items), 2):
            operations.append(
                {
                    "kind": "traverse",
                    "source": items[index - 1]["role"],
                    "target": items[index + 1]["role"],
                    **items[index],
                }
            )
            operations.append({"kind": "match", **items[index + 1]})
        return operations

    def _match(self, items, optional):
        return {
            "kind": "match_clause",
            "optional": optional,
            "operations": [op for pattern in items for op in pattern],
        }

    def match_clause(self, items):
        return self._match(items, False)

    def optional_match(self, items):
        return self._match(items, True)

    def comparison(self, items):
        return {"kind": "compare", "left": items[0], "operator": str(items[1]), "right": items[2]}

    def membership(self, items):
        return {"kind": "compare", "left": items[0], "operator": "IN", "right": items[1]}

    def exclusion(self, items):
        return {"kind": "compare", "left": items[0], "operator": "NOT IN", "right": items[1]}

    def contains(self, items):
        return {"kind": "compare", "left": items[0], "operator": "CONTAINS", "right": items[1]}

    def is_null(self, items):
        return {"kind": "compare", "left": items[0], "operator": "IS", "right": literal(None)}

    def is_not_null(self, items):
        return {"kind": "compare", "left": items[0], "operator": "IS NOT", "right": literal(None)}

    def label_test(self, items):
        return {"kind": "label", "role": items[0], "layer": str(items[1])}

    def _logical(self, items, operator):
        return (
            items[0]
            if len(items) == 1
            else {"kind": "logical", "operator": operator, "arguments": items}
        )

    def conjunction(self, items):
        return self._logical(items, "and")

    def disjunction(self, items):
        return self._logical(items, "or")

    def negation(self, items):
        return {"kind": "logical", "operator": "not", "arguments": items}

    def projection(self, items):
        return {"expression": items[0], "alias": str(items[1]) if len(items) > 1 else None}

    def projections(self, items):
        return items

    def _project(self, items, kind):
        return {"kind": kind, "distinct": items[0] is True, "items": items[-1]}

    def with_clause(self, items):
        return self._project(items, "with")

    def return_clause(self, items):
        return self._project(items, "project")

    def where_clause(self, items):
        return {"kind": "filter", "expression": items[0]}

    def ascending(self, _items):
        return "ASC"

    def descending(self, _items):
        return "DESC"

    def sort(self, items):
        return {"expression": items[0], "direction": items[1] if len(items) > 1 else "ASC"}

    def order_clause(self, items):
        return {"kind": "order", "items": items}

    def limit_clause(self, items):
        return {
            "kind": "limit",
            "value": str(items[0]) if str(items[0]).startswith("{") else int(items[0]),
        }

    def unwind_clause(self, items):
        return {"kind": "unwind", "expression": items[0], "alias": str(items[1])}


SUPPORTED_FUNCTIONS = {
    "labels",
    "point",
    "point.distance",
    "point.withinbbox",
    "count",
    "collect",
    "min",
    "max",
    "tointeger",
    "substring",
    "length",
    "tolower",
}


def walk_expression(value: Any):
    if isinstance(value, dict):
        if "kind" in value:
            yield value
        for child in value.values():
            yield from walk_expression(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_expression(child)


class RetrievalOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[
        "match",
        "traverse",
        "filter",
        "with",
        "project",
        "order",
        "limit",
        "unwind",
        "refusal",
    ]
    role: str | None = None
    layers: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False
    source: str | None = None
    target: str | None = None
    relationship: str | None = None
    relationship_variable: str | None = None
    direction: Literal["outgoing", "incoming", "undirected"] | None = None
    bounds: list[int | str] | None = None
    expression: dict[str, Any] | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    distinct: bool = False
    value: int | str | None = None
    alias: str | None = None

    @model_validator(mode="after")
    def check_required_fields(self):
        required = {
            "match": ["role"],
            "traverse": ["source", "target", "relationship", "direction", "bounds"],
            "filter": ["expression"],
            "unwind": ["expression", "alias"],
            "limit": ["value"],
        }
        for field in required.get(self.kind, []):
            if getattr(self, field) is None:
                raise ValueError(f"{self.kind} requires {field}")
        if self.kind in {"with", "project", "order"} and not self.items:
            raise ValueError(f"{self.kind} requires projection/order items")
        if self.bounds is not None and len(self.bounds) != 2:
            raise ValueError("Traversal bounds require minimum and maximum")
        return self


class TaskSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    operations: list[RetrievalOperation]


@lru_cache(maxsize=1)
def _parser():
    return Lark(GRAMMAR, parser="lalr", transformer=_AST())


@lru_cache(maxsize=4096)
def _parse_query(query: str) -> dict[str, Any]:
    if query.strip() == "// WRITE_QUERY_NOT_ALLOWED":
        return {"version": SEMANTICS_VERSION, "operations": [{"kind": "refusal"}]}
    try:
        clauses = _parser().parse(query.rstrip().rstrip(";"))
    except UnexpectedInput as exc:
        raise ValueError(
            f"Unsupported Cypher structure at line {exc.line}, column {exc.column}"
        ) from exc
    operations = []
    for clause in clauses:
        if clause["kind"] == "match_clause":
            operations.extend({**op, "optional": clause["optional"]} for op in clause["operations"])
        else:
            operations.append(clause)
    for expr in walk_expression(operations):
        if expr["kind"] == "call" and expr["name"] not in SUPPORTED_FUNCTIONS:
            raise ValueError(f"Unsupported Cypher function: {expr['name']}")
        if expr["kind"] == "call":
            arities = {
                "point": {1},
                "point.distance": {2},
                "point.withinbbox": {3},
                "substring": {2, 3},
            }
            if len(expr["arguments"]) not in arities.get(expr["name"], {1}):
                raise ValueError(f"Unsupported argument count for {expr['name']}")
    if not any(op["kind"] == "project" for op in operations):
        raise ValueError("Retrieval semantics require a RETURN projection")
    TaskSemantics.model_validate({"operations": operations})
    return {"version": SEMANTICS_VERSION, "operations": operations}


def parse_query(query: str) -> dict[str, Any]:
    return deepcopy(_parse_query(query))


def flattened_bindings(variables: dict[str, Any]) -> dict[str, Any]:
    result = {}

    def visit(value, path):
        result[path] = value
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))

    for key, value in variables.items():
        visit(value, str(key))
    return result


def resolve_semantics(spec: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    TaskSemantics.model_validate(spec)
    bindings = flattened_bindings(variables)

    def resolve(value):
        if isinstance(value, dict):
            if value.get("kind") == "binding":
                name = value["name"]
                if name not in bindings:
                    raise ValueError(f"Unresolved semantic binding: {name}")
                return literal(bindings[name])
            return {key: resolve(child) for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child) for child in value]
        if isinstance(value, str) and re.fullmatch(r"\{[A-Za-z_][A-Za-z0-9_.]*\}", value):
            name = value[1:-1]
            if name not in bindings:
                raise ValueError(f"Unresolved semantic binding: {name}")
            return bindings[name]
        return value

    return resolve(spec)


def _canonical(spec: dict[str, Any]) -> str:
    # Defaults and ordering of independent conjunctions do not change meaning.
    payload = TaskSemantics.model_validate(spec).model_dump(exclude_defaults=True)
    # Local variable names are implementation choices; returned column aliases are a contract.
    aliases = {}
    for operation in payload["operations"]:
        if operation["kind"] == "match" and operation["role"] not in aliases:
            aliases[operation["role"]] = f"entity_{len(aliases)}"
        if operation["kind"] == "with":
            for item in operation["items"]:
                if item.get("alias") and item["alias"] not in aliases:
                    aliases[item["alias"]] = f"value_{len(aliases)}"
        if operation["kind"] == "unwind" and operation["alias"] not in aliases:
            aliases[operation["alias"]] = f"value_{len(aliases)}"

    def rename(value):
        if isinstance(value, dict):
            result = {key: rename(child) for key, child in value.items()}
            for key in ("role", "source", "target"):
                if isinstance(result.get(key), str):
                    result[key] = aliases.get(result[key], result[key])
            if result.get("kind") == "ref":
                head, separator, tail = result["name"].partition(".")
                result["name"] = aliases.get(head, head) + separator + tail
            if result.get("kind") in {"with", "unwind"}:
                if result["kind"] == "with":
                    for item in result["items"]:
                        if item.get("alias"):
                            item["alias"] = aliases.get(item["alias"], item["alias"])
                else:
                    result["alias"] = aliases[result["alias"]]
            return result
        return [rename(child) for child in value] if isinstance(value, list) else value

    payload = rename(payload)

    def normalize(value):
        if isinstance(value, dict):
            value = {key: normalize(child) for key, child in value.items()}
            if value.get("kind") == "logical" and value.get("operator") in {"and", "or"}:
                value["arguments"] = sorted(
                    value["arguments"], key=lambda x: json.dumps(x, sort_keys=True)
                )
        elif isinstance(value, list):
            value = [normalize(child) for child in value]
        return value

    return json.dumps(normalize(payload), sort_keys=True, separators=(",", ":"))


def validate_query_semantics(spec: dict[str, Any], query: str) -> None:
    if _canonical(spec) != _canonical(parse_query(query)):
        raise ValueError("Cypher does not match its declared retrieval semantics")


def semantic_fingerprint(spec: dict[str, Any], goal: Any = None, fields: Any = None) -> str:
    payload = _canonical(spec) + json.dumps([goal, fields or {}], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_semantic_context(variables, template, query):
    from cypher_3dsg_agent_distillation.cypher.validation_config import resolve_contract

    spec = template.get("semantics") or parse_query(template["cypher"])
    resolved = resolve_semantics(spec, variables)
    validate_query_semantics(resolved, query)
    contract = resolve_contract(template["validation_contract"], variables)
    fields = contract["result"]["roles"]
    goal = contract["goal"]
    validate_goal_semantics(goal, fields)
    return {
        "version": REASONING_CONTEXT_VERSION,
        "renderer_version": REASONING_RENDERER_VERSION,
        "retrieval": resolved,
        "goal": deepcopy(goal),
        "grounding_fields": deepcopy(fields),
        "variables": deepcopy(variables),
        "fingerprint": semantic_fingerprint(resolved, goal, fields),
    }


def prepare_seed_semantics(seed, template):
    """Prepare a current synthetic context from explicit generation bindings."""
    from cypher_3dsg_agent_distillation.cypher.validation_config import (
        ValidationContract,
        resolve_contract,
        template_bindings,
    )

    variables = template_bindings(seed, template)
    saved = ValidationContract.model_validate(seed["validation_contract"]).model_dump(mode="json")
    selected = resolve_contract(template["validation_contract"], variables)
    # Scene scores are observations; all declarative role, goal and comparison
    # rules must still agree before creating narration from the selected catalog.
    for section in ("result", "answer"):
        saved[section]["comparison"].pop("scores", None)
        selected[section]["comparison"].pop("scores", None)
    if saved != selected:
        raise ValueError(
            "Saved validation contract differs from the selected catalog; "
            "refresh current contracts or regenerate seeds."
        )
    query = seed.get("cypher") or seed["gold_cypher"]
    context = build_semantic_context(variables, template, query)
    existing = seed["metadata"].get("reasoning_context")
    if existing is not None:
        validate_semantic_context(existing, query)
        if existing["fingerprint"] != context["fingerprint"]:
            raise ValueError(
                "Synthetic context disagrees with the current contract/catalog; regenerate seeds."
            )
    seed["metadata"]["reasoning_context"] = context
    return context


def validate_goal_semantics(goal, fields):
    if goal is None:
        return
    from cypher_3dsg_agent_distillation.graph.heracles_validation import (
        PDDL_PREDICATE_SIGNATURES,
        PDDL_SYMBOL_PATTERNS,
    )

    def visit(spec):
        if not isinstance(spec, dict):
            raise ValueError("Goal clauses must be mappings")
        if "fact" in spec:
            signature = PDDL_PREDICATE_SIGNATURES.get(spec["fact"])
            if signature is None:
                raise ValueError(f"Unknown PDDL predicate: {spec['fact']}")
            arguments = spec.get("arguments", [])
            if len(arguments) != len(signature):
                raise ValueError(f"Wrong argument count for {spec['fact']}")
            if spec.get("multiple", "or") not in {"and", "or"}:
                raise ValueError("Goal multiple must be and or or")
            for argument, symbol_type in zip(arguments, signature, strict=True):
                if isinstance(argument, dict) and "result" in argument:
                    field = argument["result"]
                    if fields and field not in fields:
                        raise ValueError(f"Goal refers to undeclared grounding field: {field}")
                    from cypher_3dsg_agent_distillation.cypher.validation_config import (
                        symbol_type as role_symbol_type,
                    )

                    actual_type = role_symbol_type(fields.get(field, {}).get("layers", []))
                    allowed = {symbol_type} | (
                        {"Place", "MeshPlace"} if symbol_type == "PlaceOrMeshPlace" else set()
                    )
                    if actual_type and actual_type not in allowed:
                        raise ValueError(f"Grounding field {field} has the wrong predicate type")
                else:
                    symbol = argument.get("literal") if isinstance(argument, dict) else argument
                    if not isinstance(symbol, str) or not PDDL_SYMBOL_PATTERNS[
                        symbol_type
                    ].fullmatch(symbol):
                        raise ValueError(f"Invalid {symbol_type} goal argument: {symbol}")
        else:
            operator = spec.get("operator")
            clauses = spec.get("clauses", [])
            if operator not in {"and", "or", "not"} or not clauses:
                raise ValueError("Goal requires nonempty and/or/not clauses")
            if operator == "not" and len(clauses) != 1:
                raise ValueError("Goal not requires exactly one clause")
            for child in clauses:
                visit(child)

    visit(goal)


def validate_semantic_context(context, query):
    if context.get("version") != REASONING_CONTEXT_VERSION or "retrieval" not in context:
        raise ValueError(
            "A current verified synthetic reasoning context is required; regenerate seeds."
        )
    validate_query_semantics(context["retrieval"], query)
    if context.get("fingerprint") != semantic_fingerprint(
        context["retrieval"], context.get("goal"), context.get("grounding_fields")
    ):
        raise ValueError("Stored semantic context fingerprint does not match its content")
    validate_goal_semantics(context.get("goal"), context.get("grounding_fields", {}))
