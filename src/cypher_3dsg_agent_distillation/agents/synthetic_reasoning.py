"""Stage-specific decisions selected from validated operations and tool feedback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from cypher_3dsg_agent_distillation.cypher.pddl_goals import render_goal_spec
from cypher_3dsg_agent_distillation.cypher.task_semantics import (
    REASONING_RENDERER_VERSION,
    validate_semantic_context,
    walk_expression,
)


@dataclass(frozen=True)
class Sentence:
    rule: str
    priority: int
    text: str
    order: int = 0
    scope: str = "task"
    critical: bool = False


def sentence_cap(config: dict[str, Any], stage: str) -> int | None:
    value = config.get(f"max_{stage}_sentences", config.get("max_sentences"))
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError("Synthetic reasoning sentence caps must be nonnegative integers or null.")
    return value


def select_sentences(candidates: list[Sentence], *, cap: int | None):
    selected = []
    seen = set()
    for sentence in sorted(candidates, key=lambda item: (item.priority, item.rule)):
        key = (sentence.scope, sentence.text)
        if key not in seen:
            seen.add(key)
            selected.append(sentence)
    kept = selected if cap is None else selected[:cap]
    dropped = selected[len(kept) :]
    rendered = sorted(kept, key=lambda item: (item.order, item.priority, item.rule))
    return " ".join(item.text for item in rendered), {
        "cap": cap,
        "candidate_count": len(selected),
        "selected_rules": [item.rule for item in kept],
        "omitted_rules": [item.rule for item in dropped],
        "critical_omitted_rules": [item.rule for item in dropped if item.critical],
        "rendered_rules": [item.rule for item in rendered],
        "selection_order": "priority_then_rule_id",
        "render_order": "semantic_order",
        "renderer_version": REASONING_RENDERER_VERSION,
    }


def _context(seed):
    context = (seed.get("metadata") or {}).get("reasoning_context") or {}
    query = str(seed.get("cypher") or seed.get("gold_cypher") or "")
    validate_semantic_context(context, query)
    return context


def _value(expr):
    return expr.get("value") if expr.get("kind") == "literal" else None


def _expand(expr, aliases):
    if isinstance(expr, list):
        return [_expand(child, aliases) for child in expr]
    if not isinstance(expr, dict):
        return expr
    if expr.get("kind") == "ref":
        name = expr["name"]
        if name in aliases:
            return aliases[name]
        head, _, tail = name.partition(".")
        mapped = aliases.get(head)
        if mapped and mapped.get("kind") == "map" and tail in mapped["entries"]:
            return mapped["entries"][tail]
    return {key: _expand(value, aliases) for key, value in expr.items()}


class QueryView:
    def __init__(self, spec):
        self.nodes = {}
        self.operations = []
        self.outputs = []
        self.output_columns = []
        self.limit = None
        aliases = {}
        for operation in spec["operations"]:
            kind = operation["kind"]
            op = _expand(operation, aliases)
            self.operations.append(op)
            if kind == "match":
                node = self.nodes.setdefault(op["role"], {"layers": [], "properties": {}})
                node["layers"] = sorted(set(node["layers"] + op.get("layers", [])))
                node["properties"].update(op.get("properties", {}))
                node["optional"] = op.get("optional", False)
            if kind in {"with", "project"}:
                for item in op["items"]:
                    if item.get("alias"):
                        aliases[item["alias"]] = item["expression"]
                if kind == "project":
                    self.outputs = op["items"]
                    self.output_columns = [
                        item.get("alias")
                        or (
                            item["expression"]["name"]
                            if item["expression"]["kind"] == "ref"
                            else None
                        )
                        for item in operation["items"]
                    ]
            elif kind == "unwind":
                expr = op["expression"]
                if expr.get("kind") == "call" and expr["name"] == "collect":
                    aliases[op["alias"]] = expr["arguments"][0]
            elif kind == "limit":
                self.limit = op["value"]
        for op in self.operations:
            for expr in walk_expression(op.get("expression", {})):
                if expr["kind"] == "label" and expr["role"] in self.nodes:
                    self.nodes[expr["role"]]["layers"].append(expr["layer"])
        for node in self.nodes.values():
            node["layers"] = sorted(set(node["layers"]))

    def target(self, role, *, include_symbol=True):
        node = self.nodes.get(role, {})
        names = {"Object": "object", "Room": "room", "Place": "place", "MeshPlace": "mesh-place"}
        layers = "/".join(names.get(layer, layer) for layer in node.get("layers", [])) or "node"
        props = node.get("properties", {})
        symbol = _value(props.get("nodeSymbol", {}))
        label = _value(props.get("class", {}))
        if symbol is not None:
            return f"{layers} {symbol}" if include_symbol else layers
        if label is not None:
            target = f"{label} {layers}s"
            if any(
                other != role
                and node.get("layers", []) == other_node.get("layers", [])
                and _value(other_node.get("properties", {}).get("class", {})) == label
                for other, other_node in self.nodes.items()
            ):
                qualifier = {"source": "reference", "target": "candidate"}.get(role, role)
                target = f"{qualifier} {target}"
            return target
        readable_role = {"source": "reference", "target": "candidate", "start": "starting"}.get(
            role
        )
        if readable_role:
            return f"{readable_role} {layers}s"
        if role in {"first", "second", "neighbor", "reference", "destination", "waypoint"}:
            return f"{role} {layers}s"
        return f"{layers}s"

    def describe(self, expr):
        kind = expr["kind"]
        if kind == "literal":
            return str(expr["value"])
        if kind == "ref":
            role, _, prop = expr["name"].partition(".")
            target = self.target(role)
            if prop == "center":
                return f"{target} center"
            if prop == "nodeSymbol":
                return f"{target} symbol"
            if prop == "class":
                return f"{target} class"
            return target if not prop else f"{target} {prop}"
        if kind == "list":
            return "[" + ", ".join(self.describe(child) for child in expr["items"]) + "]"
        if kind == "map":
            return "[" + ", ".join(self.describe(value) for value in expr["entries"].values()) + "]"
        if kind == "call":
            name = expr["name"]
            args = expr["arguments"]
            if name == "point":
                return f"point {self.describe(args[0])}"
            if name == "point.distance":
                targets = [
                    self.target(arg["name"].split(".")[0])
                    if arg.get("kind") == "ref" and arg["name"].endswith(".center")
                    else self.describe(arg)
                    for arg in args
                ]
                return f"center distance({', '.join(targets)})"
            if name == "count":
                prefix = "distinct " if expr.get("distinct") else ""
                return f"count({prefix}{self.describe(args[0])})"
            if name in {"min", "max"}:
                return f"{'minimum' if name == 'min' else 'maximum'} {self.describe(args[0])}"
            return f"{name}({', '.join(self.describe(arg) for arg in args)})"
        return json.dumps(expr, sort_keys=True)

    def grounding_target(self, field):
        for item in self.outputs:
            if item.get("alias") == field:
                expr = item["expression"]
                if expr["kind"] == "ref":
                    role = expr["name"].split(".")[0]
                    target = self.target(role, include_symbol=False)
                    # Distinguish equal-looking targets without repeating returned symbols.
                    peers = [
                        other
                        for other in self.outputs
                        if other != item
                        and other["expression"]["kind"] == "ref"
                        and self.target(
                            other["expression"]["name"].split(".")[0], include_symbol=False
                        )
                        == target
                    ]
                    return f"{field} {target}" if peers else target
        return str(field)


def _goal_description(spec, view, grounding):
    """Describe the grounded goal's structure using target roles, never result values."""
    if "fact" in spec:
        fact = spec["fact"]
        targets = []
        multiple = False
        for arg in spec.get("arguments", []):
            if isinstance(arg, dict) and "result" in arg:
                field = arg["result"]
                targets.append(view.grounding_target(field))
                multiple |= len(grounding.get(field, [])) > 1
            else:
                targets.append(str(arg.get("literal", arg)) if isinstance(arg, dict) else str(arg))
        text = f"{fact}({', '.join(targets)})"
        if spec.get("negate_each"):
            text = f"NOT({text})"
        if multiple:
            text = f"{'ALL' if spec.get('multiple') == 'and' else 'ANY'} {text}"
        return text
    operator = spec["operator"]
    descriptions = [_goal_description(child, view, grounding) for child in spec["clauses"]]
    if operator == "not":
        return f"NOT({descriptions[0]})"
    if operator == "or" and len(descriptions) == 1:
        return descriptions[0]
    return f"{operator.upper()}({', '.join(descriptions)})"


def _repair_sentence(seed, template):
    broken = str((seed.get("metadata") or {}).get("repair_cypher") or "")
    corrected = str(seed.get("cypher") or seed.get("gold_cypher") or "")
    fixes = {
        "misspelled_match_clause": "Replace MATC with MATCH.",
        "unclosed_match_parenthesis": "Close the MATCH parenthesis.",
        "missing_projection_comma": "Add the missing RETURN comma.",
        "premature_return_clause": "Move RETURN after the query clauses.",
    }
    if template in fixes:
        return fixes[template]
    if broken:
        edits = [
            (broken[a:b].strip(), corrected[c:d].strip())
            for kind, a, b, c, d in SequenceMatcher(None, broken, corrected).get_opcodes()
            if kind != "equal"
        ]
        if edits:
            before, after = edits[0]
            if template == "unknown_function":
                for function in re.findall(r"\b([A-Za-z_][\w.]*)\s*\(", corrected):
                    if broken.replace("unknownCypherFunction", function, 1) == corrected:
                        return f"Replace unknownCypherFunction with {function}."
            if template == "undefined_return_variable":
                match = re.search(r"RETURN\s+(?:DISTINCT\s+)?(\w+)\.", corrected, re.I)
                if match:
                    return f"Replace undefined_record with {match.group(1)} in RETURN."
            if after:
                return f"{'Replace ' + before + ' with' if before else 'Restore'} {after}."
    return {
        "undefined_return_variable": "Use the bound variable in RETURN.",
        "missing_property_map_value": "Restore the property value.",
        "unknown_function": "Fix the function from tool feedback.",
    }.get(template, "Fix Cypher from tool feedback.")


def _expression_key(expr):
    return json.dumps(expr, sort_keys=True)


def _expression_roles(expr):
    roles = set()
    for child in walk_expression(expr):
        if child["kind"] == "ref":
            roles.add(child["name"].split(".")[0])
        elif child["kind"] == "label":
            roles.add(child["role"])
    return roles


def _aggregate_groups(view):
    groups = {}
    for op in view.operations:
        if op["kind"] not in {"with", "project"}:
            continue
        grouping = [
            view.describe(item["expression"])
            for item in op["items"]
            if not any(
                child.get("name") in {"count", "collect", "min", "max"}
                for child in walk_expression(item["expression"])
                if child["kind"] == "call"
            )
        ]
        for item in op["items"]:
            for expr in walk_expression(item["expression"]):
                if expr["kind"] == "call" and expr["name"] == "count":
                    groups.setdefault(_expression_key(expr), grouping)
    return groups


def _numeric_symbol_role(expr):
    if expr.get("kind") != "call" or expr.get("name") != "tointeger":
        return None
    arg = expr["arguments"][0]
    if arg.get("kind") != "call" or arg.get("name") != "substring":
        return None
    source, offset, *rest = arg["arguments"]
    if (
        source.get("kind") == "ref"
        and source["name"].endswith(".nodeSymbol")
        and _value(offset) == 1
        and not rest
    ):
        return source["name"].split(".")[0]
    return None


def _filter_description(condition, view):
    kind = condition["kind"]
    if kind == "logical":
        children = [_filter_description(child, view) for child in condition["arguments"]]
        return f"{condition['operator'].upper()}({', '.join(children)})"
    if kind == "label":
        return f"{condition['role']}: {condition['layer']}"
    if kind == "call" and condition["name"] == "point.withinbbox":
        center, lower, upper = condition["arguments"]
        return (
            f"{view.describe(center)} in inclusive box "
            f"{view.describe(lower)} .. {view.describe(upper)}"
        )
    if kind == "compare":
        left, right = condition["left"], condition["right"]
        operator = condition["operator"]
        if (
            left["kind"] == right["kind"] == "ref"
            and operator == "<>"
            and "." not in left["name"]
            and "." not in right["name"]
        ):
            return f"Distinct {view.target(left['name'])} and {view.target(right['name'])}"
        if operator in {"<", ">"} and _numeric_symbol_role(left) and _numeric_symbol_role(right):
            order = "ascending" if operator == "<" else "descending"
            return f"Unordered pairs once ({order} numeric symbol IDs)"
        if (
            operator == "="
            and right["kind"] == "call"
            and right["name"] in {"min", "max"}
            and left == right["arguments"][0]
        ):
            extreme = "minimum" if right["name"] == "min" else "maximum"
            return f"Keep all {extreme} {view.describe(left)} ties"
        return f"{view.describe(left)} {operator} {view.describe(right)}"
    return view.describe(condition)


def _tool_sentences(seed, view, repair_template):
    sentences = []

    def add(rule, text, priority=20, order=20, scope="task", critical=True):
        sentences.append(Sentence(rule, priority, text, order, scope, critical))

    if repair_template:
        add("repair", _repair_sentence(seed, repair_template), 0, -1)
    for role, node in view.nodes.items():
        props = node["properties"]
        label = _value(props.get("class", {}))
        section = "room_labels" if "Room" in node["layers"] else "object_labels"
        match = re.search(
            rf"<{section}>(.*?)</{section}>", str(seed.get("labelspace_description") or ""), re.S
        )
        if (
            label is not None
            and match
            and label not in {line.strip() for line in match.group(1).splitlines() if line.strip()}
        ):
            add(
                f"label:unlisted:{role}",
                f"Unlisted class {label}; query for instances.",
                30,
                0,
                f"{section}:{label}",
                False,
            )

    groups = _aggregate_groups(view)
    filters = {}
    explained = set()
    for index, op in enumerate(view.operations):
        if op["kind"] == "filter":
            text = _filter_description(op["expression"], view)
            grouping = list(
                dict.fromkeys(
                    group
                    for child in walk_expression(op["expression"])
                    for group in groups.get(_expression_key(child), [])
                )
            )
            if grouping:
                text += "; per " + ", ".join(grouping)
            filters[index] = text
        if op["kind"] == "filter" or op["kind"] == "order" and view.limit is not None:
            explained.update(_expression_key(child) for child in walk_expression(op))

    seen = set()
    for index, op in enumerate(view.operations):
        kind = op["kind"]
        if (
            kind == "match"
            and op.get("optional")
            and not any(
                traversal["kind"] == "traverse"
                and traversal.get("optional")
                and op["role"] in {traversal["source"], traversal["target"]}
                for traversal in view.operations
            )
        ):
            add(f"optional:{index}", f"Allow missing {view.target(op['role'])}.", order=20 + index)
        if kind == "traverse":
            low, high = op["bounds"]
            source, target = view.target(op["source"]), view.target(op["target"])
            arrow = {"outgoing": "->", "incoming": "<-", "undirected": "<->"}[op["direction"]]
            depth = str(low) if low == high else f"{low}..{high}"
            text = f"{source} {arrow} {target}: {op['relationship']} {depth} hops"
            if op["relationship"] == "CONTAINS" and low == high == 2:
                text += " via Place/MeshPlace"
            if op["relationship"] != "CONTAINS":
                text += "; same layer"
            if op.get("optional"):
                text += "; keep empty starts"
            add(f"traversal:{index}", text + ".", order=20 + index)
        for expr in walk_expression(op):
            if expr["kind"] != "call":
                continue
            key = _expression_key(expr)
            if key in seen or key in explained:
                continue
            seen.add(key)
            name = expr["name"]
            if name == "count" and (groups.get(key) or expr.get("distinct")):
                group = " per " + ", ".join(groups[key]) if groups.get(key) else ""
                add(
                    f"aggregation:{index}",
                    f"{view.describe(expr)}{group}.",
                    order=20 + index,
                )
            elif name in {"min", "max"}:
                add(
                    f"extremum:{index}",
                    f"{view.describe(expr)}.",
                    order=20 + index,
                )
        if kind == "order" and view.limit is not None:
            first = op["items"][0]
            metric = first["expression"]
            ascending = first["direction"] == "ASC"
            if metric.get("kind") == "call" and metric["name"] == "point.distance":
                related = [
                    i
                    for i in filters
                    if i < index
                    and _expression_roles(view.operations[i]["expression"])
                    and _expression_roles(view.operations[i]["expression"])
                    <= _expression_roles(metric)
                ]
                already_described = any(
                    metric == child
                    for i in related
                    for child in walk_expression(view.operations[i]["expression"])
                )
                text = "Nearest" if ascending else "Farthest"
                if already_described:
                    text += " qualifying candidates"
                else:
                    text += f" by {view.describe(metric)}"
                if view.limit != 1:
                    text += f"; limit {view.limit}"
                if related:
                    text = "; ".join(filters.pop(i) for i in related) + "; " + text
            else:
                text = f"{view.describe(metric)} {first['direction']}; limit {view.limit}"
                grouping = groups.get(_expression_key(metric))
                if grouping:
                    text += "; per " + ", ".join(grouping)
            if len(op["items"]) > 1:
                text += "; ties: " + ", ".join(
                    f"{view.describe(item['expression'])} {item['direction']}"
                    for item in op["items"][1:]
                )
            add(f"selection:{index}", text + ".", order=20 + index)
    for index, text in filters.items():
        add(f"filter:{index}", text + ".", order=20 + index)
    return sentences


def _available_goal(spec, grounding):
    if "fact" in spec:
        return spec if render_goal_spec(spec, grounding) is not None else None
    children = [_available_goal(child, grounding) for child in spec["clauses"]]
    if spec["operator"] in {"and", "not"} and any(child is None for child in children):
        return None
    children = [child for child in children if child is not None]
    return {**spec, "clauses": children} if children else None


def _answer_format(formats):
    if any(f == "set[list[str]]" for f in formats):
        return "SLDP set of symbol pairs; preserve row pairs."
    if formats == ["set[point]"]:
        return "SLDP set of unique POINT(x y z); no coordinate commas."
    if any(f.startswith("set[") for f in formats):
        return "SLDP set of unique values."
    if any(f.startswith("list[") for f in formats):
        return "SLDP list; preserve order."
    if formats == ["point"]:
        return "SLDP POINT(x y z); no commas."
    if formats == ["str"]:
        return "SLDP unquoted string."
    if formats == ["int"]:
        return "SLDP integer."
    return "Declared SLDP format."


def _final_sentences(seed, context, view, rows):
    formats = seed["validation_contract"]["answer"]["formats"] or []
    formats = [formats] if isinstance(formats, str) else formats
    if any(op["kind"] == "refusal" for op in view.operations):
        return [
            Sentence(
                "outcome:refusal",
                0,
                "Read-only database.",
                critical=True,
            ),
            Sentence("answer:format", 30, _answer_format(formats), order=30),
        ]
    if seed.get("task_family") == "cypher_pddl":
        goal = context.get("goal")
        from cypher_3dsg_agent_distillation.cypher.result_validation import decode_result

        decoded = decode_result(
            query=str(seed.get("cypher") or seed.get("gold_cypher")),
            rows=rows or [],
            contract=seed["validation_contract"],
        )
        if decoded.status != "pass":
            raise ValueError("Cannot narrate unresolved PDDL result grounding")
        grounding = decoded.actual
        available = _available_goal(goal, grounding) if goal and rows is not None else None
        if rows is not None and (not any(grounding.values()) or goal and available is None):
            missing = (
                "Required goal branch unavailable"
                if any(grounding.values())
                else "No grounded targets"
            )
            return [Sentence("outcome:empty", 0, f"{missing}; empty PDDL goal.", critical=True)]
        result = []
        if available:
            if available != goal:
                result.append(
                    Sentence(
                        "outcome:partial_alternative",
                        0,
                        "Drop ungrounded alternatives.",
                        order=0,
                        critical=True,
                    )
                )
            result.append(
                Sentence(
                    "answer:composition",
                    10,
                    _goal_description(available, view, grounding) + ".",
                    order=10,
                    critical=True,
                )
            )
        result.append(Sentence("answer:format", 30, "PDDL goal expression.", order=30))
        return result
    columns = view.output_columns
    matched_columns = (
        bool(rows) and all(columns) and all(column in row for row in rows for column in columns)
    )
    empty_projection = matched_columns and all(
        value is None or isinstance(value, (list, dict, str)) and len(value) == 0
        for row in rows
        for value in (row.get(column) for column in columns)
    )
    if rows == [] or empty_projection:
        empty = (
            "Empty SLDP set: <>."
            if any(f.startswith("set[") for f in formats)
            else "Empty SLDP list: []."
            if any(f.startswith("list[") for f in formats)
            else "Empty SLDP string."
            if formats == ["str"]
            else "Empty SLDP value."
        )
        return [
            Sentence(
                "outcome:empty",
                0,
                empty,
                critical=True,
            )
        ]
    return [
        Sentence("answer:format", 30, _answer_format(formats), order=30),
    ]


def reasoning_sentences(seed, *, stage, rows=None, repair_template=None):
    if stage not in {"tool", "final"}:
        raise ValueError(f"Unknown reasoning stage: {stage}")
    context = _context(seed)
    view = QueryView(context["retrieval"])
    return (
        _tool_sentences(seed, view, repair_template)
        if stage == "tool"
        else _final_sentences(seed, context, view, rows)
    )


def add_synthetic_reasoning(message, seed, config, *, stage, rows=None, repair_template=None):
    if not config.get("enabled", False):
        return
    text, selection = select_sentences(
        reasoning_sentences(seed, stage=stage, rows=rows, repair_template=repair_template),
        cap=sentence_cap(config, stage),
    )
    message["reasoning_selection"] = selection
    if text:
        message["reasoning"] = text
        message["reasoning_kind"] = "synthetic"
