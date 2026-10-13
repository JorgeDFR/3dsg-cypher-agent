"""Short, deterministic sentences selected from the task semantics and tool outcome."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Sentence:
    rule: str
    priority: int
    text: str


def sentence_cap(config: dict[str, Any], stage: str) -> int | None:
    value = config.get(f"max_{stage}_sentences", config.get("max_sentences"))
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError("Synthetic reasoning sentence caps must be nonnegative integers or null.")
    return value


def select_sentences(
    candidates: list[Sentence],
    *,
    cap: int | None,
) -> tuple[str, dict[str, Any]]:
    """Sort relevant rules by priority, then stable rule ID; deduplicate before capping."""
    selected = []
    seen = set()
    for sentence in sorted(candidates, key=lambda item: (item.priority, item.rule)):
        if sentence.text in seen:
            continue
        seen.add(sentence.text)
        selected.append(sentence)
    kept = selected if cap is None else selected[:cap]
    dropped = selected[len(kept) :]
    return " ".join(sentence.text for sentence in kept), {
        "cap": cap,
        "candidate_count": len(selected),
        "selected_rules": [sentence.rule for sentence in kept],
        "omitted_rules": [sentence.rule for sentence in dropped],
        "selection_order": "priority_then_rule_id",
    }


def _classes(seed: dict[str, Any]) -> list[tuple[str, str]]:
    context = (seed.get("metadata") or {}).get("reasoning_context") or {}
    classes = [(str(item["layer"]), str(item["label"])) for item in context.get("classes", [])]
    if classes:
        return sorted(set(classes))
    # Compatibility with existing seeds: read schema restrictions, never solution values.
    cypher = str(seed.get("cypher") or seed.get("gold_cypher") or "")
    return _class_filters(cypher)


def _class_filters(cypher: str) -> list[tuple[str, str]]:
    classes = []
    for match in re.finditer(
        r"\(\w+:(Object|Room)\s*\{[^}]*class:\s*[\"']([^\"']+)[\"']",
        cypher,
    ):
        classes.append((match.group(1), match.group(2)))
    aliases = dict(re.findall(r"\((\w+):(Object|Room)\b", cypher))
    for alias, layer in aliases.items():
        for match in re.finditer(
            rf"\b{re.escape(alias)}\.class\s*(?:=|<>|(?:NOT\s+)?IN)\s*"
            r"(\[[^\]]*\]|\"[^\"]*\"|'[^']*')",
            cypher,
        ):
            classes.extend(
                (layer, label)
                for label in re.findall(
                    r"[\"']([^\"']+)[\"']",
                    match.group(1),
                )
            )
    return sorted(set(classes))


def semantic_context(
    variables: dict[str, Any],
    template: dict[str, Any],
    cypher: str,
) -> dict[str, Any]:
    """Retain bindings and goal structure lost by the old seed-generation pipeline."""
    return {
        "variables": variables,
        "goal": template.get("goal"),
        "variable_sources": template.get("variables", {}),
        "classes": [{"layer": layer, "label": label} for layer, label in _class_filters(cypher)],
    }


def _goal_sentences(spec: dict[str, Any], path: str = "goal") -> list[Sentence]:
    sentences = []
    meanings = {
        "safe": "inspection",
        "holding": "holding an object",
        "object-in-place": "placing an object at its destination",
        "visited-object": "an intermediate object visit",
        "visited-place": "an intermediate place visit",
        "visited-region": "an intermediate room visit",
        "at-object": "the final object destination",
        "at-place": "the final place destination",
        "in-region": "the final room destination",
    }
    fact = spec.get("fact")
    if fact:
        sentences.append(
            Sentence(f"{path}:predicate", 40, f"Map {meanings.get(fact, fact)} to {fact}.")
        )
        if spec.get("negate_each"):
            sentences.append(
                Sentence(f"{path}:negate_each", 30, "Negate each matching condition with not.")
            )
        if spec.get("multiple") in {"and", "or"}:
            op = spec["multiple"]
            meaning = "every matching target" if op == "and" else "any matching target"
            fields = [
                str(arg["result"])
                for arg in spec.get("arguments", [])
                if isinstance(arg, dict) and "result" in arg
            ]
            scope = ", ".join(fields) or path
            sentences.append(
                Sentence(
                    f"{path}:multiple", 30, f"Use {op} for {meaning} in the {scope} condition."
                )
            )
    operator = spec.get("operator")
    if operator in {"and", "or", "not"}:
        meaning = {
            "and": "all requested conditions",
            "or": "the alternative branches",
            "not": "the enclosed forbidden condition",
        }[operator]
        sentences.append(
            Sentence(f"{path}:operator", 25, f"Combine {meaning} with {operator} at {path}.")
        )
    for index, clause in enumerate(spec.get("clauses", [])):
        sentences.extend(_goal_sentences(clause, f"{path}.{index + 1}"))
    return sentences


def reasoning_sentences(
    seed: dict[str, Any],
    *,
    stage: str,
    rows: list[dict[str, Any]] | None = None,
    repair_template: str | None = None,
) -> list[Sentence]:
    question = str(seed["question"])
    lower = question.lower()
    tags = set(seed.get("tags") or [])
    context = (seed.get("metadata") or {}).get("reasoning_context") or {}
    sentences = []
    if stage == "tool":
        cypher = str(seed.get("cypher") or seed.get("gold_cypher") or "")
        restricted_layers = {layer for layer, _label in _classes(seed)}
        symbol_layers = set()
        for symbol in sorted(set(re.findall(r"\b[OR]\d+\b", question))):
            layer = "Object" if symbol.startswith("O") else "Room"
            symbol_layers.add(layer)
            sentences.append(Sentence(f"entity:{symbol}", 10, f"Resolve {layer} symbol {symbol}."))
        for layer in sorted(set(re.findall(r":(Object|Room)\b", cypher))):
            if layer not in restricted_layers | symbol_layers:
                sentences.append(Sentence(f"entity:{layer}", 10, f"Query {layer} nodes."))
        fixes = {
            "misspelled_match_clause": "Correct MATC to MATCH.",
            "unclosed_match_parenthesis": "Close the MATCH pattern parenthesis.",
            "missing_property_map_value": "Restore the missing property value.",
            "undefined_return_variable": "Use the variable bound by MATCH in RETURN.",
            "unknown_function": "Replace the unknown function with the valid Cypher function.",
            "missing_projection_comma": "Separate RETURN projections with a comma.",
            "premature_return_clause": "Move RETURN after the remaining query clauses.",
        }
        if repair_template:
            sentences.append(
                Sentence(
                    "repair",
                    0,
                    fixes.get(repair_template, "Correct the Cypher syntax using the tool error."),
                )
            )
        symbols = sorted(set(re.findall(r"\b[pP]\d+\b", question)))
        for symbol in symbols:
            layer = "Place" if symbol.startswith("p") else "MeshPlace"
            sentences.append(
                Sentence(f"layer:{symbol}", 10, f"Symbol {symbol} identifies {layer}.")
            )
        if not symbols and ("place" in lower or "location" in lower):
            layer = "MeshPlace" if re.search(r"mesh[- ]?place", lower) else "Place or MeshPlace"
            sentences.append(
                Sentence("layer:unspecified", 10, f"Resolve the requested location using {layer}.")
            )
        labelspace = str(seed.get("labelspace_description") or "")
        for layer, label in _classes(seed):
            section = "object_labels" if layer == "Object" else "room_labels"
            match = re.search(rf"<{section}>(.*?)</{section}>", labelspace, re.DOTALL)
            if match:
                labels = {line.strip() for line in match.group(1).splitlines() if line.strip()}
                available = "is" if label in labels else "is not"
                sentences.append(
                    Sentence(
                        f"class:{layer}:{label}",
                        15,
                        f"{label} {available} in the {layer.lower()} label space.",
                    )
                )
            else:
                sentences.append(
                    Sentence(
                        f"class:{layer}:{label}", 15, f"Filter {layer} nodes by class {label}."
                    )
                )
        if "nearest" in lower or "closest" in lower:
            sentences.append(
                Sentence(
                    "distance:nearest",
                    50,
                    "Rank by ascending distance and select the nearest target.",
                )
            )
        elif "farthest" in lower:
            sentences.append(
                Sentence(
                    "distance:farthest",
                    50,
                    "Rank by descending distance and select the farthest target.",
                )
            )
        if "aggregation" in tags:
            sentences.append(
                Sentence("aggregation", 50, "Aggregate the matched nodes in the query.")
            )
        if seed.get("task_family") == "cypher_pddl":
            sentences.extend(_goal_sentences(context.get("goal") or {}))
            if not context.get("goal"):
                # Legacy tags supply semantics without inventing unseen goal structure.
                for tag, text in {
                    "negation": "Use not for forbidden conditions.",
                    "disjunction": "Use or to let the planner choose an eligible alternative.",
                    "universal": "Use and to require every matching target.",
                    "conjunction": "Require all command parts with and.",
                }.items():
                    if tag in tags:
                        sentences.append(Sentence(f"legacy:{tag}", 30, text))
            if "ordering" in tags:
                sentences.append(
                    Sentence(
                        "ordering",
                        20,
                        "Use visited predicates for intermediate stops and an "
                        "at or in predicate for the final destination.",
                    )
                )
    else:
        if "WRITE_QUERY_NOT_ALLOWED" in str(seed.get("cypher") or seed.get("gold_cypher")):
            sentences.append(
                Sentence("outcome:refusal", 0, "Writes are not allowed; return the refusal token.")
            )
        no_grounding = rows == []
        if seed.get("task_family") == "cypher_pddl" and rows is not None:
            from cypher_3dsg_agent_distillation.cypher.pddl_goals import grounding_from_rows

            contract = seed.get("grounding") or {}
            if contract.get("fields"):
                no_grounding = not any(grounding_from_rows(rows, contract).values())
        if no_grounding:
            text = (
                "No grounding was returned; leave the PDDL answer empty."
                if seed.get("task_family") == "cypher_pddl"
                else "No matches were returned; use the required empty value."
            )
            sentences.append(Sentence("outcome:empty", 0, text))
        elif rows is not None:
            sentences.append(
                Sentence("outcome:grounded", 10, "Use the returned rows to construct the answer.")
            )
        formats = seed.get("solution_formats") or seed.get("solution_format") or []
        if isinstance(formats, str):
            formats = [formats]
        if formats == ["int"] and rows and len(rows) == 1 and list(rows[0].values()) == [0]:
            sentences.append(Sentence("outcome:zero", 0, "The returned count is zero; answer 0."))
        syntax = (
            "PDDL goal"
            if seed.get("task_family") == "cypher_pddl"
            else (" or ".join(formats) if formats else "SLDP value")
        )
        sentences.append(Sentence("answer:format", 5, f"Return the {syntax} inside answer tags."))
    return sentences


def add_synthetic_reasoning(
    message: dict[str, Any],
    seed: dict[str, Any],
    config: dict[str, Any],
    *,
    stage: str,
    rows: list[dict[str, Any]] | None = None,
    repair_template: str | None = None,
) -> None:
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
