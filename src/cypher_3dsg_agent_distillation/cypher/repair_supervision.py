from __future__ import annotations

import random
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CypherSyntaxErrorTemplate:
    name: str
    description: str
    mutate: Callable[[str], str | None]

    def apply(self, cypher: str) -> str | None:
        mutated = self.mutate(cypher)
        if mutated is None or mutated == cypher:
            return None
        return mutated


def replace_first(pattern: str, replacement: str, cypher: str, *, flags: int = 0) -> str | None:
    mutated, count = re.subn(pattern, replacement, cypher, count=1, flags=flags)
    return mutated if count else None


def misspell_match_clause(cypher: str) -> str | None:
    return replace_first(r"\bMATCH\b", "MATC", cypher, flags=re.IGNORECASE)


def drop_match_closing_parenthesis(cypher: str) -> str | None:
    match = re.search(r"\bMATCH\b", cypher, flags=re.IGNORECASE)
    if match is None:
        return None
    closing_index = cypher.find(")", match.end())
    if closing_index < 0:
        return None
    return cypher[:closing_index] + cypher[closing_index + 1 :]


PROPERTY_VALUE_RE = re.compile(
    r"(?P<prefix>\{\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^,}\n]+)"
)


def remove_property_map_value(cypher: str) -> str | None:
    match = PROPERTY_VALUE_RE.search(cypher)
    if match is None:
        return None
    return cypher[: match.start()] + match.group("prefix") + cypher[match.end() :]


RETURN_PROPERTY_RE = re.compile(
    r"(?im)^(?P<prefix>\s*RETURN\s+(?:DISTINCT\s+)?)"
    r"(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\."
)


def use_undefined_return_variable(cypher: str) -> str | None:
    match = RETURN_PROPERTY_RE.search(cypher)
    if match is None:
        return None
    start, end = match.span("variable")
    return cypher[:start] + "undefined_record" + cypher[end:]


KNOWN_FUNCTION_RE = re.compile(
    r"(?i)\b(?:point\.distance|point\.withinBBox|point|count|collect|length)\s*(?=\()"
)


def use_unknown_function(cypher: str) -> str | None:
    return replace_first(KNOWN_FUNCTION_RE.pattern, "unknownCypherFunction", cypher)


RETURN_LINE_RE = re.compile(r"(?im)^(?P<projection>\s*RETURN\b[^\n]*)")


def omit_projection_separator(cypher: str) -> str | None:
    match = RETURN_LINE_RE.search(cypher)
    if match is None:
        return None
    insertion_index = match.end("projection")
    return (
        cypher[:insertion_index]
        + " count(*) AS missing_separator"
        + cypher[insertion_index:]
    )


MATCH_LINE_RE = re.compile(r"(?im)^(?P<match_clause>\s*(?:OPTIONAL\s+)?MATCH\b[^\n]*)")


def insert_premature_return(cypher: str) -> str | None:
    match = MATCH_LINE_RE.search(cypher)
    if match is None:
        return None
    if re.search(r"\bRETURN\b", cypher[match.end() :], flags=re.IGNORECASE) is None:
        return None
    insertion_index = match.end("match_clause")
    return cypher[:insertion_index] + "\nRETURN *" + cypher[insertion_index:]


CYPHER_SYNTAX_ERROR_TEMPLATES = (
    CypherSyntaxErrorTemplate(
        "misspelled_match_clause",
        "Misspell the first MATCH clause keyword.",
        misspell_match_clause,
    ),
    CypherSyntaxErrorTemplate(
        "unclosed_match_parenthesis",
        "Remove a closing parenthesis from the first MATCH pattern.",
        drop_match_closing_parenthesis,
    ),
    CypherSyntaxErrorTemplate(
        "missing_property_map_value",
        "Remove a value from a node or point property map.",
        remove_property_map_value,
    ),
    CypherSyntaxErrorTemplate(
        "undefined_return_variable",
        "Replace a returned property owner with an undefined variable.",
        use_undefined_return_variable,
    ),
    CypherSyntaxErrorTemplate(
        "unknown_function",
        "Replace a valid Cypher function with an unknown function.",
        use_unknown_function,
    ),
    CypherSyntaxErrorTemplate(
        "missing_projection_comma",
        "Add a second RETURN projection without its required comma.",
        omit_projection_separator,
    ),
    CypherSyntaxErrorTemplate(
        "premature_return_clause",
        "Insert RETURN before later query clauses.",
        insert_premature_return,
    ),
)


def applicable_repair_templates(cypher: str) -> list[tuple[CypherSyntaxErrorTemplate, str]]:
    applicable = []
    for template in CYPHER_SYNTAX_ERROR_TEMPLATES:
        mutated = template.apply(cypher)
        if mutated is not None:
            applicable.append((template, mutated))
    return applicable


def validate_repair_cypher_ratio(value: Any) -> float:
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"repair_cypher_ratio must be between 0 and 1, got {ratio}.")
    return ratio


def repair_cypher_sampling_ratio(
    template: dict[str, Any],
    sampling_config: dict[str, Any],
    override: float | None,
) -> float:
    value = (
        override
        if override is not None
        else template.get(
            "repair_cypher_ratio",
            sampling_config.get("repair_cypher_ratio", 0.0),
        )
    )
    return validate_repair_cypher_ratio(value)


def apply_controlled_repair_sampling(
    records: list[dict[str, Any]],
    *,
    ratios: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    records_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_type[str(record["query_type"])].append(record)

    template_usage: Counter[str] = Counter()
    for query_type, candidates in records_by_type.items():
        ratio = validate_repair_cypher_ratio(ratios.get(query_type, 0.0))
        requested = round(len(candidates) * ratio)
        if requested <= 0:
            continue

        eligible = [
            (record, applicable_repair_templates(str(record["gold_cypher"]).strip()))
            for record in candidates
            if "invalid_input" not in (record.get("tags") or [])
        ]
        eligible = [(record, templates) for record, templates in eligible if templates]
        if len(eligible) < requested:
            warnings.warn(
                f"Question type `{query_type}` cannot meet repair_cypher_ratio={ratio:.3g}: "
                f"it needs {requested} repair questions but only {len(eligible)} have an "
                "applicable syntax-error template.",
                UserWarning,
                stacklevel=2,
            )

        rng = random.Random(f"{seed}:{query_type}:controlled-repair")
        rng.shuffle(eligible)
        local_usage: Counter[str] = Counter()
        for record, applicable in eligible[:requested]:
            minimum_usage = min(
                (template_usage[template.name], local_usage[template.name])
                for template, _mutated in applicable
            )
            choices = [
                (template, mutated)
                for template, mutated in applicable
                if (template_usage[template.name], local_usage[template.name]) == minimum_usage
            ]
            template, invalid_cypher = rng.choice(choices)
            template_usage[template.name] += 1
            local_usage[template.name] += 1
            record.setdefault("metadata", {}).update(
                {
                    "controlled_repair": True,
                    "repair_template": template.name,
                    "repair_cypher": invalid_cypher,
                }
            )
            tags = record.setdefault("tags", [])
            if "controlled_repair" not in tags:
                tags.append("controlled_repair")
    return records


def repair_template_coverage(
    question_templates: Iterable[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    by_template = {template.name: [] for template in CYPHER_SYNTAX_ERROR_TEMPLATES}
    by_question_type: dict[str, list[str]] = {}
    for question_template in question_templates:
        query_type = str(question_template["query_type"])
        applicable = applicable_repair_templates(str(question_template["cypher"]).strip())
        names = [template.name for template, _mutated in applicable]
        by_question_type[query_type] = names
        for name in names:
            by_template[name].append(query_type)
    return by_template, by_question_type
