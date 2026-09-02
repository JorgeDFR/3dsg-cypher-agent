from __future__ import annotations

import re
from typing import Any

NODE_LABEL_RE = re.compile(
    r"\(\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*)?:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?"
)
RELATIONSHIP_RE = re.compile(
    r"\[\s*(?:`?[A-Za-z_][A-Za-z0-9_]*`?\s*)?"
    r":\s*`?([A-Za-z_][A-Za-z0-9_]*)`?"
    r"(?:\s*\*(?:(\d*)\.\.(\d*)|(\d+)))?"
)
LITERAL_FILTER_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
    r"(?:\"([^\"]*)\"|'([^']*)'|(-?\d+(?:\.\d+)?))"
)
EMPTY_SOLUTIONS = {"", "<>", "[]", "{}", "null", "none"}


def _relationship_signature(cypher: str) -> tuple[tuple[str, str], ...]:
    relationships = []
    for match in RELATIONSHIP_RE.finditer(cypher):
        relationship_type, lower, upper, exact = match.groups()
        if exact is not None:
            length = exact
        elif lower or upper:
            length = f"{lower}..{upper}"
        else:
            length = "1"
        relationships.append((relationship_type, length))
    return tuple(sorted(relationships))


def _literal_filter_signature(cypher: str) -> tuple[tuple[str, str], ...]:
    filters = []
    for match in LITERAL_FILTER_RE.finditer(cypher):
        property_name, double_quoted, single_quoted, number = match.groups()
        literal = next(
            value
            for value in (double_quoted, single_quoted, number)
            if value is not None
        )
        filters.append((property_name, literal))
    return tuple(sorted(filters))


def cypher_semantic_signature(cypher: str) -> dict[str, Any]:
    """Return a small, variable-name-independent query signature for diagnostics."""
    return {
        "node_labels": tuple(sorted(NODE_LABEL_RE.findall(cypher))),
        "relationships": _relationship_signature(cypher),
        "literal_filters": _literal_filter_signature(cypher),
        "predicate_count": len(re.findall(r"\bWHERE\b", cypher, flags=re.IGNORECASE)),
    }


def _solution_is_empty(solution: Any) -> bool:
    if solution is None:
        return True
    return str(solution).strip().lower() in EMPTY_SOLUTIONS


def result_match_diagnostics(
    *,
    generated_cypher: str,
    gold_cypher: str,
    cypher_exact_match: bool,
    solution_match: bool,
    rows: list[dict[str, Any]],
    expected_solution: Any,
) -> dict[str, Any]:
    """Identify result-equivalent queries whose semantic equivalence is unproven.

    Matching one graph result cannot prove that two queries encode the same intent.
    This diagnostic keeps the ordinary solution-match score intact while exposing
    high-risk matches for review.
    """
    non_exact_result_match = bool(solution_match and not cypher_exact_match)
    reasons = []
    if non_exact_result_match:
        if not rows or _solution_is_empty(expected_solution):
            reasons.append("non_exact_cypher_matched_empty_result")

        generated_signature = cypher_semantic_signature(generated_cypher)
        gold_signature = cypher_semantic_signature(gold_cypher)
        for key, reason in (
            ("node_labels", "node_label_mismatch"),
            ("relationships", "relationship_pattern_mismatch"),
            ("literal_filters", "literal_filter_mismatch"),
            ("predicate_count", "predicate_count_mismatch"),
        ):
            if generated_signature[key] != gold_signature[key]:
                reasons.append(reason)

    return {
        "result_match_without_exact_cypher_match": non_exact_result_match,
        "potential_hidden_false_positive": bool(reasons),
        "hidden_false_positive_reasons": reasons,
    }
