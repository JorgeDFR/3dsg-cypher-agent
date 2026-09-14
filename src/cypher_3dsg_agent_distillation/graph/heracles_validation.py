from __future__ import annotations

import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from cypher_3dsg_agent_distillation.agents.agent_records import feedback_answer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERACLES_AGENTS_SRC = (
    PROJECT_ROOT / "external" / "heracles_benchmark" / "external" / "heracles_agents" / "src"
)


def ensure_heracles_agents_importable() -> None:
    try:
        import pypddl  # noqa: F401
        import sldp  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    if HERACLES_AGENTS_SRC.is_dir():
        sys.path.insert(0, str(HERACLES_AGENTS_SRC))


ensure_heracles_agents_importable()

from pypddl.pddl_goal_manipulations import pddl_goal_equals  # noqa: E402
from pypddl.pddl_goal_parser import lark_parse_pddl_goal  # noqa: E402
from pypddl.pddl_goal_types import (  # noqa: E402
    Bool,
    Conjunction,
    Disjunction,
    Fact,
    NegatedAtomic,
    NegatedClause,
    Symbol,
)
from sldp.sldp_lang import get_sldp_type, lark_parse_sldp, sldp_equals  # noqa: E402

OutputType = Literal["SLDP", "PDDL"]

PDDL_PREDICATE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "visited-place": ("PlaceOrMeshPlace",),
    "at-place": ("PlaceOrMeshPlace",),
    "visited-object": ("Object",),
    "at-object": ("Object",),
    "safe": ("Object",),
    "visited-region": ("Room",),
    "in-region": ("Room",),
    "holding": ("Object",),
    "object-in-place": ("Object", "PlaceOrMeshPlace"),
}
PDDL_SYMBOL_PATTERNS = {
    "Object": re.compile(r"^O\d+$"),
    "Place": re.compile(r"^p\d+$"),
    "MeshPlace": re.compile(r"^P\d+$"),
    "PlaceOrMeshPlace": re.compile(r"^[pP]\d+$"),
    "Room": re.compile(r"^R\d+$"),
    "Building": re.compile(r"^B\d+$"),
}


def pddl_goal_domain_issues(goal: Any) -> list[str]:
    issues: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Fact):
            signature = PDDL_PREDICATE_SIGNATURES.get(node.head)
            if signature is None:
                issues.append(f"unknown predicate: {node.head}")
                return
            if len(node.params) != len(signature):
                issues.append(
                    f"wrong arity for {node.head}: expected {len(signature)}, "
                    f"got {len(node.params)}"
                )
                return
            for value, symbol_type in zip(node.params, signature, strict=True):
                if not PDDL_SYMBOL_PATTERNS[symbol_type].fullmatch(value):
                    issues.append(f"invalid {symbol_type} symbol `{value}` for {node.head}")
            return
        if isinstance(node, Conjunction | Disjunction):
            if not node.clauses:
                issues.append(f"empty {type(node).__name__.lower()}")
            for clause in node.clauses:
                visit(clause)
            return
        if isinstance(node, NegatedClause):
            visit(node.clause)
            return
        if isinstance(node, NegatedAtomic):
            visit(node.atomic)
            return
        if isinstance(node, Bool):
            return
        if isinstance(node, Symbol):
            issues.append("free variables are not valid grounded planning goals")
            return
        issues.append(f"unsupported PDDL node: {type(node).__name__}")

    visit(goal)
    return issues


def validate_pddl_goal(answer: str) -> tuple[bool, Any | None, list[str]]:
    try:
        parsed = lark_parse_pddl_goal(str(answer))
    except Exception as exc:
        return False, None, [f"parse error: {exc}"]
    issues = pddl_goal_domain_issues(parsed)
    return not issues, parsed, issues


def comparator_for_output_type(output_type: str) -> str:
    normalized = output_type.upper()
    if normalized in {"SLDP", "PDDL"}:
        return normalized
    raise ValueError(f"Unsupported output type: {output_type}")


def answer_matches_with_heracles(
    answer: str,
    solution: str,
    output_type: OutputType,
    *,
    answer_validation: dict[str, Any] | None = None,
) -> tuple[bool, bool]:
    """Return (valid_format, correct) using Heracles answer comparators."""
    comparator = comparator_for_output_type(output_type)
    normalized_answer = str(answer).strip()
    normalized_solution = str(solution).strip()
    if not normalized_solution:
        if not normalized_answer:
            return True, True
        if comparator == "SLDP":
            try:
                lark_parse_sldp(normalized_answer)
            except Exception:
                return False, False
            return True, False
        valid_domain, _parsed_answer, _issues = validate_pddl_goal(normalized_answer)
        return valid_domain, False
    if not normalized_answer:
        return True, False

    if comparator == "SLDP":
        try:
            parsed_answer = lark_parse_sldp(str(answer))
        except Exception:
            return False, False
        exact_match = sldp_equals(str(solution), str(answer))
        if exact_match or not answer_validation:
            return True, exact_match
        if answer_validation.get("mode") == "top_k_with_ties":
            return True, top_k_with_ties_matches(parsed_answer, answer_validation)
        return True, False

    answer_feedback = feedback_answer(answer)
    solution_feedback = feedback_answer(solution)
    if answer_feedback is not None:
        return True, answer_feedback == solution_feedback
    if solution_feedback is not None:
        valid_domain, _parsed_answer, _issues = validate_pddl_goal(str(answer))
        return valid_domain, False

    valid_domain, parsed_answer, _issues = validate_pddl_goal(str(answer))
    if not valid_domain or parsed_answer is None:
        return False, False
    valid_solution, parsed_solution, _solution_issues = validate_pddl_goal(str(solution))
    if not valid_solution or parsed_solution is None:
        return False, False
    # pypddl's equality implementation prints subset-mismatch diagnostics during
    # ordinary comparisons. A mismatch is an expected validation result, so keep
    # those dependency-internal debug messages out of CLI progress output.
    captured_output = StringIO()
    with redirect_stdout(captured_output), redirect_stderr(captured_output):
        matches = pddl_goal_equals(parsed_answer, parsed_solution)
    return True, matches


def top_k_with_ties_matches(
    parsed_answer: Any,
    answer_validation: dict[str, Any],
) -> bool:
    """Validate a top-k set while allowing any members tied at the cutoff score."""
    if not isinstance(parsed_answer, tuple) or not parsed_answer or parsed_answer[0] != "set":
        return False

    values = parsed_answer[1:]
    if not all(isinstance(value, str) for value in values):
        return False
    normalized_values = [str(value).strip().lower() for value in values]
    if len(normalized_values) != len(set(normalized_values)):
        return False

    raw_scores = answer_validation.get("scores")
    if not isinstance(raw_scores, dict) or not raw_scores:
        return False
    try:
        scores = {str(value).strip().lower(): float(score) for value, score in raw_scores.items()}
        requested_limit = int(answer_validation["limit"])
    except (KeyError, TypeError, ValueError):
        return False

    target_size = min(max(requested_limit, 0), len(scores))
    if len(normalized_values) != target_size or any(
        value not in scores for value in normalized_values
    ):
        return False
    if target_size == 0:
        return True

    cutoff_score = sorted(scores.values(), reverse=True)[target_size - 1]
    required = {value for value, score in scores.items() if score > cutoff_score}
    cutoff_ties = {value for value, score in scores.items() if score == cutoff_score}
    actual = set(normalized_values)
    return required.issubset(actual) and actual.issubset(required | cutoff_ties)


def answer_type_hint(solution: str, output_type: OutputType) -> str | None:
    if output_type != "SLDP":
        return None
    sldp_type = get_sldp_type(str(solution))
    if sldp_type.lower() == "string":
        sldp_type = "primitive string"
    elif sldp_type.lower() == "number":
        sldp_type = "primitive number"
    return f"Your answer should be an SLDP {sldp_type}"


def sldp_format_description(solution_format: str) -> str:
    normalized = str(solution_format).strip().lower()
    collection_match = re.fullmatch(r"(set|list)\[(.+)\]", normalized)
    if collection_match:
        collection_type = collection_match.group(1)
        item_type = sldp_format_description(collection_match.group(2))
        if item_type.startswith("primitive "):
            item_type = item_type.removeprefix("primitive ")
        if item_type.endswith("s"):
            plural_item = item_type
        else:
            plural_item = f"{item_type}s"
        return f"{collection_type} of {plural_item}"
    if normalized in {"str", "string"}:
        return "primitive string"
    if normalized in {"int", "integer", "float", "number"}:
        return "primitive number"
    if normalized == "point":
        return "point"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    return normalized


def answer_type_hint_from_solution_formats(
    solution_formats: Any,
    output_type: OutputType,
) -> str | None:
    if output_type != "SLDP" or not solution_formats:
        return None
    formats = solution_formats if isinstance(solution_formats, list) else [solution_formats]
    descriptions = []
    for solution_format in formats:
        description = sldp_format_description(str(solution_format))
        if description not in descriptions:
            descriptions.append(description)
    if not descriptions:
        return None
    if len(descriptions) == 1:
        return f"Your answer should be an SLDP {descriptions[0]}"
    return f"Your answer should be an SLDP {' or a '.join(descriptions)}"


def answer_formatting_guidance(output_type: OutputType) -> str:
    if output_type == "SLDP":
        return """
Please format your response according to the SLDP Language:

## SLDP Language

A SLDP answer represents the answer to a direct question.
It can be a primitive string, primitive number, list, set, dictionarie, or point.
It can also be an arbitrary composition of these containers.

Use object and place symbols exactly as they appear in the scene graph. Node symbols should not contain parentheses or quotes. For example, O(1) should be written as O1.

Do not include additional information than what is explicitly asked for in the question.
E.g., if the question asks for a list of node IDs, the answer should be a list of node IDs and not a list of nodes with their properties or if the question asks for locations a list of points should be provided and not a list of nodes with their locations.

### Syntax

A primitive string is a sequence of alphanumeric characters (with no quotation).
A primitive number is a floating point representation of a number.
A `list` is written as `[element1, element2, ... elementN]`
A `set` is written as `<element1, element2, ... elementN>`
A `dict` is written as `{k1: v1, k2: v2}`
A `point` is written as `POINT(x y z)` (note the lack of comma)

### Denoting Final Answer:

Format your final answer (*not* any intermediate tool calls) as an SLDP expression wrapped between the <answer> and </answer> tags (XML-style format).
Example: <answer> <1,2,3> </answer>
Only a single pair of answer tags should appear.
"""
    if output_type == "PDDL":
        return """
Please format your response according to the PDDL Goal Language:

## PDDL Goal Language

A PDDL goal describes the desired final state for a planner. It must be a single valid goal expression using predicates from the provided PDDL domain.

Use object and place symbols exactly as they appear in the scene graph. Node symbols should not contain parentheses or quotes. For example, O(1) should be written as O1.

Do not include explanatory text inside the final answer. Do not invent predicates outside the provided PDDL domain.

### Syntax

An atomic predicate is written as `(predicate arg1 arg2 ... argN)`.
A conjunction is written as `(and goal1 goal2 ... goalN)`.
A disjunction is written as `(or goal1 goal2 ... goalN)`.
A negated atomic predicate is written as `(not (predicate arg1 arg2 ... argN))`.

Examples:
`(visited-object O1)`
`(and (visited-place R0) (visited-place R1))`
`(or (visited-object O2) (visited-object O5))`

### Denoting Final Answer:

Format your final answer (*not* any intermediate tool calls) as a PDDL goal wrapped between the <answer> and </answer> tags (XML-style format).
Example: <answer> (visited-place P100) </answer>
Only a single pair of answer tags should appear.
"""
    raise ValueError(f"Unsupported output type: {output_type}")
