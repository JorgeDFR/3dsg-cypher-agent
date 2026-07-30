from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal


def ensure_heracles_agents_importable() -> None:
    try:
        import pypddl  # noqa: F401
        import sldp  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    submodule_src = repo_root / "external" / "heracles_agents" / "src"
    if submodule_src.is_dir():
        sys.path.insert(0, str(submodule_src))


ensure_heracles_agents_importable()

from pypddl.pddl_goal_manipulations import pddl_goal_equals  # noqa: E402
from pypddl.pddl_goal_parser import lark_parse_pddl_goal  # noqa: E402
from sldp.sldp_lang import get_sldp_type, lark_parse_sldp, sldp_equals  # noqa: E402

OutputType = Literal["SLDP", "PDDL"]


def comparator_for_output_type(output_type: str) -> str:
    normalized = output_type.upper()
    if normalized in {"SLDP", "PDDL"}:
        return normalized
    raise ValueError(f"Unsupported output type: {output_type}")


def answer_matches_with_heracles(
    answer: str,
    solution: str,
    output_type: OutputType,
) -> tuple[bool, bool]:
    """Return (valid_format, correct) using Heracles answer comparators."""
    comparator = comparator_for_output_type(output_type)
    if comparator == "SLDP":
        try:
            lark_parse_sldp(str(answer))
        except Exception:
            return False, False
        return True, sldp_equals(str(solution), str(answer))

    try:
        parsed_answer = lark_parse_pddl_goal(str(answer))
    except Exception:
        return False, False
    return True, pddl_goal_equals(parsed_answer, lark_parse_pddl_goal(str(solution)))


def answer_type_hint(solution: str, output_type: OutputType) -> str | None:
    if output_type != "SLDP":
        return None
    sldp_type = get_sldp_type(str(solution))
    if sldp_type.lower() == "string":
        sldp_type = "primitive string"
    elif sldp_type.lower() == "number":
        sldp_type = "primitive number"
    return f"Your answer should be an SLDP {sldp_type}"


def answer_formatting_guidance(output_type: OutputType) -> str:
    if output_type == "SLDP":
        return """
Please format your response according to the SLDP Equality Language:

## SLDP Equality Language

Lists are equal if each element is equal. Sets are equal regardless of item order.
Dictionaries are equal if their keys and values match. Points are equal within tolerance.

A primitive string is an unquoted alphanumeric symbol.
A primitive number is a floating point representation of a number.
A list is written as [element1, element2, ... elementN].
A set is written as <element1, element2, ... elementN>.
A dict is written as {k1: v1, k2: v2}.
A point is written as POINT(x y z), with no commas.

Format your final answer as an SLDP expression wrapped between <answer> and </answer>.
Only a single pair of answer tags should appear.
"""
    if output_type == "PDDL":
        return """
Please format your response according to the PDDL goal language:

The final answer must be a single valid PDDL goal expression using predicates from the
provided PDDL domain.
Use object, place, mesh place, and room symbols exactly as they appear in the scene graph.
Do not include explanatory text inside the final answer.

Format your final answer as a PDDL goal wrapped between <answer> and </answer>.
Only a single pair of answer tags should appear.
"""
    raise ValueError(f"Unsupported output type: {output_type}")
