from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import yaml

from cypher_3dsg_agent_distillation.agents.experiment_profiles import load_experiment_profile
from cypher_3dsg_agent_distillation.core.cli_output import captured_warnings, print_summary
from cypher_3dsg_agent_distillation.core.config import (
    PROJECT_ROOT,
    ensure_parent,
    load_data_config,
    load_yaml,
)
from cypher_3dsg_agent_distillation.core.versioning import (
    config_record,
    record_stage,
    require_new_outputs,
    serializable_arguments,
)
from cypher_3dsg_agent_distillation.cypher.pddl_goals import (
    grounding_contract,
    grounding_is_non_empty,
    render_pddl_solution,
)
from cypher_3dsg_agent_distillation.cypher.repair_supervision import (
    apply_controlled_repair_sampling,
    repair_cypher_sampling_ratio,
)
from cypher_3dsg_agent_distillation.graph.heracles_validation import validate_pddl_goal
from cypher_3dsg_agent_distillation.graph.scene_graph import SceneIndex


class LiteralString(str):
    pass


class DatasetDumper(yaml.SafeDumper):
    pass


def _represent_literal_string(dumper: DatasetDumper, data: LiteralString) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


DatasetDumper.add_representer(LiteralString, _represent_literal_string)


class SolutionBlock(str):
    pass


class SolutionSet(list):
    pass


@dataclass
class QuestionCandidate:
    record: dict[str, Any]
    has_solution: bool
    variable_key: str
    solution_key: str
    sampling_group_key: str | None
    variant_index: int


def _represent_solution_block(dumper: DatasetDumper, data: SolutionBlock) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


DatasetDumper.add_representer(SolutionBlock, _represent_solution_block)


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_mapping(child, child_prefix))
        return flattened
    return {prefix: value}


def format_nested(template: str, values: dict[str, Any]) -> str:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        flattened[str(key)] = value
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, str(key)))

    def replace(match: re.Match[str]) -> str:
        field_name = match.group(1)
        if field_name not in flattened:
            raise KeyError(f"Missing template value `{field_name}`.")
        return str(flattened[field_name])

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}", replace, template)


def variable_product(
    template: dict[str, Any],
    values: dict[str, list[Any]],
) -> Iterable[dict[str, Any]]:
    variables = template.get("variables", {})
    if not variables:
        yield {}
        return

    names = list(variables)
    value_lists = []
    for variable_name in names:
        value_ref = variables[variable_name]
        if value_ref not in values:
            raise KeyError(
                f"Template variable `{variable_name}` references missing values `{value_ref}`."
            )
        value_lists.append(values[value_ref])

    for combination in itertools.product(*value_lists):
        yield dict(zip(names, combination, strict=True))


def solution_is_empty(value: Any) -> bool:
    """Return whether a solved template has no result rows/items."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict | set | list | tuple):
        return len(value) == 0
    return False


def is_zero_count_solution(value: Any) -> bool:
    """Return whether an aggregate solution is the numeric count zero."""
    if not isinstance(value, bool) and isinstance(value, int | float):
        return value == 0
    return False


def solution_has_no_matches(value: Any) -> bool:
    """Return whether a solved template represents no matched graph entities."""
    return is_zero_count_solution(value) or solution_is_empty(value)


def solution_cardinality(value: Any) -> int:
    """Estimate how many grounded entities a solution exposes to the answer."""
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(solution_cardinality(child) for child in value.values())
    if isinstance(value, list | set | tuple):
        return len(value)
    return 1


def estimated_result_rows(value: Any) -> int:
    """Estimate result rows, including Cartesian products in multi-field grounding."""
    if value is None:
        return 0
    if isinstance(value, dict):
        field_sizes = [solution_cardinality(child) for child in value.values()]
        return math.prod(field_sizes) if field_sizes else 0
    if isinstance(value, list | set | tuple):
        return len(value)
    return 1


def sampling_size_limit(
    template: dict[str, Any],
    sampling_config: dict[str, Any],
    key: str,
) -> int | None:
    """Resolve an optional positive per-template sampling size limit."""
    value = template.get(key, sampling_config.get(key))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def invalid_values_in_value(
    value: Any,
    invalid_values: set[str],
) -> list[str]:
    if isinstance(value, dict):
        return [
            invalid_value
            for child in value.values()
            for invalid_value in invalid_values_in_value(child, invalid_values)
        ]
    if isinstance(value, list | tuple | set):
        return [
            invalid_value
            for child in value
            for invalid_value in invalid_values_in_value(child, invalid_values)
        ]
    normalized = str(value)
    return [normalized] if normalized in invalid_values else []


def empty_solution_value(solution_formats: Any) -> Any:
    """Return an empty value that renders in the task's declared SLDP shape."""
    normalized_formats = normalize_solution_formats(solution_formats)
    for solution_format in normalized_formats:
        collection_kind = collection_format_kind(solution_format)
        if collection_kind == "set":
            return set()
        if collection_kind == "list":
            return []
    return ""


def requested_question_count(template: dict[str, Any]) -> int | None:
    value = template.get("target_questions")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("target_questions must be an integer.")
    if value < 0:
        raise ValueError("target_questions must be greater than or equal to zero.")
    return value


def empty_sampling_ratio(
    template: dict[str, Any],
    sampling_config: dict[str, Any],
    override: float | None,
) -> float:
    ratio = (
        override
        if override is not None
        else template.get("empty_ratio", sampling_config.get("empty_ratio", 0.1))
    )
    ratio = float(ratio)
    if ratio < 0 or ratio > 1:
        raise ValueError(f"empty_ratio must be between 0 and 1, got {ratio}.")
    return ratio


def sampled_candidates(
    candidates: list[QuestionCandidate],
    *,
    target: int,
    empty_ratio: float,
    query_type: str,
    rng: random.Random,
    warn_on_shortfall: bool = True,
) -> list[dict[str, Any]]:
    """Sample unique records while enforcing a per-query-type answer ratio."""
    if target <= 0 or not candidates:
        return []

    actual_target = min(target, len(candidates))
    non_empty = [candidate for candidate in candidates if candidate.has_solution]
    empty = [candidate for candidate in candidates if not candidate.has_solution]

    requested_empty = round(actual_target * empty_ratio)
    requested_non_empty = actual_target - requested_empty
    if warn_on_shortfall and len(empty) < requested_empty:
        warnings.warn(
            f"Template `{query_type}` cannot meet empty_ratio={empty_ratio:.3g}: "
            f"it needs {requested_empty} empty questions but only "
            f"{len(empty)} unique candidates are available.",
            UserWarning,
            stacklevel=2,
        )
    if warn_on_shortfall and len(non_empty) < requested_non_empty:
        warnings.warn(
            f"Template `{query_type}` cannot meet empty_ratio={empty_ratio:.3g}: "
            f"it needs {requested_non_empty} normal questions but only "
            f"{len(non_empty)} unique candidates are available.",
            UserWarning,
            stacklevel=2,
        )

    selected_empty = min(len(empty), requested_empty)
    selected_non_empty = min(len(non_empty), actual_target - selected_empty)
    remaining = actual_target - selected_non_empty - selected_empty
    if remaining:
        additional_empty = min(len(empty) - selected_empty, remaining)
        selected_empty += additional_empty
        remaining -= additional_empty
        selected_non_empty += min(len(non_empty) - selected_non_empty, remaining)

    variant_usage: Counter[int] = Counter()
    selected = diverse_candidate_sample(
        non_empty,
        selected_non_empty,
        rng=rng,
        variant_usage=variant_usage,
    )
    selected.extend(
        diverse_candidate_sample(
            empty,
            selected_empty,
            rng=rng,
            variant_usage=variant_usage,
        )
    )
    rng.shuffle(selected)
    return [candidate.record for candidate in selected]


def diverse_candidate_sample(
    candidates: list[QuestionCandidate],
    count: int,
    *,
    rng: random.Random,
    variant_usage: Counter[int],
) -> list[QuestionCandidate]:
    """Balance groups and variables, using solution diversity before wording."""
    if count <= 0:
        return []

    by_group: dict[str, dict[str, list[QuestionCandidate]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate in candidates:
        group_key = candidate.sampling_group_key or "__all__"
        by_group[group_key][candidate.variable_key].append(candidate)
    group_keys = list(by_group)
    rng.shuffle(group_keys)
    solution_usage: Counter[tuple[str, str]] = Counter()
    variable_usage: Counter[tuple[str, str]] = Counter()
    for variables in by_group.values():
        for variable_candidates in variables.values():
            rng.shuffle(variable_candidates)

    selected: list[QuestionCandidate] = []
    while len(selected) < count:
        available_groups = [
            group_key
            for group_key in group_keys
            if any(by_group[group_key].values())
        ]
        if not available_groups:
            break
        rng.shuffle(available_groups)
        for group_key in available_groups:
            variables = by_group[group_key]
            available_variables = [
                variable_key
                for variable_key, variable_candidates in variables.items()
                if variable_candidates
            ]
            minimum_variable_usage = min(
                variable_usage[(group_key, variable_key)]
                for variable_key in available_variables
            )
            least_used_variables = [
                variable_key
                for variable_key in available_variables
                if variable_usage[(group_key, variable_key)] == minimum_variable_usage
            ]
            available_solution_keys = {
                candidate.solution_key
                for variable_key in least_used_variables
                for candidate in variables[variable_key]
            }
            minimum_solution_usage = min(
                solution_usage[(group_key, solution_key)]
                for solution_key in available_solution_keys
            )
            least_used_solutions = {
                solution_key
                for solution_key in available_solution_keys
                if solution_usage[(group_key, solution_key)] == minimum_solution_usage
            }
            solution_variables = [
                variable_key
                for variable_key in least_used_variables
                if any(
                    candidate.solution_key in least_used_solutions
                    for candidate in variables[variable_key]
                )
            ]
            variable_key = rng.choice(solution_variables)
            choices = [
                candidate
                for candidate in variables[variable_key]
                if candidate.solution_key in least_used_solutions
            ]
            minimum_usage = min(variant_usage[candidate.variant_index] for candidate in choices)
            least_used = [
                candidate
                for candidate in choices
                if variant_usage[candidate.variant_index] == minimum_usage
            ]
            candidate = rng.choice(least_used)
            variables[variable_key].remove(candidate)
            selected.append(candidate)
            solution_usage[(group_key, candidate.solution_key)] += 1
            variable_usage[(group_key, variable_key)] += 1
            variant_usage[candidate.variant_index] += 1
            if len(selected) >= count:
                break
    return selected


def stable_uid(
    query_type: str,
    question: str,
    cypher: str,
    task_family: str = "cypher_qa",
    scene_id: str | None = None,
) -> str:
    identity = f"{query_type}\n{question}\n{cypher}"
    if task_family != "cypher_qa":
        identity = f"{task_family}\n{identity}"
    if scene_id:
        identity = f"{scene_id}\n{identity}"
    digest = hashlib.sha1(identity.encode()).hexdigest()
    return f"tpl_{digest[:12]}"


CYPHER_LABEL_PATTERN = re.compile(r":\s*(Object|Place|MeshPlace|Room|Building)\b")


def scene_supports_template(
    template: dict[str, Any], scene_summary: dict[str, Any] | None
) -> bool:
    """Return whether a scene has the layers required by a question template."""
    if not scene_summary:
        return True
    nodes = scene_summary.get("nodes") or {}
    alternative_groups = template.get("requires_any_layers") or []
    alternative_labels = {
        str(label) for alternatives in alternative_groups for label in alternatives
    }
    required = set(template.get("requires_layers") or [])
    required.update(CYPHER_LABEL_PATTERN.findall(str(template.get("cypher") or "")))
    required.difference_update(alternative_labels)
    if any(int(nodes.get(label, 0)) <= 0 for label in required):
        return False
    for alternatives in alternative_groups:
        if not any(int(nodes.get(str(label), 0)) > 0 for label in alternatives):
            return False
    return True


def format_number(value: int | float) -> str:
    number = float(value)
    if math.isfinite(number):
        if number.is_integer():
            return str(int(number))
        text = f"{number:.6f}".rstrip("0").rstrip(".")
        return text
    return str(number)


COLLECTION_FORMAT_RE = re.compile(r"^(set|list)\[(.+)\]$")
SOLUTION_TYPE_ALIASES = {
    "string": "str",
    "strings": "str",
    "integer": "int",
    "integers": "int",
    "number": "float",
    "numbers": "float",
    "point": "point",
    "points": "point",
}


def canonical_solution_type(value: str) -> str:
    normalized = value.strip().lower()
    return SOLUTION_TYPE_ALIASES.get(normalized, normalized)


def canonical_solution_format(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    human_collection = re.fullmatch(r"(set|list) of ([A-Za-z_][A-Za-z0-9_]*)", normalized)
    if human_collection:
        collection_kind, item_type = human_collection.groups()
        return f"{collection_kind}[{canonical_solution_type(item_type)}]"

    compact = normalized.replace(" ", "")
    collection = COLLECTION_FORMAT_RE.fullmatch(compact)
    if collection:
        collection_kind, item_type = collection.groups()
        return f"{collection_kind}[{canonical_solution_format(item_type)}]"
    return canonical_solution_type(normalized)


def normalize_solution_formats(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        formats = [value]
    elif isinstance(value, list | tuple):
        formats = list(value)
    else:
        raise TypeError("solution_formats must be a string or list of strings.")

    normalized = []
    for solution_format in formats:
        if not isinstance(solution_format, str):
            raise TypeError("solution_formats entries must be strings.")
        canonical = canonical_solution_format(solution_format)
        if not canonical:
            raise ValueError("solution_formats entries must not be empty.")
        normalized.append(canonical)
    return normalized


def collection_format_kind(solution_format: str) -> str | None:
    match = COLLECTION_FORMAT_RE.fullmatch(solution_format)
    return match.group(1) if match else None


def is_collection_value(value: Any) -> bool:
    return isinstance(value, SolutionSet | set | list | tuple)


def solution_collection_items(value: Any) -> list[Any]:
    if isinstance(value, set):
        return sorted(value, key=lambda item: format_solution_value(item))
    if isinstance(value, SolutionSet | list | tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def scalar_format_matches(value: Any, solution_format: str) -> bool:
    if solution_format == "bool":
        return isinstance(value, bool)
    if solution_format == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if solution_format == "float":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if solution_format == "point":
        return isinstance(value, str) and value.startswith("POINT(")
    if solution_format == "str":
        return isinstance(value, str)
    return True


def select_solution_format(value: Any, solution_formats: list[str]) -> str | None:
    if not solution_formats:
        return None

    collection_formats = [
        solution_format
        for solution_format in solution_formats
        if collection_format_kind(solution_format) is not None
    ]
    scalar_formats = [
        solution_format
        for solution_format in solution_formats
        if collection_format_kind(solution_format) is None
    ]

    if is_collection_value(value):
        if collection_formats:
            return collection_formats[0]
        items = solution_collection_items(value)
        if len(items) == 1 and scalar_formats:
            return scalar_formats[0]
        return solution_formats[0]

    compatible_scalar_formats = [
        solution_format
        for solution_format in scalar_formats
        if scalar_format_matches(value, solution_format)
    ]
    if compatible_scalar_formats:
        return compatible_scalar_formats[0]
    if collection_formats:
        return collection_formats[0]
    if scalar_formats:
        return scalar_formats[0]
    return solution_formats[0]


def coerce_solution_format(value: Any, solution_format: str | None) -> Any:
    if solution_format is None:
        return value

    collection_kind = collection_format_kind(solution_format)
    if collection_kind is not None:
        items = solution_collection_items(value)
        if collection_kind == "set":
            return SolutionSet(items)
        return items

    if is_collection_value(value):
        items = solution_collection_items(value)
        if len(items) == 1:
            return items[0]
    return value


def solution_formats_for_template(template: dict[str, Any]) -> list[str]:
    formats = template.get("solution_formats", template.get("solution_format"))
    return normalize_solution_formats(formats)


def format_solution_value(value: Any, indent: int = 0) -> str:
    if value is None:
        return "[]"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return format_number(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [
            f"{key}: {format_solution_value(child, indent=indent)}"
            for key, child in value.items()
        ]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, SolutionSet):
        return format_solution_collection("<", ">", value, indent)
    if isinstance(value, set):
        ordered = sorted(value, key=lambda item: format_solution_value(item))
        return format_solution_collection("<", ">", ordered, indent)
    if isinstance(value, list | tuple):
        return format_solution_collection("[", "]", value, indent)
    return str(value)


def format_solution_collection(
    opener: str,
    closer: str,
    values: Iterable[Any],
    indent: int,
) -> str:
    values = list(values)
    if not values:
        return f"{opener}{closer}"
    formatted = [format_solution_value(value, indent=indent + 2) for value in values]
    one_line = f"{opener}{', '.join(formatted)}{closer}"
    if len(one_line) <= 100 and "\n" not in one_line:
        return one_line

    child_indent = " " * (indent + 2)
    closing_indent = " " * indent
    lines = [opener]
    for index, item in enumerate(formatted):
        suffix = "," if index < len(formatted) - 1 else ""
        item_lines = item.splitlines()
        lines.append(f"{child_indent}{item_lines[0]}{suffix}")
        for extra_line in item_lines[1:]:
            lines.append(f"{child_indent}{extra_line}")
    lines.append(f"{closing_indent}{closer}")
    return "\n".join(lines)


def format_solution_field(
    value: Any,
    solution_formats: Any = None,
    list_style: str = "list",
) -> Any:
    normalized_formats = normalize_solution_formats(solution_formats)
    if normalized_formats:
        value = coerce_solution_format(
            value,
            select_solution_format(value, normalized_formats),
        )
    elif list_style == "set" and isinstance(value, list):
        value = SolutionSet(value)

    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else float(format_number(number))

    formatted = format_solution_value(value)
    if "\n" in formatted or ": " in formatted or formatted.startswith(("[", "{", "<")):
        return SolutionBlock(f"{formatted}\n")
    return formatted


def generate_records(
    config: dict[str, Any],
    question_templates_config: dict[str, Any] | None = None,
    limit_per_template: int | None = None,
    seed: int | None = None,
    empty_ratio: float | None = None,
    task_family: str | None = None,
    output_type: str | None = None,
    experiment_profile: str | None = None,
    question_set: str | None = None,
    warn_on_target_shortfall: bool = True,
) -> list[dict[str, Any]]:
    values = config.get("values", {})
    templates_config = (
        question_templates_config if question_templates_config is not None else config
    )
    templates = templates_config.get("templates", [])
    task_family = str(task_family or templates_config.get("task_family") or "cypher_qa")
    output_type = str(output_type or templates_config.get("output_type") or "SLDP")
    sampling_config = templates_config.get("sampling", {})
    if not isinstance(sampling_config, dict):
        raise TypeError("sampling must be a mapping.")
    sampling_seed = int(seed if seed is not None else sampling_config.get("seed", 42))
    invalid_values = {str(value) for value in (values.get("invalid_values") or [])}
    # Accept previously extracted value files as input, but ignore their legacy
    # typed feedback values. Invalid inputs now have ordinary empty answers.
    invalid_values.update(
        str(value) for value in (values.get("invalid_value_feedback") or {})
    )
    scene_index = SceneIndex(config["source_3dsg"]) if config.get("source_3dsg") else None
    records: list[dict[str, Any]] = []
    for template in templates:
        if not scene_supports_template(template, config.get("scene_summary")):
            continue
        query_type = template["query_type"]
        solution_formats = solution_formats_for_template(template)
        max_solution_size = sampling_size_limit(
            template, sampling_config, "max_solution_size"
        )
        max_result_rows = sampling_size_limit(
            template, sampling_config, "max_result_rows"
        )
        if scene_index is not None and task_family == "cypher_qa" and not solution_formats:
            raise KeyError(f"Template `{query_type}` must define solution_formats.")
        if task_family == "cypher_pddl" and not isinstance(template.get("goal"), dict):
            raise KeyError(f"PDDL template `{query_type}` must define a goal mapping.")
        if task_family == "cypher_pddl" and not (
            isinstance(template.get("grounding"), dict)
            and isinstance(template["grounding"].get("fields"), dict)
            and template["grounding"]["fields"]
        ):
            raise KeyError(f"PDDL template `{query_type}` must define grounding fields.")

        variable_combinations = list(variable_product(template, values))
        question_templates = list(template["questions"])
        has_variables = bool(template.get("variables"))
        target = requested_question_count(template) if has_variables else None
        if target is not None and limit_per_template is not None:
            target = min(target, limit_per_template)
        elif target is None and limit_per_template is not None:
            target = limit_per_template

        raw_capacity = len(variable_combinations) * len(question_templates)
        if (
            warn_on_target_shortfall
            and has_variables
            and target is not None
            and raw_capacity < target
        ):
            warnings.warn(
                f"Template `{query_type}` requests {target} questions, but its "
                f"{len(variable_combinations)} variable combinations and "
                f"{len(question_templates)} question variants can produce at most "
                f"{raw_capacity}. Generating all available unique questions.",
                UserWarning,
                stacklevel=2,
            )

        candidates: list[QuestionCandidate] = []
        seen_questions: set[str] = set()
        for variables in variable_combinations:
            matched_invalid_values = invalid_values_in_value(
                variables,
                invalid_values,
            )
            # Avoid ambiguous examples with two independently invalid references.
            if len(matched_invalid_values) > 1:
                continue
            cypher = format_nested(template["cypher"], variables).strip()
            raw_solution = scene_index.solution_for(query_type, variables) if scene_index else None
            if has_variables and raw_solution is not None:
                if (
                    max_solution_size is not None
                    and solution_cardinality(raw_solution) > max_solution_size
                ):
                    continue
                if (
                    max_result_rows is not None
                    and estimated_result_rows(raw_solution) > max_result_rows
                ):
                    continue
            # An absent value does not necessarily make a query empty. For example,
            # excluding an absent class still leaves all other objects, and an IN
            # predicate can contain one absent and one present class. Preserve the
            # computed answer for those QA cases instead of incorrectly supervising
            # them as empty answers. PDDL templates use conjunctive grounding queries,
            # so any absent reference still makes the complete grounding empty.
            invalid_input = bool(matched_invalid_values) and (
                task_family == "cypher_pddl" or solution_has_no_matches(raw_solution)
            )
            record_grounding = None
            if task_family == "cypher_pddl":
                record_grounding = grounding_contract(
                    None if invalid_input else raw_solution,
                    template.get("grounding") or {},
                    variables,
                )
                if invalid_input:
                    solution = ""
                elif grounding_is_non_empty(record_grounding):
                    solution = render_pddl_solution(template["goal"], record_grounding)
                else:
                    solution = None
                if solution:
                    valid_goal, _parsed_goal, goal_issues = validate_pddl_goal(solution)
                    if not valid_goal:
                        raise ValueError(
                            f"PDDL template `{query_type}` generated an invalid goal: "
                            + "; ".join(goal_issues)
                        )
            else:
                solution = (
                    raw_solution
                    if invalid_input and is_zero_count_solution(raw_solution)
                    else empty_solution_value(solution_formats)
                    if invalid_input
                    else raw_solution
                )
            answer_validation_for = getattr(scene_index, "answer_validation_for", None)
            answer_validation = (
                answer_validation_for(query_type, variables)
                if callable(answer_validation_for)
                else None
            )
            variable_key = json.dumps(
                variables,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            sampling_group_key = None
            sampling_group = template.get("sampling_group")
            if invalid_input:
                sampling_group_key = json.dumps("invalid_input")
            elif sampling_group is not None:
                flattened_variables = flatten_mapping(variables)
                if str(sampling_group) not in flattened_variables:
                    raise KeyError(
                        f"Template `{query_type}` sampling_group references missing "
                        f"value `{sampling_group}`."
                    )
                sampling_group_key = json.dumps(
                    flattened_variables[str(sampling_group)],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            for variant_index, question_template in enumerate(question_templates):
                question = format_nested(question_template, variables).strip()
                if question in seen_questions:
                    continue
                seen_questions.add(question)
                record_solution_formats = deepcopy(solution_formats)
                record = {
                    "uid": stable_uid(
                        query_type,
                        question,
                        cypher,
                        task_family,
                        config.get("scene_id"),
                    ),
                    "query_type": query_type,
                    "task_family": task_family,
                    "output_type": output_type,
                    "experiment_profile": experiment_profile,
                    "question": question,
                    "gold_cypher": LiteralString(f"{cypher}\n"),
                    "solution_formats": record_solution_formats,
                    "solution": (
                        str(solution)
                        if task_family == "cypher_pddl" and solution is not None
                        else format_solution_field(
                            deepcopy(solution),
                            solution_formats=record_solution_formats,
                        )
                    ),
                    "scene_id": config.get("scene_id"),
                    "labelspace_description": config.get("labelspace_description"),
                    "read_only": bool(template.get("read_only", True)),
                    "tags": deepcopy(template.get("tags", [])),
                }
                has_solution = (
                    not invalid_input
                    and (
                        grounding_is_non_empty(record_grounding)
                        if record_grounding is not None
                        else scene_index is None or not solution_has_no_matches(solution)
                    )
                )
                if (
                    (
                        invalid_input
                        or (
                            not has_solution
                            and task_family != "cypher_pddl"
                            and not is_zero_count_solution(solution)
                        )
                    )
                    and "invalid_input" not in record["tags"]
                ):
                    record["tags"].append("invalid_input")
                if answer_validation is not None:
                    record["answer_validation"] = deepcopy(answer_validation)
                if record_grounding is not None:
                    record["grounding"] = deepcopy(record_grounding)
                record["metadata"] = {
                    "experiment_profile": experiment_profile,
                    "question_set": question_set,
                    "task_family": task_family,
                    "output_type": output_type,
                }
                candidates.append(
                    QuestionCandidate(
                        record=record,
                        has_solution=has_solution,
                        variable_key=variable_key,
                        solution_key=str(record["solution"]).strip(),
                        sampling_group_key=sampling_group_key,
                        variant_index=variant_index,
                    )
                )

        if task_family == "cypher_pddl":
            candidates = [
                candidate
                for candidate in candidates
                if candidate.has_solution
                or "invalid_input" in candidate.record.get("tags", [])
            ]

        if has_variables and target is not None:
            if (
                warn_on_target_shortfall
                and len(candidates) < target
                and len(candidates) < raw_capacity
            ):
                warnings.warn(
                    f"Template `{query_type}` requests {target} questions, but only "
                    f"{len(candidates)} questions are unique after rendering. Generating "
                    "all unique questions.",
                    UserWarning,
                    stacklevel=2,
                )
            ratio = empty_sampling_ratio(template, sampling_config, empty_ratio)
            records.extend(
                sampled_candidates(
                    candidates,
                    target=target,
                    empty_ratio=ratio,
                    query_type=query_type,
                    rng=random.Random(f"{sampling_seed}:{query_type}"),
                    warn_on_shortfall=warn_on_target_shortfall,
                )
            )
        else:
            candidate_records = [candidate.record for candidate in candidates]
            records.extend(candidate_records[:target])
    return records


def dataset_question_target(
    template: dict[str, Any],
    limit_per_template: int | None,
) -> int | None:
    """Return the dataset-wide record budget for one question type."""
    target = requested_question_count(template) if template.get("variables") else None
    if limit_per_template is None:
        return target
    return limit_per_template if target is None else min(target, limit_per_template)


def effective_dataset_question_target(
    template: dict[str, Any],
    limit_per_template: int | None,
    *,
    challenge_task_types: set[str],
    challenge_questions_per_task_type: int | None,
) -> int | None:
    query_type = str(template["query_type"])
    if query_type not in challenge_task_types:
        return dataset_question_target(template, limit_per_template)
    if challenge_questions_per_task_type is None:
        raise ValueError(
            "split.challenge_questions_per_task_type is required when "
            "split.challenge_task_types is not empty."
        )
    if (
        isinstance(challenge_questions_per_task_type, bool)
        or not isinstance(challenge_questions_per_task_type, int)
        or challenge_questions_per_task_type < 0
    ):
        raise ValueError("challenge_questions_per_task_type must be a non-negative integer.")
    if limit_per_template is None:
        return challenge_questions_per_task_type
    return min(challenge_questions_per_task_type, limit_per_template)


def generated_record_has_solution(record: dict[str, Any]) -> bool:
    if "invalid_input" in (record.get("tags") or []):
        return False
    solution = record.get("solution")
    if isinstance(solution, str):
        return solution.strip() not in {"", "<>", "[]", "{}"}
    return not solution_has_no_matches(solution)


def diverse_record_sample(
    records: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Prefer distinct rendered inputs and use answers to break sampling ties."""
    if count <= 0:
        return []
    by_variable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        variable_key = str(record.get("gold_cypher") or record.get("uid") or "")
        by_variable[variable_key].append(record)

    variable_usage: Counter[str] = Counter()
    solution_usage: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available_variables = [
            key for key, variable_records in by_variable.items() if variable_records
        ]
        if not available_variables:
            break
        minimum_variable_usage = min(
            variable_usage[variable_key] for variable_key in available_variables
        )
        least_used_variables = [
            variable_key
            for variable_key in available_variables
            if variable_usage[variable_key] == minimum_variable_usage
        ]
        minimum_solution_usage = min(
            solution_usage[str(record.get("solution") or "").strip()]
            for variable_key in least_used_variables
            for record in by_variable[variable_key]
        )
        chosen_variable = next(
            variable_key
            for variable_key in least_used_variables
            if any(
                solution_usage[str(record.get("solution") or "").strip()]
                == minimum_solution_usage
                for record in by_variable[variable_key]
            )
        )
        chosen_index = next(
            index
            for index, record in enumerate(by_variable[chosen_variable])
            if solution_usage[str(record.get("solution") or "").strip()]
            == minimum_solution_usage
        )
        record = by_variable[chosen_variable].pop(chosen_index)
        selected.append(record)
        variable_usage[chosen_variable] += 1
        solution_usage[str(record.get("solution") or "").strip()] += 1
    return selected


def evenly_select_scene_records(
    records: list[dict[str, Any]],
    *,
    target: int,
    scene_ids: list[str],
    empty_ratio: float | None = None,
) -> list[dict[str, Any]]:
    """Select a dataset-wide target while keeping scene counts as even as possible."""
    if target <= 0:
        return []
    buckets = {
        scene_id: [
            record for record in records if str(record.get("scene_id")) == scene_id
        ]
        for scene_id in scene_ids
    }
    quotas: Counter[str] = Counter()
    remaining_capacity = {
        scene_id: len(scene_records) for scene_id, scene_records in buckets.items()
    }
    actual_target = min(target, sum(remaining_capacity.values()))
    while sum(quotas.values()) < actual_target:
        available = [
            scene_id
            for scene_id in scene_ids
            if quotas[scene_id] < remaining_capacity[scene_id]
        ]
        if not available:
            break
        minimum = min(quotas[scene_id] for scene_id in available)
        next_scene = next(
            scene_id
            for scene_id in available
            if quotas[scene_id] == minimum
        )
        quotas[next_scene] += 1

    if empty_ratio is None:
        return [
            record
            for scene_id in scene_ids
            for record in diverse_record_sample(
                buckets[scene_id], quotas[scene_id]
            )
        ]

    non_empty = {
        scene_id: [
            record
            for record in buckets[scene_id]
            if generated_record_has_solution(record)
        ]
        for scene_id in scene_ids
    }
    empty = {
        scene_id: [
            record
            for record in buckets[scene_id]
            if not generated_record_has_solution(record)
        ]
        for scene_id in scene_ids
    }
    minimum_empty = {
        scene_id: max(0, quotas[scene_id] - len(non_empty[scene_id]))
        for scene_id in scene_ids
    }
    maximum_empty = {
        scene_id: min(quotas[scene_id], len(empty[scene_id]))
        for scene_id in scene_ids
    }
    desired_empty = round(actual_target * empty_ratio)
    desired_empty = min(
        max(desired_empty, sum(minimum_empty.values())),
        sum(maximum_empty.values()),
    )
    empty_quotas = Counter(minimum_empty)
    while sum(empty_quotas.values()) < desired_empty:
        available = [
            scene_id
            for scene_id in scene_ids
            if empty_quotas[scene_id] < maximum_empty[scene_id]
        ]
        minimum = min(empty_quotas[scene_id] for scene_id in available)
        next_scene = next(
            scene_id
            for scene_id in available
            if empty_quotas[scene_id] == minimum
        )
        empty_quotas[next_scene] += 1

    selected_uids = {
        str(record["uid"])
        for scene_id in scene_ids
        for record in (
            diverse_record_sample(empty[scene_id], empty_quotas[scene_id])
            + diverse_record_sample(
                non_empty[scene_id], quotas[scene_id] - empty_quotas[scene_id]
            )
        )
    }
    return [record for record in records if str(record["uid"]) in selected_uids]


def apply_dataset_question_targets(
    records: list[dict[str, Any]],
    *,
    targets: dict[str, int | None],
    scene_ids: list[str],
    empty_ratios: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply per-question-type targets across all scenes, not once per scene."""
    records_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_type[str(record["query_type"])].append(record)

    selected_uids: set[str] = set()
    for query_type, candidates in records_by_type.items():
        target = targets.get(query_type)
        if target is not None and len(candidates) < target:
            warnings.warn(
                f"Question type `{query_type}` requests {target} questions across the "
                f"dataset, but only {len(candidates)} unique candidates are available.",
                UserWarning,
                stacklevel=2,
            )
        ratio = (empty_ratios or {}).get(query_type)
        actual_target = min(target, len(candidates)) if target is not None else None
        if ratio is not None and actual_target is not None:
            requested_empty = round(actual_target * ratio)
            available_empty = sum(
                not generated_record_has_solution(record) for record in candidates
            )
            available_non_empty = len(candidates) - available_empty
            if available_empty < requested_empty:
                warnings.warn(
                    f"Question type `{query_type}` cannot meet dataset-wide "
                    f"empty_ratio={ratio:.3g}: it needs {requested_empty} "
                    f"empty questions but only {available_empty} are available.",
                    UserWarning,
                    stacklevel=2,
                )
            requested_non_empty = actual_target - requested_empty
            if available_non_empty < requested_non_empty:
                warnings.warn(
                    f"Question type `{query_type}` cannot meet dataset-wide "
                    f"empty_ratio={ratio:.3g}: it needs {requested_non_empty} normal "
                    f"questions but only {available_non_empty} are available.",
                    UserWarning,
                    stacklevel=2,
                )
        selected = (
            candidates
            if target is None
            else evenly_select_scene_records(
                candidates,
                target=target,
                scene_ids=scene_ids,
                empty_ratio=ratio,
            )
        )
        selected_uids.update(str(record["uid"]) for record in selected)

    # Candidate generation is scene-grouped to minimize Neo4j reloads during
    # trace generation. Preserve that order after balanced selection.
    return [record for record in records if str(record["uid"]) in selected_uids]


def load_question_templates(
    config: dict[str, Any]
) -> dict[str, Any] | None:
    template_path = config.get("question_templates_path")
    if template_path:
        return load_yaml(template_path)
    return None


def write_questions_yaml(records: list[dict[str, Any]], path: str) -> None:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("questions:\n")
        for record in records:
            rendered = yaml.dump(
                [record],
                Dumper=DatasetDumper,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
                width=120,
            )
            handle.write(rendered)
            handle.write("\n")


def command(args: argparse.Namespace) -> None:
    data_config = load_data_config(
        args.data_config, dataset_version=args.dataset_version
    )
    split_config = data_config.get("split") or {}
    challenge_task_types = {
        str(value) for value in split_config.get("challenge_task_types", [])
    }
    challenge_questions_per_task_type = split_config.get(
        "challenge_questions_per_task_type"
    )
    question_sets = data_config.get("question_sets") or []
    question_targets: dict[str, int | None] = {}
    question_empty_ratios: dict[str, float] = {}
    question_repair_ratios: dict[str, float] = {}
    question_set_inputs = []
    for question_set in question_sets:
        templates_config = load_yaml(question_set["question_templates_path"])
        profile = load_experiment_profile(question_set["experiment_profile_path"])
        question_set_inputs.append((question_set, templates_config, profile))
        for template in templates_config.get("templates") or []:
            query_type = str(template["query_type"])
            target = effective_dataset_question_target(
                template,
                args.limit_per_template,
                challenge_task_types=challenge_task_types,
                challenge_questions_per_task_type=challenge_questions_per_task_type,
            )
            if query_type in challenge_task_types:
                template["target_questions"] = target
            if query_type in question_targets and question_targets[query_type] != target:
                raise ValueError(
                    f"Question type `{query_type}` has conflicting dataset targets."
                )
            question_targets[query_type] = target
            question_empty_ratios[query_type] = empty_sampling_ratio(
                template,
                templates_config.get("sampling") or {},
                args.empty_ratio,
            )
            question_repair_ratios[query_type] = repair_cypher_sampling_ratio(
                template,
                templates_config.get("sampling") or {},
                args.repair_cypher_ratio,
            )
            if (
                question_empty_ratios[query_type]
                + question_repair_ratios[query_type]
                > 1.0
            ):
                raise ValueError(
                    f"Question type `{query_type}` has empty_ratio + "
                    "repair_cypher_ratio greater than 1."
                )
    unknown_challenge_types = challenge_task_types - question_targets.keys()
    if unknown_challenge_types:
        raise ValueError(
            "split.challenge_task_types contains unknown question types: "
            + ", ".join(sorted(unknown_challenge_types))
        )
    all_records: list[dict[str, Any]] = []
    with captured_warnings("Question generation warnings"):
        for scene in data_config["scenes"]:
            config = load_yaml(scene["scene_template_values_path"])
            config["scene_id"] = scene["scene_id"]
            config["source_3dsg"] = scene["scene_path"]
            for question_set, templates_config, profile in question_set_inputs:
                all_records.extend(
                    generate_records(
                        config,
                        question_templates_config=templates_config,
                        limit_per_template=args.limit_per_template,
                        seed=args.seed,
                        empty_ratio=args.empty_ratio,
                        task_family=profile.task_family,
                        output_type=profile.output_type,
                        experiment_profile=profile.name,
                        question_set=question_set["name"],
                        warn_on_target_shortfall=False,
                    )
                )
    records = apply_dataset_question_targets(
        all_records,
        targets=question_targets,
        scene_ids=[str(scene["scene_id"]) for scene in data_config["scenes"]],
        empty_ratios=question_empty_ratios,
    )
    records = apply_controlled_repair_sampling(
        records,
        ratios=question_repair_ratios,
        seed=int(args.seed if args.seed is not None else 42),
    )
    overlapping_supervision = [
        str(record["uid"])
        for record in records
        if {"invalid_input", "controlled_repair"}.issubset(
            record.get("tags") or []
        )
    ]
    if overlapping_supervision:
        raise RuntimeError(
            "Question generation produced records marked as both empty and repair: "
            + ", ".join(overlapping_supervision[:3])
        )
    uids = [record["uid"] for record in records]
    if len(uids) != len(set(uids)):
        raise ValueError("Question generation produced duplicate record UIDs.")
    output_path = data_config["raw_questions_path"]
    require_new_outputs([output_path], force=args.force)
    write_questions_yaml(records, output_path)
    record_stage(
        data_config["metadata_path"],
        "generate_3dsg_seed_templates",
        arguments=serializable_arguments(args),
        outputs={"generated_questions": output_path},
        project_root=PROJECT_ROOT,
        extra={
            "data_config": config_record(
                data_config["data_config_path"], project_root=PROJECT_ROOT
            )
        },
    )
    counts = Counter(record["query_type"] for record in records)
    effective_empty_ratio = (
        args.empty_ratio if args.empty_ratio is not None else "per set"
    )
    effective_repair_ratio = (
        args.repair_cypher_ratio
        if args.repair_cypher_ratio is not None
        else "per set"
    )
    print_summary(
        "Question templates generated",
        [
            ("Dataset", data_config["dataset_id"]),
            ("Scenes", len(data_config["scenes"])),
            ("Dataset version", data_config["dataset_version"]),
            ("Questions", len(records)),
            ("Question types", len(counts)),
            ("Question sets", len(question_sets)),
            ("Empty ratio", effective_empty_ratio),
            ("Repair Cypher ratio", effective_repair_ratio),
            (
                "Empty questions",
                sum("invalid_input" in (record.get("tags") or []) for record in records),
            ),
            (
                "Cypher repair questions",
                sum(
                    "controlled_repair" in (record.get("tags") or [])
                    for record in records
                ),
            ),
            ("Output", output_path),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3DSG Cypher seed YAML.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-per-template", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--empty-ratio", type=float, default=None)
    parser.add_argument("--repair-cypher-ratio", type=float, default=None)
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
