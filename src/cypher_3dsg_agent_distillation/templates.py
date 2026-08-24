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

from cypher_3dsg_agent_distillation.config import ensure_parent, load_data_config, load_yaml
from cypher_3dsg_agent_distillation.scene_graph import SceneIndex


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


def requested_question_count(template: dict[str, Any]) -> int | None:
    value = template.get("target_questions")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("target_questions must be an integer.")
    if value < 0:
        raise ValueError("target_questions must be greater than or equal to zero.")
    return value


def sampling_ratio(
    template: dict[str, Any],
    sampling_config: dict[str, Any],
    override: float | None,
) -> float:
    ratio = (
        override
        if override is not None
        else template.get("non_empty_ratio", sampling_config.get("non_empty_ratio", 0.9))
    )
    ratio = float(ratio)
    if ratio < 0 or ratio > 1:
        raise ValueError(f"non_empty_ratio must be between 0 and 1, got {ratio}.")
    return ratio


def sampled_candidates(
    candidates: list[QuestionCandidate],
    *,
    target: int,
    non_empty_ratio: float,
    query_type: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample unique records while enforcing a per-query-type answer ratio."""
    if target <= 0 or not candidates:
        return []

    actual_target = min(target, len(candidates))
    non_empty = [candidate for candidate in candidates if candidate.has_solution]
    empty = [candidate for candidate in candidates if not candidate.has_solution]

    required_non_empty = math.ceil(actual_target * non_empty_ratio)
    if len(non_empty) < required_non_empty:
        warnings.warn(
            f"Template `{query_type}` cannot meet non_empty_ratio={non_empty_ratio:.3g}: "
            f"it needs {required_non_empty} non-empty questions but only "
            f"{len(non_empty)} unique candidates are available. Sampling the highest "
            "achievable non-empty ratio.",
            UserWarning,
            stacklevel=2,
        )

    selected_non_empty = min(
        len(non_empty),
        max(required_non_empty, actual_target - len(empty)),
    )
    selected_empty = min(len(empty), actual_target - selected_non_empty)
    remaining = actual_target - selected_non_empty - selected_empty
    if remaining:
        additional_non_empty = min(len(non_empty) - selected_non_empty, remaining)
        selected_non_empty += additional_non_empty
        remaining -= additional_non_empty
        selected_empty += min(len(empty) - selected_empty, remaining)

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
    """Maximize distinct variables before reusing one with another wording."""
    if count <= 0:
        return []

    by_variable: dict[str, list[QuestionCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_variable[candidate.variable_key].append(candidate)
    variable_keys = list(by_variable)
    rng.shuffle(variable_keys)
    for variable_candidates in by_variable.values():
        rng.shuffle(variable_candidates)

    selected: list[QuestionCandidate] = []
    while len(selected) < count:
        available_keys = [key for key in variable_keys if by_variable[key]]
        if not available_keys:
            break
        rng.shuffle(available_keys)
        for key in available_keys:
            choices = by_variable[key]
            minimum_usage = min(variant_usage[candidate.variant_index] for candidate in choices)
            least_used = [
                candidate
                for candidate in choices
                if variant_usage[candidate.variant_index] == minimum_usage
            ]
            candidate = rng.choice(least_used)
            choices.remove(candidate)
            selected.append(candidate)
            variant_usage[candidate.variant_index] += 1
            if len(selected) >= count:
                break
    return selected


def stable_uid(query_type: str, question: str, cypher: str) -> str:
    digest = hashlib.sha1(f"{query_type}\n{question}\n{cypher}".encode()).hexdigest()
    return f"tpl_{digest[:12]}"


def format_number(value: int | float) -> str:
    number = float(value)
    if math.isfinite(number):
        if number.is_integer():
            return str(int(number))
        text = f"{number:.6f}".rstrip("0").rstrip(".")
        return text
    return str(number)


COLLECTION_FORMAT_RE = re.compile(r"^(set|list)\[([A-Za-z_][A-Za-z0-9_]*)\]$")
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
        return f"{collection_kind}[{canonical_solution_type(item_type)}]"
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
    non_empty_ratio: float | None = None,
) -> list[dict[str, Any]]:
    values = config.get("values", {})
    templates_config = (
        question_templates_config if question_templates_config is not None else config
    )
    templates = templates_config.get("templates", [])
    sampling_config = templates_config.get("sampling", {})
    if not isinstance(sampling_config, dict):
        raise TypeError("sampling must be a mapping.")
    sampling_seed = int(seed if seed is not None else sampling_config.get("seed", 42))
    scene_index = SceneIndex(config["source_3dsg"]) if config.get("source_3dsg") else None
    records: list[dict[str, Any]] = []
    for template in templates:
        query_type = template["query_type"]
        solution_formats = solution_formats_for_template(template)
        if scene_index is not None and not solution_formats:
            raise KeyError(f"Template `{query_type}` must define solution_formats.")

        variable_combinations = list(variable_product(template, values))
        question_templates = list(template["questions"])
        has_variables = bool(template.get("variables"))
        target = requested_question_count(template) if has_variables else None
        if target is not None and limit_per_template is not None:
            target = min(target, limit_per_template)
        elif target is None and limit_per_template is not None:
            target = limit_per_template

        raw_capacity = len(variable_combinations) * len(question_templates)
        if has_variables and target is not None and raw_capacity < target:
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
            cypher = format_nested(template["cypher"], variables).strip()
            solution = scene_index.solution_for(query_type, variables) if scene_index else None
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
            for variant_index, question_template in enumerate(question_templates):
                question = format_nested(question_template, variables).strip()
                if question in seen_questions:
                    continue
                seen_questions.add(question)
                record = {
                    "uid": stable_uid(query_type, question, cypher),
                    "query_type": query_type,
                    "question": question,
                    "gold_cypher": LiteralString(f"{cypher}\n"),
                    "solution_formats": deepcopy(solution_formats),
                    "solution": format_solution_field(
                        deepcopy(solution),
                        solution_formats=solution_formats,
                    ),
                    "scene_id": config.get("scene_id"),
                    "read_only": bool(template.get("read_only", True)),
                    "tags": deepcopy(template.get("tags", [])),
                }
                if answer_validation is not None:
                    record["answer_validation"] = deepcopy(answer_validation)
                candidates.append(
                    QuestionCandidate(
                        record=record,
                        has_solution=scene_index is None or not solution_is_empty(solution),
                        variable_key=variable_key,
                        variant_index=variant_index,
                    )
                )

        if has_variables and target is not None:
            if len(candidates) < target and len(candidates) < raw_capacity:
                warnings.warn(
                    f"Template `{query_type}` requests {target} questions, but only "
                    f"{len(candidates)} questions are unique after rendering. Generating "
                    "all unique questions.",
                    UserWarning,
                    stacklevel=2,
                )
            ratio = sampling_ratio(template, sampling_config, non_empty_ratio)
            records.extend(
                sampled_candidates(
                    candidates,
                    target=target,
                    non_empty_ratio=ratio,
                    query_type=query_type,
                    rng=random.Random(f"{sampling_seed}:{query_type}"),
                )
            )
        else:
            candidate_records = [candidate.record for candidate in candidates]
            records.extend(candidate_records[:target])
    return records


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
    data_config = load_data_config(args.data_config)
    config = load_yaml(data_config["scene_template_values_path"])
    config["scene_id"] = data_config["scene_id"]
    config["source_3dsg"] = data_config["scene_path"]
    config["question_templates_path"] = data_config["question_templates_path"]
    question_templates_config = load_question_templates(config)
    records = generate_records(
        config,
        question_templates_config=question_templates_config,
        limit_per_template=args.limit_per_template,
        seed=args.seed,
        non_empty_ratio=args.non_empty_ratio,
    )
    write_questions_yaml(records, data_config["raw_questions_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate legacy 3DSG Cypher seed YAML.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--limit-per-template", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--non-empty-ratio", type=float, default=None)
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
