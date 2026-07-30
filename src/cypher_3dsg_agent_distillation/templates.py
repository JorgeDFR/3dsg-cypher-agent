from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import re
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import yaml

from cypher_3dsg_agent_distillation.config import ensure_parent, load_yaml
from cypher_3dsg_agent_distillation.scene_graph import SOLUTION_MODES, SceneIndex


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


def format_solution_field(value: Any, list_style: str = "list") -> Any:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else float(format_number(number))

    if list_style == "set" and isinstance(value, list):
        value = SolutionSet(value)

    formatted = format_solution_value(value)
    if "\n" in formatted or ": " in formatted or formatted.startswith(("[", "{", "<")):
        return SolutionBlock(f"{formatted}\n")
    return formatted


def generate_records(
    config: dict[str, Any],
    question_templates_config: dict[str, Any] | None = None,
    limit_per_template: int | None = None,
    solution_mode: str | None = None,
) -> list[dict[str, Any]]:
    values = config.get("values", {})
    solution_mode = solution_mode or config.get("solution_mode", "simplified")
    if solution_mode not in SOLUTION_MODES:
        raise ValueError(
            f"Unsupported solution mode `{solution_mode}`. Choose one of {sorted(SOLUTION_MODES)}."
        )
    templates = (
        question_templates_config.get("templates", [])
        if question_templates_config is not None
        else config.get("templates", [])
    )
    scene_index = SceneIndex(config["source_3dsg"]) if config.get("source_3dsg") else None
    records: list[dict[str, Any]] = []
    for template in templates:
        generated_for_template = 0
        query_type = template["query_type"]
        for variables in variable_product(template, values):
            cypher = format_nested(template["cypher"], variables).strip()
            solution = (
                scene_index.solution_for(query_type, variables, mode=solution_mode)
                if scene_index
                else None
            )
            solution_list_style = "set" if solution_mode == "informative" else "list"
            for question_template in template["questions"]:
                question = format_nested(question_template, variables).strip()
                record = {
                    "uid": stable_uid(query_type, question, cypher),
                    "query_type": query_type,
                    "question": question,
                    "gold_cypher": LiteralString(f"{cypher}\n"),
                    "solution": format_solution_field(
                        deepcopy(solution),
                        list_style=solution_list_style,
                    ),
                    "scene_id": config.get("scene_id"),
                    "read_only": bool(template.get("read_only", True)),
                    "tags": deepcopy(template.get("tags", [])),
                }
                records.append(record)
                generated_for_template += 1
                if limit_per_template is not None and generated_for_template >= limit_per_template:
                    break
            if limit_per_template is not None and generated_for_template >= limit_per_template:
                break
    return records


def load_question_templates(
    config: dict[str, Any],
    explicit_path: str | None,
) -> dict[str, Any] | None:
    template_path = explicit_path or config.get("question_templates_path")
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
    config = load_yaml(args.templates)
    question_templates_config = load_question_templates(config, args.question_templates)
    records = generate_records(
        config,
        question_templates_config=question_templates_config,
        limit_per_template=args.limit_per_template,
        solution_mode=args.solution_mode,
    )
    write_questions_yaml(records, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate legacy 3DSG Cypher seed YAML.")
    parser.add_argument(
        "--templates",
        default="data/3dsg/scene_template_values/uhumans2_scene_template_values.yaml",
    )
    parser.add_argument("--question-templates", default=None)
    parser.add_argument("--output", default="data/raw/generated_questions.yaml")
    parser.add_argument("--limit-per-template", type=int, default=None)
    parser.add_argument(
        "--solution-mode",
        choices=sorted(SOLUTION_MODES),
        default=None,
        help="Solution rendering mode. Defaults to config solution_mode or simplified.",
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
