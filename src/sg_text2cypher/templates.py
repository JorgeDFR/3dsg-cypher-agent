from __future__ import annotations

import argparse
import hashlib
import itertools
import re
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import yaml

from sg_text2cypher.config import ensure_parent, load_yaml


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


def generate_records(
    config: dict[str, Any],
    limit_per_template: int | None = None,
) -> list[dict[str, Any]]:
    values = config.get("values", {})
    records: list[dict[str, Any]] = []
    for template in config.get("templates", []):
        generated_for_template = 0
        query_type = template["query_type"]
        for variables in variable_product(template, values):
            cypher = format_nested(template["cypher"], variables).strip()
            for question_template in template["questions"]:
                question = format_nested(question_template, variables).strip()
                records.append(
                    {
                        "uid": stable_uid(query_type, question, cypher),
                        "name": query_type.replace("_", " ").title(),
                        "question": question,
                        "gold_cypher": cypher,
                        "database": config.get("database", "heracles_scene_graph"),
                        "query_type": query_type,
                        "task_type": query_type,
                        "read_only": bool(template.get("read_only", True)),
                        "scene_id": config.get("scene_id"),
                        "tags": deepcopy(template.get("tags", [])),
                    }
                )
                generated_for_template += 1
                if limit_per_template is not None and generated_for_template >= limit_per_template:
                    break
            if limit_per_template is not None and generated_for_template >= limit_per_template:
                break
    return records


def command(args: argparse.Namespace) -> None:
    config = load_yaml(args.templates)
    records = generate_records(config, limit_per_template=args.limit_per_template)
    output = ensure_parent(args.output)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"questions": records}, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic 3DSG text-to-Cypher YAML.")
    parser.add_argument("--templates", default="configs/templates_3dsg.yaml")
    parser.add_argument("--output", default="data/raw/generated_questions.yaml")
    parser.add_argument("--limit-per-template", type=int, default=None)
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
