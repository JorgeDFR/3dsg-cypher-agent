"""Load synthetic-only retrieval declarations, keyed by existing query_type names."""

from __future__ import annotations

from pathlib import Path

import yaml

from cypher_3dsg_agent_distillation.core.config import PROJECT_ROOT
from cypher_3dsg_agent_distillation.cypher.task_semantics import (
    SEMANTICS_VERSION,
    validate_goal_semantics,
    validate_query_semantics,
)

DEFAULT_TEMPLATES = [
    "configs/question_templates/cypher_qa_3dsg.yaml",
    "configs/question_templates/cypher_pddl_3dsg.yaml",
]


def default_semantics_path(template_path: str | Path) -> Path:
    return (
        PROJECT_ROOT / "configs/question_templates/synthetic_reasoning" / Path(template_path).name
    )


def load_catalog(
    template_paths, semantics_paths=None, *, validation_paths=None, validation_defaults_paths=None
):
    """Join separate files and check that their query operations still agree.

    Question generation and real teachers do not call this loader. The merged
    dictionaries are private inputs to synthetic generation and offline audits.
    Explicit semantics paths correspond to template paths in the same order.
    """
    template_paths = list(template_paths)
    semantics_paths = (
        list(semantics_paths)
        if semantics_paths is not None
        else [default_semantics_path(path) for path in template_paths]
    )
    if len(template_paths) != len(semantics_paths):
        raise ValueError("Provide one synthetic reasoning file per question template file")
    validation_paths = list(validation_paths or [None] * len(template_paths))
    validation_defaults_paths = list(validation_defaults_paths or [None] * len(template_paths))
    if len(validation_paths) != len(template_paths) or len(validation_defaults_paths) != len(
        template_paths
    ):
        raise ValueError("Provide one validation config/defaults path per question template file")
    catalog = []
    seen = set()
    for template_path, semantics_path, validation_path, defaults_path in zip(
        template_paths, semantics_paths, validation_paths, validation_defaults_paths, strict=True
    ):
        from cypher_3dsg_agent_distillation.cypher.validation_config import (
            default_validation_path,
            load_question_catalog,
        )

        if validation_path or default_validation_path(template_path).exists():
            templates = load_question_catalog(template_path, validation_path, defaults_path)[
                "templates"
            ]
        else:
            templates = yaml.safe_load(Path(template_path).read_text())["templates"]
        config = yaml.safe_load(Path(semantics_path).read_text())
        version = config.get("version")
        if type(version) is not int or version != SEMANTICS_VERSION:
            raise ValueError(
                f"Unsupported synthetic reasoning version in {semantics_path}: {version}"
            )
        declarations = config.get("templates")
        if not isinstance(declarations, dict):
            raise ValueError(
                f"Synthetic reasoning templates must be a query_type mapping: {semantics_path}"
            )
        names = {template["query_type"] for template in templates}
        missing = names - set(declarations)
        extra = set(declarations) - names
        if missing or extra:
            raise ValueError(
                f"Synthetic reasoning catalog differs from {template_path}: "
                f"missing={sorted(missing)}, obsolete={sorted(extra)}; run compile --write"
            )
        for template in templates:
            query_type = template["query_type"]
            if query_type in seen:
                raise ValueError(f"Duplicate query_type: {query_type}")
            seen.add(query_type)
            declaration = declarations[query_type]
            if not isinstance(declaration, dict) or "version" in declaration:
                raise ValueError(f"{query_type}: declare version once at the file level")
            semantics = {"version": version, **declaration}
            try:
                validate_query_semantics(semantics, template["cypher"])
                validate_goal_semantics(
                    template.get("goal"), template.get("grounding", {}).get("fields", {})
                )
            except ValueError as exc:
                raise ValueError(f"Question type {query_type}: {exc}") from exc
            catalog.append({**template, "semantics": semantics})
    return catalog
