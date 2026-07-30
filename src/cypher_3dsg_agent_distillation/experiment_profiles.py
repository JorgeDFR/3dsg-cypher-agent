from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cypher_3dsg_agent_distillation.config import load_yaml
from cypher_3dsg_agent_distillation.heracles_validation import (
    answer_formatting_guidance,
    answer_type_hint,
)

TaskFamily = Literal["cypher_qa", "cypher_pddl"]
OutputType = Literal["SLDP", "PDDL"]
ToolInterface = Literal["ollama", "openrouter", "custom"]
ReasoningPolicy = Literal["none", "brief_intent"]
SUPPORTED_TOOL_INTERFACES: tuple[str, ...] = ("ollama", "openrouter", "custom")


class ExperimentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    task_family: TaskFamily
    output_type: OutputType
    prompt_path: str
    labelspace_path: str | None = None
    scene_graph_description_path: str | None = None
    interface_description_path: str | None = None
    domain_description_path: str | None = None
    in_context_examples_path: str | None = None
    answer_type_hint: bool = False
    max_tool_calls: int = 5
    reasoning_policy: ReasoningPolicy = "none"
    tool_interfaces: list[ToolInterface] = Field(
        default_factory=lambda: ["ollama", "openrouter", "custom"]
    )

    @field_validator("tool_interfaces")
    @classmethod
    def validate_tool_interfaces(cls, value: list[ToolInterface]) -> list[ToolInterface]:
        if not value:
            raise ValueError("Experiment profile must allow at least one tool interface.")
        if set(value) - set(SUPPORTED_TOOL_INTERFACES):
            raise ValueError(f"Supported tool interfaces are {SUPPORTED_TOOL_INTERFACES}.")
        return value

    @model_validator(mode="after")
    def validate_pddl_context(self) -> ExperimentProfile:
        if self.output_type == "PDDL" and self.task_family != "cypher_pddl":
            raise ValueError("PDDL output requires task_family=cypher_pddl.")
        return self


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str
    novel_instruction_preamble: str | None = None
    novel_instruction_template: str = "{question}"
    answer_semantic_guidance: str | None = "Make your answer as concise as possible."


def resolve_path(path: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(path)))
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def load_text_section(path: str | None, preferred_keys: list[str]) -> str | None:
    if path is None:
        return None
    resolved = resolve_path(path)
    data = load_yaml(resolved)
    for key in preferred_keys:
        value = data.get(key)
        if value is not None:
            return str(value).strip()
    if len(data) == 1:
        return str(next(iter(data.values()))).strip()
    raise KeyError(f"Could not find any of {preferred_keys} in {resolved}.")


def load_experiment_profile(path: str | Path) -> ExperimentProfile:
    return ExperimentProfile.model_validate(load_yaml(resolve_path(path)))


def load_experiment_profile_from_config(data_config: dict[str, Any]) -> ExperimentProfile | None:
    profile_path = data_config.get("experiment_profile_path")
    if not profile_path:
        return None
    return load_experiment_profile(profile_path)


def profile_seed_defaults(profile: ExperimentProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    return {
        "task_family": profile.task_family,
        "output_type": profile.output_type,
        "metadata": {
            "experiment_profile": profile.name,
            "output_type": profile.output_type,
            "task_family": profile.task_family,
            "comparison_backend": "heracles_agents",
        },
    }


def load_prompt_template(profile: ExperimentProfile) -> PromptTemplate:
    return PromptTemplate.model_validate(load_yaml(resolve_path(profile.prompt_path)))


def render_profile_messages(
    profile: ExperimentProfile,
    seed: dict[str, Any],
    *,
    tool_interface: ToolInterface,
) -> list[dict[str, Any]]:
    if tool_interface not in profile.tool_interfaces:
        raise ValueError(
            f"Tool interface `{tool_interface}` is not enabled for profile `{profile.name}`."
        )

    prompt = load_prompt_template(profile)
    sections = [
        prompt.system.strip(),
        load_text_section(profile.scene_graph_description_path, ["scene_graph_description"]),
        load_text_section(profile.labelspace_path, ["labelspace_description"]),
        load_text_section(profile.interface_description_path, ["interface_description"]),
        load_text_section(profile.domain_description_path, ["domain_description"]),
        load_text_section(profile.in_context_examples_path, ["in_context_examples"]),
        prompt.novel_instruction_preamble.strip() if prompt.novel_instruction_preamble else None,
        prompt.novel_instruction_template.format(question=str(seed["question"]).strip()).strip(),
        (
            prompt.answer_semantic_guidance.strip()
            if prompt.answer_semantic_guidance
            else None
        ),
        answer_formatting_guidance(profile.output_type),
    ]
    if profile.answer_type_hint and seed.get("solution") is not None:
        sections.append(answer_type_hint(str(seed["solution"]), profile.output_type))

    return [
        {"role": "user", "content": section}
        for section in sections
        if section
    ]
