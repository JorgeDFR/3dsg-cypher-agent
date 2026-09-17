from __future__ import annotations

import argparse
import json
import math
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

from openrouter import OpenRouter
from openrouter import errors as openrouter_errors
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from cypher_3dsg_agent_distillation.agents.agent_prompts import run_cypher_tool_schema
from cypher_3dsg_agent_distillation.agents.agent_records import (
    DEFAULT_AGENT_TOOL,
    DEFAULT_TEACHER_MODEL,
    WRITE_REFUSAL_ANSWER,
    AgentTraceRecord,
    split_group_for_record,
)
from cypher_3dsg_agent_distillation.agents.experiment_profiles import (
    ExperimentProfile,
    ToolInterface,
    experiment_profile_for_seed,
    load_experiment_profiles_from_config,
    render_profile_messages,
)
from cypher_3dsg_agent_distillation.core.cli_output import print_summary
from cypher_3dsg_agent_distillation.core.config import (
    PROJECT_ROOT,
    ensure_parent,
    load_data_config,
    load_student_family_config,
    load_yaml,
    resolve_project_path,
    teacher_name_from_config,
    with_teacher_run,
)
from cypher_3dsg_agent_distillation.core.io import read_jsonl
from cypher_3dsg_agent_distillation.core.model_runtime_output import (
    configure_model_library_output,
)
from cypher_3dsg_agent_distillation.core.scene_ordering import group_records_by_scene
from cypher_3dsg_agent_distillation.core.versioning import (
    atomic_write_yaml,
    config_record,
    file_fingerprint,
    git_provenance,
    iso_now,
    serializable_arguments,
)
from cypher_3dsg_agent_distillation.cypher.evaluation import (
    ResultShapeError,
    normalize_answer,
    rows_to_solution_text,
)
from cypher_3dsg_agent_distillation.cypher.pddl_goals import validate_grounding_rows
from cypher_3dsg_agent_distillation.graph.heracles_validation import answer_matches_with_heracles
from cypher_3dsg_agent_distillation.graph.load_neo4j import load_scene
from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_URI,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    Neo4jValidator,
    is_safe_refusal,
    neo4j_uri_candidates,
    schema_policy,
    static_checks,
    suppress_neo4j_notification_logs,
)

DEFAULT_MAX_TOOL_CALLS = 3
TeacherProvider = Literal["openrouter", "synthetic"]
SUPPORTED_TEACHER_PROVIDERS: tuple[str, ...] = ("openrouter", "synthetic")
TOOL_REPAIR_MESSAGE = "The query could not be executed safely. Revise the Cypher query."


@dataclass
class TraceDecision:
    accepted: bool
    record: AgentTraceRecord | None
    reason: str
    payload: dict[str, Any]


class OpenRouterTeacherClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_TEACHER_MODEL,
        timeout: int = 120,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        sdk_client: Any | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sdk_client = sdk_client
        self._owns_sdk_client = sdk_client is None

    @property
    def timeout_ms(self) -> int:
        return int(self.timeout * 1000)

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for teacher generation.")
        return self.api_key

    def _sdk(self) -> Any:
        if self._sdk_client is None:
            self._sdk_client = OpenRouter(
                api_key=self.require_api_key(),
                timeout_ms=self.timeout_ms,
            )
        return self._sdk_client

    def close(self) -> None:
        if self._sdk_client is not None and self._owns_sdk_client:
            self._sdk_client.__exit__(None, None, None)
        self._sdk_client = None

    def __enter__(self) -> OpenRouterTeacherClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def preflight(self) -> None:
        try:
            response = self._sdk().models.list(
                q=self.model,
                limit=1000,
                timeout_ms=self.timeout_ms,
            )
        except openrouter_errors.OpenRouterError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        models = {model.id for model in response.result.data}
        if self.model not in models:
            raise RuntimeError(
                f"OpenRouter model `{self.model}` is unavailable. "
                f"Check https://openrouter.ai/models or choose a fallback."
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._sdk().chat.send(
                model=self.model,
                messages=sdk_outbound_messages(messages),
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
                timeout_ms=self.timeout_ms,
            )
        except openrouter_errors.OpenRouterError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        return normalize_assistant_message(sdk_message_to_dict(response.choices[0].message))


def object_to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: object_to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [object_to_plain_dict(item) for item in value]
    if hasattr(value, "model_dump"):
        return object_to_plain_dict(value.model_dump(exclude_none=True, by_alias=True))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return object_to_plain_dict(vars(value))
    return value


def sdk_message_to_dict(message: Any) -> dict[str, Any]:
    payload = object_to_plain_dict(message)
    if not isinstance(payload, dict):
        raise RuntimeError("OpenRouter response message was not an object.")
    payload.setdefault("role", "assistant")
    return payload


def normalize_tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(tool_call)
    function = dict(normalized.get("function") or {})
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tool arguments JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    function["arguments"] = arguments
    normalized["function"] = function
    normalized.setdefault("type", "function")
    return normalized


def normalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: value for key, value in message.items() if value is not None}
    normalized["role"] = "assistant"
    if normalized.get("tool_calls"):
        tool_calls = []
        for tool_call in normalized.get("tool_calls", []):
            try:
                tool_calls.append(normalize_tool_call_arguments(tool_call))
            except ValueError:
                raw_tool_call = dict(tool_call)
                raw_tool_call.setdefault("type", "function")
                tool_calls.append(raw_tool_call)
        normalized["tool_calls"] = tool_calls
    return normalized


def sdk_outbound_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outbound = []
    for message in messages:
        payload = {key: value for key, value in message.items() if value is not None}
        if payload.get("role") == "assistant" and payload.get("tool_calls"):
            tool_calls = []
            for tool_call in payload["tool_calls"]:
                outbound_call = dict(tool_call)
                function = dict(outbound_call.get("function") or {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, dict):
                    function["arguments"] = json.dumps(arguments, ensure_ascii=False)
                outbound_call["function"] = function
                tool_calls.append(outbound_call)
            payload["tool_calls"] = tool_calls
        if payload.get("role") == "tool":
            payload.pop("tool_name", None)
            if not payload.get("tool_call_id"):
                payload.pop("tool_call_id", None)
        outbound.append(payload)
    return outbound


def extract_single_cypher_tool_call(
    message: dict[str, Any],
) -> tuple[str, str | None, dict[str, Any]]:
    tool_call = message["tool_calls"][0]
    normalized = normalize_tool_call_arguments(tool_call)
    function = normalized["function"]
    tool_call_id = normalized.get("id") or normalized.get("tool_call_id")
    cypher = str(function.get("arguments", {}).get("cypher_string") or "").strip()
    return cypher, tool_call_id, normalized


def build_tool_result_message(tool_call_id: str | None, content: str) -> dict[str, Any]:
    message = {
        "role": "tool",
        "tool_name": DEFAULT_AGENT_TOOL,
        "content": content,
    }
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def answer_inner_text(content: str | None) -> str | None:
    if not content:
        return None
    match = re.search(r"<answer>(.*?)</answer>", content, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def tool_error_content(error_type: str, issues: list[str]) -> str:
    return json.dumps(
        {
            "ok": False,
            "error_type": error_type,
            "issues": issues,
            "message": TOOL_REPAIR_MESSAGE,
        },
        ensure_ascii=False,
    )


def tool_success_content(rows: list[dict[str, Any]]) -> str:
    return json.dumps({"ok": True, "rows": rows}, ensure_ascii=False, default=str)


def controlled_repair_spec(seed: dict[str, Any]) -> tuple[str, str] | None:
    metadata = seed.get("metadata") or {}
    if not metadata.get("controlled_repair"):
        return None
    template_name = str(metadata.get("repair_template") or "").strip()
    invalid_cypher = str(metadata.get("repair_cypher") or "").strip()
    if not template_name or not invalid_cypher:
        raise ValueError(
            "Controlled repair seeds require metadata.repair_template and metadata.repair_cypher."
        )
    return template_name, invalid_cypher


def append_controlled_repair_context(
    messages: list[dict[str, Any]],
    seed: dict[str, Any],
    *,
    validator,
) -> int | None:
    spec = controlled_repair_spec(seed)
    if spec is None:
        return None
    if validator is None:
        raise ValueError("Controlled repair supervision requires Neo4j validation.")

    _template_name, invalid_cypher = spec
    tool_call_id = f"controlled_repair_{seed.get('uid', 'seed')}"
    validation = validator.validate(invalid_cypher, execute=True)
    if validation.get("executable"):
        raise ValueError("Configured repair mutation unexpectedly produced executable Cypher.")
    if validation.get("syntax_error") is not True:
        raise ValueError("Configured repair mutation did not produce a Cypher syntax error.")

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": DEFAULT_AGENT_TOOL,
                        "arguments": {"cypher_string": invalid_cypher},
                    },
                }
            ],
        }
    )
    messages.append(
        build_tool_result_message(
            tool_call_id,
            tool_error_content(
                "execution_failed",
                [str(validation.get("error") or "Cypher syntax error")],
            ),
        )
    )
    return len(messages)


def seed_expected_solution(seed: dict[str, Any]) -> Any:
    if "solution" in seed:
        return seed["solution"]
    return seed.get("expected_solution")


def seed_real_solution(seed: dict[str, Any]) -> Any:
    return seed_expected_solution(seed)


def final_answer_value(content: str | None) -> str | None:
    if not content:
        return None
    inner = answer_inner_text(content)
    if inner is not None:
        return inner
    text = content.strip()
    return text or None


def final_answer_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return final_answer_value(message.get("content"))
    return None


def validation_with_solution_checks(
    validation: dict[str, Any],
    *,
    cypher_solution_match: bool | None = None,
    final_answer_match: bool | None = None,
    final_answer_cypher_match: bool | None = None,
) -> dict[str, Any]:
    solution_match = (
        bool(final_answer_match)
        if cypher_solution_match is None
        else bool(cypher_solution_match and final_answer_match)
    )
    return {
        **validation,
        "solution_match": solution_match,
        "cypher_solution_match": cypher_solution_match,
        "final_answer_match": final_answer_match,
        "final_answer_cypher_match": final_answer_cypher_match,
    }


def build_candidate_trace(
    seed: dict[str, Any],
    candidate: dict[str, Any],
    teacher_model: str,
) -> AgentTraceRecord:
    cypher = str(seed.get("cypher") or seed.get("gold_cypher") or "").strip()
    real_solution = seed_real_solution(seed)
    return AgentTraceRecord(
        uid=str(candidate.get("uid") or f"{seed['uid']}_teacher"),
        scene_id=seed.get("scene_id"),
        question=str(candidate.get("question") or seed["question"]),
        query_type=seed.get("query_type", "unknown"),
        task_family=seed.get("task_family", "cypher_qa"),
        output_type=seed.get("output_type", "SLDP"),
        experiment_profile=seed.get("experiment_profile"),
        messages=candidate["messages"],
        gold_cypher=cypher,
        expected_solution=real_solution,
        real_solution=real_solution,
        solution_formats=seed.get("solution_formats", []),
        answer_validation=seed.get("answer_validation"),
        grounding=seed.get("grounding"),
        final_answer=final_answer_from_messages(candidate["messages"]),
        split_group=split_group_for_record(
            seed.get("scene_id"),
            seed.get("query_type", "unknown"),
            cypher,
            seed.get("task_family", "cypher_qa"),
        ),
        teacher_model=teacher_model,
        read_only=bool(seed.get("read_only", True)),
        tags=seed.get("tags", []),
        metadata={
            **dict(seed.get("metadata") or {}),
            "seed_uid": seed.get("uid"),
            "task_family": seed.get("task_family", "cypher_qa"),
            "output_type": seed.get("output_type", "SLDP"),
        },
    )


def build_interaction_trace(
    seed: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    teacher_model: str,
    cypher: str,
    validation: dict[str, Any],
    max_tool_calls: int,
    tool_call_count: int,
    successful_tool_call_index: int | None,
    cypher_solution: str | None,
    static_only: bool = False,
) -> AgentTraceRecord:
    seed_cypher = str(seed.get("cypher") or seed.get("gold_cypher") or "").strip()
    real_solution = seed_real_solution(seed)
    metadata = {
        **dict(seed.get("metadata") or {}),
        "seed_uid": seed.get("uid"),
        "seed_gold_cypher": seed_cypher,
        "generation_mode": "tool_conversation",
        "task_family": seed.get("task_family", "cypher_qa"),
        "output_type": seed.get("output_type", "SLDP"),
        "max_tool_calls": max_tool_calls,
        "tool_call_count": tool_call_count,
        "successful_tool_call_index": successful_tool_call_index,
        "cypher_solution": cypher_solution,
        "repair_attempted": tool_call_count > 1,
    }
    if metadata.get("controlled_repair"):
        repair_cypher = str(metadata.get("repair_cypher") or "").strip()
        for index, message in enumerate(messages[:-1]):
            calls = message.get("tool_calls") or []
            arguments = (calls[0].get("function") or {}).get("arguments") if calls else None
            is_repair_call = (
                isinstance(arguments, dict)
                and str(arguments.get("cypher_string") or "").strip() == repair_cypher
            )
            if is_repair_call:
                metadata["repair_context_message_count"] = index + 2
                metadata["non_trainable_message_indices"] = [index, index + 1]
                break
    if static_only:
        metadata["static_only"] = True
        metadata["incomplete_conversation"] = True
    return AgentTraceRecord(
        uid=f"{seed['uid']}_teacher",
        scene_id=seed.get("scene_id"),
        question=str(seed["question"]),
        query_type=seed.get("query_type", "unknown"),
        task_family=seed.get("task_family", "cypher_qa"),
        output_type=seed.get("output_type", "SLDP"),
        experiment_profile=seed.get("experiment_profile"),
        messages=messages,
        gold_cypher=cypher,
        expected_solution=real_solution,
        real_solution=real_solution,
        solution_formats=seed.get("solution_formats", []),
        answer_validation=seed.get("answer_validation"),
        grounding=seed.get("grounding"),
        cypher_solution=cypher_solution,
        final_answer=final_answer_from_messages(messages),
        split_group=split_group_for_record(
            seed.get("scene_id"),
            seed.get("query_type", "unknown"),
            seed_cypher,
            seed.get("task_family", "cypher_qa"),
        ),
        teacher_model=teacher_model,
        validation=validation,
        read_only=bool(seed.get("read_only", True)),
        tags=seed.get("tags", []),
        metadata=metadata,
    )


def rejected(
    reason: str,
    *,
    messages: list[dict[str, Any]],
    cypher: str | None = None,
    validation: dict[str, Any] | None = None,
    cypher_solution: str | None = None,
    final_answer: str | None = None,
    extra: dict[str, Any] | None = None,
) -> TraceDecision:
    payload = {
        "messages": messages,
        "cypher": cypher,
        "validation": validation,
        "cypher_solution": cypher_solution,
        "final_answer": final_answer_value(final_answer),
    }
    if extra:
        payload.update(extra)
    return TraceDecision(False, None, reason, payload)


def evaluate_candidate_trace(
    seed: dict[str, Any],
    candidate: dict[str, Any],
    *,
    teacher_model: str,
    validator: Neo4jValidator | None,
    policy: dict[str, Any] | None,
) -> TraceDecision:
    real_solution = seed_real_solution(seed)
    output_type = seed.get("output_type", "SLDP")
    try:
        record = build_candidate_trace(seed, candidate, teacher_model)
    except Exception as exc:
        return TraceDecision(False, None, f"invalid_trace_schema: {exc}", candidate)

    if record.is_refusal:
        if not is_safe_refusal(record.gold_cypher):
            return TraceDecision(False, record, "unexpected_refusal", candidate)
        record.final_answer = final_answer_value(record.final_answer_text)
        record.validation.static_valid = True
        record.validation.executable = True
        record.validation.cypher_solution_match = True
        record.validation.final_answer_cypher_match = True
        record.validation.final_answer_match = normalize_answer(real_solution) in {
            "",
            normalize_answer(record.final_answer),
        } or WRITE_REFUSAL_ANSWER in normalize_answer(record.final_answer)
        record.validation.solution_match = bool(record.validation.final_answer_match)
        return TraceDecision(
            record.validation.solution_match,
            record,
            "accepted_refusal",
            candidate,
        )

    cypher = record.first_cypher_argument
    if not cypher:
        return TraceDecision(False, record, "missing_cypher_argument", candidate)
    if DEFAULT_AGENT_TOOL not in [call.function.name for call in record.tool_calls]:
        return TraceDecision(False, record, "missing_required_tool", candidate)

    static_issues = static_checks(cypher, allow_writes=not record.read_only, policy=policy)
    record.validation.static_issues = static_issues
    record.validation.static_valid = not static_issues
    if static_issues:
        return TraceDecision(False, record, "static_issues", candidate)

    if validator is None:
        record.validation.executable = True
        record.validation.solution_match = True
        return TraceDecision(True, record, "accepted_static_only", candidate)

    validation = validator.validate(cypher, execute=True)
    record.validation.executable = bool(validation.get("executable"))
    record.validation.error = validation.get("error")
    if not record.validation.executable:
        return TraceDecision(
            False,
            record,
            "execution_failed",
            {**candidate, "validation": validation},
        )

    task_family = str(seed.get("task_family") or "cypher_qa")
    if task_family == "cypher_pddl":
        cypher_matches_real, cypher_solution, _actual_grounding = validate_grounding_rows(
            validation.get("rows", []), seed.get("grounding")
        )
        valid_cypher_format = True
    else:
        try:
            cypher_solution = rows_to_solution_text(seed, validation.get("rows", []))
        except ResultShapeError as exc:
            record.validation.error = str(exc)
            return TraceDecision(
                False,
                record,
                "result_shape_error",
                {**candidate, "cypher_solution": None, "validation": validation},
            )
        valid_cypher_format, cypher_matches_real = answer_matches_with_heracles(
            cypher_solution,
            str(real_solution),
            output_type,
            answer_validation=seed.get("answer_validation"),
            solution_formats=seed.get("solution_formats") or seed.get("solution_format"),
        )
    record.cypher_solution = cypher_solution
    record.validation.cypher_solution_match = cypher_matches_real
    record.validation.tool_result_match = cypher_matches_real
    cypher_checked_validation = validation_with_solution_checks(
        validation,
        cypher_solution_match=cypher_matches_real,
    )
    if not valid_cypher_format:
        return TraceDecision(
            False,
            record,
            "invalid_answer_format",
            {
                **candidate,
                "cypher_solution": cypher_solution,
                "validation": cypher_checked_validation,
            },
        )
    if not cypher_matches_real:
        return TraceDecision(
            False,
            record,
            "solution_mismatch",
            {
                **candidate,
                "cypher_solution": cypher_solution,
                "validation": cypher_checked_validation,
            },
        )

    inner_answer = answer_inner_text(record.final_answer_text)
    record.final_answer = final_answer_value(record.final_answer_text)
    if inner_answer is None:
        return TraceDecision(
            False,
            record,
            "missing_answer_tags",
            {**candidate, "cypher_solution": cypher_solution, "validation": validation},
        )
    valid_final_format, final_matches_real = answer_matches_with_heracles(
        inner_answer,
        str(real_solution),
        output_type,
        answer_validation=seed.get("answer_validation"),
        solution_formats=seed.get("solution_formats") or seed.get("solution_format"),
    )
    if task_family == "cypher_qa":
        valid_consistency_format, final_matches_cypher = answer_matches_with_heracles(
            inner_answer,
            str(cypher_solution),
            output_type,
            solution_formats=seed.get("solution_formats") or seed.get("solution_format"),
        )
    else:
        valid_consistency_format = True
        final_matches_cypher = cypher_matches_real and final_matches_real
    record.validation.final_answer_match = final_matches_real
    record.validation.final_answer_cypher_match = final_matches_cypher
    record.validation.final_answer_format_valid = valid_final_format
    record.validation.final_answer_domain_valid = valid_final_format
    record.validation.answer_grounded = final_matches_cypher
    record.validation.solution_match = bool(
        cypher_matches_real and final_matches_real and final_matches_cypher
    )
    checked_validation = validation_with_solution_checks(
        validation,
        cypher_solution_match=cypher_matches_real,
        final_answer_match=final_matches_real,
        final_answer_cypher_match=final_matches_cypher,
    )
    if not valid_final_format or not valid_consistency_format:
        return TraceDecision(
            False,
            record,
            "invalid_answer_format",
            {
                **candidate,
                "cypher_solution": cypher_solution,
                "validation": checked_validation,
            },
        )
    if not final_matches_real:
        return TraceDecision(
            False,
            record,
            "final_answer_mismatch",
            {
                **candidate,
                "cypher_solution": cypher_solution,
                "validation": checked_validation,
            },
        )
    if not final_matches_cypher:
        return TraceDecision(
            False,
            record,
            "final_answer_cypher_mismatch",
            {
                **candidate,
                "cypher_solution": cypher_solution,
                "validation": checked_validation,
            },
        )
    return TraceDecision(True, record, "accepted", candidate)


def run_teacher_interaction(
    seed: dict[str, Any],
    *,
    profile: ExperimentProfile,
    tool_interface: ToolInterface = "openrouter",
    client: OpenRouterTeacherClient,
    teacher_model: str,
    validator: Neo4jValidator | None,
    policy: dict[str, Any] | None,
    max_tool_calls: int,
    static_only: bool,
) -> TraceDecision:
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be >= 1.")

    messages = render_profile_messages(profile, seed, tool_interface=tool_interface)

    try:
        repair_context_message_count = append_controlled_repair_context(
            messages,
            seed,
            validator=validator,
        )
    except ValueError as exc:
        return rejected(
            "invalid_controlled_repair",
            messages=messages,
            extra={"error": str(exc)},
        )
    if repair_context_message_count is not None and max_tool_calls < 2:
        return rejected(
            "invalid_controlled_repair",
            messages=messages,
            extra={"error": "Controlled repair supervision requires max_tool_calls >= 2."},
        )

    task_family = str(seed.get("task_family") or profile.task_family)
    output_type = str(seed.get("output_type") or profile.output_type)
    seed_cypher = str(seed.get("cypher") or seed.get("gold_cypher") or "").strip()
    real_solution = seed_real_solution(seed)
    expected_refusal = is_safe_refusal(seed_cypher)
    tool_call_count = 1 if repair_context_message_count is not None else 0
    successful_tool_call_index: int | None = None
    final_successful_cypher: str | None = None
    final_validation: dict[str, Any] | None = None
    cypher_solution: str | None = None
    last_failure_reason: str | None = None

    while True:
        assistant = client.chat(
            messages,
            tools=[run_cypher_tool_schema()],
            tool_choice="auto",
        )
        messages.append(assistant)
        tool_calls = assistant.get("tool_calls") or []
        final_answer = assistant.get("content")

        if expected_refusal:
            if tool_calls:
                return rejected(
                    "unexpected_tool_call_for_refusal",
                    messages=messages,
                    final_answer=final_answer,
                )
            if WRITE_REFUSAL_ANSWER not in str(final_answer or ""):
                return rejected("missing_refusal", messages=messages, final_answer=final_answer)
            record = build_interaction_trace(
                seed,
                messages=messages,
                teacher_model=teacher_model,
                cypher=WRITE_REFUSAL_ANSWER,
                validation={
                    "static_valid": True,
                    "executable": True,
                    "solution_match": True,
                    "cypher_solution_match": True,
                    "final_answer_match": True,
                    "final_answer_cypher_match": True,
                },
                max_tool_calls=max_tool_calls,
                tool_call_count=0,
                successful_tool_call_index=None,
                cypher_solution=WRITE_REFUSAL_ANSWER,
            )
            return TraceDecision(True, record, "accepted_refusal", {"messages": messages})

        if tool_calls:
            if len(tool_calls) > 1:
                return rejected("multiple_tool_calls", messages=messages)
            tool_call_count += 1
            try:
                cypher, tool_call_id, tool_call = extract_single_cypher_tool_call(assistant)
            except ValueError as exc:
                return rejected(
                    "invalid_tool_arguments",
                    messages=messages,
                    extra={"error": str(exc)},
                )
            tool_name = tool_call.get("function", {}).get("name")
            if tool_name != DEFAULT_AGENT_TOOL:
                return rejected("wrong_tool_name", messages=messages, cypher=cypher)
            if not cypher:
                return rejected("missing_cypher_argument", messages=messages)

            static_issues = static_checks(
                cypher,
                allow_writes=not bool(seed.get("read_only", True)),
                policy=policy,
            )
            if static_issues:
                last_failure_reason = "static_issues"
                if tool_call_count >= max_tool_calls:
                    return rejected(
                        "max_tool_calls_exceeded",
                        messages=messages,
                        cypher=cypher,
                        validation={"static_issues": static_issues},
                        extra={"last_failure_reason": last_failure_reason},
                    )
                messages.append(
                    build_tool_result_message(
                        tool_call_id,
                        tool_error_content("static_issues", static_issues),
                    )
                )
                continue

            if static_only:
                record = build_interaction_trace(
                    seed,
                    messages=messages,
                    teacher_model=teacher_model,
                    cypher=cypher,
                    validation={
                        "static_valid": True,
                        "executable": True,
                        "solution_match": True,
                    },
                    max_tool_calls=max_tool_calls,
                    tool_call_count=tool_call_count,
                    successful_tool_call_index=tool_call_count,
                    cypher_solution=None,
                    static_only=True,
                )
                return TraceDecision(True, record, "accepted_static_only", {"messages": messages})

            if validator is None:
                return rejected("execution_failed", messages=messages, cypher=cypher)

            validation = validator.validate(cypher, execute=True)
            if not validation.get("executable"):
                last_failure_reason = "execution_failed"
                if tool_call_count >= max_tool_calls:
                    return rejected(
                        "max_tool_calls_exceeded",
                        messages=messages,
                        cypher=cypher,
                        validation=validation,
                        extra={"last_failure_reason": last_failure_reason},
                    )
                messages.append(
                    build_tool_result_message(
                        tool_call_id,
                        tool_error_content(
                            "execution_failed",
                            [str(validation.get("error") or "execution failed")],
                        ),
                    )
                )
                continue

            final_successful_cypher = cypher
            final_validation = validation
            successful_tool_call_index = tool_call_count
            if task_family == "cypher_qa":
                try:
                    cypher_solution = rows_to_solution_text(seed, validation.get("rows", []))
                except ResultShapeError as exc:
                    last_failure_reason = "result_shape_error"
                    validation = {**validation, "result_shape_error": str(exc)}
                    if tool_call_count >= max_tool_calls:
                        return rejected(
                            "max_tool_calls_exceeded",
                            messages=messages,
                            cypher=cypher,
                            validation=validation,
                            extra={"last_failure_reason": last_failure_reason},
                        )
                    messages.append(
                        build_tool_result_message(
                            tool_call_id,
                            tool_error_content("result_shape_error", [str(exc)]),
                        )
                    )
                    continue
            elif task_family == "cypher_pddl":
                grounding_match, cypher_solution, _actual_grounding = validate_grounding_rows(
                    validation.get("rows", []), seed.get("grounding")
                )
                if not grounding_match:
                    last_failure_reason = "grounding_mismatch"
                    validation = {**validation, "grounding_mismatch": True}
                    if tool_call_count >= max_tool_calls:
                        return rejected(
                            "max_tool_calls_exceeded",
                            messages=messages,
                            cypher=cypher,
                            validation=validation_with_solution_checks(
                                validation, cypher_solution_match=False
                            ),
                            cypher_solution=cypher_solution,
                            extra={"last_failure_reason": last_failure_reason},
                        )
                    messages.append(
                        build_tool_result_message(
                            tool_call_id,
                            tool_error_content(
                                "grounding_mismatch",
                                ["The query did not retrieve the required grounded entities."],
                            ),
                        )
                    )
                    continue
            messages.append(
                build_tool_result_message(
                    tool_call_id,
                    tool_success_content(validation.get("rows", [])),
                )
            )
            continue

        if final_successful_cypher is None:
            if last_failure_reason:
                return rejected(last_failure_reason, messages=messages, final_answer=final_answer)
            return rejected("missing_required_tool", messages=messages, final_answer=final_answer)
        if not final_answer:
            return rejected("missing_final_answer", messages=messages)

        inner_answer = answer_inner_text(final_answer)
        if inner_answer is None:
            return rejected("missing_answer_tags", messages=messages, final_answer=final_answer)
        cypher_solution_match: bool | None = None
        final_answer_cypher_match: bool | None = None
        if task_family == "cypher_qa":
            valid_cypher_format, cypher_solution_match = answer_matches_with_heracles(
                str(cypher_solution),
                str(real_solution),
                output_type,
                answer_validation=seed.get("answer_validation"),
                solution_formats=seed.get("solution_formats") or seed.get("solution_format"),
            )
            cypher_checked_validation = validation_with_solution_checks(
                final_validation or {},
                cypher_solution_match=cypher_solution_match,
            )
            if not valid_cypher_format:
                return rejected(
                    "invalid_answer_format",
                    messages=messages,
                    cypher=final_successful_cypher,
                    validation=cypher_checked_validation,
                    cypher_solution=cypher_solution,
                    final_answer=final_answer,
                )
            if not cypher_solution_match:
                return rejected(
                    "solution_mismatch",
                    messages=messages,
                    cypher=final_successful_cypher,
                    validation=cypher_checked_validation,
                    cypher_solution=cypher_solution,
                    final_answer=final_answer,
                )
            valid_consistency_format, final_answer_cypher_match = answer_matches_with_heracles(
                inner_answer,
                str(cypher_solution),
                output_type,
                solution_formats=seed.get("solution_formats") or seed.get("solution_format"),
            )
            if not valid_consistency_format:
                return rejected(
                    "invalid_answer_format",
                    messages=messages,
                    cypher=final_successful_cypher,
                    validation=validation_with_solution_checks(
                        final_validation or {},
                        cypher_solution_match=cypher_solution_match,
                        final_answer_cypher_match=final_answer_cypher_match,
                    ),
                    cypher_solution=cypher_solution,
                    final_answer=final_answer,
                )
        elif task_family == "cypher_pddl":
            # The successful tool call was already checked against the seed's
            # grounding contract before its rows were added to the conversation.
            cypher_solution_match = True
            final_answer_cypher_match = None
        valid_final_format, final_matches = answer_matches_with_heracles(
            inner_answer,
            str(real_solution),
            output_type,
            answer_validation=seed.get("answer_validation"),
            solution_formats=seed.get("solution_formats") or seed.get("solution_format"),
        )
        checked_validation = validation_with_solution_checks(
            final_validation or {},
            cypher_solution_match=cypher_solution_match,
            final_answer_match=final_matches,
            final_answer_cypher_match=final_answer_cypher_match,
        )
        checked_validation.update(
            {
                "tool_result_match": cypher_solution_match,
                "final_answer_format_valid": valid_final_format,
                "final_answer_domain_valid": valid_final_format,
                "answer_grounded": bool(cypher_solution_match and final_matches),
            }
        )
        if not valid_final_format:
            return rejected(
                "invalid_answer_format",
                messages=messages,
                cypher=final_successful_cypher,
                validation=checked_validation,
                cypher_solution=cypher_solution,
                final_answer=final_answer,
            )
        if not final_matches:
            return rejected(
                "final_answer_mismatch",
                messages=messages,
                cypher=final_successful_cypher,
                validation=checked_validation,
                cypher_solution=cypher_solution,
                final_answer=final_answer,
            )
        if final_answer_cypher_match is False:
            return rejected(
                "final_answer_cypher_mismatch",
                messages=messages,
                cypher=final_successful_cypher,
                validation=checked_validation,
                cypher_solution=cypher_solution,
                final_answer=final_answer,
            )

        record = build_interaction_trace(
            seed,
            messages=messages,
            teacher_model=teacher_model,
            cypher=final_successful_cypher,
            validation={
                **checked_validation,
                "static_valid": True,
                "executable": True,
                "error": None,
            },
            max_tool_calls=max_tool_calls,
            tool_call_count=tool_call_count,
            successful_tool_call_index=successful_tool_call_index,
            cypher_solution=cypher_solution,
        )
        return TraceDecision(True, record, "accepted", {"messages": messages})


def configured_teacher_provider(config: dict[str, Any]) -> TeacherProvider:
    provider = str(config.get("provider") or "").strip().lower()
    if provider not in SUPPORTED_TEACHER_PROVIDERS:
        raise ValueError(
            f"Unsupported teacher provider `{provider or '<missing>'}`. "
            f"Expected one of {SUPPORTED_TEACHER_PROVIDERS}."
        )
    return "synthetic" if provider == "synthetic" else "openrouter"


def run_synthetic_teacher_interaction(
    seed: dict[str, Any],
    *,
    profile: ExperimentProfile,
    tool_interface: ToolInterface = "openrouter",
    validator: Neo4jValidator,
    policy: dict[str, Any] | None,
    max_tool_calls: int = 2,
) -> TraceDecision:
    """Construct a validated trace from gold Cypher, gold solution, and Neo4j rows."""
    messages = render_profile_messages(profile, seed, tool_interface=tool_interface)
    cypher = str(seed.get("cypher") or seed.get("gold_cypher") or "").strip()
    if not cypher:
        return rejected("missing_gold_cypher", messages=messages)

    real_solution = seed_real_solution(seed)
    if real_solution is None:
        return rejected("missing_gold_solution", messages=messages, cypher=cypher)

    if is_safe_refusal(cypher):
        messages.append(
            {"role": "assistant", "content": f"<answer>{WRITE_REFUSAL_ANSWER}</answer>"}
        )
        record = build_interaction_trace(
            seed,
            messages=messages,
            teacher_model=None,
            cypher=cypher,
            validation=validation_with_solution_checks(
                {"static_valid": True, "executable": True, "error": None},
                cypher_solution_match=True,
                final_answer_match=True,
                final_answer_cypher_match=True,
            ),
            max_tool_calls=max_tool_calls,
            tool_call_count=0,
            successful_tool_call_index=None,
            cypher_solution=WRITE_REFUSAL_ANSWER,
        )
        record.metadata["teacher_provider"] = "synthetic"
        record.metadata["generation_mode"] = "synthetic_gold_cypher"
        return TraceDecision(True, record, "accepted_refusal", {"messages": messages})

    static_issues = static_checks(
        cypher,
        allow_writes=not bool(seed.get("read_only", True)),
        policy=policy,
    )
    if static_issues:
        return rejected(
            "static_issues",
            messages=messages,
            cypher=cypher,
            validation={"static_valid": False, "static_issues": static_issues},
        )

    try:
        repair_context_message_count = append_controlled_repair_context(
            messages,
            seed,
            validator=validator,
        )
    except ValueError as exc:
        return rejected(
            "invalid_controlled_repair",
            messages=messages,
            cypher=cypher,
            extra={"error": str(exc)},
        )
    if repair_context_message_count is not None and max_tool_calls < 2:
        return rejected(
            "invalid_controlled_repair",
            messages=messages,
            cypher=cypher,
            extra={"error": "Controlled repair supervision requires max_tool_calls >= 2."},
        )

    tool_call_id = f"synthetic_{seed.get('uid', 'seed')}"
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": DEFAULT_AGENT_TOOL,
                        "arguments": {"cypher_string": cypher},
                    },
                }
            ],
        }
    )
    validation = validator.validate(cypher, execute=True)
    validation = {**validation, "static_valid": True}
    if not validation.get("executable"):
        messages.append(
            build_tool_result_message(
                tool_call_id,
                tool_error_content(
                    "execution_failed",
                    [str(validation.get("error") or "execution failed")],
                ),
            )
        )
        return rejected(
            "execution_failed",
            messages=messages,
            cypher=cypher,
            validation=validation,
        )
    messages.append(
        build_tool_result_message(
            tool_call_id,
            tool_success_content(validation.get("rows", [])),
        )
    )

    task_family = str(seed.get("task_family") or profile.task_family)
    output_type = str(seed.get("output_type") or profile.output_type)
    cypher_solution: str | None = None
    cypher_solution_match: bool | None = None
    final_answer_cypher_match: bool | None = None
    if task_family == "cypher_qa":
        try:
            cypher_solution = rows_to_solution_text(seed, validation.get("rows", []))
        except ResultShapeError as exc:
            return rejected(
                "result_shape_error",
                messages=messages,
                cypher=cypher,
                validation={**validation, "result_shape_error": str(exc)},
            )
        valid_cypher_format, cypher_solution_match = answer_matches_with_heracles(
            cypher_solution,
            str(real_solution),
            output_type,
            solution_formats=seed.get("solution_formats") or seed.get("solution_format"),
        )
        if not valid_cypher_format:
            return rejected(
                "invalid_answer_format",
                messages=messages,
                cypher=cypher,
                validation=validation,
                cypher_solution=cypher_solution,
            )
        if not cypher_solution_match:
            return rejected(
                "gold_solution_mismatch",
                messages=messages,
                cypher=cypher,
                validation=validation_with_solution_checks(
                    validation,
                    cypher_solution_match=False,
                ),
                cypher_solution=cypher_solution,
            )
        final_answer_cypher_match = True
    elif task_family == "cypher_pddl":
        cypher_solution_match, cypher_solution, _actual_grounding = validate_grounding_rows(
            validation.get("rows", []), seed.get("grounding")
        )
        if not cypher_solution_match:
            return rejected(
                "grounding_mismatch",
                messages=messages,
                cypher=cypher,
                validation=validation_with_solution_checks(
                    validation,
                    cypher_solution_match=False,
                ),
                cypher_solution=cypher_solution,
            )

    final_answer = str(real_solution).strip()
    valid_final_format, final_answer_match = answer_matches_with_heracles(
        final_answer,
        str(real_solution),
        output_type,
        solution_formats=seed.get("solution_formats") or seed.get("solution_format"),
    )
    if not valid_final_format or not final_answer_match:
        return rejected(
            "invalid_gold_solution",
            messages=messages,
            cypher=cypher,
            validation=validation,
            cypher_solution=cypher_solution,
            final_answer=final_answer,
        )

    messages.append({"role": "assistant", "content": f"<answer>{final_answer}</answer>"})
    checked_validation = validation_with_solution_checks(
        validation,
        cypher_solution_match=cypher_solution_match,
        final_answer_match=final_answer_match,
        final_answer_cypher_match=final_answer_cypher_match,
    )
    checked_validation.update(
        {
            "tool_result_match": cypher_solution_match,
            "final_answer_format_valid": valid_final_format,
            "final_answer_domain_valid": valid_final_format,
            "answer_grounded": bool(cypher_solution_match and final_answer_match),
        }
    )
    record = build_interaction_trace(
        seed,
        messages=messages,
        teacher_model=None,
        cypher=cypher,
        validation=checked_validation,
        max_tool_calls=max_tool_calls,
        tool_call_count=2 if repair_context_message_count is not None else 1,
        successful_tool_call_index=2 if repair_context_message_count is not None else 1,
        cypher_solution=cypher_solution,
    )
    record.metadata["teacher_provider"] = "synthetic"
    record.metadata["generation_mode"] = "synthetic_gold_cypher"
    return TraceDecision(True, record, "accepted", {"messages": messages})


def configured_max_tool_calls(
    args: argparse.Namespace,
    teacher_config: dict[str, Any],
    profile: ExperimentProfile,
) -> int:
    value = (
        args.max_tool_calls
        if args.max_tool_calls is not None
        else teacher_config.get(
            "max_tool_calls",
            profile.max_tool_calls,
        )
    )
    max_tool_calls = int(value)
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be >= 1.")
    return max_tool_calls


def progress_description(
    seed_index: int,
    seed_count: int,
    accepted_count: int,
    rejected_count: int,
) -> str:
    return (
        f"teacher traces seed {seed_index}/{seed_count} "
        f"accepted={accepted_count} rejected={rejected_count}"
    )


def nearest_rank_percentile(values: list[int], percentile: int) -> int | None:
    """Return an observed value at the requested nearest-rank percentile."""
    if not values:
        return None
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in the range 1..100.")
    ordered = sorted(values)
    rank = math.ceil(percentile / 100 * len(ordered))
    return ordered[rank - 1]


def trace_token_length_report(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Measure traces exactly as they are rendered and tokenized for training."""
    from cypher_3dsg_agent_distillation.training.train import (
        rendered_record_text,
        tokenize_rendered_text,
    )

    if max_length <= 0:
        raise ValueError("training.max_length must be greater than zero.")

    traces = []
    lengths = []
    for record in records:
        text = rendered_record_text(record, tokenizer)
        token_length = len(tokenize_rendered_text(tokenizer, text))
        lengths.append(token_length)
        traces.append(
            {
                "uid": str(record.get("uid") or ""),
                "token_length": token_length,
                "exceeds_max_length": token_length > max_length,
            }
        )

    overlength_count = sum(trace["exceeds_max_length"] for trace in traces)
    return {
        "trace_count": len(traces),
        "max_length": max_length,
        "overlength_count": overlength_count,
        "overlength_rate": overlength_count / len(traces) if traces else 0.0,
        "percentile_method": "nearest_rank",
        "percentiles": {
            "p50": nearest_rank_percentile(lengths, 50),
            "p90": nearest_rank_percentile(lengths, 90),
            "p95": nearest_rank_percentile(lengths, 95),
            "p99": nearest_rank_percentile(lengths, 99),
            "max": max(lengths) if lengths else None,
        },
        "traces": traces,
    }


def debug_trace_record(
    seed: dict[str, Any],
    decision: TraceDecision,
    *,
    record_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record_json = record_json or {}
    payload = decision.payload or {}
    metadata = record_json.get("metadata") or {}
    messages = payload.get("messages") or record_json.get("messages") or []
    cypher_solution = payload.get("cypher_solution")
    if cypher_solution is None:
        cypher_solution = (
            record_json["cypher_solution"]
            if record_json.get("cypher_solution") is not None
            else metadata.get("cypher_solution")
        )
    final_answer = payload.get("final_answer")
    if final_answer is None:
        final_answer = (
            record_json["final_answer"]
            if record_json.get("final_answer") is not None
            else final_answer_from_messages(messages)
        )

    standard_payload_keys = {
        "messages",
        "cypher",
        "validation",
        "cypher_solution",
        "final_answer",
    }
    debug_record = {
        "uid": record_json.get("uid") or f"{seed.get('uid')}_teacher",
        "seed_uid": seed.get("uid"),
        "scene_id": record_json.get("scene_id") or seed.get("scene_id"),
        "query_type": record_json.get("task_type") or seed.get("query_type", "unknown"),
        "task_family": record_json.get("task_family") or seed.get("task_family", "cypher_qa"),
        "output_type": record_json.get("output_type") or seed.get("output_type", "SLDP"),
        "experiment_profile": record_json.get("experiment_profile")
        or seed.get("experiment_profile"),
        "question": record_json.get("question") or seed.get("question"),
        "cypher": payload.get("cypher") or record_json.get("gold_cypher"),
        "real_solution": seed_real_solution(seed),
        "solution_formats": seed.get("solution_formats", []),
        "answer_validation": seed.get("answer_validation"),
        "grounding": seed.get("grounding"),
        "cypher_solution": cypher_solution,
        "final_answer": final_answer,
        "validation": payload.get("validation") or record_json.get("validation"),
        "messages": messages,
        "tags": seed.get("tags", []),
        "details": {
            key: value for key, value in payload.items() if key not in standard_payload_keys
        },
    }
    if not decision.accepted:
        debug_record["reason"] = decision.reason
    return debug_record


def generate_command(args: argparse.Namespace) -> None:
    configure_model_library_output()
    suppress_neo4j_notification_logs()
    teacher_config = load_yaml(args.teacher_config)
    student_config = load_student_family_config(args.student_config)
    from cypher_3dsg_agent_distillation.training.train import load_tokenizer

    training_max_length = int((student_config.get("training") or {}).get("max_length", 4096))
    if training_max_length <= 0:
        raise ValueError("training.max_length must be greater than zero.")
    tokenizer = load_tokenizer(student_config)
    provider = configured_teacher_provider(teacher_config)
    data_config = load_data_config(args.data_config, dataset_version=args.dataset_version)
    profiles = load_experiment_profiles_from_config(data_config)
    if not profiles:
        raise ValueError("Teacher trace generation requires at least one experiment profile.")
    model = (
        args.teacher_model or teacher_config.get("model", DEFAULT_TEACHER_MODEL)
        if provider == "openrouter"
        else None
    )
    traces_per_seed = int(
        args.traces_per_seed
        if args.traces_per_seed is not None
        else teacher_config.get("traces_per_seed", 1)
    )
    if traces_per_seed < 1:
        raise ValueError("traces_per_seed must be >= 1.")
    if provider == "synthetic" and traces_per_seed != 1:
        raise ValueError("The synthetic teacher requires traces_per_seed=1.")
    if provider == "synthetic" and args.static_only:
        raise ValueError("The synthetic teacher requires Neo4j execution; disable --static-only.")
    effective_teacher_config = dict(teacher_config)
    if model is not None:
        effective_teacher_config["model"] = model
    data_config = with_teacher_run(
        data_config,
        teacher_name=args.teacher or teacher_name_from_config(effective_teacher_config),
        teacher_run=args.teacher_run,
        create_version=True,
    )
    profile_inputs = {}
    for question_set in data_config.get("question_sets") or []:
        profile_path = question_set["experiment_profile_path"]
        profile_inputs[f"profile:{question_set['name']}"] = config_record(
            profile_path, project_root=PROJECT_ROOT
        )
        for key, value in load_yaml(profile_path).items():
            if key.endswith("_path") and value:
                profile_inputs[f"{question_set['name']}:{key}"] = config_record(
                    resolve_project_path(value), project_root=PROJECT_ROOT
                )
    created_at = iso_now()
    atomic_write_yaml(
        {
            "schema_version": 1,
            "artifact_type": "teacher_run",
            "teacher_name": data_config["teacher_name"],
            "version": data_config["teacher_run"],
            "created_at": created_at,
            "updated_at": created_at,
            "status": "running",
            "dataset": {
                "dataset_id": data_config["dataset_id"],
                "scene_ids": [scene["scene_id"] for scene in data_config["scenes"]],
                "version": data_config["dataset_version"],
            },
            "teacher_config": config_record(
                resolve_project_path(args.teacher_config), project_root=PROJECT_ROOT
            ),
            "student_config": config_record(
                student_config["student_config_path"], project_root=PROJECT_ROOT
            ),
            "data_config": config_record(
                data_config["data_config_path"], project_root=PROJECT_ROOT
            ),
            "profile_inputs": profile_inputs,
            "effective": {
                "provider": provider,
                "model": model,
                "arguments": serializable_arguments(args),
            },
            "input_seed_records": file_fingerprint(
                data_config["seed_records_jsonl_path"], relative_to=PROJECT_ROOT
            ),
            "code": git_provenance(PROJECT_ROOT),
        },
        data_config["teacher_metadata_path"],
    )
    client_context = (
        OpenRouterTeacherClient(
            # The model is always resolved for the OpenRouter provider above.
            model=str(model),
            timeout=int(teacher_config.get("timeout", 120)),
            temperature=float(teacher_config.get("temperature", 0.2)),
            max_tokens=int(teacher_config.get("max_tokens", 2048)),
        )
        if provider == "openrouter"
        else nullcontext(None)
    )
    with client_context as client:
        if args.preflight and provider == "openrouter":
            assert client is not None
            client.preflight()

        schema_config = load_yaml(data_config["schema_path"])
        policy = schema_policy(schema_config)
        validator = None
        if not args.static_only:
            validator = Neo4jValidator(
                uri=neo4j_uri_candidates(args.uri),
                username=args.username,
                password=args.password,
                database=args.database,
                timeout=args.neo4j_timeout,
            )

        scene_database_reloads = 0
        try:
            seeds = read_jsonl(data_config["seed_records_jsonl_path"])
            seeds = group_records_by_scene(
                seeds,
                seed=0,
                scene_order=[str(scene["scene_id"]) for scene in data_config["scenes"]],
            )
            scenes_by_id = {str(scene["scene_id"]): scene for scene in data_config["scenes"]}
            current_scene_id: str | None = None
            teacher_records: list[dict[str, Any]] = []
            accepted_debug_records: list[dict[str, Any]] = []
            rejected_debug_records: list[dict[str, Any]] = []
            total_attempts = len(seeds) * traces_per_seed
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task_id = progress.add_task(
                    progress_description(0, len(seeds), 0, 0),
                    total=total_attempts,
                )
                for seed_index, seed in enumerate(seeds, start=1):
                    seed_scene_id = str(seed.get("scene_id") or "")
                    if seed_scene_id not in scenes_by_id:
                        raise ValueError(
                            f"Seed {seed.get('uid')} references unknown scene `{seed_scene_id}`."
                        )
                    if validator is not None and seed_scene_id != current_scene_id:
                        scene = scenes_by_id[seed_scene_id]
                        load_scene(
                            scene_path=scene["scene_path"],
                            uri=neo4j_uri_candidates(args.uri),
                            username=args.username,
                            password=args.password,
                            database=args.database,
                            clear=True,
                        )
                        current_scene_id = seed_scene_id
                        scene_database_reloads += 1
                    profile = experiment_profile_for_seed(seed, profiles)
                    max_tool_calls = configured_max_tool_calls(args, teacher_config, profile)
                    for _ in range(traces_per_seed):
                        progress.update(
                            task_id,
                            description=progress_description(
                                seed_index,
                                len(seeds),
                                len(teacher_records),
                                len(rejected_debug_records),
                            ),
                        )
                        if provider == "openrouter":
                            assert client is not None
                            assert model is not None
                            decision = run_teacher_interaction(
                                seed,
                                profile=profile,
                                tool_interface="openrouter",
                                client=client,
                                teacher_model=model,
                                validator=validator,
                                policy=policy,
                                max_tool_calls=max_tool_calls,
                                static_only=args.static_only,
                            )
                        else:
                            if validator is None:
                                raise RuntimeError("Synthetic teacher requires Neo4j validation.")
                            decision = run_synthetic_teacher_interaction(
                                seed,
                                profile=profile,
                                tool_interface="openrouter",
                                validator=validator,
                                policy=policy,
                                max_tool_calls=max_tool_calls,
                            )
                        if decision.accepted and decision.record is not None:
                            record_json = decision.record.model_dump(
                                by_alias=True,
                                exclude_none=True,
                            )
                            teacher_records.append(record_json)
                            accepted_debug_records.append(
                                debug_trace_record(seed, decision, record_json=record_json)
                            )
                        else:
                            rejected_debug_records.append(debug_trace_record(seed, decision))
                        progress.update(
                            task_id,
                            advance=1,
                            description=progress_description(
                                seed_index,
                                len(seeds),
                                len(teacher_records),
                                len(rejected_debug_records),
                            ),
                        )
        finally:
            if validator is not None:
                validator.close()

    teacher_traces_path = data_config.get("teacher_traces_jsonl_path")
    ensure_parent(teacher_traces_path).write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in teacher_records),
        encoding="utf-8",
    )
    ensure_parent(data_config["accepted_traces_path"]).write_text(
        json.dumps(accepted_debug_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ensure_parent(data_config["rejected_traces_path"]).write_text(
        json.dumps(rejected_debug_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    token_report = trace_token_length_report(
        teacher_records,
        tokenizer,
        max_length=training_max_length,
    )
    token_report.update(
        {
            "schema_version": 1,
            "artifact_type": "teacher_trace_token_lengths",
            "created_at": iso_now(),
            "tokenizer": str(
                student_config["model"].get("tokenizer") or student_config["model"]["base_model"]
            ),
            "student_config": config_record(
                student_config["student_config_path"], project_root=PROJECT_ROOT
            ),
        }
    )
    token_length_report_path = str(
        ensure_parent(data_config["teacher_traces_jsonl_path"]).parent / "token_length_report.yaml"
    )
    atomic_write_yaml(token_report, token_length_report_path)
    teacher_metadata = load_yaml(data_config["teacher_metadata_path"])
    teacher_metadata.update(
        {
            "updated_at": iso_now(),
            "status": "complete",
            "counts": {
                "seeds": len(seeds),
                "attempts": total_attempts,
                "accepted": len(teacher_records),
                "rejected": len(rejected_debug_records),
                "scene_database_reloads": scene_database_reloads,
            },
            "outputs": {
                "teacher_traces": file_fingerprint(
                    data_config["teacher_traces_jsonl_path"], relative_to=PROJECT_ROOT
                ),
                "accepted": file_fingerprint(
                    data_config["accepted_traces_path"], relative_to=PROJECT_ROOT
                ),
                "rejected": file_fingerprint(
                    data_config["rejected_traces_path"], relative_to=PROJECT_ROOT
                ),
                "token_length_report": file_fingerprint(
                    token_length_report_path, relative_to=PROJECT_ROOT
                ),
            },
            "token_lengths": {
                key: value
                for key, value in token_report.items()
                if key not in {"traces", "student_config"}
            },
        }
    )
    atomic_write_yaml(teacher_metadata, data_config["teacher_metadata_path"])
    dataset_metadata = load_yaml(data_config["metadata_path"])
    dataset_metadata["active_teacher_run"] = {
        "teacher_name": data_config["teacher_name"],
        "version": data_config["teacher_run"],
    }
    dataset_metadata["updated_at"] = iso_now()
    atomic_write_yaml(dataset_metadata, data_config["metadata_path"])
    acceptance_rate = len(teacher_records) / total_attempts if total_attempts else 0.0
    token_percentiles = token_report["percentiles"]
    print_summary(
        "Teacher traces generated",
        [
            ("Provider", provider),
            ("Dataset version", data_config["dataset_version"]),
            ("Teacher run", f"{data_config['teacher_name']}/{data_config['teacher_run']}"),
            ("Seeds", len(seeds)),
            ("Accepted", len(teacher_records)),
            ("Rejected", len(rejected_debug_records)),
            ("Acceptance rate", f"{acceptance_rate:.1%}"),
            (
                "Token length p50",
                token_percentiles["p50"] if token_percentiles["p50"] is not None else "n/a",
            ),
            (
                "Token length p90",
                token_percentiles["p90"] if token_percentiles["p90"] is not None else "n/a",
            ),
            (
                "Token length p95",
                token_percentiles["p95"] if token_percentiles["p95"] is not None else "n/a",
            ),
            (
                "Token length p99",
                token_percentiles["p99"] if token_percentiles["p99"] is not None else "n/a",
            ),
            (
                "Token length max",
                token_percentiles["max"] if token_percentiles["max"] is not None else "n/a",
            ),
            (
                f"Traces over max_length={training_max_length}",
                f"{token_report['overlength_count']} ({token_report['overlength_rate']:.1%})",
            ),
            ("Scene database reloads", scene_database_reloads),
            (
                "Max tool calls",
                args.max_tool_calls
                if args.max_tool_calls is not None
                else teacher_config.get("max_tool_calls", "per profile"),
            ),
            ("Training traces", teacher_traces_path),
            ("Accepted debug", data_config["accepted_traces_path"]),
            ("Rejected debug", data_config["rejected_traces_path"]),
            ("Token length report", token_length_report_path),
        ],
    )


def preflight_command(args: argparse.Namespace) -> None:
    config = load_yaml(args.teacher_config)
    provider = configured_teacher_provider(config)
    if provider == "synthetic":
        print_summary(
            "Teacher preflight complete",
            [("Provider", provider), ("Model check", "not required")],
        )
        return
    with OpenRouterTeacherClient(
        model=args.teacher_model or config.get("model", DEFAULT_TEACHER_MODEL)
    ) as client:
        client.preflight()
        print_summary(
            "Teacher preflight complete",
            [("Provider", provider), ("Model", client.model), ("Available", True)],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate profile-based teacher agent traces.")
    subparsers = parser.add_subparsers(required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--teacher-config",
        default="configs/teacher_gpt_oss_120b.yaml",
    )
    preflight_parser.add_argument("--teacher-model", default=None)
    preflight_parser.set_defaults(func=preflight_command)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--teacher-config",
        default="configs/teacher_gpt_oss_120b.yaml",
    )
    generate_parser.add_argument("--data-config", default="configs/data.yaml")
    generate_parser.add_argument(
        "--student-config",
        default="configs/student_qwen3.5_4b.yaml",
        help="Student tokenizer and training.max_length used for the token-length report.",
    )
    generate_parser.add_argument("--dataset-version", default=None)
    generate_parser.add_argument(
        "--teacher",
        default=None,
        help="Optional path-safe teacher folder name.",
    )
    generate_parser.add_argument(
        "--teacher-run",
        default=None,
        help="Optional new run version in YYYY-MM-DD-T-HH-MM-SS-Z format.",
    )
    generate_parser.add_argument("--teacher-model", default=None)
    generate_parser.add_argument("--traces-per-seed", type=int, default=None)
    generate_parser.add_argument("--preflight", action="store_true")
    generate_parser.add_argument("--max-tool-calls", type=int, default=None)
    generate_parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip Neo4j execution; intended only for dry runs.",
    )
    generate_parser.add_argument("--neo4j-timeout", type=float, default=10.0)
    generate_parser.add_argument(
        "--uri",
        default=None,
        help=(
            f"Neo4j URI or comma-separated URI fallback list. "
            f"Default: {DEFAULT_HERACLES_NEO4J_URI}, then localhost fallback."
        ),
    )
    generate_parser.add_argument("--username", default=DEFAULT_HERACLES_NEO4J_USERNAME)
    generate_parser.add_argument("--password", default=DEFAULT_HERACLES_NEO4J_PASSWORD)
    generate_parser.add_argument("--database", default=DEFAULT_HERACLES_NEO4J_DATABASE)
    generate_parser.set_defaults(func=generate_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
