from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

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

from cypher_3dsg_agent_distillation.agent_prompts import (
    build_teacher_interaction_messages,
    run_cypher_tool_schema,
)
from cypher_3dsg_agent_distillation.agent_records import (
    DEFAULT_AGENT_TOOL,
    DEFAULT_TEACHER_MODEL,
    WRITE_REFUSAL_ANSWER,
    AgentTraceRecord,
    split_group_for_record,
)
from cypher_3dsg_agent_distillation.config import ensure_parent, load_data_config, load_yaml
from cypher_3dsg_agent_distillation.evaluation import (
    ResultShapeError,
    normalize_answer,
    rows_to_solution_text,
)
from cypher_3dsg_agent_distillation.experiment_profiles import (
    ExperimentProfile,
    ToolInterface,
    load_experiment_profile_from_config,
    render_profile_messages,
)
from cypher_3dsg_agent_distillation.heracles_validation import answer_matches_with_heracles
from cypher_3dsg_agent_distillation.io import read_jsonl
from cypher_3dsg_agent_distillation.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_URI,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    Neo4jValidator,
    is_safe_refusal,
    neo4j_uri_candidates,
    schema_policy,
    static_checks,
)

DEFAULT_MAX_TOOL_CALLS = 3
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
        normalized["tool_calls"] = [
            normalize_tool_call_arguments(tool_call)
            for tool_call in normalized.get("tool_calls", [])
        ]
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
        messages=candidate["messages"],
        gold_cypher=cypher,
        expected_solution=real_solution,
        real_solution=real_solution,
        final_answer=final_answer_from_messages(candidate["messages"]),
        split_group=split_group_for_record(
            seed.get("scene_id"),
            seed.get("query_type", "unknown"),
            cypher,
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
    if static_only:
        metadata["static_only"] = True
        metadata["incomplete_conversation"] = True
    return AgentTraceRecord(
        uid=f"{seed['uid']}_teacher",
        scene_id=seed.get("scene_id"),
        question=str(seed["question"]),
        query_type=seed.get("query_type", "unknown"),
        messages=messages,
        gold_cypher=cypher,
        expected_solution=real_solution,
        real_solution=real_solution,
        cypher_solution=cypher_solution,
        final_answer=final_answer_from_messages(messages),
        split_group=split_group_for_record(
            seed.get("scene_id"),
            seed.get("query_type", "unknown"),
            seed_cypher,
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
    record.cypher_solution = cypher_solution
    valid_cypher_format, cypher_matches_real = answer_matches_with_heracles(
        cypher_solution,
        str(real_solution),
        output_type,
    )
    record.validation.cypher_solution_match = cypher_matches_real
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
    )
    valid_consistency_format, final_matches_cypher = answer_matches_with_heracles(
        inner_answer,
        cypher_solution,
        output_type,
    )
    record.validation.final_answer_match = final_matches_real
    record.validation.final_answer_cypher_match = final_matches_cypher
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
    schema: str | None = None,
    profile: ExperimentProfile | None = None,
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

    if profile is not None:
        messages = render_profile_messages(profile, seed, tool_interface=tool_interface)
    elif schema is not None:
        messages = build_teacher_interaction_messages(schema, seed["question"])
    else:
        raise ValueError("Either schema or profile must be provided.")

    task_family = str(seed.get("task_family") or (profile.task_family if profile else "cypher_qa"))
    output_type = str(seed.get("output_type") or (profile.output_type if profile else "SLDP"))
    seed_cypher = str(seed.get("cypher") or seed.get("gold_cypher") or "").strip()
    real_solution = seed_real_solution(seed)
    expected_refusal = is_safe_refusal(seed_cypher)
    tool_call_count = 0
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
        valid_final_format, final_matches = answer_matches_with_heracles(
            inner_answer,
            str(real_solution),
            output_type,
        )
        checked_validation = validation_with_solution_checks(
            final_validation or {},
            cypher_solution_match=cypher_solution_match,
            final_answer_match=final_matches,
            final_answer_cypher_match=final_answer_cypher_match,
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
            validation=validation_with_solution_checks(
                {
                    "static_valid": True,
                    "executable": True,
                    "error": None,
                },
                cypher_solution_match=cypher_solution_match,
                final_answer_match=final_matches,
                final_answer_cypher_match=final_answer_cypher_match,
            ),
            max_tool_calls=max_tool_calls,
            tool_call_count=tool_call_count,
            successful_tool_call_index=successful_tool_call_index,
            cypher_solution=cypher_solution,
        )
        return TraceDecision(True, record, "accepted", {"messages": messages})


def configured_max_tool_calls(
    args: argparse.Namespace,
    teacher_config: dict[str, Any],
    profile: ExperimentProfile | None = None,
) -> int:
    value = (
        args.max_tool_calls
        if args.max_tool_calls is not None
        else teacher_config.get(
            "max_tool_calls",
            profile.max_tool_calls if profile is not None else DEFAULT_MAX_TOOL_CALLS,
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
        cypher_solution = record_json.get("cypher_solution") or metadata.get("cypher_solution")
    final_answer = payload.get("final_answer")
    if final_answer is None:
        final_answer = record_json.get("final_answer") or final_answer_from_messages(messages)

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
        "question": record_json.get("question") or seed.get("question"),
        "cypher": payload.get("cypher") or record_json.get("gold_cypher"),
        "real_solution": seed_real_solution(seed),
        "cypher_solution": cypher_solution,
        "final_answer": final_answer,
        "validation": payload.get("validation") or record_json.get("validation"),
        "messages": messages,
        "details": {
            key: value
            for key, value in payload.items()
            if key not in standard_payload_keys
        },
    }
    if not decision.accepted:
        debug_record["reason"] = decision.reason
    return debug_record


def generate_command(args: argparse.Namespace) -> None:
    teacher_config = load_yaml(args.teacher_config)
    data_config = load_data_config(args.data_config)
    profile = load_experiment_profile_from_config(data_config)
    model = args.teacher_model or teacher_config.get("model", DEFAULT_TEACHER_MODEL)
    max_tool_calls = configured_max_tool_calls(args, teacher_config, profile)
    traces_per_seed = int(
        args.traces_per_seed
        if args.traces_per_seed is not None
        else teacher_config.get("traces_per_seed", 1)
    )
    if traces_per_seed < 1:
        raise ValueError("traces_per_seed must be >= 1.")

    with OpenRouterTeacherClient(
        model=model,
        timeout=int(teacher_config.get("timeout", 120)),
        temperature=float(teacher_config.get("temperature", 0.2)),
        max_tokens=int(teacher_config.get("max_tokens", 2048)),
    ) as client:
        if args.preflight:
            client.preflight()

        schema_config = load_yaml(data_config["schema_path"])
        schema = schema_config["interface_description"]
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

        try:
            seeds = read_jsonl(data_config["seed_records_jsonl_path"])
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
                        decision = run_teacher_interaction(
                            seed,
                            schema=schema,
                            profile=profile,
                            tool_interface="openrouter",
                            client=client,
                            teacher_model=model,
                            validator=validator,
                            policy=policy,
                            max_tool_calls=max_tool_calls,
                            static_only=args.static_only,
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


def preflight_command(args: argparse.Namespace) -> None:
    config = load_yaml(args.teacher_config)
    with OpenRouterTeacherClient(
        model=args.teacher_model or config.get("model", DEFAULT_TEACHER_MODEL)
    ) as client:
        client.preflight()
        print(f"OpenRouter model `{client.model}` is available.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OpenRouter teacher agent traces.")
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
