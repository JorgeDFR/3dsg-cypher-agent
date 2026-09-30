from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.agents.evaluation_cache import (
    completed_evaluation,
    evaluation_completed,
    evaluation_identity,
)
from cypher_3dsg_agent_distillation.agents.raw_generation_runtimes import (
    DEFAULT_OLLAMA_CONTEXT_LENGTH,
    RAW_GENERATION_RUNTIMES,
    TRANSFORMERS_RUNTIME,
    GenerationResult,
    OllamaRawGenerationRuntime,
    RawGenerationRuntime,
    TransformersRawGenerationRuntime,
    generation_artifacts_present,
    jsonable,
)
from cypher_3dsg_agent_distillation.agents.student_eval_loss import (
    combine_eval_loss_summaries,
    evaluate_eval_loss,
    evaluate_eval_loss_splits,
)
from cypher_3dsg_agent_distillation.agents.student_model import (
    default_adapter_dir,
    tokenizer_source,
)
from cypher_3dsg_agent_distillation.core.cli_output import print_breakdown, print_summary
from cypher_3dsg_agent_distillation.core.config import (
    ensure_parent,
    load_data_config,
    load_student_config,
    load_student_family_config,
    load_yaml,
    model_reports_dir,
    ollama_export_metadata,
)
from cypher_3dsg_agent_distillation.core.model_runtime_output import (
    configure_model_library_output,
)
from cypher_3dsg_agent_distillation.core.scene_ordering import group_records_by_scene
from cypher_3dsg_agent_distillation.core.versioning import (
    confirm_dataset_lineage,
    read_metadata,
)

EVAL_LOSS_MODE = "eval-loss"
RAW_GENERATION_MODE = "raw-generation"
SUPPORTED_SPLITS = ("train", "eval", "test", "challenge")
ALL_SPLITS = "all"
SPLIT_CHOICES = (*SUPPORTED_SPLITS, ALL_SPLITS)
FINAL_ANSWER_CHECK = "final_answer_from_trace"
TOOL_CALL_CHECK = "tool_call_from_prompt"
END_TO_END_CHECK = "end_to_end_final_answer"
DEFAULT_MAX_QUERY_RETRIES = 1
DEFAULT_E2E_MAX_TOOL_ROWS = 128
DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS = 16384
DEFAULT_E2E_MAX_QUERY_CHARS = 4096
E2E_REPAIR_MESSAGE = (
    "The query was not executed successfully. Return one corrected run_cypher_query tool call."
)


@dataclass(frozen=True)
class ToolCallParseResult:
    tool_call_generated: bool
    tool_calls: list[dict[str, Any]]
    error: str | None = None


def report_output_path(
    *,
    data_config: dict[str, Any],
    split: str,
    mode: str,
    output: str | None,
    baseline: bool = False,
    runtime: str = TRANSFORMERS_RUNTIME,
    runtime_model: str | None = None,
    reports_directory: str | Path | None = None,
) -> Path:
    if output:
        return Path(output)
    model_name = "baseline" if baseline else "student"
    runtime_suffix = ""
    if runtime != TRANSFORMERS_RUNTIME:
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", runtime_model or "model")
        runtime_suffix = f".{runtime}.{safe_model}"
    directory = (
        Path(reports_directory)
        if reports_directory is not None
        else Path(data_config["test_report_path"]).parent
    )
    if split == ALL_SPLITS:
        return directory / f"{model_name}{runtime_suffix}.{mode}.report.json"
    return directory / f"{split}.{model_name}{runtime_suffix}.{mode}.report.json"


QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\s]+)>\s*"
    r"(?P<body>.*?)</function>\s*</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)
QWEN_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
    flags=re.IGNORECASE | re.DOTALL,
)
CUSTOM_TOOL_CALL_RE = re.compile(
    r"<tool>\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<arguments>.*?)\)\s*</tool>",
    flags=re.IGNORECASE | re.DOTALL,
)
QUOTED_ARGUMENT_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<literal>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")",
    flags=re.DOTALL,
)
CODE_BLOCK_RE = re.compile(
    r"```(?:json|python)?\s*(?P<body>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def record_expected_solution(record: dict[str, Any]) -> Any:
    for key in ("expected_solution", "real_solution", "solution"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def record_gold_cypher(record: dict[str, Any]) -> str:
    return str(record.get("gold_cypher") or record.get("cypher") or "").strip()


def record_output_type(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(record.get("output_type") or metadata.get("output_type") or "SLDP")


def record_task_family(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(record.get("task_family") or metadata.get("task_family") or "cypher_qa")


def record_query_type(record: dict[str, Any]) -> str:
    return str(record.get("query_type") or record.get("task_type") or "")


def record_for_solution_conversion(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("query_type"):
        return record
    return {**record, "query_type": record_query_type(record)}


def record_is_refusal(record: dict[str, Any]) -> bool:
    from cypher_3dsg_agent_distillation.agents.agent_records import (
        WRITE_REFUSAL_ANSWER,
        normalize_cypher,
    )

    gold_cypher = normalize_cypher(record_gold_cypher(record))
    expected_solution = record_expected_solution(record)
    return (
        gold_cypher
        in {
            WRITE_REFUSAL_ANSWER,
            f"// {WRITE_REFUSAL_ANSWER}",
        }
        or str(expected_solution or "").strip() == WRITE_REFUSAL_ANSWER
    )


def initial_prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    from cypher_3dsg_agent_distillation.training.train import normalize_chat_messages

    messages = normalize_chat_messages(record.get("messages", []))
    repair_context_message_count = (record.get("metadata") or {}).get(
        "repair_context_message_count"
    )
    if repair_context_message_count is not None:
        context_count = int(repair_context_message_count)
        if not 0 <= context_count <= len(messages):
            raise ValueError(
                f"Record {record.get('uid')} has invalid repair_context_message_count="
                f"{context_count}."
            )
        return messages[:context_count]
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            return messages[:index]
    return messages


def final_answer_prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    from cypher_3dsg_agent_distillation.training.train import normalize_chat_messages

    messages = normalize_chat_messages(record.get("messages", []))
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            message.get("role") == "assistant"
            and message.get("content")
            and not message.get("tool_calls")
        ):
            return messages[:index]
    return messages


def generation_report_fields(result: GenerationResult) -> dict[str, int | None]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "max_new_tokens": result.max_new_tokens,
        "token_budget": result.token_budget,
    }


def add_raw_generation_debug_fields(
    score: dict[str, Any],
    generation: GenerationResult,
) -> None:
    score["raw_generated_text"] = generation.raw_text
    score["raw_generated_text_had_artifacts"] = generation_artifacts_present(generation.raw_text)


def generation_debug_report(
    generation: GenerationResult | None,
) -> dict[str, Any]:
    return {
        "rendered_prompt": generation.prompt if generation is not None else None,
        "raw_completion": generation.raw_text if generation is not None else None,
        "structured_tool_calls": (list(generation.tool_calls) if generation is not None else []),
        "provider_metadata": (generation.provider_metadata if generation is not None else {}),
    }


def expected_final_answer(record: dict[str, Any]) -> str | None:
    solution = record_expected_solution(record)
    return f"<answer>{solution}</answer>" if solution is not None else None


def normalized_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def qwen_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in QWEN_TOOL_CALL_RE.finditer(text):
        arguments = {
            parameter.group("name").strip(): parameter.group("value").strip()
            for parameter in QWEN_PARAMETER_RE.finditer(match.group("body"))
        }
        calls.append(normalized_tool_call(match.group("name").strip(), arguments))
    return calls


def normalize_structured_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
        normalize_tool_call_arguments,
    )

    if "function" not in tool_call and "name" in tool_call:
        tool_call = {
            **{key: value for key, value in tool_call.items() if key not in {"name", "arguments"}},
            "type": "function",
            "function": {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments", {}),
            },
        }
    return normalize_tool_call_arguments(tool_call)


def structured_tool_calls_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "tool_calls" in payload:
            raw_calls = payload.get("tool_calls") or []
        elif "function" in payload or "name" in payload:
            raw_calls = [payload]
        else:
            return []
    elif isinstance(payload, list):
        raw_calls = payload
    else:
        return []

    return [
        normalize_structured_tool_call(tool_call)
        for tool_call in raw_calls
        if isinstance(tool_call, dict)
    ]


def parse_json_like_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def json_candidate_texts(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates.extend(match.group("body").strip() for match in CODE_BLOCK_RE.finditer(text))
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        candidates.append(text[first_bracket : last_bracket + 1])
    return [candidate for candidate in candidates if candidate]


def json_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    errors = []
    for candidate in json_candidate_texts(text):
        try:
            tool_calls = structured_tool_calls_from_payload(parse_json_like_payload(candidate))
        except (SyntaxError, ValueError, TypeError) as exc:
            errors.append(str(exc))
            continue
        if tool_calls:
            return tool_calls
    if "tool_calls" in text:
        raise ValueError("; ".join(errors) if errors else "malformed JSON tool call")
    return []


def custom_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in CUSTOM_TOOL_CALL_RE.finditer(text):
        arguments = {}
        for argument in QUOTED_ARGUMENT_RE.finditer(match.group("arguments")):
            try:
                value = ast.literal_eval(argument.group("literal"))
            except (SyntaxError, ValueError):
                value = argument.group("literal").strip("\"'")
            arguments[argument.group("name")] = value
        calls.append(normalized_tool_call(match.group("name").strip(), arguments))
    return calls


def generated_tool_marker_present(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in ("<tool_call", "<tool>", "tool_calls", "run_cypher_query")
    )


def parse_generated_tool_calls(
    text: str,
    structured_tool_calls: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> ToolCallParseResult:
    try:
        if structured_tool_calls:
            tool_calls = [
                normalize_structured_tool_call(tool_call) for tool_call in structured_tool_calls
            ]
            return ToolCallParseResult(True, tool_calls)

        tool_calls = qwen_tool_calls_from_text(text)
        if tool_calls:
            return ToolCallParseResult(True, tool_calls)

        tool_calls = json_tool_calls_from_text(text)
        if tool_calls:
            return ToolCallParseResult(True, tool_calls)

        tool_calls = custom_tool_calls_from_text(text)
        if tool_calls:
            return ToolCallParseResult(True, tool_calls)
    except ValueError as exc:
        return ToolCallParseResult(True, [], str(exc))

    if generated_tool_marker_present(text):
        return ToolCallParseResult(True, [], "tool-call markers were present but unparseable")
    return ToolCallParseResult(False, [])


def score_final_answer_generation(
    record: dict[str, Any],
    generated_text: str,
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import answer_inner_text
    from cypher_3dsg_agent_distillation.graph.heracles_validation import (
        answer_matches_with_heracles,
    )

    expected_solution = record_expected_solution(record)
    answer = answer_inner_text(generated_text)
    format_valid = False
    answer_match = False
    if answer is not None and expected_solution is not None:
        format_valid, answer_match = answer_matches_with_heracles(
            answer,
            str(expected_solution),
            record_output_type(record),
            answer_validation=record.get("answer_validation"),
            solution_formats=record.get("solution_formats") or record.get("solution_format"),
        )

    status = "pass"
    if expected_solution is None:
        status = "missing_expected_solution"
    elif not generated_text.strip():
        status = "missing_generation"
    elif answer is None:
        status = "missing_answer_tags"
    elif not format_valid:
        status = "invalid_answer_format"
    elif not answer_match:
        status = "final_answer_mismatch"

    return {
        "status": status,
        "generated_text": generated_text,
        "generated_answer": answer,
        "expected_solution": expected_solution,
        "final_answer_generated": bool(generated_text.strip()),
        "final_answer_has_tags": answer is not None,
        "final_answer_format_valid": format_valid,
        "final_answer_domain_valid": format_valid,
        "final_answer_match": answer_match,
    }


def final_answer_generation_error(record: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "status": "generation_error",
        "generation_error": f"{type(exc).__name__}: {exc}",
        "generated_text": "",
        "generated_answer": None,
        "expected_solution": record_expected_solution(record),
        "final_answer_generated": False,
        "final_answer_has_tags": False,
        "final_answer_format_valid": False,
        "final_answer_match": False,
    }


def bounded_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    marker = "...[truncated]"
    return f"{text[: max(0, max_chars - len(marker))]}{marker}"


def end_to_end_error_content(
    error_type: str,
    issues: list[Any],
    *,
    max_chars: int,
) -> str:
    """Build small retry feedback without copying arbitrary database errors."""
    payload = {
        "ok": False,
        "error_type": error_type,
        "issues": [bounded_text(issue, 512) for issue in issues[:3]],
        "message": E2E_REPAIR_MESSAGE,
    }
    content = json.dumps(payload, ensure_ascii=False, default=str)
    if len(content) <= max_chars:
        return content
    # max_chars is validated to be >= 256, so this compact payload remains valid
    # JSON instead of slicing JSON at an arbitrary byte/character boundary.
    return json.dumps(
        {
            "ok": False,
            "error_type": bounded_text(error_type, 64),
            "message": "Return one corrected run_cypher_query tool call.",
        },
        ensure_ascii=False,
    )


def bounded_tool_success_content(
    rows: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str | None:
    """Serialize tool rows incrementally and stop before an oversized copy is built."""
    encoder = json.JSONEncoder(ensure_ascii=False, default=str)
    chunks = []
    length = 0
    for chunk in encoder.iterencode({"ok": True, "rows": rows}):
        length += len(chunk)
        if length > max_chars:
            return None
        chunks.append(chunk)
    return "".join(chunks)


def end_to_end_skipped_result(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": reason,
        "final_answer_match": False,
        "generated_text": None,
        "generated_answer": None,
        "query_attempts": 0,
        "retries_used": 0,
        "attempts": [],
    }


def end_to_end_failure_result(
    record: dict[str, Any],
    *,
    status: str,
    query_attempts: int,
    max_query_retries: int,
    attempts: list[dict[str, Any]],
    last_failure_reason: str | None = None,
    generation_error: str | None = None,
) -> dict[str, Any]:
    result = {
        "status": status,
        "final_answer_match": False,
        "generated_text": None,
        "generated_answer": None,
        "expected_solution": record_expected_solution(record),
        "query_attempts": query_attempts,
        "retries_used": max(0, query_attempts - 1),
        "max_query_retries": max_query_retries,
        "attempts": attempts,
    }
    if last_failure_reason is not None:
        result["last_failure_reason"] = last_failure_reason
    if generation_error is not None:
        result["generation_error"] = generation_error
    return result


def append_end_to_end_retry_feedback(
    messages: list[dict[str, Any]],
    *,
    generated_text: str,
    tool_call: dict[str, Any] | None,
    feedback: str,
    max_query_chars: int,
) -> None:
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
        build_tool_result_message,
    )

    if tool_call is not None:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call],
            }
        )
        tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        messages.append(build_tool_result_message(tool_call_id, feedback))
        return

    # An unparseable or wrongly named call cannot be followed by a valid tool-role
    # message. Keep only a bounded preview and provide corrective user feedback.
    messages.append(
        {
            "role": "assistant",
            "content": bounded_text(generated_text, max_query_chars),
        }
    )
    messages.append({"role": "user", "content": feedback})


def evaluate_end_to_end_record(
    record: dict[str, Any],
    *,
    runtime: RawGenerationRuntime,
    validator,
    policy: dict[str, Any] | None,
    tool_schemas: list[dict[str, Any]],
    max_new_tokens: int | None,
    enable_thinking: bool,
    max_query_retries: int = DEFAULT_MAX_QUERY_RETRIES,
    max_tool_rows: int = DEFAULT_E2E_MAX_TOOL_ROWS,
    max_tool_response_chars: int = DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS,
    max_query_chars: int = DEFAULT_E2E_MAX_QUERY_CHARS,
    progress=None,
    progress_task_id=None,
    progress_description: str | None = None,
    generation_debug_turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one bounded query/tool/final-answer interaction.

    Query failures receive compact feedback and may be retried. The only quality
    metric produced by this interaction is ``final_answer_match``; query fields are
    operational diagnostics for understanding retries and resource guards.
    """
    from cypher_3dsg_agent_distillation.agents.agent_records import DEFAULT_AGENT_TOOL
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
        static_checks_with_explain,
    )

    if max_query_retries < 0:
        raise ValueError("max_query_retries must be zero or greater.")
    if max_tool_rows <= 0:
        raise ValueError("max_tool_rows must be greater than zero.")
    if max_tool_response_chars < 256:
        raise ValueError("max_tool_response_chars must be at least 256.")
    if max_query_chars < 256:
        raise ValueError("max_query_chars must be at least 256.")

    base_messages = initial_prompt_messages(record)
    messages = list(base_messages)
    attempts: list[dict[str, Any]] = []
    query_attempts = 0
    max_query_attempts = max_query_retries + 1

    while True:
        try:
            generation = runtime.generate(
                messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                tools=tool_schemas,
                progress=progress,
                progress_task_id=progress_task_id,
                progress_description=progress_description,
            )
        except Exception as exc:
            return end_to_end_failure_result(
                record,
                status="generation_error",
                query_attempts=query_attempts,
                max_query_retries=max_query_retries,
                attempts=attempts,
                generation_error=bounded_text(f"{type(exc).__name__}: {exc}", 1024),
            )

        generated_text = generation.text
        parsed = parse_generated_tool_calls(generated_text, generation.tool_calls)
        if generation_debug_turns is not None:
            generation_debug_turns.append(
                {
                    "turn": len(generation_debug_turns) + 1,
                    **generation_debug_report(generation),
                }
            )
        attempt = {
            "turn": len(attempts) + 1,
            "completion_preview": bounded_text(generated_text, 1024),
            **generation_report_fields(generation),
        }
        attempts.append(attempt)

        if not parsed.tool_calls and not parsed.tool_call_generated:
            score = score_final_answer_generation(record, generated_text)
            attempt["status"] = score["status"]
            return {
                **score,
                "query_attempts": query_attempts,
                "retries_used": max(0, query_attempts - 1),
                "max_query_retries": max_query_retries,
                "attempts": attempts,
            }

        if query_attempts >= max_query_attempts:
            attempt["status"] = "query_retry_limit_reached"
            return end_to_end_failure_result(
                record,
                status="query_retries_exhausted",
                query_attempts=query_attempts,
                max_query_retries=max_query_retries,
                attempts=attempts,
                last_failure_reason="query_retry_limit_reached",
            )

        query_attempts += 1
        retry_tool_call = None
        failure_reason = None
        failure_issues: list[Any] = []
        validation = None
        tool_content = None

        if not parsed.tool_calls:
            failure_reason = "tool_call_parse_failed"
            failure_issues = [parsed.error or "The tool call could not be parsed."]
        elif len(parsed.tool_calls) != 1:
            failure_reason = "multiple_tool_calls"
            failure_issues = ["Return exactly one run_cypher_query tool call."]
        else:
            tool_call = parsed.tool_calls[0]
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or {}
            tool_name = function.get("name")
            cypher = str(arguments.get("cypher_string") or "").strip()
            attempt["tool_name"] = tool_name
            attempt["generated_cypher"] = bounded_text(cypher, max_query_chars)

            if tool_name != DEFAULT_AGENT_TOOL:
                failure_reason = "wrong_tool_name"
                failure_issues = [f"Expected tool {DEFAULT_AGENT_TOOL}."]
            else:
                retry_tool_call = tool_call
                if not cypher:
                    failure_reason = "missing_cypher_argument"
                    failure_issues = ["cypher_string must be a non-empty query."]
                elif len(cypher) > max_query_chars:
                    retry_tool_call = None
                    failure_reason = "query_too_large"
                    failure_issues = [
                        f"Query exceeded the {max_query_chars}-character safety limit."
                    ]
                else:
                    static_issues, explain_validation = static_checks_with_explain(
                        cypher,
                        allow_writes=not bool(record.get("read_only", True)),
                        policy=policy,
                        validator=validator,
                    )
                    if explain_validation is not None:
                        attempt["explain_validation"] = jsonable(explain_validation)
                    if static_issues:
                        failure_reason = "static_error"
                        failure_issues = static_issues
                    else:
                        try:
                            validation = validator.validate(
                                cypher,
                                execute=True,
                                max_rows=max_tool_rows,
                                compute_result_hash=False,
                            )
                        except Exception as exc:
                            failure_reason = "execution_error"
                            failure_issues = [f"{type(exc).__name__}: {exc}"]
                        else:
                            attempt["executable"] = bool(validation.get("executable"))
                            attempt["elapsed_ms"] = validation.get("elapsed_ms")
                            rows = validation.get("rows") or []
                            attempt["row_count"] = len(rows)
                            attempt["rows_truncated"] = bool(validation.get("rows_truncated"))
                            if not validation.get("executable"):
                                failure_reason = "execution_error"
                                failure_issues = [
                                    validation.get("error") or "Query execution failed."
                                ]
                            elif validation.get("rows_truncated"):
                                failure_reason = "result_too_large"
                                failure_issues = [
                                    f"Result exceeded the {max_tool_rows}-row safety limit."
                                ]
                            else:
                                candidate_content = bounded_tool_success_content(
                                    rows,
                                    max_chars=max_tool_response_chars,
                                )
                                if candidate_content is None:
                                    failure_reason = "result_too_large"
                                    failure_issues = [
                                        "Serialized result exceeded the "
                                        f"{max_tool_response_chars}-character safety limit."
                                    ]
                                else:
                                    attempt["tool_response_chars"] = len(candidate_content)
                                    tool_content = candidate_content

        if tool_content is not None and retry_tool_call is not None:
            attempt["status"] = "tool_success"
            # Keep the initial prompt plus only the most recent tool exchange. Old
            # failed/incorrect outputs must not grow the next GPU input each retry.
            messages = list(base_messages)
            append_end_to_end_retry_feedback(
                messages,
                generated_text=generated_text,
                tool_call=retry_tool_call,
                feedback=tool_content,
                max_query_chars=max_query_chars,
            )
            del generation, validation, tool_content
            continue

        failure_reason = failure_reason or "query_failed"
        attempt["status"] = failure_reason
        feedback = end_to_end_error_content(
            failure_reason,
            failure_issues,
            max_chars=max_tool_response_chars,
        )
        attempt["tool_response_chars"] = len(feedback)
        if query_attempts >= max_query_attempts:
            return end_to_end_failure_result(
                record,
                status="query_retries_exhausted",
                query_attempts=query_attempts,
                max_query_retries=max_query_retries,
                attempts=attempts,
                last_failure_reason=failure_reason,
            )
        messages = list(base_messages)
        append_end_to_end_retry_feedback(
            messages,
            generated_text=generated_text,
            tool_call=retry_tool_call,
            feedback=feedback,
            max_query_chars=max_query_chars,
        )
        del generation, validation, feedback


def skipped_tool_call_result(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": reason,
        "generated_text": None,
        "tool_call_generated": False,
        "tool_call_parse_success": False,
        "correct_tool_name": False,
        "cypher_argument_present": False,
        "cypher_exact_match": False,
        "static_valid": False,
        "executable": False,
        "cypher_solution_match": False,
        "tool_result_match": False,
        "result_match_without_exact_cypher_match": False,
        "potential_hidden_false_positive": False,
        "hidden_false_positive_reasons": [],
    }


def tool_call_generation_error(record: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "status": "generation_error",
        "generation_error": f"{type(exc).__name__}: {exc}",
        "generated_text": "",
        "tool_call_generated": False,
        "tool_call_parse_success": False,
        "correct_tool_name": False,
        "cypher_argument_present": False,
        "cypher_exact_match": False,
        "static_valid": False,
        "executable": False,
        "gold_cypher": record_gold_cypher(record),
        "expected_solution": record_expected_solution(record),
        "cypher_solution_match": False,
        "tool_result_match": False,
        "result_match_without_exact_cypher_match": False,
        "potential_hidden_false_positive": False,
        "hidden_false_positive_reasons": [],
    }


def score_refusal_tool_generation(
    record: dict[str, Any],
    generated_text: str,
    parsed: ToolCallParseResult,
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.agent_records import WRITE_REFUSAL_ANSWER
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import answer_inner_text

    answer = answer_inner_text(generated_text)
    refusal_match = str(answer or "").strip() == WRITE_REFUSAL_ANSWER
    status = "pass"
    if parsed.tool_call_generated:
        status = "unexpected_tool_call_for_refusal"
    elif not refusal_match:
        status = "missing_refusal"

    tool_calls = parsed.tool_calls
    function = (tool_calls[0].get("function") or {}) if tool_calls else {}
    arguments = function.get("arguments") or {}
    generated_cypher = str(arguments.get("cypher_string") or "").strip()
    return {
        "status": status,
        "generated_text": generated_text,
        "generated_answer": answer,
        "generated_cypher": generated_cypher or None,
        "tool_calls": tool_calls,
        "expected_solution": record_expected_solution(record),
        "tool_call_generated": parsed.tool_call_generated,
        "tool_call_parse_success": bool(parsed.tool_calls),
        "correct_tool_name": not parsed.tool_call_generated,
        "cypher_argument_present": False,
        "cypher_exact_match": False,
        "static_valid": refusal_match and not parsed.tool_call_generated,
        "executable": refusal_match and not parsed.tool_call_generated,
        "cypher_solution": WRITE_REFUSAL_ANSWER if refusal_match else None,
        "cypher_solution_match": refusal_match,
        "tool_result_match": refusal_match,
        "result_match_without_exact_cypher_match": False,
        "potential_hidden_false_positive": False,
        "hidden_false_positive_reasons": [],
    }


def score_tool_call_generation(
    record: dict[str, Any],
    generated_text: str,
    *,
    validator,
    policy: dict[str, Any] | None,
    max_rows: int | None = None,
    max_result_chars: int | None = None,
    structured_tool_calls: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.agent_records import (
        DEFAULT_AGENT_TOOL,
        normalize_cypher,
    )
    from cypher_3dsg_agent_distillation.cypher.cypher_match_diagnostics import (
        result_match_diagnostics,
    )
    from cypher_3dsg_agent_distillation.cypher.evaluation import (
        ResultShapeError,
        rows_to_solution_text,
    )
    from cypher_3dsg_agent_distillation.cypher.pddl_goals import validate_grounding_rows
    from cypher_3dsg_agent_distillation.graph.heracles_validation import (
        answer_matches_with_heracles,
    )
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
        result_hash,
        static_checks_with_explain,
    )

    parsed = parse_generated_tool_calls(generated_text, structured_tool_calls)
    base = {
        "generated_text": generated_text,
        "tool_call_generated": parsed.tool_call_generated,
        "tool_call_parse_success": bool(parsed.tool_calls),
        "parse_error": parsed.error,
        "correct_tool_name": False,
        "cypher_argument_present": False,
        "cypher_exact_match": False,
        "static_valid": False,
        "static_issues": [],
        "explain_validation": None,
        "executable": False,
        "syntax_error": False,
        "timeout": False,
        "elapsed_ms": None,
        "row_count": 0,
        "result_hash": None,
        "execution_error": None,
        "result_shape_error": None,
        "generated_cypher": None,
        "gold_cypher": record_gold_cypher(record),
        "cypher_solution": None,
        "expected_solution": record_expected_solution(record),
        "cypher_solution_format_valid": False,
        "cypher_solution_match": False,
        "tool_result_match": False,
        "result_match_without_exact_cypher_match": False,
        "potential_hidden_false_positive": False,
        "hidden_false_positive_reasons": [],
    }

    if record_is_refusal(record):
        return score_refusal_tool_generation(record, generated_text, parsed)
    if not parsed.tool_calls:
        status = "tool_call_parse_failed" if parsed.tool_call_generated else "missing_required_tool"
        return {**base, "status": status}
    if len(parsed.tool_calls) > 1:
        return {**base, "status": "multiple_tool_calls", "tool_calls": parsed.tool_calls}

    tool_call = parsed.tool_calls[0]
    function = tool_call.get("function") or {}
    arguments = function.get("arguments") or {}
    tool_name = function.get("name")
    cypher = str(arguments.get("cypher_string") or "").strip()
    correct_tool_name = tool_name == DEFAULT_AGENT_TOOL
    cypher_argument_present = bool(cypher)
    base.update(
        {
            "tool_calls": [tool_call],
            "tool_name": tool_name,
            "correct_tool_name": correct_tool_name,
            "cypher_argument_present": cypher_argument_present,
            "generated_cypher": cypher or None,
            "cypher_exact_match": normalize_cypher(cypher)
            == normalize_cypher(record_gold_cypher(record)),
        }
    )
    if not correct_tool_name:
        return {**base, "status": "wrong_tool_name"}
    if not cypher_argument_present:
        return {**base, "status": "missing_cypher_argument"}

    static_issues, explain_validation = static_checks_with_explain(
        cypher,
        allow_writes=not bool(record.get("read_only", True)),
        policy=policy,
        validator=validator,
    )
    base.update(
        {
            "static_valid": not static_issues,
            "static_issues": static_issues,
            "explain_validation": jsonable(explain_validation),
        }
    )
    if static_issues:
        if explain_validation is not None:
            base["syntax_error"] = bool(explain_validation.get("syntax_error"))
            base["execution_error"] = explain_validation.get("error")
        return {**base, "status": "static_error"}

    try:
        validation = (
            validator.validate(
                cypher,
                execute=True,
                max_rows=max_rows,
                compute_result_hash=False,
            )
            if max_rows is not None
            else validator.validate(cypher, execute=True)
        )
    except Exception as exc:
        return {
            **base,
            "status": "execution_error",
            "execution_error": f"{type(exc).__name__}: {exc}",
        }
    rows = validation.get("rows", [])
    result_too_large = bool(validation.get("rows_truncated"))
    if (
        not result_too_large
        and max_result_chars is not None
        and bounded_tool_success_content(rows, max_chars=max_result_chars) is None
    ):
        result_too_large = True
    report_validation = (
        {**validation, "rows": [], "rows_omitted": True} if result_too_large else validation
    )
    base.update(
        {
            "validation": jsonable(report_validation),
            "executable": bool(validation.get("executable")),
            "syntax_error": bool(validation.get("syntax_error")),
            "timeout": bool(validation.get("timeout")),
            "elapsed_ms": validation.get("elapsed_ms"),
            "row_count": len(rows),
            "result_hash": result_hash(rows) if validation.get("executable") else None,
            "execution_error": validation.get("error"),
        }
    )
    if not validation.get("executable"):
        return {**base, "status": "execution_error"}
    if result_too_large:
        return {**base, "status": "result_too_large"}

    expected_solution = record_expected_solution(record)
    is_pddl = record_task_family(record) == "cypher_pddl"
    if is_pddl:
        cypher_solution_match, cypher_solution, actual_grounding = validate_grounding_rows(
            rows, record.get("grounding")
        )
        format_valid = True
        base["expected_grounding"] = record.get("grounding")
        base["actual_grounding"] = actual_grounding
        base["grounding_match"] = cypher_solution_match
    else:
        try:
            cypher_solution = rows_to_solution_text(record_for_solution_conversion(record), rows)
        except ResultShapeError as exc:
            return {**base, "status": "result_shape_error", "result_shape_error": str(exc)}
        format_valid = False
        cypher_solution_match = False
        if expected_solution is not None:
            format_valid, cypher_solution_match = answer_matches_with_heracles(
                str(cypher_solution),
                str(expected_solution),
                record_output_type(record),
                answer_validation=record.get("answer_validation"),
                solution_formats=record.get("solution_formats") or record.get("solution_format"),
            )
    status = "pass"
    if expected_solution is None:
        status = "missing_expected_solution"
    elif not format_valid:
        status = "invalid_cypher_solution_format"
    elif not cypher_solution_match:
        status = (
            "grounding_mismatch"
            if record_task_family(record) == "cypher_pddl"
            else "cypher_solution_mismatch"
        )

    match_diagnostics = result_match_diagnostics(
        generated_cypher=cypher,
        gold_cypher=record_gold_cypher(record),
        cypher_exact_match=bool(base["cypher_exact_match"]),
        solution_match=bool(cypher_solution_match),
        rows=rows,
        expected_solution=expected_solution,
    )

    return {
        **base,
        **match_diagnostics,
        "status": status,
        "cypher_solution": cypher_solution,
        "cypher_solution_format_valid": format_valid,
        "cypher_solution_match": cypher_solution_match,
        "tool_result_match": cypher_solution_match,
    }


def summarize_raw_section(
    records: list[dict[str, Any]],
    section: str,
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    total = len(records)
    section_records = [record.get(section) or {"status": "skipped"} for record in records]
    status_counts = Counter(str(record.get("status") or "unknown") for record in section_records)
    skipped = status_counts.get("skipped", 0)
    evaluated = total - skipped
    metrics = {
        key: sum(1 for record in section_records if bool(record.get(key))) for key in metric_keys
    }
    denominator = evaluated if evaluated else total
    return {
        "total": total,
        "evaluated": evaluated,
        "skipped": skipped,
        "metrics": metrics,
        "rates": {
            key: (value / denominator if denominator else 0.0) for key, value in metrics.items()
        },
        "statuses": dict(status_counts),
    }


def summarize_raw_generation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "examples": len(records),
        FINAL_ANSWER_CHECK: summarize_raw_section(
            records,
            FINAL_ANSWER_CHECK,
            (
                "final_answer_generated",
                "final_answer_has_tags",
                "final_answer_format_valid",
                "final_answer_match",
            ),
        ),
        TOOL_CALL_CHECK: summarize_raw_section(
            records,
            TOOL_CALL_CHECK,
            (
                "tool_call_generated",
                "tool_call_parse_success",
                "correct_tool_name",
                "cypher_argument_present",
                "cypher_exact_match",
                "static_valid",
                "executable",
                "cypher_solution_match",
                "tool_result_match",
                "result_match_without_exact_cypher_match",
                "potential_hidden_false_positive",
            ),
        ),
        END_TO_END_CHECK: summarize_raw_section(
            records,
            END_TO_END_CHECK,
            ("final_answer_match",),
        ),
    }
    by_family = {}
    for family in sorted({record.get("task_family", "cypher_qa") for record in records}):
        family_records = [
            record for record in records if record.get("task_family", "cypher_qa") == family
        ]
        family_tool_metrics = [
            "static_valid",
            "executable",
            "tool_result_match",
            "potential_hidden_false_positive",
        ]
        if family == "cypher_pddl":
            family_tool_metrics.append("grounding_match")
        else:
            family_tool_metrics.append("cypher_solution_match")
        by_family[family] = {
            "examples": len(family_records),
            FINAL_ANSWER_CHECK: summarize_raw_section(
                family_records,
                FINAL_ANSWER_CHECK,
                (
                    "final_answer_format_valid",
                    "final_answer_domain_valid",
                    "final_answer_match",
                ),
            ),
            TOOL_CALL_CHECK: summarize_raw_section(
                family_records,
                TOOL_CALL_CHECK,
                tuple(family_tool_metrics),
            ),
            END_TO_END_CHECK: summarize_raw_section(
                family_records,
                END_TO_END_CHECK,
                ("final_answer_match",),
            ),
        }
    summary["by_task_family"] = by_family
    return summary


def raw_generation_progress_context(show_progress: bool):
    if not show_progress:
        return nullcontext(None)

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def split_display_name(split: str) -> str:
    return split.replace("_", " ").replace("-", " ").title()


def raw_generation_progress_description(
    *,
    split: str,
    phase: str,
) -> str:
    phase_labels = {
        "final-answer": "Validating final answer",
        "final-answer tokens": "Generating final answer",
        "tool-call": "Validating tool call",
        "tool-call tokens": "Generating tool call",
        "end-to-end": "Validating end to end",
        "end-to-end tokens": "Generating end-to-end response",
        "done": "Validating records",
    }
    return f"{phase_labels[phase]} · {split_display_name(split)}"


def raw_generation_report_record(
    record: dict[str, Any],
    *,
    final_answer_result: dict[str, Any],
    tool_call_result: dict[str, Any],
    final_generation_debug: dict[str, Any],
    tool_generation_debug: dict[str, Any],
    end_to_end_result: dict[str, Any] | None = None,
    end_to_end_generation_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one raw-generation report record with task-aware result fields."""
    family = record_task_family(record)
    expected_solution = record_expected_solution(record)
    expected = {
        "cypher": record_gold_cypher(record),
        "cypher_solution": expected_solution,
        "final_answer": expected_final_answer(record),
    }
    generated = {
        "cypher": tool_call_result.get("generated_cypher"),
        "cypher_solution": tool_call_result.get("cypher_solution"),
        "final_answer": final_answer_result.get("generated_text"),
        "final_answer_value": final_answer_result.get("generated_answer"),
        "cypher_tool_output": (tool_call_result.get("validation") or {}).get("rows"),
        "end_to_end_final_answer": (end_to_end_result or {}).get("generated_text"),
    }
    if family == "cypher_pddl":
        empty_answer = expected_solution is not None and not str(expected_solution).strip()
        expected.update(
            {
                "pddl_goal": None if empty_answer else expected_solution,
                "empty_answer": empty_answer,
                "grounding": record.get("grounding"),
            }
        )
        generated["actual_grounding"] = tool_call_result.get("actual_grounding")

    return {
        "uid": record.get("uid"),
        "query_type": record_query_type(record),
        "task_family": family,
        "output_type": record_output_type(record),
        "question": record.get("question"),
        "controlled_repair": bool((record.get("metadata") or {}).get("controlled_repair")),
        "repair_template": (record.get("metadata") or {}).get("repair_template"),
        "expected": expected,
        "generated": generated,
        "generation_debug": {
            "cypher_generation": tool_generation_debug,
            "final_answer_generation": final_generation_debug,
            "end_to_end_generation": end_to_end_generation_debug or {"turns": []},
        },
        FINAL_ANSWER_CHECK: final_answer_result,
        TOOL_CALL_CHECK: tool_call_result,
        END_TO_END_CHECK: end_to_end_result
        or end_to_end_skipped_result("end-to-end validation was not run"),
    }


def evaluate_raw_generation(
    *,
    data_config: dict[str, Any],
    split: str,
    runtime: RawGenerationRuntime,
    max_new_tokens: int | None,
    limit: int | None,
    validator,
    tool_call_skip_reason: str | None,
    policy: dict[str, Any] | None,
    show_progress: bool,
    enable_thinking: bool,
    max_query_retries: int = DEFAULT_MAX_QUERY_RETRIES,
    max_tool_rows: int = DEFAULT_E2E_MAX_TOOL_ROWS,
    max_tool_response_chars: int = DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS,
    max_query_chars: int = DEFAULT_E2E_MAX_QUERY_CHARS,
    scene_loader=None,
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.agent_prompts import run_cypher_tool_schema
    from cypher_3dsg_agent_distillation.core.io import read_jsonl

    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero when provided.")
    if max_query_retries < 0:
        raise ValueError("max_query_retries must be zero or greater.")
    if max_tool_rows <= 0:
        raise ValueError("max_tool_rows must be greater than zero.")
    if max_tool_response_chars < 256:
        raise ValueError("max_tool_response_chars must be at least 256.")
    if max_query_chars < 256:
        raise ValueError("max_query_chars must be at least 256.")
    if tool_call_skip_reason is None and validator is None:
        raise ValueError("validator is required when tool-call evaluation is not skipped.")

    records = read_jsonl(data_config["splits"][split])
    scene_order = [str(scene["scene_id"]) for scene in data_config.get("scenes", [])]
    records = group_records_by_scene(
        records,
        seed=0,
        scene_order=scene_order,
    )
    if limit is not None:
        records = records[:limit]
    scenes_by_id = {str(scene["scene_id"]): scene for scene in data_config.get("scenes", [])}
    current_scene_id: str | None = None
    scene_database_reloads = 0
    tool_schemas = [run_cypher_tool_schema()]

    evaluated_records = []
    with raw_generation_progress_context(show_progress) as progress:
        record_task_id = None
        token_task_id = None
        if progress is not None:
            record_task_id = progress.add_task(
                f"Validating records · {split_display_name(split)}",
                total=len(records),
            )
            token_task_id = progress.add_task(
                "Generating response",
                total=1,
                visible=False,
            )

        for record in records:
            record_scene_id = str(record.get("scene_id") or "")
            if validator is not None and scene_loader is not None:
                if record_scene_id not in scenes_by_id:
                    raise ValueError(
                        f"Record {record.get('uid')} references unknown scene `{record_scene_id}`."
                    )
                if record_scene_id != current_scene_id:
                    scene_loader(scenes_by_id[record_scene_id])
                    current_scene_id = record_scene_id
                    scene_database_reloads += 1
            final_messages = final_answer_prompt_messages(record)
            tool_messages = initial_prompt_messages(record)
            if progress is not None:
                progress.update(
                    record_task_id,
                    description=raw_generation_progress_description(
                        split=split,
                        phase="final-answer",
                    ),
                )
            try:
                final_generation = runtime.generate(
                    final_messages,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    tools=tool_schemas,
                    progress=progress,
                    progress_task_id=token_task_id,
                    progress_description=raw_generation_progress_description(
                        split=split,
                        phase="final-answer tokens",
                    ),
                )
                final_answer_result = score_final_answer_generation(
                    record,
                    final_generation.text,
                )
                add_raw_generation_debug_fields(final_answer_result, final_generation)
                final_answer_result["generation"] = generation_report_fields(final_generation)
                final_generation_debug = generation_debug_report(
                    final_generation,
                )
            except Exception as exc:
                final_answer_result = final_answer_generation_error(record, exc)
                final_generation_debug = generation_debug_report(
                    None,
                )

            if tool_call_skip_reason is not None:
                tool_call_result = skipped_tool_call_result(tool_call_skip_reason)
                tool_generation_debug = generation_debug_report(
                    None,
                )
            else:
                if progress is not None:
                    progress.update(
                        record_task_id,
                        description=raw_generation_progress_description(
                            split=split,
                            phase="tool-call",
                        ),
                    )
                try:
                    tool_generation = runtime.generate(
                        tool_messages,
                        max_new_tokens=max_new_tokens,
                        enable_thinking=enable_thinking,
                        tools=tool_schemas,
                        progress=progress,
                        progress_task_id=token_task_id,
                        progress_description=raw_generation_progress_description(
                            split=split,
                            phase="tool-call tokens",
                        ),
                    )
                    tool_call_result = score_tool_call_generation(
                        record,
                        tool_generation.text,
                        validator=validator,
                        policy=policy,
                        max_rows=max_tool_rows,
                        max_result_chars=max_tool_response_chars,
                        structured_tool_calls=tool_generation.tool_calls,
                    )
                    add_raw_generation_debug_fields(tool_call_result, tool_generation)
                    tool_call_result["generation"] = generation_report_fields(tool_generation)
                    tool_generation_debug = generation_debug_report(
                        tool_generation,
                    )
                except Exception as exc:
                    tool_call_result = tool_call_generation_error(record, exc)
                    tool_generation_debug = generation_debug_report(
                        None,
                    )

            end_to_end_generation_debug: dict[str, Any] = {"turns": []}
            if tool_call_skip_reason is not None:
                end_to_end_result = end_to_end_skipped_result(tool_call_skip_reason)
            else:
                if progress is not None:
                    progress.update(
                        record_task_id,
                        description=raw_generation_progress_description(
                            split=split,
                            phase="end-to-end",
                        ),
                    )
                try:
                    end_to_end_result = evaluate_end_to_end_record(
                        record,
                        runtime=runtime,
                        validator=validator,
                        policy=policy,
                        tool_schemas=tool_schemas,
                        max_new_tokens=max_new_tokens,
                        enable_thinking=enable_thinking,
                        max_query_retries=max_query_retries,
                        max_tool_rows=max_tool_rows,
                        max_tool_response_chars=max_tool_response_chars,
                        max_query_chars=max_query_chars,
                        progress=progress,
                        progress_task_id=token_task_id,
                        progress_description=raw_generation_progress_description(
                            split=split,
                            phase="end-to-end tokens",
                        ),
                        generation_debug_turns=end_to_end_generation_debug["turns"],
                    )
                except Exception as exc:
                    end_to_end_result = end_to_end_failure_result(
                        record,
                        status="evaluation_error",
                        query_attempts=0,
                        max_query_retries=max_query_retries,
                        attempts=[],
                        generation_error=bounded_text(f"{type(exc).__name__}: {exc}", 1024),
                    )

            evaluated_records.append(
                raw_generation_report_record(
                    record,
                    final_answer_result=final_answer_result,
                    tool_call_result=tool_call_result,
                    final_generation_debug=final_generation_debug,
                    tool_generation_debug=tool_generation_debug,
                    end_to_end_result=end_to_end_result,
                    end_to_end_generation_debug=end_to_end_generation_debug,
                )
            )
            runtime.release_memory()
            if progress is not None:
                progress.update(
                    record_task_id,
                    advance=1,
                    description=raw_generation_progress_description(
                        split=split,
                        phase="done",
                    ),
                )
        if progress is not None:
            progress.update(
                record_task_id,
                description=f"{split_display_name(split)} validation complete",
            )
            progress.update(token_task_id, visible=False)

    return {
        "summary": summarize_raw_generation_records(evaluated_records),
        "records": evaluated_records,
        "generation_config": runtime.generation_config(max_new_tokens=max_new_tokens),
        "runtime": runtime.metadata(),
        "scene_database_reloads": scene_database_reloads,
    }


def build_report(
    *,
    summary: dict[str, Any],
    data_config_path: str,
    student_config_path: str,
    data_config: dict[str, Any],
    student_config: dict[str, Any],
    split: str,
    mode: str,
    adapter_dir: str,
    explicit_tokenizer: str | None,
    baseline: bool = False,
    batch_size: int | None = None,
    max_length: int | None = None,
    loss_mode: str | None = None,
    records: list[dict[str, Any]] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    runtime: str = TRANSFORMERS_RUNTIME,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "split": split,
        "data_config": data_config_path,
        "dataset_version": data_config.get("dataset_version"),
        "evaluated_on_dataset": {
            "dataset_id": data_config.get("dataset_id"),
            "scene_ids": [scene["scene_id"] for scene in data_config.get("scenes", [])],
            "version": data_config.get("dataset_version"),
            "teacher_name": data_config.get("teacher_name"),
        },
        "student_config": student_config_path,
        "model_family": student_config.get("model_family"),
        "model_version": None if baseline else student_config.get("model_version"),
        "split_path": (
            data_config["splits"][split]
            if split in data_config["splits"]
            else {name: data_config["splits"][name] for name in SUPPORTED_SPLITS}
        ),
        "base_model": student_config["model"]["base_model"],
        "model_variant": "baseline" if baseline else "fine-tuned",
        "adapter_dir": None if baseline else adapter_dir,
        "runtime": runtime,
        "tokenizer": (
            tokenizer_source(
                adapter_dir=adapter_dir,
                explicit_tokenizer=explicit_tokenizer,
                model_config=student_config["model"],
                use_adapter=not baseline,
            )
            if runtime == TRANSFORMERS_RUNTIME
            else None
        ),
    }
    metadata["trained_on_dataset"] = None if baseline else (model_metadata or {}).get("dataset")
    if batch_size is not None:
        metadata["batch_size"] = batch_size
    if max_length is not None:
        metadata["max_length"] = max_length
    if loss_mode is not None:
        metadata["loss_mode"] = loss_mode
    if extra_metadata:
        metadata.update(extra_metadata)

    report = {
        "metadata": metadata,
        "summary": summary,
    }
    if records is not None:
        report["records"] = records
    return report


def write_report(report: dict[str, Any], output: str | Path) -> Path:
    output_path = ensure_parent(output)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".report-", dir=output_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def connect_raw_generation_validator(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
        Neo4jValidator,
        neo4j_connection_settings,
    )

    connection = neo4j_connection_settings(args)
    metadata = {
        "uri_candidates": connection["uri_candidates"],
        "username": connection["username"],
        "database": connection["database"],
        "connected": False,
        "connected_uri": None,
        "skip_reason": None,
    }
    try:
        validator = Neo4jValidator(
            uri=connection["uri_candidates"],
            username=connection["username"],
            password=connection["password"],
            database=connection["database"],
            timeout=args.neo4j_timeout,
            connection_timeout=args.neo4j_connection_timeout,
        )
    except Exception as exc:
        metadata["skip_reason"] = f"Neo4j unavailable: {type(exc).__name__}: {exc}"
        return None, metadata

    metadata["connected"] = True
    metadata["connected_uri"] = validator.uri
    return validator, metadata


def configure_neo4j_notification_output(*, show_notifications: bool) -> None:
    """Keep Neo4j notifications out of the terminal unless explicitly requested."""
    if show_notifications:
        return
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
        suppress_neo4j_notification_logs,
    )

    suppress_neo4j_notification_logs()


def build_raw_generation_runtime(
    args: argparse.Namespace,
    *,
    student_config: dict[str, Any],
    adapter_dir: str,
) -> RawGenerationRuntime:
    if args.runtime == TRANSFORMERS_RUNTIME:
        return TransformersRawGenerationRuntime.load(
            student_config=student_config,
            adapter_dir=adapter_dir,
            explicit_tokenizer=args.tokenizer,
            show_progress=not args.no_progress,
            baseline=args.baseline,
        )

    if not args.ollama_model:
        raise ValueError("--ollama-model is required with --runtime ollama.")
    runtime = OllamaRawGenerationRuntime(
        model=args.ollama_model,
        host=args.ollama_host,
        timeout=args.ollama_timeout,
        temperature=args.ollama_temperature,
        seed=args.ollama_seed,
        context_length=(
            args.ollama_context_length
            if args.ollama_context_length is not None
            else max(
                DEFAULT_OLLAMA_CONTEXT_LENGTH,
                int(student_config["training"].get("max_length", 4096))
                + int(args.max_new_tokens or 0),
            )
        ),
        keep_alive=args.ollama_keep_alive,
    )
    runtime.validate()
    return runtime


def print_eval_loss_summary(
    *,
    split: str,
    mode: str,
    loss_mode: str,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    perplexity = summary["perplexity"]
    perplexity_text = f"{perplexity:.6f}" if perplexity is not None else "overflow"
    print_summary(
        "Student eval-loss complete",
        [
            ("Split", split),
            ("Mode", mode),
            ("Loss mode", loss_mode),
            ("Examples", summary["examples"]),
            ("Tokens", summary["tokens"]),
            ("Evaluation loss", f"{summary['eval_loss']:.6f}"),
            ("Perplexity", perplexity_text),
            ("Evaluation entropy", f"{summary['eval_entropy']:.6f}"),
            ("Mean token accuracy", f"{summary['eval_mean_token_accuracy']:.6f}"),
            ("Report", output_path),
        ],
    )


def print_eval_loss_all_splits_summary(
    *,
    mode: str,
    loss_mode: str,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    perplexity = summary["perplexity"]
    perplexity_text = f"{perplexity:.6f}" if perplexity is not None else "overflow"
    print_summary(
        "Student eval-loss complete",
        [
            ("Split", ALL_SPLITS),
            ("Mode", mode),
            ("Loss mode", loss_mode),
            ("Examples", summary["examples"]),
            ("Tokens", summary["tokens"]),
            ("Evaluation loss", f"{summary['eval_loss']:.6f}"),
            ("Perplexity", perplexity_text),
            ("Evaluation entropy", f"{summary['eval_entropy']:.6f}"),
            ("Mean token accuracy", f"{summary['eval_mean_token_accuracy']:.6f}"),
            ("Report", output_path),
        ],
    )
    print_breakdown(
        "Eval loss by split",
        ("Split", "Examples", "Tokens", "Evaluation loss"),
        (
            (
                split,
                split_summary["examples"],
                split_summary["tokens"],
                f"{split_summary['eval_loss']:.6f}",
            )
            for split, split_summary in summary["splits"].items()
        ),
        right_align=(1, 2, 3),
    )


def format_count_rate(section: dict[str, Any], metric: str) -> str:
    count = section["metrics"][metric]
    evaluated = section["evaluated"]
    rate = section["rates"][metric]
    return f"{count}/{evaluated} ({rate:.6f})"


def print_raw_generation_summary(
    *,
    split: str,
    mode: str,
    summary: dict[str, Any],
    output_path: Path,
    baseline: bool = False,
) -> None:
    final_answer = summary[FINAL_ANSWER_CHECK]
    tool_call = summary[TOOL_CALL_CHECK]
    end_to_end = summary[END_TO_END_CHECK]
    rows = [
        ("Split", split),
        ("Mode", mode),
        ("Examples", summary["examples"]),
        ("Final-answer match", format_count_rate(final_answer, "final_answer_match")),
        ("Final-answer tags", format_count_rate(final_answer, "final_answer_has_tags")),
    ]
    if end_to_end["skipped"] == end_to_end["total"]:
        rows.append(
            (
                "End-to-end validation",
                f"skipped {end_to_end['skipped']}/{end_to_end['total']}",
            )
        )
    else:
        rows.append(
            (
                "End-to-end final-answer match",
                format_count_rate(end_to_end, "final_answer_match"),
            )
        )
    if tool_call["skipped"]:
        rows.append(
            ("Tool-call validation", f"skipped {tool_call['skipped']}/{tool_call['total']}")
        )
    else:
        rows.extend(
            [
                (
                    "Tool-call parse success",
                    format_count_rate(tool_call, "tool_call_parse_success"),
                ),
                ("Static-valid Cypher", format_count_rate(tool_call, "static_valid")),
                ("Executable Cypher", format_count_rate(tool_call, "executable")),
                ("Cypher solution match", format_count_rate(tool_call, "cypher_solution_match")),
                (
                    "Potential hidden false positives",
                    format_count_rate(tool_call, "potential_hidden_false_positive"),
                ),
            ]
        )
    rows.append(("Report", output_path))
    title = (
        "Baseline raw-generation validation complete"
        if baseline
        else "Student raw-generation validation complete"
    )
    print_summary(title, rows)


def command(args: argparse.Namespace) -> None:
    configure_model_library_output()
    data_config = load_data_config(args.data_config, dataset_version=args.dataset_version)
    family_config = load_student_family_config(args.student_config)
    if args.baseline and args.mode != RAW_GENERATION_MODE:
        raise ValueError("--baseline is supported only with --mode raw-generation.")
    if args.mode == EVAL_LOSS_MODE and args.runtime != TRANSFORMERS_RUNTIME:
        raise ValueError("eval-loss is supported only with --runtime transformers.")

    export_metadata = None
    if args.runtime != TRANSFORMERS_RUNTIME:
        if not args.ollama_model:
            raise ValueError("--ollama-model is required with --runtime ollama.")
        export_metadata = ollama_export_metadata(family_config, args.ollama_model)

    model_metadata = None
    if args.baseline:
        student_config = family_config
        adapter_dir = args.adapter_dir or str(Path(family_config["model_family_root"]) / "baseline")
        selected_reports_dir = model_reports_dir(data_config, student_config, baseline=True)
    else:
        exported_version = None
        adapter_metadata = None
        adapter_version = None
        if args.adapter_dir and (Path(args.adapter_dir) / "metadata.yaml").is_file():
            adapter_metadata = read_metadata(Path(args.adapter_dir) / "metadata.yaml")
            adapter_version = adapter_metadata.get("version")
        if export_metadata:
            exported_version = (export_metadata.get("source_model") or {}).get("version")
            if not exported_version:
                raise ValueError(
                    "The Ollama export metadata does not identify its source LoRA version."
                )
            if args.model_version not in (None, "latest", exported_version):
                raise ValueError(
                    f"Requested model version {args.model_version} does not match Ollama "
                    f"export source version {exported_version}."
                )
        if adapter_version and args.model_version not in (None, "latest", adapter_version):
            raise ValueError(
                f"Requested model version {args.model_version} does not match adapter "
                f"metadata version {adapter_version}."
            )
        if args.adapter_dir:
            student_config = dict(family_config)
            training = dict(student_config["training"])
            training["output_dir"] = args.adapter_dir
            student_config["training"] = training
            student_config["model_version"] = str(adapter_version or "external")
            student_config["model_dir"] = args.adapter_dir
        elif export_metadata is not None:
            student_config = dict(family_config)
            source_model = export_metadata["source_model"]
            student_config["model_family"] = str(
                source_model.get("family") or family_config["model_family"]
            )
            student_config["model_version"] = str(exported_version)
            student_config["model_dir"] = source_model.get("path")
            training = dict(student_config["training"])
            training["output_dir"] = str(source_model.get("path") or "")
            student_config["training"] = training
        else:
            student_config = load_student_config(
                args.student_config,
                model_version=str(exported_version or args.model_version or "latest"),
            )
        adapter_dir = args.adapter_dir or default_adapter_dir(student_config)
        metadata_file = Path(adapter_dir) / "metadata.yaml"
        if export_metadata is not None:
            model_metadata = export_metadata
        elif adapter_metadata is not None:
            model_metadata = adapter_metadata
        elif metadata_file.is_file():
            model_metadata = read_metadata(metadata_file)
        elif not args.allow_dataset_mismatch:
            raise RuntimeError(
                f"Adapter metadata is missing at {metadata_file}. Re-run with "
                "--allow-dataset-mismatch to use an adapter with unknown lineage."
            )
        if model_metadata is not None:
            if not confirm_dataset_lineage(
                model_metadata=model_metadata,
                data_config=data_config,
                allow_mismatch=args.allow_dataset_mismatch,
            ):
                return
        selected_reports_dir = model_reports_dir(data_config, student_config)

    output_path = report_output_path(
        data_config=data_config,
        split=args.split,
        mode=args.mode,
        output=args.output,
        baseline=args.baseline,
        runtime=args.runtime,
        runtime_model=args.ollama_model,
        reports_directory=selected_reports_dir,
    )
    runtime_identity = None
    if args.runtime != TRANSFORMERS_RUNTIME:
        runtime_identity = OllamaRawGenerationRuntime(
            model=args.ollama_model, host=args.ollama_host, timeout=args.ollama_timeout,
        ).cache_identity()
    identity = evaluation_identity(
        args, data_config, student_config, adapter_dir, runtime_identity=runtime_identity,
    )
    if not args.force and completed_evaluation(output_path, identity):
        print_summary("Validation already complete; skipping", [("Report", output_path)])
        return

    records = None
    extra_metadata = None
    batch_size = None
    max_length = None
    loss_mode = None
    if args.mode == EVAL_LOSS_MODE:
        from cypher_3dsg_agent_distillation.training.train import (
            loss_mode_from_training_config,
            normalize_loss_mode,
        )

        batch_size = args.batch_size
        max_length = args.max_length or int(student_config["training"].get("max_length", 4096))
        loss_mode = (
            normalize_loss_mode(args.loss_mode)
            if args.loss_mode is not None
            else loss_mode_from_training_config(student_config["training"])
        )
        if args.split == ALL_SPLITS:
            summary = combine_eval_loss_summaries(
                evaluate_eval_loss_splits(
                    data_config=data_config,
                    student_config=student_config,
                    splits=SUPPORTED_SPLITS,
                    adapter_dir=adapter_dir,
                    explicit_tokenizer=args.tokenizer,
                    batch_size=batch_size,
                    max_length=max_length,
                    loss_mode=loss_mode,
                    show_progress=not args.no_progress,
                )
            )
        else:
            summary = evaluate_eval_loss(
                data_config=data_config,
                student_config=student_config,
                split=args.split,
                adapter_dir=adapter_dir,
                explicit_tokenizer=args.tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                loss_mode=loss_mode,
                show_progress=not args.no_progress,
            )
    else:
        if args.split == ALL_SPLITS:
            raise ValueError("raw-generation mode does not support `--split all`.")
        from cypher_3dsg_agent_distillation.graph.neo4j_validation import schema_policy

        show_db_notifications = bool(getattr(args, "show_db_notifications", False))
        configure_neo4j_notification_output(
            show_notifications=show_db_notifications,
        )

        runtime = build_raw_generation_runtime(
            args,
            student_config=student_config,
            adapter_dir=adapter_dir,
        )
        validator, neo4j_metadata = connect_raw_generation_validator(args)
        scene_loader = None
        if validator is not None:
            from cypher_3dsg_agent_distillation.graph.load_neo4j import load_scene
            from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
                neo4j_connection_settings,
            )

            connection = neo4j_connection_settings(args)

            def scene_loader(scene):
                return load_scene(
                    scene_path=scene["scene_path"],
                    uri=connection["uri_candidates"],
                    username=connection["username"],
                    password=connection["password"],
                    database=connection["database"],
                    clear=True,
                )

        try:
            raw_generation = evaluate_raw_generation(
                data_config=data_config,
                split=args.split,
                runtime=runtime,
                max_new_tokens=args.max_new_tokens,
                limit=args.limit,
                validator=validator,
                tool_call_skip_reason=neo4j_metadata["skip_reason"],
                policy=schema_policy(load_yaml(data_config["schema_path"])),
                show_progress=not args.no_progress,
                enable_thinking=args.enable_thinking,
                max_query_retries=args.max_query_retries,
                max_tool_rows=args.e2e_max_tool_rows,
                max_tool_response_chars=args.e2e_max_tool_response_chars,
                max_query_chars=args.e2e_max_query_chars,
                scene_loader=scene_loader,
            )
        finally:
            if validator is not None:
                validator.close()
        summary = raw_generation["summary"]
        records = raw_generation["records"]
        extra_metadata = {
            "raw_generation": {
                "checks": [FINAL_ANSWER_CHECK, TOOL_CALL_CHECK, END_TO_END_CHECK],
                "limit": args.limit,
                "max_new_tokens": args.max_new_tokens,
                "max_query_retries": args.max_query_retries,
                "max_query_attempts": args.max_query_retries + 1,
                "e2e_max_tool_rows": args.e2e_max_tool_rows,
                "e2e_max_tool_response_chars": args.e2e_max_tool_response_chars,
                "e2e_max_query_chars": args.e2e_max_query_chars,
                "enable_thinking": args.enable_thinking,
                "progress": not args.no_progress,
                "model_variant": "baseline" if args.baseline else "fine-tuned",
                "runtime": raw_generation["runtime"],
                "generation_config": raw_generation["generation_config"],
                "neo4j": neo4j_metadata,
                "show_db_notifications": show_db_notifications,
                "scene_database_reloads": raw_generation["scene_database_reloads"],
            }
        }

    report = build_report(
        summary=summary,
        data_config_path=args.data_config,
        student_config_path=args.student_config,
        data_config=data_config,
        student_config=student_config,
        split=args.split,
        mode=args.mode,
        adapter_dir=adapter_dir,
        explicit_tokenizer=args.tokenizer,
        baseline=args.baseline,
        batch_size=batch_size,
        max_length=max_length,
        loss_mode=loss_mode,
        records=records,
        extra_metadata=extra_metadata,
        runtime=args.runtime,
        model_metadata=model_metadata,
    )
    report["metadata"].update({
        "evaluation_identity": identity,
        "evaluation_status": "complete" if evaluation_completed(report) else "incomplete",
    })
    write_report(report, output_path)

    if args.mode == EVAL_LOSS_MODE:
        if args.split == ALL_SPLITS:
            print_eval_loss_all_splits_summary(
                mode=args.mode,
                loss_mode=loss_mode,
                summary=summary,
                output_path=output_path,
            )
        else:
            print_eval_loss_summary(
                split=args.split,
                mode=args.mode,
                loss_mode=loss_mode,
                summary=summary,
                output_path=output_path,
            )
    else:
        print_raw_generation_summary(
            split=args.split,
            mode=args.mode,
            summary=summary,
            output_path=output_path,
            baseline=args.baseline,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned or baseline student model on configured splits."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run validation even when a matching completed report exists.",
    )
    parser.add_argument(
        "--student-config",
        default="configs/student_qwen3.5_4b.yaml",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="LoRA version. Defaults to the latest completed version.",
    )
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Continue non-interactively when model and dataset lineage differ.",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="test",
        help="Configured split to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=(EVAL_LOSS_MODE, RAW_GENERATION_MODE),
        default=EVAL_LOSS_MODE,
        help=(
            "`eval-loss` computes held-out loss/perplexity. "
            "`raw-generation` evaluates the existing isolated checks plus a bounded "
            "end-to-end query/tool/final-answer interaction."
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=RAW_GENERATION_RUNTIMES,
        default=TRANSFORMERS_RUNTIME,
        help=(
            "Inference runtime for raw-generation validation. Eval-loss requires "
            "Transformers (default: transformers)."
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help=(
            "LoRA adapter directory. Defaults to the selected/latest completed version "
            "below training.output_dir."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Evaluate the configured base student model without loading its LoRA "
            "adapter when using Transformers. With Ollama, this labels the selected "
            "registered model as the baseline. Supported only in raw-generation mode."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer path/name. Fine-tuned evaluation defaults to the adapter "
            "tokenizer when available; baseline evaluation uses the configured model "
            "tokenizer."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--loss-mode",
        choices=("full", "assistant-only"),
        default=None,
        help=(
            "Token labels to score for eval-loss. Defaults to training.loss_mode in "
            "the student config, or assistant-only when unset."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Maximum tokenized example length for eval-loss mode. Defaults to "
            "training.max_length in the student config."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to evaluate in raw-generation mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Optional generated-token limit for raw-generation mode. When unset, "
            "Transformers uses the loaded generation_config and Ollama uses the "
            "registered model/server default."
        ),
    )
    parser.add_argument(
        "--max-query-retries",
        type=int,
        default=DEFAULT_MAX_QUERY_RETRIES,
        help=(
            "Maximum corrected query retries in end-to-end validation after the "
            "initial query attempt (default: 1)."
        ),
    )
    parser.add_argument(
        "--e2e-max-tool-rows",
        type=int,
        default=DEFAULT_E2E_MAX_TOOL_ROWS,
        help=(
            "Maximum rows streamed from an end-to-end query before it is retried "
            f"(default: {DEFAULT_E2E_MAX_TOOL_ROWS})."
        ),
    )
    parser.add_argument(
        "--e2e-max-tool-response-chars",
        type=int,
        default=DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS,
        help=(
            "Maximum serialized tool-response characters fed back to the model "
            f"(default: {DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS})."
        ),
    )
    parser.add_argument(
        "--e2e-max-query-chars",
        type=int,
        default=DEFAULT_E2E_MAX_QUERY_CHARS,
        help="Maximum generated Cypher characters accepted for execution.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Allow tokenizer chat-template thinking mode for raw-generation. By "
            "default, thinking is disabled when the tokenizer supports it."
        ),
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Registered Ollama model name. Required with --runtime ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama API URL. Defaults to OLLAMA_HOST or http://ollama:11434.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for each non-streaming Ollama request.",
    )
    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=0.0,
        help="Ollama sampling temperature (default: 0 for paired comparisons).",
    )
    parser.add_argument(
        "--ollama-seed",
        type=int,
        default=42,
        help="Ollama generation seed (default: 42).",
    )
    parser.add_argument(
        "--ollama-context-length",
        type=int,
        default=DEFAULT_OLLAMA_CONTEXT_LENGTH,
        help=(
            "Ollama context length (default: 16384, sized for the longest current "
            "challenge tool-response conversations)."
        ),
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default="5m",
        help="How long Ollama keeps the selected model loaded (default: 5m).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Neo4j URI or comma-separated URI fallback list for raw-generation mode.",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Neo4j username for raw-generation mode.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Neo4j password for raw-generation mode.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Neo4j database name for raw-generation mode.",
    )
    parser.add_argument(
        "--neo4j-timeout",
        type=float,
        default=10.0,
        help="Per-query Neo4j timeout in seconds for raw-generation mode.",
    )
    parser.add_argument(
        "--neo4j-connection-timeout",
        type=float,
        default=3.0,
        help="Neo4j connection timeout in seconds for raw-generation mode.",
    )
    parser.add_argument(
        "--show-db-notifications",
        action="store_true",
        help=(
            "Show Neo4j DBMS notification warnings during raw-generation queries. "
            "They are hidden by default but remain available in validation results."
        ),
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
