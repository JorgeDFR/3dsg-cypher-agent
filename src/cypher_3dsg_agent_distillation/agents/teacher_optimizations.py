from __future__ import annotations

import json
import warnings
from copy import deepcopy
from typing import Any

from cypher_3dsg_agent_distillation.agents.agent_records import (
    DEFAULT_TEACHER_MODEL,
    AgentTraceRecord,
    split_group_for_record,
)
from cypher_3dsg_agent_distillation.agents.reasoning import (
    reasoning_text,
    split_generated_reasoning,
)
from cypher_3dsg_agent_distillation.graph.heracles_validation import answer_matches_with_heracles


def configured_teacher_optimizations(config: dict[str, Any]) -> dict[str, Any]:
    """Validate teacher optimizations and disable those unsupported by the provider."""
    raw = config.get("optimizations", {})
    if not isinstance(raw, dict):
        raise ValueError("optimizations must be a mapping.")
    unknown = raw.keys() - {
        "repair_invalid_final_answers", "recover_earlier_retrievals", "tuning_prompt",
    }
    if unknown:
        raise ValueError(f"Unknown teacher optimizations: {', '.join(sorted(unknown))}")
    repair = raw.get("repair_invalid_final_answers", False)
    recover = raw.get("recover_earlier_retrievals", False)
    prompt = raw.get("tuning_prompt", "")
    if not isinstance(repair, bool):
        raise ValueError("optimizations.repair_invalid_final_answers must be a boolean.")
    if not isinstance(recover, bool):
        raise ValueError("optimizations.recover_earlier_retrievals must be a boolean.")
    if not isinstance(prompt, str):
        raise ValueError("optimizations.tuning_prompt must be a string.")
    effective = {
        "repair_invalid_final_answers": repair, "recover_earlier_retrievals": recover,
        "tuning_prompt": prompt.strip(),
    }
    if str(config.get("provider", "")).strip().lower() == "synthetic":
        for name, value in effective.items():
            if value:
                warnings.warn(
                    f"optimizations.{name} is not supported for provider: synthetic; "
                    "ignoring it for this teacher.",
                    UserWarning,
                    stacklevel=2,
                )
        return {
            "repair_invalid_final_answers": False, "recover_earlier_retrievals": False,
            "tuning_prompt": "",
        }
    return effective


def teacher_request_messages(
    messages: list[dict[str, Any]], tuning_prompt: str,
) -> list[dict[str, Any]]:
    """Add guidance to a request copy without changing the saved conversation."""
    if not tuning_prompt:
        return messages
    outbound = [dict(message) for message in messages]
    for message in reversed(outbound):
        if message.get("role") == "user":
            message["content"] = f"{message.get('content') or ''}\n\n{tuning_prompt}"
            return outbound
    raise ValueError("A teacher tuning prompt requires a user message.")


def rejected_trace_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    nested_payload = record.get("payload")
    if isinstance(nested_payload, dict):
        payload = {
            **nested_payload,
            **{key: value for key, value in record.items() if key != "payload"},
        }
    if "cypher_solution" not in payload and "actual_solution" in payload:
        payload["cypher_solution"] = payload["actual_solution"]
    return payload


def tool_call_cypher(tool_call: dict[str, Any]) -> str | None:
    function = tool_call.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    value = arguments.get("cypher_string")
    return str(value).strip() if value is not None else None


def tool_call_count(messages: list[dict[str, Any]]) -> int:
    return sum(len(message.get("tool_calls") or []) for message in messages)


def successful_tool_call_index(messages: list[dict[str, Any]], cypher: str) -> int | None:
    target = cypher.strip()
    call_index = 0
    successful_index = None
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            call_index += 1
            if tool_call_cypher(tool_call) == target:
                successful_index = call_index
    return successful_index


def successful_tool_cypher(messages: list[dict[str, Any]], rows: Any) -> str | None:
    """Recover the query paired with successful feedback matching saved execution rows."""
    for index in range(len(messages) - 1, 0, -1):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        try:
            feedback = json.loads(message.get("content") or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(feedback, dict) or feedback.get("ok") is not True:
            continue
        if feedback.get("rows") != rows:
            continue
        calls = messages[index - 1].get("tool_calls") or []
        if len(calls) == 1 and (
            not message.get("tool_call_id") or calls[0].get("id") == message["tool_call_id"]
        ):
            return tool_call_cypher(calls[0])
    return None


def repaired_messages_with_final_answer(
    messages: list[dict[str, Any]],
    final_answer: str,
) -> list[dict[str, Any]] | None:
    repaired = deepcopy(messages)
    for message in reversed(repaired):
        if (
            message.get("role") == "assistant"
            and message.get("content")
            and not message.get("tool_calls")
        ):
            _, inline_reasoning, status = split_generated_reasoning(str(message["content"]))
            if status != "absent":
                text, kind = reasoning_text(message)
                message["reasoning"] = "\n".join(
                    dict.fromkeys(part for part in (text, inline_reasoning) if part)
                )
                message["reasoning_kind"] = kind or "text"
            message["content"] = f"<answer>{final_answer}</answer>"
            return repaired
    return None


def repair_rejected_invalid_answer_trace(
    record: dict[str, Any],
) -> AgentTraceRecord | None:
    trace = rejected_trace_payload(record)
    if trace.get("reason") != "invalid_answer_format":
        return None

    validation = dict(trace.get("validation") or {})
    if validation.get("executable") is False:
        return None

    real_solution = trace.get("real_solution")
    cypher_solution = trace.get("cypher_solution")
    cypher = str(trace.get("cypher") or "").strip()
    question = trace.get("question")
    messages = trace.get("messages") or []
    contract_checks: dict[str, Any] = {}
    if trace.get("validation_contract"):
        from cypher_3dsg_agent_distillation.cypher.result_validation import (
            compare_answers,
            final_grounding_check,
            validate_record_result,
        )

        rows = validation.get("rows")
        if rows is None or validation.get("rows_truncated"):
            return None
        cypher = cypher or successful_tool_cypher(messages, rows) or ""
        outcome = validate_record_result(trace, rows, cypher, validation.get("columns"))
        if outcome["status"] != "pass":
            return None
        cypher_solution = outcome.get("solution")
        answer = compare_answers(
            str(cypher_solution), str(real_solution), trace["validation_contract"],
            trace.get("answer_validation"),
        )
        grounded = final_grounding_check(trace, cypher_solution, outcome)
        if answer["match"] is not True or grounded["match"] is not True:
            return None
        contract_checks = {
            "result_validation": outcome,
            "semantic_check": outcome.get("semantic_check"),
            "answer_check": answer,
            "grounding_check": grounded,
        }
    if (
        real_solution is None
        or cypher_solution is None
        or not cypher
        or not question
        or not messages
    ):
        return None

    output_type = str(
        trace.get("output_type") or (trace.get("metadata") or {}).get("output_type") or "SLDP"
    )
    if output_type.upper() != "SLDP":
        return None
    valid_cypher_format, cypher_matches_real = answer_matches_with_heracles(
        str(cypher_solution),
        str(real_solution),
        output_type,
        answer_validation=trace.get("answer_validation"),
        solution_formats=trace.get("solution_formats") or trace.get("solution_format"),
    )
    if not contract_checks and (not valid_cypher_format or not cypher_matches_real):
        return None

    repaired_messages = repaired_messages_with_final_answer(messages, str(cypher_solution))
    if repaired_messages is None:
        return None

    raw_tool_call_count = tool_call_count(repaired_messages)
    successful_index = successful_tool_call_index(repaired_messages, cypher)
    if successful_index is None and raw_tool_call_count:
        successful_index = raw_tool_call_count

    metadata = dict(trace.get("metadata") or {})
    metadata.update(
        {
            "generation_mode": metadata.get("generation_mode", "tool_conversation"),
            "seed_uid": trace.get("seed_uid"),
            "source_trace_uid": trace.get("uid"),
            "source_trace_status": "rejected",
            "source_rejection_reason": "invalid_answer_format",
            "final_answer_repaired": True,
            "repair_type": "replace_invalid_final_answer_with_cypher_solution",
            "original_final_answer": trace.get("final_answer"),
            "tool_call_count": raw_tool_call_count,
            "successful_tool_call_index": successful_index,
        }
    )
    validation.update(
        {
            **contract_checks,
            "static_valid": True,
            "executable": True,
            "solution_match": True,
            "cypher_solution_match": True,
            "final_answer_match": True,
            "final_answer_cypher_match": True,
            "tool_result_match": True,
            "final_answer_format_valid": True,
            "final_answer_domain_valid": True,
            "answer_grounded": True,
            "error": None,
        }
    )

    uid = str(trace.get("uid") or trace.get("seed_uid") or "rejected_trace")
    return AgentTraceRecord.model_validate(
        {
            "uid": f"{uid}_repaired",
            "scene_id": trace.get("scene_id"),
            "question": question,
            "query_type": trace.get("query_type", "unknown"),
            "task_family": trace.get("task_family", "cypher_qa"),
            "output_type": output_type,
            "experiment_profile": trace.get("experiment_profile"),
            "messages": repaired_messages,
            "gold_cypher": cypher,
            "expected_solution": real_solution,
            "real_solution": real_solution,
            "solution_formats": trace.get("solution_formats", []),
            "answer_validation": trace.get("answer_validation"),
            "grounding": trace.get("grounding"),
            "validation_contract": trace.get("validation_contract"),
            "cypher_solution": cypher_solution,
            "final_answer": str(cypher_solution),
            "split_group": trace.get("split_group"),
            "teacher_model": trace.get("teacher_model", DEFAULT_TEACHER_MODEL),
            "validation": validation,
            "read_only": bool(trace.get("read_only", True)),
            "tags": trace.get("tags", []),
            "metadata": metadata,
        }
    )


def recover_rejected_earlier_retrieval(record: dict[str, Any]) -> AgentTraceRecord | None:
    """Retain the conversation, accepting only a proven, correct earlier retrieval."""
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import answer_inner_text
    from cypher_3dsg_agent_distillation.agents.trace_evidence import (
        EARLIER_RETRIEVAL_POLICY,
        select_earlier_retrieval,
        tool_executions,
    )
    from cypher_3dsg_agent_distillation.cypher.result_validation import (
        compare_answers,
        final_grounding_check,
        validate_record_result,
    )
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import result_hash

    trace = rejected_trace_payload(record)
    if trace.get("reason") not in {
        "invalid_result_shape", "mapping_unresolved", "mismatch", "final_answer_cypher_mismatch",
    }:
        return None
    messages = trace.get("messages") or []
    if (not trace.get("validation_contract") or not messages
            or messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls")):
        return None
    selected = select_earlier_retrieval(
        messages, trace["validation_contract"],
        (trace.get("metadata") or {}).get("seed_gold_cypher"),
    )
    if selected is None:
        return None
    final = answer_inner_text(messages[-1].get("content"))
    expected = trace.get("real_solution", trace.get("expected_solution"))
    if final is None or expected is None:
        return None
    execution = selected.execution
    # Correctness and final grounding are checked only after deterministic selection.
    outcome = validate_record_result(trace, execution.rows, execution.query, execution.columns)
    answer = compare_answers(
        final, str(expected), trace["validation_contract"], trace.get("answer_validation"),
    )
    grounded = final_grounding_check(trace, final, outcome)
    if outcome["status"] != "pass" or answer["match"] is not True or grounded["match"] is not True:
        return None
    executions = tool_executions(messages)
    metadata = {
        **dict(trace.get("metadata") or {}),
        "seed_uid": trace.get("seed_uid"),
        "source_trace_uid": trace.get("uid"),
        "source_trace_status": "rejected",
        "source_rejection_reason": trace["reason"],
        "earlier_retrieval_recovered": True,
        "retrieval_evidence_policy": EARLIER_RETRIEVAL_POLICY,
        "grounding_tool_call_index": execution.call_index,
        "diagnostic_tool_call_indices": sorted(selected.diagnostics),
        "diagnostic_tool_call_reasons": {str(k): v for k, v in selected.diagnostics.items()},
        "tool_call_count": tool_call_count(messages),
        "successful_tool_call_index": executions[-1].call_index,
        "cypher_solution": outcome["solution"],
    }
    validation = {
        **(trace.get("validation") or {}),
        "static_valid": True, "static_issues": [], "executable": True,
        "rows": execution.rows, "columns": execution.columns,
        "rows_truncated": False, "rows_omitted": False,
        "returns_results": bool(execution.rows), "result_hash": result_hash(execution.rows),
        "elapsed_ms": None, "notifications": [], "row_limit": None,
        "result_validation": outcome, "semantic_check": outcome.get("semantic_check"),
        "answer_check": answer, "grounding_check": grounded,
        "solution_match": True, "cypher_solution_match": True, "tool_result_match": True,
        "final_answer_match": True, "final_answer_format_valid": True,
        "final_answer_domain_valid": True, "final_answer_cypher_match": True,
        "answer_grounded": True, "error": None,
    }
    return AgentTraceRecord.model_validate({
        "uid": f"{trace.get('uid') or trace.get('seed_uid')}_recovered",
        "scene_id": trace.get("scene_id"), "question": trace["question"],
        "query_type": trace.get("query_type", "unknown"),
        "task_family": trace.get("task_family", "cypher_qa"),
        "output_type": trace.get("output_type", "SLDP"),
        "experiment_profile": trace.get("experiment_profile"),
        "messages": deepcopy(messages), "gold_cypher": execution.query,
        "expected_solution": expected, "real_solution": expected,
        "solution_formats": trace.get("solution_formats", []),
        "answer_validation": trace.get("answer_validation"), "grounding": trace.get("grounding"),
        "validation_contract": trace["validation_contract"],
        "cypher_solution": outcome["solution"], "final_answer": final,
        "split_group": trace.get("split_group") or split_group_for_record(
            trace.get("scene_id"), trace.get("query_type", "unknown"),
            metadata.get("seed_gold_cypher") or execution.query,
            trace.get("task_family", "cypher_qa"),
        ),
        "teacher_model": trace.get("teacher_model"),
        "validation": validation, "read_only": bool(trace.get("read_only", True)),
        "tags": trace.get("tags", []), "metadata": metadata,
    })
