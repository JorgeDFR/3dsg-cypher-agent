from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.agents.agent_records import DEFAULT_AGENT_TOOL, AgentTraceRecord
from cypher_3dsg_agent_distillation.core.cli_output import print_breakdown, print_summary
from cypher_3dsg_agent_distillation.core.config import ensure_parent, load_data_config
from cypher_3dsg_agent_distillation.core.io import read_jsonl
from cypher_3dsg_agent_distillation.cypher.evaluation import build_summary as build_cypher_summary


def saved_execution(record):
    """Return saved evidence under the trace's verified selection policy."""
    from cypher_3dsg_agent_distillation.agents.trace_evidence import (
        marked_earlier_execution,
        uses_earlier_retrieval,
    )

    payload = record.model_dump()
    if uses_earlier_retrieval(payload):
        return marked_earlier_execution(payload)
    query = None
    latest = None
    for message in record.messages:
        if message.tool_calls:
            query = None
            for call in message.tool_calls:
                if call.function.name == DEFAULT_AGENT_TOOL:
                    query = call.function.arguments.get("cypher_string")
        if message.role == "tool" and query:
            try:
                payload = json.loads(message.content or "")
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict) and payload.get("ok") is not False:
                rows = payload.get("rows", payload.get("result"))
                if isinstance(rows, list):
                    latest = (query, rows, payload.get("columns"))
            elif isinstance(payload, list):
                latest = (query, payload, None)
    # Only reuse saved execution when its exact query is recorded or is the
    # validated successful query retained on a generated trace.
    extras = record.validation.model_dump()
    if (
        latest is not None and latest[2] is None
        and latest[0] == record.gold_cypher and latest[1] == extras.get("rows")
        and not extras.get("rows_truncated") and not extras.get("rows_omitted")
    ):
        latest = (latest[0], latest[1], extras.get("columns"))
    if latest is None and extras.get("executable") and isinstance(extras.get("rows"), list):
        latest = (record.gold_cypher, extras["rows"], extras.get("columns"))
    return latest


def recompute_trace_validation(record):
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import answer_inner_text
    from cypher_3dsg_agent_distillation.agents.student_evaluation import (
        score_final_answer_generation,
    )
    from cypher_3dsg_agent_distillation.cypher.result_validation import (
        final_grounding_check,
        validate_record_result,
    )

    payload = record.model_dump()
    execution = saved_execution(record)
    result = (
        validate_record_result(payload, execution[1], execution[0], execution[2])
        if execution
        else {
            "status": "not_checked",
            "match": None,
            "issues": ["No saved successful execution evidence"],
        }
    )
    final = score_final_answer_generation(payload, record.final_answer_text or "")
    grounded = final_grounding_check(payload, answer_inner_text(record.final_answer_text), result)
    return {
        **final,
        "result_validation": result,
        "cypher_solution_match": result["match"],
        "tool_result_match": result["match"],
        "answer_grounded": grounded["match"],
        "grounding_check": grounded,
    }


def evaluate_trace_record(record: AgentTraceRecord) -> dict[str, Any]:
    tool_calls = record.tool_calls
    tool_names = [call.function.name for call in tool_calls]
    valid_cypher_argument = record.is_refusal or bool(record.first_cypher_argument)
    correct_tool_name = record.is_refusal or DEFAULT_AGENT_TOOL in tool_names
    hallucinated_tool = any(name not in record.tools for name in tool_names)
    native_tool_call_parse_success = record.is_refusal or bool(tool_calls)

    final_answer = record.final_answer_text or ""
    if record.is_refusal:
        refusal_correct = "WRITE_QUERY_NOT_ALLOWED" in final_answer
    else:
        refusal_correct = "WRITE_QUERY_NOT_ALLOWED" not in final_answer
    final_answer_match = (
        record.validation.final_answer_match
        if record.validation.final_answer_match is not None
        else record.validation.solution_match
    )

    recomputed = recompute_trace_validation(record) if record.validation_contract else {}
    return {
        **recomputed,
        "uid": record.uid,
        "task_type": record.task_type,
        "task_family": record.task_family,
        "output_type": record.output_type,
        "split_group": record.split_group,
        "native_tool_call_parse_success": native_tool_call_parse_success,
        "correct_tool_name": correct_tool_name,
        "valid_cypher_argument": valid_cypher_argument,
        "static_read_only_safe": bool(record.validation.static_valid),
        "cypher_execution_success": bool(record.validation.executable),
        "final_answer_match": recomputed.get("final_answer_match", bool(final_answer_match)),
        "n_tool_calls": len(tool_calls),
        "hallucinated_tool": hallucinated_tool,
        "refusal": record.is_refusal,
        "refusal_correct": refusal_correct,
    }


def build_agent_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counters = Counter()
    by_task: dict[str, Counter] = {}
    by_family: dict[str, Counter] = {}
    for record in records:
        task_type = record.get("task_type") or "unknown"
        task_family = record.get("task_family") or "cypher_qa"
        by_task.setdefault(task_type, Counter())
        by_family.setdefault(task_family, Counter())
        for key in [
            "native_tool_call_parse_success",
            "correct_tool_name",
            "valid_cypher_argument",
            "static_read_only_safe",
            "cypher_execution_success",
            "final_answer_match",
            "hallucinated_tool",
            "refusal",
            "refusal_correct",
        ]:
            if record.get(key):
                counters[key] += 1
                by_task[task_type][key] += 1
                by_family[task_family][key] += 1
        counters["n_tool_calls"] += int(record.get("n_tool_calls", 0))
        by_task[task_type]["n_tool_calls"] += int(record.get("n_tool_calls", 0))
        by_family[task_family]["n_tool_calls"] += int(record.get("n_tool_calls", 0))

    total = len(records)
    validation_counts = Counter(
        (r.get("result_validation") or {"status": "not_checked"})["status"] for r in records
    )
    return {
        "result_validation_status_counts": dict(validation_counts),
        "total": total,
        "metrics": dict(counters),
        "rates": {
            key: counters[key] / total if total else 0.0
            for key in [
                "native_tool_call_parse_success",
                "correct_tool_name",
                "valid_cypher_argument",
                "static_read_only_safe",
                "cypher_execution_success",
                "final_answer_match",
                "hallucinated_tool",
                "refusal_correct",
            ]
        },
        "average_tool_calls": counters["n_tool_calls"] / total if total else 0.0,
        "by_task_type": {
            task_type: {
                "total": sum(1 for record in records if record.get("task_type") == task_type),
                "metrics": dict(counter),
            }
            for task_type, counter in sorted(by_task.items())
        },
        "by_task_family": {
            task_family: {
                "total": sum(
                    1
                    for record in records
                    if (record.get("task_family") or "cypher_qa") == task_family
                ),
                "metrics": dict(counter),
            }
            for task_family, counter in sorted(by_family.items())
        },
    }


def evaluate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_trace_record(AgentTraceRecord.model_validate(record)) for record in records]


def write_report(evaluated: list[dict[str, Any]], output: str | Path) -> None:
    report = {
        "summary": build_agent_summary(evaluated),
        "cypher_summary_compatible": build_cypher_summary(
            [
                {
                    "query_type": record["task_type"],
                    "static_valid": record["static_read_only_safe"],
                    "executable": record["cypher_execution_success"],
                    "solution_match": record["final_answer_match"],
                    "status": "pass" if record["final_answer_match"] else "agent_failure",
                }
                for record in evaluated
            ]
        ),
        "records": evaluated,
    }
    output_path = ensure_parent(output)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def command(args: argparse.Namespace) -> None:
    data_config = load_data_config(
        args.data_config, dataset_version=args.dataset_version
    )
    split_path = data_config["splits"][args.split]
    report_path = Path(data_config["test_report_path"]).with_name(
        f"{args.split}.agent.report.json"
    )
    evaluated = evaluate_records(read_jsonl(split_path))
    write_report(evaluated, report_path)
    summary = build_agent_summary(evaluated)
    metrics = summary["metrics"]
    print_summary(
        "Agent traces evaluated",
        [
            ("Split", args.split),
            ("Dataset version", data_config["dataset_version"]),
            ("Records", summary["total"]),
            (
                "Final-answer matches",
                f"{metrics.get('final_answer_match', 0)}/{summary['total']} "
                f"({summary['rates']['final_answer_match']:.1%})",
            ),
            (
                "Executable Cypher",
                f"{metrics.get('cypher_execution_success', 0)}/{summary['total']} "
                f"({summary['rates']['cypher_execution_success']:.1%})",
            ),
            ("Average tool calls", f"{summary['average_tool_calls']:.2f}"),
            ("Report", report_path),
        ],
    )
    print_breakdown(
        "Results by question type",
        ("Question type", "Records", "Final-answer matches"),
        (
            (
                task_type,
                task_summary["total"],
                task_summary["metrics"].get("final_answer_match", 0),
            )
            for task_type, task_summary in summary["by_task_type"].items()
        ),
        right_align=(1, 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent trace records.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument(
        "--split",
        choices=["train", "eval", "test", "challenge"],
        default="train",
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
