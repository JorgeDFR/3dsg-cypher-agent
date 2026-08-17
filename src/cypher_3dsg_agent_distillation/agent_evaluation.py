from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.agent_records import DEFAULT_AGENT_TOOL, AgentTraceRecord
from cypher_3dsg_agent_distillation.config import ensure_parent, load_data_config
from cypher_3dsg_agent_distillation.evaluation import build_summary as build_cypher_summary
from cypher_3dsg_agent_distillation.io import read_jsonl


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

    return {
        "uid": record.uid,
        "task_type": record.task_type,
        "split_group": record.split_group,
        "native_tool_call_parse_success": native_tool_call_parse_success,
        "correct_tool_name": correct_tool_name,
        "valid_cypher_argument": valid_cypher_argument,
        "static_read_only_safe": bool(record.validation.static_valid),
        "cypher_execution_success": bool(record.validation.executable),
        "final_answer_match": bool(final_answer_match),
        "n_tool_calls": len(tool_calls),
        "hallucinated_tool": hallucinated_tool,
        "refusal": record.is_refusal,
        "refusal_correct": refusal_correct,
    }


def build_agent_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counters = Counter()
    by_task: dict[str, Counter] = {}
    for record in records:
        task_type = record.get("task_type") or "unknown"
        by_task.setdefault(task_type, Counter())
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
        counters["n_tool_calls"] += int(record.get("n_tool_calls", 0))
        by_task[task_type]["n_tool_calls"] += int(record.get("n_tool_calls", 0))

    total = len(records)
    return {
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
    data_config = load_data_config(args.data_config)
    split_path = data_config["splits"][args.split]
    report_path = Path(data_config["test_report_path"]).with_name(
        f"{args.split}.agent.report.json"
    )
    evaluated = evaluate_records(read_jsonl(split_path))
    write_report(evaluated, report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent trace records.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "eval", "test", "challenge"],
        default="test",
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
