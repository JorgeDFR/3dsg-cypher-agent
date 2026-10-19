from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from cypher_3dsg_agent_distillation.core.config import ensure_parent
from cypher_3dsg_agent_distillation.cypher.templates import format_solution_field
from cypher_3dsg_agent_distillation.graph.heracles_validation import answer_matches_with_heracles
from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
    is_safe_refusal,
    result_hash,
    static_checks,
)


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(format_solution_field(value))
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([<>\[\]{},:])\s*", r"\1", text)
    return text


def answer_matches(
    actual: Any,
    expected: Any,
    output_type: str = "SLDP",
    *,
    answer_validation: dict[str, Any] | None = None,
    solution_formats: Any = None,
) -> bool:
    valid_format, correct = answer_matches_with_heracles(
        str(actual),
        str(expected),
        output_type,
        answer_validation=answer_validation,
        solution_formats=solution_formats,
    )
    return valid_format and correct


def evaluate_records(records, cypher_field, validator, policy):
    from cypher_3dsg_agent_distillation.cypher.result_validation import (
        compare_answers,
        validate_record_result,
    )
    from cypher_3dsg_agent_distillation.cypher.validation_config import contract_for_record

    evaluated = []
    for record in records:
        contract_for_record(record)
        cypher = record.get(cypher_field, "")
        is_refusal = is_safe_refusal(cypher)
        issues = static_checks(cypher, False, policy=policy)
        validation = {
            "executable": False,
            "error": None,
            "syntax_error": False,
            "timeout": False,
            "elapsed_ms": None,
            "rows": [],
        }
        outcome = {"status": "not_checked", "match": None, "solution": None}
        if not issues and is_refusal:
            outcome = {
                **compare_answers(
                    "WRITE_QUERY_NOT_ALLOWED",
                    str(record.get("solution")),
                    contract_for_record(record),
                ),
                "solution": "WRITE_QUERY_NOT_ALLOWED",
            }
            validation["executable"] = True
        elif not issues:
            validation = validator.validate(cypher, execute=True)
            if validation.get("executable") and not any(
                validation.get(key) for key in ("rows_truncated", "rows_omitted")
            ):
                outcome = validate_record_result(
                    record, validation["rows"], cypher, validation.get("columns")
                )
        status = (
            "static_error"
            if issues
            else "execution_error"
            if validation.get("error")
            else outcome["status"]
        )
        rows = validation.get("rows", [])
        evaluated.append(
            {
                "uid": record.get("uid"),
                "query_type": record["query_type"],
                "question": record.get("question"),
                "cypher": cypher,
                "expected_solution": record.get("solution"),
                "actual_solution": outcome.get("solution"),
                "solution_match": outcome["match"],
                "status": status,
                "static_valid": not issues,
                "static_issues": issues,
                "refusal": is_refusal,
                "executed": not issues and not is_refusal,
                **validation,
                "row_count": len(rows),
                "result_hash": result_hash(rows) if validation.get("executable") else None,
                "result_validation": outcome,
            }
        )
    return evaluated


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    counters = Counter()
    by_query_type: dict[str, Counter] = {}
    for record in records:
        query_type = record.get("query_type") or "unknown"
        by_query_type.setdefault(query_type, Counter())
        for key in [
            "static_valid",
            "executed",
            "executable",
            "syntax_error",
            "timeout",
            "solution_match",
            "refusal",
        ]:
            if record.get(key):
                counters[key] += 1
                by_query_type[query_type][key] += 1
        if record.get("error"):
            counters["execution_error"] += 1
            by_query_type[query_type]["execution_error"] += 1
        if record.get("status") == "mismatch":
            counters["solution_mismatch"] += 1
            by_query_type[query_type]["solution_mismatch"] += 1
        if record.get("status") == "static_error":
            counters["static_error"] += 1
            by_query_type[query_type]["static_error"] += 1

    return {
        "total": total,
        "metrics": dict(counters),
        "rates": {
            key: (counters[key] / total if total else 0.0)
            for key in [
                "static_valid",
                "executable",
                "solution_match",
                "execution_error",
                "solution_mismatch",
                "static_error",
            ]
        },
        "by_query_type": {
            query_type: {
                "total": sum(1 for record in records if record.get("query_type") == query_type),
                "metrics": dict(counter),
            }
            for query_type, counter in sorted(by_query_type.items())
        },
    }


def build_report(
    records: list[dict[str, Any]],
    *,
    input_path: str,
    scene_path: str,
    cypher_field: str,
) -> dict[str, Any]:
    return {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "input": input_path,
            "scene": scene_path,
            "cypher_field": cypher_field,
        },
        "summary": build_summary(records),
        "records": records,
    }


def write_report(report: dict[str, Any], output: str | Path) -> None:
    output_path = ensure_parent(output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_report_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    total = summary["total"]
    metrics = Counter(summary.get("metrics", {}))

    console = Console()
    table = Table(title="Cypher Validation Report")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")
    for key, label in [
        ("static_valid", "Static valid"),
        ("executable", "Executed/refusal OK"),
        ("solution_match", "Solution match"),
        ("execution_error", "Execution errors"),
        ("solution_mismatch", "Solution mismatches"),
        ("static_error", "Static errors"),
        ("timeout", "Timeouts"),
        ("syntax_error", "Syntax errors"),
    ]:
        count = metrics[key]
        rate = count / total if total else 0.0
        table.add_row(label, str(count), f"{rate:.2%}")
    console.print(table)

    query_table = Table(title="By Query Type")
    query_table.add_column("Query Type")
    query_table.add_column("N", justify="right")
    query_table.add_column("Exec OK", justify="right")
    query_table.add_column("Match", justify="right")
    query_table.add_column("Errors", justify="right")
    for query_type, payload in summary.get("by_query_type", {}).items():
        n = payload["total"]
        query_metrics = Counter(payload.get("metrics", {}))
        errors = (
            query_metrics["execution_error"]
            + query_metrics["solution_mismatch"]
            + query_metrics["static_error"]
        )
        query_table.add_row(
            query_type,
            str(n),
            f"{query_metrics['executable'] / n:.2%}" if n else "0.00%",
            f"{query_metrics['solution_match'] / n:.2%}" if n else "0.00%",
            str(errors),
        )
    console.print(query_table)
