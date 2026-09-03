from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from cypher_3dsg_agent_distillation.core.io import read_jsonl
from cypher_3dsg_agent_distillation.cypher.evaluation import print_report_summary


def command(args: argparse.Namespace) -> None:
    input_text = Path(args.input).read_text(encoding="utf-8").lstrip()
    if input_text.startswith("{"):
        try:
            report = json.loads(input_text)
        except json.JSONDecodeError:
            report = None
        if isinstance(report, dict) and "summary" in report and "records" in report:
            print_report_summary(report)
            return

    records = read_jsonl(args.input)
    total = len(records)
    counters = Counter()
    by_query_type: dict[str, Counter] = {}

    for record in records:
        validation = record.get("validation", {})
        query_type = record.get("query_type", "unknown")
        by_query_type.setdefault(query_type, Counter())
        for key in ["static_valid", "syntax_error", "timeout", "executable", "returns_results"]:
            value = bool(validation.get(key, False))
            if value:
                counters[key] += 1
                by_query_type[query_type][key] += 1
        if validation.get("error"):
            counters["errors"] += 1
            by_query_type[query_type]["errors"] += 1

    console = Console()
    table = Table(title=f"Validation Summary: {args.input}")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")
    for metric in [
        "static_valid",
        "executable",
        "returns_results",
        "syntax_error",
        "timeout",
        "errors",
    ]:
        count = counters[metric]
        rate = (count / total) if total else 0
        table.add_row(metric, str(count), f"{rate:.2%}")
    console.print(table)

    query_table = Table(title="By Query Type")
    query_table.add_column("Query Type")
    query_table.add_column("N", justify="right")
    query_table.add_column("Executable", justify="right")
    query_table.add_column("Syntax Error", justify="right")
    for query_type, query_counter in sorted(by_query_type.items()):
        n = sum(1 for record in records if record.get("query_type", "unknown") == query_type)
        query_table.add_row(
            query_type,
            str(n),
            f"{query_counter['executable'] / n:.2%}" if n else "0.00%",
            f"{query_counter['syntax_error'] / n:.2%}" if n else "0.00%",
        )
    console.print(query_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Cypher validation results.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
