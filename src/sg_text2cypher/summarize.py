from __future__ import annotations

import argparse
from collections import Counter

from rich.console import Console
from rich.table import Table

from sg_text2cypher.dataset import read_jsonl


def command(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    total = len(records)
    counters = Counter()
    by_task: dict[str, Counter] = {}

    for record in records:
        validation = record.get("validation", {})
        task_type = record.get("task_type", "unknown")
        by_task.setdefault(task_type, Counter())
        for key in ["static_valid", "syntax_error", "timeout", "executable", "returns_results"]:
            value = bool(validation.get(key, False))
            if value:
                counters[key] += 1
                by_task[task_type][key] += 1
        if validation.get("error"):
            counters["errors"] += 1
            by_task[task_type]["errors"] += 1

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

    task_table = Table(title="By Task Type")
    task_table.add_column("Task")
    task_table.add_column("N", justify="right")
    task_table.add_column("Executable", justify="right")
    task_table.add_column("Syntax Error", justify="right")
    for task_type, task_counter in sorted(by_task.items()):
        n = sum(1 for record in records if record.get("task_type", "unknown") == task_type)
        task_table.add_row(
            task_type,
            str(n),
            f"{task_counter['executable'] / n:.2%}" if n else "0.00%",
            f"{task_counter['syntax_error'] / n:.2%}" if n else "0.00%",
        )
    console.print(task_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Cypher validation results.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
