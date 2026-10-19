"""Explicit catalog audits and saved-output rescoring."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cypher_3dsg_agent_distillation.cypher.result_validation import (
    compare_answers,
    final_grounding_check,
    validate_record_result,
)
from cypher_3dsg_agent_distillation.cypher.validation_config import (
    contract_for_record,
    fingerprint,
    implementation_fingerprint,
    load_question_catalog,
    strict_yaml,
)

DEFAULT_CATALOGS = [
    f"configs/question_templates/{name}_3dsg.yaml" for name in ("cypher_qa", "cypher_pddl")
]


def load_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = strict_yaml(path) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("records", data.get("questions", []))


def load_templates(paths):
    templates = [t for path in paths for t in load_question_catalog(path)["templates"]]
    if len({t["query_type"] for t in templates}) != len(templates):
        raise ValueError("Duplicate query types across selected catalogs")
    return templates


def saved_result(record):
    """Read a current report, paired conversation or standalone execution record."""
    from cypher_3dsg_agent_distillation.agents.trace_evidence import (
        marked_earlier_execution,
        tool_executions,
        uses_earlier_retrieval,
    )

    tool = record.get("tool_call_from_prompt") or {}
    validation = tool.get("validation") or record.get("validation") or {}
    if validation.get("rows_truncated") or validation.get("rows_omitted"):
        return None
    # A student report's generated evidence is independent of its source trace markers.
    if tool:
        query = tool.get("generated_cypher")
        rows = validation.get("rows")
    elif record.get("messages"):
        if uses_earlier_retrieval(record):
            return marked_earlier_execution(record)
        executions = tool_executions(record["messages"])
        if not executions or not executions[-1].complete:
            return None
        execution = executions[-1]
        columns = execution.columns
        if columns is None and validation.get("rows") == execution.rows:
            columns = validation.get("columns")
        return execution.query, execution.rows, columns
    else:
        query = record.get("cypher")
        rows = validation.get("rows")
    if query and isinstance(rows, list) and validation.get("executable") is True:
        return query, rows, validation.get("columns")
    return None


def rescore_records(records, sources=None):
    sources = {r["uid"]: r for r in sources or []}
    results = []
    for original in records:
        source = sources.get(original.get("uid"), original)
        query_record = {**source}
        contract_for_record(query_record)
        evidence = saved_result(original)
        outcome = (
            validate_record_result(query_record, evidence[1], evidence[0], evidence[2])
            if evidence
            else {
                "status": "not_checked",
                "match": None,
                "issues": ["No complete saved execution; fresh execution required"],
            }
        )
        text = (original.get("final_answer_from_trace") or {}).get("generated_text") or (
            original.get("generated") or {}
        ).get("final_answer")
        if text is None and original.get("messages"):
            text = next(
                (
                    m.get("content")
                    for m in reversed(original["messages"])
                    if m["role"] == "assistant" and not m.get("tool_calls")
                ),
                None,
            )
        if text is None and "<answer>" in str(original.get("final_answer") or "").lower():
            text = original["final_answer"]
        from cypher_3dsg_agent_distillation.agents.openrouter_teacher import answer_inner_text

        answer = answer_inner_text(text)
        expected = next(
            (
                query_record[k]
                for k in ("expected_solution", "solution")
                if query_record.get(k) is not None
            ),
            None,
        )
        contract = contract_for_record(query_record)
        final = (
            compare_answers(answer, str(expected), contract)
            if answer is not None and expected is not None and contract
            else {
                "status": "not_checked"
                if text is None or not contract
                else "invalid_answer_format",
                "match": None,
            }
        )
        before = (
            original.get("tool_call_from_prompt")
            or original.get("tool_call_check")
            or original.get("validation")
            or {}
        ).get("cypher_solution_match")
        results.append(
            {
                "uid": original.get("uid"),
                "before_match": before,
                "changed": before is not None and outcome["match"] != before,
                "result_validation": outcome,
                "answer_check": final,
                "grounding_check": final_grounding_check(query_record, answer, outcome),
                "source_hash": fingerprint(original),
                "contract_hash": fingerprint(contract) if contract else None,
            }
        )
    return {
        "implementation": implementation_fingerprint(),
        "records": results,
        "status_counts": dict(Counter(r["result_validation"]["status"] for r in results)),
        "changed_grades": sum(r["changed"] for r in results),
    }


def write_new(path, payload, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects both historical inputs and previous audit output.
    with path.open("x") as handle:
        if jsonl:
            for record in payload:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("audit")
    command.add_argument("--templates", nargs="+", default=DEFAULT_CATALOGS)
    command = commands.add_parser("rescore")
    command.add_argument("--records", required=True)
    command.add_argument("--source-records")
    command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "audit":
        templates = load_templates(args.templates)
        for t in templates:
            from cypher_3dsg_agent_distillation.cypher.query_provenance import projections

            projections(t["cypher"])
        print(
            json.dumps(
                {
                    "status": "pass",
                    "templates": len(templates),
                    "implementation": implementation_fingerprint(),
                }
            )
        )
    elif args.command == "rescore":
        report = rescore_records(
            load_records(args.records),
            load_records(args.source_records) if args.source_records else None,
        )
        write_new(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
