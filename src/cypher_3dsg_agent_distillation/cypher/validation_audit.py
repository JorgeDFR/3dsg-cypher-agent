"""Explicit catalog audits, historical migration and saved-output rescoring."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from cypher_3dsg_agent_distillation.cypher.result_validation import (
    compare_answers,
    final_grounding_check,
    validate_record_result,
)
from cypher_3dsg_agent_distillation.cypher.task_semantics import recover_bindings
from cypher_3dsg_agent_distillation.cypher.validation_config import (
    ValidationContract,
    contract_for_record,
    fingerprint,
    implementation_fingerprint,
    load_question_catalog,
    resolve_contract,
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


def migrate_records(records, templates, *, refresh_contracts=False):
    """Explicit selection plus verified gold query/bindings; never lookup by type alone."""
    catalog = {t["query_type"]: t for t in templates}
    implementation = implementation_fingerprint()
    migrated, issues = [], []
    for original in records:
        record = deepcopy(original)
        try:
            contract = contract_for_record(record)
            if contract and not refresh_contracts:
                ValidationContract.model_validate(contract)
            else:
                query_type = record.get("query_type", record.get("task_type"))
                template = catalog.get(query_type)
                if template is None:
                    raise ValueError("No selected source template")
                # Teacher traces retain the original gold query in metadata.
                query = (
                    (record.get("metadata") or {}).get("seed_gold_cypher")
                    or record.get("cypher")
                    or record.get("gold_cypher")
                )
                if not query:
                    raise ValueError("No saved trusted query")
                variables = recover_bindings(template["cypher"], query)
                previous_contract = contract
                contract = resolve_contract(template["validation_contract"], variables)
                if record.get("answer_validation"):
                    contract["answer"]["comparison"].update(record["answer_validation"])
                # Scores are scene observations, not template declarations. Reuse
                # them only when the scoring rule and resolved limit are unchanged.
                for section in ("result", "answer"):
                    old = (previous_contract or {}).get(section, {}).get("comparison", {})
                    new = contract[section]["comparison"]
                    if all(old.get(key) == new.get(key) for key in (
                        "mode", "score_source", "limit",
                    )) and old.get("scores") is not None:
                        new["scores"] = deepcopy(old["scores"])
                ValidationContract.model_validate(contract)
                record["validation_contract"] = contract
                if record.get("grounding"):
                    record["grounding"]["validation_contract"] = deepcopy(contract)
                record.setdefault("metadata", {})["validation_provenance"] = template[
                    "validation_provenance"
                ]
            meta = record.setdefault("metadata", {})
            meta["validation_contract_hash"] = fingerprint(contract)
            meta["validation_migration"] = {
                "status": "migrated",
                "source_record_hash": fingerprint(original),
                "implementation": implementation,
            }
            expected = next(
                (
                    record[k]
                    for k in ("expected_solution", "real_solution", "solution")
                    if record.get(k) is not None
                ),
                None,
            )
            if (
                expected is not None
                and compare_answers(str(expected), str(expected), contract)["status"]
                == "invalid_expected_solution"
            ):
                raise ValueError(
                    "Historical expected solution violates selected contract; label unchanged"
                )
            if contract.get("goal"):
                from cypher_3dsg_agent_distillation.cypher.pddl_goals import render_goal_spec

                expected_roles = (record.get("grounding") or {}).get("expected")
                if expected_roles is None:
                    raise ValueError(
                        "Missing historical expected grounding; cannot reconstruct oracle"
                    )
                rendered = render_goal_spec(contract["goal"], expected_roles) or ""
                if (
                    expected is not None
                    and not compare_answers(rendered, str(expected), contract)["match"]
                ):
                    raise ValueError(
                        "Historical goal disagrees with configured goal tree; label unchanged"
                    )
        except (ValueError, KeyError, TypeError) as exc:
            diagnostic = {
                "uid": record.get("uid"),
                "status": "migration_unresolved",
                "issue": str(exc),
            }
            record.setdefault("metadata", {})["validation_migration"] = diagnostic
            issues.append(diagnostic)
        migrated.append(record)
    return migrated, issues


def saved_result(record):
    """Read actual saved execution evidence; absence is never interpreted as empty rows."""
    from cypher_3dsg_agent_distillation.agents.trace_evidence import (
        marked_earlier_execution,
        uses_earlier_retrieval,
    )

    tool = (
        record.get("tool_call_from_prompt")
        or record.get("tool_call_check")
        or record.get("tool_call_generation")
        or {}
    )
    validation = tool.get("validation") or record.get("validation") or {}
    query = tool.get("generated_cypher") or (record.get("generated") or {}).get("cypher")
    rows = validation.get("rows")
    if not query:
        if uses_earlier_retrieval(record):
            # Canonical traces and accepted debug records share the same proof.
            # Invalid markers must never fall back to a cached grade.
            return marked_earlier_execution(record)
        from cypher_3dsg_agent_distillation.agents.agent_evaluation import saved_execution
        from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord

        if record.get("messages"):
            try:
                execution = saved_execution(AgentTraceRecord.model_validate(record))
                if execution:
                    return execution
            except ValueError:
                pass
        query = record.get("cypher") or record.get("gold_cypher")
    if rows is None:
        rows = (record.get("generated") or {}).get("cypher_tool_output")
    if validation.get("rows_truncated") or validation.get("rows_omitted"):
        return None
    if query and isinstance(rows, list) and validation.get("executable") is not False:
        return query, rows, validation.get("columns")
    return None


def rescore_records(records, sources=None):
    sources = {r["uid"]: r for r in sources or []}
    results = []
    for original in records:
        source = sources.get(original.get("uid"), original)
        query_record = {**source}
        if not contract_for_record(query_record):
            outcome = {
                "status": "not_checked",
                "match": None,
                "issues": ["Explicit contract migration required"],
            }
        elif (query_record.get("metadata") or {}).get("validation_migration", {}).get(
            "status"
        ) == "migration_unresolved":
            outcome = {
                "status": "invalid_expected_solution",
                "match": None,
                "issues": ["Resolve migration diagnostics before rescoring"],
            }
        else:
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
        text = (
            original.get("final_answer_from_trace") or original.get("final_answer_generation") or {}
        ).get("generated_text") or (original.get("generated") or {}).get("final_answer")
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
                for k in ("expected_solution", "real_solution", "solution")
                if query_record.get(k) is not None
            ),
            None,
        )
        contract = contract_for_record(query_record)
        final = (
            compare_answers(answer, str(expected), contract, query_record.get("answer_validation"))
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
    for name in ("audit", "migrate"):
        command = commands.add_parser(name)
        command.add_argument("--templates", nargs="+", default=DEFAULT_CATALOGS)
        if name == "migrate":
            command.add_argument("--records", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--refresh-contracts", action="store_true")
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
    elif args.command == "migrate":
        records, issues = migrate_records(
            load_records(args.records), load_templates(args.templates),
            refresh_contracts=args.refresh_contracts,
        )
        write_new(args.output, records, jsonl=True)
        print(json.dumps({"records": len(records), "issues": len(issues), "diagnostics": issues}))
    elif args.command == "rescore":
        report = rescore_records(
            load_records(args.records),
            load_records(args.source_records) if args.source_records else None,
        )
        write_new(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
