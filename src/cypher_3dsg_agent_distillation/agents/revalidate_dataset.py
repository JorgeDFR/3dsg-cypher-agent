"""Regrade a completed teacher dataset from saved evidence, without model requests."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

from cypher_3dsg_agent_distillation.agents.agent_evaluation import saved_execution
from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord
from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
    TraceDecision,
    build_interaction_trace,
    configured_trace_decision,
    debug_trace_record,
    optimize_teacher_attempts,
    trace_token_length_report,
)
from cypher_3dsg_agent_distillation.agents.reasoning import without_controlled_repair
from cypher_3dsg_agent_distillation.agents.teacher_pilot import (
    TeacherRunJournal,
    attempt_payload,
    pilot_report,
)
from cypher_3dsg_agent_distillation.agents.trace_evidence import tool_executions
from cypher_3dsg_agent_distillation.core.config import PROJECT_ROOT, load_data_config, load_yaml
from cypher_3dsg_agent_distillation.core.io import read_jsonl, write_jsonl
from cypher_3dsg_agent_distillation.core.versioning import (
    atomic_write_yaml,
    file_fingerprint,
    iso_now,
    record_stage,
    sha256_file,
    version_id,
)
from cypher_3dsg_agent_distillation.cypher.validation_audit import load_templates
from cypher_3dsg_agent_distillation.cypher.validation_config import implementation_fingerprint
from cypher_3dsg_agent_distillation.cypher.validation_refresh import refresh_contracts


def revalidate_attempt(item, seed, *, teacher_model, max_tool_calls, tokenizer, max_length):
    """Use complete paired evidence; keep interrupted/no-tool conversations rejected."""
    original = deepcopy(item)
    seed = without_controlled_repair(seed)
    original["debug_record"]["validation_contract"] = deepcopy(seed["validation_contract"])
    original["debug_record"]["grounding"] = deepcopy(seed.get("grounding"))
    record = None
    if item.get("record"):
        payload = deepcopy(item["record"])
        payload["validation_contract"] = seed["validation_contract"]
        payload["grounding"] = seed.get("grounding")
        payload.setdefault("metadata", {}).update(
            {
                key: value
                for key, value in seed.get("metadata", {}).items()
                if key.startswith("validation_")
            }
        )
        record = AgentTraceRecord.model_validate(payload)
        if record.is_refusal:
            decision = TraceDecision(
                True,
                record,
                item["reason"],
                {"messages": payload["messages"]},
            )
        else:
            execution = saved_execution(record)
    else:
        debug = item["debug_record"]
        messages = debug.get("messages") or []
        # Incomplete provider output and protocol violations are not repaired by
        # inventing final answers or dropping unexecuted tool calls.
        if item["reason"] not in {
            "invalid_result_shape",
            "mapping_unresolved",
            "mismatch",
            "final_answer_cypher_mismatch",
            "invalid_answer_format",
        }:
            return original
        executions = [e for e in tool_executions(messages) if e.complete]
        if not executions:
            return original
        last = executions[-1]
        record = build_interaction_trace(
            seed,
            messages=deepcopy(messages),
            teacher_model=teacher_model,
            cypher=last.query,
            validation=debug.get("validation") or {},
            max_tool_calls=max_tool_calls,
            tool_call_count=sum(len(m.get("tool_calls") or []) for m in messages),
            successful_tool_call_index=last.call_index,
            cypher_solution=None,
        )
        execution = saved_execution(record)
    if not record.is_refusal:
        validation = record.validation.model_dump()
        evidence = (
            {**validation, "executable": True, "rows": execution[1], "columns": execution[2]}
            if execution
            and not validation.get("rows_truncated")
            and not validation.get("rows_omitted")
            else {}
        )
        if execution:
            record.gold_cypher = execution[0]
        decision = configured_trace_decision(
            seed,
            record,
            {"messages": [m.model_dump(exclude_none=True) for m in record.messages]},
            evidence,
        )
    if decision.accepted:
        decision.reason = item["reason"] if item["accepted"] else "accepted_revalidated"
        record.metadata["saved_evidence_revalidated"] = True
    result = attempt_payload(
        seed,
        decision,
        attempt_id=item["attempt_id"],
        tokenizer=tokenizer,
        max_length=max_length,
    )
    record_json = record.model_dump(by_alias=True, exclude_none=True)
    result["debug_record"] = debug_trace_record(seed, decision, record_json=record_json)
    if item.get("optimization"):
        result["optimization"] = deepcopy(item["optimization"])
    return result


def revalidate_dataset(data_config, *, tokenizer, apply=False):
    dataset = Path(data_config["dataset_dir"])
    traces = Path(data_config["teacher_metadata_path"]).parent
    metadata = load_yaml(traces / "metadata.yaml")
    if metadata.get("status") != "complete":
        raise ValueError("Saved-evidence revalidation requires a completed teacher run.")
    if (traces / "pending_request.json").exists():
        raise ValueError("Reconcile the pending provider request before revalidation.")
    requests_hash = sha256_file(traces / "requests.jsonl")
    templates = load_templates([q["question_templates_path"] for q in data_config["question_sets"]])
    seeds = refresh_contracts(
        read_jsonl(data_config["seed_records_jsonl_path"]),
        templates,
    )
    questions_doc = load_yaml(data_config["raw_questions_path"])
    questions = refresh_contracts(
        questions_doc["questions"],
        templates,
    )
    questions_doc["questions"] = questions
    seed_lookup = {s["uid"]: s for s in seeds}
    original_attempts = read_jsonl(traces / "attempts.jsonl")
    requests = read_jsonl(traces / "requests.jsonl")
    student = metadata["student_config"]["snapshot"]
    teacher = metadata["teacher_config"]["snapshot"]
    max_length = int(student["training"]["max_length"])
    updated = [
        revalidate_attempt(
            item,
            seed_lookup[item["seed_uid"]],
            teacher_model=teacher["model"],
            max_tool_calls=int(teacher["max_tool_calls"]),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for item in original_attempts
    ]

    with tempfile.TemporaryDirectory(prefix=".revalidate-", dir=dataset) as temporary:
        staging = Path(temporary)
        stage_traces = staging / traces.relative_to(dataset)
        stage_traces.mkdir(parents=True)
        write_jsonl(updated, stage_traces / "attempts.jsonl")
        journal = TeacherRunJournal(stage_traces)
        optimize_teacher_attempts(
            journal,
            seeds=seeds,
            optimizations=teacher.get("optimizations") or {},
            tokenizer=tokenizer,
            max_length=max_length,
            teacher_model=teacher["model"],
        )
        updated = journal.attempts
        changes = [
            {
                "attempt_id": before["attempt_id"],
                "accepted_before": before["accepted"],
                "accepted_after": after["accepted"],
                "reason_before": before["reason"],
                "reason_after": after["reason"],
            }
            for before, after in zip(original_attempts, updated, strict=True)
            if (before["accepted"], before["reason"]) != (after["accepted"], after["reason"])
        ]
        report = {
            "created_at": iso_now(),
            "implementation": implementation_fingerprint(),
            "teacher_requests_added": 0,
            "request_journal_sha256": requests_hash,
            "accepted_before": sum(i["accepted"] for i in original_attempts),
            "accepted_after": sum(i["accepted"] for i in updated),
            "rejected_after": sum(not i["accepted"] for i in updated),
            "changes": changes,
            "rejection_reasons": dict(Counter(i["reason"] for i in updated if not i["accepted"])),
        }
        if not apply:
            return report
        backup = dataset / "revalidation_backups" / version_id()
        report["backup_directory"] = str(backup)
        records = [i["record"] for i in updated if i["accepted"]]
        write_jsonl(records, stage_traces / "teacher_traces.jsonl")
        for name, accepted in (("accepted.json", True), ("rejected.json", False)):
            (stage_traces / name).write_text(
                json.dumps(
                    [i["debug_record"] for i in updated if i["accepted"] is accepted],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        lengths = trace_token_length_report(
            records,
            tokenizer,
            max_length=max_length,
            include_reasoning=student["training"].get("include_reasoning", False),
        )
        lengths = {
            **load_yaml(traces / "token_length_report.yaml"),
            **lengths,
            "revalidated_at": report["created_at"],
        }
        atomic_write_yaml(lengths, stage_traces / "token_length_report.yaml")
        pilot = {
            **load_yaml(traces / "pilot_report.yaml"),
            **pilot_report(seeds, updated, requests),
        }
        pilot["revalidation"] = {k: v for k, v in report.items() if k != "changes"}
        atomic_write_yaml(pilot, stage_traces / "pilot_report.yaml")
        (stage_traces / "matched_retained_uids.json").write_text(
            json.dumps(pilot["matched_retained_uids"], indent=2) + "\n",
        )
        write_jsonl(
            seeds,
            staging / Path(data_config["seed_records_jsonl_path"]).relative_to(dataset),
        )
        atomic_write_yaml(
            questions_doc,
            staging / Path(data_config["raw_questions_path"]).relative_to(dataset),
        )
        metadata.setdefault("original_generation_counts", deepcopy(metadata["counts"]))
        metadata["counts"].update(
            {
                "accepted": report["accepted_after"],
                "rejected": report["rejected_after"],
                "accepted_before_optimizations": pilot["optimizations"]["accepted_before"],
                "rejected_before_optimizations": pilot["optimizations"]["rejected_before"],
                "repaired": pilot["optimizations"]["repaired"],
                "recovered": pilot["optimizations"]["recovered"],
            }
        )
        metadata.update(
            updated_at=report["created_at"],
            optimizations=pilot["optimizations"],
            revalidation=report,
            token_lengths={
                k: v for k, v in lengths.items() if k not in {"traces", "student_config"}
            },
        )
        # Fingerprints describe the final relative paths, not the staging directory.
        for name, filename in {
            "pilot_report": "pilot_report.yaml",
            "attempt_journal": "attempts.jsonl",
            "teacher_traces": "teacher_traces.jsonl",
            "accepted": "accepted.json",
            "rejected": "rejected.json",
            "token_length_report": "token_length_report.yaml",
        }.items():
            fp = file_fingerprint(stage_traces / filename)
            fp["path"] = file_fingerprint(traces / filename, relative_to=PROJECT_ROOT)["path"]
            metadata["outputs"][name] = fp
        atomic_write_yaml(metadata, stage_traces / "metadata.yaml")
        report_path = staging / "reports/revalidation.report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        outputs = [p for p in staging.rglob("*") if p.is_file()]
        # Publish the teacher status last; retain a byte-for-byte backup for rollback.
        outputs.sort(key=lambda p: p == stage_traces / "metadata.yaml")
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(data_config["metadata_path"], backup / "metadata.yaml")
        for path in outputs:
            relative = path.relative_to(staging)
            destination = dataset / relative
            if destination.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, saved)
        try:
            for path in outputs:
                destination = dataset / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
            assert sha256_file(traces / "requests.jsonl") == requests_hash
            record_stage(
                data_config["metadata_path"],
                "revalidate_teacher_traces",
                arguments={"saved_evidence_only": True, "refresh_contracts": True},
                outputs={
                    "report": dataset / "reports/revalidation.report.json",
                    "traces": traces / "teacher_traces.jsonl",
                },
                project_root=PROJECT_ROOT,
                extra={"implementation": report["implementation"]},
            )
        except BaseException:
            for path in outputs:
                relative = path.relative_to(staging)
                destination, saved = dataset / relative, backup / relative
                if saved.exists():
                    shutil.copy2(saved, destination)
                else:
                    destination.unlink(missing_ok=True)
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and replace dataset artifacts.",
    )
    args = parser.parse_args()
    config = load_data_config(args.data_config, dataset_version=args.dataset_version)
    teacher = load_yaml(config["teacher_metadata_path"])
    from cypher_3dsg_agent_distillation.training.train import load_tokenizer

    result = revalidate_dataset(
        config,
        tokenizer=load_tokenizer(teacher["student_config"]["snapshot"]),
        apply=args.apply,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
