from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.agent_records import (
    DEFAULT_AGENT_TOOL,
    DEFAULT_TEACHER_MODEL,
    AgentTraceRecord,
)
from cypher_3dsg_agent_distillation.agent_rendering import RenderView, render_trace
from cypher_3dsg_agent_distillation.config import ensure_parent, load_data_config
from cypher_3dsg_agent_distillation.cypher_seed_records import (
    load_raw_seed_records_from_config,
    prepare_seed_records,
)
from cypher_3dsg_agent_distillation.experiment_profiles import (
    load_experiment_profile_from_config,
    profile_seed_defaults,
)
from cypher_3dsg_agent_distillation.heracles_validation import answer_matches_with_heracles
from cypher_3dsg_agent_distillation.io import read_jsonl, write_jsonl


def read_agent_jsonl(path: str | Path) -> list[AgentTraceRecord]:
    return [AgentTraceRecord.model_validate(record) for record in read_jsonl(path)]


def cap_by_task_type(
    records: list[AgentTraceRecord],
    max_per_task_type: int | None,
) -> list[AgentTraceRecord]:
    if max_per_task_type is None:
        return records
    counts: Counter[str] = Counter()
    capped = []
    for record in records:
        if counts[record.task_type] >= max_per_task_type:
            continue
        capped.append(record)
        counts[record.task_type] += 1
    return capped


def read_json_array(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array at {resolved}.")
    return data


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
    if (
        real_solution is None
        or cypher_solution is None
        or not cypher
        or not question
        or not messages
    ):
        return None

    output_type = str(
        trace.get("output_type")
        or (trace.get("metadata") or {}).get("output_type")
        or "SLDP"
    )
    valid_cypher_format, cypher_matches_real = answer_matches_with_heracles(
        str(cypher_solution),
        str(real_solution),
        output_type,
        answer_validation=trace.get("answer_validation"),
    )
    if not valid_cypher_format or not cypher_matches_real:
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
            "static_valid": True,
            "executable": True,
            "solution_match": True,
            "cypher_solution_match": True,
            "final_answer_match": True,
            "final_answer_cypher_match": True,
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
            "messages": repaired_messages,
            "gold_cypher": cypher,
            "expected_solution": real_solution,
            "real_solution": real_solution,
            "solution_formats": trace.get("solution_formats", []),
            "answer_validation": trace.get("answer_validation"),
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


def repair_rejected_invalid_answer_traces(
    records: list[dict[str, Any]],
) -> list[AgentTraceRecord]:
    repaired, _counts = repair_rejected_invalid_answer_traces_with_counts(records)
    return repaired


def repair_rejected_invalid_answer_traces_with_counts(
    records: list[dict[str, Any]],
) -> tuple[list[AgentTraceRecord], Counter[str]]:
    repaired = []
    counts: Counter[str] = Counter()
    for record in records:
        payload = rejected_trace_payload(record)
        reason = str(payload.get("reason") or "unknown")
        counts[f"rejected_reason:{reason}"] += 1
        trace = repair_rejected_invalid_answer_trace(record)
        if trace is not None:
            repaired.append(trace)
            counts["repaired_invalid_answer_format"] += 1
    counts["rejected_total"] = len(records)
    return repaired, counts


def read_student_trace_records_with_counts(
    config: dict[str, Any],
    *,
    include_repaired_rejected: bool,
) -> tuple[list[AgentTraceRecord], Counter[str]]:
    records = read_agent_jsonl(
        config.get("teacher_traces_jsonl_path") or config["traces_jsonl_path"]
    )
    counts: Counter[str] = Counter({"accepted_teacher_traces": len(records)})
    if include_repaired_rejected:
        rejected_records = read_json_array(config["rejected_traces_path"])
        repaired_records, repair_counts = repair_rejected_invalid_answer_traces_with_counts(
            rejected_records
        )
        records.extend(repaired_records)
        counts.update(repair_counts)
    return records, counts


def read_student_trace_records(
    config: dict[str, Any],
    *,
    include_repaired_rejected: bool,
) -> list[AgentTraceRecord]:
    records, _counts = read_student_trace_records_with_counts(
        config,
        include_repaired_rejected=include_repaired_rejected,
    )
    return records


def optimize_student_trace(record: AgentTraceRecord) -> AgentTraceRecord:
    raw_tool_call_count = int(record.metadata.get("tool_call_count") or len(record.tool_calls))
    if raw_tool_call_count <= 1 or record.metadata.get("incomplete_conversation"):
        return record

    successful_index = record.metadata.get("successful_tool_call_index")
    if not isinstance(successful_index, int) or successful_index < 1:
        return record

    messages = [message.model_dump(by_alias=True, exclude_none=True) for message in record.messages]
    prefix = [message for message in messages if message.get("role") in {"system", "user"}]
    assistant_index = None
    call_count = 0
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            call_count += 1
            if call_count == successful_index:
                assistant_index = index
                break
    if assistant_index is None:
        return record

    assistant_message = messages[assistant_index]
    tool_call = assistant_message["tool_calls"][0]
    tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
    tool_result = None
    for message in messages[assistant_index + 1 :]:
        if message.get("role") != "tool":
            continue
        if tool_call_id and message.get("tool_call_id") != tool_call_id:
            continue
        tool_result = message
        break
    if tool_result is None:
        return record

    final_answer = None
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content") and not message.get(
            "tool_calls"
        ):
            final_answer = message
            break
    if final_answer is None:
        return record

    payload = record.model_dump(by_alias=True, exclude_none=True)
    payload["messages"] = prefix + [assistant_message, tool_result, final_answer]
    payload["metadata"] = {
        **record.metadata,
        "student_optimized": True,
        "raw_tool_call_count": raw_tool_call_count,
        "dropped_tool_call_count": raw_tool_call_count - 1,
    }
    payload["tools"] = [DEFAULT_AGENT_TOOL]
    return AgentTraceRecord.model_validate(payload)


def optimize_student_traces(records: list[AgentTraceRecord]) -> list[AgentTraceRecord]:
    optimized, _counts = optimize_student_traces_with_counts(records)
    return optimized


def optimize_student_traces_with_counts(
    records: list[AgentTraceRecord],
) -> tuple[list[AgentTraceRecord], Counter[str]]:
    optimized = []
    counts: Counter[str] = Counter()
    for record in records:
        optimized_record = optimize_student_trace(record)
        optimized.append(optimized_record)
        if optimized_record is not record and optimized_record.metadata.get("student_optimized"):
            counts["pruned_failed_tool_attempts"] += 1
            counts["dropped_tool_calls"] += int(
                optimized_record.metadata.get("dropped_tool_call_count") or 0
            )
    return optimized, counts


SplitResult = tuple[
    list[AgentTraceRecord],
    list[AgentTraceRecord],
    list[AgentTraceRecord],
    list[AgentTraceRecord],
]


def validate_split_sizes(train_size: float, eval_size: float, test_size: float) -> None:
    sizes = {
        "train_size": train_size,
        "eval_size": eval_size,
        "test_size": test_size,
    }
    for name, value in sizes.items():
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}.")
    total = train_size + eval_size + test_size
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "train_size, eval_size, and test_size must sum to 1.0, "
            f"got {total:.6g}."
        )


def grouped_split_records(
    records: list[AgentTraceRecord],
    *,
    train_size: float,
    eval_size: float,
    test_size: float,
    seed: int,
    challenge_task_types: set[str] | None = None,
) -> SplitResult:
    validate_split_sizes(train_size, eval_size, test_size)
    challenge_task_types = challenge_task_types or set()
    challenge = [record for record in records if record.task_type in challenge_task_types]
    main_records = [record for record in records if record.task_type not in challenge_task_types]

    train: list[AgentTraceRecord] = []
    eval_records: list[AgentTraceRecord] = []
    test: list[AgentTraceRecord] = []
    split_lists = {"train": train, "eval": eval_records, "test": test}
    split_sizes = {"train": train_size, "eval": eval_size, "test": test_size}
    records_by_task_type: dict[str, list[AgentTraceRecord]] = defaultdict(list)
    for record in main_records:
        records_by_task_type[record.task_type].append(record)

    record_level_task_types: set[str] = set()
    for task_type in sorted(records_by_task_type):
        task_records = records_by_task_type[task_type]
        task_groups: dict[str, list[AgentTraceRecord]] = defaultdict(list)
        for record in task_records:
            task_groups[record.split_group or "unknown"].append(record)

        rng = random.Random(f"{seed}:{task_type}")
        group_items = list(task_groups.values())
        rng.shuffle(group_items)
        active_split_count = sum(size > 0 for size in split_sizes.values())

        if len(group_items) >= active_split_count:
            target_counts = stratified_split_counts(len(task_records), split_sizes)
            assignments = balanced_group_assignments(group_items, target_counts, split_sizes)
            task_splits = {"train": [], "eval": [], "test": []}
            for group_records, split_name in zip(group_items, assignments, strict=True):
                task_splits[split_name].extend(group_records)
            if rebalance_task_splits(task_splits, target_counts):
                record_level_task_types.add(task_type)
                warnings.warn(
                    f"Task type `{task_type}` cannot reach the exact "
                    f"{train_size:.3g}/{eval_size:.3g}/{test_size:.3g} split while "
                    "keeping every Cypher-equivalent group intact. Moving the minimum "
                    "number of question variants needed to meet the exact per-type counts.",
                    UserWarning,
                    stacklevel=2,
                )
            for split_name in ("train", "eval", "test"):
                split_lists[split_name].extend(task_splits[split_name])
            continue

        # A task such as a no-variable template can have several wordings but only
        # one Cypher group. Split those wordings so the task is still measurable in
        # eval and test, and explicitly exempt only that task from group/Cypher checks.
        shuffled_records = list(task_records)
        rng.shuffle(shuffled_records)
        record_counts = stratified_split_counts(len(shuffled_records), split_sizes)
        if sum(count > 0 for count in record_counts.values()) > len(group_items):
            record_level_task_types.add(task_type)
            warnings.warn(
                f"Task type `{task_type}` has {len(shuffled_records)} questions but only "
                f"{len(group_items)} leakage-safe group(s). Splitting question variants "
                "across train/eval/test so this low-count task remains represented.",
                UserWarning,
                stacklevel=2,
            )
        offset = 0
        for split_name in ("train", "eval", "test"):
            count = record_counts[split_name]
            split_lists[split_name].extend(shuffled_records[offset : offset + count])
            offset += count

    output_rng = random.Random(seed)
    for split_records in split_lists.values():
        output_rng.shuffle(split_records)

    assert_no_split_leakage(
        split_lists,
        allow_group_leakage_task_types=record_level_task_types,
    )
    return train, eval_records, test, challenge


def stratified_split_counts(
    total: int,
    split_sizes: dict[str, float],
) -> dict[str, int]:
    """Apportion one task type while retaining eval/test representation."""
    names = ("train", "eval", "test")
    counts = {name: 0 for name in names}
    active = [name for name in names if split_sizes[name] > 0]
    if total <= 0 or not active:
        return counts

    if total < len(active):
        # With two examples, prioritize both held-out sets as requested. A singleton
        # remains in train because no three-way representation is possible.
        if total == 1 and "train" in active:
            counts["train"] = 1
            return counts
        priority = [name for name in ("eval", "test", "train") if name in active]
        for name in priority[:total]:
            counts[name] += 1
        return counts

    quotas = {name: total * split_sizes[name] for name in active}
    for name in active:
        counts[name] = math.floor(quotas[name])
    remaining = total - sum(counts.values())
    remainder_order = sorted(
        active,
        key=lambda name: (quotas[name] - counts[name], split_sizes[name]),
        reverse=True,
    )
    for name in remainder_order[:remaining]:
        counts[name] += 1

    for name in active:
        if counts[name] > 0:
            continue
        donors = [candidate for candidate in active if counts[candidate] > 1]
        if not donors:
            continue
        donor = max(donors, key=lambda candidate: counts[candidate] - quotas[candidate])
        counts[donor] -= 1
        counts[name] += 1
    return counts


def rebalance_task_splits(
    task_splits: dict[str, list[AgentTraceRecord]],
    target_counts: dict[str, int],
) -> bool:
    """Move the fewest records necessary to make per-task split counts exact."""
    changed = False
    for receiver in ("eval", "test", "train"):
        while len(task_splits[receiver]) < target_counts[receiver]:
            donors = [
                name
                for name in ("train", "eval", "test")
                if len(task_splits[name]) > target_counts[name]
            ]
            if not donors:
                raise RuntimeError("Unable to rebalance task splits to requested counts.")
            donor = max(
                donors,
                key=lambda name: len(task_splits[name]) - target_counts[name],
            )
            task_splits[receiver].append(task_splits[donor].pop())
            changed = True
    return changed


def balanced_group_assignments(
    groups: list[list[AgentTraceRecord]],
    target_counts: dict[str, int],
    split_sizes: dict[str, float],
) -> tuple[str, ...]:
    """Find the grouped assignment closest to per-task record targets."""
    split_names = ("train", "eval", "test")
    active = [name for name in split_names if split_sizes[name] > 0]
    bit = {"train": 1, "eval": 2, "test": 4}
    required_mask = sum(bit[name] for name in active)

    # State tracks held-out record counts and which splits have at least one group.
    states: dict[tuple[int, int, int], tuple[str, ...]] = {(0, 0, 0): ()}
    for group in groups:
        group_size = len(group)
        next_states: dict[tuple[int, int, int], tuple[str, ...]] = {}
        for (eval_count, test_count, mask), assignments in states.items():
            for split_name in active:
                next_eval = eval_count + (group_size if split_name == "eval" else 0)
                next_test = test_count + (group_size if split_name == "test" else 0)
                key = (next_eval, next_test, mask | bit[split_name])
                next_states.setdefault(key, (*assignments, split_name))
        states = next_states

    total = sum(len(group) for group in groups)
    eligible = [
        (state, assignments)
        for state, assignments in states.items()
        if state[2] & required_mask == required_mask
    ]

    def score(item: tuple[tuple[int, int, int], tuple[str, ...]]) -> tuple[int, int, int]:
        (eval_count, test_count, _mask), _assignments = item
        actual = {
            "train": total - eval_count - test_count,
            "eval": eval_count,
            "test": test_count,
        }
        deviations = [abs(actual[name] - target_counts[name]) for name in split_names]
        return (sum(deviations), max(deviations), abs(eval_count - test_count))

    return min(eligible, key=score)[1]


def values_by(
    records: list[AgentTraceRecord],
    field: str,
    excluded_task_types: set[str] | None = None,
) -> set[str]:
    excluded_task_types = excluded_task_types or set()
    values = set()
    for record in records:
        if field != "uid" and record.task_type in excluded_task_types:
            continue
        if field == "uid":
            values.add(record.uid)
        elif field == "split_group":
            values.add(record.split_group or "")
        elif field == "gold_cypher":
            values.add(record.gold_cypher.strip())
        else:
            raise ValueError(f"Unsupported leakage field: {field}")
    return values


def assert_no_split_leakage(
    splits: dict[str, list[AgentTraceRecord]],
    *,
    allow_group_leakage_task_types: set[str] | None = None,
) -> None:
    allowed = allow_group_leakage_task_types or set()
    for field in ("uid", "split_group", "gold_cypher"):
        names = list(splits)
        for index, left_name in enumerate(names):
            left = values_by(splits[left_name], field, allowed)
            for right_name in names[index + 1 :]:
                overlap = left & values_by(splits[right_name], field, allowed)
                if overlap:
                    sample = sorted(overlap)[:3]
                    raise ValueError(
                        f"Split leakage by {field} between {left_name} and {right_name}: {sample}"
                    )


def write_rendered_split(
    records: list[AgentTraceRecord],
    path: str | Path,
    view: str | RenderView,
) -> None:
    write_jsonl([render_trace(record, view) for record in records], path)


def record_task_type_counts(records: list[AgentTraceRecord]) -> Counter[str]:
    return Counter(record.task_type for record in records)


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def split_report_output_path(config: dict[str, Any], output: str | None = None) -> Path:
    if output:
        return Path(output)
    return Path(config["test_report_path"]).with_name("split.report.json")


def counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def optimization_kinds(record: AgentTraceRecord) -> list[str]:
    kinds = []
    if record.metadata.get("student_optimized"):
        kinds.append("pruned_failed_tool_attempts")
    if record.metadata.get("final_answer_repaired"):
        kinds.append("repaired_invalid_answer_format")
    return kinds


def optimized_question_payload(record: AgentTraceRecord, split_name: str) -> dict[str, Any]:
    metadata = record.metadata
    return {
        "uid": record.uid,
        "scene_id": record.scene_id,
        "split": split_name,
        "task_type": record.task_type,
        "question": record.question,
        "split_group": record.split_group,
        "gold_cypher": record.gold_cypher,
        "expected_solution": record.expected_solution,
        "final_answer": record.final_answer,
        "optimization_kinds": optimization_kinds(record),
        "student_optimized": bool(metadata.get("student_optimized")),
        "final_answer_repaired": bool(metadata.get("final_answer_repaired")),
        "raw_tool_call_count": metadata.get("raw_tool_call_count")
        or metadata.get("tool_call_count"),
        "dropped_tool_call_count": int(metadata.get("dropped_tool_call_count") or 0),
        "successful_tool_call_index": metadata.get("successful_tool_call_index"),
        "seed_uid": metadata.get("seed_uid"),
        "source_trace_uid": metadata.get("source_trace_uid"),
        "source_trace_status": metadata.get("source_trace_status"),
        "source_rejection_reason": metadata.get("source_rejection_reason"),
        "repair_type": metadata.get("repair_type"),
        "original_final_answer": metadata.get("original_final_answer"),
    }


def split_records_payload(
    split_records: dict[str, list[AgentTraceRecord]],
) -> dict[str, Any]:
    payload = {}
    for split_name, records in split_records.items():
        payload[split_name] = {
            "questions": len(records),
            "task_type_counts": counter_to_sorted_dict(record_task_type_counts(records)),
            "optimized_questions": sum(1 for record in records if optimization_kinds(record)),
            "uids": [record.uid for record in records],
        }
    return payload


def split_report_payload(
    *,
    data_config: dict[str, Any],
    render_view: str,
    source_counts: Counter[str],
    optimization_counts: Counter[str],
    cap_dropped_counts: Counter[str],
    split_records: dict[str, list[AgentTraceRecord]],
    split_paths: dict[str, str],
    optimize_enabled: bool,
) -> dict[str, Any]:
    optimized_questions = []
    for split_name in ("train", "eval", "test", "challenge"):
        for record in split_records[split_name]:
            if optimization_kinds(record):
                optimized_questions.append(optimized_question_payload(record, split_name))

    return {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "scene_id": data_config.get("scene_id"),
            "render_view": render_view,
            "optimize_student_traces": optimize_enabled,
            "max_per_task_type": data_config.get("max_per_task_type"),
            "split_config": data_config.get("split", {}),
        },
        "paths": {
            "teacher_traces_jsonl_path": data_config.get("teacher_traces_jsonl_path"),
            "rejected_traces_path": data_config.get("rejected_traces_path"),
            "splits": split_paths,
        },
        "source_counts": counter_to_sorted_dict(source_counts),
        "optimization_counts": counter_to_sorted_dict(optimization_counts),
        "cap_dropped_counts": counter_to_sorted_dict(cap_dropped_counts),
        "splits": split_records_payload(split_records),
        "optimized_questions": optimized_questions,
    }


def write_split_report(report: dict[str, Any], output: str | Path) -> Path:
    output_path = ensure_parent(output)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def split_report_lines(
    *,
    source_counts: Counter[str],
    optimization_counts: Counter[str],
    cap_dropped_counts: Counter[str],
    split_records: dict[str, list[AgentTraceRecord]],
    split_paths: dict[str, str],
    optimize_enabled: bool,
) -> list[str]:
    lines = [
        "Dataset split summary:",
        f"  accepted teacher traces loaded: {source_counts['accepted_teacher_traces']}",
    ]
    if optimize_enabled:
        lines.extend(
            [
                (
                    "  rejected traces scanned: "
                    f"{source_counts['rejected_total']} "
                    f"(invalid_answer_format="
                    f"{source_counts['rejected_reason:invalid_answer_format']})"
                ),
                "  optimizations:",
                (
                    "    final answer repairs from invalid_answer_format rejects: "
                    f"{source_counts['repaired_invalid_answer_format']}"
                ),
                (
                    "    pruned failed tool attempts: "
                    f"{optimization_counts['pruned_failed_tool_attempts']} traces, "
                    f"{optimization_counts['dropped_tool_calls']} tool calls dropped"
                ),
            ]
        )
    else:
        lines.extend(
            [
                "  rejected traces scanned: 0 (optimization disabled)",
                "  optimizations: disabled",
            ]
        )

    if cap_dropped_counts:
        lines.append(f"  dropped by max_per_task_type: {format_counter(cap_dropped_counts)}")

    lines.append("  split counts:")
    for split_name in ("train", "eval", "test", "challenge"):
        records = split_records[split_name]
        lines.append(
            f"    {split_name}: {len(records)} questions "
            f"({format_counter(record_task_type_counts(records))}) -> "
            f"{split_paths[split_name]}"
        )
    return lines


def split_command(args: argparse.Namespace) -> None:
    config = load_data_config(args.data_config)
    optimize_enabled = args.optimize_student_traces or bool(
        config.get("optimize_student_traces", False)
    )
    records, source_counts = read_student_trace_records_with_counts(
        config,
        include_repaired_rejected=optimize_enabled,
    )
    optimization_counts: Counter[str] = Counter()
    if optimize_enabled:
        records, optimization_counts = optimize_student_traces_with_counts(records)
    records_before_cap = records
    records = cap_by_task_type(records, config.get("max_per_task_type"))
    cap_dropped_counts = record_task_type_counts(records_before_cap) - record_task_type_counts(
        records
    )
    split_config = config.get("split", {})
    train, eval_records, test, challenge = grouped_split_records(
        records,
        train_size=float(split_config.get("train_size", 0.8)),
        eval_size=float(split_config.get("eval_size", 0.1)),
        test_size=float(split_config.get("test_size", 0.1)),
        seed=int(split_config.get("seed", 42)),
        challenge_task_types=set(split_config.get("challenge_task_types", [])),
    )
    paths = config["splits"]
    write_rendered_split(train, paths["train"], args.render_view)
    write_rendered_split(eval_records, paths["eval"], args.render_view)
    write_rendered_split(test, paths["test"], args.render_view)
    write_rendered_split(challenge, paths["challenge"], args.render_view)
    split_records = {
        "train": train,
        "eval": eval_records,
        "test": test,
        "challenge": challenge,
    }
    report_path = write_split_report(
        split_report_payload(
            data_config=config,
            render_view=args.render_view,
            source_counts=source_counts,
            optimization_counts=optimization_counts,
            cap_dropped_counts=cap_dropped_counts,
            split_records=split_records,
            split_paths=paths,
            optimize_enabled=optimize_enabled,
        ),
        split_report_output_path(config, args.report_output),
    )
    print(
        "\n".join(
            split_report_lines(
                source_counts=source_counts,
                optimization_counts=optimization_counts,
                cap_dropped_counts=cap_dropped_counts,
                split_records=split_records,
                split_paths=paths,
                optimize_enabled=optimize_enabled,
            )
            + [f"  report: {report_path}"]
        )
    )


def prepare_seeds_command(args: argparse.Namespace) -> None:
    data_config = load_data_config(args.data_config)
    profile = load_experiment_profile_from_config(data_config)
    if profile is None:
        raise ValueError(
            "Seed preparation requires a valid `experiment_profile_path` in the data config."
        )
    seed_defaults = profile_seed_defaults(profile)
    raw_records = load_raw_seed_records_from_config(data_config)
    prepared = prepare_seed_records(
        raw_records=raw_records,
        task_family=seed_defaults.get("task_family", "cypher_qa"),
        output_type=seed_defaults.get("output_type", "SLDP"),
        metadata_defaults=seed_defaults.get("metadata", {}),
    )
    write_jsonl(prepared, data_config["seed_records_jsonl_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare agent trace datasets.")
    subparsers = parser.add_subparsers(required=True)

    seeds_parser = subparsers.add_parser("prepare-seeds")
    seeds_parser.add_argument("--data-config", default="configs/data.yaml")
    seeds_parser.set_defaults(func=prepare_seeds_command)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--data-config", default="configs/data.yaml")
    split_parser.add_argument(
        "--render-view",
        choices=[view.value for view in RenderView],
        default=RenderView.OLLAMA_NATIVE.value,
    )
    split_parser.add_argument("--optimize-student-traces", action="store_true")
    split_parser.add_argument(
        "--report-output",
        default=None,
        help=(
            "Optional JSON split report path. Defaults to "
            "data/3dsg/{scene_id}/reports/split.report.json."
        ),
    )
    split_parser.set_defaults(func=split_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
