from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.agents.agent_records import (
    AgentTraceRecord,
    normalize_cypher,
)
from cypher_3dsg_agent_distillation.agents.agent_rendering import RenderView, render_trace
from cypher_3dsg_agent_distillation.agents.experiment_profiles import (
    experiment_profile_for_seed,
    load_experiment_profiles_from_config,
    profile_seed_defaults,
)
from cypher_3dsg_agent_distillation.core.cli_output import print_summary
from cypher_3dsg_agent_distillation.core.config import (
    PROJECT_ROOT,
    ensure_parent,
    load_data_config,
    load_yaml,
)
from cypher_3dsg_agent_distillation.core.io import read_jsonl, write_jsonl
from cypher_3dsg_agent_distillation.core.scene_ordering import group_records_by_scene
from cypher_3dsg_agent_distillation.core.versioning import (
    atomic_write_yaml,
    config_record,
    record_stage,
    require_new_outputs,
    serializable_arguments,
    split_fingerprints,
)
from cypher_3dsg_agent_distillation.cypher.cypher_seed_records import (
    load_raw_seed_records_from_config,
    prepare_seed_records,
)


def read_agent_jsonl(path: str | Path) -> list[AgentTraceRecord]:
    return [AgentTraceRecord.model_validate(record) for record in read_jsonl(path)]


def record_task_key(record: AgentTraceRecord) -> str:
    if record.task_family in {"cypher_qa", "cypher_pddl"}:
        return record.task_type
    return f"{record.task_family}/{record.task_type}"


def record_stratum_key(record: AgentTraceRecord) -> str:
    """Identify the family/type stratum used for balancing and leakage rules."""
    return f"{record.task_family}/{record.task_type}"


def split_reference_query(record: AgentTraceRecord) -> str:
    """Split by the task's query, independently of the teacher's implementation."""
    return normalize_cypher(record.metadata.get("seed_gold_cypher") or record.gold_cypher)


def split_record_groups(records: list[AgentTraceRecord]) -> list[list[AgentTraceRecord]]:
    """Connect seed-query equivalents and explicitly shared split groups."""
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    identities = {}
    for index, record in enumerate(records):
        keys = [("query", split_reference_query(record))]
        if record.split_group:
            keys.append(("group", record.split_group))
        for key in keys:
            previous = identities.setdefault(key, index)
            parents[find(index)] = find(previous)
    groups: dict[int, list[AgentTraceRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    return list(groups.values())


def cap_by_task_type(
    records: list[AgentTraceRecord],
    max_per_task_type: int | None,
) -> list[AgentTraceRecord]:
    if max_per_task_type is None:
        return records
    counts: Counter[tuple[str, str]] = Counter()
    capped = []
    for record in records:
        task_key = (record.task_family, record.task_type)
        if counts[task_key] >= max_per_task_type:
            continue
        capped.append(record)
        counts[task_key] += 1
    return capped


def read_student_trace_records_with_counts(
    config: dict[str, Any],
) -> tuple[list[AgentTraceRecord], Counter[str]]:
    """Load finalized generation outputs; splitting never repairs rejected attempts."""
    records = read_agent_jsonl(config["teacher_traces_jsonl_path"])
    counts: Counter[str] = Counter(
        {
            "accepted_teacher_traces": len(records),
            "repaired_invalid_answer_format": sum(
                bool(record.metadata.get("final_answer_repaired")) for record in records
            ),
            "recovered_earlier_retrieval": sum(
                bool(record.metadata.get("earlier_retrieval_recovered")) for record in records
            ),
        }
    )
    return records, counts


def read_student_trace_records(config: dict[str, Any]) -> list[AgentTraceRecord]:
    records, _counts = read_student_trace_records_with_counts(config)
    return records


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
        raise ValueError(f"train_size, eval_size, and test_size must sum to 1.0, got {total:.6g}.")


def grouped_split_records(
    records: list[AgentTraceRecord],
    *,
    train_size: float,
    eval_size: float,
    test_size: float,
    seed: int,
    challenge_task_types: set[str] | None = None,
    scene_order: list[str] | None = None,
    warn_on_leakage: bool = True,
) -> SplitResult:
    validate_split_sizes(train_size, eval_size, test_size)
    challenge_task_types = challenge_task_types or set()
    challenge = [record for record in records if record.task_type in challenge_task_types]
    challenge_uids = {record.uid for record in challenge}
    main_records = [record for record in records if record.uid not in challenge_uids]

    train: list[AgentTraceRecord] = []
    eval_records: list[AgentTraceRecord] = []
    test: list[AgentTraceRecord] = []
    split_lists = {"train": train, "eval": eval_records, "test": test}
    split_sizes = {"train": train_size, "eval": eval_size, "test": test_size}
    records_by_task_type: dict[str, list[AgentTraceRecord]] = defaultdict(list)
    for record in main_records:
        records_by_task_type[record_stratum_key(record)].append(record)

    record_level_task_types: set[str] = set()
    for task_types in cross_task_components(records_by_task_type):
        if len(task_types) > 1:
            component_records = [
                record for task_type in task_types for record in records_by_task_type[task_type]
            ]
            target_counts = {
                task_type: stratified_split_counts(
                    len(records_by_task_type[task_type]), split_sizes
                )
                for task_type in task_types
            }
            grouped_assignments = joint_cypher_group_assignments(
                component_records,
                task_types=task_types,
                target_counts=target_counts,
                split_sizes=split_sizes,
                seed=seed,
            )
            if grouped_assignments is not None:
                for group_records, split_name in grouped_assignments:
                    split_lists[split_name].extend(group_records)
                continue

            if warn_on_leakage:
                warnings.warn(
                    "Cypher-equivalent groups shared by task types "
                    f"{', '.join(task_types)} cannot reach the exact "
                    f"{train_size:.3g}/{eval_size:.3g}/{test_size:.3g} split while "
                    "remaining intact. Falling back to exact per-type counts and "
                    "allowing the minimum necessary shared-query leakage.",
                    UserWarning,
                    stacklevel=2,
                )
            component_splits = {"train": [], "eval": [], "test": []}
            for task_type in task_types:
                task_splits, used_record_level_split = split_one_task_type(
                    task_type,
                    records_by_task_type[task_type],
                    split_sizes=split_sizes,
                    seed=seed,
                    warn=False,
                )
                if used_record_level_split:
                    record_level_task_types.add(task_type)
                for split_name in component_splits:
                    component_splits[split_name].extend(task_splits[split_name])
            consolidate_cross_task_cypher(component_splits)
            rebalance_task_types_after_consolidation(
                component_splits,
                records_by_task_type={
                    task_type: records_by_task_type[task_type] for task_type in task_types
                },
                split_sizes=split_sizes,
                leakage_exemptions=record_level_task_types,
            )
            for split_name in split_lists:
                split_lists[split_name].extend(component_splits[split_name])
            continue

        task_type = task_types[0]
        task_splits, used_record_level_split = split_one_task_type(
            task_type,
            records_by_task_type[task_type],
            split_sizes=split_sizes,
            seed=seed,
            warn=warn_on_leakage,
        )
        if used_record_level_split:
            record_level_task_types.add(task_type)
        for split_name in split_lists:
            split_lists[split_name].extend(task_splits[split_name])

    for records_to_group in (eval_records, test, challenge):
        records_to_group[:] = group_records_by_scene(
            records_to_group,
            seed=seed,
            scene_order=scene_order or (),
        )

    assert_no_split_leakage(
        split_lists,
        allow_group_leakage_task_types=record_level_task_types,
    )
    return train, eval_records, test, challenge


def split_one_task_type(
    task_type: str,
    task_records: list[AgentTraceRecord],
    *,
    split_sizes: dict[str, float],
    seed: int,
    warn: bool = True,
) -> tuple[dict[str, list[AgentTraceRecord]], bool]:
    """Split one task type exactly, breaking groups only when unavoidable."""
    task_splits: dict[str, list[AgentTraceRecord]] = {
        "train": [],
        "eval": [],
        "test": [],
    }
    rng = random.Random(f"{seed}:{task_type}")
    group_items = split_record_groups(task_records)
    rng.shuffle(group_items)
    active_split_count = sum(size > 0 for size in split_sizes.values())

    if len(group_items) >= active_split_count:
        target_counts = stratified_split_counts(len(task_records), split_sizes)
        assignments = balanced_group_assignments(group_items, target_counts, split_sizes)
        for group_records, split_name in zip(group_items, assignments, strict=True):
            task_splits[split_name].extend(group_records)
        used_record_level_split = rebalance_task_splits(task_splits, target_counts)
        if used_record_level_split and warn:
            warnings.warn(
                f"Task type `{task_type}` cannot reach the exact "
                f"{split_sizes['train']:.3g}/{split_sizes['eval']:.3g}/"
                f"{split_sizes['test']:.3g} split while keeping every "
                "Cypher-equivalent group intact. Moving the minimum number of "
                "question variants needed to meet the exact per-type counts.",
                UserWarning,
                stacklevel=2,
            )
        return task_splits, used_record_level_split

    # A task such as a no-variable template can have several wordings but only
    # one Cypher group. Split those wordings so the task is still measurable in
    # eval and test, and explicitly exempt only that task from group/Cypher checks.
    shuffled_records = list(task_records)
    rng.shuffle(shuffled_records)
    record_counts = stratified_split_counts(len(shuffled_records), split_sizes)
    used_record_level_split = sum(count > 0 for count in record_counts.values()) > len(group_items)
    if used_record_level_split and warn:
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
        task_splits[split_name].extend(shuffled_records[offset : offset + count])
        offset += count
    return task_splits, used_record_level_split


def cross_task_components(
    records_by_task_type: dict[str, list[AgentTraceRecord]],
) -> list[tuple[str, ...]]:
    """Return task components connected by normalized-equivalent Cypher."""
    neighbors = {task_type: set() for task_type in records_by_task_type}
    records = [record for group in records_by_task_type.values() for record in group]
    for group in split_record_groups(records):
        task_types = {record_stratum_key(record) for record in group}
        for task_type in task_types:
            neighbors[task_type].update(task_types - {task_type})

    components = []
    remaining = set(neighbors)
    while remaining:
        root = min(remaining)
        stack = [root]
        remaining.remove(root)
        component = []
        while stack:
            task_type = stack.pop()
            component.append(task_type)
            new_neighbors = sorted(neighbors[task_type] & remaining, reverse=True)
            stack.extend(new_neighbors)
            remaining.difference_update(new_neighbors)
        components.append(tuple(sorted(component)))
    return sorted(components)


def joint_cypher_group_assignments(
    records: list[AgentTraceRecord],
    *,
    task_types: tuple[str, ...],
    target_counts: dict[str, dict[str, int]],
    split_sizes: dict[str, float],
    seed: int,
) -> list[tuple[list[AgentTraceRecord], str]] | None:
    """Assign shared Cypher groups while meeting every per-task split target."""
    task_index = {task_type: index for index, task_type in enumerate(task_types)}
    groups = split_record_groups(records)
    rng = random.Random(f"{seed}:{':'.join(task_types)}")
    rng.shuffle(groups)
    # Processing the most strongly coupled groups first keeps the DP state small.
    groups.sort(
        key=lambda group: len({record_stratum_key(record) for record in group}),
        reverse=True,
    )
    vectors = []
    for group in groups:
        vector = [0] * len(task_types)
        for record in group:
            vector[task_index[record_stratum_key(record)]] += 1
        vectors.append(tuple(vector))

    active_splits = tuple(
        split_name for split_name in ("train", "eval", "test") if split_sizes[split_name] > 0
    )
    if not active_splits:
        return None
    base_split = max(
        active_splits,
        key=lambda split_name: sum(
            target_counts[task_type][split_name] for task_type in task_types
        ),
    )
    tracked_splits = tuple(split_name for split_name in active_splits if split_name != base_split)
    width = len(task_types)
    zero_state = (0,) * (len(tracked_splits) * width)
    states: dict[tuple[int, ...], int] = {zero_state: 0}
    processed = [0] * width
    totals = [sum(vector[index] for vector in vectors) for index in range(width)]
    radix = len(active_splits)
    max_states = 250_000

    for vector in vectors:
        processed = [count + increment for count, increment in zip(processed, vector, strict=True)]
        remaining = [total - count for total, count in zip(totals, processed, strict=True)]
        next_states: dict[tuple[int, ...], int] = {}
        for state, code in states.items():
            for choice_index, split_name in enumerate(active_splits):
                candidate = list(state)
                if split_name != base_split:
                    split_offset = tracked_splits.index(split_name) * width
                    for index, increment in enumerate(vector):
                        candidate[split_offset + index] += increment

                possible = True
                for index, task_type in enumerate(task_types):
                    tracked_total = 0
                    for tracked_index, tracked_split in enumerate(tracked_splits):
                        count = candidate[tracked_index * width + index]
                        target = target_counts[task_type][tracked_split]
                        if count > target or count + remaining[index] < target:
                            possible = False
                            break
                        tracked_total += count
                    if not possible:
                        break
                    base_count = processed[index] - tracked_total
                    base_target = target_counts[task_type][base_split]
                    if base_count > base_target or base_count + remaining[index] < base_target:
                        possible = False
                        break
                if possible:
                    candidate_state = tuple(candidate)
                    next_states.setdefault(
                        candidate_state,
                        code * radix + choice_index,
                    )
                    if len(next_states) > max_states:
                        return None
        states = next_states
        if not states:
            return None

    goal_state = tuple(
        target_counts[task_type][split_name]
        for split_name in tracked_splits
        for task_type in task_types
    )
    code = states.get(goal_state)
    if code is None:
        return None
    assignments = []
    for _group in reversed(groups):
        choice_index = code % radix
        assignments.append(active_splits[choice_index])
        code //= radix
    assignments.reverse()
    return list(zip(groups, assignments, strict=True))


def rebalance_task_types_after_consolidation(
    splits: dict[str, list[AgentTraceRecord]],
    *,
    records_by_task_type: dict[str, list[AgentTraceRecord]],
    split_sizes: dict[str, float],
    leakage_exemptions: set[str],
) -> None:
    """Restore exact per-task counts after the grouped fallback consolidation."""
    for task_type, task_records in records_by_task_type.items():
        task_splits = {
            split_name: [
                record for record in split_records if record_stratum_key(record) == task_type
            ]
            for split_name, split_records in splits.items()
        }
        target_counts = stratified_split_counts(len(task_records), split_sizes)
        if not rebalance_task_splits(task_splits, target_counts):
            continue
        leakage_exemptions.add(task_type)
        for split_name in splits:
            splits[split_name][:] = [
                record for record in splits[split_name] if record_stratum_key(record) != task_type
            ]
            splits[split_name].extend(task_splits[split_name])


def consolidate_cross_task_cypher(
    splits: dict[str, list[AgentTraceRecord]],
) -> None:
    """Keep a Cypher-equivalent group in one split even across task families."""
    records = [record for group in splits.values() for record in group]
    split_for_uid = {record.uid: name for name, group in splits.items() for record in group}
    priority = {"train": 2, "eval": 1, "test": 0}
    for group in split_record_groups(records):
        by_split: dict[str, list[AgentTraceRecord]] = defaultdict(list)
        for record in group:
            by_split[split_for_uid[record.uid]].append(record)
        if len(by_split) <= 1 or len({record_stratum_key(r) for r in group}) <= 1:
            continue
        target = max(
            by_split,
            key=lambda name: (len(by_split[name]), priority.get(name, -1)),
        )
        for split_name, records in by_split.items():
            if split_name == target:
                continue
            record_ids = {id(record) for record in records}
            splits[split_name][:] = [
                record for record in splits[split_name] if id(record) not in record_ids
            ]
            splits[target].extend(records)


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
        if field != "uid" and record_stratum_key(record) in excluded_task_types:
            continue
        if field == "uid":
            values.add(record.uid)
        elif field == "split_group":
            values.add(record.split_group or "")
        elif field == "seed_cypher":
            values.add(split_reference_query(record))
        else:
            raise ValueError(f"Unsupported leakage field: {field}")
    return values


def assert_no_split_leakage(
    splits: dict[str, list[AgentTraceRecord]],
    *,
    allow_group_leakage_task_types: set[str] | None = None,
) -> None:
    allowed = allow_group_leakage_task_types or set()
    for field in ("uid", "split_group", "seed_cypher"):
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
    return Counter(record_task_key(record) for record in records)


def record_task_family_counts(records: list[AgentTraceRecord]) -> Counter[str]:
    return Counter(record.task_family for record in records)


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
    if record.metadata.get("final_answer_repaired"):
        kinds.append("repaired_invalid_answer_format")
    if record.metadata.get("earlier_retrieval_recovered"):
        kinds.append("recovered_earlier_retrieval")
    return kinds


def optimized_question_payload(record: AgentTraceRecord, split_name: str) -> dict[str, Any]:
    metadata = record.metadata
    return {
        "uid": record.uid,
        "scene_id": record.scene_id,
        "split": split_name,
        "task_type": record.task_type,
        "task_family": record.task_family,
        "output_type": record.output_type,
        "question": record.question,
        "split_group": record.split_group,
        "gold_cypher": record.gold_cypher,
        "expected_solution": record.expected_solution,
        "final_answer": record.final_answer,
        "optimization_kinds": optimization_kinds(record),
        "final_answer_repaired": bool(metadata.get("final_answer_repaired")),
        "earlier_retrieval_recovered": bool(metadata.get("earlier_retrieval_recovered")),
        "grounding_tool_call_index": metadata.get("grounding_tool_call_index"),
        "diagnostic_tool_call_indices": metadata.get("diagnostic_tool_call_indices"),
        "tool_call_count": metadata.get("tool_call_count"),
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
        scene_sequence = [str(record.scene_id or "unknown") for record in records]
        payload[split_name] = {
            "questions": len(records),
            "task_type_counts": counter_to_sorted_dict(record_task_type_counts(records)),
            "task_family_counts": counter_to_sorted_dict(record_task_family_counts(records)),
            "optimized_questions": sum(1 for record in records if optimization_kinds(record)),
            "uids": [record.uid for record in records],
            "scene_counts": counter_to_sorted_dict(Counter(scene_sequence)),
            "scene_runs": (
                0
                if not scene_sequence
                else 1
                + sum(
                    left != right
                    for left, right in zip(scene_sequence, scene_sequence[1:], strict=False)
                )
            ),
        }
    return payload


def split_report_payload(
    *,
    data_config: dict[str, Any],
    render_view: str,
    source_counts: Counter[str],
    cap_dropped_counts: Counter[str],
    split_records: dict[str, list[AgentTraceRecord]],
    split_paths: dict[str, str],
) -> dict[str, Any]:
    optimized_questions = []
    for split_name in ("train", "eval", "test", "challenge"):
        for record in split_records[split_name]:
            if optimization_kinds(record):
                optimized_questions.append(optimized_question_payload(record, split_name))

    return {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": data_config.get("dataset_id"),
            "scene_ids": [scene["scene_id"] for scene in data_config.get("scenes", [])],
            "dataset_version": data_config.get("dataset_version"),
            "teacher_name": data_config.get("teacher_name"),
            "render_view": render_view,
            "optimization_stage": "generation",
            "max_per_task_type": data_config.get("max_per_task_type"),
            "split_config": data_config.get("split", {}),
        },
        "paths": {
            "teacher_traces_jsonl_path": data_config.get("teacher_traces_jsonl_path"),
            "rejected_traces_path": data_config.get("rejected_traces_path"),
            "splits": split_paths,
        },
        "source_counts": counter_to_sorted_dict(source_counts),
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
    cap_dropped_counts: Counter[str],
    split_records: dict[str, list[AgentTraceRecord]],
    split_paths: dict[str, str],
) -> list[str]:
    lines = [
        "Dataset split summary:",
        f"  accepted teacher traces loaded: {source_counts['accepted_teacher_traces']}",
    ]
    lines.append(
        "  generation final-answer repairs included: "
        f"{source_counts['repaired_invalid_answer_format']}"
    )

    if cap_dropped_counts:
        lines.append(f"  dropped by max_per_task_type: {format_counter(cap_dropped_counts)}")

    lines.append("  split counts:")
    for split_name in ("train", "eval", "test", "challenge"):
        records = split_records[split_name]
        lines.append(f"    {split_name}: {len(records)} traces -> {split_paths[split_name]}")
    return lines


def split_command(args: argparse.Namespace) -> None:
    config = load_data_config(args.data_config, dataset_version=args.dataset_version)
    split_config = config.get("split", {})
    if getattr(args, "split_by_scene", False) or bool(split_config.get("split_by_scene", False)):
        print("Scene-aware splitting is not implemented yet.")
        return
    teacher_metadata = load_yaml(config["teacher_metadata_path"])
    if teacher_metadata.get("status") != "complete":
        raise ValueError("Teacher traces are not complete. Generate traces before splitting.")
    prospective_report = split_report_output_path(config, args.report_output)
    require_new_outputs([*config["splits"].values(), prospective_report], force=args.force)
    records, source_counts = read_student_trace_records_with_counts(config)
    records_before_cap = records
    records = cap_by_task_type(records, config.get("max_per_task_type"))
    cap_dropped_counts = record_task_type_counts(records_before_cap) - record_task_type_counts(
        records
    )
    train, eval_records, test, challenge = grouped_split_records(
        records,
        train_size=float(split_config.get("train_size", 0.8)),
        eval_size=float(split_config.get("eval_size", 0.1)),
        test_size=float(split_config.get("test_size", 0.1)),
        seed=int(split_config.get("seed", 42)),
        challenge_task_types=set(split_config.get("challenge_task_types", [])),
        scene_order=[str(scene["scene_id"]) for scene in config["scenes"]],
        warn_on_leakage=False,
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
            cap_dropped_counts=cap_dropped_counts,
            split_records=split_records,
            split_paths=paths,
        ),
        split_report_output_path(config, args.report_output),
    )
    split_records_metadata, split_fingerprint = split_fingerprints(
        config["splits"], project_root=PROJECT_ROOT
    )
    metadata = record_stage(
        config["metadata_path"],
        "split",
        arguments=serializable_arguments(args),
        outputs={
            **{f"split:{name}": path for name, path in config["splits"].items()},
            "split_report": report_path,
        },
        project_root=PROJECT_ROOT,
        extra={
            "teacher_name": config["teacher_name"],
            "split_fingerprint": split_fingerprint,
            "data_config": config_record(config["data_config_path"], project_root=PROJECT_ROOT),
        },
    )
    metadata["status"] = "complete"
    metadata["split_files"] = split_records_metadata
    metadata["split_fingerprint"] = split_fingerprint
    atomic_write_yaml(metadata, config["metadata_path"])
    summary_rows = [
        ("Accepted traces", source_counts["accepted_teacher_traces"]),
        ("Dataset version", config["dataset_version"]),
        ("Teacher", config["teacher_name"]),
        ("Generation final-answer repairs", source_counts["repaired_invalid_answer_format"]),
        ("Render view", args.render_view),
        (
            "Split traces",
            ", ".join(
                f"{name}={len(split_records[name])}"
                for name in ("train", "eval", "test", "challenge")
            ),
        ),
    ]
    if cap_dropped_counts:
        summary_rows.append(("Dropped by task cap", format_counter(cap_dropped_counts)))
    summary_rows.append(("Report", report_path))
    print_summary("Dataset splits created", summary_rows)


def prepare_seeds_command(args: argparse.Namespace) -> None:
    data_config = load_data_config(args.data_config, dataset_version=args.dataset_version)
    profiles = load_experiment_profiles_from_config(data_config)
    if not profiles:
        raise ValueError("Seed preparation requires at least one experiment profile.")
    default_profile = next(iter(profiles.values()))
    seed_defaults = profile_seed_defaults(default_profile)
    raw_records = load_raw_seed_records_from_config(data_config)
    for raw_record in raw_records:
        experiment_profile_for_seed(
            raw_record.model_dump(by_alias=True, exclude_none=True), profiles
        )
    prepared = prepare_seed_records(
        raw_records=raw_records,
        task_family=seed_defaults.get("task_family", "cypher_qa"),
        output_type=seed_defaults.get("output_type", "SLDP"),
        metadata_defaults=seed_defaults.get("metadata", {}),
    )
    output_path = data_config["seed_records_jsonl_path"]
    require_new_outputs([output_path], force=args.force)
    write_jsonl(prepared, output_path)
    record_stage(
        data_config["metadata_path"],
        "prepare_seeds",
        arguments=serializable_arguments(args),
        outputs={"seed_records": output_path},
        project_root=PROJECT_ROOT,
        extra={
            "data_config": config_record(data_config["data_config_path"], project_root=PROJECT_ROOT)
        },
    )
    counts = Counter(record.query_type for record in prepared)
    family_counts = Counter(record.task_family for record in prepared)
    print_summary(
        "Seed records prepared",
        [
            ("Dataset", data_config["dataset_id"]),
            ("Scenes", len(data_config["scenes"])),
            ("Dataset version", data_config["dataset_version"]),
            ("Records", len(prepared)),
            ("Question types", len(counts)),
            ("Task families", format_counter(family_counts)),
            ("Output", output_path),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare agent trace datasets.")
    subparsers = parser.add_subparsers(required=True)

    seeds_parser = subparsers.add_parser("prepare-seeds")
    seeds_parser.add_argument("--data-config", default="configs/data.yaml")
    seeds_parser.add_argument("--dataset-version", default=None)
    seeds_parser.add_argument("--force", action="store_true")
    seeds_parser.set_defaults(func=prepare_seeds_command)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--data-config", default="configs/data.yaml")
    split_parser.add_argument("--dataset-version", default=None)

    split_parser.add_argument("--force", action="store_true")
    split_parser.add_argument(
        "--render-view",
        choices=[view.value for view in RenderView],
        default=RenderView.OLLAMA_NATIVE.value,
    )
    split_parser.add_argument(
        "--split-by-scene",
        action="store_true",
        help=(
            "Reserve complete scenes for non-training splits. This option is "
            "declared for the multi-scene interface but is not implemented yet."
        ),
    )
    split_parser.add_argument(
        "--report-output",
        default=None,
        help=(
            "Optional JSON split report path. Defaults to "
            "data/datasets/{dataset_id}/{version}/reports/split.report.json."
        ),
    )
    split_parser.set_defaults(func=split_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
