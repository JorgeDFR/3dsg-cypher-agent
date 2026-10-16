"""Reproducible teacher sampling, durable request replay, and cost/coverage estimates."""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.agents.reasoning import trace_reasoning_coverage
from cypher_3dsg_agent_distillation.core.versioning import atomic_write_yaml, mapping_fingerprint


def stratum(seed: dict[str, Any]) -> str:
    return "/".join(
        str(seed.get(key) or "unknown")
        for key in (
            "scene_id",
            "task_family",
            "query_type",
        )
    )


def select_pilot_seeds(
    seeds: list[dict[str, Any]],
    *,
    per_task_type: int = 2,
    random_seed: int = 42,
    extra_per_tag: int = 2,
) -> list[dict[str, Any]]:
    if per_task_type < 1 or extra_per_tag < 0:
        raise ValueError("per_task_type must be positive and extra_per_tag nonnegative.")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        groups[(str(seed.get("task_family")), str(seed.get("query_type")))].append(seed)
    selected = {}
    for key, group in sorted(groups.items()):
        scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for seed in sorted(group, key=lambda item: str(item["uid"])):
            scenes[str(seed.get("scene_id"))].append(seed)
        rng = random.Random(f"{random_seed}:{key}")
        for candidates in scenes.values():
            rng.shuffle(candidates)
        # At least one example per represented scene, even if the requested minimum is 1.
        target = min(len(group), max(per_task_type, len(scenes)))
        while sum(seed["uid"] in selected for seed in group) < target:
            for scene in sorted(scenes):
                if scenes[scene]:
                    seed = scenes[scene].pop()
                    selected[seed["uid"]] = seed
                if sum(seed["uid"] in selected for seed in group) >= target:
                    break
    for tag in (
        "invalid_input",
        "negation",
        "conjunction",
        "disjunction",
        "universal",
        "ordering",
        "place",
        "mesh_place",
    ):
        candidates = sorted(
            [seed for seed in seeds if tag in seed.get("tags", []) and seed["uid"] not in selected],
            key=lambda seed: str(seed["uid"]),
        )
        random.Random(f"{random_seed}:{tag}").shuffle(candidates)
        selected.update((seed["uid"], seed) for seed in candidates[:extra_per_tag])
    return sorted(selected.values(), key=lambda seed: str(seed["uid"]))


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt journal: {path}, line {index + 1}.") from None
            # A crash during the last append can leave a partial line. Remove it before
            # future appends; the pending request marker still prevents double billing.
            path.write_bytes(b"".join(lines[:index]))
    return result


class TeacherRunJournal:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.requests_path = self.directory / "requests.jsonl"
        self.attempts_path = self.directory / "attempts.jsonl"
        self.pending_path = self.directory / "pending_request.json"
        self.requests = read_journal(self.requests_path)
        self.attempts = read_journal(self.attempts_path)
        self.cache = {(item["attempt_id"], item["request_index"]): item for item in self.requests}
        self.completed = {item["attempt_id"]: item for item in self.attempts}
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text(encoding="utf-8"))
            if (pending["attempt_id"], pending["request_index"]) in self.cache:
                self.pending_path.unlink()
            else:
                raise RuntimeError(
                    "An interrupted request has no saved response and may have been billed. "
                    "Reconcile pending_request.json with OpenRouter before retrying; "
                    "automatic replay cannot safely determine its cost."
                )

    def cached_request(
        self,
        attempt_id: str,
        request_index: int,
        request: dict[str, Any],
    ) -> dict[str, Any] | None:
        cached = self.cache.get((attempt_id, request_index))
        if cached is None:
            return None
        if cached["request_fingerprint"] != mapping_fingerprint(request):
            raise ValueError(
                "Resumed conversation differs from the saved request. "
                "Keep configuration, schema, and database outcomes unchanged."
            )
        return cached["assistant"]

    def start_request(
        self,
        attempt_id: str,
        request_index: int,
        request: dict[str, Any],
    ) -> None:
        # Atomic rename ensures readers never see a partially-written pending marker.
        target = self.pending_path.with_suffix(".tmp")
        target.write_text(
            json.dumps(
                {
                    "attempt_id": attempt_id,
                    "request_index": request_index,
                    "request_fingerprint": mapping_fingerprint(request),
                }
            ),
            encoding="utf-8",
        )
        target.replace(self.pending_path)

    def finish_request(
        self,
        entry: dict[str, Any],
        request: dict[str, Any],
        assistant: dict[str, Any],
    ) -> None:
        record = {
            **entry,
            "request_fingerprint": mapping_fingerprint(request),
            "assistant": assistant,
        }
        append_jsonl(self.requests_path, record)
        self.requests.append(record)
        self.cache[(entry["attempt_id"], entry["request_index"])] = record
        self.pending_path.unlink(missing_ok=True)

    def finish_attempt(self, item: dict[str, Any]) -> None:
        if item["attempt_id"] in self.completed:
            raise ValueError(f"Duplicate completed attempt: {item['attempt_id']}.")
        append_jsonl(self.attempts_path, item)
        self.attempts.append(item)
        self.completed[item["attempt_id"]] = item

    def replace_attempts(self, attempts: list[dict[str, Any]]) -> None:
        """Commit generation optimizations atomically without appending duplicate attempts."""
        if [item["attempt_id"] for item in attempts] != [
            item["attempt_id"] for item in self.attempts
        ]:
            raise ValueError("Optimizations must preserve the completed attempt IDs and order.")
        target = self.attempts_path.with_suffix(".tmp")
        with target.open("w", encoding="utf-8") as handle:
            for item in attempts:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        target.replace(self.attempts_path)
        self.attempts = attempts
        self.completed = {item["attempt_id"]: item for item in attempts}


def attempt_payload(
    seed: dict[str, Any],
    decision,
    *,
    attempt_id: str,
    tokenizer,
    max_length: int,
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.training.train import (
        rendered_record_text,
        tokenize_rendered_text,
    )

    record = (
        decision.record.model_dump(by_alias=True, exclude_none=True)
        if (decision.accepted and decision.record is not None)
        else None
    )
    messages = record["messages"] if record else decision.payload.get("messages", [])
    lengths = {}
    if record:
        for include in (False, True):
            lengths["with_reasoning" if include else "without_reasoning"] = len(
                tokenize_rendered_text(
                    tokenizer,
                    rendered_record_text(
                        record,
                        tokenizer,
                        include_reasoning=include,
                    ),
                ),
            )
    excluded = (
        set((record.get("metadata") or {}).get("non_trainable_message_indices") or [])
        if record
        else set()
    )
    coverage = trace_reasoning_coverage(
        [message for index, message in enumerate(messages) if index not in excluded]
    )
    return {
        "attempt_id": attempt_id,
        "seed_uid": seed["uid"],
        "stratum": stratum(seed),
        "scene_id": seed.get("scene_id"),
        "task_family": seed.get("task_family"),
        "query_type": seed.get("query_type"),
        "tags": seed.get("tags", []),
        "accepted": decision.accepted,
        "reason": decision.reason,
        "reasoning": coverage,
        "token_lengths": lengths,
        "trainable_with_reasoning": bool(record) and lengths["with_reasoning"] <= max_length,
        "matched_trainable": bool(record) and max(lengths.values()) <= max_length,
        "record": record,
        "debug": decision.payload,
    }


def _rates(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    accepted = sum(bool(item["accepted"]) for item in items)
    optimized = [item for item in items if item.get("optimization") and item["accepted"]]
    kinds = Counter(item["optimization"]["kind"] for item in optimized)
    repaired = kinds["repair_invalid_final_answers"]
    recovered = kinds["recover_earlier_retrievals"]
    readable = sum(item["accepted"] and item["reasoning"]["both_stages"] for item in items)
    usable = sum(
        item["accepted"] and item["reasoning"]["both_stages"] and item["trainable_with_reasoning"]
        for item in items
    )
    return {
        "attempts": count,
        "accepted": accepted,
        "rejected": count - accepted,
        "accepted_before_optimizations": accepted - len(optimized),
        "rejected_before_optimizations": count - accepted + len(optimized),
        "repaired": repaired,
        "recovered": recovered,
        "accepted_with_both_stage_reasoning": readable,
        "usable_reasoning_traces": usable,
        "acceptance_rate": accepted / count if count else None,
        "both_stage_reasoning_rate": readable / count if count else None,
        "usable_reasoning_rate": usable / count if count else None,
        "rejection_reasons": dict(
            Counter(item["reason"] for item in items if not item["accepted"])
        ),
        "reasoning_kinds": dict(
            Counter(kind for item in items for kind in item["reasoning"].get("kinds", []))
        ),
    }


def pilot_report(
    population: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    traces_per_seed: int = 1,
) -> dict[str, Any]:
    costs: dict[str, list[float | None]] = defaultdict(list)
    for request in requests:
        costs[request["attempt_id"]].append(request.get("cost_usd"))
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        entries = costs.get(attempt["attempt_id"], [])
        attempt = {
            **attempt,
            "cost_usd": sum(entries)
            if entries and all(value is not None for value in entries)
            else None,
        }
        by_stratum[attempt["stratum"]].append(attempt)
        by_type[f"{attempt['task_family']}/{attempt['query_type']}"].append(attempt)
        by_scene[str(attempt["scene_id"])].append(attempt)
        for tag in attempt["tags"]:
            by_tag[tag].append(attempt)
    sizes = Counter(stratum(seed) for seed in population)
    projected_cost = 0.0
    projected_usable = 0.0
    variance = 0.0
    cost_complete = True
    variance_complete = True
    coverage_complete = True
    strata = {}
    for key, size in sorted(sizes.items()):
        items = by_stratum.get(key, [])
        rates = _rates(items)
        entries = [item["cost_usd"] for item in items]
        mean = (
            sum(entries) / len(entries)
            if entries and all(value is not None for value in entries)
            else None
        )
        strata[key] = {"population": size, **rates, "mean_conversation_cost_usd": mean}
        if mean is None:
            cost_complete = False
        else:
            projected_cost += size * traces_per_seed * mean
            if len(entries) > 1:
                sample_variance = sum((value - mean) ** 2 for value in entries) / (len(entries) - 1)
                variance += (size * traces_per_seed) ** 2 * sample_variance / len(entries)
            else:
                variance_complete = False
        if rates["usable_reasoning_rate"] is None:
            coverage_complete = False
        else:
            projected_usable += size * traces_per_seed * rates["usable_reasoning_rate"]
    known_cost = sum(
        float(request["cost_usd"]) for request in requests if request.get("cost_usd") is not None
    )
    unknown = sum(request.get("cost_usd") is None for request in requests)
    overall = _rates(attempts)
    accepted = overall["accepted"]
    usable = overall["usable_reasoning_traces"]
    complete_cost = known_cost if not unknown else None
    interval = None
    if cost_complete and variance_complete:
        margin = 1.96 * math.sqrt(variance)
        interval = [max(0.0, projected_cost - margin), projected_cost + margin]
    return {
        "schema_version": 1,
        "optimizations": {
            "stage": "generation",
            "accepted_before": overall["accepted_before_optimizations"],
            "rejected_before": overall["rejected_before_optimizations"],
            "repaired": overall["repaired"],
            "recovered": overall["recovered"],
            "accepted_after": overall["accepted"],
            "rejected_after": overall["rejected"],
            "by_kind": dict(Counter(
                item["optimization"]["kind"] for item in attempts if item.get("optimization")
            )),
        },
        "population_seeds": len(population),
        "population_task_types": len(
            {(seed.get("task_family"), seed.get("query_type")) for seed in population}
        ),
        "coverage": overall,
        "by_stratum": strata,
        "by_task_type": {key: _rates(items) for key, items in sorted(by_type.items())},
        "by_scene": {key: _rates(items) for key, items in sorted(by_scene.items())},
        "by_tag": {key: _rates(items) for key, items in sorted(by_tag.items())},
        "billing": {
            "requests": len(requests),
            "known_cost_usd": known_cost,
            "unknown_cost_requests": unknown,
            "total_cost_usd": complete_cost,
            "cost_per_accepted_trace_usd": complete_cost / accepted
            if complete_cost is not None and accepted
            else None,
            "cost_per_usable_reasoning_trace_usd": complete_cost / usable
            if complete_cost is not None and usable
            else None,
            "prompt_tokens": sum(
                (r.get("usage") or {}).get("prompt_tokens") or 0 for r in requests
            ),
            "completion_tokens": sum(
                (r.get("usage") or {}).get("completion_tokens") or 0 for r in requests
            ),
            "reasoning_tokens": sum(
                ((r.get("usage") or {}).get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
                or 0
                for r in requests
            ),
            "latency_seconds": sum(r.get("latency_seconds") or 0 for r in requests),
        },
        "projection": {
            "method": "scene_family_task_type_weighted_conversation_means",
            "full_dataset_cost_usd": projected_cost if cost_complete else None,
            "approximate_95_percent_cost_interval_usd": interval,
            "expected_usable_reasoning_traces": projected_usable if coverage_complete else None,
            "unsampled_strata": [key for key in sizes if not by_stratum.get(key)],
            "note": "Pilot estimates are exploratory; small strata and changed provider/cache "
            "behavior limit precision. Reasoning is already part of completion billing.",
        },
        "matched_retained_uids": sorted(
            {item["record"]["uid"] for item in attempts if item.get("matched_trainable")}
        ),
    }


def write_pilot_report(path: Path, report: dict[str, Any]) -> None:
    atomic_write_yaml(report, path)
