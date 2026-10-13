"""Small, bounded teacher capability checks without creating a training dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.agents.agent_records import DEFAULT_TEACHER_MODEL
from cypher_3dsg_agent_distillation.agents.experiment_profiles import (
    experiment_profile_for_seed,
    load_experiment_profiles_from_config,
)
from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
    OpenRouterTeacherClient,
    TeacherBudgetExceeded,
    configured_teacher_provider,
    run_teacher_interaction,
)
from cypher_3dsg_agent_distillation.agents.reasoning import (
    reasoning_text,
    trace_reasoning_coverage,
    without_controlled_repair,
    without_reasoning,
)
from cypher_3dsg_agent_distillation.agents.teacher_check_reporting import (
    example_report,
    print_check_report,
    reasoning_stage_report,
)
from cypher_3dsg_agent_distillation.core.config import ensure_parent, load_data_config, load_yaml
from cypher_3dsg_agent_distillation.core.io import read_jsonl
from cypher_3dsg_agent_distillation.core.versioning import require_new_outputs
from cypher_3dsg_agent_distillation.graph.load_neo4j import load_scene
from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    Neo4jValidator,
    is_safe_refusal,
    neo4j_uri_candidates,
    schema_policy,
    suppress_neo4j_notification_logs,
)


class TeacherCheckRequestLimit(RuntimeError):
    """The check never silently expands into additional conversations or retries."""


class TeacherCheckClient(OpenRouterTeacherClient):
    def __init__(self, *, max_requests: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_requests = max_requests
        self.observed_messages: list[dict[str, Any]] = []

    def chat(self, messages, *, tools=None, tool_choice=None):
        self.observed_messages = deepcopy(messages)
        if len(self.requests) >= self.max_requests:
            raise TeacherCheckRequestLimit(f"Reached the {self.max_requests}-request check limit.")
        assistant = super().chat(messages, tools=tools, tool_choice=tool_choice)
        self.observed_messages = deepcopy(messages + [assistant])
        return assistant


def select_check_seed(
    population: list[dict[str, Any]],
    seed_uid: str | None,
) -> dict[str, Any]:
    if seed_uid is not None:
        matches = [seed for seed in population if seed["uid"] == seed_uid]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one dataset seed with UID {seed_uid!r}.")
        return without_controlled_repair(matches[0])
    candidates = [
        seed
        for seed in population
        if seed.get("read_only", True)
        and not is_safe_refusal(str(seed.get("cypher") or seed.get("gold_cypher") or ""))
        and (seed.get("task_family") or "cypher_qa") == "cypher_qa"
    ]
    if not candidates:
        raise ValueError("No answerable Cypher QA seeds; select one explicitly with --seed-uid.")
    # A deterministic middle-sized QA example, rather than the cheapest refusal.
    candidates.sort(
        key=lambda seed: (
            len(str(seed.get("question") or "")) + len(str(seed.get("cypher") or "")),
            str(seed["uid"]),
        )
    )
    return without_controlled_repair(candidates[len(candidates) // 2])


def _sum_known(values: list[int | float | None]) -> int | float | None:
    return sum(values) if values and all(value is not None for value in values) else None


def _price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def check_report(
    *,
    model_info: dict[str, Any],
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    wall_seconds: float,
    outcome: dict[str, Any],
    target_traces: int | None,
    seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one observed conversation, keeping missing billing and coverage explicit."""
    pricing = model_info.get("pricing") or {}
    prompt_price, completion_price = (
        _price(pricing.get("prompt")),
        _price(pricing.get("completion")),
    )
    per_request = []
    for entry, response in zip(requests, responses, strict=True):
        usage = entry.get("usage") or {}
        assistant = (response.get("choices") or [{}])[0].get("message") or {}
        text, kind = reasoning_text(assistant)
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        seconds = entry.get("latency_seconds")
        estimate = (
            prompt * prompt_price + completion * completion_price
            if None not in (prompt, completion, prompt_price, completion_price)
            else None
        )
        per_request.append(
            {
                **entry,
                "readable_reasoning": bool(text),
                "reasoning_kind": kind,
                "reasoning_characters": len(text),
                "reasoning_detail_types": sorted(
                    {str(block.get("type")) for block in assistant.get("reasoning_details") or []}
                ),
                "completion_tokens_per_second": completion / seconds
                if completion is not None and seconds and seconds > 0
                else None,
                "catalog_estimated_cost_usd": estimate,
                "raw_response": response,
            }
        )
    prompt_tokens = _sum_known(
        [(item.get("usage") or {}).get("prompt_tokens") for item in requests]
    )
    completion_tokens = _sum_known(
        [(item.get("usage") or {}).get("completion_tokens") for item in requests]
    )
    reasoning_tokens = _sum_known(
        [
            ((item.get("usage") or {}).get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            for item in requests
        ]
    )
    api_seconds = _sum_known([item.get("latency_seconds") for item in requests])
    billed = _sum_known([item.get("cost_usd") for item in requests])
    estimated = _sum_known([item["catalog_estimated_cost_usd"] for item in per_request])
    complete = outcome.get("status") == "completed"
    cost_basis = billed if billed is not None else estimated
    readable = [item["readable_reasoning"] for item in per_request]
    truncated = any(item.get("finish_reason") == "length" for item in requests)
    project = complete and not truncated and bool(target_traces)
    reason_status = (
        "observed"
        if any(readable)
        else "inconclusive_truncated"
        if truncated
        else "not_observed"
        if requests
        else "not_tested"
    )
    coverage = trace_reasoning_coverage(messages)
    returned = [(response.get("choices") or [{}])[0].get("message") or {} for response in responses]
    accepted = outcome.get("accepted")
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "dataset",
        "model_catalog": model_info,
        "outcome": outcome,
        "capability": {
            "readable_reasoning_status": reason_status,
            "reasoning_coverage": coverage,
            "reasoning_stages": reasoning_stage_report(returned, messages),
            "tool_call_observed": any(message.get("tool_calls") for message in messages),
            "dataset_example_accepted": accepted,
            "both_stage_reasoning_and_accepted": bool(accepted and coverage["both_stages"])
            if complete
            else None,
            "note": "One dataset example tests "
            "that conversation, not overall acceptance or coverage. Missing text in one "
            "response does not establish that the model never exposes reasoning.",
        },
        "metrics": {
            "requests": len(requests),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "billed_cost_usd": billed,
            "unknown_cost_requests": sum(item.get("cost_usd") is None for item in requests),
            "catalog_estimated_cost_usd": estimated,
            "catalog_prompt_usd_per_token": prompt_price,
            "catalog_completion_usd_per_token": completion_price,
            "effective_billed_usd_per_total_token": billed / (prompt_tokens + completion_tokens)
            if billed is not None
            and prompt_tokens is not None
            and completion_tokens is not None
            and prompt_tokens + completion_tokens > 0
            else None,
            "api_latency_seconds": api_seconds,
            "conversation_wall_seconds": wall_seconds,
            "completion_tokens_per_second": completion_tokens / api_seconds
            if completion_tokens is not None and api_seconds and api_seconds > 0
            else None,
            "note": "Throughput includes request, prompt processing and network time; "
            "it is not decoder-only speed or time to first token. Reasoning tokens are "
            "already included in completion tokens and cost. Catalog estimates ignore "
            "provider differences, caching and additional fees; billed cost takes precedence.",
        },
        "projection": {
            "method": "one_conversation_times_target_traces",
            "target_traces": target_traces,
            "full_dataset_cost_usd": cost_basis * target_traces
            if project and cost_basis is not None
            else None,
            "cost_basis": "billed"
            if billed is not None
            else "catalog_estimate"
            if estimated is not None
            else "unknown",
            "sequential_api_seconds": api_seconds * target_traces
            if project and api_seconds is not None
            else None,
            "sequential_conversation_seconds": wall_seconds * target_traces if project else None,
            "note": "Projections assume every attempt resembles this example. "
            "Rejected completed attempts are included; no accepted-dataset yield is inferred. "
            "Scene loading, rate limits, parallelism and task variation are not modeled. "
            "Interrupted checks do not project full conversation costs.",
        },
        "requests": per_request,
        "messages": messages,
        "example": example_report(messages, seed),
    }


def check_command(args: argparse.Namespace) -> None:
    config = load_yaml(args.teacher_config)
    if configured_teacher_provider(config) != "openrouter":
        raise ValueError("The paid teacher check requires provider: openrouter.")
    reasoning = dict(config.get("reasoning") or {})
    reasoning["enabled"] = args.include_reasoning
    if args.reasoning_effort == "default":
        reasoning.pop("effort", None)
    elif args.reasoning_effort is not None:
        reasoning["effort"] = args.reasoning_effort
    if args.max_requests < 1 or args.traces_per_seed is not None and args.traces_per_seed < 1:
        raise ValueError("Request and trace counts must be positive.")
    if args.dataset_size is not None and args.dataset_size < 1:
        raise ValueError("--dataset-size must be positive.")
    output = (
        Path(args.output)
        if args.output
        else Path("outputs/teacher_checks")
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    )
    require_new_outputs([output], force=args.force)
    requests, responses = [], []
    messages: list[dict[str, Any]] = []
    outcome: dict[str, Any] = {"status": "not_started"}
    target_traces = None
    wall_seconds = 0.0
    with TeacherCheckClient(
        model=args.teacher_model or config.get("model") or DEFAULT_TEACHER_MODEL,
        timeout=int(config.get("timeout", 120)),
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=args.max_tokens,
        reasoning=reasoning,
        provider=config.get("routing"),
        max_cost_usd=args.max_cost_usd,
        max_requests=args.max_requests,
    ) as client:
        client.require_api_key()
        client.preflight()
        # Older configurations have no routing price ceilings. Supply advertised
        # prices as routing limits so the check's spending reservation still applies.
        routing = dict(client.provider or {})
        ceilings = dict(routing.get("max_price") or {})
        for key in ("prompt", "completion"):
            if key not in ceilings:
                price = _price((client.model_info.get("pricing") or {}).get(key))
                if price is None:
                    raise ValueError("Teacher check requires routing.max_price or catalog pricing.")
                ceilings[key] = price * 1_000_000
        routing["max_price"] = ceilings
        routing.setdefault("require_parameters", True)
        client.provider = routing

        # Preserve all response fields locally, including provider-specific metrics.
        def capture(entry):
            requests.append(deepcopy(entry))
            responses.append(deepcopy(client.last_response))

        client.on_response = capture
        data = load_data_config(args.data_config, dataset_version=args.dataset_version)
        population = read_jsonl(data["seed_records_jsonl_path"])
        seed = select_check_seed(population, args.seed_uid)
        profile = experiment_profile_for_seed(seed, load_experiment_profiles_from_config(data))
        count = args.dataset_size if args.dataset_size is not None else len(population)
        copies = (
            args.traces_per_seed
            if args.traces_per_seed is not None
            else int(config.get("traces_per_seed", 1))
        )
        if copies < 1:
            raise ValueError("traces_per_seed must be positive.")
        target_traces = count * copies
        scene = next(item for item in data["scenes"] if item["scene_id"] == seed["scene_id"])
        suppress_neo4j_notification_logs()
        load_scene(
            scene_path=scene["scene_path"],
            uri=neo4j_uri_candidates(args.uri),
            username=args.username,
            password=args.password,
            database=args.database,
            clear=True,
        )
        validator = Neo4jValidator(
            uri=neo4j_uri_candidates(args.uri),
            username=args.username,
            password=args.password,
            database=args.database,
            timeout=args.neo4j_timeout,
        )
        started = time.monotonic()
        try:
            decision = run_teacher_interaction(
                seed,
                profile=profile,
                client=client,
                teacher_model=client.model,
                validator=validator,
                policy=schema_policy(load_yaml(data["schema_path"])),
                max_tool_calls=max(1, args.max_requests - 1),
                static_only=False,
            )
            messages = decision.payload.get("messages") or []
            outcome = {
                "status": "completed",
                "accepted": decision.accepted,
                "reason": decision.reason,
                "seed_uid": seed["uid"],
                "scene_id": seed["scene_id"],
                "task_family": seed.get("task_family"),
                "query_type": seed.get("query_type"),
                "validation": decision.payload.get("validation")
                or (decision.record.validation.model_dump() if decision.record else None),
            }
        except (TeacherBudgetExceeded, TeacherCheckRequestLimit, RuntimeError) as exc:
            outcome = {
                "status": "incomplete",
                "error": str(exc),
                "seed_uid": seed["uid"],
                "scene_id": seed["scene_id"],
                "task_family": seed.get("task_family"),
                "query_type": seed.get("query_type"),
            }
            messages = client.observed_messages
            if not client.include_reasoning:
                messages = [without_reasoning(message) for message in messages]
        finally:
            wall_seconds = time.monotonic() - started
            if validator is not None:
                validator.close()
        report = check_report(
            model_info=client.model_info,
            requests=requests,
            responses=responses,
            messages=messages,
            wall_seconds=wall_seconds,
            outcome=outcome,
            target_traces=target_traces,
            seed=seed,
        )
        report["effective"] = {
            "model": client.model,
            "reasoning": reasoning,
            "routing": client.provider,
            "max_tokens": client.max_tokens,
            "max_requests": client.max_requests,
            "max_cost_usd": client.max_cost_usd,
        }
    ensure_parent(output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print_check_report(report, output)


def add_check_parser(subparsers) -> None:
    parser = subparsers.add_parser("check", help="Check one dataset conversation with tool calls.")
    parser.add_argument(
        "--teacher-config", default="configs/teachers/teacher_deepseek_v4_1_flash.yaml"
    )
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--include-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reasoning-effort", default=None, help="Override effort; 'default' omits it."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Output limit per API response."
    )
    parser.add_argument(
        "--max-requests", type=int, default=2, help="Maximum API requests for the conversation."
    )
    parser.add_argument("--max-cost-usd", type=float, default=0.10, help="Check spending limit.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--seed-uid", default=None)
    parser.add_argument(
        "--dataset-size", type=int, default=None, help="Override projection seed count."
    )
    parser.add_argument("--traces-per-seed", type=int, default=None)
    parser.add_argument("--uri", default=None)
    parser.add_argument("--username", default=DEFAULT_HERACLES_NEO4J_USERNAME)
    parser.add_argument("--password", default=DEFAULT_HERACLES_NEO4J_PASSWORD)
    parser.add_argument("--database", default=DEFAULT_HERACLES_NEO4J_DATABASE)
    parser.add_argument("--neo4j-timeout", type=float, default=10.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(func=check_command)
