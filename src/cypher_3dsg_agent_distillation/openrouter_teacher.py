from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from cypher_3dsg_agent_distillation.agent_prompts import build_teacher_prompt
from cypher_3dsg_agent_distillation.agent_records import (
    DEFAULT_AGENT_TOOL,
    DEFAULT_TEACHER_MODEL,
    AgentTraceRecord,
    split_group_for_record,
)
from cypher_3dsg_agent_distillation.config import ensure_parent, load_yaml
from cypher_3dsg_agent_distillation.evaluation import normalize_answer, rows_to_solution_text
from cypher_3dsg_agent_distillation.io import read_jsonl
from cypher_3dsg_agent_distillation.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_URI,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    Neo4jValidator,
    is_safe_refusal,
    schema_policy,
    static_checks,
)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass
class TraceDecision:
    accepted: bool
    record: AgentTraceRecord | None
    reason: str
    payload: dict[str, Any]


class OpenRouterTeacherClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_TEACHER_MODEL,
        timeout: int = 120,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for teacher generation.")
        return self.api_key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.require_api_key()}",
            "Content-Type": "application/json",
        }

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"OpenRouter request timed out: {exc}") from exc

    def get_json(self, url: str) -> dict[str, Any]:
        req = request.Request(url, headers=self._headers(), method="GET")
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def preflight(self) -> None:
        data = self.get_json(OPENROUTER_MODELS_URL)
        models = {item.get("id") for item in data.get("data", []) if isinstance(item, dict)}
        if self.model not in models:
            raise RuntimeError(
                f"OpenRouter model `{self.model}` is unavailable. "
                f"Check https://openrouter.ai/models or choose a fallback."
            )

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        data = self.post_json(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        content = data["choices"][0]["message"].get("content") or "{}"
        return json.loads(content)


def build_candidate_trace(
    seed: dict[str, Any],
    candidate: dict[str, Any],
    teacher_model: str,
) -> AgentTraceRecord:
    cypher = str(seed.get("cypher") or seed.get("gold_cypher") or "").strip()
    return AgentTraceRecord(
        uid=str(candidate.get("uid") or f"{seed['uid']}_teacher"),
        scene_id=seed.get("scene_id"),
        question=str(candidate.get("question") or seed["question"]),
        query_type=seed.get("query_type", "unknown"),
        messages=candidate["messages"],
        gold_cypher=cypher,
        expected_solution=seed.get("solution") or seed.get("expected_solution"),
        split_group=split_group_for_record(
            seed.get("scene_id"),
            seed.get("query_type", "unknown"),
            cypher,
        ),
        teacher_model=teacher_model,
        read_only=bool(seed.get("read_only", True)),
        tags=seed.get("tags", []),
        metadata={"seed_uid": seed.get("uid")},
    )


def evaluate_candidate_trace(
    seed: dict[str, Any],
    candidate: dict[str, Any],
    *,
    teacher_model: str,
    validator: Neo4jValidator | None,
    policy: dict[str, Any] | None,
) -> TraceDecision:
    try:
        record = build_candidate_trace(seed, candidate, teacher_model)
    except Exception as exc:
        return TraceDecision(False, None, f"invalid_trace_schema: {exc}", candidate)

    if record.is_refusal:
        if not is_safe_refusal(record.gold_cypher):
            return TraceDecision(False, record, "unexpected_refusal", candidate)
        record.validation.static_valid = True
        record.validation.executable = True
        record.validation.solution_match = normalize_answer(record.expected_solution) in {
            "",
            normalize_answer(record.final_answer_text),
        } or "WRITE_QUERY_NOT_ALLOWED" in normalize_answer(record.final_answer_text)
        return TraceDecision(
            record.validation.solution_match,
            record,
            "accepted_refusal",
            candidate,
        )

    cypher = record.first_cypher_argument
    if not cypher:
        return TraceDecision(False, record, "missing_cypher_argument", candidate)
    if DEFAULT_AGENT_TOOL not in [call.function.name for call in record.tool_calls]:
        return TraceDecision(False, record, "missing_required_tool", candidate)

    static_issues = static_checks(cypher, allow_writes=not record.read_only, policy=policy)
    record.validation.static_issues = static_issues
    record.validation.static_valid = not static_issues
    if static_issues:
        return TraceDecision(False, record, "static_issues", candidate)

    if validator is None:
        record.validation.executable = True
        record.validation.solution_match = True
        return TraceDecision(True, record, "accepted_static_only", candidate)

    validation = validator.validate(cypher, execute=True)
    record.validation.executable = bool(validation.get("executable"))
    record.validation.error = validation.get("error")
    if not record.validation.executable:
        return TraceDecision(
            False,
            record,
            "execution_failed",
            {**candidate, "validation": validation},
        )

    actual = rows_to_solution_text(seed, validation.get("rows", []))
    record.validation.solution_match = normalize_answer(actual) == normalize_answer(
        record.expected_solution
    )
    if not record.validation.solution_match:
        return TraceDecision(
            False,
            record,
            "solution_mismatch",
            {**candidate, "actual_solution": actual, "validation": validation},
        )
    return TraceDecision(True, record, "accepted", candidate)


def generate_command(args: argparse.Namespace) -> None:
    teacher_config = load_yaml(args.teacher_config)
    data_config = load_yaml(args.data_config)
    model = args.teacher_model or teacher_config.get("model", DEFAULT_TEACHER_MODEL)
    client = OpenRouterTeacherClient(
        model=model,
        timeout=int(teacher_config.get("timeout", 120)),
        temperature=float(teacher_config.get("temperature", 0.2)),
        max_tokens=int(teacher_config.get("max_tokens", 2048)),
    )
    if args.preflight:
        client.preflight()

    schema_config = load_yaml(data_config["schema_path"])
    schema = schema_config["interface_description"]
    policy = schema_policy(schema_config)
    validator = None
    if not args.static_only:
        validator = Neo4jValidator(
            uri=args.uri,
            username=args.username,
            password=args.password,
            database=args.database,
            timeout=args.neo4j_timeout,
        )

    try:
        seeds = read_jsonl(args.input)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for seed in seeds:
            prompt = build_teacher_prompt(
                schema,
                seed["question"],
                seed.get("solution"),
                seed.get("read_only", True),
            )
            for _ in range(args.traces_per_seed):
                candidate = client.generate(prompt)
                decision = evaluate_candidate_trace(
                    seed,
                    candidate,
                    teacher_model=model,
                    validator=validator,
                    policy=policy,
                )
                if decision.accepted and decision.record is not None:
                    accepted.append(
                        decision.record.model_dump(by_alias=True, exclude_none=True)
                    )
                else:
                    rejected.append(
                        {
                            "seed_uid": seed.get("uid"),
                            "reason": decision.reason,
                            "payload": decision.payload,
                        }
                    )
    finally:
        if validator is not None:
            validator.close()

    ensure_parent(args.output).write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in accepted),
        encoding="utf-8",
    )
    ensure_parent(args.rejected_output).write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def preflight_command(args: argparse.Namespace) -> None:
    config = load_yaml(args.teacher_config)
    client = OpenRouterTeacherClient(
        model=args.teacher_model or config.get("model", DEFAULT_TEACHER_MODEL)
    )
    client.preflight()
    print(f"OpenRouter model `{client.model}` is available.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OpenRouter teacher agent traces.")
    subparsers = parser.add_subparsers(required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--teacher-config",
        default="configs/teacher_openrouter_gpt_oss_120b.yaml",
    )
    preflight_parser.add_argument("--teacher-model", default=None)
    preflight_parser.set_defaults(func=preflight_command)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--teacher-config",
        default="configs/teacher_openrouter_gpt_oss_120b.yaml",
    )
    generate_parser.add_argument("--data-config", default="configs/data.yaml")
    generate_parser.add_argument("--teacher-model", default=None)
    generate_parser.add_argument("--input", default="data/3dsg/seeds/cypher_seed_records.jsonl")
    generate_parser.add_argument(
        "--output",
        default="data/agent/traces_openrouter.accepted.jsonl",
    )
    generate_parser.add_argument(
        "--rejected-output",
        default="data/agent/traces_openrouter.rejected.json",
    )
    generate_parser.add_argument("--traces-per-seed", type=int, default=3)
    generate_parser.add_argument("--preflight", action="store_true")
    generate_parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip Neo4j execution; intended only for dry runs.",
    )
    generate_parser.add_argument("--neo4j-timeout", type=float, default=10.0)
    generate_parser.add_argument("--uri", default=DEFAULT_HERACLES_NEO4J_URI)
    generate_parser.add_argument("--username", default=DEFAULT_HERACLES_NEO4J_USERNAME)
    generate_parser.add_argument("--password", default=DEFAULT_HERACLES_NEO4J_PASSWORD)
    generate_parser.add_argument("--database", default=DEFAULT_HERACLES_NEO4J_DATABASE)
    generate_parser.set_defaults(func=generate_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
