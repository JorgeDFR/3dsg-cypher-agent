"""Human-readable teacher metrics and the generated dataset conversation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from cypher_3dsg_agent_distillation.agents.reasoning import reasoning_text, without_reasoning
from cypher_3dsg_agent_distillation.core import cli_output


def _generated_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Profile examples precede the dataset question. Teacher generation only adds
    # assistant/tool turns after the final user instruction.
    last_user = max(
        (i for i, message in enumerate(messages) if message.get("role") == "user"), default=-1
    )
    return messages[last_user + 1 :]


def reasoning_formats(message: dict[str, Any]) -> list[str]:
    text, kind = reasoning_text(message)
    formats = {"summary" if kind == "summary" else "text"} if text else set()
    for block in message.get("reasoning_details") or []:
        block_type = block.get("type")
        if block_type == "reasoning.encrypted":
            formats.add("encrypted")
        elif block_type == "reasoning.text" and block.get("text"):
            formats.add("text")
        elif block_type == "reasoning.summary" and block.get("summary"):
            formats.add("summary")
        elif block_type not in {"reasoning.text", "reasoning.summary"}:
            formats.add("opaque")
    return sorted(formats) or ["none"]


def reasoning_stage_report(
    returned: list[dict[str, Any]],
    saved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Distinguish text availability from opaque blocks and dataset inclusion."""

    def stage_messages(messages, stage):
        return [
            message
            for message in _generated_messages(messages)
            if message.get("role") == "assistant"
            and bool(message.get("tool_calls")) == (stage == "tool_call")
        ]

    result = {}
    for stage in ("tool_call", "final_answer"):
        observed = stage_messages(returned, stage)
        retained = stage_messages(saved, stage)
        readable = sum(bool(reasoning_text(message)[0]) for message in observed)
        result[stage] = {
            "turns": len(observed),
            "formats": sorted(
                {value for message in observed for value in reasoning_formats(message)}
            ),
            "readable_turns": readable,
            "encrypted_turns": sum(
                "encrypted" in reasoning_formats(message) for message in observed
            ),
            "all_turns_readable": bool(observed) and readable == len(observed),
            "saved_readable_turns": sum(bool(reasoning_text(message)[0]) for message in retained),
        }
    return result


def example_report(messages: list[dict[str, Any]], seed: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only question, model turns and feedback for display; omit prompt scaffolding."""
    seed = seed or {}
    messages = _generated_messages(messages)
    turns = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        text, kind = reasoning_text(message)
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass
            call_id = call.get("id") or call.get("tool_call_id")
            feedback = None
            for following in messages[index + 1 :]:
                if following.get("role") == "assistant":
                    break
                if following.get("role") == "tool" and (
                    not call_id
                    or not following.get("tool_call_id")
                    or following.get("tool_call_id") == call_id
                ):
                    feedback = following.get("content")
                    if isinstance(feedback, str):
                        try:
                            feedback = json.loads(feedback)
                        except ValueError:
                            pass
                    break
            calls.append(
                {"name": function.get("name"), "arguments": arguments, "feedback": feedback}
            )
        turns.append(
            {
                "stage": "tool_call" if calls else "final_answer",
                "reasoning": text,
                "reasoning_kind": kind,
                "reasoning_formats": reasoning_formats(message),
                "content": without_reasoning(message).get("content"),
                "tool_calls": calls,
                "finish_reason": (message.get("generation_metadata") or {}).get("finish_reason"),
            }
        )
    return {
        "question": seed.get("question"),
        "expected_answer": seed.get(
            "real_solution", seed.get("solution", seed.get("expected_solution"))
        ),
        "reference_cypher": seed.get("cypher") or seed.get("gold_cypher"),
        "turns": turns,
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "Unknown"
    if seconds < 60 and round(seconds, 2) < 60:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min {remainder} s" if hours else f"{minutes} min {remainder} s"


def format_cost(cost: float | None, *, dataset: bool = False) -> str:
    if cost is None:
        return "Unknown"
    precision = 2 if dataset else 6 if cost >= 0.000001 or cost == 0 else 8
    return f"${cost:,.{precision}f}"


def _check(value: bool | None) -> str:
    return "Pass" if value is True else "Fail" if value is False else "Not checked"


def _panel(title: str, content: Any, *, cypher: bool = False, flatten: bool = False) -> Panel:
    if content is None:
        content = "Not recorded"
    elif content == "":
        content = "Empty string"
    elif not isinstance(content, str):
        content = json.dumps(content, indent=2, ensure_ascii=False)
    if flatten:
        # Only the terminal representation changes; the saved answer retains its formatting.
        content = " ".join(
            line.strip()
            for line in content.replace(r"\r\n", "\n").replace(r"\n", "\n").splitlines()
            if line.strip()
        )
    body = (
        Syntax(content, "cypher", word_wrap=True, background_color="default")
        if cypher
        else Text(content)
    )
    return Panel(body, title=Text(title, style="bold"))


def _notes(*paragraphs: tuple[str, str]) -> Text:
    text = Text("Notes", style="default")
    text.stylize("bold", 0, len(text))
    for label, explanation in paragraphs:
        text.append("\n\n")
        text.append(f"{label}: ", style="bold")
        text.append(explanation)
    return text


def _table_with_notes(title: str, table: Table, notes: Text) -> None:
    cli_output.console.print(
        Panel(Group(table, Rule(), notes), title=Text(title, style="bold cyan"))
    )


def _summary_table(rows: list[tuple[str, Any]]) -> Table:
    table = Table(show_header=False, box=None, expand=True)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for label, value in rows:
        table.add_row(Text(label), Text("Unknown" if value is None else str(value)))
    return table


def print_check_report(report: dict[str, Any], output: str | Path) -> None:
    metrics, projection, outcome = report["metrics"], report["projection"], report["outcome"]
    requests = report.get("requests") or []
    effective = report.get("effective") or {}
    providers = sorted({str(item["provider"]) for item in requests if item.get("provider")})
    accepted = outcome.get("accepted")
    verdict = "Accepted" if accepted is True else "Rejected" if accepted is False else "Incomplete"
    status = verdict + (f" ({outcome['reason'].replace('_', ' ')})" if accepted is False else "")
    rows = [
        ("Model", effective.get("model") or (report.get("model_catalog") or {}).get("id")),
        ("Provider(s)", ", ".join(providers) or "Unknown"),
        ("Example result", status),
        ("API requests", metrics["requests"]),
        (
            "Input tokens",
            f"{metrics['prompt_tokens']:,}"
            if metrics.get("prompt_tokens") is not None
            else "Unknown",
        ),
        (
            "Output tokens (includes reasoning)",
            f"{metrics['completion_tokens']:,}"
            if metrics.get("completion_tokens") is not None
            else "Unknown",
        ),
        (
            "Reasoning tokens",
            f"{metrics['reasoning_tokens']:,}"
            if metrics.get("reasoning_tokens") is not None
            else "Unknown",
        ),
        ("Billed check cost", format_cost(metrics.get("billed_cost_usd"))),
        ("Time in model requests", format_duration(metrics.get("api_latency_seconds"))),
        ("Example generation time", format_duration(metrics.get("conversation_wall_seconds"))),
        (
            "Generation speed",
            f"{metrics['completion_tokens_per_second']:.2f} tokens/s"
            if metrics.get("completion_tokens_per_second") is not None
            else "Unknown",
        ),
        (
            "Dataset traces estimated",
            f"{projection['target_traces']:,}"
            if projection.get("target_traces") is not None
            else "Unknown",
        ),
        (
            "Estimated dataset cost",
            format_cost(projection.get("full_dataset_cost_usd"), dataset=True),
        ),
        (
            "Estimated dataset generation time",
            format_duration(projection.get("sequential_conversation_seconds")),
        ),
        ("Full JSON report", str(output)),
    ]
    if metrics.get("billed_cost_usd") is None:
        rows.insert(
            8,
            (
                "Estimated check cost (catalog)",
                format_cost(metrics.get("catalog_estimated_cost_usd")),
            ),
        )
    basis = "billed cost" if projection.get("cost_basis") == "billed" else "catalog prices"
    traces = projection.get("target_traces")
    multiplier = f"{traces:,} traces" if traces is not None else "the target number of traces"
    _table_with_notes(
        "Teacher check results",
        _summary_table(rows),
        _notes(
            (
                "Model request time",
                "The sum of the time from sending each API request to receiving "
                "its full response. This includes prompt processing, reasoning "
                "and answer generation, "
                "network time, and any provider wait.",
            ),
            (
                "Example generation time",
                "The full conversation, including model requests, Cypher "
                "query execution, and answer validation. Loading the Neo4j scene and discovering "
                "the model happen before this timer starts and are excluded.",
            ),
            (
                "Dataset estimate",
                f"The measured conversation duration (about "
                f"{format_duration(metrics.get('conversation_wall_seconds'))}) is multiplied by "
                f"{multiplier}, assuming one conversation at a time. The cost estimate likewise "
                f"uses this example's {basis}. Totals use unrounded values. Other question types, "
                "provider/cache behavior, scene loading, and rate limits "
                "can change the actual totals.",
            ),
            (
                "Generation speed",
                "Output tokens divided by total model request time. Output "
                "includes reasoning tokens. This measures the observed request rate, rather than "
                "the model's decoder speed alone.",
            ),
        ),
    )
    if outcome.get("error"):
        cli_output.console.print(_panel("Check stopped", outcome["error"]))

    returned = [
        (item.get("raw_response", {}).get("choices") or [{}])[0].get("message") or {}
        for item in requests
    ]
    stages = report["capability"].get("reasoning_stages") or reasoning_stage_report(
        returned,
        report.get("messages") or [],
    )
    table = Table(box=box.SIMPLE_HEAVY, show_edge=False, expand=True, header_style="bold")
    for label in ("Stage", "Returned format", "Readable turns", "Saved readable turns"):
        table.add_column(label)
    format_labels = {
        "text": "Text",
        "summary": "Summary",
        "encrypted": "Encrypted",
        "none": "None",
        "opaque": "Other opaque blocks",
    }
    for stage, label in (("tool_call", "Tool calls"), ("final_answer", "Final answer")):
        values = stages[stage]
        formats = " + ".join(format_labels.get(value, value) for value in values["formats"])
        table.add_row(
            label,
            formats or "Not generated",
            f"{values['readable_turns']}/{values['turns']}",
            f"{values['saved_readable_turns']}/{values['turns']}",
        )
    _table_with_notes(
        "Reasoning availability",
        table,
        _notes(
            (
                "Returned format",
                "Text is the provider's readable reasoning. Summary is a readable "
                "synopsis that may omit reasoning steps. Encrypted/opaque blocks cannot be used as "
                "training text. None means no reasoning was returned. Not generated means there "
                "was no model response at that stage.",
            ),
        ),
    )

    validation = outcome.get("validation") or {}
    checks = [
        ("Cypher safety/schema", validation.get("static_valid")),
        ("Cypher execution", validation.get("executable")),
        ("Query result/grounding matches expected", validation.get("cypher_solution_match")),
        ("Final answer format", validation.get("final_answer_format_valid")),
        ("Final answer matches expected", validation.get("final_answer_match")),
        ("Answer agrees with tool result", validation.get("final_answer_cypher_match")),
    ]
    _table_with_notes(
        "Answer validation",
        _summary_table([(label, _check(value)) for label, value in checks]),
        _notes(
            (
                "Results",
                "Pass means the check was performed and succeeded. Fail means it was "
                "performed and failed. Not checked means no result is available, for example "
                "when generation stopped before an answer could be validated.",
            ),
            (
                "Correctness",
                "The query's returned results or PDDL grounding are compared with "
                "the dataset requirements; equivalent Cypher queries can pass. The final answer "
                "is checked against the expected answer and, where applicable, the tool result. "
                "These checks do not validate every claim in the reasoning text.",
            ),
        ),
    )
    details = [str(value) for value in validation.get("static_issues") or []]
    if validation.get("error"):
        details.append(str(validation["error"]))
    if details:
        cli_output.console.print(_panel("Validation details", "\n".join(details)))

    example = report.get("example") or example_report(report.get("messages") or [], None)
    parts: list[Any] = [
        Text(
            f"Seed: {outcome.get('seed_uid', 'Unknown')}\n"
            f"Scene: {outcome.get('scene_id', 'Unknown')}\n"
            f"Task: {outcome.get('query_type', 'Unknown')}"
        ),
        _panel("Question", example.get("question")),
    ]
    tool_index, final_index = 0, 0
    for turn in example["turns"]:
        is_tool = turn["stage"] == "tool_call"
        tool_index += int(is_tool)
        final_index += int(not is_tool)
        label = f"Tool call {tool_index}" if is_tool else f"Final answer {final_index}"
        if turn.get("reasoning"):
            kind = "summary" if turn.get("reasoning_kind") == "summary" else "text"
            parts.append(_panel(f"{label} — reasoning {kind}", turn["reasoning"]))
        else:
            description = (
                "Encrypted/opaque reasoning only; no readable text."
                if any(kind in {"encrypted", "opaque"} for kind in turn["reasoning_formats"])
                else "No readable reasoning returned or saved."
            )
            parts.append(Text(f"{label}: {description}"))
        for call in turn["tool_calls"]:
            arguments = call["arguments"]
            cypher = arguments.get("cypher_string") if isinstance(arguments, dict) else None
            parts.append(
                _panel(
                    f"{label} — {call['name']}"
                    + (" — Cypher" if cypher is not None else " — arguments"),
                    cypher if cypher is not None else arguments,
                    cypher=cypher is not None,
                )
            )
            feedback = call.get("feedback")
            if isinstance(feedback, dict):
                if feedback.get("ok") is True:
                    rows = feedback.get("rows") or []
                    parts.append(Text(f"{label}: query executed; {len(rows)} result rows."))
                else:
                    parts.append(_panel(f"{label} — tool error", feedback))
            elif feedback:
                parts.append(_panel(f"{label} — tool feedback", feedback))
            else:
                parts.append(Text(f"{label}: no tool feedback recorded."))
        if turn.get("content") or not is_tool:
            parts.append(
                _panel(
                    f"{label} — generated answer" if not is_tool else f"{label} — assistant text",
                    turn.get("content") or "No answer generated.",
                )
            )
        if turn.get("finish_reason") not in (None, "stop", "tool_calls"):
            parts.append(
                Text(f"{label}: generation ended with {turn['finish_reason']}.", style="yellow")
            )
    if not final_index:
        parts.append(Text("No final answer generated.", style="yellow"))
    parts.append(_panel("Expected answer", example.get("expected_answer"), flatten=True))
    cli_output.console.print(
        Panel(Group(*parts), title=Text("Generated text example", style="bold cyan"))
    )
