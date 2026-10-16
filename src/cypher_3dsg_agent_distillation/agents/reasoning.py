"""Reasoning normalization and student rendering, independent of provider replay."""

from __future__ import annotations

import re
from typing import Any

REASONING_FIELDS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "reasoning_kind",
    "thinking",
    "reasoning_selection",
)
THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.IGNORECASE | re.DOTALL)


def reasoning_text(message: dict[str, Any]) -> tuple[str, str | None]:
    """Extract readable text only; encrypted blocks remain opaque replay data."""
    for key in ("reasoning", "reasoning_content", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            types = {item.get("type") for item in message.get("reasoning_details") or []}
            kind = (
                "summary"
                if "reasoning.summary" in types and "reasoning.text" not in types
                else "text"
            )
            return value.strip(), message.get("reasoning_kind") or kind
    details = message.get("reasoning_details") or []
    texts = [
        str(item["text"]).strip()
        for item in details
        if item.get("type") == "reasoning.text" and item.get("text")
    ]
    if texts:
        return "\n".join(texts), "text"
    summaries = [
        str(item["summary"]).strip()
        for item in details
        if item.get("type") == "reasoning.summary" and item.get("summary")
    ]
    if summaries:
        return "\n".join(summaries), "summary"
    blocks = THINK_BLOCK.findall(str(message.get("content") or ""))
    return ("\n".join(block.strip() for block in blocks), "text") if blocks else ("", None)


def normalize_reasoning(message: dict[str, Any]) -> dict[str, Any]:
    payload = dict(message)
    text, kind = reasoning_text(payload)
    if text:
        payload["reasoning"] = text
        payload["reasoning_kind"] = kind
    if isinstance(payload.get("content"), str):
        payload["content"] = THINK_BLOCK.sub("", payload["content"])
    return payload


def has_reasoning(message: dict[str, Any]) -> bool:
    """Detect readable, inline, and opaque provider reasoning."""
    return bool(
        reasoning_text(message)[0]
        or message.get("reasoning_details")
        or re.search(r"<think>", str(message.get("content") or ""), re.IGNORECASE)
    )


def without_reasoning(message: dict[str, Any]) -> dict[str, Any]:
    """Remove reasoning from saved messages, including inline think blocks."""
    payload = {key: value for key, value in message.items() if key not in REASONING_FIELDS}
    if payload.get("role") == "assistant" and isinstance(payload.get("content"), str):
        content = THINK_BLOCK.sub("", payload["content"])
        payload["content"] = re.split(r"<think>", content, maxsplit=1, flags=re.IGNORECASE)[0]
    return payload


def student_messages(
    messages: list[dict[str, Any]],
    tokenizer,
    *,
    include_reasoning: bool,
) -> list[dict[str, Any]]:
    """Adapt canonical reasoning to a native field or portable think delimiters."""
    native = "reasoning_content" in str(getattr(tokenizer, "chat_template", "") or "")
    result = []
    for message in messages:
        text, _kind = reasoning_text(message)
        payload = {key: value for key, value in message.items() if key not in REASONING_FIELDS}
        content = str(payload.get("content") or "")
        payload["content"] = (
            THINK_BLOCK.sub("", content) if payload.get("role") == "assistant" else content
        )
        if include_reasoning and text and payload.get("role") == "assistant":
            if native:
                payload["reasoning_content"] = text
            else:
                payload["content"] = f"<think>\n{text}\n</think>\n\n{payload['content']}"
        result.append(payload)
    return result


def training_template(tokenizer, *, include_reasoning: bool) -> str | None:
    """Make Qwen's training transcript prefix-stable even with XML tool feedback."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return None
    condition = "{%- if loop.index0 > ns.last_query_index %}"
    if "reasoning_content" in template and condition in template:
        return template.replace(condition, "{%- if true %}")
    return None


def split_generated_reasoning(
    text: str,
    *,
    thinking_started: bool = False,
) -> tuple[str, str, str]:
    """Return action text, reasoning, and status without executing unfinished thoughts."""
    endings = list(re.finditer(r"</think>", text, re.IGNORECASE))
    if endings:
        before = text[: endings[0].start()]
        before = re.sub(r"^.*?<think>\s*", "", before, flags=re.IGNORECASE | re.DOTALL)
        blocks = THINK_BLOCK.findall(text[endings[0].end() : endings[-1].end()])
        reasoning = "\n".join([before.strip(), *(block.strip() for block in blocks)])
        action = text[endings[-1].end() :].lstrip()
        opened = re.search(r"<think>", action, re.IGNORECASE)
        if opened:
            return "", reasoning + "\n" + action[opened.end() :].strip(), "incomplete"
        return action, reasoning, "complete"
    start = re.search(r"<think>", text, re.IGNORECASE)
    if start or thinking_started:
        return "", text[start.end() :].strip() if start else text.strip(), "incomplete"
    return text, "", "absent"


def trace_reasoning_coverage(messages: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [
        message
        for message in messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    finals = [
        message
        for message in messages
        if message.get("role") == "assistant"
        and not message.get("tool_calls")
        and message.get("content")
    ]
    tool = bool(calls) and all(reasoning_text(message)[0] for message in calls)
    final = bool(finals) and bool(reasoning_text(finals[-1])[0])
    return {
        "tool_reasoning": tool,
        "final_reasoning": final,
        "both_stages": final and (tool or not calls),
        "kinds": sorted(
            {
                kind
                for message in messages
                if message.get("role") == "assistant"
                if (kind := reasoning_text(message)[1]) is not None
            }
        ),
    }


def without_controlled_repair(seed: dict[str, Any]) -> dict[str, Any]:
    """Keep the question and its identity, removing synthetic error injection semantics."""
    metadata = dict(seed.get("metadata") or {})
    if metadata.pop("controlled_repair", False):
        metadata["controlled_repair_skipped"] = True
    for key in (
        "repair_template",
        "repair_cypher",
        "repair_context_message_count",
        "non_trainable_message_indices",
    ):
        metadata.pop(key, None)
    return {
        **seed,
        "metadata": metadata,
        "tags": [tag for tag in seed.get("tags", []) if tag != "controlled_repair"],
    }
