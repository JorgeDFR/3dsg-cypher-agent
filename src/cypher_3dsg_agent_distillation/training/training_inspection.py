from __future__ import annotations

import argparse
import json
from typing import Any

from cypher_3dsg_agent_distillation.core.config import load_data_config, load_yaml
from cypher_3dsg_agent_distillation.core.io import read_jsonl
from cypher_3dsg_agent_distillation.training.train import (
    ASSISTANT_ONLY_LOSS_MODE,
    IGNORE_INDEX,
    SUPPORTED_LOSS_MODES,
    active_label_count,
    build_training_example,
    load_tokenizer,
    normalize_chat_messages,
    normalize_loss_mode,
    rendered_record_text,
)


def select_record(records: list[dict[str, Any]], *, index: int, uid: str | None) -> dict[str, Any]:
    if uid is None:
        try:
            return records[index]
        except IndexError as exc:
            raise IndexError(
                f"Record index {index} is out of range for {len(records)} records."
            ) from exc

    for record in records:
        if str(record.get("uid")) == uid:
            return record
    raise KeyError(f"Could not find record with uid `{uid}`.")


def message_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    common_fields = {"role", "content", "tool_calls", "tool_name", "tool_call_id"}
    for index, message in enumerate(messages):
        tool_calls = message.get("tool_calls") or []
        summary.append(
            {
                "index": index,
                "role": message.get("role"),
                "content_chars": len(message.get("content") or ""),
                "tool_call_count": len(tool_calls),
                "tool_calls": [
                    {
                        "name": ((tool_call.get("function") or {}).get("name")),
                        "arguments": ((tool_call.get("function") or {}).get("arguments") or {}),
                    }
                    for tool_call in tool_calls
                ],
                "extra_fields": sorted(set(message) - common_fields),
            }
        )
    return summary


def label_runs(labels: list[int]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, label in enumerate(labels):
        if label != IGNORE_INDEX and start is None:
            start = index
        elif label == IGNORE_INDEX and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(labels)))
    return runs


def decoded_label_spans(tokenizer, input_ids: list[int], labels: list[int]) -> list[dict[str, Any]]:
    spans = []
    for index, (start, end) in enumerate(label_runs(labels)):
        spans.append(
            {
                "index": index,
                "start_token": start,
                "end_token": end,
                "tokens": end - start,
                "text": tokenizer.decode(input_ids[start:end], skip_special_tokens=False),
            }
        )
    return spans


def truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n...[truncated]..."


def inspect_record(
    record: dict[str, Any],
    tokenizer,
    *,
    loss_mode: str,
    max_length: int,
    max_chars: int,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    loss_mode = normalize_loss_mode(loss_mode)
    messages = normalize_chat_messages(record.get("messages", []))
    rendered_text = rendered_record_text(record, tokenizer, include_reasoning=include_reasoning)
    example = build_training_example(
        record,
        tokenizer,
        loss_mode=loss_mode,
        max_length=max_length,
        include_reasoning=include_reasoning,
    )
    supervised_tokens = active_label_count(example["labels"])
    return {
        "uid": record.get("uid"),
        "split_group": record.get("split_group"),
        "task_type": record.get("task_type") or record.get("query_type"),
        "render_view": record.get("render_view"),
        "loss_mode": loss_mode,
        "include_reasoning": include_reasoning,
        "max_length": max_length,
        "total_tokens": len(example["input_ids"]),
        "supervised_tokens": supervised_tokens,
        "masked_tokens": len(example["input_ids"]) - supervised_tokens,
        "messages": message_summary(messages),
        "rendered_text": truncate_text(rendered_text, max_chars),
        "supervised_spans": [
            {
                **span,
                "text": truncate_text(span["text"], max_chars),
            }
            for span in decoded_label_spans(
                tokenizer,
                example["input_ids"],
                example["labels"],
            )
        ],
    }


def print_text_report(report: dict[str, Any]) -> None:
    print(f"uid: {report['uid']}")
    print(f"task_type: {report['task_type']}")
    print(f"render_view: {report['render_view']}")
    print(f"loss_mode: {report['loss_mode']}")
    print(f"tokens: {report['total_tokens']}")
    print(f"supervised_tokens: {report['supervised_tokens']}")
    print(f"masked_tokens: {report['masked_tokens']}")
    print("")
    print("messages:")
    for message in report["messages"]:
        extras = ",".join(message["extra_fields"]) if message["extra_fields"] else "-"
        tool_names = ",".join(
            tool_call["name"] or "<unknown>" for tool_call in message["tool_calls"]
        )
        tool_names = tool_names or "-"
        print(
            f"  [{message['index']}] role={message['role']} "
            f"content_chars={message['content_chars']} "
            f"tool_calls={tool_names} extra_fields={extras}"
        )
    print("")
    print("rendered_text:")
    print(report["rendered_text"])
    print("")
    print("supervised_spans:")
    for span in report["supervised_spans"]:
        print(
            f"  span {span['index']}: tokens "
            f"{span['start_token']}..{span['end_token']} ({span['tokens']})"
        )
        print(span["text"])


def command(args: argparse.Namespace) -> None:
    data_config = load_data_config(
        args.data_config, dataset_version=args.dataset_version
    )
    student_config = load_yaml(args.student_config)
    loss_mode = normalize_loss_mode(args.loss_mode)
    max_length = args.max_length or int(student_config["training"].get("max_length", 4096))
    tokenizer = load_tokenizer(student_config, tokenizer_id=args.tokenizer)
    records = read_jsonl(data_config["splits"][args.split])
    record = select_record(records, index=args.index, uid=args.uid)
    report = inspect_record(
        record,
        tokenizer,
        loss_mode=loss_mode,
        max_length=max_length,
        max_chars=args.max_chars,
        include_reasoning=(getattr(args, "include_reasoning", None)
                           if getattr(args, "include_reasoning", None) is not None
                           else bool(student_config["training"].get("include_reasoning", False))),
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_text_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect rendered training text and supervised token spans."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument(
        "--student-config",
        default="configs/students/student_qwen3.5_4b.yaml",
    )
    parser.add_argument(
        "--split",
        choices=("train", "eval", "test", "challenge"),
        default="train",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--uid", default=None)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path/name. Defaults to model.tokenizer or model.base_model.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=SUPPORTED_LOSS_MODES,
        default=ASSISTANT_ONLY_LOSS_MODE,
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--include-reasoning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
