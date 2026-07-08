from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

from text2cypher_3dsg.config import ensure_parent, load_yaml
from text2cypher_3dsg.prompts import DEFAULT_SYSTEM_PROMPT, build_messages
from text2cypher_3dsg.records import PreparedRecord, Text2CypherRecord


def load_raw_records(path: str | Path) -> list[Text2CypherRecord]:
    data = load_yaml(path)
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        raise TypeError("Expected `questions` to be a list.")
    return [Text2CypherRecord.model_validate(question) for question in questions]


def load_raw_records_from_config(data_config: dict) -> list[Text2CypherRecord]:
    paths = [data_config["raw_questions_path"]]
    paths.extend(data_config.get("additional_raw_questions_paths", []))
    records: list[Text2CypherRecord] = []
    seen_uids: set[str] = set()
    for path in paths:
        for record in load_raw_records(path):
            if record.uid in seen_uids:
                raise ValueError(f"Duplicate uid `{record.uid}` while loading {path}.")
            seen_uids.add(record.uid)
            records.append(record)
    return records


def prepare_records(
    raw_records: Iterable[Text2CypherRecord],
    schema_name: str,
    schema_text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[PreparedRecord]:
    prepared: list[PreparedRecord] = []
    for record in raw_records:
        prepared.append(
            PreparedRecord(
                uid=record.uid,
                question=record.question,
                cypher=record.cypher.strip(),
                query_type=record.query_type,
                read_only=record.read_only,
                expected_result_hash=record.expected_result_hash,
                solution=record.solution,
                tags=record.tags,
                scene_id=record.scene_id,
                schema_name=schema_name,
                schema=schema_text.strip(),
                messages=build_messages(schema_text, record.question, record.cypher, system_prompt),
                metadata=record.metadata,
            )
        )
    return prepared


def write_jsonl(records: Iterable[PreparedRecord | dict], path: str | Path) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, PreparedRecord):
                payload = record.model_dump(by_alias=True)
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _stratify_values(records: list[dict], field: str | None) -> list[str] | None:
    if not field:
        return None
    values = [str(record.get(field, "unknown")) for record in records]
    counts = {value: values.count(value) for value in set(values)}
    if any(count < 2 for count in counts.values()):
        return None
    return values


def _split_once(
    records: list[dict],
    test_size: float,
    seed: int,
    stratify_by: str | None,
) -> tuple[list[dict], list[dict]]:
    if not records:
        return [], []

    rng = random.Random(seed)
    n_test = round(len(records) * test_size)
    if test_size > 0 and n_test == 0 and len(records) > 1:
        n_test = 1
    n_test = min(max(n_test, 0), len(records) - 1 if len(records) > 1 else len(records))

    stratify = _stratify_values(records, stratify_by)
    if stratify is None:
        shuffled = list(records)
        rng.shuffle(shuffled)
        return shuffled[n_test:], shuffled[:n_test]

    groups: dict[str, list[dict]] = {}
    for record, value in zip(records, stratify, strict=True):
        groups.setdefault(value, []).append(record)

    selected: list[dict] = []
    remaining: list[dict] = []
    for group_records in groups.values():
        rng.shuffle(group_records)
        group_n = round(len(group_records) * test_size)
        selected.extend(group_records[:group_n])
        remaining.extend(group_records[group_n:])

    while len(selected) < n_test and remaining:
        selected.append(remaining.pop())
    while len(selected) > n_test:
        remaining.append(selected.pop())

    rng.shuffle(selected)
    rng.shuffle(remaining)
    return remaining, selected


def split_records(
    records: list[dict],
    test_size: float,
    eval_size: float,
    seed: int,
    stratify_by: str | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not records:
        return [], [], []
    train_eval, test = _split_once(records, test_size, seed, stratify_by)
    eval_fraction = eval_size / (1.0 - test_size)
    train, eval_records = _split_once(train_eval, eval_fraction, seed, stratify_by)
    return train, eval_records, test


def prepare_command(args: argparse.Namespace) -> None:
    data_config = load_yaml(args.config)
    schema_config = load_yaml(data_config["schema_path"])
    raw_records = load_raw_records_from_config(data_config)
    prepared = prepare_records(
        raw_records=raw_records,
        schema_name=schema_config.get("name", "3dsg"),
        schema_text=schema_config["interface_description"],
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
    )
    write_jsonl(prepared, data_config["prepared_jsonl_path"])


def split_command(args: argparse.Namespace) -> None:
    data_config = load_yaml(args.config)
    records = read_jsonl(data_config["prepared_jsonl_path"])
    split_config = data_config.get("split", {})
    train, eval_records, test = split_records(
        records,
        test_size=float(split_config.get("test_size", 0.15)),
        eval_size=float(split_config.get("eval_size", 0.15)),
        seed=int(split_config.get("seed", 42)),
        stratify_by=split_config.get("stratify_by"),
    )
    write_jsonl(train, data_config["train_jsonl_path"])
    write_jsonl(eval_records, data_config["eval_jsonl_path"])
    write_jsonl(test, data_config["test_jsonl_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 3DSG text-to-Cypher datasets.")
    subparsers = parser.add_subparsers(required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", default="configs/data.yaml")
    prepare_parser.add_argument("--system-prompt", default=None)
    prepare_parser.set_defaults(func=prepare_command)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--config", default="configs/data.yaml")
    split_parser.set_defaults(func=split_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
