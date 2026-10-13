from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cypher_3dsg_agent_distillation.core.cli_output import print_summary
from cypher_3dsg_agent_distillation.core.config import (
    PROJECT_ROOT,
    ensure_parent,
    existing_report_path,
    load_data_config,
    load_student_config,
    model_reports_dir,
    resolve_project_path,
)
from cypher_3dsg_agent_distillation.core.io import read_jsonl
from cypher_3dsg_agent_distillation.core.versioning import (
    confirm_dataset_lineage,
    read_metadata,
)
from cypher_3dsg_agent_distillation.cypher.cypher_match_diagnostics import (
    result_match_diagnostics,
)

SPLITS = ("train", "eval", "test", "challenge")
DEFAULT_REPORT_NAME = "workflow_report.html"
FINAL_ANSWER_CHECK = "final_answer_from_trace"
TOOL_CALL_CHECK = "tool_call_from_prompt"
END_TO_END_CHECK = "end_to_end_final_answer"
SCENE_CLASS_LABELS = ("Object", "MeshPlace", "Room")
LABEL_OVERRIDES = {
    "id": "ID",
    "uid": "UID",
    "jsonl": "JSONL",
    "3d": "3D",
    "3dsg": "3DSG",
    "pddl": "PDDL",
    "qa": "QA",
    "sldp": "SLDP",
    "usd": "USD",
    "url": "URL",
    "utc": "UTC",
}
NEGATIVE_BOOLEAN_KEYS = {"rejected", "potential_hidden_false_positive"}
CLEAN_DETAIL_FIELD_TOKENS = ("answer", "cypher", "solution")
PRE_TABLE_KEYS = {"gold_cypher", "model_cypher", "sample", "solution"}


def workflow_report_output_path(
    data_config: dict[str, Any],
    output: str | None = None,
    student_config: dict[str, Any] | None = None,
) -> Path:
    if output:
        return Path(output)
    if student_config is not None and student_config.get("model_version"):
        return model_reports_dir(data_config, student_config) / DEFAULT_REPORT_NAME
    return Path(data_config["test_report_path"]).with_name(DEFAULT_REPORT_NAME)


def relative_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    resolved = Path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def read_yaml_if_exists(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_json_if_exists(path: str | Path) -> Any:
    resolved = existing_report_path(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def read_json_array_if_exists(path: str | Path) -> list[dict[str, Any]]:
    data = read_json_if_exists(path)
    return data if isinstance(data, list) else []


def read_jsonl_if_exists(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    return read_jsonl(resolved)


def file_info(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False}
    resolved = Path(path)
    info: dict[str, Any] = {
        "path": relative_path(resolved),
        "exists": resolved.exists(),
    }
    if resolved.exists():
        stat = resolved.stat()
        info.update(
            {
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            }
        )
    return info


def task_type(record: dict[str, Any]) -> str:
    return str(record.get("task_type") or record.get("query_type") or "unknown")


def count_by_task_type(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(task_type(record) for record in records).items()))


def task_family(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(record.get("task_family") or metadata.get("task_family") or "cypher_qa")


def count_by_task_family(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(task_family(record) for record in records).items()))


def output_type(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    value = record.get("output_type") or metadata.get("output_type")
    if value:
        return str(value)
    return "PDDL" if task_family(record) == "cypher_pddl" else "SLDP"


def is_pddl_record(record: dict[str, Any]) -> bool:
    return task_family(record) == "cypher_pddl"


def count_tags(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for tag in record.get("tags") or []:
            counter[str(tag)] += 1
    return dict(sorted(counter.items()))


def generated_question_supervision_counts(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    empty_questions = 0
    repair_questions = 0
    for record in records:
        tags = {str(tag) for tag in record.get("tags") or []}
        metadata = record.get("metadata") or {}
        empty_questions += int("invalid_input" in tags)
        repair_questions += int(
            "controlled_repair" in tags or bool(metadata.get("controlled_repair"))
        )
    return {
        "empty questions": empty_questions,
        "Cypher repair questions": repair_questions,
    }


def one_line_text(value: Any) -> str:
    return " ".join(str(value).split())


def expanded_text(value: Any) -> str:
    return str(value).replace("\\n", "\n").strip()


def clean_detail_text(value: Any) -> str:
    text = one_line_text(str(value).replace("\\n", " ").replace("\\r", " ").strip())
    while len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    for opener, closer in (("<", ">"), ("[", "]"), ("{", "}")):
        text = text.replace(f"{opener} ", opener).replace(f" {closer}", closer)
    return text


def field_text(value: Any, key: str | None = None) -> str:
    text = format_cell(value, key)
    if key and any(token in key.lower() for token in CLEAN_DETAIL_FIELD_TOKENS):
        return clean_detail_text(text)
    return text


def record_gold_cypher(record: dict[str, Any]) -> Any:
    return record.get("gold_cypher") or record.get("solution_cypher")


def tool_call_cypher(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    arguments = function.get("arguments", tool_call.get("arguments"))
    if isinstance(arguments, dict):
        return str(
            arguments.get("cypher_string")
            or arguments.get("cypher")
            or arguments.get("query")
            or ""
        )
    if not isinstance(arguments, str):
        return ""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        match = re.search(
            r'"(?:cypher_string|cypher|query)"\s*:\s*"(?P<cypher>.*)',
            arguments,
            flags=re.DOTALL,
        )
        return match.group("cypher").strip() if match else arguments.strip()
    if not isinstance(parsed, dict):
        return ""
    return str(
        parsed.get("cypher_string") or parsed.get("cypher") or parsed.get("query") or ""
    )


def model_cypher_from_messages(record: dict[str, Any]) -> str:
    cyphers = []
    for message in record.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            cypher = tool_call_cypher(tool_call).strip()
            if cypher and cypher not in cyphers:
                cyphers.append(cypher)
    return "\n\n".join(cyphers)


def record_model_cypher(record: dict[str, Any]) -> Any:
    return record.get("cypher") or record.get("model_cypher") or model_cypher_from_messages(record)


def record_solution(record: dict[str, Any]) -> Any:
    for key in ("expected_solution", "real_solution", "solution"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def bool_or_none(value: Any) -> bool | Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return value


def count_tool_calls(record: dict[str, Any]) -> int:
    messages = record.get("messages") or []
    return sum(len(message.get("tool_calls") or []) for message in messages)


def validation_status(record: dict[str, Any]) -> str:
    validation = record.get("validation") or {}
    if validation.get("solution_match") or validation.get("final_answer_match"):
        return "pass"
    if validation.get("error"):
        return "error"
    if validation:
        return "fail"
    return ""


def compact_question_record(record: dict[str, Any], split: str | None = None) -> dict[str, Any]:
    row = {
        "uid": record.get("uid"),
        "split": split,
        "task_family": task_family(record),
        "output_type": output_type(record),
        "task_type": task_type(record),
        "question": record.get("question"),
        "tags": record.get("tags") or [],
        "gold_cypher": clean_detail_text(record_gold_cypher(record) or ""),
        "solution": clean_detail_text(record_solution(record) or ""),
    }
    return {key: value for key, value in row.items() if value not in (None, "", [])}


def expected_output_blocks(record: dict[str, Any]) -> dict[str, Any]:
    solution = record_solution(record)
    if is_pddl_record(record):
        empty_answer = solution is not None and not str(solution).strip()
        return {
            "expected_pddl_goal": None if empty_answer else solution,
            "expected_empty_answer": empty_answer,
            "expected_grounding": record.get("grounding"),
        }
    return {"solution": solution}


def optimized_from_record(record: dict[str, Any], split: str) -> dict[str, Any] | None:
    metadata = record.get("metadata") or {}
    kinds = []
    if metadata.get("final_answer_repaired"):
        kinds.append("repaired_invalid_answer_format")
    if not kinds:
        return None
    return {
        "uid": record.get("uid"),
        "split": split,
        "task_type": task_type(record),
        "question": record.get("question"),
        "optimization_kinds": kinds,
        "tool_call_count": metadata.get("tool_call_count"),
        "source_trace_uid": metadata.get("source_trace_uid"),
        "source_rejection_reason": metadata.get("source_rejection_reason"),
    }


def optimization_lookup(
    optimized_questions: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in optimized_questions:
        uid = str(item.get("uid") or "")
        split = str(item.get("split") or "")
        if not uid or not split:
            continue
        lookup[(uid, split)] = item
    return lookup


def optimization_kinds_from_info(info: dict[str, Any] | None) -> list[str]:
    if not info:
        return []
    kinds = info.get("optimization_kinds") or []
    return [str(kind) for kind in kinds] or ["optimized"]


def record_lookup_keys(record: dict[str, Any]) -> set[str]:
    metadata = record.get("metadata") or {}
    keys = {
        str(value)
        for value in (
            record.get("uid"),
            record.get("seed_uid"),
            metadata.get("seed_uid"),
            metadata.get("source_trace_uid"),
        )
        if value
    }
    for key in list(keys):
        for suffix in ("_teacher_repaired", "_teacher"):
            if key.endswith(suffix):
                keys.add(key[: -len(suffix)])
    return keys


def build_tag_lookup(*record_groups: list[dict[str, Any]]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for records in record_groups:
        for record in records:
            tags = [str(tag) for tag in record.get("tags") or []]
            if not tags:
                continue
            for key in record_lookup_keys(record):
                lookup.setdefault(key, tags)
    return lookup


def build_gold_cypher_lookup(
    *record_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for records in record_groups:
        for record in records:
            cypher = record_gold_cypher(record)
            if not cypher:
                continue
            for key in record_lookup_keys(record):
                lookup.setdefault(key, cypher)
    return lookup


def tags_for_record(record: dict[str, Any], tag_lookup: dict[str, list[str]]) -> list[str]:
    tags = [str(tag) for tag in record.get("tags") or []]
    if tags:
        return tags
    for key in record_lookup_keys(record):
        if key in tag_lookup:
            return tag_lookup[key]
    return []


def gold_cypher_for_record(
    record: dict[str, Any],
    gold_cypher_lookup: dict[str, Any],
) -> Any:
    cypher = record_gold_cypher(record)
    if cypher:
        return cypher
    for key in record_lookup_keys(record):
        if key in gold_cypher_lookup:
            return gold_cypher_lookup[key]
    return ""


def question_detail(record: dict[str, Any], *, title: str | None = None) -> dict[str, Any]:
    return {
        "title": title or str(record.get("question") or record.get("uid") or "Question"),
        "fields": {
            "uid": record.get("uid"),
            "task_family": task_family(record),
            "output_type": output_type(record),
            "task_type": task_type(record),
            "question": record.get("question"),
            "tags": record.get("tags") or [],
        },
        "sections": {
            "gold_cypher": record_gold_cypher(record),
            **expected_output_blocks(record),
            "metadata": record.get("metadata"),
        },
    }


def split_question_row(
    record: dict[str, Any],
    split: str,
    optimized_lookup: dict[tuple[str, str], dict[str, Any]],
    tag_lookup: dict[str, list[str]],
) -> dict[str, Any]:
    uid = str(record.get("uid") or "")
    info = optimized_lookup.get((uid, split)) or optimized_from_record(record, split)
    kinds = optimization_kinds_from_info(info)
    return {
        "uid": record.get("uid"),
        "split": split,
        "task_family": task_family(record),
        "output_type": output_type(record),
        "task_type": task_type(record),
        "question": record.get("question"),
        "tags": tags_for_record(record, tag_lookup),
        "optimized": bool(kinds),
    }


def split_question_detail(
    record: dict[str, Any],
    split: str,
    optimized_lookup: dict[tuple[str, str], dict[str, Any]],
    tag_lookup: dict[str, list[str]],
    gold_cypher_lookup: dict[str, Any],
) -> dict[str, Any]:
    row = split_question_row(record, split, optimized_lookup, tag_lookup)
    metadata = record.get("metadata") or {}
    uid = str(record.get("uid") or "")
    info = optimized_lookup.get((uid, split)) or optimized_from_record(record, split) or {}
    kinds = optimization_kinds_from_info(info)
    tool_calls = (
        info.get("tool_call_count")
        or metadata.get("tool_call_count")
        or count_tool_calls(record)
    )
    return {
        "title": "Split Question Details",
        "fields": {
            key: value for key, value in row.items() if key != "question"
        }
        | {
            "optimization_type": kinds or ["none"],
            "tool_calls": tool_calls,
        },
        "blocks": {
            "question": record.get("question"),
            "gold_cypher": gold_cypher_for_record(record, gold_cypher_lookup),
            "model_cypher": record_model_cypher(record),
            **expected_output_blocks(record),
        },
        "show_empty_blocks": True,
        "sections": {
            "messages": record.get("messages"),
        },
    }


def merged_trace_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    traces = data["traces"]
    rows: dict[str, dict[str, Any]] = {}

    def key_for(record: dict[str, Any]) -> str:
        return str(
            record.get("uid")
            or record.get("seed_uid")
            or record.get("question")
            or id(record)
        )

    for record in traces["accepted_records"]:
        key = key_for(record)
        rows[key] = {
            "record": record,
            "uid": record.get("uid"),
            "seed_uid": record.get("seed_uid") or (record.get("metadata") or {}).get("seed_uid"),
            "task_family": task_family(record),
            "output_type": output_type(record),
            "task_type": task_type(record),
            "question": record.get("question"),
            "rejected": False,
            "reason": "",
            "validation": validation_status(record),
            "tool_calls": count_tool_calls(record),
            "final_answer": record.get("final_answer"),
        }

    for record in traces["rejected_records"]:
        key = key_for(record)
        rows[key] = {
            "record": record,
            "uid": record.get("uid"),
            "seed_uid": record.get("seed_uid") or (record.get("metadata") or {}).get("seed_uid"),
            "task_family": task_family(record),
            "output_type": output_type(record),
            "task_type": task_type(record),
            "question": record.get("question"),
            "rejected": True,
            "reason": record.get("reason") or "unknown",
            "validation": validation_status(record),
            "tool_calls": count_tool_calls(record),
            "final_answer": record.get("final_answer"),
        }

    for record in traces["teacher_records"]:
        key = key_for(record)
        if key not in rows:
            rows[key] = {
                "record": record,
                "uid": record.get("uid"),
                "seed_uid": record.get("seed_uid")
                or (record.get("metadata") or {}).get("seed_uid"),
                "task_family": task_family(record),
                "output_type": output_type(record),
                "task_type": task_type(record),
                "question": record.get("question"),
                "rejected": False,
                "reason": "",
                "validation": validation_status(record),
                "tool_calls": count_tool_calls(record),
                "final_answer": record.get("final_answer"),
            }
        else:
            rows[key]["record"] = rows[key]["record"] or record
    return list(rows.values())


def trace_detail(row: dict[str, Any], gold_cypher_lookup: dict[str, Any]) -> dict[str, Any]:
    record = row["record"]
    validation = record.get("validation") or {}
    return {
        "title": "Trace Details",
        "fields": {
            key: value
            for key, value in row.items()
            if key not in {
                "record",
                "in_teacher_jsonl",
                "question",
                "final_answer",
            }
        }
        | {
            key: validation.get(key)
            for key in (
                "tool_result_match",
                "final_answer_domain_valid",
                "answer_grounded",
            )
            if validation.get(key) is not None
        },
        "blocks": {
            "question": row.get("question"),
            "gold_cypher": gold_cypher_for_record(record, gold_cypher_lookup),
            "model_cypher": record_model_cypher(record),
            **expected_output_blocks(record),
            "answer": record.get("final_answer"),
        },
        "show_empty_blocks": True,
        "sections": {
            "messages": record.get("messages"),
            "validation": record.get("validation"),
            "metadata": record.get("metadata"),
        },
    }


def split_task_type_pivot(by_task_type: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    task_types = sorted(
        {
            task_type_name
            for counts in by_task_type.values()
            for task_type_name in counts
        }
    )
    rows = []
    for task_type_name in task_types:
        row = {"task_type": task_type_name}
        for split in SPLITS:
            row[split] = by_task_type.get(split, {}).get(task_type_name, 0)
        row["total"] = sum(int(row[split]) for split in SPLITS)
        rows.append(row)
    return rows


def split_task_family_pivot(
    by_task_family: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    families = sorted(
        {
            family
            for counts in by_task_family.values()
            for family in counts
        }
    )
    rows = []
    for family in families:
        row = {"task_family": family}
        for split in SPLITS:
            row[split] = by_task_family.get(split, {}).get(family, 0)
        row["total"] = sum(int(row[split]) for split in SPLITS)
        rows.append(row)
    return rows


def reports_dir(data_config: dict[str, Any]) -> Path:
    return Path(data_config["test_report_path"]).parent


def report_path(data_config: dict[str, Any], name: str) -> Path:
    return reports_dir(data_config) / name


def collect_report_files(data_config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_reports = reports_dir(data_config)
    directory = dataset_reports.parent if dataset_reports.name == "dataset" else dataset_reports
    if not directory.exists():
        return []
    return [file_info(path) for path in sorted(directory.rglob("*.report.json"))]


def report_quantization(report: dict[str, Any], path: Path) -> str:
    metadata = report.get("metadata") or {}
    raw_generation = metadata.get("raw_generation") or {}
    runtime = raw_generation.get("runtime") or {}
    model_details = runtime.get("model_details") or {}
    quantization = (
        model_details.get("quantization_level")
        or model_details.get("quantization")
    )
    if quantization:
        return str(quantization).strip().upper()

    model_name = str(runtime.get("model") or path.name)
    match = re.search(
        r"(?:^|[-_.])(BF16|F16|I?Q\d+(?:_[A-Z0-9]+)*)(?=$|[-.])",
        model_name,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else "Unknown"


def collect_quantization_reports(
    data_config: dict[str, Any],
    student_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    directory = (
        model_reports_dir(data_config, student_config)
        if student_config and student_config.get("model_version")
        else reports_dir(data_config)
    )
    reports = {}
    pattern = "challenge.student.ollama.*.raw-generation.report.json"
    legacy = directory
    for root in directory.parents:
        if root.name == "reports":
            legacy = root / "models" / directory.relative_to(root)
            break
    paths = {path.name: path for path in legacy.glob(pattern)}
    paths.update({path.name: path for path in directory.glob(pattern)})
    for path in sorted(paths.values()):
        report = read_json_if_exists(path)
        if not isinstance(report, dict):
            continue
        metadata = report.get("metadata") or {}
        if metadata.get("split") not in (None, "challenge"):
            continue
        quantization = report_quantization(report, path)
        key = quantization
        if key in reports:
            runtime = (metadata.get("raw_generation") or {}).get("runtime") or {}
            model_name = str(runtime.get("model") or path.stem)
            key = f"{quantization} ({model_name})"
        reports[key] = report
    return reports


def collect_gguf_artifacts(
    student_config: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    export = (student_config or {}).get("ollama_export") or {}
    export_root = export.get("export_root")
    model_name = export.get("model_name")
    if not export_root or not model_name:
        return {}
    directory = resolve_project_path(export_root) / str(model_name)
    if not directory.exists():
        return {}

    artifacts = {}
    for path in sorted(directory.glob("*.gguf")):
        quantization = report_quantization({}, path)
        artifacts[quantization] = file_info(path)
    return artifacts


def collect_trainer_state(student_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not student_config:
        return None
    output_dir = student_config.get("training", {}).get("output_dir")
    if not output_dir:
        return None
    resolved = resolve_project_path(output_dir)
    candidates = list(resolved.glob("trainer_state.json"))
    candidates.extend(resolved.glob("checkpoint-*/trainer_state.json"))
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    data = read_json_if_exists(latest)
    return {
        "file": file_info(latest),
        "state": data if isinstance(data, dict) else {},
    }


def collect_workflow_report_data(
    data_config: dict[str, Any],
    student_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scene_graphs = {}
    scene_artifacts = {}
    configured_scenes = data_config.get("scenes") or [
        {
            "scene_id": data_config.get("scene_id", "scene"),
            "scene_path": data_config.get("scene_path"),
            "scene_template_values_path": data_config["scene_template_values_path"],
        }
    ]
    for scene in configured_scenes:
        scene_values_doc = read_yaml_if_exists(scene["scene_template_values_path"])
        scene_graphs[scene["scene_id"]] = {
            "summary": scene_values_doc.get("scene_summary") or {},
            "values": scene_values_doc.get("values") or {},
            "raw": scene_values_doc,
        }
        scene_artifacts[scene["scene_id"]] = {
            "scene": file_info(scene["scene_path"]),
            "scene_template_values": file_info(scene["scene_template_values_path"]),
        }
    generated_questions_doc = read_yaml_if_exists(data_config["raw_questions_path"])
    generated_questions = generated_questions_doc.get("questions") or []
    seed_records = read_jsonl_if_exists(data_config["seed_records_jsonl_path"])
    teacher_traces = read_jsonl_if_exists(data_config["teacher_traces_jsonl_path"])
    accepted_traces = read_json_array_if_exists(data_config["accepted_traces_path"])
    rejected_traces = read_json_array_if_exists(data_config["rejected_traces_path"])
    split_records = {
        split: read_jsonl_if_exists(data_config["splits"][split]) for split in SPLITS
    }
    split_report = read_json_if_exists(report_path(data_config, "split.report.json"))
    selected_model_reports_dir = (
        model_reports_dir(data_config, student_config)
        if student_config and student_config.get("model_version")
        else reports_dir(data_config)
    )
    baseline_reports_dir = (
        model_reports_dir(data_config, student_config, baseline=True)
        if student_config and student_config.get("model_family")
        else reports_dir(data_config)
    )
    derived_optimized = [
        optimized
        for split, records in split_records.items()
        for record in records
        if (optimized := optimized_from_record(record, split)) is not None
    ]
    eval_loss_reports = {
        split: read_json_if_exists(
            selected_model_reports_dir / f"{split}.student.eval-loss.report.json"
        )
        for split in SPLITS
    }
    raw_generation_reports = {
        split: read_json_if_exists(
            selected_model_reports_dir / f"{split}.student.raw-generation.report.json"
        )
        for split in SPLITS
    }
    baseline_raw_generation_reports = {
        split: read_json_if_exists(
            baseline_reports_dir / f"{split}.baseline.raw-generation.report.json"
        )
        for split in SPLITS
    }
    quantization_raw_generation_reports = collect_quantization_reports(
        data_config, student_config
    )
    agent_reports = {
        split: read_json_if_exists(report_path(data_config, f"{split}.agent.report.json"))
        for split in SPLITS
    }
    tag_lookup = build_tag_lookup(
        generated_questions,
        seed_records,
        teacher_traces,
        accepted_traces,
        rejected_traces,
    )
    gold_cypher_lookup = build_gold_cypher_lookup(
        generated_questions,
        seed_records,
        teacher_traces,
        accepted_traces,
        rejected_traces,
        *split_records.values(),
    )

    return {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": data_config.get("dataset_id", data_config.get("scene_id")),
            "scene_ids": [scene["scene_id"] for scene in configured_scenes],
            "dataset_version": data_config.get("dataset_version"),
            "model_version": (student_config or {}).get("model_version"),
            "data_config": relative_path(data_config.get("data_config_path")),
        },
        "artifacts": {
            "scenes": scene_artifacts,
            "scene": file_info(configured_scenes[0].get("scene_path")),
            "scene_template_values": file_info(
                configured_scenes[0]["scene_template_values_path"]
            ),
            "generated_questions": file_info(data_config["raw_questions_path"]),
            "seed_records": file_info(data_config["seed_records_jsonl_path"]),
            "teacher_traces": file_info(data_config["teacher_traces_jsonl_path"]),
            "accepted_traces": file_info(data_config["accepted_traces_path"]),
            "rejected_traces": file_info(data_config["rejected_traces_path"]),
            "reports": collect_report_files(data_config),
            "splits": {
                split: file_info(data_config["splits"][split]) for split in SPLITS
            },
        },
        "scene_graphs": scene_graphs,
        "scene_graph": next(
            iter(scene_graphs.values()),
            {"summary": {}, "values": {}, "raw": {}},
        ),
        "generated_questions": {
            "records": generated_questions,
            "seed_records": seed_records,
            "by_task_type": count_by_task_type(generated_questions),
            "by_task_family": count_by_task_family(generated_questions),
            "tag_counts": count_tags(generated_questions),
        },
        "traces": {
            "teacher_records": teacher_traces,
            "accepted_records": accepted_traces,
            "rejected_records": rejected_traces,
            "teacher_by_task_type": count_by_task_type(teacher_traces),
            "teacher_by_task_family": count_by_task_family(teacher_traces),
            "accepted_by_task_type": count_by_task_type(accepted_traces),
            "accepted_by_task_family": count_by_task_family(accepted_traces),
            "rejected_by_task_type": count_by_task_type(rejected_traces),
            "rejected_by_task_family": count_by_task_family(rejected_traces),
            "rejected_reasons": dict(
                sorted(
                    Counter(
                        str(record.get("reason") or "unknown")
                        for record in rejected_traces
                    ).items()
                )
            ),
            "gold_cypher_lookup": gold_cypher_lookup,
        },
        "splits": {
            "records": split_records,
            "by_task_type": {
                split: count_by_task_type(records) for split, records in split_records.items()
            },
            "by_task_family": {
                split: count_by_task_family(records)
                for split, records in split_records.items()
            },
            "split_report": split_report,
            "optimized_questions": (
                (split_report or {}).get("optimized_questions") or derived_optimized
            ),
            "tag_lookup": tag_lookup,
            "gold_cypher_lookup": gold_cypher_lookup,
        },
        "train": {
            "student_config": student_config or {},
            "trainer_state": collect_trainer_state(student_config),
            "aggregate_eval_loss": read_json_if_exists(
                selected_model_reports_dir / "student.eval-loss.report.json"
            ),
            "eval_loss_reports": eval_loss_reports,
        },
        "validation": {
            "agent_reports": agent_reports,
            "raw_generation_reports": raw_generation_reports,
        },
        "baseline": {
            "raw_generation_reports": baseline_raw_generation_reports,
        },
        "deploy_quantization": {
            "raw_generation_reports": quantization_raw_generation_reports,
            "gguf_artifacts": collect_gguf_artifacts(student_config),
        },
    }


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def humanize_label(value: Any) -> str:
    text = str(value)
    if not text:
        return ""
    quantization_tokens: dict[str, str] = {}

    def protect_quantization(match: re.Match[str]) -> str:
        placeholder = f"QUANTIZATIONTOKEN{len(quantization_tokens)}X"
        quantization_tokens[placeholder] = match.group(0).upper().replace("_", "-")
        return placeholder

    text = re.sub(
        r"\bI?Q\d+(?:_[A-Z0-9]+)+\b",
        protect_quantization,
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("_", " ").replace("-", " ").replace("/", " / ")
    text = "".join(
        f" {char}" if index > 0 and char.isupper() and text[index - 1].islower() else char
        for index, char in enumerate(text)
    )
    parts = []
    for part in text.split():
        key = part.lower()
        parts.append(LABEL_OVERRIDES.get(key, part[:1].upper() + part[1:]))
    rendered = " ".join(parts)
    for placeholder, quantization in quantization_tokens.items():
        rendered = rendered.replace(placeholder, quantization)
    return rendered


def format_number(value: int | float, digits: int = 6) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def is_percentage_key(key: str | None) -> bool:
    if not key:
        return False
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    if normalized == "learning_rate":
        return False
    if normalized.endswith(("_static_valid", "_domain_valid")):
        return True
    tokens = set(normalized.split("_"))
    return any(
        token in tokens
        for token in (
            "accuracy",
            "executable",
            "match",
            "percent",
            "rate",
            "success",
        )
    )


def is_seconds_key(key: str | None) -> bool:
    if not key:
        return False
    normalized = key.lower()
    return normalized.endswith("_seconds") or "latency" in normalized


def is_bytes_key(key: str | None) -> bool:
    if not key:
        return False
    return "bytes" in key.lower()


def format_percent(value: int | float) -> str:
    numeric = float(value)
    if abs(numeric) <= 1:
        numeric *= 100
    return f"{numeric:.2f}%"


def format_bytes(value: int | float) -> str:
    numeric = float(value)
    units = ("B", "KiB", "MiB", "GiB")
    index = 0
    while abs(numeric) >= 1024 and index < len(units) - 1:
        numeric /= 1024
        index += 1
    if index == 0:
        return f"{int(numeric):,} {units[index]}"
    return f"{numeric:.1f} {units[index]}"


def format_cell(value: Any, key: str | None = None) -> str:
    if value is None:
        return ""
    if key == "quantization":
        return str(value).replace("_", "-")
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int | float):
        if is_percentage_key(key):
            return format_percent(value)
        if is_bytes_key(key):
            return format_bytes(value)
        formatted = format_number(value)
        if is_seconds_key(key):
            return f"{formatted}s"
        return formatted
    if isinstance(value, list | tuple | set):
        return ", ".join(format_cell(item) for item in value)
    if isinstance(value, dict):
        if all(not isinstance(item, list | tuple | set | dict) for item in value.values()):
            return ", ".join(
                f"{humanize_label(key)}: {format_cell(item)}"
                for key, item in value.items()
            )
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def boolean_class(key: str | None, value: bool) -> str:
    is_negative = bool(key and key.lower() in NEGATIVE_BOOLEAN_KEYS)
    is_good = (value and not is_negative) or (not value and is_negative)
    return "status-good" if is_good else "status-bad"


def render_status(value: str) -> str | None:
    normalized = value.lower()
    classes = {
        "pass": "status-good",
        "passed": "status-good",
        "success": "status-good",
        "ok": "status-good",
        "true": "status-good",
        "fail": "status-bad",
        "failed": "status-bad",
        "false": "status-bad",
        "error": "status-warn",
    }
    css_class = classes.get(normalized)
    if css_class is None:
        return None
    return f'<span class="status-pill {css_class}">{escape(value)}</span>'


def render_cell(value: Any, key: str | None = None) -> str:
    if isinstance(value, bool):
        return (
            f'<span class="status-pill {boolean_class(key, value)}">'
            f"{format_cell(value, key)}</span>"
        )
    text = field_text(value, key)
    escaped = escape(text)
    if key in {"path", "artifact"}:
        return escaped
    if key in PRE_TABLE_KEYS or "\n" in text or len(text) > 120:
        return f"<pre>{escaped}</pre>"
    return escaped


def sort_value(value: Any, key: str | None = None) -> str:
    if value is None:
        return ""
    if key == "sample":
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return str(value)
    return field_text(value, key)


def render_filter_row(headers: list[tuple[str, str]]) -> str:
    cells = "".join(
        "<th>"
        f'<input class="column-filter" type="search" data-filter-index="{index}" '
        f'placeholder="Filter {escape(humanize_label(label))}" '
        f'aria-label="Filter {escape(humanize_label(label))}">'
        "</th>"
        for index, (_key, label) in enumerate(headers)
    )
    return f'<tr class="filter-row">{cells}</tr>'


def render_table(
    headers: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    *,
    table_class: str = "",
    filterable: bool = False,
    collapsible_summary_rows: bool = False,
) -> str:
    if not rows:
        return '<p class="empty">No records found.</p>'
    head = "".join(
        f'<th data-sort-index="{index}">{escape(humanize_label(label))}</th>'
        for index, (_key, label) in enumerate(headers)
    )
    filters = render_filter_row(headers) if filterable else ""
    body = []
    for index, row in enumerate(rows):
        cells = "".join(
            f'<td data-sort-value="{escape(sort_value(row.get(key), key))}">'
            f"{render_cell(row.get(key), key)}</td>"
            for key, _label in headers
        )
        attributes = ""
        if collapsible_summary_rows and index > 0:
            attributes = ' class="question-type-row" hidden'
        body.append(f"<tr{attributes}>{cells}</tr>")
    return (
        '<div class="table-wrap"><table '
        f'class="data-table sortable {"filterable" if filterable else ""} '
        f'{escape(table_class)}"><thead><tr>'
        f"{head}</tr>{filters}</thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def render_question_types_toggle(question_type_count: int) -> str:
    if question_type_count <= 0:
        return ""
    return (
        '<button type="button" class="question-types-toggle" '
        'aria-expanded="false" '
        f'data-count="{question_type_count}">'
        f"Show question types ({question_type_count})"
        "</button>"
    )


def stable_detail_id(section_id: str, index: int, row: dict[str, Any]) -> str:
    key = str(row.get("uid") or row.get("question") or index)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{section_id}-{index}-{digest}"


def render_kv(fields: dict[str, Any]) -> str:
    items = []
    for key, value in fields.items():
        if value in (None, "", [], {}):
            continue
        items.append(
            f"<dt>{escape(humanize_label(key))}</dt><dd>{render_cell(value, key)}</dd>"
        )
    if not items:
        return ""
    return f'<dl class="kv">{"".join(items)}</dl>'


def render_pre_text(value: Any, *, preserve_newlines: bool = False) -> str:
    if isinstance(value, str):
        text = expanded_text(value) if preserve_newlines else clean_detail_text(value)
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return f"<pre>{escape(text)}</pre>"


def render_message_card(message: dict[str, Any], index: int) -> str:
    role = humanize_label(message.get("role") or f"Message {index}")
    tool_name = message.get("tool_name")
    heading = f"{index}. {role}"
    if tool_name:
        heading = f"{heading} / {tool_name}"

    parts = [
        '<div class="message-card">',
        f'<div class="message-role">{escape(heading)}</div>',
    ]
    if message.get("content") not in (None, ""):
        parts.append(render_pre_text(message.get("content"), preserve_newlines=True))
    if message.get("tool_calls"):
        parts.append("<h5>Tool Calls</h5>")
        parts.append(render_pre_text(message["tool_calls"], preserve_newlines=True))
    extra = {
        key: value
        for key, value in message.items()
        if key not in {"role", "content", "tool_calls", "tool_name"}
        and value not in (None, "", [], {})
    }
    if extra:
        parts.append("<h5>Metadata</h5>")
        parts.append(render_pre_text(extra, preserve_newlines=True))
    parts.append("</div>")
    return "".join(parts)


def render_messages_section(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    cards = [
        render_message_card(message, index)
        for index, message in enumerate(messages, start=1)
        if isinstance(message, dict)
    ]
    if not cards:
        return ""
    return '<div class="message-list">' + "".join(cards) + "</div>"


def render_detail_section(
    name: str,
    payload: Any,
    *,
    include_empty: bool = False,
) -> str:
    if not include_empty and payload in (None, "", [], {}):
        return ""
    title = escape(humanize_label(name))
    if name == "messages":
        messages = render_messages_section(payload)
        if messages:
            return f"<h4>{title}</h4>{messages}"
    render_payload = "" if payload in (None, "", [], {}) else payload
    rendered = render_pre_text(
        render_payload,
        preserve_newlines=(
            name in {"expected_cypher", "generated_cypher"}
            or name.endswith("_pddl_goal")
            or name.endswith(("_rendered_prompt", "_raw_completion"))
        ),
    )
    if name.endswith("_rendered_prompt"):
        return f"<details><summary>{title}</summary>{rendered}</details>"
    return f"<h4>{title}</h4>{rendered}"


def render_detail_blocks(blocks: dict[str, Any], *, include_empty: bool = False) -> str:
    return "".join(
        render_detail_section(name, payload, include_empty=include_empty)
        for name, payload in blocks.items()
    )


def render_detail_group(group: dict[str, Any]) -> str:
    title = group.get("title") or humanize_label(group.get("name") or "Details")
    blocks = group.get("blocks") or {}
    sections = group.get("sections") or {}
    content = render_detail_blocks(
        blocks,
        include_empty=bool(group.get("show_empty_blocks")),
    ) + "".join(
        render_detail_section(name, payload) for name, payload in sections.items()
    )
    open_attribute = " open" if group.get("open") else ""
    return (
        f'<details class="validation-detail-group"{open_attribute}>'
        f"<summary>{escape(str(title))}</summary>"
        f'<div class="validation-detail-group-body">{content}</div>'
        "</details>"
    )


def render_record_detail(detail: dict[str, Any]) -> str:
    title = detail.get("title") or "Record"
    fields = detail.get("fields") or {}
    blocks = detail.get("blocks") or {}
    sections = detail.get("sections") or {}
    groups = detail.get("groups") or []
    show_empty_blocks = bool(detail.get("show_empty_blocks"))
    section_html = []
    for name, payload in sections.items():
        section_html.append(render_detail_section(name, payload))
    return (
        f"<h3>{escape(title)}</h3>"
        + render_kv(fields)
        + render_detail_blocks(blocks, include_empty=show_empty_blocks)
        + "".join(section_html)
        + "".join(render_detail_group(group) for group in groups)
    )


def render_interactive_record_table(
    *,
    section_id: str,
    headers: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
    empty_message: str = "No records found.",
    filterable: bool = True,
    visibility_field: str | None = None,
    visible_values: set[str] | None = None,
) -> str:
    if not rows:
        return f'<p class="empty">{escape(empty_message)}</p>'
    head = "".join(
        f'<th data-sort-index="{index}">{escape(humanize_label(label))}</th>'
        for index, (_key, label) in enumerate(headers)
    )
    filters = render_filter_row(headers) if filterable else ""
    body = []
    templates = []
    first_visible_index = 0
    active_assigned = False
    for index, row in enumerate(rows):
        detail_id = stable_detail_id(section_id, index, row)
        cells = "".join(
            f'<td data-sort-value="{escape(sort_value(row.get(key), key))}">'
            f"{render_cell(row.get(key), key)}</td>"
            for key, _label in headers
        )
        group_value = str(row.get(visibility_field) or "") if visibility_field else ""
        visible = visible_values is None or group_value in visible_values
        active = ""
        if visible and not active_assigned:
            active = " active"
            active_assigned = True
            first_visible_index = index
        group_attribute = (
            f' data-report-group-value="{escape(group_value)}"'
            if visibility_field
            else ""
        )
        hidden_attribute = "" if visible else " hidden"
        body.append(
            f'<tr class="record-row{active}" data-detail-id="{escape(detail_id)}" '
            f'data-detail-target="{escape(section_id)}-detail"{group_attribute}'
            f"{hidden_attribute}>{cells}</tr>"
        )
        templates.append(
            f'<template id="{escape(detail_id)}">{render_record_detail(details[index])}</template>'
        )
    first_detail_id = stable_detail_id(
        section_id, first_visible_index, rows[first_visible_index]
    )
    return (
        '<div class="record-grid">'
        '<div class="table-wrap">'
        f'<table class="data-table sortable {"filterable" if filterable else ""}">'
        f"<thead><tr>{head}</tr>{filters}</thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
        "</div>"
        f'<aside id="{escape(section_id)}-detail" class="panel detail">'
        f"{render_record_detail(details[first_visible_index])}"
        "</aside>"
        "</div>"
        f'<!-- initial-detail:{escape(first_detail_id)} -->'
        + "".join(templates)
    )


def render_counter_bars(
    counter: dict[str, int],
    *,
    label: str,
    empty_message: str = "0 records",
) -> str:
    title = humanize_label(label)
    if not counter:
        return (
            '<section class="counter-panel">'
            f"<h3>{escape(title)}</h3>"
            f'<div class="counter-empty">{escape(empty_message)}</div>'
            "</section>"
        )
    max_count = max(counter.values()) or 1
    rows = []
    for key, count in counter.items():
        width = max(4, round((count / max_count) * 100))
        rows.append(
            '<div class="bar-row">'
            f'<div class="bar-label" title="{escape(key)}">{escape(key)}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width}%"></div>'
            "</div>"
            f'<div class="bar-count">{count}</div>'
            "</div>"
        )
    return (
        '<section class="counter-panel">'
        f"<h3>{escape(title)}</h3>"
        f'{"".join(rows)}'
        "</section>"
    )


def render_counter_grid(
    sections: list[tuple[str, dict[str, int]]],
    *,
    empty_message: str = "0 records",
) -> str:
    return (
        '<div class="summary-grid">'
        + "".join(
            render_counter_bars(counter, label=label, empty_message=empty_message)
            for label, counter in sections
        )
        + "</div>"
    )


def render_metric_grid(metrics: dict[str, Any]) -> str:
    cards = []
    for label, value in metrics.items():
        cards.append(
            '<div class="metric">'
            f'<div class="metric-label">{escape(humanize_label(label))}</div>'
            f'<div class="metric-value">{render_cell(value, label)}</div>'
            "</div>"
        )
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def render_json_details(title: str, payload: Any) -> str:
    if payload in (None, {}, []):
        return ""
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    return (
        "<details>"
        f"<summary>{escape(humanize_label(title))}</summary>"
        f"<pre>{escape(text)}</pre>"
        "</details>"
    )


def rate_value(report: dict[str, Any] | None, *path: str) -> Any:
    value: Any = report or {}
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def scene_class_label_from_values_key(key: str) -> str:
    label = key.removesuffix("_classes")
    mapping = {
        "object": "Object",
        "mesh_place": "MeshPlace",
        "room": "Room",
    }
    return mapping.get(label, humanize_label(label).replace(" ", ""))


def scene_class_sections(
    classes: dict[str, Any],
    values: dict[str, Any],
) -> list[tuple[str, dict[str, int]]]:
    labels = list(SCENE_CLASS_LABELS)
    for label in classes:
        if label not in labels:
            labels.append(label)
    for key in values:
        if not key.endswith("_classes"):
            continue
        label = scene_class_label_from_values_key(str(key))
        if label not in labels:
            labels.append(label)
    sections = []
    for label in labels:
        counts = classes.get(label) or {}
        sections.append((f"{label} Classes", counts if isinstance(counts, dict) else {}))
    return sections


def plural_scene_node_label(label: str) -> str:
    labels = {
        "Object": "Objects",
        "Place": "Places",
        "MeshPlace": "Mesh Places",
        "Room": "Rooms",
        "Building": "Buildings",
    }
    return labels.get(label, f"{humanize_label(label)}s")


def scene_graph_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    nodes = summary.get("nodes") or {}
    metrics: dict[str, Any] = {"edges": summary.get("edges", 0)}
    labels = list(nodes)
    if "Room" not in labels:
        labels.append("Room")
    for label in labels:
        metrics[plural_scene_node_label(str(label))] = nodes.get(label, 0)
    return metrics


def scene_template_sample_text(_name: str, value: Any) -> str:
    """Render template values as comma-separated scalar or dictionary samples."""
    def format_value(item: Any) -> str:
        if isinstance(item, dict):
            fields = ", ".join(
                f"{key}: {format_value(child)}" for key, child in item.items()
            )
            return "{" + fields + "}"
        if isinstance(item, list | tuple):
            return "[" + ", ".join(format_value(child) for child in item) + "]"
        if isinstance(item, set):
            ordered = sorted(item, key=str)
            return "{" + ", ".join(format_value(child) for child in ordered) + "}"
        if isinstance(item, bool):
            return str(item).lower()
        if item is None:
            return "null"
        return str(item)

    if isinstance(value, list | tuple):
        return ", ".join(format_value(item) for item in value)
    return format_value(value)


def render_scene_graph_panel(scene: dict[str, Any]) -> str:
    summary = scene["summary"]
    classes = summary.get("classes") or {}
    values = scene["values"]
    value_rows = [
        {
            "name": key,
            "count": len(value) if isinstance(value, list | dict) else "",
            "sample": scene_template_sample_text(str(key), value),
        }
        for key, value in values.items()
    ]
    class_sections = scene_class_sections(classes, values)
    return (
        render_metric_grid(scene_graph_metrics(summary))
        + render_counter_grid(class_sections, empty_message="0 classes")
        + "<h3>Template Values</h3>"
        + render_table(
            [("name", "name"), ("count", "count"), ("sample", "sample")],
            value_rows,
        )
        + render_json_details("Raw scene template values", scene["raw"])
    )


def render_scene_graph_tab(data: dict[str, Any]) -> str:
    scenes = data.get("scene_graphs") or {}
    if not scenes:
        return '<h2>3D Scene Graph</h2><p class="empty">No scenes found.</p>'
    scene_ids = list(scenes)
    selected = scene_ids[0]
    options = "".join(
        f'<option value="{escape(scene_id)}">{escape(scene_id)}</option>'
        for scene_id in scene_ids
    )
    selector = (
        '<label class="quantization-selector-label" for="scene-graph-selector">Scene'
        '<select id="scene-graph-selector" class="scene-graph-selector quantization-selector">'
        f"{options}</select></label>"
    )
    panels = "".join(
        f'<div class="scene-graph-group" data-scene-graph-value="{escape(scene_id)}"'
        f'{"" if scene_id == selected else " hidden"}>'
        f"{render_scene_graph_panel(scene)}</div>"
        for scene_id, scene in scenes.items()
    )
    return (
        '<div class="raw-tab-title"><h2>3D Scene Graph</h2>'
        f"{selector}</div>{panels}"
    )


def render_generated_questions_tab(data: dict[str, Any]) -> str:
    section = data["generated_questions"]
    rows = [compact_question_record(record) for record in section["records"]]
    supervision_counts = generated_question_supervision_counts(section["records"])
    details = [
        question_detail(record, title="Generated Question Details")
        for record in section["records"]
    ]
    return (
        "<h2>Generated Questions</h2>"
        + render_metric_grid(
            {
                "generated questions": len(section["records"]),
                "task types": len(section["by_task_type"]),
                "task families": len(section["by_task_family"]),
                **supervision_counts,
            }
        )
        + render_counter_grid(
            [
                ("Tags", section["tag_counts"]),
                ("Task Type", section["by_task_type"]),
                ("Task Family", section["by_task_family"]),
            ]
        )
        + "<h3>Questions</h3>"
        + render_interactive_record_table(
            section_id="generated-questions",
            headers=[
                ("uid", "uid"),
                ("task_family", "task_family"),
                ("task_type", "task_type"),
                ("tags", "tags"),
                ("question", "question"),
                ("gold_cypher", "gold_cypher"),
                ("solution", "solution"),
            ],
            rows=rows,
            details=details,
        )
    )


def render_traces_tab(data: dict[str, Any]) -> str:
    traces = data["traces"]
    rows = merged_trace_records(data)
    details = [trace_detail(row, traces["gold_cypher_lookup"]) for row in rows]
    return (
        "<h2>Teacher Traces</h2>"
        + render_metric_grid(
            {
                "accepted": len(traces["accepted_records"]),
                "rejected": len(traces["rejected_records"]),
            }
        )
        + render_counter_grid(
            [
                ("Accepted By Task Type", traces["accepted_by_task_type"]),
                ("Rejected By Task Type", traces["rejected_by_task_type"]),
                ("Rejected Reasons", traces["rejected_reasons"]),
            ]
        )
        + "<h3>Trace Outcomes</h3>"
        + render_interactive_record_table(
            section_id="trace-outcomes",
            headers=[
                ("uid", "uid"),
                ("task_family", "task_family"),
                ("task_type", "task_type"),
                ("question", "question"),
                ("rejected", "rejected"),
                ("reason", "reason"),
            ],
            rows=rows,
            details=details,
        )
    )


def render_splits_tab(data: dict[str, Any]) -> str:
    splits = data["splits"]
    records_by_split = splits["records"]
    optimized_lookup = optimization_lookup(splits["optimized_questions"])
    tag_lookup = splits["tag_lookup"]
    gold_cypher_lookup = splits["gold_cypher_lookup"]
    split_rows = [
        {
            "split": split,
            "questions": len(records),
            "task_types": len(splits["by_task_type"][split]),
            "optimized": sum(1 for record in records if optimized_from_record(record, split)),
        }
        for split, records in records_by_split.items()
    ]
    task_rows = split_task_type_pivot(splits["by_task_type"])
    family_rows = split_task_family_pivot(splits["by_task_family"])
    split_question_rows = [
        split_question_row(record, split, optimized_lookup, tag_lookup)
        for split, records in records_by_split.items()
        for record in records
    ]
    split_question_details = [
        split_question_detail(
            record,
            split,
            optimized_lookup,
            tag_lookup,
            gold_cypher_lookup,
        )
        for split, records in records_by_split.items()
        for record in records
    ]
    return (
        "<h2>Splits</h2>"
        + render_table(
            [
                ("split", "split"),
                ("questions", "questions"),
                ("task_types", "task_types"),
                ("optimized", "optimized"),
            ],
            split_rows,
        )
        + "<h3>Task Types Per Split</h3>"
        + render_table(
            [
                ("task_type", "task_type"),
                ("train", "train"),
                ("eval", "eval"),
                ("test", "test"),
                ("challenge", "challenge"),
                ("total", "total"),
            ],
            task_rows,
            table_class="equal-columns",
        )
        + "<h3>Task Families Per Split</h3>"
        + render_table(
            [
                ("task_family", "task_family"),
                ("train", "train"),
                ("eval", "eval"),
                ("test", "test"),
                ("challenge", "challenge"),
                ("total", "total"),
            ],
            family_rows,
            table_class="equal-columns",
        )
        + "<h3>Split Questions</h3>"
        + render_interactive_record_table(
            section_id="split-questions",
            headers=[
                ("uid", "uid"),
                ("split", "split"),
                ("task_family", "task_family"),
                ("task_type", "task_type"),
                ("question", "question"),
                ("tags", "tags"),
                ("optimized", "optimized"),
            ],
            rows=split_question_rows,
            details=split_question_details,
        )
        + render_json_details("Raw split report", splits["split_report"])
    )


def eval_loss_rows(train: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate = train["aggregate_eval_loss"]
    rows = []
    if isinstance(aggregate, dict):
        summary = aggregate.get("summary") or {}
        for split, split_summary in (summary.get("splits") or {}).items():
            rows.append({"split": split, **split_summary})
        if summary:
            rows.append(
                {
                    key: value
                    for key, value in {"split": "all", **summary}.items()
                    if key != "splits"
                }
            )
        return rows

    for split, report in train["eval_loss_reports"].items():
        if not isinstance(report, dict):
            continue
        summary = report.get("summary") or {}
        rows.append({"split": split, **summary})
    return rows


def render_train_tab(data: dict[str, Any]) -> str:
    train = data["train"]
    student_config = train["student_config"]
    trainer_state = train["trainer_state"]
    log_rows = []
    if trainer_state:
        log_rows = (trainer_state["state"].get("log_history") or [])[-25:]
    return (
        "<h2>Train</h2>"
        + "<h3>Eval Loss Reports</h3>"
        + render_table(
            [
                ("split", "split"),
                ("examples", "examples"),
                ("tokens", "tokens"),
                ("eval_loss", "eval_loss"),
                ("perplexity", "perplexity"),
                ("eval_entropy", "eval_entropy"),
                ("eval_mean_token_accuracy", "eval_mean_token_accuracy"),
            ],
            eval_loss_rows(train),
            table_class="equal-columns",
        )
        + "<h3>Student Config</h3>"
        + render_json_details("Student config", student_config)
        + "<h3>Trainer State Log</h3>"
        + render_table(
            [
                ("step", "step"),
                ("epoch", "epoch"),
                ("loss", "loss"),
                ("eval_loss", "eval_loss"),
                ("learning_rate", "learning_rate"),
            ],
            log_rows,
            table_class="equal-columns",
        )
        + render_json_details("Aggregate eval-loss report", train["aggregate_eval_loss"])
    )


def bool_rate(records: list[dict[str, Any]], section: str, key: str) -> float | None:
    evaluated = []
    for record in records:
        if not isinstance(record.get(section), dict) or record[section].get(key) is None:
            continue
        evaluated.append(bool_or_none(record[section][key]))
    if not evaluated:
        return None
    return sum(1 for value in evaluated if bool(value)) / len(evaluated)


def bool_rate_first(
    records: list[dict[str, Any]],
    section: str,
    keys: tuple[str, ...],
) -> float | None:
    evaluated = []
    for record in records:
        section_record = record.get(section)
        if not isinstance(section_record, dict):
            continue
        value = next(
            (section_record.get(key) for key in keys if section_record.get(key) is not None),
            None,
        )
        if value is not None:
            evaluated.append(bool_or_none(value))
    if not evaluated:
        return None
    return sum(1 for value in evaluated if bool(value)) / len(evaluated)


def record_hidden_false_positive_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    tool_call = record.get(TOOL_CALL_CHECK) or {}
    existing = tool_call.get("potential_hidden_false_positive")
    if existing is not None:
        return {
            "potential_hidden_false_positive": bool_or_none(existing),
            "hidden_false_positive_reasons": tool_call.get(
                "hidden_false_positive_reasons"
            )
            or [],
        }

    expected = record.get("expected") or {}
    generated = record.get("generated") or {}
    pddl = task_family(record) == "cypher_pddl"
    solution_match = tool_call.get(
        "grounding_match" if pddl else "cypher_solution_match",
        tool_call.get("tool_result_match", tool_call.get("cypher_solution_match")),
    )
    return result_match_diagnostics(
        generated_cypher=str(
            generated.get("cypher") or tool_call.get("generated_cypher") or ""
        ),
        gold_cypher=str(expected.get("cypher") or tool_call.get("gold_cypher") or ""),
        cypher_exact_match=bool(tool_call.get("cypher_exact_match")),
        solution_match=bool(solution_match),
        rows=(
            generated.get("cypher_tool_output")
            or (tool_call.get("validation") or {}).get("rows")
            or []
        ),
        expected_solution=expected.get(
            "cypher_solution", tool_call.get("expected_solution")
        ),
    )


def hidden_false_positive_rate(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return sum(
        1
        for record in records
        if record_hidden_false_positive_diagnostics(record)[
            "potential_hidden_false_positive"
        ]
    ) / len(records)


def raw_generation_summary_fallback_row(
    report: dict[str, Any],
    task_family_name: str,
) -> dict[str, Any]:
    summary = report.get("summary") or {}
    family_summary = (summary.get("by_task_family") or {}).get(task_family_name) or {}
    summary_section = family_summary or summary
    final_answer = summary_section.get(FINAL_ANSWER_CHECK) or {}
    tool_call = summary_section.get(TOOL_CALL_CHECK) or {}
    end_to_end = summary_section.get(END_TO_END_CHECK) or {}
    match_key = "grounding_match" if task_family_name == "cypher_pddl" else "cypher_solution_match"
    result_match = rate_value(tool_call, "rates", match_key)
    if result_match is None:
        result_match = rate_value(tool_call, "rates", "tool_result_match")
    if result_match is None:
        result_match = rate_value(tool_call, "rates", "cypher_solution_match")
    return {
        "task_type": "All",
        "examples": summary_section.get(
            "examples",
            final_answer.get("total", summary.get("examples")),
        ),
        "final_answer_match": rate_value(final_answer, "rates", "final_answer_match"),
        "end_to_end_match": rate_value(
            end_to_end, "rates", "final_answer_match"
        ),
        "final_answer_domain_valid": rate_value(
            final_answer, "rates", "final_answer_domain_valid"
        ),
        "tool_call_static_valid": rate_value(tool_call, "rates", "static_valid"),
        "tool_call_executable": rate_value(tool_call, "rates", "executable"),
        "result_match": result_match,
        "potential_hidden_false_positive": rate_value(
            tool_call, "rates", "potential_hidden_false_positive"
        ),
    }


def raw_generation_task_type_rows(
    report: dict[str, Any],
    task_family_name: str,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in report.get("records") or []
        if task_family(record) == task_family_name
    ]
    if not records:
        return [raw_generation_summary_fallback_row(report, task_family_name)]

    grouped: dict[str, list[dict[str, Any]]] = {"All": list(records)}
    for record in records:
        grouped.setdefault(task_type(record), []).append(record)

    rows = []
    for task_type_name, task_records in grouped.items():
        rows.append(
            {
                "task_type": task_type_name,
                "examples": len(task_records),
                "final_answer_match": bool_rate(
                    task_records,
                    FINAL_ANSWER_CHECK,
                    "final_answer_match",
                ),
                "end_to_end_match": bool_rate(
                    task_records,
                    END_TO_END_CHECK,
                    "final_answer_match",
                ),
                "final_answer_domain_valid": bool_rate(
                    task_records,
                    FINAL_ANSWER_CHECK,
                    "final_answer_domain_valid",
                ),
                "tool_call_static_valid": bool_rate(
                    task_records,
                    TOOL_CALL_CHECK,
                    "static_valid",
                ),
                "tool_call_executable": bool_rate(
                    task_records,
                    TOOL_CALL_CHECK,
                    "executable",
                ),
                "result_match": bool_rate_first(
                    task_records,
                    TOOL_CALL_CHECK,
                    (
                        "grounding_match"
                        if task_family_name == "cypher_pddl"
                        else "cypher_solution_match",
                        "tool_result_match",
                        "cypher_solution_match",
                    ),
                ),
                "potential_hidden_false_positive": hidden_false_positive_rate(
                    task_records
                ),
            }
        )
    return rows[:1] + sorted(rows[1:], key=lambda row: str(row["task_type"]))


def validation_example_rows(
    raw_generation_reports: dict[str, dict[str, Any] | None],
    *,
    detail_title: str = "Validation Example Details",
    group_field: str = "split",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    details = []
    for group_value, report in raw_generation_reports.items():
        if not isinstance(report, dict):
            continue
        for record in report.get("records") or []:
            family = task_family(record)
            pddl = family == "cypher_pddl"
            final_answer = record.get(FINAL_ANSWER_CHECK) or {}
            tool_call = record.get(TOOL_CALL_CHECK) or {}
            end_to_end = record.get(END_TO_END_CHECK) or {}
            hidden_false_positive = record_hidden_false_positive_diagnostics(record)
            result_match = bool_or_none(
                tool_call.get(
                    "grounding_match" if pddl else "cypher_solution_match",
                    tool_call.get(
                        "tool_result_match", tool_call.get("cypher_solution_match")
                    ),
                )
            )
            row = {
                "uid": record.get("uid"),
                group_field: group_value,
                "task_family": family,
                "output_type": output_type(record),
                "task_type": task_type(record),
                "question": record.get("question"),
                "tool_call_status": tool_call.get("status"),
                "final_answer_status": final_answer.get("status"),
                "tool_call_parse_success": bool_or_none(
                    tool_call.get("tool_call_parse_success")
                ),
                "tool_call_static_valid": bool_or_none(
                    tool_call.get("static_valid")
                ),
                "tool_call_executable": bool_or_none(tool_call.get("executable")),
                "cypher_solution_match": None if pddl else result_match,
                "grounding_match": result_match if pddl else None,
                "potential_hidden_false_positive": bool_or_none(
                    hidden_false_positive["potential_hidden_false_positive"]
                ),
                "final_answer_domain_valid": bool_or_none(
                    final_answer.get("final_answer_domain_valid")
                ),
                "final_answer_match": bool_or_none(
                    final_answer.get("final_answer_match")
                ),
                "end_to_end_match": bool_or_none(
                    end_to_end.get("final_answer_match")
                ),
            }
            rows.append(row)
            expected = record.get("expected") or {
                "cypher": tool_call.get("gold_cypher"),
                "cypher_solution": tool_call.get("expected_solution"),
                "final_answer": final_answer.get("expected_solution"),
            }
            generated = record.get("generated") or {
                "cypher": tool_call.get("generated_cypher"),
                "cypher_solution": tool_call.get("cypher_solution"),
                "cypher_tool_output": (
                    tool_call.get("validation") or {}
                ).get("rows"),
                "final_answer": final_answer.get("generated_text"),
                "final_answer_value": final_answer.get("generated_answer"),
            }
            if pddl:
                expected = {
                    **expected,
                    "pddl_goal": (
                        expected.get("pddl_goal")
                        or (
                            expected.get("cypher_solution")
                            if str(expected.get("cypher_solution") or "").strip()
                            else None
                        )
                    ),
                    "grounding": expected.get("grounding")
                    or tool_call.get("expected_grounding"),
                }
                generated = {
                    **generated,
                    "actual_grounding": generated.get("actual_grounding")
                    or tool_call.get("actual_grounding"),
                }
            generation_debug = record.get("generation_debug") or {}
            cypher_generation = generation_debug.get("cypher_generation") or {}
            final_answer_generation = (
                generation_debug.get("final_answer_generation") or {}
            )
            end_to_end_generation = (
                generation_debug.get("end_to_end_generation") or {}
            )
            # Read the previous report shape only for prompt/completion compatibility.
            legacy_conversations = record.get("conversations") or {}
            if not cypher_generation:
                cypher_generation = legacy_conversations.get("cypher_generation") or {}
            if not final_answer_generation:
                final_answer_generation = (
                    legacy_conversations.get("final_answer_generation") or {}
                )
            if not end_to_end_generation:
                end_to_end_generation = (
                    legacy_conversations.get("end_to_end_generation") or {}
                )
            cypher_validation_issues = {
                key: value
                for key, value in {
                    "static_issues": tool_call.get("static_issues"),
                    "parse_error": tool_call.get("parse_error"),
                    "execution_error": tool_call.get("execution_error"),
                    "result_shape_error": tool_call.get("result_shape_error"),
                    "generation_error": tool_call.get("generation_error"),
                    "hidden_false_positive_reasons": hidden_false_positive[
                        "hidden_false_positive_reasons"
                    ],
                }.items()
                if value not in (None, "", [], {})
            }
            final_answer_validation_issues = {
                key: value
                for key, value in {
                    "generation_error": final_answer.get("generation_error"),
                    "generated_answer_value": generated.get("final_answer_value"),
                }.items()
                if value not in (None, "", [], {})
            }
            end_to_end_validation_issues = {
                key: value
                for key, value in {
                    "last_failure_reason": end_to_end.get("last_failure_reason"),
                    "generation_error": end_to_end.get("generation_error"),
                }.items()
                if value not in (None, "", [], {})
            }
            cypher_blocks = {
                "expected_cypher": expected.get("cypher"),
                "generated_cypher": generated.get("cypher"),
                "cypher_tool_output": generated.get("cypher_tool_output"),
            }
            if pddl:
                cypher_blocks.update(
                    {
                        "expected_pddl_goal": expected.get("pddl_goal"),
                        "expected_grounding": expected.get("grounding"),
                        "actual_grounding": generated.get("actual_grounding"),
                        "cypher_validation_issues": cypher_validation_issues,
                        "grounding_validation_issues": {
                            "status": tool_call.get("status"),
                            "grounding_match": result_match,
                        }
                        if tool_call.get("status") != "pass" or result_match is False
                        else {},
                    }
                )
            else:
                cypher_blocks.update(
                    {
                        "expected_cypher_solution": expected.get("cypher_solution"),
                        "generated_cypher_solution": generated.get("cypher_solution"),
                        "cypher_validation_issues": cypher_validation_issues,
                    }
                )
            end_to_end_turns = end_to_end_generation.get("turns") or []
            end_to_end_sections = {}
            for turn_index, turn in enumerate(end_to_end_turns, start=1):
                if not isinstance(turn, dict):
                    continue
                turn_number = turn.get("turn") or turn_index
                end_to_end_sections.update(
                    {
                        f"turn_{turn_number}_rendered_prompt": turn.get(
                            "rendered_prompt"
                        ),
                        f"turn_{turn_number}_raw_completion": turn.get(
                            "raw_completion"
                        ),
                    }
                )
            details.append(
                {
                    "title": detail_title,
                    "fields": {
                        key: value for key, value in row.items() if key != "question"
                    },
                    "blocks": {"question": record.get("question")},
                    "show_empty_blocks": True,
                    "groups": [
                        {
                            "name": "cypher_validation",
                            "title": "Cypher Validation",
                            "blocks": cypher_blocks,
                            "show_empty_blocks": True,
                            "sections": {
                                "cypher_generation_rendered_prompt": (
                                    cypher_generation.get("rendered_prompt")
                                ),
                                "cypher_generation_raw_completion": (
                                    cypher_generation.get("raw_completion")
                                ),
                            },
                        },
                        {
                            "name": "final_answer_validation",
                            "title": "Final Answer Validation",
                            "blocks": {
                                "expected_final_answer": expected.get("final_answer"),
                                "generated_final_answer": generated.get("final_answer"),
                                "final_answer_validation_issues": (
                                    final_answer_validation_issues
                                ),
                            },
                            "show_empty_blocks": True,
                            "sections": {
                                "final_answer_generation_rendered_prompt": (
                                    final_answer_generation.get("rendered_prompt")
                                ),
                                "final_answer_generation_raw_completion": (
                                    final_answer_generation.get("raw_completion")
                                ),
                            },
                        },
                        {
                            "name": "end_to_end_validation",
                            "title": "End-to-End Validation",
                            "blocks": {
                                "expected_end_to_end_final_answer": expected.get(
                                    "final_answer"
                                ),
                                "generated_end_to_end_final_answer": generated.get(
                                    "end_to_end_final_answer"
                                )
                                or end_to_end.get("generated_text"),
                                "end_to_end_validation_issues": (
                                    end_to_end_validation_issues
                                ),
                                "end_to_end_attempts": end_to_end.get("attempts"),
                            },
                            "show_empty_blocks": True,
                            "sections": end_to_end_sections,
                        },
                    ],
                }
            )
    return rows, details


def validation_main_rows(
    raw_generation_reports: dict[str, dict[str, Any] | None],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows = {}
    for split, report in raw_generation_reports.items():
        if isinstance(report, dict):
            records = report.get("records") or []
            families = {task_family(record) for record in records}
            families.update(
                ((report.get("summary") or {}).get("by_task_family") or {}).keys()
            )
            if not families:
                families.add("cypher_qa")
            rows[split] = {
                family: raw_generation_task_type_rows(report, family)
                for family in sorted(families)
            }
    return rows


def raw_generation_rate(
    report: dict[str, Any] | None,
    section: str,
    metric: str,
) -> float | None:
    value = rate_value(report, "summary", section, "rates", metric)
    return float(value) if isinstance(value, int | float) else None


def percentage_point_gap(candidate: float | None, reference: float | None) -> str | None:
    if candidate is None or reference is None:
        return None
    return f"{(candidate - reference) * 100:+.2f} pp"


ComparisonMetric = tuple[str, Any, str | None, str]


def render_comparison_metrics(
    rows: list[tuple[str, list[ComparisonMetric]]],
) -> str:
    rendered_rows = []
    for row_title, cards in rows:
        rendered_cards = []
        for label, value, tone, note in cards:
            if value is None:
                continue
            tone_class = f" metric-{tone}" if tone else ""
            rendered_cards.append(
                f'<div class="metric comparison-metric{tone_class}">'
                '<div class="metric-label">'
                f"{escape(humanize_label(label))}"
                '<span class="metric-help-container">'
                '<button type="button" class="metric-help" '
                f'aria-label="Show metric help: {escape(note)}">?</button>'
                f'<span class="metric-tooltip" role="note">{escape(note)}</span>'
                "</span>"
                "</div>"
                f'<div class="metric-value">{render_cell(value, label)}</div>'
                "</div>"
            )
        if rendered_cards:
            rendered_rows.append(
                '<div class="comparison-metric-row">'
                f"<h4>{escape(row_title)}</h4>"
                f'<div class="metric-grid comparison-metric-grid">'
                f'{"".join(rendered_cards)}</div></div>'
            )
    if not rendered_rows:
        return ""
    return (
        '<section class="comparison-metrics">'
        '<h3>Comparison Metrics</h3>'
        f'{"".join(rendered_rows)}'
        "</section>"
    )


def gap_tone(value: str | None) -> str | None:
    if value is None or value.startswith("+0.00") or value.startswith("-0.00"):
        return "neutral"
    return "good" if value.startswith("+") else "bad"


END_TO_END_RATE_NOTE = (
    "End-to-end match = correct final answers from the complete tool-using run "
    "÷ evaluated questions."
)


def validation_comparison_metrics(
    student_reports: dict[str, dict[str, Any] | None],
) -> str:
    cards: list[ComparisonMetric] = []
    for split in ("test", "challenge"):
        rate = raw_generation_rate(
            student_reports.get(split), END_TO_END_CHECK, "final_answer_match"
        )
        cards.append(
            (f"{split} student end-to-end match", rate, None, END_TO_END_RATE_NOTE)
        )
    return render_comparison_metrics([("End-to-End Quality", cards)])


def baseline_comparison_metrics(
    student_reports: dict[str, dict[str, Any] | None],
    baseline_reports: dict[str, dict[str, Any] | None],
) -> str:
    rows = []
    for split in ("test", "challenge"):
        student_rate = raw_generation_rate(
            student_reports.get(split), END_TO_END_CHECK, "final_answer_match"
        )
        baseline_rate = raw_generation_rate(
            baseline_reports.get(split), END_TO_END_CHECK, "final_answer_match"
        )
        gap = percentage_point_gap(student_rate, baseline_rate)
        rows.append(
            (
                humanize_label(split),
                [
                    (
                        f"{split} baseline end-to-end match",
                        baseline_rate,
                        None,
                        END_TO_END_RATE_NOTE,
                    ),
                    (
                        f"{split} fine-tuned end-to-end match",
                        student_rate,
                        None,
                        END_TO_END_RATE_NOTE,
                    ),
                    (
                        f"{split} fine-tune end-to-end gap",
                        gap,
                        gap_tone(gap),
                        "Fine-tune gap = fine-tuned end-to-end match − baseline "
                        "end-to-end match. Positive values favor the fine-tuned model.",
                    ),
                ],
            )
        )
    return render_comparison_metrics(rows)


def end_to_end_generation_throughput(report: dict[str, Any] | None) -> float | None:
    generated_tokens = 0.0
    eval_duration_ns = 0.0
    for record in (report or {}).get("records") or []:
        attempts = ((record.get(END_TO_END_CHECK) or {}).get("attempts") or [])
        turns = (
            ((record.get("generation_debug") or {}).get("end_to_end_generation") or {})
            .get("turns")
            or []
        )
        for attempt, turn in zip(attempts, turns, strict=False):
            tokens = attempt.get("generated_tokens")
            duration = (turn.get("provider_metadata") or {}).get("eval_duration")
            if isinstance(tokens, int | float) and isinstance(duration, int | float):
                if tokens >= 0 and duration > 0:
                    generated_tokens += float(tokens)
                    eval_duration_ns += float(duration)
    if eval_duration_ns <= 0:
        return None
    return generated_tokens / (eval_duration_ns / 1_000_000_000)


def quantization_report(
    reports: dict[str, dict[str, Any] | None], quantization: str
) -> dict[str, Any] | None:
    expected = quantization.upper()
    for label, report in reports.items():
        if label.split(" (", 1)[0].upper() == expected and isinstance(report, dict):
            return report
    return None


def quantization_artifact(
    artifacts: dict[str, dict[str, Any]], quantization: str
) -> dict[str, Any] | None:
    expected = quantization.upper()
    for label, artifact in artifacts.items():
        if label.upper() == expected:
            return artifact
    return None


def deploy_comparison_metrics(
    data: dict[str, Any], selected_quantization: str
) -> str:
    deploy_data = data["deploy_quantization"]
    reports = deploy_data["raw_generation_reports"]
    transformers = data["validation"]["raw_generation_reports"].get("challenge")
    bf16 = quantization_report(reports, "BF16")
    selected_report = reports.get(selected_quantization)
    quantization = selected_quantization.split(" (", 1)[0]
    quantization_label = humanize_label(quantization)
    transformers_rate = raw_generation_rate(
        transformers, END_TO_END_CHECK, "final_answer_match"
    )
    bf16_rate = raw_generation_rate(bf16, END_TO_END_CHECK, "final_answer_match")
    selected_rate = raw_generation_rate(
        selected_report, END_TO_END_CHECK, "final_answer_match"
    )
    deployment_gap = percentage_point_gap(bf16_rate, transformers_rate)
    quantization_gap = percentage_point_gap(selected_rate, bf16_rate)

    artifacts = deploy_data.get("gguf_artifacts") or {}
    bf16_artifact = quantization_artifact(artifacts, "BF16") or {}
    selected_artifact = quantization_artifact(artifacts, quantization) or {}
    bf16_bytes = bf16_artifact.get("bytes")
    selected_bytes = selected_artifact.get("bytes")
    size_reduction = None
    if (
        isinstance(bf16_bytes, int | float)
        and isinstance(selected_bytes, int | float)
        and bf16_bytes > 0
    ):
        size_reduction = (
            f"{(bf16_bytes - selected_bytes) / bf16_bytes * 100:.2f}%"
        )

    bf16_throughput = end_to_end_generation_throughput(bf16)
    selected_throughput = end_to_end_generation_throughput(selected_report)
    speed_change = None
    speed_tone = None
    if bf16_throughput and selected_throughput is not None:
        speed_ratio = selected_throughput / bf16_throughput
        if speed_ratio >= 1:
            speed_change = f"{speed_ratio:.2f}× faster"
            speed_tone = "good" if speed_ratio > 1 else "neutral"
        elif speed_ratio > 0:
            speed_change = f"{1 / speed_ratio:.2f}× slower"
            speed_tone = "bad"

    throughput_note = (
        "End-to-end generation throughput = total generated tokens across all "
        "end-to-end Ollama turns ÷ summed Ollama eval_duration. Prompt evaluation, "
        "model loading, Neo4j execution, and other request overhead are excluded."
    )
    size_note = "GGUF size is the local artifact file size on disk."
    rows: list[tuple[str, list[ComparisonMetric]]] = [
        (
            "End-to-End Quality",
            [
                (
                    "Transformers end-to-end match",
                    transformers_rate,
                    None,
                    END_TO_END_RATE_NOTE,
                ),
                ("BF16 end-to-end match", bf16_rate, None, END_TO_END_RATE_NOTE),
                (
                    f"{quantization} end-to-end match",
                    selected_rate,
                    None,
                    END_TO_END_RATE_NOTE,
                ),
            ],
        ),
        (
            "Quality Gaps",
            [
                (
                    "Deployment end-to-end gap",
                    deployment_gap,
                    gap_tone(deployment_gap),
                    "Deployment gap = Ollama BF16 end-to-end match − Transformers "
                    "end-to-end match. Negative values indicate degradation.",
                ),
                (
                    "Quantization end-to-end gap",
                    quantization_gap,
                    gap_tone(quantization_gap),
                    f"Quantization gap = Ollama {quantization_label} end-to-end "
                    "match − Ollama BF16 end-to-end match. Negative values indicate "
                    "degradation.",
                ),
            ],
        ),
        (
            "GGUF Artifact Size",
            [
                (
                    "BF16 GGUF size",
                    format_bytes(bf16_bytes) if bf16_bytes is not None else None,
                    None,
                    size_note,
                ),
                (
                    f"{quantization} GGUF size",
                    (
                        format_bytes(selected_bytes)
                        if selected_bytes is not None
                        else None
                    ),
                    None,
                    size_note,
                ),
                (
                    f"{quantization} GGUF size reduction",
                    size_reduction,
                    "good" if size_reduction is not None else None,
                    f"{quantization_label} size reduction = (BF16 bytes − "
                    f"{quantization_label} bytes) ÷ BF16 bytes.",
                ),
            ],
        ),
        (
            "End-to-End Generation Throughput",
            [
                (
                    "BF16 throughput",
                    f"{bf16_throughput:.2f} tok/s" if bf16_throughput is not None else None,
                    None,
                    throughput_note,
                ),
                (
                    f"{quantization} throughput",
                    (
                        f"{selected_throughput:.2f} tok/s"
                        if selected_throughput is not None
                        else None
                    ),
                    None,
                    throughput_note,
                ),
                (
                    f"{quantization} speed change",
                    speed_change,
                    speed_tone,
                    f"{quantization_label} speed ratio = {quantization_label} tok/s "
                    "÷ BF16 tok/s. Ratios above 1 are reported as that many times "
                    "faster; ratios below 1 use the reciprocal and are reported as "
                    "that many times slower.",
                ),
            ],
        ),
    ]
    return render_comparison_metrics(rows)


def render_raw_generation_tab(
    raw_generation_reports: dict[str, dict[str, Any] | None],
    *,
    title: str,
    summary_title: str,
    section_id: str,
    group_field: str = "split",
    comparison_metrics: str = "",
    show_title: bool = True,
    visible_report_groups: set[str] | None = None,
) -> str:
    raw_rows_by_split = validation_main_rows(raw_generation_reports)
    example_rows, example_details = validation_example_rows(
        raw_generation_reports,
        detail_title=f"{title} Example Details",
        group_field=group_field,
    )
    summary_tables = []
    for split, family_rows in raw_rows_by_split.items():
        family_tables = []
        family_order = {"cypher_qa": 0, "cypher_pddl": 1}
        for family in sorted(
            family_rows,
            key=lambda name: (family_order.get(name, 2), str(name)),
        ):
            rows = family_rows[family]
            result_label = (
                "grounding_match" if family == "cypher_pddl" else "cypher_solution_match"
            )
            heading = (
                '<div class="raw-family-heading">'
                f"<h5>{escape(humanize_label(family))}</h5>"
                f"{render_question_types_toggle(len(rows) - 1)}"
                "</div>"
            )
            family_tables.append(
                '<section class="raw-family-summary">'
                '<div class="collapsible-summary-table">'
                + heading
                + render_table(
                    [
                        ("task_type", "task_type"),
                        ("examples", "examples"),
                        ("tool_call_executable", "tool_call_executable"),
                        ("result_match", result_label),
                        ("final_answer_match", "final_answer_match"),
                        ("end_to_end_match", "end_to_end_match"),
                    ],
                    rows,
                    table_class="equal-columns",
                    collapsible_summary_rows=True,
                )
                + "</div></section>"
            )
        group_attribute = ""
        hidden_attribute = ""
        if visible_report_groups is not None:
            group_attribute = f' data-report-group-value="{escape(split)}"'
            hidden_attribute = "" if split in visible_report_groups else " hidden"
        summary_tables.append(
            f'<section class="raw-summary-group"{group_attribute}{hidden_attribute}>'
            f"<h4>{escape(humanize_label(split))}</h4>"
            f'<div class="raw-family-grid">{"".join(family_tables)}</div>'
            "</section>"
        )
    examples = (
        '<section class="raw-examples">'
        f"<h3>{escape(title)} Examples</h3>"
        + render_interactive_record_table(
            section_id=f"{section_id}-examples",
            headers=[
                ("uid", "uid"),
                (group_field, group_field),
                ("task_family", "task_family"),
                ("task_type", "task_type"),
                ("question", "question"),
                ("tool_call_executable", "tool_call_executable"),
                ("cypher_solution_match", "cypher_solution_match"),
                ("grounding_match", "grounding_match"),
                ("final_answer_match", "final_answer_match"),
                ("end_to_end_match", "end_to_end_match"),
            ],
            rows=example_rows,
            details=example_details,
            visibility_field=(group_field if visible_report_groups is not None else None),
            visible_values=visible_report_groups,
        )
        + "</section>"
    )
    if visible_report_groups is None:
        raw_report_details = render_json_details(
            f"{title} raw generation reports", raw_generation_reports
        )
    else:
        raw_report_details = "".join(
            f'<div class="raw-report-json" data-report-group-value="{escape(group)}"'
            f'{"" if group in visible_report_groups else " hidden"}>'
            f'{render_json_details(f"{title} {group} raw generation report", report)}'
            "</div>"
            for group, report in raw_generation_reports.items()
        )
    return (
        (f"<h2>{escape(title)}</h2>" if show_title else "")
        + comparison_metrics
        + f"<h3>{escape(summary_title)}</h3>"
        + ("".join(summary_tables) if summary_tables else '<p class="empty">No records found.</p>')
        + examples
        + raw_report_details
    )


def render_validation_tab(data: dict[str, Any]) -> str:
    return render_raw_generation_tab(
        data["validation"]["raw_generation_reports"],
        title="Validation",
        summary_title="Student Raw Generation Summary",
        section_id="validation",
        comparison_metrics=validation_comparison_metrics(
            data["validation"]["raw_generation_reports"]
        ),
    )


def render_baseline_tab(data: dict[str, Any]) -> str:
    return render_raw_generation_tab(
        data["baseline"]["raw_generation_reports"],
        title="Baseline",
        summary_title="Base Student Raw Generation Summary",
        section_id="baseline",
        comparison_metrics=baseline_comparison_metrics(
            data["validation"]["raw_generation_reports"],
            data["baseline"]["raw_generation_reports"],
        ),
    )


def render_deploy_quantization_tab(data: dict[str, Any]) -> str:
    reports = data["deploy_quantization"]["raw_generation_reports"]
    bf16_key = next(
        (label for label in reports if label.split(" (", 1)[0].upper() == "BF16"),
        None,
    )
    selectable = [
        label
        for label in reports
        if label.split(" (", 1)[0].upper() != "BF16"
    ]
    selected = selectable[0] if selectable else None
    visible_groups = {value for value in (bf16_key, selected) if value is not None}

    selector = ""
    comparison_metrics = ""
    if selectable:
        options = "".join(
            f'<option value="{escape(label)}">{escape(humanize_label(label))}</option>'
            for label in selectable
        )
        selector = (
            '<label class="quantization-selector-label" '
            'for="deploy-quantization-selector">Quantization'
            '<select id="deploy-quantization-selector" class="quantization-selector" '
            f'data-reference-value="{escape(bf16_key or "")}">{options}</select>'
            "</label>"
        )
        comparison_metrics = "".join(
            f'<div class="quantization-metrics" data-report-metrics-value="{escape(label)}"'
            f'{"" if label == selected else " hidden"}>'
            f"{deploy_comparison_metrics(data, label)}</div>"
            for label in selectable
        )

    title = (
        '<div class="raw-tab-title">'
        "<h2>Deploy / Quantization</h2>"
        f"{selector}</div>"
    )
    return title + render_raw_generation_tab(
        reports,
        title="Deploy / Quantization",
        summary_title="Challenge Raw Generation Summary",
        section_id="deploy-quantization",
        group_field="quantization",
        comparison_metrics=comparison_metrics,
        show_title=False,
        visible_report_groups=visible_groups,
    )


def artifact_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """List each file once, retaining the first (most specific) artifact label."""
    artifacts = data["artifacts"]
    rows = []
    for key, value in artifacts.items():
        if key == "reports":
            rows.extend(
                {"artifact": f"report:{Path(info['path']).name}", **info} for info in value
            )
        elif key == "splits":
            rows.extend({"artifact": f"split:{split}", **info} for split, info in value.items())
        elif key == "scenes":
            for scene_id, scene_files in value.items():
                rows.extend(
                    {"artifact": f"scene:{scene_id}:{name}", **info}
                    for name, info in scene_files.items()
                )
        else:
            rows.append({"artifact": key, **value})
    unique_rows = []
    seen_paths = set()
    for row in rows:
        path = row.get("path")
        if path:
            canonical_path = (PROJECT_ROOT / path).resolve()
            if canonical_path in seen_paths:
                continue
            seen_paths.add(canonical_path)
        unique_rows.append(row)
    names = Counter(row["artifact"] for row in unique_rows)
    for row in unique_rows:
        if names[row["artifact"]] > 1:
            row["artifact"] = f"{row['artifact']}:{row['path']}"
    return unique_rows


def render_artifacts(data: dict[str, Any]) -> str:
    return render_table(
        [
            ("artifact", "artifact"),
            ("path", "path"),
            ("exists", "exists"),
            ("bytes", "bytes"),
            ("modified_at", "modified_at"),
        ],
        artifact_rows(data),
    )


def render_artifacts_tab(data: dict[str, Any]) -> str:
    return "<h2>Artifacts</h2>" + render_artifacts(data)


def render_workflow_report(data: dict[str, Any]) -> str:
    dataset_id = data["metadata"].get("dataset_id") or "unknown"
    tabs = [
        ("scene", "3D Scene Graph", render_scene_graph_tab(data)),
        ("questions", "Generated Questions", render_generated_questions_tab(data)),
        ("traces", "Traces", render_traces_tab(data)),
        ("splits", "Splits", render_splits_tab(data)),
        ("train", "Train", render_train_tab(data)),
        ("validation", "Validation", render_validation_tab(data)),
        ("baseline", "Baseline", render_baseline_tab(data)),
    ]
    quantization_reports = data.get("deploy_quantization", {}).get(
        "raw_generation_reports"
    )
    if quantization_reports:
        tabs.append(
            (
                "deploy-quantization",
                "Deploy / Quantization",
                render_deploy_quantization_tab(data),
            )
        )
    tabs.append(("artifacts", "Artifacts", render_artifacts_tab(data)))
    buttons = "".join(
        f'<button class="tab-button{" active" if index == 0 else ""}" '
        f'data-tab="{escape(tab_id)}">{escape(label)}</button>'
        for index, (tab_id, label, _content) in enumerate(tabs)
    )
    panels = "".join(
        f'<section id="{escape(tab_id)}" class="tab-panel{" active" if index == 0 else ""}">'
        f"{content}</section>"
        for index, (tab_id, _label, content) in enumerate(tabs)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3DSG Workflow Report - {escape(dataset_id)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #657282;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-soft: #e1f3f1;
      --good: #117a37;
      --good-bg: #e7f6ec;
      --bad: #b42318;
      --bad-bg: #fde7e4;
      --warn: #a15c00;
      --warn-bg: #fff3d6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .page-header {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 0 rgba(15, 23, 42, 0.03);
    }}
    header {{
      padding: 24px 28px 14px;
      background: var(--panel);
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 20px; }}
    h3 {{ font-size: 15px; }}
    h4 {{ font-size: 14px; margin: 18px 0 8px; }}
    .raw-tab-title {{
      align-items: center;
      display: flex;
      gap: 16px;
      justify-content: space-between;
      margin-bottom: 12px;
    }}
    .raw-tab-title h2 {{ margin: 0; }}
    .quantization-selector-label {{
      align-items: center;
      color: var(--muted);
      display: flex;
      font-size: 12px;
      font-weight: 650;
      gap: 8px;
    }}
    .quantization-selector {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 5px;
      color: var(--text);
      font: inherit;
      min-width: 130px;
      padding: 6px 30px 6px 9px;
    }}
    .subtitle {{ color: var(--muted); }}
    .tabs {{
      display: flex;
      gap: 6px;
      padding: 0 28px;
      background: var(--panel);
      overflow-x: auto;
    }}
    .tab-button {{
      appearance: none;
      border: 1px solid transparent;
      border-bottom: 0;
      background: transparent;
      color: var(--muted);
      padding: 10px 12px;
      font: inherit;
      cursor: pointer;
    }}
    .tab-button.active {{
      color: var(--accent);
      background: var(--accent-soft);
      border-color: var(--line);
    }}
    main {{ padding: 22px 28px 34px; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin: 12px 0 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px;
    }}
    .metric-good {{ border-top: 3px solid var(--good); }}
    .metric-bad {{ border-top: 3px solid var(--bad); }}
    .metric-neutral {{ border-top: 3px solid var(--muted); }}
    .metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .metric-value {{ font-size: 20px; font-weight: 650; margin-top: 4px; }}
    .comparison-metrics {{ margin: 14px 0 24px; }}
    .comparison-metric-row {{ margin-top: 14px; }}
    .comparison-metric-row h4 {{ color: var(--muted); margin: 0 0 7px; }}
    .comparison-metric-grid {{
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      margin: 0;
    }}
    .comparison-metric {{ position: relative; }}
    .comparison-metric .metric-label {{
      align-items: center;
      display: flex;
      gap: 6px;
      justify-content: space-between;
    }}
    .metric-help-container {{ display: inline-flex; position: relative; }}
    .metric-help {{
      align-items: center;
      background: transparent;
      border: 1px solid var(--line);
      border-radius: 50%;
      color: inherit;
      cursor: help;
      display: inline-flex;
      flex: 0 0 17px;
      font: inherit;
      font-size: 11px;
      height: 17px;
      justify-content: center;
      padding: 0;
      text-transform: none;
      width: 17px;
    }}
    .metric-tooltip {{
      background: #17202b;
      border-radius: 5px;
      color: #fff;
      display: none;
      font-size: 12px;
      font-weight: 400;
      right: -1px;
      line-height: 1.4;
      max-width: 420px;
      min-width: min(300px, calc(100vw - 80px));
      padding: 9px 10px;
      position: absolute;
      text-transform: none;
      top: calc(100% + 6px);
      z-index: 30;
    }}
    .metric-help:hover + .metric-tooltip,
    .metric-help:focus + .metric-tooltip {{ display: block; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin: 12px 0 18px;
    }}
    .counter-panel, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 12px;
    }}
    .raw-summary-group {{ margin: 18px 0 24px; }}
    .raw-family-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .raw-family-summary {{ min-width: 0; }}
    .raw-family-heading {{
      align-items: center;
      display: flex;
      gap: 10px;
      justify-content: space-between;
      margin-bottom: 8px;
      min-height: 29px;
    }}
    .raw-family-heading h5 {{ color: var(--muted); margin: 0; }}
    .raw-family-summary .table-wrap {{ margin-bottom: 0; }}
    .question-types-toggle {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 5px;
      color: var(--accent);
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 650;
      padding: 5px 9px;
    }}
    .question-types-toggle:hover {{ background: var(--accent-soft); }}
    .raw-examples {{ margin-top: 26px; }}
    .counter-empty {{
      color: var(--muted);
      background: #f5f7fa;
      border: 1px dashed var(--line);
      padding: 12px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(120px, 1fr) minmax(100px, 2fr) 48px;
      gap: 8px;
      align-items: center;
      margin: 7px 0;
    }}
    .bar-label {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--text);
    }}
    .bar-track {{
      height: 10px;
      background: #edf1f5;
      border: 1px solid var(--line);
    }}
    .bar-fill {{
      height: 100%;
      background: var(--accent);
    }}
    .bar-count {{
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      margin-bottom: 14px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    table.equal-columns {{ table-layout: fixed; }}
    th, td {{
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
    }}
    th {{
      background: #eef1f5;
      cursor: pointer;
      font-size: 13px;
      font-weight: 750;
      color: #394656;
      z-index: 1;
      white-space: nowrap;
      user-select: none;
    }}
    th.sort-asc::after {{ content: " ▲"; color: var(--accent); }}
    th.sort-desc::after {{ content: " ▼"; color: var(--accent); }}
    .filter-row th {{
      background: var(--panel);
      cursor: default;
      padding: 6px 8px;
    }}
    .column-filter {{
      width: 100%;
      min-width: 96px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--text);
      padding: 5px 7px;
      font: inherit;
      font-size: 12px;
    }}
    td {{
      max-width: 360px;
      overflow-wrap: anywhere;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f2f4f7;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      max-height: 260px;
      overflow: auto;
      margin: 0;
      font: inherit;
    }}
    .record-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 440px);
      gap: 14px;
      align-items: start;
    }}
    .detail {{
      position: sticky;
      top: 128px;
      max-height: calc(100vh - 148px);
      overflow: auto;
    }}
    .detail h4 {{ margin: 16px 0 8px; font-size: 13px; }}
    .detail h5 {{ margin: 10px 0 6px; font-size: 12px; color: var(--muted); }}
    .kv {{
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      gap: 6px 12px;
      margin: 0 0 12px;
    }}
    .kv dt {{ color: var(--muted); font-weight: 650; }}
    .kv dd {{ margin: 0; min-width: 0; }}
    .status-pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 1px 8px 2px;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.5;
      white-space: nowrap;
    }}
    .status-good {{
      color: var(--good);
      background: var(--good-bg);
    }}
    .status-bad {{
      color: var(--bad);
      background: var(--bad-bg);
    }}
    .status-warn {{
      color: var(--warn);
      background: var(--warn-bg);
    }}
    .record-row {{ cursor: pointer; }}
    .record-row:hover {{ background: #f2f6f5; }}
    .record-row.active {{ background: var(--accent-soft); }}
    .message-list {{
      display: grid;
      gap: 10px;
    }}
    .message-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px 12px;
    }}
    .message-role {{
      font-weight: 700;
      margin-bottom: 8px;
      color: #394656;
    }}
    .message-card pre + h5 {{ margin-top: 12px; }}
    details {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 10px 12px;
      margin: 14px 0;
    }}
    summary {{ cursor: pointer; font-weight: 650; }}
    .validation-detail-group {{ margin: 10px 0; }}
    .validation-detail-group > summary {{
      color: var(--accent);
      font-size: 14px;
    }}
    .validation-detail-group-body {{
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 2px;
    }}
    .validation-detail-group-body > details {{ margin: 10px 0; }}
    .empty {{ color: var(--muted); }}
    @media (max-width: 980px) {{
      .record-grid {{ grid-template-columns: 1fr; }}
      .detail {{ position: static; max-height: none; }}
      .raw-family-grid {{ grid-template-columns: 1fr; }}
      .raw-tab-title {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class="page-header">
    <header>
      <h1>3DSG Workflow Report: {escape(dataset_id)}</h1>
      <div class="subtitle">Generated at {escape(data["metadata"]["created_at"])}</div>
    </header>
    <nav class="tabs" aria-label="Workflow report tabs">{buttons}</nav>
  </div>
  <main>
    {panels}
  </main>
  <script>
    const buttons = document.querySelectorAll(".tab-button");
    const panels = document.querySelectorAll(".tab-panel");
    buttons.forEach((button) => {{
      button.addEventListener("click", () => {{
        const tab = button.dataset.tab;
        buttons.forEach((item) => item.classList.toggle("active", item === button));
        panels.forEach((panel) => panel.classList.toggle("active", panel.id === tab));
      }});
    }});
    function compareSortValues(left, right, direction) {{
      const leftMissing = left === "";
      const rightMissing = right === "";
      if (leftMissing && rightMissing) return 0;
      if (leftMissing) return 1;
      if (rightMissing) return -1;
      const leftNumber = Number(left);
      const rightNumber = Number(right);
      if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {{
        return (leftNumber - rightNumber) * direction;
      }}
      return left.localeCompare(right, undefined, {{ numeric: true }}) * direction;
    }}
    function selectRecordRow(row) {{
      if (!row) return;
      const template = document.getElementById(row.dataset.detailId);
      const target = document.getElementById(row.dataset.detailTarget);
      if (!template || !target) return;
      const table = row.closest("table");
      if (table) {{
        table.querySelectorAll(".record-row").forEach((item) => {{
          item.classList.toggle("active", item === row);
        }});
      }}
      target.innerHTML = template.innerHTML;
    }}
    function firstVisibleRecordRow(table) {{
      return Array.from(table.querySelectorAll(".record-row[data-detail-id]"))
        .find((row) => !row.hidden && row.style.display !== "none");
    }}
    function applyColumnFilters(table) {{
      const filters = Array.from(table.querySelectorAll(".column-filter"));
      const rows = Array.from(table.tBodies[0].rows);
      rows.forEach((row) => {{
        const visible = filters.every((filter) => {{
          const term = filter.value.trim().toLowerCase();
          if (!term) return true;
          const cell = row.cells[Number(filter.dataset.filterIndex)];
          if (!cell) return false;
          const value = `${{cell.dataset.sortValue || ""}} ${{cell.textContent || ""}}`;
          return value.toLowerCase().includes(term);
        }});
        row.style.display = visible ? "" : "none";
      }});
      const active = table.querySelector(".record-row.active");
      if (active && !active.hidden && active.style.display !== "none") return;
      selectRecordRow(firstVisibleRecordRow(table));
    }}
    function applyReportGroupSelection(selector) {{
      const panel = selector.closest(".tab-panel");
      if (!panel) return;
      const visible = new Set([selector.dataset.referenceValue, selector.value]);
      panel.querySelectorAll(
        ".raw-summary-group[data-report-group-value], " +
        ".raw-report-json[data-report-group-value]"
      )
        .forEach((group) => {{
          group.hidden = !visible.has(group.dataset.reportGroupValue);
        }});
      panel.querySelectorAll(".quantization-metrics[data-report-metrics-value]")
        .forEach((metrics) => {{
          metrics.hidden = metrics.dataset.reportMetricsValue !== selector.value;
        }});
      const tables = new Set();
      panel.querySelectorAll(".record-row[data-report-group-value]").forEach((row) => {{
        row.hidden = !visible.has(row.dataset.reportGroupValue);
        const table = row.closest("table");
        if (table) tables.add(table);
      }});
      tables.forEach((table) => applyColumnFilters(table));
    }}
    document
      .querySelectorAll(".quantization-selector:not(.scene-graph-selector)")
      .forEach((selector) => {{
      selector.addEventListener("change", () => applyReportGroupSelection(selector));
      }});
    document.querySelectorAll(".scene-graph-selector").forEach((selector) => {{
      selector.addEventListener("change", () => {{
        const panel = selector.closest(".tab-panel");
        if (!panel) return;
        panel.querySelectorAll(".scene-graph-group[data-scene-graph-value]")
          .forEach((group) => {{
            group.hidden = group.dataset.sceneGraphValue !== selector.value;
          }});
      }});
    }});
    document.querySelectorAll("table.sortable th[data-sort-index]").forEach((header) => {{
      header.addEventListener("click", () => {{
        const table = header.closest("table");
        const body = table && table.tBodies[0];
        if (!body) return;
        const index = Number(header.dataset.sortIndex);
        const direction = header.classList.contains("sort-asc") ? -1 : 1;
        table.querySelectorAll("th").forEach((item) => {{
          item.classList.remove("sort-asc", "sort-desc");
          item.removeAttribute("aria-sort");
        }});
        header.classList.toggle("sort-asc", direction === 1);
        header.classList.toggle("sort-desc", direction === -1);
        header.setAttribute("aria-sort", direction === 1 ? "ascending" : "descending");
        const rows = Array.from(body.rows);
        rows.sort((left, right) => {{
          const leftCell = left.cells[index];
          const rightCell = right.cells[index];
          const leftValue = (leftCell && leftCell.dataset.sortValue) || "";
          const rightValue = (rightCell && rightCell.dataset.sortValue) || "";
          return compareSortValues(leftValue, rightValue, direction);
        }});
        rows.forEach((row) => body.appendChild(row));
      }});
    }});
    document.querySelectorAll("table.filterable .column-filter").forEach((input) => {{
      input.addEventListener("input", () => {{
        const table = input.closest("table");
        if (table) applyColumnFilters(table);
      }});
    }});
    document.querySelectorAll(".question-types-toggle").forEach((button) => {{
      button.addEventListener("click", () => {{
        const tableWrap = button.closest(".collapsible-summary-table");
        if (!tableWrap) return;
        const expanded = button.getAttribute("aria-expanded") !== "true";
        tableWrap.querySelectorAll(".question-type-row").forEach((row) => {{
          row.hidden = !expanded;
        }});
        button.setAttribute("aria-expanded", String(expanded));
        const action = expanded ? "Hide" : "Show";
        button.textContent = `${{action}} question types (${{button.dataset.count}})`;
      }});
    }});
    document.querySelectorAll(".record-row[data-detail-id]").forEach((row) => {{
      row.addEventListener("click", () => {{
        selectRecordRow(row);
      }});
    }});
  </script>
</body>
</html>
"""


def write_workflow_report(
    data_config: dict[str, Any],
    student_config: dict[str, Any] | None,
    output: str | Path,
) -> Path:
    data = collect_workflow_report_data(data_config, student_config)
    output_path = ensure_parent(output)
    output_path.write_text(render_workflow_report(data), encoding="utf-8")
    return output_path


def command(args: argparse.Namespace) -> None:
    data_config = load_data_config(
        args.data_config, dataset_version=args.dataset_version
    )
    student_config = None
    if args.student_config:
        student_config = load_student_config(
            args.student_config, model_version=args.model_version
        )
        if not confirm_dataset_lineage(
            model_metadata=read_metadata(student_config["model_metadata_path"]),
            data_config=data_config,
            allow_mismatch=args.allow_dataset_mismatch,
        ):
            return
    output_path = write_workflow_report(
        data_config,
        student_config,
        workflow_report_output_path(data_config, args.output, student_config),
    )
    print_summary(
        "Workflow report generated",
        [
            ("Dataset", data_config["dataset_id"]),
            ("Scenes", len(data_config["scenes"])),
            ("Dataset version", data_config["dataset_version"]),
            ("Model version", (student_config or {}).get("model_version", "none")),
            ("Output", output_path),
            ("Size", f"{output_path.stat().st_size / (1024 * 1024):.1f} MiB"),
            ("Open", f"file://{output_path}"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML report for a configured 3DSG dataset."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument(
        "--student-config",
        default="configs/students/student_qwen3.5_4b.yaml",
        help="Optional student config used to show training settings and trainer state.",
    )
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--allow-dataset-mismatch", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional HTML output path. Defaults to "
            "the selected dataset/model report directory."
        ),
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
