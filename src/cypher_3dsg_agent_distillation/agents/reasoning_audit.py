"""Offline compilation, migration and coverage inspection for synthetic reasoning."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from cypher_3dsg_agent_distillation.agents.reasoning_config import (
    DEFAULT_TEMPLATES,
    default_semantics_path,
    load_catalog,
)
from cypher_3dsg_agent_distillation.agents.synthetic_reasoning import (
    reasoning_sentences,
    select_sentences,
)
from cypher_3dsg_agent_distillation.cypher.task_semantics import (
    REASONING_RENDERER_VERSION,
    SEMANTICS_VERSION,
    enrich_seed_semantics,
    parse_query,
)


def compile_templates(
    path: str | Path, *, output_path: str | Path | None = None, write: bool = False
):
    """Compile a separate synthetic configuration; never modify question files."""
    path = Path(path)
    catalog = yaml.safe_load(path.read_text())["templates"]
    declarations = {}
    for template in catalog:
        query_type = template["query_type"]
        if query_type in declarations:
            raise ValueError(f"Duplicate query_type: {query_type}")
        declarations[query_type] = {"operations": parse_query(template["cypher"])["operations"]}
    result = yaml.safe_dump(
        {"version": SEMANTICS_VERSION, "templates": declarations},
        sort_keys=False,
        default_flow_style=None,
        width=95,
    )
    if write:
        destination = Path(output_path) if output_path is not None else default_semantics_path(path)
        if destination.resolve() == path.resolve():
            raise ValueError("Synthetic reasoning output must be separate from question templates")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result)
    return result


def migrate_seeds(seeds, catalog, template_mapping=None):
    by_type = {template["query_type"]: template for template in catalog}
    if len(by_type) != len(catalog):
        raise ValueError("Migration requires unique query_type names")
    mapping = template_mapping or {}
    migrated = []
    diagnostics = []
    for original in seeds:
        seed = deepcopy(original)
        metadata = seed.setdefault("metadata", {})
        context = metadata.get("reasoning_context") or {}
        query_type = mapping.get(seed["query_type"], seed["query_type"])
        template = by_type.get(query_type)
        try:
            if template is None and not context.get("retrieval"):
                raise ValueError("No unambiguous source template; supply a template mapping")
            enrich_seed_semantics(seed, template)
        except ValueError as exc:
            metadata["reasoning_migration"] = {"status": "unresolved", "error": str(exc)}
            diagnostics.append(
                {"uid": seed["uid"], "query_type": seed["query_type"], "error": str(exc)}
            )
        migrated.append(seed)
    return migrated, diagnostics


def audit_seeds(
    seeds, catalog, *, traces=None, caps=(None, 1, 2, 4), template_mapping=None, tokenizer=None
):
    seeds, diagnostics = migrate_seeds(seeds, catalog, template_mapping)
    trace_map = {str(trace["uid"]).removesuffix("_teacher"): trace for trace in traces or []}
    report: dict[str, Any] = {
        "semantics_version": SEMANTICS_VERSION,
        "renderer_version": REASONING_RENDERER_VERSION,
        "template_count": len(catalog),
        "seed_count": len(seeds),
        "verified_seeds": len(seeds) - len(diagnostics),
        "diagnostics": diagnostics,
        "validation": "offline structure and saved tool feedback; no fresh database execution",
        "by_type": {},
        "rules": {},
        "caps": {},
        "examples": [],
    }
    rules = Counter()
    lengths = []
    tokens = []
    stage_lengths = {"tool": [], "final": []}
    stage_tokens = {"tool": [], "final": []}
    for seed in seeds:
        entry = report["by_type"].setdefault(
            seed["query_type"], {"seeds": 0, "verified": 0, "tool_rules": [], "final_rules": []}
        )
        entry["seeds"] += 1
        if seed["metadata"].get("reasoning_migration", {}).get("status") == "unresolved":
            continue
        entry["verified"] += 1
        trace = trace_map.get(seed["uid"])
        feedback = (trace or {}).get("validation", {})
        rows = feedback.get("rows")
        if rows is None and trace:
            for message in reversed(trace["messages"]):
                if message["role"] == "tool":
                    try:
                        payload = json.loads(message["content"])
                        if payload.get("ok"):
                            rows = payload.get("rows")
                            break
                    except (ValueError, TypeError):
                        pass
        refusal = "WRITE_QUERY_NOT_ALLOWED" in seed["cypher"]
        stages = [] if refusal else ["tool"]
        if rows is not None or refusal:
            stages.append("final")
        example = {
            "uid": seed["uid"],
            "query_type": seed["query_type"],
            "question": seed["question"],
        }
        observed_repair = False
        if trace:
            for message in trace.get("messages", []):
                for call in message.get("tool_calls") or []:
                    arguments = call["function"]["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    observed_repair |= (
                        arguments.get("cypher_string", "").strip()
                        == (seed["metadata"].get("repair_cypher") or "").strip()
                    )
        for stage in stages:
            candidates = reasoning_sentences(
                seed,
                stage=stage,
                rows=rows,
                repair_template=(
                    seed["metadata"].get("repair_template")
                    if stage == "tool" and observed_repair
                    else None
                ),
            )
            text, selection = select_sentences(candidates, cap=None)
            example[stage] = text
            lengths.append(len(text.split()))
            stage_lengths[stage].append(lengths[-1])
            if tokenizer:
                tokens.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
                stage_tokens[stage].append(tokens[-1])
            rules.update(item.rule.split(":")[0] for item in candidates)
            entry[stage + "_rules"] = sorted(
                set(entry[stage + "_rules"]) | {item.rule.split(":")[0] for item in candidates}
            )
            for cap in caps:
                cap_report = report["caps"].setdefault(
                    str(cap), {"turns": 0, "turns_with_critical_omissions": 0}
                )
                cap_report["turns"] += 1
                _, capped = select_sentences(candidates, cap=cap)
                cap_report["turns_with_critical_omissions"] += bool(
                    capped["critical_omitted_rules"]
                )
        if not any(e["query_type"] == seed["query_type"] for e in report["examples"]):
            report["examples"].append(example)
    report["rules"] = dict(rules)
    report["types_without_seeds"] = sorted(
        {template["query_type"] for template in catalog} - set(report["by_type"])
    )
    report["reasoning_words_per_turn"] = {
        "median": statistics.median(lengths) if lengths else 0,
        "max": max(lengths, default=0),
    }
    report["reasoning_by_stage"] = {}
    for stage, values in stage_lengths.items():
        metrics = {
            "turns": len(values),
            "turns_with_reasoning": sum(value > 0 for value in values),
            "words": {
                "mean": statistics.mean(values) if values else 0,
                "median": statistics.median(values) if values else 0,
                "max": max(values, default=0),
                "total": sum(values),
            },
        }
        if tokenizer:
            values = stage_tokens[stage]
            metrics["tokens"] = {
                "mean": statistics.mean(values) if values else 0,
                "median": statistics.median(values) if values else 0,
                "max": max(values, default=0),
                "total": sum(values),
            }
        report["reasoning_by_stage"][stage] = metrics
    if tokenizer:
        report["reasoning_tokens_per_turn"] = {
            "median": statistics.median(tokens) if tokens else 0,
            "max": max(tokens, default=0),
        }
    return report, seeds


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["compile", "audit", "migrate"])
    parser.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    parser.add_argument(
        "--semantics", nargs="+", help="Separate reasoning files, in the same order as --templates"
    )
    parser.add_argument(
        "--write", action="store_true", help="Refresh separate synthetic reasoning files"
    )
    parser.add_argument("--seeds")
    parser.add_argument("--traces")
    parser.add_argument(
        "--template-mapping", help="JSON mapping from legacy query_type names to current names"
    )
    parser.add_argument("--output", help="New JSON audit report or migrated JSONL file")
    parser.add_argument(
        "--tokenizer", help="Optional local tokenizer directory for exact reasoning lengths"
    )
    args = parser.parse_args(argv)
    if args.semantics and len(args.semantics) != len(args.templates):
        parser.error("--semantics requires one file per --templates entry")
    if args.command == "compile":
        for index, path in enumerate(args.templates):
            content = compile_templates(
                path,
                output_path=args.semantics[index] if args.semantics else None,
                write=args.write,
            )
            if not args.write:
                print(content)
        return
    catalog = load_catalog(args.templates, args.semantics)
    if not args.seeds:
        if args.command == "migrate":
            parser.error("migrate requires --seeds and --output")
        print(f"Validated {len(catalog)} semantic template declarations.")
        return
    mapping = json.loads(Path(args.template_mapping).read_text()) if args.template_mapping else None
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report, seeds = audit_seeds(
        _read_jsonl(args.seeds),
        catalog,
        traces=_read_jsonl(args.traces) if args.traces else None,
        template_mapping=mapping,
        tokenizer=tokenizer,
    )
    if args.command == "migrate":
        if not args.output:
            parser.error("migrate requires --output; input datasets are never overwritten")
        _write_new(args.output, "".join(json.dumps(seed) + "\n" for seed in seeds))
    elif args.output:
        _write_new(args.output, json.dumps(report, indent=2) + "\n")
    print(
        f"Templates: {len(catalog)}; verified seeds: {report['verified_seeds']}/{len(seeds)}; "
        f"migration diagnostics: {len(report['diagnostics'])}."
    )
    for diagnostic in report["diagnostics"][:5]:
        print(f"{diagnostic['uid']}: {diagnostic['error']}")


if __name__ == "__main__":
    main()
