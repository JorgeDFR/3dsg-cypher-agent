from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from typing import Any

from neo4j import GraphDatabase, Query
from neo4j.exceptions import Neo4jError

from cypher_3dsg_agent_distillation.config import ensure_parent, load_yaml
from cypher_3dsg_agent_distillation.io import read_jsonl

FORBIDDEN_SUBSTRINGS = ("apoc.", " distance(")
WRITE_KEYWORDS = ("CREATE ", "MERGE ", "DELETE ", "DETACH ", "SET ", "REMOVE ", "DROP ")
SAFE_REFUSALS = {"// WRITE_QUERY_NOT_ALLOWED", "WRITE_QUERY_NOT_ALLOWED"}
DEFAULT_HERACLES_NEO4J_URI = "neo4j://127.0.0.1:7683"
DEFAULT_HERACLES_NEO4J_USERNAME = "neo4j"
DEFAULT_HERACLES_NEO4J_PASSWORD = "neo4j_pw"
DEFAULT_HERACLES_NEO4J_DATABASE = "neo4j"
NEO4J_NOTIFICATION_LOGGER = "neo4j.notifications"


def schema_policy(schema_config: dict[str, Any]) -> dict[str, Any]:
    labels = set(schema_config.get("labels", {}))
    relationships = set(schema_config.get("relationships", {}))
    properties = {
        property_name
        for label in schema_config.get("labels", {}).values()
        for property_name in label.get("properties", [])
    }
    safety = schema_config.get("safety", {})
    return {
        "labels": labels,
        "relationships": relationships,
        "properties": properties,
        "forbidden_functions": set(safety.get("forbidden_functions", [])),
        "forbidden_write_keywords": tuple(
            f"{keyword.upper()} " for keyword in safety.get("forbidden_write_keywords", [])
        )
        or WRITE_KEYWORDS,
        "require_bounded_variable_length_paths": bool(
            safety.get("require_bounded_variable_length_paths", True)
        ),
    }


def is_safe_refusal(cypher: str) -> bool:
    return cypher.strip() in SAFE_REFUSALS


def result_hash(rows: list[dict[str, Any]]) -> str:
    normalized = json.dumps(rows, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def suppress_neo4j_notification_logs() -> None:
    logging.getLogger(NEO4J_NOTIFICATION_LOGGER).setLevel(logging.ERROR)


def static_checks(
    cypher: str,
    allow_writes: bool,
    policy: dict[str, Any] | None = None,
) -> list[str]:
    if is_safe_refusal(cypher):
        return []

    policy = policy or {}
    upper = f" {cypher.upper()} "
    issues = []
    for substring in FORBIDDEN_SUBSTRINGS:
        if substring in cypher.lower():
            issues.append(f"forbidden substring: {substring.strip()}")
    if re.search(r"(^|[^.A-Za-z_])distance\s*\(", cypher, flags=re.IGNORECASE):
        issues.append("forbidden function: distance")
    if not allow_writes:
        for keyword in policy.get("forbidden_write_keywords", WRITE_KEYWORDS):
            if keyword in upper:
                issues.append(f"write keyword not allowed: {keyword.strip()}")

    allowed_labels = policy.get("labels")
    if allowed_labels:
        for label in sorted(set(re.findall(r":`?([A-Za-z_][A-Za-z0-9_]*)`?", cypher))):
            if label not in allowed_labels and label not in policy.get("relationships", set()):
                issues.append(f"unknown label: {label}")

    allowed_relationships = policy.get("relationships")
    if allowed_relationships:
        for relationship in sorted(set(re.findall(r"\[:`?([A-Z_][A-Z0-9_]*)`?", cypher))):
            if relationship not in allowed_relationships:
                issues.append(f"unknown relationship: {relationship}")

    allowed_properties = policy.get("properties")
    if allowed_properties:
        property_matches = re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.`?([A-Za-z_][A-Za-z0-9_]*)`?",
            cypher,
        )
        for variable_name, property_name in sorted(set(property_matches)):
            if variable_name == "point":
                continue
            if property_name not in allowed_properties:
                issues.append(f"unknown property: {property_name}")

    if policy.get("require_bounded_variable_length_paths", True):
        for variable_path in re.findall(r"\[[^\]]*\*[^\]]*\]", cypher):
            if ".." not in variable_path:
                issues.append(f"unbounded variable-length path: {variable_path}")
            elif re.search(r"\*\d*\.\.\s*\]", variable_path):
                issues.append(f"missing upper bound on variable-length path: {variable_path}")
    return issues


class Neo4jValidator:
    def __init__(self, uri: str, username: str, password: str, database: str, timeout: float):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.timeout = timeout

    def close(self) -> None:
        self.driver.close()

    def validate(self, cypher: str, execute: bool) -> dict[str, Any]:
        started = time.perf_counter()
        query_text = cypher if execute else f"EXPLAIN\n{cypher}"
        try:
            records, summary, _ = self.driver.execute_query(
                Query(query_text, timeout=self.timeout),
                database_=self.database,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            rows = [record.data() for record in records] if execute else []
            return {
                "syntax_error": False,
                "timeout": False,
                "executable": True,
                "returns_results": bool(records),
                "elapsed_ms": elapsed_ms,
                "notifications": [str(item) for item in (summary.notifications or [])],
                "rows": rows,
                "result_hash": result_hash(rows) if execute else None,
                "error": None,
            }
        except Neo4jError as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            code = getattr(exc, "code", "") or ""
            return {
                "syntax_error": "SyntaxError" in code,
                "timeout": "TransactionTimedOut" in code or "ClientError.Transaction" in code,
                "executable": False,
                "returns_results": False,
                "elapsed_ms": elapsed_ms,
                "notifications": [],
                "rows": [],
                "result_hash": None,
                "error": f"{code}: {exc}",
            }


def neo4j_connection_settings(args: argparse.Namespace) -> dict[str, str]:
    return {
        "uri": args.uri or DEFAULT_HERACLES_NEO4J_URI,
        "username": args.username or DEFAULT_HERACLES_NEO4J_USERNAME,
        "password": args.password or DEFAULT_HERACLES_NEO4J_PASSWORD,
        "database": args.database or DEFAULT_HERACLES_NEO4J_DATABASE,
    }


def validate_records(
    records: list[dict[str, Any]],
    cypher_field: str,
    execute: bool,
    allow_writes: bool,
    validator: Neo4jValidator | None,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validated = []
    for record in records:
        cypher = record.get(cypher_field, "")
        record_allows_writes = allow_writes or not bool(record.get("read_only", True))
        issues = static_checks(cypher, record_allows_writes, policy=policy)
        result = {
            "uid": record.get("uid"),
            "question": record.get("question"),
            "cypher": cypher,
            "static_issues": issues,
            "static_valid": not issues,
            "refusal": is_safe_refusal(cypher),
        }
        if validator is not None and not issues and not is_safe_refusal(cypher):
            result.update(validator.validate(cypher, execute=execute))
        validated.append({**record, "validation": result})
    return validated


def command(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    policy = schema_policy(load_yaml(args.schema_config)) if args.schema_config else None
    validator = None
    if not args.static_only:
        connection = neo4j_connection_settings(args)
        validator = Neo4jValidator(
            uri=connection["uri"],
            username=connection["username"],
            password=connection["password"],
            database=connection["database"],
            timeout=args.timeout,
        )
    try:
        validated = validate_records(
            records=records,
            cypher_field=args.cypher_field,
            execute=args.execute,
            allow_writes=args.allow_writes,
            validator=validator,
            policy=policy,
        )
    finally:
        if validator is not None:
            validator.close()

    output_path = ensure_parent(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Cypher records against static rules and Neo4j."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cypher-field", default="cypher")
    parser.add_argument("--schema-config", default="configs/schema_3dsg.yaml")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute queries instead of using EXPLAIN.",
    )
    parser.add_argument("--allow-writes", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--uri", default=None, help=f"Default: {DEFAULT_HERACLES_NEO4J_URI}")
    parser.add_argument(
        "--username",
        default=None,
        help=f"Default: {DEFAULT_HERACLES_NEO4J_USERNAME}",
    )
    parser.add_argument(
        "--password",
        default=None,
        help=f"Default: {DEFAULT_HERACLES_NEO4J_PASSWORD}",
    )
    parser.add_argument(
        "--database",
        default=None,
        help=f"Default: {DEFAULT_HERACLES_NEO4J_DATABASE}",
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
