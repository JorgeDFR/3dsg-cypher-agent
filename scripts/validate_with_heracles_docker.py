import argparse
import os
import shutil
import subprocess
import time

import _bootstrap  # noqa: F401
from neo4j import GraphDatabase

from text2cypher_3dsg.config import load_yaml
from text2cypher_3dsg.dataset import read_jsonl
from text2cypher_3dsg.evaluation import (
    build_report,
    evaluate_records,
    print_report_summary,
    write_report,
)
from text2cypher_3dsg.heracles_loader import load_scene_with_heracles
from text2cypher_3dsg.neo4j_validation import Neo4jValidator, schema_policy

DEFAULT_COMPOSE_FILE = "external/heracles/docker/docker-compose.yaml"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def start_neo4j(compose_file: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is not available on PATH.")
    run(["docker", "compose", "-f", compose_file, "up", "-d", "neo4j"])


def wait_for_neo4j(uri: str, username: str, password: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            driver.verify_connectivity()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
        finally:
            driver.close()
    raise TimeoutError(f"Neo4j did not become reachable at {uri}: {last_error}")


def command(args: argparse.Namespace) -> None:
    uri = args.uri or os.getenv("NEO4J_URI") or "neo4j://127.0.0.1:7683"
    username = args.username or os.getenv("NEO4J_USERNAME") or "neo4j"
    password = args.password or os.getenv("NEO4J_PASSWORD") or "neo4j_pw"
    database = args.database or os.getenv("NEO4J_DATABASE") or "neo4j"

    if not args.skip_docker:
        start_neo4j(args.compose_file)
    wait_for_neo4j(uri, username, password, args.wait_timeout)

    if not args.skip_load:
        load_scene_with_heracles(
            scene_path=args.scene,
            uri=uri,
            username=username,
            password=password,
            object_labelspace=args.object_labelspace,
            room_labelspace=args.room_labelspace,
        )

    policy = schema_policy(load_yaml(args.schema_config))
    validator = Neo4jValidator(
        uri=uri,
        username=username,
        password=password,
        database=database,
        timeout=args.query_timeout,
    )
    try:
        records = evaluate_records(
            records=read_jsonl(args.input),
            cypher_field=args.cypher_field,
            allow_writes=args.allow_writes,
            validator=validator,
            policy=policy,
        )
    finally:
        validator.close()

    report = build_report(
        records,
        input_path=args.input,
        scene_path=args.scene,
        cypher_field=args.cypher_field,
    )
    write_report(report, args.output)
    print_report_summary(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start Heracles' Docker Neo4j service, load a 3DSG through Heracles, "
            "and validate Cypher records."
        )
    )
    parser.add_argument("--compose-file", default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--scene", default="data/scenes/3dsg_uhumans2.json")
    parser.add_argument("--input", default="data/processed/text2cypher.jsonl")
    parser.add_argument("--output", default="data/processed/text2cypher.heracles.report.json")
    parser.add_argument("--schema-config", default="configs/schema_3dsg.yaml")
    parser.add_argument("--cypher-field", default="cypher")
    parser.add_argument("--uri", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--object-labelspace", default="ade20k_mit_label_space.yaml")
    parser.add_argument("--room-labelspace", default="b45_label_space.yaml")
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--query-timeout", type=float, default=30.0)
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--allow-writes", action="store_true")
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
