import argparse
import shutil
import subprocess
import time

import _bootstrap  # noqa: F401
from neo4j import GraphDatabase

from cypher_3dsg_agent_distillation.config import load_yaml
from cypher_3dsg_agent_distillation.evaluation import (
    build_report,
    evaluate_records,
    print_report_summary,
    write_report,
)
from cypher_3dsg_agent_distillation.heracles_loader import load_scene_with_heracles
from cypher_3dsg_agent_distillation.io import read_jsonl
from cypher_3dsg_agent_distillation.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_URI,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    Neo4jValidator,
    neo4j_uri_candidates,
    schema_policy,
    suppress_neo4j_notification_logs,
)

DEFAULT_COMPOSE_FILE = "docker/docker-compose.yaml"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def start_neo4j(compose_file: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is not available on PATH.")
    run(["docker", "compose", "-f", compose_file, "up", "-d", "neo4j"])


def wait_for_neo4j(
    uris: tuple[str, ...],
    username: str,
    password: str,
    timeout_s: float,
) -> str:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        for uri in uris:
            driver = GraphDatabase.driver(uri, auth=(username, password), connection_timeout=3.0)
            try:
                driver.verify_connectivity()
                return uri
            except Exception as exc:
                last_error = f"{uri}: {type(exc).__name__}: {exc}"
            finally:
                driver.close()
        time.sleep(1)
    raise TimeoutError(f"Neo4j did not become reachable at {uris}: {last_error}")


def command(args: argparse.Namespace) -> None:
    if not args.show_db_notifications:
        suppress_neo4j_notification_logs()

    uri_candidates = neo4j_uri_candidates(args.uri)
    username = args.username or DEFAULT_HERACLES_NEO4J_USERNAME
    password = args.password or DEFAULT_HERACLES_NEO4J_PASSWORD
    database = args.database or DEFAULT_HERACLES_NEO4J_DATABASE

    if not args.skip_docker:
        start_neo4j(args.compose_file)
    uri = wait_for_neo4j(uri_candidates, username, password, args.wait_timeout)

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
            "Start the project Docker Neo4j service, load a 3DSG through Heracles, "
            "and validate Cypher records."
        )
    )
    parser.add_argument("--compose-file", default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--scene", default="data/scenes/uhumans2_office.json")
    parser.add_argument("--input", default="data/legacy/processed/cypher_records.jsonl")
    parser.add_argument(
        "--output",
        default="data/legacy/reports/cypher_records.heracles.report.json",
    )
    parser.add_argument("--schema-config", default="configs/schema_3dsg.yaml")
    parser.add_argument("--cypher-field", default="cypher")
    parser.add_argument(
        "--uri",
        default=None,
        help=(
            f"Neo4j URI or comma-separated URI fallback list. "
            f"Default: {DEFAULT_HERACLES_NEO4J_URI}, then localhost fallback."
        ),
    )
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
    parser.add_argument("--object-labelspace", default="ade20k_mit_label_space.yaml")
    parser.add_argument("--room-labelspace", default="b45_label_space.yaml")
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--query-timeout", type=float, default=30.0)
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--allow-writes", action="store_true")
    parser.add_argument(
        "--show-db-notifications",
        action="store_true",
        help="Show Neo4j driver notification logs for query warnings.",
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
