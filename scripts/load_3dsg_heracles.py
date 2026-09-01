import argparse
import os

import _bootstrap  # noqa: F401

from cypher_3dsg_agent_distillation.heracles_loader import load_scene_with_heracles
from cypher_3dsg_agent_distillation.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_URI,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    connect_neo4j_driver,
    neo4j_uri_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a 3DSG JSON into Neo4j through Heracles."
    )
    parser.add_argument("--scene", default="data/scenes/uhumans2_office.json")
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
    parser.add_argument("--object-labelspace", default="ade20k_mit_label_space.yaml")
    parser.add_argument("--room-labelspace", default="b45_label_space.yaml")
    args = parser.parse_args()

    uri = args.uri or os.getenv("NEO4J_URI") or os.getenv("HERACLES_NEO4J_URI")
    username = args.username or os.getenv("NEO4J_USERNAME") or os.getenv(
        "HERACLES_NEO4J_USERNAME",
        DEFAULT_HERACLES_NEO4J_USERNAME,
    )
    password = args.password or os.getenv(
        "NEO4J_PASSWORD",
    ) or os.getenv("HERACLES_NEO4J_PASSWORD", DEFAULT_HERACLES_NEO4J_PASSWORD)
    uri, driver = connect_neo4j_driver(
        neo4j_uri_candidates(uri),
        username=username,
        password=password,
        database=DEFAULT_HERACLES_NEO4J_DATABASE,
    )
    driver.close()

    load_scene_with_heracles(
        scene_path=args.scene,
        uri=uri,
        username=username,
        password=password,
        object_labelspace=args.object_labelspace,
        room_labelspace=args.room_labelspace,
    )


if __name__ == "__main__":
    main()
