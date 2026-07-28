import argparse
import os

import _bootstrap  # noqa: F401

from cypher_3dsg_agent_distillation.heracles_loader import load_scene_with_heracles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a 3DSG JSON into Neo4j through Heracles."
    )
    parser.add_argument("--scene", default="data/scenes/3dsg_uhumans2.json")
    parser.add_argument("--uri", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--object-labelspace", default="ade20k_mit_label_space.yaml")
    parser.add_argument("--room-labelspace", default="b45_label_space.yaml")
    args = parser.parse_args()

    uri = args.uri or os.getenv("NEO4J_URI") or os.getenv("HERACLES_NEO4J_URI")
    username = args.username or os.getenv("NEO4J_USERNAME") or os.getenv(
        "HERACLES_NEO4J_USERNAME"
    )
    password = args.password or os.getenv("NEO4J_PASSWORD") or os.getenv("HERACLES_NEO4J_PASSWORD")
    if not uri or not username or not password:
        raise SystemExit("Neo4j URI, username, and password are required.")

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
