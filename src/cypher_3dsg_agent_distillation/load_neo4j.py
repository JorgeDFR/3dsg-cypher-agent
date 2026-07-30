from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neo4j.exceptions import DatabaseError

from cypher_3dsg_agent_distillation.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_URI,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    connect_neo4j_driver,
    neo4j_uri_candidates,
)
from cypher_3dsg_agent_distillation.scene_graph import (
    load_graph,
    node_id_value,
    node_symbol,
    nodes_by_label,
    semantic_class,
)

INDEXES = {
    "object_node_symbol": (
        "CREATE INDEX object_node_symbol IF NOT EXISTS FOR (n:Object) ON (n.nodeSymbol)"
    ),
    "place_node_symbol": (
        "CREATE INDEX place_node_symbol IF NOT EXISTS FOR (n:Place) ON (n.nodeSymbol)"
    ),
    "mesh_place_node_symbol": (
        "CREATE INDEX mesh_place_node_symbol IF NOT EXISTS FOR (n:MeshPlace) ON (n.nodeSymbol)"
    ),
    "room_node_symbol": (
        "CREATE INDEX room_node_symbol IF NOT EXISTS FOR (n:Room) ON (n.nodeSymbol)"
    ),
    "building_node_symbol": (
        "CREATE INDEX building_node_symbol IF NOT EXISTS FOR (n:Building) ON (n.nodeSymbol)"
    ),
}


def point_map(position: Any) -> dict[str, float] | None:
    if position is None or len(position) < 3:
        return None
    return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}


def node_record(graph, label: str, node) -> dict[str, Any]:
    attrs = node.attributes
    bbox = getattr(attrs, "bounding_box", None)
    record = {
        "id": node_id_value(node),
        "label": label,
        "nodeSymbol": node_symbol(node.id),
        "center": point_map(getattr(attrs, "position", None)),
        "class": semantic_class(graph, node, label),
        "name": getattr(attrs, "name", None) or None,
    }
    record["bbox_center"] = point_map(getattr(bbox, "world_P_center", None))
    record["bbox_dim"] = point_map(getattr(bbox, "dimensions", None))
    return record


def relationship_for(source_label: str, target_label: str) -> tuple[str, str, str] | None:
    pair = {source_label, target_label}
    if pair == {"Object", "Place"}:
        return "Place", "CONTAINS", "Object"
    if pair == {"Object", "MeshPlace"}:
        return "MeshPlace", "CONTAINS", "Object"
    if pair == {"Place"}:
        return source_label, "PLACE_CONNECTED", target_label
    if pair == {"MeshPlace"}:
        return source_label, "MESH_PLACE_CONNECTED", target_label
    if pair == {"Room"}:
        return source_label, "ROOM_CONNECTED", target_label
    if pair == {"Building"}:
        return source_label, "BUILDING_CONNECTED", target_label
    if pair == {"Room", "Place"}:
        return "Room", "CONTAINS", "Place"
    if pair == {"Room", "MeshPlace"}:
        return "Room", "CONTAINS", "MeshPlace"
    if pair == {"Room", "Building"}:
        return "Building", "CONTAINS", "Room"
    return None


def extract_graph_records(
    scene_path: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict]]:
    graph = load_graph(scene_path)
    graph_nodes = nodes_by_label(graph)
    node_records_by_label: dict[str, list[dict[str, Any]]] = {
        "Object": [],
        "Place": [],
        "MeshPlace": [],
        "Room": [],
        "Building": [],
    }
    id_to_record: dict[int, dict[str, Any]] = {}
    for label, nodes in graph_nodes.items():
        for node in nodes:
            record = node_record(graph, label, node)
            node_records_by_label[label].append(record)
            id_to_record[record["id"]] = record

    relationships = []
    for edge in graph.edges:
        source = id_to_record.get(int(edge.source))
        target = id_to_record.get(int(edge.target))
        if source is None or target is None:
            continue
        rel_spec = relationship_for(source["label"], target["label"])
        if rel_spec is None:
            continue
        from_label, rel_type, to_label = rel_spec
        if source["label"] == from_label and target["label"] == to_label:
            from_node, to_node = source, target
        elif target["label"] == from_label and source["label"] == to_label:
            from_node, to_node = target, source
        else:
            from_node, to_node = source, target
        relationships.append(
            {
                "from": from_node["nodeSymbol"],
                "from_label": from_node["label"],
                "to": to_node["nodeSymbol"],
                "to_label": to_node["label"],
                "type": rel_type,
            }
        )
    return node_records_by_label, relationships


def run_query(driver, database: str, query: str, **parameters):
    return driver.execute_query(query, database_=database, **parameters)


def initialize_database(driver, database: str, clear: bool) -> None:
    if clear:
        run_query(driver, database, "MATCH (n) DETACH DELETE n")
    for query in INDEXES.values():
        try:
            run_query(driver, database, query)
        except DatabaseError:
            # Older Neo4j versions may not support IF NOT EXISTS for every index form.
            pass


def insert_nodes(driver, database: str, label: str, nodes: list[dict[str, Any]]) -> None:
    if not nodes:
        return
    query = f"""
    UNWIND $nodes AS node
    WITH node,
         CASE WHEN node.center IS NULL THEN null ELSE point(node.center) END AS center,
         CASE
           WHEN node.bbox_center IS NULL THEN null
           ELSE point(node.bbox_center)
         END AS bbox_center,
         CASE WHEN node.bbox_dim IS NULL THEN null ELSE point(node.bbox_dim) END AS bbox_dim
    MERGE (n:{label} {{nodeSymbol: node.nodeSymbol}})
    SET n.center = center
    SET n.class = node.class
    SET n.name = node.name
    SET n.bbox_center = bbox_center
    SET n.bbox_dim = bbox_dim
    """
    run_query(driver, database, query, nodes=nodes)


def insert_relationships(driver, database: str, relationships: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for relationship in relationships:
        key = (relationship["from_label"], relationship["type"], relationship["to_label"])
        grouped.setdefault(key, []).append(relationship)

    for (from_label, rel_type, to_label), rels in grouped.items():
        query = f"""
        UNWIND $relationships AS relationship
        MATCH (a:{from_label} {{nodeSymbol: relationship.from}})
        MATCH (b:{to_label} {{nodeSymbol: relationship.to}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        run_query(driver, database, query, relationships=rels)


def load_scene(
    scene_path: str | Path,
    uri: str | Sequence[str],
    username: str,
    password: str,
    database: str,
    clear: bool,
) -> dict[str, Any]:
    nodes_by_label, relationships = extract_graph_records(scene_path)
    _resolved_uri, driver = connect_neo4j_driver(
        uri,
        username=username,
        password=password,
        database=database,
    )
    try:
        initialize_database(driver, database, clear=clear)
        for label, nodes in nodes_by_label.items():
            insert_nodes(driver, database, label, nodes)
        insert_relationships(driver, database, relationships)
    finally:
        driver.close()

    return {
        "nodes": {label: len(nodes) for label, nodes in nodes_by_label.items()},
        "relationships": len(relationships),
    }


def command(args: argparse.Namespace) -> None:
    uri = args.uri or os.environ.get("NEO4J_URI")
    summary = load_scene(
        scene_path=args.scene,
        uri=neo4j_uri_candidates(uri),
        username=args.username or os.environ.get("NEO4J_USERNAME", DEFAULT_HERACLES_NEO4J_USERNAME),
        password=args.password or os.environ.get("NEO4J_PASSWORD", DEFAULT_HERACLES_NEO4J_PASSWORD),
        database=args.database or os.environ.get("NEO4J_DATABASE", DEFAULT_HERACLES_NEO4J_DATABASE),
        clear=not args.no_clear,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a Spark DSG JSON into Neo4j.")
    parser.add_argument("--scene", default="data/scenes/3dsg_uhumans2.json")
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
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
