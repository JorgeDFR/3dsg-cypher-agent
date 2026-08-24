from __future__ import annotations

import argparse
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any

import spark_dsg
import yaml

from cypher_3dsg_agent_distillation.config import ensure_parent, load_data_config

LOCAL_ID_MASK = (1 << 56) - 1

LAYER_SPECS = [
    ("Object", spark_dsg.DsgLayers.OBJECTS),
    ("Place", spark_dsg.DsgLayers.PLACES),
    ("MeshPlace", spark_dsg.DsgLayers.MESH_PLACES),
    ("Room", spark_dsg.DsgLayers.ROOMS),
    ("Building", spark_dsg.DsgLayers.BUILDINGS),
]

LABELSPACE_KEYS = {
    "Object": ["_l2p0"],
    "MeshPlace": ["_l3p1", "mesh"],
    "Room": ["room_labelspace", "_l4p0"],
}

LABEL_TO_DSG_LAYER = {
    "Object": spark_dsg.DsgLayers.OBJECTS,
    "Place": spark_dsg.DsgLayers.PLACES,
    "MeshPlace": spark_dsg.DsgLayers.MESH_PLACES,
    "Room": spark_dsg.DsgLayers.ROOMS,
    "Building": spark_dsg.DsgLayers.BUILDINGS,
}

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def load_graph(scene_path: str | Path) -> spark_dsg.DynamicSceneGraph:
    return spark_dsg.DynamicSceneGraph.load(str(scene_path))


def node_symbol(node_id: Any) -> str:
    if hasattr(node_id, "str"):
        return node_id.str(True)
    raw_id = int(node_id)
    prefix = chr(raw_id >> 56)
    return f"{prefix}{raw_id & LOCAL_ID_MASK}"


def node_id_value(node: spark_dsg.SceneGraphNode) -> int:
    return int(node.id.value)


def layer_nodes(
    graph: spark_dsg.DynamicSceneGraph,
    layer: spark_dsg.DsgLayers,
) -> list[spark_dsg.SceneGraphNode]:
    try:
        return list(graph.get_layer(layer).nodes)
    except (IndexError, RuntimeError):
        return []


def nodes_by_label(
    graph: spark_dsg.DynamicSceneGraph,
) -> dict[str, list[spark_dsg.SceneGraphNode]]:
    return {label: layer_nodes(graph, layer) for label, layer in LAYER_SPECS}


def metadata_labelspace(graph: spark_dsg.DynamicSceneGraph, label: str) -> dict[int, str]:
    metadata = graph.metadata.get() or {}
    labelspaces = metadata.get("labelspaces", {})
    for key in LABELSPACE_KEYS.get(label, []):
        if key in labelspaces:
            return {int(label_id): name for label_id, name in labelspaces[key]}
    return {}


def semantic_class(
    graph: spark_dsg.DynamicSceneGraph,
    node: spark_dsg.SceneGraphNode,
    label: str,
) -> str | None:
    semantic_label = getattr(node.attributes, "semantic_label", None)
    if semantic_label is None:
        return None
    try:
        semantic_label = int(semantic_label)
    except (TypeError, ValueError):
        return None
    if semantic_label == getattr(node.attributes, "NO_SEMANTIC_LABEL", -1):
        return None

    labelspace = metadata_labelspace(graph, label)
    if labelspace:
        return labelspace.get(semantic_label)

    try:
        labelspace_obj = graph.get_labelspace(LABEL_TO_DSG_LAYER[label])
        return labelspace_obj.get_node_category(node, "UNKNOWN")
    except Exception:
        return None


def point_from_position(position: Any) -> dict[str, float] | None:
    if position is None or len(position) < 3:
        return None
    return {
        "x": round(float(position[0]), 3),
        "y": round(float(position[1]), 3),
        "z": round(float(position[2]), 3),
    }


def point_string(point: dict[str, float] | None) -> str | None:
    if point is None:
        return None
    return f"POINT({point['x']:.3f} {point['y']:.3f} {point['z']:.3f})"


def distance(a: dict[str, float] | None, b: dict[str, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return round(
        math.sqrt(
            (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2
        ),
        6,
    )


def sorted_symbols(nodes: list[spark_dsg.SceneGraphNode], limit: int) -> list[str]:
    symbols = [node_symbol(node.id) for node in nodes]
    return sorted(symbols, key=lambda value: (value[0], int(value[1:])))[:limit]


def symbol_sort_key(value: str) -> tuple[str, int]:
    return (value[0], int(value[1:]))


def scalar_or_set(values: list[Any]) -> Any:
    if not values:
        return set()
    if len(values) == 1:
        return values[0]
    return set(values)


def template_value(variables: dict[str, Any], key: str) -> Any:
    if key in variables:
        return variables[key]
    case = variables.get("case", {})
    if isinstance(case, dict) and key in case:
        return case[key]
    raise KeyError(f"Missing template variable `{key}`.")


def select_balanced_cases(
    non_empty: list[dict[str, Any]],
    empty: list[dict[str, Any]],
    limit: int,
    non_empty_ratio: float,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    target_non_empty = min(len(non_empty), round(limit * non_empty_ratio))
    target_empty = min(len(empty), limit - target_non_empty)

    selected = non_empty[:target_non_empty] + empty[:target_empty]
    remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(non_empty[target_non_empty : target_non_empty + remaining])
    remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(empty[target_empty : target_empty + remaining])
    return selected


def finite_positions(nodes: list[spark_dsg.SceneGraphNode]) -> list[dict[str, float]]:
    points = []
    for node in nodes:
        point = point_from_position(getattr(node.attributes, "position", None))
        if point is not None and all(math.isfinite(value) for value in point.values()):
            points.append(point)
    return points


def scene_bbox(points: list[dict[str, float]]) -> dict[str, dict[str, float]] | None:
    if not points:
        return None
    return {
        "lower": {
            "x": min(point["x"] for point in points),
            "y": min(point["y"] for point in points),
            "z": min(point["z"] for point in points),
        },
        "upper": {
            "x": max(point["x"] for point in points),
            "y": max(point["y"] for point in points),
            "z": max(point["z"] for point in points),
        },
    }


def centroid(points: list[dict[str, float]]) -> dict[str, float] | None:
    if not points:
        return None
    return {
        axis: round(sum(point[axis] for point in points) / len(points), 3)
        for axis in ["x", "y", "z"]
    }


def sample_points(points: list[dict[str, float]], limit: int) -> list[dict[str, float]]:
    if not points:
        return []
    selected = []
    center = centroid(points)
    if center is not None:
        selected.append(center)
    for point in points:
        if point not in selected:
            selected.append(point)
        if len(selected) >= limit:
            break
    return selected


def select_object_classes_by_solution_size(
    counts: Counter[str],
    limit: int,
    max_solution_size: int,
) -> list[str]:
    if limit <= 0:
        return []

    rare_enough = [
        (class_name, count)
        for class_name, count in counts.items()
        if count <= max_solution_size
    ]
    selected = sorted(rare_enough, key=lambda item: (item[1], item[0]))
    if not selected:
        selected = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    selected = selected[:limit]
    return [class_name for class_name, _ in selected]


def ranked_class_names(counts: Counter[str], limit: int) -> list[str]:
    """Match Cypher's `ORDER BY count DESC, class ASC` ranking."""
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [class_name for class_name, _ in ranked[:limit]]


def top_k_with_ties_validation(
    counts: Counter[str],
    limit: int,
) -> dict[str, Any]:
    return {
        "mode": "top_k_with_ties",
        "limit": limit,
        "scores": dict(sorted(counts.items())),
    }


def bbox_around_point(
    point: dict[str, float],
    half_extent: float,
) -> dict[str, dict[str, float]]:
    return {
        "lower": {
            axis: round(point[axis] - half_extent, 3)
            for axis in ["x", "y", "z"]
        },
        "upper": {
            axis: round(point[axis] + half_extent, 3)
            for axis in ["x", "y", "z"]
        },
    }


def point_in_bbox(
    point: dict[str, float] | None,
    bbox: dict[str, dict[str, float]],
) -> bool:
    if point is None:
        return False
    lower = bbox["lower"]
    upper = bbox["upper"]
    return all(lower[axis] <= point[axis] <= upper[axis] for axis in ["x", "y", "z"])


def bbox_key(bbox: dict[str, dict[str, float]]) -> tuple[float, ...]:
    return tuple(
        bbox[corner][axis]
        for corner in ["lower", "upper"]
        for axis in ["x", "y", "z"]
    )


def scene_values(
    scene_path: str | Path,
    max_classes: int,
    max_object_symbols: int,
    max_place_symbols: int,
    max_mesh_place_symbols: int,
    max_points: int,
    max_bboxes: int = 50,
    max_place_object_cases: int = 50,
    max_radius_cases: int = 50,
    max_nearest_mesh_cases: int = 60,
    max_nearest_object_cases: int = 50,
    max_mesh_place_hop_cases: int = 50,
    max_place_hop_cases: int = 50,
    non_empty_ratio: float = 0.9,
    max_object_class_solution_size: int = 10,
    max_bbox_solution_size: int = 20,
) -> dict[str, Any]:
    graph = load_graph(scene_path)
    by_label = nodes_by_label(graph)
    index = SceneIndex(scene_path)

    object_class_counts = Counter()
    for node in by_label["Object"]:
        class_name = semantic_class(graph, node, "Object")
        if class_name:
            object_class_counts[class_name] += 1

    mesh_class_counts = Counter()
    for node in by_label["MeshPlace"]:
        class_name = semantic_class(graph, node, "MeshPlace")
        if class_name:
            mesh_class_counts[class_name] += 1

    room_class_counts = Counter()
    for node in by_label["Room"]:
        class_name = semantic_class(graph, node, "Room")
        if class_name:
            room_class_counts[class_name] += 1

    object_points = finite_positions(by_label["Object"])
    all_points = finite_positions(by_label["Object"] + by_label["Place"] + by_label["MeshPlace"])
    object_classes = select_object_classes_by_solution_size(
        object_class_counts,
        max_classes,
        max_object_class_solution_size,
    )
    mesh_place_symbols = sorted_symbols(by_label["MeshPlace"], max_mesh_place_symbols)

    return {
        "object_classes": object_classes,
        "mesh_place_classes": [
            class_name for class_name, _ in mesh_class_counts.most_common(max_classes)
        ],
        "room_classes": [
            class_name for class_name, _ in room_class_counts.most_common(max_classes)
        ],
        "object_symbols": sorted_symbols(by_label["Object"], max_object_symbols),
        "room_symbols": sorted_symbols(by_label["Room"], max_place_symbols),
        "place_symbols": sorted_symbols(by_label["Place"], max_place_symbols),
        "mesh_place_symbols": mesh_place_symbols,
        "building_symbols": sorted_symbols(by_label["Building"], max_place_symbols),
        "top_class_limits": list(range(1, len(object_class_counts) + 1)),
        "radii": [2, 5, 10],
        "hop_counts": [2, 3, 4, 5],
        "points": sample_points(object_points or all_points, max_points),
        "bboxes": build_bounding_box_cases(
            index=index,
            limit=max_bboxes,
            max_solution_size=max_bbox_solution_size,
        ),
        "place_object_cases": build_place_object_cases(
            index=index,
            limit=max_place_object_cases,
            non_empty_ratio=non_empty_ratio,
        ),
        "object_radius_cases": build_object_radius_cases(
            index=index,
            object_classes=object_classes,
            limit=max_radius_cases,
            non_empty_ratio=non_empty_ratio,
        ),
        "nearest_object_mesh_cases": build_nearest_object_mesh_cases(
            object_classes=object_classes,
            mesh_place_symbols=mesh_place_symbols,
            limit=max_nearest_mesh_cases,
        ),
        "nearest_object_cases": build_nearest_object_cases(
            index=index,
            object_classes=object_classes,
            limit=max_nearest_object_cases,
        ),
        "mesh_place_hop_cases": build_hop_cases(
            symbols=mesh_place_symbols,
            hop_counts=[2, 3, 4, 5],
            symbol_key="mesh_place",
            limit=max_mesh_place_hop_cases,
        ),
        "place_hop_cases": build_hop_cases(
            symbols=sorted_symbols(by_label["Place"], max_place_symbols),
            hop_counts=[2, 3, 4, 5],
            symbol_key="place",
            limit=max_place_hop_cases,
        ),
    }


def scene_summary(scene_path: str | Path) -> dict[str, Any]:
    graph = load_graph(scene_path)
    by_label = nodes_by_label(graph)
    classes: dict[str, Counter] = {"Object": Counter(), "MeshPlace": Counter(), "Room": Counter()}
    for label, nodes in by_label.items():
        if label not in classes:
            continue
        for node in nodes:
            class_name = semantic_class(graph, node, label)
            if class_name:
                classes[label][class_name] += 1
    return {
        "nodes": {label: len(nodes) for label, nodes in by_label.items() if nodes},
        "classes": {label: dict(counter.most_common()) for label, counter in classes.items()},
        "edges": int(graph.num_edges()),
    }


class SceneIndex:
    def __init__(self, scene_path: str | Path):
        self.graph = load_graph(scene_path)
        self.nodes_by_label = nodes_by_label(self.graph)
        self.nodes_by_symbol: dict[str, spark_dsg.SceneGraphNode] = {}
        self.node_labels_by_id: dict[int, str] = {}
        self.classes_by_symbol: dict[str, str | None] = {}
        self.points_by_symbol: dict[str, dict[str, float] | None] = {}
        for label, nodes in self.nodes_by_label.items():
            for node in nodes:
                symbol = node_symbol(node.id)
                self.nodes_by_symbol[symbol] = node
                self.node_labels_by_id[node_id_value(node)] = label
                self.classes_by_symbol[symbol] = semantic_class(self.graph, node, label)
                self.points_by_symbol[symbol] = point_from_position(
                    getattr(node.attributes, "position", None)
                )

    def nodes_with_label(self, label: str) -> list[spark_dsg.SceneGraphNode]:
        return self.nodes_by_label.get(label, [])

    def object_class_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for node in self.nodes_with_label("Object"):
            class_name = self.classes_by_symbol[node_symbol(node.id)]
            if class_name is not None:
                counts[class_name] += 1
        return counts

    def objects_with_class(self, class_name: str | None = None) -> list[spark_dsg.SceneGraphNode]:
        nodes = self.nodes_with_label("Object")
        if class_name is None:
            return sorted(nodes, key=lambda node: symbol_sort_key(node_symbol(node.id)))
        return sorted(
            [node for node in nodes if self.classes_by_symbol[node_symbol(node.id)] == class_name],
            key=lambda node: symbol_sort_key(node_symbol(node.id)),
        )

    def node_result(
        self,
        node: spark_dsg.SceneGraphNode,
        key: str = "nodeSymbol",
    ) -> dict[str, Any]:
        symbol = node_symbol(node.id)
        result = {
            key: symbol,
            "center": point_string(self.points_by_symbol[symbol]),
        }
        class_name = self.classes_by_symbol.get(symbol)
        if class_name is not None:
            result["class"] = class_name
        return result

    def direct_place_objects(self, place_symbol: str) -> list[spark_dsg.SceneGraphNode]:
        place = self.nodes_by_symbol.get(place_symbol)
        if place is None:
            return []
        object_ids = [
            child
            for child in place.children()
            if self.node_labels_by_id.get(int(child)) == "Object"
        ]
        nodes = [self.graph.get_node(node_id) for node_id in object_ids]
        return sorted(nodes, key=lambda node: symbol_sort_key(node_symbol(node.id)))

    def contained_objects(
        self,
        container: spark_dsg.SceneGraphNode,
    ) -> list[spark_dsg.SceneGraphNode]:
        found: list[spark_dsg.SceneGraphNode] = []
        queue = deque(container.children())
        visited: set[int] = set()
        while queue:
            child_id = int(queue.popleft())
            if child_id in visited:
                continue
            visited.add(child_id)
            label = self.node_labels_by_id.get(child_id)
            child = self.graph.get_node(child_id)
            if label == "Object":
                found.append(child)
                continue
            queue.extend(child.children())
        return sorted(found, key=lambda node: symbol_sort_key(node_symbol(node.id)))

    def rooms_with_class(self, class_name: str | None = None) -> list[spark_dsg.SceneGraphNode]:
        nodes = self.nodes_with_label("Room")
        if class_name is None:
            return sorted(nodes, key=lambda node: symbol_sort_key(node_symbol(node.id)))
        return sorted(
            [node for node in nodes if self.classes_by_symbol[node_symbol(node.id)] == class_name],
            key=lambda node: symbol_sort_key(node_symbol(node.id)),
        )

    def containing_places(self, object_symbol: str) -> list[spark_dsg.SceneGraphNode]:
        obj = self.nodes_by_symbol.get(object_symbol)
        if obj is None:
            return []
        place_ids = [
            parent
            for parent in obj.parents()
            if self.node_labels_by_id.get(int(parent)) == "Place"
        ]
        nodes = [self.graph.get_node(node_id) for node_id in place_ids]
        return sorted(nodes, key=lambda node: symbol_sort_key(node_symbol(node.id)))

    def hop_symbols(self, start_symbol: str, label: str, max_hops: int) -> list[str]:
        start = self.nodes_by_symbol.get(start_symbol)
        if start is None:
            return []
        visited = {node_id_value(start)}
        found: set[str] = set()
        queue: deque[tuple[spark_dsg.SceneGraphNode, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor_id in node.siblings():
                neighbor_id = int(neighbor_id)
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                if self.node_labels_by_id.get(neighbor_id) != label:
                    continue
                neighbor = self.graph.get_node(neighbor_id)
                found.add(node_symbol(neighbor.id))
                queue.append((neighbor, depth + 1))
        return sorted(found, key=symbol_sort_key)

    def solution_for(
        self,
        query_type: str,
        variables: dict[str, Any],
    ) -> Any:
        if query_type == "object_class_inventory":
            class_names = sorted(
                {
                    self.classes_by_symbol[node_symbol(node.id)]
                    for node in self.nodes_with_label("Object")
                    if self.classes_by_symbol[node_symbol(node.id)] is not None
                }
            )
            return scalar_or_set(class_names)
        if query_type == "top_object_classes_by_count":
            limit = int(variables.get("limit", variables.get("case", {}).get("limit", 2)))
            return ranked_class_names(self.object_class_counts(), limit)
        if query_type == "object_class_centers":
            centers = [
                point_string(self.points_by_symbol[node_symbol(node.id)])
                for node in self.objects_with_class(str(template_value(variables, "class")))
            ]
            return scalar_or_set([center for center in centers if center is not None])
        if query_type == "object_class_count":
            return len(self.objects_with_class(str(template_value(variables, "class"))))
        if query_type == "objects_in_room_symbol":
            room = self.nodes_by_symbol.get(str(template_value(variables, "room")))
            if room is None:
                return set()
            symbols = [node_symbol(node.id) for node in self.contained_objects(room)]
            return scalar_or_set(symbols)
        if query_type == "objects_in_room_class":
            symbols = []
            for room in self.rooms_with_class(str(template_value(variables, "room_class"))):
                symbols.extend(node_symbol(node.id) for node in self.contained_objects(room))
            return scalar_or_set(sorted(set(symbols), key=symbol_sort_key))
        if query_type == "objects_in_place_symbol":
            symbols = [
                node_symbol(node.id)
                for node in self.direct_place_objects(str(template_value(variables, "place")))
            ]
            return scalar_or_set(symbols)
        if query_type == "object_place_lookup":
            places = [
                node_symbol(node.id)
                for node in self.containing_places(str(template_value(variables, "object")))
            ]
            return scalar_or_set(places)
        if query_type == "nearest_object_to_mesh_place":
            mesh_place = self.nodes_by_symbol.get(str(template_value(variables, "mesh_place")))
            if mesh_place is None:
                return None
            mesh_point = self.points_by_symbol[node_symbol(mesh_place.id)]
            candidates = []
            for node in self.objects_with_class(str(template_value(variables, "class"))):
                symbol = node_symbol(node.id)
                node_distance = distance(mesh_point, self.points_by_symbol[symbol])
                if node_distance is not None:
                    candidates.append({"object": symbol, "distance": node_distance})
            sorted_candidates = sorted(
                candidates,
                key=lambda item: (item["distance"], symbol_sort_key(item["object"])),
            )
            if not sorted_candidates:
                return set()
            return sorted_candidates[0]["object"]
        if query_type == "object_radius_from_point":
            query_point = template_value(variables, "point")
            radius = float(template_value(variables, "radius"))
            symbols = []
            for node in self.objects_with_class(str(template_value(variables, "class"))):
                symbol = node_symbol(node.id)
                node_distance = distance(query_point, self.points_by_symbol[symbol])
                if node_distance is not None and node_distance < radius:
                    symbols.append(symbol)
            return scalar_or_set(sorted(set(symbols), key=symbol_sort_key))
        if query_type == "nearest_object_to_object":
            source = self.nodes_by_symbol.get(str(template_value(variables, "object")))
            if source is None:
                return set()
            source_symbol = node_symbol(source.id)
            source_point = self.points_by_symbol[source_symbol]
            candidates = []
            for node in self.objects_with_class(str(template_value(variables, "class"))):
                symbol = node_symbol(node.id)
                if symbol == source_symbol:
                    continue
                node_distance = distance(source_point, self.points_by_symbol[symbol])
                if node_distance is not None:
                    candidates.append({"object": symbol, "distance": node_distance})
            sorted_candidates = sorted(
                candidates,
                key=lambda item: (item["distance"], symbol_sort_key(item["object"])),
            )
            if not sorted_candidates:
                return set()
            return sorted_candidates[0]["object"]
        if query_type == "objects_in_bounding_box":
            bbox = variables["bbox"]
            lower = bbox["lower"]
            upper = bbox["upper"]
            symbols = []
            for node in self.objects_with_class():
                symbol = node_symbol(node.id)
                point = self.points_by_symbol[symbol]
                if point is None:
                    continue
                if all(lower[axis] <= point[axis] <= upper[axis] for axis in ["x", "y", "z"]):
                    symbols.append(symbol)
            return scalar_or_set(symbols)
        if query_type == "mesh_place_hops":
            return scalar_or_set(
                self.hop_symbols(
                    str(template_value(variables, "mesh_place")),
                    "MeshPlace",
                    int(template_value(variables, "hops")),
                )
            )
        if query_type == "room_adjacency_count":
            room = self.nodes_by_symbol.get(str(variables["room"]))
            if room is None:
                return 0
            neighbor_ids = {
                int(neighbor_id)
                for neighbor_id in room.siblings()
                if self.node_labels_by_id.get(int(neighbor_id)) == "Room"
            }
            return len(neighbor_ids)
        if query_type == "place_hops":
            return scalar_or_set(
                self.hop_symbols(
                    str(template_value(variables, "place")),
                    "Place",
                    int(template_value(variables, "hops")),
                )
            )
        if query_type == "write_refusal":
            return "WRITE_QUERY_NOT_ALLOWED"
        return None

    def answer_validation_for(
        self,
        query_type: str,
        variables: dict[str, Any],
    ) -> dict[str, Any] | None:
        if query_type != "top_object_classes_by_count":
            return None
        limit = int(variables.get("limit", variables.get("case", {}).get("limit", 2)))
        return top_k_with_ties_validation(self.object_class_counts(), limit)


def build_place_object_cases(
    index: SceneIndex,
    limit: int,
    non_empty_ratio: float,
) -> list[dict[str, Any]]:
    non_empty = []
    empty = []
    for node in index.nodes_with_label("Place"):
        symbol = node_symbol(node.id)
        case = {"place": symbol}
        if index.direct_place_objects(symbol):
            non_empty.append(case)
        else:
            empty.append(case)
    non_empty = sorted(non_empty, key=lambda case: symbol_sort_key(case["place"]))
    empty = sorted(empty, key=lambda case: symbol_sort_key(case["place"]))
    return select_balanced_cases(non_empty, empty, limit, non_empty_ratio)


def objects_in_bbox(
    index: SceneIndex,
    bbox: dict[str, dict[str, float]],
) -> list[str]:
    symbols = []
    for node in index.objects_with_class():
        symbol = node_symbol(node.id)
        if point_in_bbox(index.points_by_symbol[symbol], bbox):
            symbols.append(symbol)
    return symbols


def build_bounding_box_cases(
    index: SceneIndex,
    limit: int,
    max_solution_size: int,
    half_extents: tuple[float, ...] = (1.5, 2.5, 4.0),
) -> list[dict[str, dict[str, float]]]:
    if limit <= 0 or max_solution_size <= 0:
        return []

    candidates: list[tuple[int, str, dict[str, dict[str, float]]]] = []
    seen_solutions: set[tuple[str, ...]] = set()
    seen_bboxes: set[tuple[float, ...]] = set()
    for node in index.objects_with_class():
        symbol = node_symbol(node.id)
        point = index.points_by_symbol[symbol]
        if point is None:
            continue
        for half_extent in half_extents:
            bbox = bbox_around_point(point, half_extent)
            key = bbox_key(bbox)
            if key in seen_bboxes:
                continue
            seen_bboxes.add(key)
            solution = tuple(objects_in_bbox(index, bbox))
            if not solution or len(solution) > max_solution_size:
                continue
            if solution in seen_solutions:
                continue
            seen_solutions.add(solution)
            candidates.append((len(solution), symbol, bbox))

    candidates.sort(key=lambda item: (-item[0], symbol_sort_key(item[1]), bbox_key(item[2])))
    return [bbox for _, _, bbox in candidates[:limit]]


def build_object_radius_cases(
    index: SceneIndex,
    object_classes: list[str],
    limit: int,
    non_empty_ratio: float,
) -> list[dict[str, Any]]:
    object_nodes = index.objects_with_class()
    point_objects = [
        node
        for node in object_nodes
        if index.points_by_symbol[node_symbol(node.id)] is not None
    ]
    non_empty = []
    empty = []

    for class_name in object_classes:
        for node in index.objects_with_class(class_name):
            symbol = node_symbol(node.id)
            point = index.points_by_symbol[symbol]
            if point is not None:
                non_empty.append({"class": class_name, "radius": 2, "point": point})

    used_empty_points: set[tuple[float, float, float]] = set()
    for class_name in object_classes:
        fallback_case = None
        for node in point_objects:
            point = index.points_by_symbol[node_symbol(node.id)]
            if point is None:
                continue
            has_match = False
            for candidate in index.objects_with_class(class_name):
                candidate_distance = distance(
                    point,
                    index.points_by_symbol[node_symbol(candidate.id)],
                )
                if candidate_distance is not None and candidate_distance < 2:
                    has_match = True
                    break
            if not has_match:
                case = {"class": class_name, "radius": 2, "point": point}
                point_key = (point["x"], point["y"], point["z"])
                if point_key not in used_empty_points:
                    empty.append(case)
                    used_empty_points.add(point_key)
                    break
                if fallback_case is None:
                    fallback_case = case
        else:
            if fallback_case is not None:
                empty.append(fallback_case)

    deduped_empty = []
    seen_empty = set()
    for case in empty:
        key = (
            case["class"],
            case["radius"],
            case["point"]["x"],
            case["point"]["y"],
            case["point"]["z"],
        )
        if key in seen_empty:
            continue
        seen_empty.add(key)
        deduped_empty.append(case)

    return select_balanced_cases(non_empty, deduped_empty, limit, non_empty_ratio)


def build_nearest_object_mesh_cases(
    object_classes: list[str],
    mesh_place_symbols: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    cases = []
    if not object_classes or not mesh_place_symbols:
        return cases
    for index in range(limit):
        cases.append(
            {
                "class": object_classes[index % len(object_classes)],
                "mesh_place": mesh_place_symbols[index % len(mesh_place_symbols)],
            }
        )
    return cases


def build_nearest_object_cases(
    index: SceneIndex,
    object_classes: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    cases = []
    if not object_classes:
        return cases
    object_symbols = [node_symbol(node.id) for node in index.objects_with_class()]
    for offset in range(len(object_classes)):
        for object_index, object_symbol in enumerate(object_symbols):
            class_name = object_classes[(object_index + offset) % len(object_classes)]
            candidates = [
                node_symbol(node.id)
                for node in index.objects_with_class(class_name)
                if node_symbol(node.id) != object_symbol
            ]
            if candidates:
                cases.append({"object": object_symbol, "class": class_name})
                if len(cases) >= limit:
                    return cases
    return cases


def build_hop_cases(
    symbols: list[str],
    hop_counts: list[int],
    symbol_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    cases = []
    if not symbols or not hop_counts:
        return cases
    for index in range(limit):
        cases.append(
            {
                symbol_key: symbols[index % len(symbols)],
                "hops": hop_counts[index % len(hop_counts)],
            }
        )
    return cases


def build_template_config(
    scene_path: str | Path,
    scene_id: str,
    limits: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "scene_summary": scene_summary(scene_path),
        "values": scene_values(
            scene_path=scene_path,
            max_classes=limits.max_classes,
            max_object_symbols=limits.max_object_symbols,
            max_place_symbols=limits.max_place_symbols,
            max_mesh_place_symbols=limits.max_mesh_place_symbols,
            max_points=limits.max_points,
            max_bboxes=limits.max_bboxes,
            max_place_object_cases=limits.max_place_object_cases,
            max_radius_cases=limits.max_radius_cases,
            max_nearest_mesh_cases=limits.max_nearest_mesh_cases,
            max_nearest_object_cases=limits.max_nearest_object_cases,
            max_mesh_place_hop_cases=limits.max_mesh_place_hop_cases,
            max_place_hop_cases=limits.max_place_hop_cases,
            non_empty_ratio=limits.non_empty_ratio,
            max_object_class_solution_size=limits.max_object_class_solution_size,
            max_bbox_solution_size=limits.max_bbox_solution_size,
        ),
    }


def command(args: argparse.Namespace) -> None:
    data_config = load_data_config(args.data_config)
    config = build_template_config(
        scene_path=data_config["scene_path"],
        scene_id=data_config["scene_id"],
        limits=args,
    )
    output = ensure_parent(data_config["scene_template_values_path"])
    with output.open("w", encoding="utf-8") as handle:
        yaml.dump(
            config,
            handle,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dataset template values from a 3DSG JSON."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--max-classes", type=int, default=30)
    parser.add_argument("--max-object-symbols", type=int, default=50)
    parser.add_argument("--max-place-symbols", type=int, default=50)
    parser.add_argument("--max-mesh-place-symbols", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=50)
    parser.add_argument("--max-bboxes", type=int, default=50)
    parser.add_argument("--max-place-object-cases", type=int, default=50)
    parser.add_argument("--max-radius-cases", type=int, default=50)
    parser.add_argument("--max-nearest-mesh-cases", type=int, default=60)
    parser.add_argument("--max-nearest-object-cases", type=int, default=50)
    parser.add_argument("--max-mesh-place-hop-cases", type=int, default=50)
    parser.add_argument("--max-place-hop-cases", type=int, default=50)
    parser.add_argument("--non-empty-ratio", type=float, default=0.9)
    parser.add_argument("--max-object-class-solution-size", type=int, default=10)
    parser.add_argument("--max-bbox-solution-size", type=int, default=20)
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
