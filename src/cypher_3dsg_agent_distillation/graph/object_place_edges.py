from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TARGET_LAYERS = {"mesh-place": "MESH_PLACES", "place": "PLACES"}
DEFAULT_LAYER_KEYS = {"OBJECTS": (2, 0), "MESH_PLACES": (3, 1), "PLACES": (3, 0)}


@dataclass(frozen=True)
class RepairResult:
    objects: int
    added: int
    already_connected: int
    invalid_positions: int


def layer_key(scene: dict[str, Any], name: str) -> tuple[int, int]:
    """Honor named layer aliases and partitions, including the MIT place alias."""
    key = scene.get("layer_names", {}).get(name)
    if key is None:
        return DEFAULT_LAYER_KEYS[name]
    if isinstance(key, dict):
        return int(key["layer"]), int(key.get("partition", 0))
    return int(key), 0


def position(node: dict[str, Any]) -> np.ndarray | None:
    try:
        point = np.asarray(node.get("attributes", {}).get("position"), dtype=float)
    except (TypeError, ValueError):
        return None
    return point if point.shape == (3,) and np.isfinite(point).all() else None


def add_missing_object_place_edges(scene: dict[str, Any], target_layer: str) -> RepairResult:
    """Append one nearest-place edge for objects unconnected to the chosen layer.

    Existing edges in either orientation are preserved. Distance is Euclidean in
    XYZ, with the lowest node ID breaking ties. Only scene['edges'] is changed.
    """
    if target_layer not in TARGET_LAYERS:
        raise ValueError(f"Unsupported target layer: {target_layer!r}")
    object_key = layer_key(scene, "OBJECTS")
    target_key = layer_key(scene, TARGET_LAYERS[target_layer])
    objects = []
    targets = []
    for node in scene["nodes"]:
        key = int(node["layer"]), int(node.get("partition", 0))
        if key == object_key:
            objects.append(node)
        elif key == target_key:
            targets.append(node)
    targets.sort(key=lambda node: int(node["id"]))
    target_ids = {int(node["id"]) for node in targets}
    usable_targets = [(node, point) for node in targets if (point := position(node)) is not None]
    if not usable_targets:
        raise ValueError(
            f"No {target_layer} nodes with finite XYZ positions in layer {target_key}."
        )
    points = np.asarray([point for _, point in usable_targets])
    object_ids = {int(node["id"]) for node in objects}
    connected = set()
    edges = scene["edges"]
    for edge in edges:
        source, target = int(edge["source"]), int(edge["target"])
        if source in object_ids and target in target_ids:
            connected.add(source)
        elif target in object_ids and source in target_ids:
            connected.add(target)

    new_edges = []
    invalid = 0
    for obj in sorted(objects, key=lambda node: int(node["id"])):
        object_id = int(obj["id"])
        if object_id in connected:
            continue
        point = position(obj)
        if point is None:
            invalid += 1
            continue
        squared_distances = np.sum((points - point) ** 2, axis=1)
        nearest, _ = usable_targets[int(np.argmin(squared_distances))]
        new_edges.append(
            {
                "source": int(nearest["id"]),
                "target": object_id,
                "info": {
                    "type": "EdgeAttributes",
                    "metadata": {},
                    "weight": 1.0,
                    "weighted": False,
                },
            }
        )
    edges.extend(new_edges)
    return RepairResult(len(objects), len(new_edges), len(connected), invalid)


def save_scene(scene: dict[str, Any], output_path: Path) -> None:
    """Write JSON atomically without a Spark DSG round trip changing attributes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=output_path.parent, suffix=".json")
    temporary_path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(scene, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect objects missing edges to their nearest mesh place or place (XYZ)."
    )
    parser.add_argument("scene_json", type=Path, help="Input 3DSG JSON file")
    parser.add_argument("--target-layer", choices=TARGET_LAYERS, required=True)
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing output, including the input",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report repairs without writing a file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_path = args.scene_json
    if not source_path.is_file():
        parser.error(f"input scene does not exist: {source_path}")
    output_path = args.output or source_path.with_name(
        f"{source_path.stem}.object_place_edges{source_path.suffix}"
    )
    if not args.dry_run and not args.force:
        if output_path.resolve() == source_path.resolve():
            parser.error("refusing to overwrite the input scene without --force")
        if output_path.exists():
            parser.error(f"output already exists: {output_path} (use --force to replace it)")
    try:
        scene = json.loads(source_path.read_text(encoding="utf-8"))
        result = add_missing_object_place_edges(scene, args.target_layer)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    action = "Would add" if args.dry_run else "Added"
    print(f"{action} {result.added} object -> {args.target_layer} connections.")
    print(
        f"Objects: {result.objects}; already connected: {result.already_connected}; "
        f"skipped invalid positions: {result.invalid_positions}."
    )
    if not args.dry_run:
        save_scene(scene, output_path)
        print(f"Wrote repaired scene: {output_path}")
    return 0
