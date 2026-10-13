from __future__ import annotations

import argparse
import copy
import os
import tempfile
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import spark_dsg

ROOM_LABELSPACE_KEY = "_l4p0"
OBJECT_LABELSPACE_KEYS = ("_l2p0", "_l3p1", "mesh")

# Canonical room and region types. Spaces are part of the semantic class name.
ROOM_LABELS = (
    "alcove",
    "anteroom",
    "archive",
    "atrium",
    "auditorium",
    "balcony",
    "ballroom",
    "bar",
    "basement",
    "bathroom",
    "beach",
    "bedroom",
    "billiard room",
    "boardroom",
    "boiler room",
    "break room",
    "bridge",
    "cafeteria",
    "carport",
    "cellar",
    "chapel",
    "classroom",
    "cleanroom",
    "cloakroom",
    "closet",
    "computer room",
    "conference room",
    "conservatory",
    "control room",
    "corridor",
    "courtroom",
    "courtyard",
    "darkroom",
    "deck",
    "den",
    "dining room",
    "dormitory",
    "driveway",
    "elevator",
    "entryway",
    "family room",
    "forest",
    "foyer",
    "gallery",
    "game room",
    "garage",
    "garden",
    "greenhouse",
    "gymnasium",
    "hallway",
    "kitchen",
    "laboratory",
    "laundry room",
    "lawn",
    "library",
    "living room",
    "loading dock",
    "lobby",
    "locker room",
    "lounge",
    "machine room",
    "mailroom",
    "mezzanine",
    "mudroom",
    "nursery",
    "office",
    "operating room",
    "orchard",
    "pantry",
    "park",
    "parking lot",
    "parlor",
    "patio",
    "picnic area",
    "playground",
    "playroom",
    "plaza",
    "pool area",
    "porch",
    "reception",
    "road",
    "rooftop",
    "seating area",
    "server room",
    "showroom",
    "sidewalk",
    "sports field",
    "stairwell",
    "storage room",
    "studio",
    "study",
    "sunroom",
    "terrace",
    "theater",
    "trail",
    "utility room",
    "vault",
    "vegetation",
    "vestibule",
    "waiting room",
    "walkway",
    "workshop",
)

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


class LabelingAborted(RuntimeError):
    """Raised when the operator cancels before a scene is written."""


def node_symbol(node: spark_dsg.SceneGraphNode) -> str:
    return str(node.id.str(True))


def symbol_sort_key(symbol: str) -> tuple[str, int | str]:
    prefix = symbol[:1]
    suffix = symbol[1:]
    return (prefix, int(suffix) if suffix.isdigit() else suffix)


def sorted_layer_nodes(
    graph: spark_dsg.DynamicSceneGraph,
    layer: spark_dsg.DsgLayers,
) -> list[spark_dsg.SceneGraphNode]:
    return sorted(
        list(graph.get_layer(layer).nodes),
        key=lambda node: symbol_sort_key(node_symbol(node)),
    )


def metadata_dict(graph: spark_dsg.DynamicSceneGraph) -> dict[str, Any]:
    # spark_dsg exposes the top-level metadata object as a mappingproxy. Convert that
    # proxy before copying so it can be passed back through metadata.set/json.dumps.
    return copy.deepcopy(dict(graph.metadata.get() or {}))


def labelspace_mapping(
    graph: spark_dsg.DynamicSceneGraph,
    keys: Iterable[str],
) -> dict[int, str]:
    labelspaces = metadata_dict(graph).get("labelspaces") or {}
    if not isinstance(labelspaces, dict):
        return {}
    for key in keys:
        entries = labelspaces.get(key)
        if not isinstance(entries, list):
            continue
        mapping: dict[int, str] = {}
        for entry in entries:
            if not isinstance(entry, list | tuple) or len(entry) != 2:
                continue
            try:
                mapping[int(entry[0])] = str(entry[1])
            except (TypeError, ValueError):
                continue
        if mapping:
            return mapping
    return {}


def current_room_label(
    room: spark_dsg.SceneGraphNode,
    room_labelspace: dict[int, str],
) -> str | None:
    try:
        return room_labelspace.get(int(room.attributes.semantic_label))
    except (AttributeError, TypeError, ValueError):
        return None


def contained_objects(
    graph: spark_dsg.DynamicSceneGraph,
    room: spark_dsg.SceneGraphNode,
) -> list[spark_dsg.SceneGraphNode]:
    object_nodes = sorted_layer_nodes(graph, spark_dsg.DsgLayers.OBJECTS)
    object_ids = {int(node.id.value) for node in object_nodes}
    found: dict[int, spark_dsg.SceneGraphNode] = {}
    visited: set[int] = set()
    queue = deque(int(child_id) for child_id in room.children())
    while queue:
        child_id = queue.popleft()
        if child_id in visited:
            continue
        visited.add(child_id)
        child = graph.get_node(child_id)
        if child_id in object_ids:
            found[child_id] = child
            continue
        queue.extend(int(grandchild_id) for grandchild_id in child.children())
    return sorted(found.values(), key=lambda node: symbol_sort_key(node_symbol(node)))


def object_class_counts(
    graph: spark_dsg.DynamicSceneGraph,
    room: spark_dsg.SceneGraphNode,
) -> Counter[str]:
    object_labelspace = labelspace_mapping(graph, OBJECT_LABELSPACE_KEYS)
    counts: Counter[str] = Counter()
    for obj in contained_objects(graph, room):
        try:
            label_id = int(obj.attributes.semantic_label)
        except (AttributeError, TypeError, ValueError):
            counts["unknown"] += 1
            continue
        counts[object_labelspace.get(label_id, f"unknown:{label_id}")] += 1
    return counts


def format_label_choices(labels: tuple[str, ...] = ROOM_LABELS, columns: int = 3) -> str:
    numbered = [f"{index:>2}. {label}" for index, label in enumerate(labels, start=1)]
    width = max(len(value) for value in numbered) + 3
    rows = []
    for start in range(0, len(numbered), columns):
        row = "".join(value.ljust(width) for value in numbered[start : start + columns])
        rows.append(row.rstrip())
    return "\n".join(rows)


def canonical_room_label(value: str, labels: tuple[str, ...] = ROOM_LABELS) -> str | None:
    normalized = " ".join(value.lower().split())
    return normalized if normalized in labels else None


def parse_label_choice(value: str, labels: tuple[str, ...] = ROOM_LABELS) -> str | None:
    normalized = value.strip()
    if normalized.isdigit():
        index = int(normalized) - 1
        return labels[index] if 0 <= index < len(labels) else None
    return canonical_room_label(normalized, labels)


def format_object_counts(counts: Counter[str]) -> list[str]:
    if not counts:
        return ["  (no contained objects)"]
    return [
        f"  {class_name:<24} {count:>4}"
        for class_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


@dataclass
class ScenePlot:
    figure: Any
    axes: Any
    room_patches: dict[str, Any]

    def highlight(self, room_symbol: str) -> None:
        for symbol, patch in self.room_patches.items():
            active = symbol == room_symbol
            patch.set_edgecolor("darkorange" if active else "red")
            patch.set_linewidth(3.0 if active else 1.5)
            patch.set_alpha(1.0 if active else 0.55)
            patch.set_zorder(5 if active else 2)
        self.axes.set_title(f"3D Scene Graph — classify {room_symbol}")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)


def visualize_scene(
    graph: spark_dsg.DynamicSceneGraph,
    *,
    show: bool = True,
    figure_output: Path | None = None,
) -> ScenePlot:
    spark_dsg.add_bounding_boxes_to_layer(
        graph,
        spark_dsg.DsgLayers.ROOMS,
        child_layer=spark_dsg.DsgLayers.MESH_PLACES,
    )
    figure, axes = plt.subplots(figsize=(12, 10))
    axes.set_title("3D Scene Graph — room classification")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")

    object_positions = []
    for obj in sorted_layer_nodes(graph, spark_dsg.DsgLayers.OBJECTS):
        position = np.asarray(obj.attributes.position, dtype=float)[:2]
        if np.isfinite(position).all():
            object_positions.append(position)
    if object_positions:
        positions = np.asarray(object_positions)
        axes.scatter(
            positions[:, 0],
            positions[:, 1],
            c="blue",
            marker="^",
            s=45,
            alpha=0.65,
            label="Objects",
            zorder=3,
        )

    room_patches: dict[str, Any] = {}
    for index, room in enumerate(sorted_layer_nodes(graph, spark_dsg.DsgLayers.ROOMS)):
        bounding_box = room.attributes.bounding_box
        if not bounding_box.is_valid():
            continue
        symbol = node_symbol(room)
        minimum = np.asarray(bounding_box.min, dtype=float)[:2]
        dimensions = np.asarray(bounding_box.dimensions, dtype=float)[:2]
        center = np.asarray(bounding_box.world_P_center, dtype=float)[:2]
        patch = plt.Rectangle(
            minimum - np.asarray([0.1, 0.1]),
            dimensions[0] + 0.2,
            dimensions[1] + 0.2,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.55,
            label="Rooms / regions" if index == 0 else None,
            zorder=2,
        )
        axes.add_patch(patch)
        axes.text(
            center[0],
            center[1],
            symbol,
            fontsize=9,
            fontweight="bold",
            color="red",
            horizontalalignment="center",
            verticalalignment="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            zorder=6,
        )
        room_patches[symbol] = patch

    if not room_patches:
        plt.close(figure)
        raise ValueError("The scene has no rooms/regions with computable bounding boxes.")

    axes.set_aspect("equal", adjustable="box")
    axes.legend(loc="upper right")
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    if figure_output is not None:
        figure_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_output, dpi=180, bbox_inches="tight")
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    return ScenePlot(figure=figure, axes=axes, room_patches=room_patches)


def prompt_for_room_labels(
    graph: spark_dsg.DynamicSceneGraph,
    *,
    scene_plot: ScenePlot | None = None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> dict[str, str]:
    rooms = sorted_layer_nodes(graph, spark_dsg.DsgLayers.ROOMS)
    if not rooms:
        raise ValueError("The scene does not contain any rooms/regions.")
    room_labelspace = labelspace_mapping(graph, (ROOM_LABELSPACE_KEY,))
    choices = format_label_choices()
    output_fn("Allowed room / outdoor region labels:")
    output_fn(choices)
    output_fn("Enter a label name or number. Enter '?' to show the list again; 'q' aborts.\n")

    assignments: dict[str, str] = {}
    for index, room in enumerate(rooms, start=1):
        symbol = node_symbol(room)
        current = current_room_label(room, room_labelspace)
        if scene_plot is not None:
            scene_plot.highlight(symbol)
        counts = object_class_counts(graph, room)
        output_fn(f"[{index}/{len(rooms)}] Region {symbol}")
        output_fn(f"Current label: {current or '(unlabeled)'}")
        output_fn(f"Contained objects: {sum(counts.values())}")
        for line in format_object_counts(counts):
            output_fn(line)

        while True:
            default = canonical_room_label(current) if current else None
            default_hint = f" [Enter keeps '{default}']" if default else ""
            try:
                response = input_fn(f"Label for {symbol}{default_hint}: ").strip()
            except (EOFError, KeyboardInterrupt) as exc:
                raise LabelingAborted("Room labeling was cancelled; no file was written.") from exc
            if response.lower() in {"q", "quit", "abort"}:
                raise LabelingAborted("Room labeling was cancelled; no file was written.")
            if response == "?":
                output_fn(choices)
                continue
            if not response and default is not None:
                assignments[symbol] = default
                break
            selected = parse_label_choice(response)
            if selected is None:
                output_fn("Invalid label. Choose one of the displayed names or numbers.")
                continue
            assignments[symbol] = selected
            break
        output_fn(f"Assigned {symbol} -> {assignments[symbol]}\n")
    return assignments


def updated_labelspace_entries(
    existing_entries: Any,
    assignments: dict[str, str],
) -> tuple[list[list[int | str]], dict[str, int]]:
    entries: list[list[int | str]] = []
    label_to_id: dict[str, int] = {}
    used_ids: set[int] = set()
    if isinstance(existing_entries, list):
        for entry in existing_entries:
            if not isinstance(entry, list | tuple) or len(entry) != 2:
                continue
            try:
                label_id = int(entry[0])
            except (TypeError, ValueError):
                continue
            label = " ".join(str(entry[1]).lower().split())
            if not label or label_id in used_ids:
                continue
            entries.append([label_id, label])
            used_ids.add(label_id)
            label_to_id.setdefault(label, label_id)

    next_id = max(used_ids, default=-1) + 1
    for label in assignments.values():
        if label not in ROOM_LABELS:
            raise ValueError(f"Unsupported room label: {label!r}")
        if label in label_to_id:
            continue
        while next_id in used_ids:
            next_id += 1
        entries.append([next_id, label])
        label_to_id[label] = next_id
        used_ids.add(next_id)
        next_id += 1
    entries.sort(key=lambda entry: int(entry[0]))
    return entries, label_to_id


def apply_room_labels(
    graph: spark_dsg.DynamicSceneGraph,
    assignments: dict[str, str],
) -> None:
    rooms = {
        node_symbol(room): room
        for room in sorted_layer_nodes(graph, spark_dsg.DsgLayers.ROOMS)
    }
    missing = sorted(set(assignments) - set(rooms), key=symbol_sort_key)
    if missing:
        raise KeyError(f"Unknown room/region symbols: {', '.join(missing)}")

    metadata = metadata_dict(graph)
    labelspaces = metadata.setdefault("labelspaces", {})
    if not isinstance(labelspaces, dict):
        raise TypeError("Scene metadata.labelspaces must be a dictionary.")
    entries, label_to_id = updated_labelspace_entries(
        labelspaces.get(ROOM_LABELSPACE_KEY), assignments
    )
    labelspaces[ROOM_LABELSPACE_KEY] = entries
    graph.metadata.set(metadata)
    for symbol, label in assignments.items():
        rooms[symbol].attributes.semantic_label = label_to_id[label]


def default_output_path(scene_path: Path) -> Path:
    return scene_path.with_name(f"{scene_path.stem}.room_labels{scene_path.suffix}")


def save_labeled_scene(
    source_path: Path,
    output_path: Path,
    assignments: dict[str, str],
) -> None:
    graph = spark_dsg.DynamicSceneGraph.load(str(source_path))
    apply_room_labels(graph, assignments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".json",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        graph.save(str(temporary_path), include_mesh=True)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively classify every room/region in a spark_dsg scene and write "
            "an updated scene JSON."
        )
    )
    parser.add_argument("scene_json", nargs="?", type=Path, help="Input 3DSG JSON file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing output file, including the input file",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib window (useful for headless sessions)",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        help="Optionally save the top-down visualization as an image",
    )
    parser.add_argument(
        "--list-labels",
        action="store_true",
        help="Print the allowed room labels and exit when no scene is provided",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_labels:
        print(format_label_choices())
        if args.scene_json is None:
            return 0
    if args.scene_json is None:
        parser.error("scene_json is required unless --list-labels is used")

    source_path = args.scene_json
    if not source_path.is_file():
        parser.error(f"input scene does not exist: {source_path}")
    output_path = args.output or default_output_path(source_path)
    if output_path.exists() and not args.force:
        parser.error(f"output already exists: {output_path} (use --force to replace it)")
    try:
        same_path = source_path.resolve() == output_path.resolve()
    except OSError:
        same_path = source_path.absolute() == output_path.absolute()
    if same_path and not args.force:
        parser.error("refusing to overwrite the input scene without --force")

    display_graph = spark_dsg.DynamicSceneGraph.load(str(source_path))
    scene_plot = None
    if not args.no_show or args.figure_output is not None:
        scene_plot = visualize_scene(
            display_graph,
            show=not args.no_show,
            figure_output=args.figure_output,
        )
    try:
        assignments = prompt_for_room_labels(display_graph, scene_plot=scene_plot)
    except LabelingAborted as exc:
        print(f"\n{exc}")
        if scene_plot is not None:
            plt.close(scene_plot.figure)
        return 130

    save_labeled_scene(source_path, output_path, assignments)
    print(f"Updated {len(assignments)} rooms/regions.")
    print(f"Wrote labeled scene: {output_path}")
    if args.figure_output is not None:
        print(f"Wrote visualization: {args.figure_output}")
    if scene_plot is not None and not args.no_show:
        scene_plot.axes.set_title("3D Scene Graph — room classification complete")
        scene_plot.figure.canvas.draw_idle()
        print("Close the visualization window to exit.")
        plt.show()
    elif scene_plot is not None:
        plt.close(scene_plot.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
