from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import spark_dsg
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERACLES_SRC = PROJECT_ROOT / "external" / "heracles" / "heracles" / "src"

OBJECT_LABELSPACE_KEYS = ["_l2p0", "_l3p1", "mesh"]
ROOM_LABELSPACE_KEYS = ["room_labelspace", "_l4p0"]


def ensure_heracles_on_path() -> None:
    if not HERACLES_SRC.exists():
        raise RuntimeError(
            "Heracles submodule is missing. Run "
            "`git submodule update --init --recursive external/heracles`."
        )
    src = str(HERACLES_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def labelspace_from_metadata(
    graph: spark_dsg.DynamicSceneGraph,
    keys: list[str],
) -> dict[str, str]:
    metadata = graph.metadata.get() or {}
    labelspaces = metadata.get("labelspaces", {})
    if not isinstance(labelspaces, dict):
        return {}

    merged: dict[str, str] = {}
    for key in keys:
        values = labelspaces.get(key)
        if not values:
            continue
        for label_id, name in values:
            merged.setdefault(str(label_id), str(name))
    return merged


def write_heracles_labelspace(labelspace: dict[str, str], path: Path) -> None:
    payload: dict[str, list[dict[str, Any]]] = {
        "label_names": [
            {"label": str(label_id), "name": name}
            for label_id, name in sorted(labelspace.items(), key=lambda item: int(item[0]))
        ]
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_scene_with_heracles(
    scene_path: str | Path,
    uri: str,
    username: str,
    password: str,
    object_labelspace: str = "ade20k_mit_label_space.yaml",
    room_labelspace: str = "b45_label_space.yaml",
) -> None:
    ensure_heracles_on_path()
    from heracles.utils import load_dsg_to_db

    graph = spark_dsg.DynamicSceneGraph.load(str(scene_path))
    object_labels = labelspace_from_metadata(graph, OBJECT_LABELSPACE_KEYS)
    room_labels = labelspace_from_metadata(graph, ROOM_LABELSPACE_KEYS)

    with tempfile.TemporaryDirectory(prefix="3dsg_heracles_labelspaces_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        object_labelspace_path = object_labelspace
        room_labelspace_path = room_labelspace

        if object_labels:
            object_labelspace_path = str(tmp_path / "object_labelspace.yaml")
            write_heracles_labelspace(object_labels, Path(object_labelspace_path))
        if room_labels:
            room_labelspace_path = str(tmp_path / "room_labelspace.yaml")
            write_heracles_labelspace(room_labels, Path(room_labelspace_path))

        load_dsg_to_db(
            object_labelspace_path,
            room_labelspace_path,
            uri,
            (username, password),
            graph,
        )
