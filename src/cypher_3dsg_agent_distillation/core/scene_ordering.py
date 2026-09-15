from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

RecordT = TypeVar("RecordT")


def default_scene_id(record: object) -> str:
    if isinstance(record, dict):
        return str(record.get("scene_id") or "unknown")
    return str(getattr(record, "scene_id", None) or "unknown")


def group_records_by_scene(
    records: Sequence[RecordT],
    *,
    seed: int,
    scene_order: Iterable[str] = (),
    scene_id: Callable[[RecordT], str] = default_scene_id,
) -> list[RecordT]:
    """Group records by scene and shuffle deterministically within each scene."""
    indexed_records = list(enumerate(records))
    if len(indexed_records) < 2:
        return list(records)

    buckets: dict[str, list[tuple[int, RecordT]]] = {}
    ordered_scene_ids: list[str] = []
    for configured_scene_id in scene_order:
        normalized = str(configured_scene_id)
        if normalized not in buckets:
            buckets[normalized] = []
            ordered_scene_ids.append(normalized)
    for indexed_record in indexed_records:
        record_scene_id = scene_id(indexed_record[1])
        if record_scene_id not in buckets:
            buckets[record_scene_id] = []
            ordered_scene_ids.append(record_scene_id)
        buckets[record_scene_id].append(indexed_record)

    ordered_scene_ids = [value for value in ordered_scene_ids if buckets[value]]
    for record_scene_id in ordered_scene_ids:
        random.Random(f"{seed}:scene-order:{record_scene_id}").shuffle(
            buckets[record_scene_id]
        )

    return [
        record
        for record_scene_id in ordered_scene_ids
        for _index, record in buckets[record_scene_id]
    ]
