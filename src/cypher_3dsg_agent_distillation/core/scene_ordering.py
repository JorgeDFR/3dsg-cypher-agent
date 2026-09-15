from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

RecordT = TypeVar("RecordT")


def validate_scene_interleave_ratio(value: float) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(
            f"train_scene_interleave_ratio must be between 0 and 1, got {ratio}."
        )
    return ratio


def default_scene_id(record: object) -> str:
    if isinstance(record, dict):
        return str(record.get("scene_id") or "unknown")
    return str(getattr(record, "scene_id", None) or "unknown")


def order_records_by_scene_interleave(
    records: Sequence[RecordT],
    *,
    ratio: float,
    seed: int,
    scene_order: Iterable[str] = (),
    scene_id: Callable[[RecordT], str] = default_scene_id,
) -> list[RecordT]:
    """Order records between scene-grouped (0) and round-robin (1).

    Records are shuffled deterministically inside each scene first. Intermediate
    ratios divide each scene into progressively smaller chunks, providing a
    stable continuum between the two endpoint schedules.
    """
    ratio = validate_scene_interleave_ratio(ratio)
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

    grouped = [
        indexed_record
        for record_scene_id in ordered_scene_ids
        for indexed_record in buckets[record_scene_id]
    ]
    if ratio == 0.0:
        return [record for _index, record in grouped]

    maximum_scene_size = max(len(buckets[value]) for value in ordered_scene_ids)
    rounds = 1 + round(ratio * (maximum_scene_size - 1))
    chunk_size = math.ceil(maximum_scene_size / rounds)
    interleaved: list[tuple[int, RecordT]] = []
    offsets = {record_scene_id: 0 for record_scene_id in ordered_scene_ids}
    while len(interleaved) < len(indexed_records):
        for record_scene_id in ordered_scene_ids:
            offset = offsets[record_scene_id]
            bucket = buckets[record_scene_id]
            if offset < len(bucket):
                next_offset = min(offset + chunk_size, len(bucket))
                interleaved.extend(bucket[offset:next_offset])
                offsets[record_scene_id] = next_offset
    return [record for _index, record in interleaved]
