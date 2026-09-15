import pytest

from cypher_3dsg_agent_distillation.core.scene_ordering import (
    order_records_by_scene_interleave,
)


def records_by_scene(count: int = 4):
    return [
        {"uid": f"{scene}-{index}", "scene_id": scene}
        for scene in ("scene-1", "scene-2", "scene-3")
        for index in range(count)
    ]


def scene_sequence(records):
    return [record["scene_id"] for record in records]


def test_zero_scene_interleave_groups_validation_records_by_scene():
    ordered = order_records_by_scene_interleave(
        records_by_scene(),
        ratio=0.0,
        seed=42,
        scene_order=("scene-1", "scene-2", "scene-3"),
    )

    assert scene_sequence(ordered) == [
        *(["scene-1"] * 4),
        *(["scene-2"] * 4),
        *(["scene-3"] * 4),
    ]


def test_full_scene_interleave_is_round_robin_and_deterministic():
    kwargs = {
        "ratio": 1.0,
        "seed": 42,
        "scene_order": ("scene-1", "scene-2", "scene-3"),
    }

    first = order_records_by_scene_interleave(records_by_scene(), **kwargs)
    second = order_records_by_scene_interleave(records_by_scene(), **kwargs)

    assert scene_sequence(first) == ["scene-1", "scene-2", "scene-3"] * 4
    assert [record["uid"] for record in first] == [record["uid"] for record in second]


def test_partial_scene_interleave_lies_between_the_endpoint_switch_counts():
    records = records_by_scene(10)
    schedules = [
        order_records_by_scene_interleave(records, ratio=ratio, seed=7)
        for ratio in (0.0, 0.3, 1.0)
    ]

    switch_counts = [
        sum(
            left != right
            for left, right in zip(scenes, scenes[1:], strict=False)
        )
        for scenes in map(scene_sequence, schedules)
    ]
    assert switch_counts[0] < switch_counts[1] < switch_counts[2]


def test_scene_interleave_ratio_must_be_between_zero_and_one():
    with pytest.raises(ValueError, match="between 0 and 1"):
        order_records_by_scene_interleave([], ratio=1.1, seed=1)
