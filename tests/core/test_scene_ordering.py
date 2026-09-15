from cypher_3dsg_agent_distillation.core.scene_ordering import (
    group_records_by_scene,
)


def records_by_scene(count: int = 4):
    return [
        {"uid": f"{scene}-{index}", "scene_id": scene}
        for scene in ("scene-1", "scene-2", "scene-3")
        for index in range(count)
    ]


def scene_sequence(records):
    return [record["scene_id"] for record in records]


def test_groups_validation_records_by_scene_deterministically():
    ordered = group_records_by_scene(
        records_by_scene(),
        seed=42,
        scene_order=("scene-1", "scene-2", "scene-3"),
    )

    second = group_records_by_scene(
        records_by_scene(),
        seed=42,
        scene_order=("scene-1", "scene-2", "scene-3"),
    )

    assert scene_sequence(ordered) == [
        *(["scene-1"] * 4),
        *(["scene-2"] * 4),
        *(["scene-3"] * 4),
    ]
    assert [record["uid"] for record in ordered] == [
        record["uid"] for record in second
    ]
