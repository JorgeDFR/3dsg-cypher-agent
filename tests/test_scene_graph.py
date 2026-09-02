from collections import Counter
from pathlib import Path

import spark_dsg

from cypher_3dsg_agent_distillation.heracles_loader import (
    labelspace_from_metadata,
    write_heracles_labelspace,
)
from cypher_3dsg_agent_distillation.load_neo4j import (
    extract_graph_records,
    relationship_for,
)
from cypher_3dsg_agent_distillation.scene_graph import (
    SceneIndex,
    node_symbol,
    objects_in_bbox,
    ranked_class_names,
    scene_values,
    top_k_with_ties_validation,
)

SCENE_PATH = Path("data/scenes/uhumans2_office.json")


def test_node_symbol_matches_spark_dsg_encoding():
    assert node_symbol(0x4F0000000000002A) == "O42"
    assert node_symbol(0x7000000000000007) == "p7"
    assert node_symbol(0x5000000000001626) == "P5670"


def test_top_class_gold_ranking_breaks_count_ties_alphabetically():
    counts = Counter({"plant": 7, "flag": 7, "painting": 7, "light": 10})

    assert ranked_class_names(counts, 3) == ["light", "flag", "painting"]
    assert top_k_with_ties_validation(counts, 3) == {
        "mode": "top_k_with_ties",
        "limit": 3,
        "scores": {"flag": 7, "light": 10, "painting": 7, "plant": 7},
    }


def test_extract_scene_values_from_spark_dsg():
    values = scene_values(SCENE_PATH, 30, 10, 10, 10, 3)
    assert values["object_classes"][:3] == ["bookcase", "bulletin", "mirror"]
    assert "light" not in values["object_classes"]
    assert values["object_symbols"][:3] == ["O0", "O1", "O3"]
    assert values["place_symbols"][:3] == ["p0", "p1", "p4"]
    assert values["mesh_place_symbols"][:3] == ["P5670", "P5671", "P5672"]
    assert {"O2", "R8", "p2", "P0"} <= set(values["invalid_values"])
    index = SceneIndex(SCENE_PATH)
    assert len(values["bboxes"]) == 50
    assert len(values["room_containing_object_cases"]) == 50
    assert {
        case["region"] for case in values["room_containing_object_cases"][:8]
    } == {f"R{index}" for index in range(8)}
    assert all(len(objects_in_bbox(index, bbox)) <= 20 for bbox in values["bboxes"])
    assert len(values["place_object_cases"]) == 50
    assert len(values["object_radius_cases"]) == 50
    assert len(values["nearest_object_mesh_cases"]) == 60
    assert len(values["nearest_object_cases"]) == 50
    assert len(values["mesh_place_hop_cases"]) == 50
    assert len(values["place_hop_cases"]) == 50
    assert sum(
        case["object"] in values["invalid_values"]
        for case in values["room_containing_object_cases"]
    ) == 5
    assert {case["hops"] for case in values["mesh_place_hop_cases"]} <= {2, 3, 4}
    assert {case["hops"] for case in values["place_hop_cases"]} <= {2, 3, 4}
    assert all(
        len(index.hop_symbols(case["mesh_place"], "MeshPlace", case["hops"])) < 50
        for case in values["mesh_place_hop_cases"]
    )
    assert all(
        len(index.hop_symbols(case["place"], "Place", case["hops"])) < 50
        for case in values["place_hop_cases"]
    )


def test_extract_graph_records_from_spark_dsg():
    nodes, relationships = extract_graph_records(SCENE_PATH)
    assert {label: len(items) for label, items in nodes.items()} == {
        "Object": 213,
        "Place": 1130,
        "MeshPlace": 1494,
        "Room": 8,
        "Building": 1,
    }
    assert {
        (relationship["from_label"], relationship["type"], relationship["to_label"])
        for relationship in relationships
    } == {
        ("Place", "CONTAINS", "Object"),
        ("Place", "PLACE_CONNECTED", "Place"),
        ("MeshPlace", "MESH_PLACE_CONNECTED", "MeshPlace"),
        ("Room", "CONTAINS", "Place"),
        ("Room", "ROOM_CONNECTED", "Room"),
    }


def test_neo4j_loader_models_room_to_object_containment_as_two_edges():
    assert relationship_for("Room", "Place") == ("Room", "CONTAINS", "Place")
    assert relationship_for("Room", "MeshPlace") == (
        "Room",
        "CONTAINS",
        "MeshPlace",
    )
    assert relationship_for("Place", "Object") == ("Place", "CONTAINS", "Object")
    assert relationship_for("MeshPlace", "Object") == (
        "MeshPlace",
        "CONTAINS",
        "Object",
    )


def test_heracles_labelspace_uses_embedded_spark_dsg_labels(tmp_path):
    graph = spark_dsg.DynamicSceneGraph.load(str(SCENE_PATH))
    labelspace = labelspace_from_metadata(graph, ["_l2p0", "_l3p1", "mesh", "_l4p0"])
    output = tmp_path / "labelspace.yaml"

    write_heracles_labelspace(labelspace, output)

    assert labelspace["82"] == "light"
    assert "82" in output.read_text()
    assert "light" in output.read_text()


def test_solution_for_returns_simplified_values():
    index = SceneIndex(SCENE_PATH)
    variables = {"class": "stool"}

    solution = index.solution_for("object_class_centers", variables)

    assert solution == "POINT(11.830 27.001 1.435)"


def test_simplified_spatial_solutions_use_object_symbols():
    index = SceneIndex(SCENE_PATH)

    nearest = index.solution_for(
        "nearest_object_to_mesh_place",
        {"class": "light", "mesh_place": "P7047"},
    )
    radius = index.solution_for(
        "object_radius_from_point",
        {
            "class": "light",
            "radius": 5,
            "point": {"x": 2.055, "y": 27.813, "z": 2.247},
        },
    )

    assert nearest == "O8"
    assert radius == "O166"


def test_object_class_inventory_uses_object_labels_only():
    index = SceneIndex(SCENE_PATH)

    inventory = index.solution_for("object_class_inventory", {})

    assert "light" in inventory
    assert "floor" not in inventory


def test_room_pddl_solutions_support_class_and_object_containment_references():
    index = SceneIndex(SCENE_PATH)

    assert [node_symbol(node.id) for node in index.containing_rooms("O0")] == ["R0"]
    assert index.solution_for("pddl_visit_room_containing_object", {"object": "O0"}) == [
        "R0"
    ]
    assert index.solution_for("pddl_end_in_room_class", {"room_class": "lounge"}) == [
        "R0"
    ]


def test_room_containing_object_cases_are_balanced_across_regions():
    values = scene_values(SCENE_PATH, 30, 50, 10, 10, 3)
    region_counts = Counter(
        case["region"] for case in values["room_containing_object_cases"][:20]
    )

    assert set(region_counts) == {f"R{index}" for index in range(8)}
    assert max(region_counts.values()) - min(region_counts.values()) <= 1
    assert len(
        {case["object"] for case in values["room_containing_object_cases"]}
    ) == len(values["room_containing_object_cases"])


def test_curated_cases_include_empty_and_non_empty_solutions():
    index = SceneIndex(SCENE_PATH)
    values = scene_values(SCENE_PATH, 30, 10, 10, 10, 3)

    place_solutions = [
        index.solution_for("objects_in_place_symbol", {"case": case})
        for case in values["place_object_cases"]
    ]
    radius_solutions = [
        index.solution_for("object_radius_from_point", {"case": case})
        for case in values["object_radius_cases"]
    ]

    assert sum(solution == set() for solution in place_solutions) == 5
    assert sum(solution != set() for solution in place_solutions) == 45
    assert sum(solution == set() for solution in radius_solutions) == 5
    assert sum(solution != set() for solution in radius_solutions) == 45
