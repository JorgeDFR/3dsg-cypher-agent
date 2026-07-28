from pathlib import Path

import spark_dsg

from cypher_3dsg_agent_distillation.heracles_loader import (
    labelspace_from_metadata,
    write_heracles_labelspace,
)
from cypher_3dsg_agent_distillation.load_neo4j import extract_graph_records
from cypher_3dsg_agent_distillation.scene_graph import SceneIndex, node_symbol, scene_values

SCENE_PATH = Path("data/scenes/3dsg_uhumans2.json")


def test_node_symbol_matches_spark_dsg_encoding():
    assert node_symbol(0x4F0000000000002A) == "O42"
    assert node_symbol(0x7000000000000007) == "p7"
    assert node_symbol(0x5000000000001626) == "P5670"


def test_extract_scene_values_from_spark_dsg():
    values = scene_values(SCENE_PATH, 30, 10, 10, 10, 3)
    assert values["object_classes"][:3] == ["light", "cabinet", "desk"]
    assert values["object_symbols"][:3] == ["O0", "O1", "O3"]
    assert values["place_symbols"][:3] == ["p0", "p1", "p4"]
    assert values["mesh_place_symbols"][:3] == ["P5670", "P5671", "P5672"]
    assert len(values["place_object_cases"]) == 20
    assert len(values["object_radius_cases"]) == 40
    assert len(values["nearest_object_mesh_cases"]) == 60
    assert len(values["nearest_object_cases"]) == 40
    assert len(values["mesh_place_hop_cases"]) == 25
    assert len(values["place_hop_cases"]) == 25


def test_extract_graph_records_from_spark_dsg():
    nodes, relationships = extract_graph_records(SCENE_PATH)
    assert {label: len(items) for label, items in nodes.items()} == {
        "Object": 255,
        "Place": 1338,
        "MeshPlace": 2191,
        "Room": 0,
        "Building": 0,
    }
    assert {
        (relationship["from_label"], relationship["type"], relationship["to_label"])
        for relationship in relationships
    } == {
        ("Place", "CONTAINS", "Object"),
        ("Place", "PLACE_CONNECTED", "Place"),
        ("MeshPlace", "MESH_PLACE_CONNECTED", "MeshPlace"),
    }


def test_heracles_labelspace_uses_embedded_spark_dsg_labels(tmp_path):
    graph = spark_dsg.DynamicSceneGraph.load(str(SCENE_PATH))
    labelspace = labelspace_from_metadata(graph, ["_l2p0", "_l3p1", "mesh"])
    output = tmp_path / "labelspace.yaml"

    write_heracles_labelspace(labelspace, output)

    assert labelspace["82"] == "light"
    assert "82" in output.read_text()
    assert "light" in output.read_text()


def test_solution_modes_can_be_selected():
    index = SceneIndex(SCENE_PATH)
    variables = {"class": "stool"}

    simplified = index.solution_for("object_class_centers", variables, mode="simplified")
    informative = index.solution_for("object_class_centers", variables, mode="informative")

    assert simplified == "POINT(11.832 26.993 1.432)"
    assert informative == [
        {
            "nodeSymbol": "O481",
            "center": "POINT(11.832 26.993 1.432)",
            "class": "stool",
        }
    ]


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
    assert radius == {"O166", "O261", "O498"}


def test_object_class_inventory_uses_object_labels_only():
    index = SceneIndex(SCENE_PATH)

    inventory = index.solution_for("object_class_inventory", {})

    assert "light" in inventory
    assert "floor" not in inventory


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

    assert sum(solution == set() for solution in place_solutions) == 6
    assert sum(solution != set() for solution in place_solutions) == 14
    assert sum(solution == set() for solution in radius_solutions) >= 10
    assert sum(solution != set() for solution in radius_solutions) >= 25
