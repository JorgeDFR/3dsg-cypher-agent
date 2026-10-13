import copy
import json
from pathlib import Path

import pytest
import spark_dsg

from cypher_3dsg_agent_distillation.graph.object_place_edges import (
    add_missing_object_place_edges,
    main,
)


def node(node_id, layer, partition, point):
    return {
        "id": node_id,
        "layer": layer,
        "partition": partition,
        "attributes": {"position": point},
    }


@pytest.fixture
def scene():
    return {
        "layer_names": {
            "OBJECTS": {"layer": 2, "partition": 0},
            "MESH_PLACES": {"layer": 3, "partition": 1},
            "PLACES": {"layer": 3, "partition": 0},
        },
        "nodes": [
            node(1, 2, 0, [0, 0, 0]),
            node(2, 2, 0, [10, 0, 0]),
            node(3, 2, 97, [0, 0, 0]),  # Agent, not an object.
            node(11, 3, 1, [0, 0, 5]),  # XY nearest, but not XYZ nearest.
            node(12, 3, 1, [1, 0, 0]),
            node(13, 3, 1, [9, 0, 0]),
            node(21, 3, 0, [-1, 0, 0]),
            node(22, 3, 0, [11, 0, 0]),
        ],
        "edges": [{"source": 11, "target": 12, "info": {"metadata": {"keep": True}}}],
        "metadata": {"keep": [1, 2, 3]},
        "mesh": {"keep": True},
    }


@pytest.mark.parametrize("target_layer,parents", [("mesh-place", [12, 13]), ("place", [21, 22])])
def test_repairs_selected_layer_by_xyz_distance_and_preserves_other_data(
    scene, target_layer, parents
):
    before = copy.deepcopy(scene)
    result = add_missing_object_place_edges(scene, target_layer)
    assert result.added == result.objects == 2
    assert result.already_connected == result.invalid_positions == 0
    assert [(edge["source"], edge["target"]) for edge in scene["edges"][1:]] == [
        (parents[0], 1),
        (parents[1], 2),
    ]
    assert scene["edges"][:1] == before["edges"]
    assert {key: value for key, value in scene.items() if key != "edges"} == {
        key: value for key, value in before.items() if key != "edges"
    }
    assert add_missing_object_place_edges(scene, target_layer).added == 0


def test_existing_edges_in_both_orientations_are_kept(scene):
    scene["edges"].extend([{"source": 1, "target": 11}, {"source": 13, "target": 2}])
    before = copy.deepcopy(scene["edges"])
    result = add_missing_object_place_edges(scene, "mesh-place")
    assert result.already_connected == 2
    assert result.added == 0
    assert scene["edges"] == before


def test_other_layer_connections_do_not_prevent_repair_and_named_aliases_are_honored(scene):
    scene["edges"].append({"source": 21, "target": 1})
    scene["layer_names"]["PLACES"] = {"layer": 3, "partition": 1}
    result = add_missing_object_place_edges(scene, "place")
    assert result.added == 2
    assert scene["edges"][-2]["source"] == 12


def test_equal_distances_choose_lowest_id_independent_of_node_order(scene):
    scene["nodes"].append(node(10, 3, 1, [-1, 0, 0]))
    add_missing_object_place_edges(scene, "mesh-place")
    assert scene["edges"][-2]["source"] == 10


def test_invalid_object_and_target_positions_are_reported_and_skipped(scene):
    scene["nodes"][0]["attributes"]["position"] = None
    scene["nodes"].append(node(14, 3, 1, [float("nan"), 0, 0]))
    result = add_missing_object_place_edges(scene, "mesh-place")
    assert result.invalid_positions == 1
    assert result.added == 1
    assert scene["edges"][-1]["target"] == 2


def test_empty_target_layer_fails_without_mutating_edges(scene):
    scene["nodes"] = [n for n in scene["nodes"] if n["layer"] != 3]
    before = copy.deepcopy(scene["edges"])
    with pytest.raises(ValueError, match="No mesh-place nodes"):
        add_missing_object_place_edges(scene, "mesh-place")
    assert scene["edges"] == before


def test_cli_dry_run_default_output_and_overwrite_protection(scene, tmp_path, capsys):
    source = tmp_path / "scene.json"
    source.write_text(json.dumps(scene))
    original = source.read_bytes()
    args = [str(source), "--target-layer", "mesh-place"]
    output = tmp_path / "scene.object_place_edges.json"
    assert main([*args, "--dry-run"]) == 0
    assert "Would add 2" in capsys.readouterr().out
    assert not output.exists()
    assert main(args) == 0
    assert source.read_bytes() == original
    assert len(json.loads(output.read_text())["edges"]) == 3
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        main([*args, "--output", str(source)])
    assert main([*args, "--output", str(source), "--force"]) == 0
    assert len(json.loads(source.read_text())["edges"]) == 3


def test_repaired_scene_loads_in_spark_dsg_and_gives_objects_place_parents(tmp_path):
    source = Path(__file__).resolve().parents[2] / "data/scenes/uhumans2_office.json"
    output = tmp_path / "repaired.json"
    assert main([str(source), "--target-layer", "place", "--output", str(output)]) == 0
    graph = spark_dsg.DynamicSceneGraph.load(str(output))
    objects = list(graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes)
    place_ids = {int(n.id.value) for n in graph.get_layer(spark_dsg.DsgLayers.PLACES).nodes}
    assert objects
    assert all(any(int(parent) in place_ids for parent in obj.parents()) for obj in objects)
