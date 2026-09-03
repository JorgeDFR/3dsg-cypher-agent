from pathlib import Path

import matplotlib
import spark_dsg

matplotlib.use("Agg")

from cypher_3dsg_agent_distillation.graph.room_label_editor import (  # noqa: E402
    ROOM_LABELS,
    ROOM_LABELSPACE_KEY,
    apply_room_labels,
    default_output_path,
    format_label_choices,
    object_class_counts,
    parse_label_choice,
    prompt_for_room_labels,
    save_labeled_scene,
    sorted_layer_nodes,
    updated_labelspace_entries,
    visualize_scene,
)

SCENE_PATH = Path(__file__).resolve().parents[2] / "data" / "scenes" / "uhumans2_office.json"


def test_room_label_vocabulary_is_unique_single_word_and_selectable():
    assert len(ROOM_LABELS) == len(set(ROOM_LABELS))
    assert ROOM_LABELS == tuple(sorted(ROOM_LABELS))
    assert all(label.isalpha() and label == label.lower() for label in ROOM_LABELS)
    office_number = ROOM_LABELS.index("office") + 1

    assert parse_label_choice(" Office ") == "office"
    assert parse_label_choice(str(office_number)) == "office"
    assert parse_label_choice("meeting room") is None
    assert parse_label_choice("999") is None
    assert "office" in format_label_choices()


def test_updated_labelspace_preserves_ids_and_appends_selected_labels():
    entries, label_to_id = updated_labelspace_entries(
        [[0, "lounge"], [1, "hallway"]],
        {"R0": "office", "R1": "lounge", "R2": "kitchen"},
    )

    assert entries == [[0, "lounge"], [1, "hallway"], [2, "office"], [3, "kitchen"]]
    assert label_to_id == {"lounge": 0, "hallway": 1, "office": 2, "kitchen": 3}


def test_prompt_lists_object_class_counts_and_visits_rooms_in_symbol_order():
    graph = spark_dsg.DynamicSceneGraph.load(str(SCENE_PATH))
    room_count = len(sorted_layer_nodes(graph, spark_dsg.DsgLayers.ROOMS))
    responses = iter(["office"] * room_count)
    output: list[str] = []

    assignments = prompt_for_room_labels(
        graph,
        input_fn=lambda _prompt: next(responses),
        output_fn=output.append,
    )

    assert list(assignments) == [f"R{index}" for index in range(room_count)]
    assert set(assignments.values()) == {"office"}
    assert any("[1/8] Region R0" in line for line in output)
    assert any("Contained objects:" in line for line in output)
    assert any("light" in line for line in output)
    first_room = next(graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes)
    assert sum(object_class_counts(graph, first_room).values()) > 0


def test_save_labeled_scene_updates_spark_dsg_metadata_and_room_attributes(tmp_path):
    output_path = tmp_path / "labeled.json"

    save_labeled_scene(
        SCENE_PATH,
        output_path,
        {"R0": "office", "R1": "kitchen"},
    )

    graph = spark_dsg.DynamicSceneGraph.load(str(output_path))
    metadata = dict(graph.metadata.get())
    room_labels = {
        int(label_id): name
        for label_id, name in metadata["labelspaces"][ROOM_LABELSPACE_KEY]
    }
    rooms = {
        room.id.str(True): room
        for room in graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes
    }
    assert room_labels[int(rooms["R0"].attributes.semantic_label)] == "office"
    assert room_labels[int(rooms["R1"].attributes.semantic_label)] == "kitchen"
    assert room_labels[0] == "lounge"
    assert room_labels[1] == "hallway"


def test_visualization_draws_all_room_boxes_and_can_save_image(tmp_path):
    graph = spark_dsg.DynamicSceneGraph.load(str(SCENE_PATH))
    figure_path = tmp_path / "top_down.png"

    scene_plot = visualize_scene(graph, show=False, figure_output=figure_path)

    assert figure_path.is_file()
    assert set(scene_plot.room_patches) == {f"R{index}" for index in range(8)}
    assert scene_plot.axes.collections  # Blue object triangle collection.
    scene_plot.highlight("R3")
    assert scene_plot.room_patches["R3"].get_edgecolor()[-1] == 1.0
    matplotlib.pyplot.close(scene_plot.figure)


def test_apply_room_labels_rejects_unknown_symbols_and_default_output_is_separate():
    graph = spark_dsg.DynamicSceneGraph.load(str(SCENE_PATH))

    try:
        apply_room_labels(graph, {"R999": "office"})
    except KeyError as exc:
        assert "R999" in str(exc)
    else:
        raise AssertionError("Unknown room symbol was accepted")

    assert default_output_path(Path("scene.json")) == Path("scene.room_labels.json")
