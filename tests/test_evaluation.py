from cypher_3dsg_agent_distillation.evaluation import (
    ResultShapeError,
    answer_matches,
    build_summary,
    evaluate_records,
    normalize_answer,
    point_to_solution,
    rows_to_solution_text,
)


class CartesianPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def test_rows_to_solution_text_matches_simplified_set_solution():
    record = {
        "query_type": "object_radius_from_point",
        "solution": "<O166, O261, O498>",
        "solution_formats": ["set[str]", "str"],
    }
    rows = [{"object": "O498"}, {"object": "O166"}, {"object": "O261"}]

    actual = rows_to_solution_text(record, rows)

    assert normalize_answer(actual) == normalize_answer(record["solution"])


def test_answer_matches_sldp_set_with_loose_order():
    expected = "<\n  armchair,\n  chair,\n  sofa\n>\n"
    actual = "<sofa, armchair, chair>"

    assert answer_matches(actual, expected)


def test_answer_matches_does_not_accept_missing_set_items():
    expected = "<armchair, chair, sofa>"
    actual = "sofa, chair"

    assert not answer_matches(actual, expected)


def test_rows_to_solution_text_uses_declared_set_format():
    record = {
        "query_type": "top_object_classes_by_count",
        "solution": "<light, cabinet>",
        "solution_formats": ["set[str]"],
    }
    rows = [
        {"class": "light", "count": 57},
        {"class": "cabinet", "count": 30},
    ]

    actual = rows_to_solution_text(record, rows)

    assert actual == "<light, cabinet>"


def test_rows_to_solution_text_infers_set_format_from_existing_trace_solution():
    top_classes = {
        "query_type": "top_object_classes_by_count",
        "expected_solution": "<light, cabinet>",
    }
    singleton_objects = {
        "query_type": "objects_in_place_symbol",
        "expected_solution": "<O14>",
    }
    singleton_points = {
        "query_type": "object_class_centers",
        "expected_solution": "<POINT(11.832 26.993 1.432)>",
    }

    assert rows_to_solution_text(
        top_classes,
        [{"class": "light", "count": 57}, {"class": "cabinet", "count": 28}],
    ) == "<light, cabinet>"
    assert rows_to_solution_text(singleton_objects, [{"object": "O14"}]) == "<O14>"
    assert rows_to_solution_text(
        singleton_points,
        [{"center": [11.832, 26.993, 1.432]}],
    ) == "<POINT(11.832 26.993 1.432)>"


def test_rows_to_solution_text_accepts_unaliased_neo4j_expression_keys():
    count_record = {"query_type": "object_class_count", "solution": 17}
    center_record = {
        "query_type": "object_class_centers",
        "solution": "POINT(1.000 2.000 3.000)",
        "solution_formats": ["set[point]", "point"],
    }

    assert rows_to_solution_text(count_record, [{"count(o)": 17}]) == "17"
    assert rows_to_solution_text(center_record, [{"o.center": [1, 2, 3]}]) == (
        "POINT(1.000 2.000 3.000)"
    )


def test_room_adjacency_count_accepts_gold_and_generated_count_aliases():
    record = {
        "query_type": "room_adjacency_count",
        "solution": 3,
        "solution_formats": ["int"],
    }

    assert rows_to_solution_text(
        record,
        [{"room": "R1", "adjacent_rooms": 3}],
    ) == "3"


def test_invalid_input_query_results_convert_to_empty_answer():
    record = {
        "query_type": "object_class_count",
        "solution": "",
        "solution_formats": ["int"],
        "tags": ["invalid_input"],
    }

    assert rows_to_solution_text(record, []) == ""
    assert rows_to_solution_text(record, [{"count": 0}]) == ""
    assert rows_to_solution_text(
        record,
        [{"class": "missing", "object_count": 0}],
    ) == ""
    assert rows_to_solution_text(
        record,
        [{"room": "R1", "neighbor_count": 3}],
    ) == "3"


def test_rows_to_solution_text_flattens_collected_point_values():
    record = {
        "query_type": "object_class_centers",
        "solution": "<POINT(1.000 2.000 3.000), POINT(4.000 5.000 6.000)>",
        "solution_formats": ["set[point]"],
    }
    rows = [
        {
            "center": [
                CartesianPoint(1, 2, 3),
                CartesianPoint(4, 5, 6),
            ]
        }
    ]

    actual = rows_to_solution_text(record, rows)

    assert normalize_answer(actual) == normalize_answer(record["solution"])


def test_rows_to_solution_text_flattens_collected_alias_values():
    class_record = {
        "query_type": "object_class_inventory",
        "solution": "<chair, table>",
        "solution_formats": ["set[str]"],
    }
    object_record = {
        "query_type": "objects_in_place_symbol",
        "solution": "<O1, O2>",
        "solution_formats": ["set[str]"],
    }

    class_solution = rows_to_solution_text(
        class_record,
        [{"classes": ["table", "chair"]}],
    )
    assert normalize_answer(class_solution) == normalize_answer(class_record["solution"])
    assert normalize_answer(rows_to_solution_text(object_record, [{"objects": ["O1", "O2"]}])) == (
        normalize_answer(object_record["solution"])
    )


def test_point_to_solution_reports_non_point_collection_as_shape_error():
    try:
        point_to_solution([CartesianPoint(1, 2, 3), CartesianPoint(4, 5, 6)])
    except ResultShapeError as exc:
        assert "expected point value" in str(exc)
    else:
        raise AssertionError("Expected ResultShapeError")


def test_rows_to_solution_text_infers_symbol_from_result_aliases():
    result_record = {
        "query_type": "nearest_object_to_mesh_place",
        "solution": "O8",
        "solution_formats": ["str"],
    }
    light_record = {
        "query_type": "nearest_object_to_object",
        "solution": "O8",
        "solution_formats": ["str"],
    }

    assert rows_to_solution_text(result_record, [{"result": "O8"}]) == "O8"
    assert rows_to_solution_text(light_record, [{"light": "O8", "dist": 4.75}]) == "O8"


def test_evaluate_records_marks_unexpected_return_shape_as_mismatch():
    class Validator:
        def validate(self, cypher, execute):
            return {
                "executable": True,
                "syntax_error": False,
                "timeout": False,
                "elapsed_ms": 1.0,
                "rows": [{"wrong_column": "O1"}],
                "error": None,
            }

    records = evaluate_records(
        records=[
            {
                "uid": "one",
                "query_type": "object_class_centers",
                "question": "Where are chairs?",
                "predicted_cypher": "MATCH (o:Object) RETURN o",
                "solution": "<POINT(1.000 2.000 3.000)>",
                "solution_formats": ["set[point]", "point"],
                "read_only": True,
            }
        ],
        cypher_field="predicted_cypher",
        allow_writes=False,
        validator=Validator(),
        policy=None,
    )

    assert records[0]["status"] == "solution_mismatch"
    assert records[0]["actual_solution"] is None
    assert records[0]["result_shape_error"] == "missing `center` in returned columns: wrong_column"


def test_build_summary_counts_solution_matches_and_errors():
    summary = build_summary(
        [
            {
                "query_type": "object_class_count",
                "static_valid": True,
                "executable": True,
                "solution_match": True,
                "status": "pass",
            },
            {
                "query_type": "object_class_count",
                "static_valid": True,
                "executable": True,
                "solution_match": False,
                "status": "solution_mismatch",
            },
        ]
    )

    assert summary["total"] == 2
    assert summary["metrics"]["solution_match"] == 1
    assert summary["metrics"]["solution_mismatch"] == 1
