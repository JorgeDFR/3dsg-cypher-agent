from text2cypher_3dsg.evaluation import (
    build_summary,
    evaluate_records,
    normalize_answer,
    rows_to_solution_text,
)


def test_rows_to_solution_text_matches_simplified_set_solution():
    record = {
        "query_type": "object_radius_from_point",
        "solution": "<O166, O261, O498>",
    }
    rows = [{"object": "O498"}, {"object": "O166"}, {"object": "O261"}]

    actual = rows_to_solution_text(record, rows)

    assert normalize_answer(actual) == normalize_answer(record["solution"])


def test_rows_to_solution_text_uses_informative_set_of_dicts():
    record = {
        "query_type": "object_class_centers",
        "solution": "<{nodeSymbol: O1, center: POINT(1.000 2.000 3.000), class: chair}>",
    }
    rows = [
        {
            "nodeSymbol": "O1",
            "center": [1, 2, 3],
            "class": "chair",
        }
    ]

    actual = rows_to_solution_text(record, rows)

    assert actual == "<{nodeSymbol: O1, center: POINT(1.000 2.000 3.000), class: chair}>"


def test_rows_to_solution_text_accepts_unaliased_neo4j_expression_keys():
    count_record = {"query_type": "object_class_count", "solution": 17}
    center_record = {
        "query_type": "object_class_centers",
        "solution": "<POINT(1.000 2.000 3.000)>",
    }

    assert rows_to_solution_text(count_record, [{"count(o)": 17}]) == "17"
    assert rows_to_solution_text(center_record, [{"o.center": [1, 2, 3]}]) == (
        "POINT(1.000 2.000 3.000)"
    )


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
