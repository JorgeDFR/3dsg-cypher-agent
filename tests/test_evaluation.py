from text2cypher_3dsg.evaluation import (
    build_summary,
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
