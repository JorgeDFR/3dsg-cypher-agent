from cypher_3dsg_agent_distillation.templates import (
    SolutionBlock,
    format_solution_field,
    generate_records,
)


def test_generate_records_expands_nested_values():
    records = generate_records(
        {
            "scene_id": "scene_a",
            "values": {
                "classes": ["trash"],
                "points": [{"x": 1, "y": 2, "z": 3}],
            },
            "templates": [
                {
                    "query_type": "radius",
                    "variables": {"class": "classes", "point": "points"},
                    "cypher": "MATCH (o:Object {class: \"{class}\"}) RETURN {point.x}",
                    "questions": ["Where is {class} near {point.x}?"],
                }
            ],
        }
    )
    assert records[0]["question"] == "Where is trash near 1?"
    assert records[0]["gold_cypher"].strip() == 'MATCH (o:Object {class: "trash"}) RETURN 1'
    assert records[0]["scene_id"] == "scene_a"


def test_format_solution_field_uses_answer_notation():
    assert format_solution_field(3) == 3
    assert format_solution_field("O12") == "O12"
    assert format_solution_field({"O1", "O2"}) == SolutionBlock("<O1, O2>\n")
    assert format_solution_field([{"object": "O1"}], list_style="set") == SolutionBlock(
        "<{object: O1}>\n"
    )
    assert format_solution_field({"object": "O1", "distance": 1.25}) == SolutionBlock(
        "{object: O1, distance: 1.25}\n"
    )
