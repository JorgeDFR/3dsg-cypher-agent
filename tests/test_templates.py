from sg_text2cypher.templates import generate_records


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
    assert records[0]["gold_cypher"] == 'MATCH (o:Object {class: "trash"}) RETURN 1'
    assert records[0]["scene_id"] == "scene_a"

