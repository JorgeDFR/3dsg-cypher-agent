import pytest

import cypher_3dsg_agent_distillation.templates as templates_module
from cypher_3dsg_agent_distillation.config import load_yaml
from cypher_3dsg_agent_distillation.templates import (
    SolutionBlock,
    format_solution_field,
    generate_records,
    normalize_solution_formats,
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
                    "solution_formats": ["int"],
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
    assert records[0]["solution_formats"] == ["int"]


def test_format_solution_field_uses_answer_notation():
    assert format_solution_field(3) == 3
    assert format_solution_field("O12") == "O12"
    assert format_solution_field({"O1", "O2"}, solution_formats=["set[str]"]) == SolutionBlock(
        "<O1, O2>\n"
    )
    assert format_solution_field(["light", "cabinet"], solution_formats=["set[str]"]) == (
        SolutionBlock("<light, cabinet>\n")
    )
    assert format_solution_field("chair", solution_formats=["set[str]"]) == SolutionBlock(
        "<chair>\n"
    )
    assert format_solution_field("O12", solution_formats=["set[str]", "str"]) == "O12"


def test_solution_format_aliases_are_normalized():
    assert normalize_solution_formats(["set of points", "string"]) == ["set[point]", "str"]


def test_variable_template_samples_target_number_of_unique_questions():
    template = {
        "query_type": "lookup",
        "target_questions": 5,
        "variables": {"class": "classes"},
        "cypher": 'MATCH (o {class: "{class}"}) RETURN o',
        "questions": ["Where is {class}?", "Find {class}."],
    }

    records = generate_records(
        {"values": {"classes": ["chair", "desk", "lamp", "rug"]}, "templates": [template]}
    )

    assert len(records) == 5
    assert len({record["question"] for record in records}) == 5
    assert records == generate_records(
        {"values": {"classes": ["chair", "desk", "lamp", "rug"]}, "templates": [template]}
    )


def test_variable_sampling_uses_distinct_variables_before_wording_reuse():
    values = [f"value-{index}" for index in range(10)]
    records = generate_records(
        {
            "values": {"values": values},
            "templates": [
                {
                    "query_type": "lookup",
                    "target_questions": 10,
                    "variables": {"value": "values"},
                    "cypher": "RETURN '{value}'",
                    "questions": ["Find {value}.", "Locate {value}.", "Show {value}."],
                }
            ],
        }
    )

    assert len({record["gold_cypher"] for record in records}) == 10


def test_variable_template_warns_when_target_exceeds_variant_capacity():
    with pytest.warns(UserWarning, match="can produce at most 2"):
        records = generate_records(
            {
                "values": {"classes": ["chair"]},
                "templates": [
                    {
                        "query_type": "lookup",
                        "target_questions": 10,
                        "variables": {"class": "classes"},
                        "cypher": "RETURN '{class}'",
                        "questions": ["Where is {class}?", "Find {class}."],
                    }
                ],
            }
        )

    assert len(records) == 2


def test_non_empty_ratio_is_enforced_for_each_variable_template(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, variables):
            return set() if variables["value"].startswith("empty") else {variables["value"]}

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    values = [*(f"full{i}" for i in range(5)), *(f"empty{i}" for i in range(5))]
    records = generate_records(
        {"source_3dsg": "unused.json", "values": {"values": values}},
        question_templates_config={
            "sampling": {"seed": 7, "non_empty_ratio": 0.9},
            "templates": [
                {
                    "query_type": "lookup",
                    "target_questions": 10,
                    "solution_formats": ["set[str]"],
                    "variables": {"value": "values"},
                    "cypher": "RETURN '{value}'",
                    "questions": ["Find {value}.", "Locate {value}."],
                }
            ],
        },
    )

    assert len(records) == 10
    assert sum(str(record["solution"]).strip() != "<>" for record in records) >= 9


def test_empty_sampling_prioritizes_distinct_variables(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, variables):
            return set() if variables["value"].startswith("empty") else {variables["value"]}

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    values = [*(f"full{i}" for i in range(10)), *(f"empty{i}" for i in range(10))]
    records = generate_records(
        {"source_3dsg": "unused.json", "values": {"values": values}},
        question_templates_config={
            "sampling": {"seed": 7, "non_empty_ratio": 0.5},
            "templates": [
                {
                    "query_type": "lookup",
                    "target_questions": 20,
                    "solution_formats": ["set[str]"],
                    "variables": {"value": "values"},
                    "cypher": "RETURN '{value}'",
                    "questions": ["Find {value}.", "Locate {value}."],
                }
            ],
        },
    )

    empty_records = [record for record in records if str(record["solution"]).strip() == "<>"]
    assert len(empty_records) == 10
    assert len({record["gold_cypher"] for record in empty_records}) == 10


def test_no_variable_template_keeps_each_wording_variant():
    records = generate_records(
        {
            "values": {},
            "templates": [
                {
                    "query_type": "inventory",
                    "target_questions": 50,
                    "cypher": "MATCH (n) RETURN n",
                    "questions": ["What exists?", "List everything."],
                }
            ],
        }
    )

    assert [record["question"] for record in records] == ["What exists?", "List everything."]


def test_write_refusal_template_has_ten_unique_question_variants():
    templates_config = load_yaml("configs/question_templates/cypher_qa_3dsg.yaml")
    write_refusal = next(
        template
        for template in templates_config["templates"]
        if template["query_type"] == "write_refusal"
    )

    records = generate_records({"values": {}, "templates": [write_refusal]})

    assert len(records) == 10
    assert len({record["question"] for record in records}) == 10
    assert {record["query_type"] for record in records} == {"write_refusal"}
