import hashlib
from collections import Counter

import pytest

import cypher_3dsg_agent_distillation.cypher.templates as templates_module
from cypher_3dsg_agent_distillation.core.config import load_yaml
from cypher_3dsg_agent_distillation.cypher.templates import (
    SolutionBlock,
    apply_dataset_question_targets,
    dataset_question_target,
    effective_dataset_question_target,
    evenly_select_scene_records,
    format_solution_field,
    generate_records,
    normalize_solution_formats,
    stable_uid,
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
                    "cypher": 'MATCH (o:Object {class: "{class}"}) RETURN {point.x}',
                    "questions": ["Where is {class} near {point.x}?"],
                }
            ],
        }
    )
    assert records[0]["question"] == "Where is trash near 1?"
    assert records[0]["gold_cypher"].strip() == 'MATCH (o:Object {class: "trash"}) RETURN 1'
    assert records[0]["scene_id"] == "scene_a"
    assert records[0]["solution_formats"] == ["int"]


def test_generation_skips_templates_for_missing_scene_layers():
    records = generate_records(
        {
            "scene_id": "spot",
            "scene_summary": {"nodes": {"Object": 3, "MeshPlace": 5}},
            "values": {"symbols": ["O1"]},
            "templates": [
                {
                    "query_type": "room_lookup",
                    "variables": {"symbol": "symbols"},
                    "cypher": "MATCH (r:Room) RETURN r",
                    "questions": ["Find the room for {symbol}."],
                },
                {
                    "query_type": "object_lookup",
                    "variables": {"symbol": "symbols"},
                    "cypher": "MATCH (o:Object) RETURN o",
                    "questions": ["Find {symbol}."],
                },
            ],
        }
    )

    assert [record["query_type"] for record in records] == ["object_lookup"]


def test_any_layer_requirement_does_not_require_every_label_mentioned_in_cypher():
    records = generate_records(
        {
            "scene_id": "spot",
            "scene_summary": {"nodes": {"Object": 3, "MeshPlace": 5}},
            "values": {"cases": [{"object": "O1"}]},
            "templates": [
                {
                    "query_type": "parent_lookup",
                    "requires_any_layers": [["Place", "MeshPlace"]],
                    "variables": {"case": "cases"},
                    "cypher": (
                        "MATCH (p)-[:CONTAINS]->(o:Object {nodeSymbol: "
                        '"{case.object}"}) WHERE p:Place OR p:MeshPlace '
                        "RETURN p.nodeSymbol AS place"
                    ),
                    "questions": ["Where is {case.object}?"],
                }
            ],
        }
    )

    assert len(records) == 1


def test_generation_carries_scene_labelspace_and_scene_specific_uid():
    config = {
        "scene_id": "scene_a",
        "labelspace_description": "scene-specific labels",
        "values": {},
        "templates": [
            {
                "query_type": "inventory",
                "cypher": "MATCH (o:Object) RETURN o",
                "questions": ["List objects."],
            }
        ],
    }

    record = generate_records(config)[0]

    assert record["labelspace_description"] == "scene-specific labels"
    assert stable_uid("inventory", "List objects.", "MATCH", scene_id="a") != (
        stable_uid("inventory", "List objects.", "MATCH", scene_id="b")
    )


def test_dataset_question_target_is_not_multiplied_by_scene_count():
    template = {
        "query_type": "lookup",
        "target_questions": 5,
        "variables": {"value": "values"},
    }

    assert dataset_question_target(template, None) == 5
    assert dataset_question_target(template, 3) == 3


def test_challenge_quota_overrides_template_question_target():
    template = {
        "query_type": "place_hops",
        "target_questions": 20,
        "variables": {"value": "values"},
    }

    assert (
        effective_dataset_question_target(
            template,
            None,
            challenge_task_types={"place_hops"},
            challenge_questions_per_task_type=10,
        )
        == 10
    )
    assert (
        effective_dataset_question_target(
            template,
            4,
            challenge_task_types={"place_hops"},
            challenge_questions_per_task_type=10,
        )
        == 4
    )


def test_scene_record_selection_is_even_and_redistributes_unused_capacity():
    records = [{"uid": f"a{index}", "scene_id": "a"} for index in range(4)] + [
        {"uid": f"b{index}", "scene_id": "b"} for index in range(4)
    ]

    balanced = evenly_select_scene_records(records, target=5, scene_ids=["a", "b"])
    constrained = evenly_select_scene_records(
        [records[0], *records[4:]],
        target=5,
        scene_ids=["a", "b"],
    )

    assert Counter(record["scene_id"] for record in balanced) == {"a": 3, "b": 2}
    assert Counter(record["scene_id"] for record in constrained) == {"a": 1, "b": 4}


def test_scene_record_selection_uses_solution_diversity_to_break_variable_ties():
    records = [
        {
            "uid": f"record-{index}",
            "scene_id": "office",
            "gold_cypher": f"RETURN {index}",
            "solution": solution,
        }
        for index, solution in enumerate(("R0", "R0", "R0", "R0", "R1", "R2"))
    ]

    selected = evenly_select_scene_records(
        records,
        target=4,
        scene_ids=["office"],
    )

    assert len({record["gold_cypher"] for record in selected}) == 4
    assert {record["solution"] for record in selected} == {"R0", "R1", "R2"}


def test_scene_record_selection_preserves_dataset_empty_ratio():
    records = [
        {
            "uid": f"{scene}-{kind}-{index}",
            "scene_id": scene,
            "solution": "O1" if kind == "full" else "<>",
            "tags": [],
        }
        for scene, kinds in (
            ("office", (("full", 8), ("empty", 2))),
            ("spot", (("full", 4), ("empty", 6))),
        )
        for kind, count in kinds
        for index in range(count)
    ]

    selected = evenly_select_scene_records(
        records,
        target=10,
        scene_ids=["office", "spot"],
        empty_ratio=0.2,
    )

    assert Counter(record["scene_id"] for record in selected) == {
        "office": 5,
        "spot": 5,
    }
    assert sum(record["solution"] != "<>" for record in selected) == 8


def test_dataset_targets_preserve_scene_grouped_generation_order():
    records = [
        {
            "uid": f"{scene}-{query_type}-{index}",
            "scene_id": scene,
            "query_type": query_type,
        }
        for scene in ("office", "spot")
        for query_type, count in (("shared", 4), ("room_only", 3))
        for index in range(count)
        if not (scene == "spot" and query_type == "room_only")
    ]

    selected = apply_dataset_question_targets(
        records,
        targets={"shared": 4, "room_only": 3},
        scene_ids=["office", "spot"],
    )

    shared_scene_counts = Counter(
        record["scene_id"] for record in selected if record["query_type"] == "shared"
    )
    assert shared_scene_counts == {"office": 2, "spot": 2}
    assert sum(record["query_type"] == "room_only" for record in selected) == 3
    assert [record["scene_id"] for record in selected] == sorted(
        (record["scene_id"] for record in selected),
        key={"office": 0, "spot": 1}.get,
    )


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
    assert format_solution_field(None, solution_formats=["str"]) == ""
    assert format_solution_field([], solution_formats=["str"]) == ""
    assert format_solution_field(set(), solution_formats=["str"]) == ""


def test_solution_format_aliases_are_normalized():
    assert normalize_solution_formats(["set of points", "string"]) == ["set[point]", "str"]
    assert normalize_solution_formats(["set[list[str]]"]) == ["set[list[str]]"]


def test_format_solution_field_supports_sets_of_pairs():
    assert format_solution_field(
        [["O1", "O2"], ["O3", "O4"]],
        solution_formats=["set[list[str]]"],
    ) == SolutionBlock("<[O1, O2], [O3, O4]>\n")


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
            "sampling": {"empty_ratio": 0.0},
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


def test_variable_sampling_balances_solutions_before_repeating_them(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, variables):
            return {
                "O1": "R0",
                "O2": "R0",
                "O3": "R0",
                "O4": "R0",
                "O5": "R0",
                "O6": "R0",
                "O7": "R1",
                "O8": "R2",
            }[variables["object"]]

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {"objects": [f"O{index}" for index in range(1, 9)]},
        },
        question_templates_config={
            "sampling": {"seed": 7, "empty_ratio": 0.0},
            "templates": [
                {
                    "query_type": "room_lookup",
                    "target_questions": 6,
                    "solution_formats": ["str"],
                    "variables": {"object": "objects"},
                    "cypher": "RETURN '{object}' AS object",
                    "questions": ["Find {object}.", "Locate {object}."],
                }
            ],
        },
    )

    solution_counts = Counter(str(record["solution"]) for record in records)
    assert len({record["gold_cypher"] for record in records}) == 6
    assert set(solution_counts) == {"R0", "R1", "R2"}


def test_variable_sampling_balances_explicit_sampling_groups():
    cases = [
        {"object": f"O{region}{offset}", "region": f"R{region}"}
        for region in range(4)
        for offset in range(5)
    ]
    records = generate_records(
        {
            "sampling": {"empty_ratio": 0.0},
            "values": {"cases": cases},
            "templates": [
                {
                    "query_type": "room_lookup",
                    "target_questions": 12,
                    "variables": {"case": "cases"},
                    "sampling_group": "case.region",
                    "cypher": "RETURN '{case.region}' AS region, '{case.object}' AS object",
                    "questions": ["Find {case.object}.", "Locate {case.object}."],
                }
            ],
        }
    )

    region_counts = Counter(str(record["gold_cypher"]).split("'")[1] for record in records)
    assert region_counts == Counter({"R0": 3, "R1": 3, "R2": 3, "R3": 3})


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


def test_empty_ratio_is_enforced_for_each_variable_template(monkeypatch):
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
            "sampling": {"seed": 7, "empty_ratio": 0.1},
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
    assert all(
        ("invalid_input" in record["tags"]) == (str(record["solution"]).strip() == "<>")
        for record in records
    )


def test_invalid_inputs_use_empty_answers_with_template_empty_ratio(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, variables):
            return None if variables["object"].startswith("O9") else variables["object"]

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "objects": ["O1", "O2", "O3", "O4", "O90", "O91"],
                "invalid_values": ["O90", "O91"],
            },
        },
        question_templates_config={
            "sampling": {"empty_ratio": 0.1},
            "templates": [
                {
                    "query_type": "lookup",
                    "target_questions": 10,
                    "empty_ratio": 0.2,
                    "solution_formats": ["set[str]"],
                    "variables": {"object": "objects"},
                    "cypher": "RETURN '{object}' AS object",
                    "questions": ["Find {object}.", "Locate {object}."],
                }
            ],
        },
    )

    invalid_records = [record for record in records if "invalid_input" in record["tags"]]
    assert len(records) == 10
    assert len(invalid_records) == 2
    assert all(record["tags"] == ["invalid_input"] for record in invalid_records)
    assert {str(record["solution"]).strip() for record in invalid_records} == {"<>"}
    assert all("empty_result_feedback" not in record for record in invalid_records)


def test_pddl_invalid_inputs_are_retained_as_empty_answer_candidates(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, variables):
            return None if variables["object"].startswith("O9") else variables["object"]

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "objects": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O90", "O91", "O92"],
                "invalid_values": ["O90", "O91", "O92"],
            },
        },
        question_templates_config={
            "sampling": {"empty_ratio": 0.3},
            "templates": [
                {
                    "query_type": "pddl_visit_object_symbol",
                    "target_questions": 10,
                    "variables": {"object": "objects"},
                    "cypher": (
                        "MATCH (o:Object {nodeSymbol: '{object}'}) RETURN o.nodeSymbol AS object"
                    ),
                    "grounding": {
                        "fields": {
                            "object": {
                                "source": "value",
                                "aliases": ["object"],
                                "symbol_type": "Object",
                            }
                        }
                    },
                    "goal": {
                        "fact": "visited-object",
                        "arguments": [{"result": "object"}],
                    },
                    "questions": ["Visit {object}."],
                }
            ],
        },
        task_family="cypher_pddl",
        output_type="PDDL",
    )

    invalid_records = [record for record in records if "invalid_input" in record["tags"]]
    assert len(records) == 10
    assert len(invalid_records) == 3
    assert all(record["tags"] == ["invalid_input"] for record in invalid_records)
    assert all(record["solution"] == "" for record in invalid_records)
    assert all(record["grounding"]["expected"] == {"object": []} for record in invalid_records)
    assert all("empty_result_feedback" not in record for record in invalid_records)


def test_pddl_partial_invalid_grounding_is_fully_empty(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, _variables):
            return {"first": ["O1"], "second": []}

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "first_classes": ["chair"],
                "second_classes": ["missing"],
                "invalid_values": ["missing"],
            },
        },
        question_templates_config={
            "templates": [
                {
                    "query_type": "pddl_pair",
                    "target_questions": 1,
                    "empty_ratio": 1.0,
                    "variables": {
                        "first_class": "first_classes",
                        "second_class": "second_classes",
                    },
                    "cypher": "RETURN '{first_class}', '{second_class}'",
                    "grounding": {
                        "fields": {
                            "first": {"source": "first", "cardinality": "set"},
                            "second": {"source": "second", "cardinality": "set"},
                        }
                    },
                    "goal": {
                        "operator": "and",
                        "clauses": [
                            {"fact": "safe", "arguments": [{"result": "first"}]},
                            {"fact": "safe", "arguments": [{"result": "second"}]},
                        ],
                    },
                    "questions": ["Use {first_class} and {second_class}."],
                }
            ]
        },
        task_family="cypher_pddl",
        output_type="PDDL",
    )

    assert records[0]["solution"] == ""
    assert records[0]["grounding"]["expected"] == {"first": [], "second": []}
    assert "invalid_input" in records[0]["tags"]


def test_qa_nonempty_result_with_absent_filter_value_is_not_invalid(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, _variables):
            return "O2"

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "objects": ["O1"],
                "excluded_classes": ["missing"],
                "invalid_values": ["missing"],
            },
        },
        question_templates_config={
            "templates": [
                {
                    "query_type": "nearest_excluding",
                    "target_questions": 1,
                    "solution_formats": ["str"],
                    "variables": {
                        "object": "objects",
                        "excluded_class": "excluded_classes",
                    },
                    "cypher": "RETURN 'O2' AS object",
                    "questions": ["Nearest to {object}, excluding {excluded_class}?"],
                }
            ]
        },
        task_family="cypher_qa",
    )

    assert records[0]["solution"] == "O2"
    assert "invalid_input" not in records[0]["tags"]


def test_qa_zero_count_with_absent_filter_value_is_sampled_as_empty(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, variables):
            return 0 if variables["class"] == "missing" else 1

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "classes": ["chair", "desk", "lamp", "table", "missing"],
                "invalid_values": ["missing"],
            },
        },
        question_templates_config={
            "sampling": {"seed": 7, "empty_ratio": 0.2},
            "templates": [
                {
                    "query_type": "object_class_count",
                    "target_questions": 10,
                    "solution_formats": ["int"],
                    "variables": {"class": "classes"},
                    "cypher": "RETURN count('{class}') AS count",
                    "questions": ["Count {class}.", "How many {class}?"],
                }
            ],
        },
    )

    empty_records = [record for record in records if "invalid_input" in record["tags"]]
    assert len(records) == 10
    assert len(empty_records) == 2
    assert all(record["solution"] == 0 for record in empty_records)
    assert {"missing"} == {
        record["question"].removeprefix("Count ").removeprefix("How many ").rstrip(".?")
        for record in empty_records
    }


def test_qa_zero_count_with_valid_input_is_empty_but_not_invalid(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, variables):
            return 0 if variables["class"] == "empty" else 1

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {"classes": ["chair", "empty"]},
        },
        question_templates_config={
            "sampling": {"seed": 7, "empty_ratio": 0.5},
            "templates": [
                {
                    "query_type": "object_class_count",
                    "target_questions": 2,
                    "solution_formats": ["int"],
                    "variables": {"class": "classes"},
                    "cypher": "RETURN count('{class}') AS count",
                    "questions": ["Count {class}."],
                }
            ],
        },
    )

    zero_record = next(record for record in records if record["solution"] == 0)
    assert templates_module.generated_record_has_solution(zero_record) is False
    assert "invalid_input" not in zero_record["tags"]


def test_generation_bounds_solution_size_and_cartesian_result_rows(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, variables):
            if variables["case"] == "large_solution":
                return {"first": [f"O{i}" for i in range(11)], "second": ["O20"]}
            if variables["case"] == "large_product":
                return {
                    "first": [f"O{i}" for i in range(5)],
                    "second": [f"O{i}" for i in range(10, 15)],
                }
            return {"first": ["O1", "O2"], "second": ["O3", "O4"]}

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {"cases": ["large_solution", "large_product", "bounded"]},
        },
        question_templates_config={
            "sampling": {
                "empty_ratio": 0.0,
                "max_solution_size": 10,
                "max_result_rows": 20,
            },
            "templates": [
                {
                    "query_type": "pddl_pair",
                    "target_questions": 1,
                    "variables": {"case": "cases"},
                    "cypher": "RETURN '{case}'",
                    "grounding": {
                        "fields": {
                            "first": {"source": "first", "cardinality": "set"},
                            "second": {"source": "second", "cardinality": "set"},
                        }
                    },
                    "goal": {
                        "operator": "and",
                        "clauses": [
                            {"fact": "safe", "arguments": [{"result": "first"}]},
                            {"fact": "safe", "arguments": [{"result": "second"}]},
                        ],
                    },
                    "questions": ["Use {case}."],
                }
            ],
        },
        task_family="cypher_pddl",
        output_type="PDDL",
    )

    assert [record["question"] for record in records] == ["Use bounded."]


@pytest.mark.parametrize("key", ["max_solution_size", "max_result_rows"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_generation_rejects_invalid_sampling_size_limits(key, value):
    with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
        generate_records(
            {"values": {"values": ["one"]}},
            question_templates_config={
                "sampling": {key: value},
                "templates": [
                    {
                        "query_type": "lookup",
                        "variables": {"value": "values"},
                        "cypher": "RETURN '{value}'",
                        "questions": ["Find {value}."],
                    }
                ],
            },
        )


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
            "sampling": {"seed": 7, "empty_ratio": 0.5},
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


def test_pddl_template_generates_goal_and_grounding_contract(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, _variables):
            return ["O2", "O1"]

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {"source_3dsg": "unused.json", "values": {"classes": ["chair"]}},
        question_templates_config={
            "templates": [
                {
                    "query_type": "pddl_hold_object_class",
                    "target_questions": 1,
                    "variables": {"class": "classes"},
                    "cypher": "RETURN '{class}' AS object",
                    "grounding": {
                        "fields": {
                            "object": {
                                "source": "value",
                                "aliases": ["object"],
                                "symbol_type": "Object",
                                "cardinality": "set",
                            }
                        }
                    },
                    "goal": {
                        "fact": "holding",
                        "arguments": [{"result": "object"}],
                        "multiple": "or",
                    },
                    "questions": ["Pick up a {class}."],
                }
            ]
        },
        task_family="cypher_pddl",
        output_type="PDDL",
        experiment_profile="cypher_pddl_template",
        question_set="pddl",
    )

    assert records[0]["solution"] == "(or (holding O1) (holding O2))"
    assert records[0]["grounding"]["expected"] == {"object": ["O1", "O2"]}
    assert records[0]["task_family"] == "cypher_pddl"


def test_pddl_disjunction_preserves_available_branch(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, _variables):
            return {"first": ["O1", "O2"], "second": []}

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "first_classes": ["light"],
                "second_classes": ["missing"],
                "invalid_values": ["missing"],
            },
        },
        question_templates_config={
            "sampling": {"empty_ratio": 0.0},
            "templates": [
                {
                    "query_type": "pddl_atomic_alternative",
                    "target_questions": 1,
                    "variables": {
                        "first_class": "first_classes",
                        "second_class": "second_classes",
                    },
                    "cypher": "RETURN '{first_class}', '{second_class}'",
                    "grounding": {
                        "fields": {
                            "first": {
                                "source": "first",
                                "aliases": ["first"],
                                "symbol_type": "Object",
                                "cardinality": "set",
                            },
                            "second": {
                                "source": "second",
                                "aliases": ["second"],
                                "symbol_type": "Object",
                                "cardinality": "set",
                            },
                        }
                    },
                    "goal": {
                        "operator": "or",
                        "clauses": [
                            {
                                "fact": "safe",
                                "arguments": [{"result": "first"}],
                                "multiple": "or",
                            },
                            {
                                "fact": "visited-object",
                                "arguments": [{"result": "second"}],
                                "multiple": "or",
                            },
                        ],
                    },
                    "questions": ["Inspect {first_class} or visit {second_class}."],
                }
            ],
        },
        task_family="cypher_pddl",
        output_type="PDDL",
    )

    assert records[0]["solution"] == "(or (safe O1) (safe O2))"
    assert records[0]["grounding"]["expected"] == {
        "first": ["O1", "O2"],
        "second": [],
    }
    assert templates_module.generated_record_has_solution(records[0]) is True


def test_valid_empty_scalar_solution_uses_empty_answer(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, _query_type, _variables):
            return None

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    records = generate_records(
        {"source_3dsg": "unused.json", "values": {"classes": ["bed"]}},
        question_templates_config={
            "sampling": {"empty_ratio": 1.0},
            "templates": [
                {
                    "query_type": "nearest_object_between_classes",
                    "target_questions": 1,
                    "solution_formats": ["str"],
                    "variables": {"class": "classes"},
                    "cypher": "RETURN '{class}'",
                    "questions": ["Find {class}."],
                }
            ],
        },
        task_family="cypher_qa",
        output_type="SLDP",
    )

    assert records[0]["solution"] == ""


def test_pddl_alternative_queries_preserve_only_satisfiable_branches():
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]
    by_query_type = {template["query_type"]: template for template in templates}

    atomic = by_query_type["pddl_atomic_alternative"]["cypher"]
    assert atomic.count("OPTIONAL MATCH") == 2

    compound = by_query_type["pddl_compound_alternative"]["cypher"]
    assert "OPTIONAL MATCH (first:Object" in compound
    assert "MATCH (second:Object" in compound
    assert "OPTIONAL MATCH (second:Object" not in compound


def test_qa_stable_uid_remains_backward_compatible():
    query_type = "object_count"
    question = "How many chairs?"
    cypher = "MATCH (o:Object) RETURN count(o) AS count"
    expected = hashlib.sha1(f"{query_type}\n{question}\n{cypher}".encode()).hexdigest()

    assert templates_module.stable_uid(query_type, question, cypher) == (f"tpl_{expected[:12]}")


def test_pddl_template_without_scene_grounding_does_not_emit_invalid_goal():
    records = generate_records(
        {"values": {"objects": ["O1"]}},
        question_templates_config={
            "templates": [
                {
                    "query_type": "pddl_visit_object_symbol",
                    "variables": {"object": "objects"},
                    "cypher": "RETURN '{object}' AS object",
                    "grounding": {"fields": {"object": {"source": "value", "aliases": ["object"]}}},
                    "goal": {
                        "fact": "visited-object",
                        "arguments": [{"result": "object"}],
                    },
                    "questions": ["Visit {object}."],
                }
            ]
        },
        task_family="cypher_pddl",
        output_type="PDDL",
    )

    assert records == []


def test_pddl_template_rejects_missing_grounding_contract():
    with pytest.raises(KeyError, match="must define grounding fields"):
        generate_records(
            {"values": {}},
            question_templates_config={
                "templates": [
                    {
                        "query_type": "pddl_invalid",
                        "cypher": "RETURN 'O1' AS object",
                        "goal": {
                            "fact": "visited-object",
                            "arguments": [{"result": "object"}],
                        },
                        "questions": ["Visit the object."],
                    }
                ]
            },
            task_family="cypher_pddl",
            output_type="PDDL",
        )


def test_pddl_templates_have_multiple_distinct_instruction_variants():
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]

    assert all(len(template["questions"]) >= 5 for template in templates)
    assert all(
        len(template["questions"]) == len(set(template["questions"])) for template in templates
    )


def test_qa_templates_have_at_least_five_distinct_question_variants():
    templates = load_yaml("configs/question_templates/cypher_qa_3dsg.yaml")["templates"]

    assert all(len(template["questions"]) >= 5 for template in templates)
    assert all(
        len(template["questions"]) == len(set(template["questions"])) for template in templates
    )


def test_qa_templates_disable_empty_sampling_for_deterministic_and_count_queries():
    templates = load_yaml("configs/question_templates/cypher_qa_3dsg.yaml")["templates"]

    zero_empty_ratio_types = {
        template["query_type"] for template in templates if template.get("empty_ratio") == 0.0
    }

    assert zero_empty_ratio_types == {
        "object_class_inventory",
        "object_class_count",
        "top_object_classes_by_count",
        "objects_in_bounding_box",
        "room_adjacency_count",
        "object_total",
        "room_total",
        "combined_place_total",
        "room_object_count",
        "most_connected_rooms",
        "least_connected_rooms",
        "count_objects_of_class_in_room_class",
        "place_object_count",
        "closest_cross_class_object_pairs",
        "closest_room_pairs",
        "write_refusal",
    }


def test_qa_templates_return_only_fields_required_by_the_answer():
    templates = load_yaml("configs/question_templates/cypher_qa_3dsg.yaml")["templates"]
    final_returns = {
        template["query_type"]: [
            line.strip()
            for line in template["cypher"].splitlines()
            if line.strip().startswith("RETURN ")
        ][-1]
        for template in templates
        if template["query_type"] != "write_refusal"
    }

    assert final_returns == {
        "object_class_inventory": "RETURN DISTINCT o.class AS class",
        "object_class_centers": "RETURN o.center AS center",
        "object_class_count": "RETURN count(o) AS count",
        "top_object_classes_by_count": "RETURN class",
        "objects_in_room_symbol": "RETURN o.nodeSymbol AS object",
        "objects_in_place_symbol": "RETURN o.nodeSymbol AS object",
        "object_place_lookup": "RETURN p.nodeSymbol AS place",
        "nearest_object_to_mesh_place": "RETURN o.nodeSymbol AS object",
        "object_radius_from_point": "RETURN o.nodeSymbol AS object",
        "objects_in_bounding_box": "RETURN o.nodeSymbol AS object",
        "mesh_place_hops": "RETURN DISTINCT mp.nodeSymbol AS meshPlace",
        "room_adjacency_count": "RETURN count(DISTINCT neighbor) AS adjacent_rooms",
        "nearest_object_to_object": "RETURN target.nodeSymbol AS object",
        "place_hops": "RETURN DISTINCT p.nodeSymbol AS place",
        "location_hops": "RETURN DISTINCT location.nodeSymbol AS location",
        "objects_in_room_class": "RETURN DISTINCT o.nodeSymbol AS object",
        "object_total": "RETURN count(o) AS count",
        "room_total": "RETURN count(r) AS count",
        "combined_place_total": "RETURN count(p) AS count",
        "object_class_by_symbol": "RETURN o.class AS class",
        "object_center_by_symbol": "RETURN o.center AS center",
        "room_object_count": "RETURN count(DISTINCT o) AS count",
        "object_classes_in_room_symbol": "RETURN DISTINCT o.class AS class",
        "adjacent_rooms": "RETURN DISTINCT neighbor.nodeSymbol AS room",
        "most_connected_rooms": "RETURN row.symbol AS room",
        "least_connected_rooms": "RETURN row.symbol AS room",
        "nearest_object_any_class": "RETURN target.nodeSymbol AS object",
        "farthest_object_from_object": "RETURN target.nodeSymbol AS object",
        "nearest_object_different_class": "RETURN target.nodeSymbol AS object",
        "objects_within_radius_of_object": "RETURN target.nodeSymbol AS object",
        "objects_of_class_in_room_symbol": "RETURN DISTINCT o.nodeSymbol AS object",
        "objects_of_class_in_room_class": "RETURN DISTINCT o.nodeSymbol AS object",
        "count_objects_of_class_in_room_class": "RETURN count(DISTINCT o) AS count",
        "rooms_containing_object_class": "RETURN DISTINCT r.nodeSymbol AS room",
        "object_classes_in_room_class": "RETURN DISTINCT o.class AS class",
        "nearest_object_same_class": "RETURN target.nodeSymbol AS object",
        "nearest_object_excluding_class": "RETURN target.nodeSymbol AS object",
        "nearest_object_in_class_set": "RETURN target.nodeSymbol AS object",
        "nearest_object_between_classes": "RETURN target.nodeSymbol AS object",
        "farthest_object_of_class": "RETURN target.nodeSymbol AS object",
        "objects_of_class_within_radius_of_object": "RETURN DISTINCT target.nodeSymbol AS object",
        "objects_of_class_near_class": "RETURN DISTINCT target.nodeSymbol AS object",
        "adjacent_rooms_of_class": "RETURN DISTINCT neighbor.nodeSymbol AS room",
        "objects_of_class_in_place": "RETURN DISTINCT o.nodeSymbol AS object",
        "objects_excluding_class_in_room_class": "RETURN DISTINCT o.nodeSymbol AS object",
        "place_object_count": "RETURN count(DISTINCT o) AS count",
        "object_classes_in_place_symbol": "RETURN DISTINCT o.class AS class",
        "closest_cross_class_object_pairs": "RETURN row.first AS first, row.second AS second",
        "closest_room_pairs": "RETURN row.first AS first, row.second AS second",
        "nearest_place_to_object": "RETURN p.nodeSymbol AS place",
        "nearest_room_to_object": "RETURN r.nodeSymbol AS room",
    }


def test_object_containment_queries_support_both_parent_layer_labels():
    templates = load_yaml("configs/question_templates/cypher_qa_3dsg.yaml")["templates"]
    by_query_type = {template["query_type"]: template for template in templates}

    for query_type in ("objects_in_place_symbol", "object_place_lookup"):
        cypher = by_query_type[query_type]["cypher"]
        assert "WHERE p:Place OR p:MeshPlace" in cypher
        assert "{case.place_label}" not in cypher


def test_place_and_mesh_place_templates_include_symbol_only_contrast_pairs():
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]
    by_query_type = {template["query_type"]: template for template in templates}

    pairs = (
        ("pddl_visit_place_symbol", "pddl_visit_mesh_place_symbol"),
        ("pddl_end_at_place_symbol", "pddl_end_at_mesh_place_symbol"),
    )
    for place_query_type, mesh_query_type in pairs:
        place_questions = {
            question.replace("{place}", "{symbol}")
            for question in by_query_type[place_query_type]["questions"]
        }
        mesh_questions = {
            question.replace("{mesh_place}", "{symbol}")
            for question in by_query_type[mesh_query_type]["questions"]
        }

        assert len(place_questions & mesh_questions) >= 2

    qa_templates = load_yaml("configs/question_templates/cypher_qa_3dsg.yaml")["templates"]
    qa_by_query_type = {template["query_type"]: template for template in qa_templates}
    place_hop_questions = {
        question.replace("{case.place}", "{case.symbol}")
        for question in qa_by_query_type["place_hops"]["questions"]
    }
    mesh_hop_questions = {
        question.replace("{case.mesh_place}", "{case.symbol}")
        for question in qa_by_query_type["mesh_place_hops"]["questions"]
    }

    assert len(place_hop_questions & mesh_hop_questions) >= 2


def test_location_hops_composes_both_place_layers():
    templates = load_yaml("configs/question_templates/cypher_qa_3dsg.yaml")["templates"]
    template = next(template for template in templates if template["query_type"] == "location_hops")

    assert template["variables"] == {"case": "location_hop_cases"}
    assert template["requires_layers"] == ["Place", "MeshPlace"]
    assert template["sampling_group"] == "case.layer"
    assert ":{case.relationship}*1..{case.hops}" in template["cypher"]


def test_pddl_room_templates_cover_symbols_classes_and_object_containment(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, query_type, _variables):
            if query_type.endswith("_room_class"):
                return ["R2", "R1"]
            if query_type.endswith("_room_containing_object"):
                return ["R3"]
            return "R0"

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]
    room_query_types = {
        "pddl_visit_room_symbol",
        "pddl_end_in_room_symbol",
        "pddl_visit_room_class",
        "pddl_end_in_room_class",
        "pddl_visit_room_containing_object",
        "pddl_end_in_room_containing_object",
    }
    room_templates = [
        template for template in templates if template["query_type"] in room_query_types
    ]
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "room_symbols": ["R0"],
                "room_classes": ["office"],
                "room_containing_object_cases": [{"object": "O1", "region": "R3"}],
            },
        },
        question_templates_config={"templates": room_templates},
        limit_per_template=1,
        task_family="cypher_pddl",
        output_type="PDDL",
    )

    goals = {record["query_type"]: record["solution"] for record in records}
    assert goals == {
        "pddl_visit_room_symbol": "(visited-region R0)",
        "pddl_end_in_room_symbol": "(in-region R0)",
        "pddl_visit_room_class": "(or (visited-region R1) (visited-region R2))",
        "pddl_end_in_room_class": "(or (in-region R1) (in-region R2))",
        "pddl_visit_room_containing_object": "(visited-region R3)",
        "pddl_end_in_room_containing_object": "(in-region R3)",
    }


def test_all_pddl_room_groundings_accept_region_result_aliases():
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]
    room_fields = [
        field
        for template in templates
        for field in (template.get("grounding", {}).get("fields", {}).values())
        if field.get("symbol_type") == "Room"
    ]

    assert room_fields
    assert all("region" in field.get("aliases", []) for field in room_fields)


def test_pddl_at_templates_cover_objects_places_and_spatial_references(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, query_type, _variables):
            if query_type == "pddl_end_at_object_class":
                return ["O2", "O1"]
            if query_type == "pddl_end_at_nearest_object":
                return "O3"
            if query_type == "pddl_end_at_place_symbol":
                return "p4"
            if query_type == "pddl_end_at_mesh_place_symbol":
                return "P5"
            return "O0"

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]
    at_query_types = {
        "pddl_end_at_object_symbol",
        "pddl_end_at_object_class",
        "pddl_end_at_nearest_object",
        "pddl_end_at_place_symbol",
        "pddl_end_at_mesh_place_symbol",
    }
    at_templates = [template for template in templates if template["query_type"] in at_query_types]
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "object_symbols": ["O0"],
                "object_classes": ["chair"],
                "nearest_object_cases": [{"object": "O8", "class": "chair"}],
                "place_symbols": ["p4"],
                "mesh_place_symbols": ["P5"],
            },
        },
        question_templates_config={"templates": at_templates},
        limit_per_template=1,
        task_family="cypher_pddl",
        output_type="PDDL",
    )

    assert {record["query_type"]: record["solution"] for record in records} == {
        "pddl_end_at_object_symbol": "(at-object O0)",
        "pddl_end_at_object_class": "(or (at-object O1) (at-object O2))",
        "pddl_end_at_nearest_object": "(at-object O3)",
        "pddl_end_at_place_symbol": "(at-place p4)",
        "pddl_end_at_mesh_place_symbol": "(at-place P5)",
    }


def test_pddl_navigation_template_tags_distinguish_waypoints_from_destinations():
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]

    for template in templates:
        fact = template["goal"].get("fact")
        tags = set(template.get("tags", []))
        if fact in {"visited-object", "visited-place", "visited-region"}:
            assert "waypoint" in tags, template["query_type"]
            assert "destination" not in tags, template["query_type"]
        elif fact in {"at-object", "at-place", "in-region"}:
            assert "destination" in tags, template["query_type"]
            assert "waypoint" not in tags, template["query_type"]


def test_pddl_compound_templates_generate_and_not_goals(monkeypatch):
    class FakeSceneIndex:
        def __init__(self, _path):
            pass

        def solution_for(self, query_type, _variables):
            if query_type == "pddl_inspect_object_and_end_at_place":
                return {"object": "O0", "place": "p4"}
            if query_type == "pddl_visit_object_and_end_at_object":
                return {"waypoint": "O1", "destination": "O2"}
            if query_type == "pddl_visit_place_and_end_in_room":
                return {"place": "p4", "room": "R1"}
            if query_type == "pddl_not_holding_object_class":
                return ["O2", "O1"]
            if query_type == "pddl_object_not_in_place":
                return {"object": "O0", "place": "p4"}
            return "O0"

    monkeypatch.setattr(templates_module, "SceneIndex", FakeSceneIndex)
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]
    compound_query_types = {
        "pddl_inspect_object_and_end_at_place",
        "pddl_visit_object_and_end_at_object",
        "pddl_visit_place_and_end_in_room",
        "pddl_not_holding_object_symbol",
        "pddl_not_holding_object_class",
        "pddl_object_not_in_place",
        "pddl_visit_object_without_holding",
    }
    compound_templates = [
        template for template in templates if template["query_type"] in compound_query_types
    ]
    records = generate_records(
        {
            "source_3dsg": "unused.json",
            "values": {
                "object_symbols": ["O0", "O1", "O2"],
                "object_classes": ["chair"],
                "place_symbols": ["p4"],
                "place_or_mesh_place_symbols": ["p4"],
                "room_symbols": ["R1"],
            },
        },
        question_templates_config={"templates": compound_templates},
        limit_per_template=1,
        task_family="cypher_pddl",
        output_type="PDDL",
    )

    assert {record["query_type"]: record["solution"] for record in records} == {
        "pddl_inspect_object_and_end_at_place": ("(and (safe O0) (at-place p4))"),
        "pddl_visit_object_and_end_at_object": ("(and (visited-object O1) (at-object O2))"),
        "pddl_visit_place_and_end_in_room": ("(and (visited-place p4) (in-region R1))"),
        "pddl_not_holding_object_symbol": "(not (holding O0))",
        "pddl_not_holding_object_class": ("(not (or (holding O1) (holding O2)))"),
        "pddl_object_not_in_place": "(not (object-in-place O0 p4))",
        "pddl_visit_object_without_holding": ("(and (visited-object O0) (not (holding O0)))"),
    }


def test_pddl_object_in_place_templates_accept_both_place_layers():
    templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]
    by_query_type = {template["query_type"]: template for template in templates}

    for query_type in ("pddl_move_object_to_place", "pddl_object_not_in_place"):
        template = by_query_type[query_type]
        assert template["variables"]["place"] == "place_or_mesh_place_symbols"
        assert template["requires_any_layers"] == [["Place", "MeshPlace"]]
        assert "WHERE p:Place OR p:MeshPlace" in template["cypher"]
        assert template["grounding"]["fields"]["place"]["symbol_type"] == ("PlaceOrMeshPlace")


def test_room_object_queries_follow_the_loaded_two_edge_hierarchy():
    qa_templates = load_yaml("configs/question_templates/cypher_qa_3dsg.yaml")["templates"]
    pddl_templates = load_yaml("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"]
    affected_query_types = {
        "objects_in_room_symbol",
        "objects_in_room_class",
        "pddl_visit_room_containing_object",
        "pddl_end_in_room_containing_object",
    }
    affected_templates = [
        template
        for template in [*qa_templates, *pddl_templates]
        if template["query_type"] in affected_query_types
    ]

    assert {template["query_type"] for template in affected_templates} == affected_query_types
    assert all(
        "(r:Room" in template["cypher"] and "[:CONTAINS*2..2]->(o:Object" in template["cypher"]
        for template in affected_templates
    )
