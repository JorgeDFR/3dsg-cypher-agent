from collections import Counter

import pytest

from cypher_3dsg_agent_distillation.core.config import load_yaml
from cypher_3dsg_agent_distillation.cypher.repair_supervision import (
    CYPHER_SYNTAX_ERROR_TEMPLATES,
    apply_controlled_repair_sampling,
    repair_template_coverage,
    validate_repair_cypher_ratio,
)


def test_each_repair_template_mutates_an_applicable_query():
    queries = [
        "MATCH (o:Object {class: 'chair'}) RETURN o.nodeSymbol AS object",
        (
            "MATCH (o:Object)\n"
            "WITH o, point.distance(o.center, point({x: 0, y: 0, z: 0})) AS distance\n"
            "ORDER BY distance\n"
            "RETURN o.nodeSymbol AS object"
        ),
    ]

    mutations = {
        template.name: next(
            mutated
            for query in queries
            if (mutated := template.apply(query)) is not None
        )
        for template in CYPHER_SYNTAX_ERROR_TEMPLATES
    }

    assert set(mutations) == {template.name for template in CYPHER_SYNTAX_ERROR_TEMPLATES}
    assert all(mutated not in queries for mutated in mutations.values())


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_repair_ratio_must_be_between_zero_and_one(ratio):
    with pytest.raises(ValueError, match="repair_cypher_ratio must be between 0 and 1"):
        validate_repair_cypher_ratio(ratio)


def test_controlled_repair_sampling_marks_exact_ratio_and_balances_templates():
    records = [
        {
            "uid": f"record-{index}",
            "query_type": "lookup",
            "gold_cypher": (
                "MATCH (o:Object {class: 'chair'})\n"
                "WITH o, point.distance(o.center, point({x: 0, y: 0, z: 0})) AS distance\n"
                "ORDER BY distance\n"
                "RETURN o.nodeSymbol AS object"
            ),
            "tags": [],
            "metadata": {},
        }
        for index in range(20)
    ]

    sampled = apply_controlled_repair_sampling(
        records,
        ratios={"lookup": 0.2},
        seed=42,
    )
    repairs = [record for record in sampled if record["metadata"].get("controlled_repair")]

    assert len(repairs) == 4
    assert all(record["tags"] == ["controlled_repair"] for record in repairs)
    assert all(record["metadata"]["repair_cypher"] != record["gold_cypher"] for record in repairs)
    assert max(Counter(record["metadata"]["repair_template"] for record in repairs).values()) == 1


def test_controlled_repair_sampling_excludes_empty_questions():
    records = [
        {
            "uid": f"record-{index}",
            "query_type": "lookup",
            "gold_cypher": "MATCH (o:Object) RETURN o.nodeSymbol AS object",
            "solution": "<>" if index < 5 else "<O1>",
            "tags": ["invalid_input"] if index < 5 else [],
            "metadata": {},
        }
        for index in range(10)
    ]

    sampled = apply_controlled_repair_sampling(
        records,
        ratios={"lookup": 0.5},
        seed=42,
    )
    repairs = [record for record in sampled if "controlled_repair" in record["tags"]]

    assert len(repairs) == 5
    assert all("invalid_input" not in record["tags"] for record in repairs)
    assert not any(
        {"invalid_input", "controlled_repair"}.issubset(record["tags"])
        for record in sampled
    )


def test_configured_repair_templates_have_broad_question_type_coverage():
    question_templates = []
    for path in (
        "configs/question_templates/cypher_qa_3dsg.yaml",
        "configs/question_templates/cypher_pddl_3dsg.yaml",
    ):
        config = load_yaml(path)
        assert config["sampling"]["repair_cypher_ratio"] == 0.2
        question_templates.extend(config["templates"])

    write_refusal = next(
        template
        for template in question_templates
        if template["query_type"] == "write_refusal"
    )
    assert write_refusal["repair_cypher_ratio"] == 0.0

    by_template, by_question_type = repair_template_coverage(question_templates)

    assert {name: len(types) for name, types in by_template.items()} == {
        "misspelled_match_clause": 90,
        "unclosed_match_parenthesis": 90,
        "missing_property_map_value": 85,
        "undefined_return_variable": 81,
        "unknown_function": 33,
        "missing_projection_comma": 90,
        "premature_return_clause": 90,
    }
    assert len(by_question_type) == 91
    assert sum(bool(templates) for templates in by_question_type.values()) == 90
