from cypher_3dsg_agent_distillation.pddl_goals import (
    grounding_contract,
    render_pddl_solution,
    validate_grounding_rows,
)


def test_render_pddl_solution_uses_or_for_ambiguous_grounding():
    grounding = {
        "expected": {"object": ["O1", "O2"]},
    }
    goal = {
        "fact": "holding",
        "arguments": [{"result": "object"}],
        "multiple": "or",
    }

    assert render_pddl_solution(goal, grounding) == (
        "(or (holding O1) (holding O2))"
    )


def test_render_pddl_solution_supports_compound_goals():
    grounding = {"expected": {"first": ["p1"], "second": ["P2"]}}
    goal = {
        "operator": "and",
        "clauses": [
            {"fact": "visited-place", "arguments": [{"result": "first"}]},
            {"fact": "at-place", "arguments": [{"result": "second"}]},
        ],
    }

    assert render_pddl_solution(goal, grounding) == (
        "(and (visited-place p1) (at-place P2))"
    )


def test_render_pddl_solution_supports_nested_negation():
    grounding = {"expected": {"object": ["O1"]}}
    goal = {
        "operator": "and",
        "clauses": [
            {"fact": "visited-object", "arguments": [{"result": "object"}]},
            {
                "operator": "not",
                "clauses": [
                    {"fact": "holding", "arguments": [{"result": "object"}]}
                ],
            },
        ],
    }

    assert render_pddl_solution(goal, grounding) == (
        "(and (visited-object O1) (not (holding O1)))"
    )


def test_grounding_contract_and_row_validation_are_alias_aware():
    contract = grounding_contract(
        ["O2", "O1"],
        {
            "fields": {
                "object": {
                    "source": "value",
                    "aliases": ["object", "nodeSymbol"],
                    "symbol_type": "Object",
                    "cardinality": "set",
                }
            }
        },
        {},
    )

    matches, rendered, actual = validate_grounding_rows(
        [{"nodeSymbol": "O1"}, {"nodeSymbol": "O2", "distance": 1.0}],
        contract,
    )

    assert matches
    assert rendered == '{"object": ["O1", "O2"]}'
    assert actual == {"object": ["O1", "O2"]}


def test_grounding_validation_rejects_wrong_entities():
    contract = {
        "expected": {"object": ["O1"]},
        "fields": {
            "object": {
                "aliases": ["object"],
                "symbol_type": "Object",
                "cardinality": "set",
            }
        },
        "match": "exact",
    }

    assert validate_grounding_rows([{"object": "O2"}], contract)[0] is False
