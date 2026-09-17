from cypher_3dsg_agent_distillation.cypher.pddl_goals import (
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

    assert render_pddl_solution(goal, grounding) == ("(or (holding O1) (holding O2))")


def test_render_pddl_solution_supports_compound_goals():
    grounding = {"expected": {"first": ["p1"], "second": ["P2"]}}
    goal = {
        "operator": "and",
        "clauses": [
            {"fact": "visited-place", "arguments": [{"result": "first"}]},
            {"fact": "at-place", "arguments": [{"result": "second"}]},
        ],
    }

    assert render_pddl_solution(goal, grounding) == ("(and (visited-place p1) (at-place P2))")


def test_render_pddl_solution_skips_only_unavailable_disjunction_branches():
    grounding = {"expected": {"first": ["O1", "O2"], "second": []}}
    goal = {
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
    }

    assert render_pddl_solution(goal, grounding) == "(or (safe O1) (safe O2))"


def test_render_pddl_solution_rejects_partial_conjunction():
    grounding = {"expected": {"first": ["O1"], "second": []}}
    goal = {
        "operator": "and",
        "clauses": [
            {"fact": "safe", "arguments": [{"result": "first"}]},
            {"fact": "visited-object", "arguments": [{"result": "second"}]},
        ],
    }

    assert render_pddl_solution(goal, grounding) is None


def test_render_pddl_solution_keeps_available_nested_or_branch():
    grounding = {"expected": {"first": [], "second": ["O3"]}}
    goal = {
        "operator": "or",
        "clauses": [
            {
                "operator": "and",
                "clauses": [
                    {"fact": "safe", "arguments": [{"result": "first"}]},
                    {
                        "fact": "visited-object",
                        "arguments": [{"result": "second"}],
                    },
                ],
            },
            {"fact": "at-object", "arguments": [{"result": "second"}]},
        ],
    }

    assert render_pddl_solution(goal, grounding) == "(at-object O3)"


def test_render_pddl_solution_supports_nested_negation():
    grounding = {"expected": {"object": ["O1"]}}
    goal = {
        "operator": "and",
        "clauses": [
            {"fact": "visited-object", "arguments": [{"result": "object"}]},
            {
                "operator": "not",
                "clauses": [{"fact": "holding", "arguments": [{"result": "object"}]}],
            },
        ],
    }

    assert render_pddl_solution(goal, grounding) == ("(and (visited-object O1) (not (holding O1)))")


def test_render_pddl_solution_can_negate_each_ambiguous_fact():
    grounding = {"expected": {"object": ["O1", "O2"]}}
    goal = {
        "fact": "holding",
        "arguments": [{"result": "object"}],
        "multiple": "and",
        "negate_each": True,
    }

    assert render_pddl_solution(goal, grounding) == ("(and (not (holding O1)) (not (holding O2)))")


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


def test_grounding_validation_accepts_semantically_mapped_aliases():
    contract = {
        "expected": {"first": ["O1", "O2"], "second": ["O3"]},
        "fields": {
            "first": {
                "aliases": ["first"],
                "symbol_type": "Object",
                "cardinality": "set",
            },
            "second": {
                "aliases": ["second"],
                "symbol_type": "Object",
                "cardinality": "set",
            },
        },
        "match": "exact",
    }

    matches, _rendered, actual = validate_grounding_rows(
        [
            {"object": "O1", "bench": "O3"},
            {"object": "O2", "bench": "O3"},
        ],
        contract,
    )

    assert matches
    assert actual == {"first": ["O1", "O2"], "second": ["O3"]}


def test_grounding_validation_rejects_columns_that_collapse_distinct_roles():
    contract = {
        "expected": {"first": ["O1", "O2"], "second": ["O3"]},
        "fields": {
            "first": {"aliases": ["first"], "symbol_type": "Object"},
            "second": {"aliases": ["second"], "symbol_type": "Object"},
        },
        "match": "exact",
    }

    assert (
        validate_grounding_rows([{"object": "O1"}, {"object": "O2"}, {"object": "O3"}], contract)[0]
        is False
    )


def test_empty_grounding_rejects_symbols_hidden_behind_unknown_alias():
    contract = {
        "expected": {"first": [], "second": []},
        "fields": {
            "first": {"aliases": ["first"], "symbol_type": "Object"},
            "second": {"aliases": ["second"], "symbol_type": "Object"},
        },
        "match": "exact",
    }

    assert validate_grounding_rows([{"object": "O1"}], contract)[0] is False


def test_room_grounding_accepts_region_result_alias():
    contract = {
        "expected": {"room": ["R3"]},
        "fields": {
            "room": {
                "aliases": ["room", "region", "nodeSymbol"],
                "symbol_type": "Room",
                "cardinality": "set",
            }
        },
        "match": "exact",
    }

    matches, _rendered, actual = validate_grounding_rows(
        [{"region": "R3"}],
        contract,
    )

    assert matches
    assert actual == {"room": ["R3"]}


def test_grounding_validation_infers_fields_from_unambiguous_symbol_patterns():
    contract = {
        "expected": {"object": ["O1"], "place": ["p38"], "room": ["R4"]},
        "fields": {
            "object": {
                "aliases": ["object"],
                "symbol_type": "Object",
                "cardinality": "set",
            },
            "place": {
                "aliases": ["place"],
                "symbol_type": "Place",
                "cardinality": "set",
            },
            "room": {
                "aliases": ["room"],
                "symbol_type": "Room",
                "cardinality": "set",
            },
        },
        "match": "exact",
    }

    matches, _rendered, actual = validate_grounding_rows(
        [{"subject": "O1", "waypoint": "p38", "destination": "R4"}],
        contract,
    )

    assert matches
    assert actual == {"object": ["O1"], "place": ["p38"], "room": ["R4"]}


def test_grounding_pattern_inference_preserves_place_symbol_case():
    contract = {
        "expected": {"place": ["p1"], "mesh_place": ["P1"]},
        "fields": {
            "place": {"symbol_type": "Place", "cardinality": "set"},
            "mesh_place": {"symbol_type": "MeshPlace", "cardinality": "set"},
        },
        "match": "exact",
    }

    matches, _rendered, actual = validate_grounding_rows(
        [{"first": "p1", "second": "P1"}], contract
    )

    assert matches
    assert actual == {"place": ["p1"], "mesh_place": ["P1"]}


def test_grounding_pattern_inference_does_not_guess_counts_or_ambiguous_fields():
    contract = {
        "expected": {"source": [], "target": [], "count": ["2"]},
        "fields": {
            "source": {"symbol_type": "Object", "cardinality": "set"},
            "target": {"symbol_type": "Object", "cardinality": "set"},
            "count": {
                "aliases": ["count"],
                "symbol_type": "Count",
                "cardinality": "set",
            },
        },
        "match": "exact",
    }

    matches, _rendered, actual = validate_grounding_rows(
        [{"left": "O1", "right": "O2", "count": 2}], contract
    )

    assert matches
    assert actual == {"source": [], "target": [], "count": ["2"]}


def test_empty_grounding_renders_empty_answer():
    contract = {
        "expected": {"object": []},
        "fields": {
            "object": {
                "aliases": ["object"],
                "symbol_type": "Object",
                "cardinality": "set",
            }
        },
        "match": "exact",
    }

    matches, rendered, actual = validate_grounding_rows([], contract)

    assert matches
    assert rendered == ""
    assert actual == {"object": []}
