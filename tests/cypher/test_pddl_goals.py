from cypher_3dsg_agent_distillation.cypher.pddl_goals import (
    render_pddl_solution,
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
