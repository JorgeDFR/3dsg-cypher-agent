from cypher_3dsg_agent_distillation.heracles_validation import (
    answer_matches_with_heracles,
    answer_type_hint,
    answer_type_hint_from_solution_formats,
    ensure_heracles_agents_importable,
)


def test_heracles_agents_submodule_validation_packages_are_importable():
    ensure_heracles_agents_importable()
    import pypddl  # noqa: F401
    import sldp  # noqa: F401


def test_sldp_equality_uses_heracles_set_semantics():
    valid_format, correct = answer_matches_with_heracles("<A, B>", "<B, A>", "SLDP")

    assert valid_format
    assert correct


def test_sldp_invalid_format_is_reported():
    valid_format, correct = answer_matches_with_heracles("A, B", "<B, A>", "SLDP")

    assert not valid_format
    assert not correct


def test_top_k_with_ties_accepts_any_cutoff_tie_member():
    validation = {
        "mode": "top_k_with_ties",
        "limit": 3,
        "scores": {"light": 10, "chair": 8, "flag": 7, "plant": 7},
    }

    valid_format, correct = answer_matches_with_heracles(
        "<light, chair, plant>",
        "<light, chair, flag>",
        "SLDP",
        answer_validation=validation,
    )

    assert valid_format
    assert correct


def test_top_k_with_ties_rejects_missing_higher_ranked_or_wrong_size_answers():
    validation = {
        "mode": "top_k_with_ties",
        "limit": 3,
        "scores": {"light": 10, "chair": 8, "flag": 7, "plant": 7},
    }

    assert answer_matches_with_heracles(
        "<light, flag, plant>",
        "<light, chair, flag>",
        "SLDP",
        answer_validation=validation,
    ) == (True, False)
    assert answer_matches_with_heracles(
        "<light, chair>",
        "<light, chair, flag>",
        "SLDP",
        answer_validation=validation,
    ) == (True, False)


def test_pddl_equality_uses_heracles_goal_semantics():
    valid_format, correct = answer_matches_with_heracles(
        "(and (visited-object O1) (visited-room R2))",
        "(and (visited-room R2) (visited-object O1))",
        "PDDL",
    )

    assert valid_format
    assert correct


def test_pddl_invalid_format_is_reported():
    valid_format, correct = answer_matches_with_heracles(
        "visited-object O1",
        "(visited-object O1)",
        "PDDL",
    )

    assert not valid_format
    assert not correct


def test_sldp_answer_type_hint_uses_heracles_parser():
    assert answer_type_hint("<A, B>", "SLDP") == "Your answer should be an SLDP set"


def test_sldp_answer_type_hint_uses_solution_formats_when_available():
    assert (
        answer_type_hint_from_solution_formats(["set[str]"], "SLDP")
        == "Your answer should be an SLDP set of strings"
    )
    assert (
        answer_type_hint_from_solution_formats(["set[point]", "point"], "SLDP")
        == "Your answer should be an SLDP set of points or a point"
    )
    assert answer_type_hint_from_solution_formats(["str"], "PDDL") is None
