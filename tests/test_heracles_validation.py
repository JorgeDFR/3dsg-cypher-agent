from cypher_3dsg_agent_distillation.heracles_validation import (
    answer_matches_with_heracles,
    answer_type_hint,
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
