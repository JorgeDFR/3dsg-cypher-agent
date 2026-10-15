from cypher_3dsg_agent_distillation.graph.heracles_validation import (
    answer_matches_with_heracles,
    answer_type_hint,
    answer_type_hint_from_solution_formats,
    ensure_heracles_agents_importable,
    validate_pddl_goal,
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
        "(and (visited-object O1) (visited-region R2))",
        "(and (visited-region R2) (visited-object O1))",
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


def test_pddl_mismatch_does_not_print_dependency_diagnostics(capsys):
    valid_format, correct = answer_matches_with_heracles(
        "(at-place p31)",
        "(and (visited-place p31) (in-region R2))",
        "PDDL",
    )

    captured = capsys.readouterr()
    assert valid_format
    assert not correct
    assert captured.out == ""
    assert captured.err == ""


def test_pddl_validation_accepts_empty_answer_for_empty_solution():
    assert answer_matches_with_heracles("", "", "PDDL") == (True, True)
    assert answer_matches_with_heracles("", "(visited-object O1)", "PDDL") == (
        True,
        False,
    )
    assert answer_matches_with_heracles(
        "OBJECT_SYMBOL_NOT_VALID",
        "",
        "PDDL",
    ) == (False, False)


def test_pddl_domain_validation_rejects_unknown_predicate_wrong_arity_and_type():
    assert validate_pddl_goal("(teleported-object O1)")[0] is False
    assert validate_pddl_goal("(holding O1 O2)")[0] is False
    assert validate_pddl_goal("(holding R1)")[0] is False


def test_pddl_domain_validation_accepts_place_and_mesh_place_symbols():
    assert validate_pddl_goal("(visited-place p1)")[0]
    assert validate_pddl_goal("(visited-place P2)")[0]


def test_pddl_domain_validation_uses_region_predicates_for_room_symbols():
    assert validate_pddl_goal("(visited-region R1)")[0]
    assert validate_pddl_goal("(in-region R2)")[0]
    assert validate_pddl_goal("(visited-room R1)")[0] is False
    assert validate_pddl_goal("(in-room R2)")[0] is False


def test_pddl_domain_validation_accepts_conjunction_and_negation():
    assert validate_pddl_goal("(and (safe O1) (at-place p2) (not (holding O1)))")[0]


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
    assert (
        answer_type_hint_from_solution_formats(["set[list[str]]"], "SLDP")
        == "Your answer should be an SLDP set of lists of strings"
    )
    assert answer_type_hint_from_solution_formats(["str"], "PDDL") is None


def test_sldp_validation_enforces_declared_solution_format():
    assert answer_matches_with_heracles("[]", "[]", "SLDP", solution_formats=["str"]) == (
        False,
        False,
    )
    assert answer_matches_with_heracles("", "", "SLDP", solution_formats=["str"]) == (True, True)
    assert answer_matches_with_heracles("<O1>", "<O1>", "SLDP", solution_formats=["set[str]"]) == (
        True,
        True,
    )


def test_sldp_parser_reuses_grammar_without_changing_heracles_results(monkeypatch):
    import pytest
    from lark.exceptions import UnexpectedInput
    from sldp.lark_parser import lark_parse_sldp as reference_parse

    from cypher_3dsg_agent_distillation.graph import heracles_validation as validation

    samples = [
        "0",
        "3.25",
        "chair",
        "<>",
        "[]",
        "<O1, O2>",
        "[O1, [p1, P2]]",
        "POINT(-1.25 2 3)",
        "{target: O1}",
    ]
    expected = [reference_parse(value) for value in samples]
    original = validation.Lark
    compiled = []

    def count_compilation(*args, **kwargs):
        compiled.append(True)
        return original(*args, **kwargs)

    validation._sldp_parser.cache_clear()
    monkeypatch.setattr(validation, "Lark", count_compilation)
    try:
        assert [validation.lark_parse_sldp(value) for value in samples] == expected
        with pytest.raises(UnexpectedInput):
            validation.lark_parse_sldp("<broken")
        assert validation.lark_parse_sldp("<O3>") == ("set", "O3")
        assert len(compiled) == 1
    finally:
        validation._sldp_parser.cache_clear()
