import pytest

from cypher_3dsg_agent_distillation.cypher.evaluation import (
    answer_matches,
    build_summary,
    evaluate_records,
)
from tests.helpers import qa_contract


def test_answer_matches_sldp_set_with_loose_order():
    expected = "<\n  armchair,\n  chair,\n  sofa\n>\n"
    actual = "<sofa, armchair, chair>"
    assert answer_matches(actual, expected)


def test_answer_matches_does_not_accept_missing_set_items():
    assert not answer_matches("sofa, chair", "<armchair, chair, sofa>")


@pytest.mark.parametrize("value,status", [(3, "pass"), (4, "mismatch")])
def test_standalone_evaluation_uses_the_current_contract(value, status):
    class Validator:
        def validate(self, _query, execute=True):
            return {"executable": True, "rows": [{"count": value}], "error": None}

    record = {
        "uid": "count",
        "query_type": "count",
        "solution": "3",
        "cypher": "MATCH (o:Object) RETURN count(o) AS count",
        "validation_contract": qa_contract(["int"]),
    }
    result = evaluate_records([record], "cypher", Validator(), None)[0]
    assert result["status"] == status
    assert result["solution_match"] is (value == 3)


def test_standalone_evaluation_rejects_missing_contract_before_execution():
    with pytest.raises(ValueError, match="validation_contract.*regenerate"):
        evaluate_records([{"cypher": "RETURN 1"}], "cypher", None, None)


def test_standalone_refusal_never_executes_and_still_checks_the_expected_answer():
    refusal = "WRITE_QUERY_NOT_ALLOWED"
    contract = qa_contract()
    contract["query_policy"]["refusal"] = refusal
    record = {
        "uid": "refusal",
        "query_type": "refusal",
        "cypher": refusal,
        "solution": refusal,
        "validation_contract": contract,
    }
    result = evaluate_records([record], "cypher", None, None)[0]
    assert result["status"] == "pass"
    assert result["executed"] is False
    record["solution"] = "another"
    result = evaluate_records([record], "cypher", None, None)[0]
    assert result["solution_match"] is False


def test_build_summary_counts_solution_matches_and_errors():
    summary = build_summary(
        [
            {
                "query_type": "object_class_count",
                "static_valid": True,
                "executable": True,
                "solution_match": True,
                "status": "pass",
            },
            {
                "query_type": "object_class_count",
                "static_valid": True,
                "executable": True,
                "solution_match": False,
                "status": "mismatch",
            },
        ]
    )

    assert summary["total"] == 2
    assert summary["metrics"]["solution_match"] == 1
    assert summary["metrics"]["solution_mismatch"] == 1
