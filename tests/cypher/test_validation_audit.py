from copy import deepcopy

import pytest

from cypher_3dsg_agent_distillation.cypher.validation_audit import (
    load_templates,
    rescore_records,
    write_new,
)
from cypher_3dsg_agent_distillation.cypher.validation_config import resolve_contract
from cypher_3dsg_agent_distillation.cypher.validation_refresh import refresh_contracts
from tests.helpers import qa_contract, source_metadata


def source_seed():
    templates = load_templates(["configs/question_templates/cypher_pddl_3dsg.yaml"])
    template = next(t for t in templates if t["query_type"] == "pddl_visit_object_symbol")
    bindings = {"object": "O1"}
    seed = {
        "uid": "current",
        "query_type": template["query_type"],
        "cypher": template["cypher"].replace("{object}", "O1"),
        "solution": "(visited-object O1)",
        "grounding": {"expected": {"object": ["O1"]}},
        "validation_contract": resolve_contract(template["validation_contract"], bindings),
        "metadata": source_metadata(template, bindings),
    }
    return seed, templates


def test_contract_refresh_verifies_explicit_bindings_and_preserves_labels_and_inputs():
    seed, templates = source_seed()
    original = deepcopy(seed)
    refreshed = refresh_contracts([seed], templates)[0]
    assert refreshed["validation_contract"]["result"]["roles"]["object"]["origin"] == {
        "nodeSymbol": "O1"
    }
    assert refreshed["solution"] == seed["solution"]
    assert seed == original
    assert "validation_migration" not in refreshed["metadata"]
    assert "validation_contract" not in refreshed["grounding"]


@pytest.mark.parametrize("change", ["query", "identity", "bindings", "label", "contract"])
def test_refresh_rejects_missing_or_contradictory_current_inputs(change):
    seed, templates = source_seed()
    if change == "query":
        seed["cypher"] = "MATCH (r:Room) RETURN r.nodeSymbol AS object"
    elif change == "identity":
        seed["metadata"]["template_identity"] = "wrong"
    elif change == "bindings":
        seed["metadata"].pop("template_bindings")
    elif change == "label":
        seed["solution"] = "(visited-object O2)"
    else:
        seed.pop("validation_contract")
    with pytest.raises(ValueError):
        refresh_contracts([seed], templates)


def test_rescore_does_not_treat_missing_rows_as_empty_execution():
    record = {
        "uid": "a",
        "validation_contract": qa_contract(["int"]),
        "solution": "2",
        "cypher": "RETURN 2 AS total",
    }
    assert rescore_records([record])["records"][0]["result_validation"]["status"] == "not_checked"
    record["validation"] = {
        "executable": True,
        "rows": [{"total": 2}],
        "cypher_solution_match": False,
    }
    report = rescore_records([record])
    assert report["changed_grades"] == 1
    assert report["records"][0]["result_validation"]["status"] == "pass"


def test_audit_outputs_cannot_overwrite_existing_data(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("original")
    with pytest.raises(FileExistsError):
        write_new(output, {"different": True})
    assert output.read_text() == "original"


def ranking_seed():
    comparison = {
        "mode": "top_k_with_ties",
        "limit": {"binding": "limit"},
        "score_source": {"kind": "property_counts", "layer": "Object", "property": "class"},
    }
    template = {
        "query_type": "ranking",
        "cypher": "MATCH (o:Object) RETURN o.class AS class LIMIT {limit}",
        "variables": {"limit": "limits"},
        "validation_provenance": {},
        "validation_contract": {
            "result": {"kind": "set", "type": "string", "comparison": deepcopy(comparison)},
            "answer": {"formats": ["set[str]"], "comparison": deepcopy(comparison)},
        },
    }
    bindings = {"limit": 2}
    contract = resolve_contract(template["validation_contract"], bindings)
    for section in ("answer", "result"):
        contract[section]["comparison"]["scores"] = {"chair": 5, "table": 3, "desk": 3}
    record = {
        "uid": "ranked",
        "query_type": "ranking",
        "cypher": template["cypher"].replace("{limit}", "2"),
        "solution": "<chair, table>",
        "validation_contract": contract,
        "metadata": source_metadata(template, bindings),
    }
    return record, template


def test_refresh_preserves_observed_scores_only_for_the_same_scoring_rule():
    record, template = ranking_seed()
    refreshed = refresh_contracts([record], [template])[0]
    assert refreshed["validation_contract"]["answer"]["comparison"]["scores"] == {
        "chair": 5,
        "table": 3,
        "desk": 3,
    }
    # Changing to exact comparison drops scene scores.
    for section in ("answer", "result"):
        template["validation_contract"][section]["comparison"] = {"mode": "exact"}
    refreshed = refresh_contracts([record], [template])[0]
    assert refreshed["validation_contract"]["answer"]["comparison"]["scores"] is None


def test_refresh_cannot_reuse_scores_for_a_different_observation_source():
    record, template = ranking_seed()
    for section in ("answer", "result"):
        template["validation_contract"][section]["comparison"]["score_source"]["property"] = (
            "nodeSymbol"
        )
    original = deepcopy(record)
    with pytest.raises(ValueError, match="saved label violates"):
        refresh_contracts([record], [template])
    assert record == original


def test_refresh_rejects_retired_record_level_comparison_overrides():
    record, template = ranking_seed()
    record["answer_validation"] = {"mode": "top_k_with_ties", "limit": 2}
    with pytest.raises(ValueError, match="answer_validation.*regenerate"):
        refresh_contracts([record], [template])
