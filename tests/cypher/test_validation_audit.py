from copy import deepcopy

import pytest

from cypher_3dsg_agent_distillation.cypher.validation_audit import (
    load_templates,
    migrate_records,
    rescore_records,
    write_new,
)


def test_migration_requires_query_agreement_and_preserves_historical_labels():
    templates = load_templates(["configs/question_templates/cypher_pddl_3dsg.yaml"])
    old = {
        "uid": "old",
        "query_type": "pddl_visit_object_symbol",
        "cypher": 'MATCH (o:Object {nodeSymbol: "O1"}) RETURN o.nodeSymbol AS object',
        "solution": "(visited-object O1)",
        "grounding": {"expected": {"object": ["O1"]}},
    }
    source = deepcopy(old)
    migrated, issues = migrate_records([old], templates)
    assert not issues
    assert migrated[0]["validation_contract"]["result"]["roles"]["object"]["origin"] == {
        "nodeSymbol": "O1"
    }
    assert old == source
    wrong = {**old, "cypher": "MATCH (r:Room) RETURN r.nodeSymbol AS object"}
    _records, issues = migrate_records([wrong], templates)
    assert issues[0]["status"] == "migration_unresolved"
    flawed_label = {**old, "solution": "(visited-object O2)"}
    migrated, issues = migrate_records([flawed_label], templates)
    assert issues
    assert migrated[0]["solution"] == flawed_label["solution"]


def test_rescore_does_not_treat_missing_rows_as_empty_execution():
    c = {
        "result": {"kind": "scalar", "type": "integer"},
        "answer": {"formats": ["int"], "empty": "0"},
    }
    record = {"uid": "a", "validation_contract": c, "solution": "2", "cypher": "RETURN 2 AS total"}
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
