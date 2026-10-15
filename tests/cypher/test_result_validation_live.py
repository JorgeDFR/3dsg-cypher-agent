"""Opt-in engine checks against rollback-only graphs on a reachable Neo4j server."""

import os

import pytest
from neo4j import GraphDatabase

from cypher_3dsg_agent_distillation.cypher.result_validation import (
    final_grounding_check,
    validate_record_result,
)
from tests.graph.rollback_neo4j import RollbackGraph

pytestmark = pytest.mark.skipif(
    not os.environ.get("VALIDATION_LIVE_NEO4J_URI"), reason="Explicit live Neo4j URI unavailable"
)


@pytest.fixture
def graph():
    with GraphDatabase.driver(
        os.environ["VALIDATION_LIVE_NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    ) as driver:
        graph = RollbackGraph(driver, os.environ.get("VALIDATION_LIVE_NEO4J_DATABASE", "neo4j"))
        try:
            yield graph
        finally:
            graph.close()


@pytest.mark.parametrize("representation", ["scalar", "collected", "nodes", "point", "count"])
def test_real_driver_values_and_cosmetic_aliases(graph, representation):
    graph.reset()
    graph.run(
        "CREATE (:Object {nodeSymbol:'O1',class:'chair',center:point({x:1,y:2,z:3})}), "
        "(:Object {nodeSymbol:'O2',class:'chair',center:point({x:4,y:5,z:6})})"
    ).consume()
    projection, result, expected = {
        "scalar": (
            "o.nodeSymbol",
            {"kind": "set", "type": "node_symbol", "layers": ["Object"], "property": "nodeSymbol"},
            "<O1,O2>",
        ),
        "collected": (
            "collect(o.nodeSymbol)",
            {"kind": "set", "type": "node_symbol", "layers": ["Object"], "property": "nodeSymbol"},
            "<O1,O2>",
        ),
        "nodes": (
            "o",
            {"kind": "set", "type": "node_symbol", "layers": ["Object"], "property": "nodeSymbol"},
            "<O1,O2>",
        ),
        "point": (
            "o.center",
            {"kind": "set", "type": "point", "layers": ["Object"], "property": "center"},
            "<POINT(1 2 3),POINT(4 5 6)>",
        ),
        "count": ("count(o)", {"kind": "scalar", "type": "integer"}, "2"),
    }[representation]
    contract = {"result": result}
    query = f"MATCH (o:Object) RETURN {projection} AS teacher_chosen_alias"
    validator = graph.validator_class(set())()
    execution = validator.validate(query, execute=True)
    assert execution["executable"], execution
    record = {"validation_contract": contract, "solution": expected}
    outcome = validate_record_result(record, execution["rows"], query, execution["columns"])
    assert outcome["status"] == "pass", outcome
    assert final_grounding_check(record, expected, outcome)["match"] is True


def test_same_type_pddl_roles_follow_provenance_and_report_ambiguity(graph):
    graph.reset()
    graph.run(
        "CREATE (:Object {nodeSymbol:'O1',class:'chair'}), "
        "(:Object {nodeSymbol:'O2',class:'table'})"
    ).consume()
    contract = {
        "result": {
            "kind": "role_sets",
            "roles": {
                "first": {
                    "type": "node_symbol",
                    "layers": ["Object"],
                    "property": "nodeSymbol",
                    "origin": {"class": "chair"},
                },
                "second": {
                    "type": "node_symbol",
                    "layers": ["Object"],
                    "property": "nodeSymbol",
                    "origin": {"class": "table"},
                },
            },
        },
        "goal": {
            "operator": "and",
            "clauses": [
                {"fact": "visited-object", "arguments": [{"result": "first"}], "multiple": "and"},
                {"fact": "at-object", "arguments": [{"result": "second"}], "multiple": "or"},
            ],
        },
        "answer": {"comparison": {"mode": "pddl_equivalence"}},
    }
    record = {
        "validation_contract": contract,
        "grounding": {"expected": {"first": ["O1"], "second": ["O2"]}},
    }
    query = (
        "MATCH (x:Object {class:'chair'}),(y:Object {class:'table'}) "
        "RETURN y.nodeSymbol AS destination_from_teacher, x.nodeSymbol AS inspected_by_teacher"
    )
    execution = graph.validator_class(set())().validate(query, execute=True)
    assert execution["executable"], execution
    outcome = validate_record_result(record, execution["rows"], query, execution["columns"])
    assert outcome["status"] == "pass", outcome
    assert (
        final_grounding_check(record, "(and (visited-object O1) (at-object O2))", outcome)["match"]
        is True
    )
    assert (
        final_grounding_check(record, "(and (visited-object O2) (at-object O1))", outcome)["match"]
        is False
    )
    for spec in contract["result"]["roles"].values():
        spec["origin"] = {}
    outcome = validate_record_result(record, execution["rows"], query, execution["columns"])
    assert outcome["status"] == "mapping_unresolved", outcome
