from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from cypher_3dsg_agent_distillation.cypher.query_provenance import semantic_evidence
from cypher_3dsg_agent_distillation.cypher.result_validation import (
    compare_answers,
    decode_result,
    final_grounding_check,
    validate_record_result,
)
from cypher_3dsg_agent_distillation.cypher.validation_config import (
    INLINE_FIELDS,
    ValidationContract,
    load_question_catalog,
    resolve_contract,
)


def roles():
    return {
        "result": {
            "kind": "role_sets",
            "roles": {
                "first": {
                    "type": "node_symbol",
                    "layers": ["Object"],
                    "property": "nodeSymbol",
                    "origin": {"class": "armchair"},
                },
                "second": {
                    "type": "node_symbol",
                    "layers": ["Object"],
                    "property": "nodeSymbol",
                    "origin": {"class": "bench"},
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


QUERY = (
    'MATCH (a:Object {class:"armchair"}), (b:Object {class:"bench"}) '
    "RETURN a.nodeSymbol AS seat, b.nodeSymbol AS finish"
)


def test_aliases_follow_provenance_and_ignore_expected_values():
    contract = roles()
    rows = [{"finish": "O3", "seat": "O1"}, {"finish": "O3", "seat": "O2"}]
    decoded = decode_result(QUERY, rows, contract)
    assert decoded.status == "pass"
    assert decoded.actual == {"first": ["O1", "O2"], "second": ["O3"]}
    for expected in [
        {"first": ["O1", "O2"], "second": ["O3"]},
        {"first": ["O3"], "second": ["O1", "O2"]},
    ]:
        outcome = validate_record_result(
            {"validation_contract": contract, "grounding": {"expected": expected}}, rows, QUERY
        )
        assert outcome["actual"] == decoded.actual
        assert outcome["bindings"] == decoded.bindings
    assert outcome["match"] is False


def test_explicit_swapped_roles_cannot_be_reassigned_to_match_gold():
    query = QUERY.replace("AS seat", "AS second").replace("AS finish", "AS first")
    result = decode_result(query, [{"second": "O1", "first": "O3"}], roles())
    assert result.status == "invalid_result_shape"
    assert "origin conflicts" in result.issues[0]


def test_same_type_ambiguity_is_unresolved_without_gold_guessing():
    contract = roles()
    for role in contract["result"]["roles"].values():
        role["origin"] = {}
    result = decode_result("unsupported query", [{"a": "O1", "b": "O2"}], contract)
    assert result.status == "mapping_unresolved"


@pytest.mark.parametrize("value", ["R999", ["O1", 12], ["O1", None]])
def test_invalid_values_are_not_filtered(value):
    c = {"result": {"kind": "set", "type": "node_symbol", "layers": ["Object"]}}
    assert (
        decode_result("RETURN anything", [{"renamed": value}], c).status == "invalid_result_shape"
    )


def test_extra_column_requires_explicit_typed_declaration():
    c = {"result": {"kind": "set", "type": "node_symbol", "layers": ["Object"]}}
    rows = [{"answer": "O1", "unrelated": "O999"}]
    assert decode_result("", rows, c).status != "pass"
    c["result"].update(
        element={"type": "node_symbol", "layers": ["Object"], "column": "answer"},
        extra_columns="declared",
        auxiliary={"unrelated": {"type": "node_symbol", "layers": ["Object"]}},
    )
    assert decode_result("", rows, c).actual == ["O1"]
    rows[0]["unrelated"] = 42
    assert decode_result("", rows, c).status == "invalid_result_shape"


def test_empty_expected_role_does_not_hide_returned_room():
    c = {
        "result": {
            "kind": "role_sets",
            "roles": {"room": {"type": "node_symbol", "layers": ["Room"]}},
        }
    }
    r = validate_record_result(
        {"validation_contract": c, "grounding": {"expected": {"room": []}}},
        [{"region": "R999"}],
        "",
    )
    assert r["match"] is False
    assert r["extra"] == {"room": ["R999"]}


@pytest.mark.parametrize(
    "value",
    [
        "O1",
        ["O1"],
        {"nodeSymbol": "O1"},
        {"__neo4j_type__": "node", "labels": ["Object"], "properties": {"nodeSymbol": "O1"}},
    ],
)
def test_scalar_collection_map_and_full_node_forms(value):
    c = {
        "result": {
            "kind": "set",
            "type": "node_symbol",
            "layers": ["Object"],
            "property": "nodeSymbol",
            "accepted_forms": ["scalar_columns", "collected_columns", "nodes", "maps"],
        }
    }
    assert decode_result(
        "MATCH (x:Object) RETURN x AS whatever", [{"whatever": value}], c
    ).actual == ["O1"]


def test_map_layout_is_explicit():
    c = {
        "result": {
            "kind": "scalar",
            "element": {"type": "integer", "column": "payload", "path": ["count"]},
            "accepted_forms": ["maps", "scalar_columns"],
        }
    }
    assert decode_result("", [{"payload": {"count": 2}}], c).actual == 2


def test_points_are_not_flattened_into_coordinates():
    c = {"result": {"kind": "list", "type": "point"}}
    assert decode_result("", [{"xyz": [1, 2, 3]}], c).actual == [(1.0, 2.0, 3.0)]
    assert decode_result("", [{"xyz": [[1, 2, 3], [4, 5, 6]]}], c).actual == [
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    ]


def test_tuple_associations_and_ordered_results_are_preserved():
    c = {
        "result": {
            "kind": "tuples",
            "roles": {
                "object": {"type": "node_symbol", "layers": ["Object"]},
                "place": {"type": "node_symbol", "layers": ["Place"]},
            },
        }
    }
    a = decode_result("", [{"a": "O1", "b": "p1"}, {"a": "O2", "b": "p2"}], c)
    b = decode_result("", [{"a": "O1", "b": "p2"}, {"a": "O2", "b": "p1"}], c)
    assert a.actual != b.actual
    c = {"result": {"kind": "list"}, "answer": {"formats": ["list[str]"], "empty": "[]"}}
    assert compare_answers("[b,a]", "[a,b]", c)["match"] is False
    c["result"]["kind"] = "multiset"
    assert compare_answers("[b,a,a]", "[a,b,a]", c)["match"] is True
    assert compare_answers("[b,a]", "[a,b,a]", c)["match"] is False


def test_counts_empty_rows_nulls_and_case():
    c = {
        "result": {"kind": "scalar", "type": "integer", "row_count": {"min": 1, "max": 1}},
        "answer": {"formats": ["int"], "empty": "0"},
    }
    assert decode_result("", [{"n": 0}], c).actual == 0
    assert decode_result("", [], c).status == "invalid_result_shape"
    assert decode_result("", [{"n": None}], c).status == "invalid_result_shape"
    assert decode_result("", [{"n": True}], c).status == "invalid_result_shape"
    assert compare_answers("1.5", "1", c)["format_valid"] is False
    c = {
        "result": {"kind": "set", "type": "node_symbol", "layers": ["Place", "MeshPlace"]},
        "answer": {"formats": ["set[str]"], "empty": "<>"},
    }
    assert compare_answers("<p119>", "<P119>", c)["match"] is False


def test_actual_final_grounding_uses_actual_goal_and_reports_invalid_gold():
    c = roles()
    r = {"validation_contract": c, "grounding": {"expected": {"first": ["O1"], "second": ["O2"]}}}
    outcome = validate_record_result(r, [{"seat": "O1", "finish": "O2"}], QUERY)
    assert final_grounding_check(r, "(and (visited-object O1) (at-object O2))", outcome)["match"]
    assert not final_grounding_check(r, "(and (visited-object O2) (at-object O1))", outcome)[
        "match"
    ]
    assert (
        compare_answers("(at-object O1)", "(at-object R2)", c)["status"]
        == "invalid_expected_solution"
    )


def test_semantic_evidence_is_conservative():
    q = "MATCH (a:Place)-[:NEAR*2]->(b:Place) RETURN b.nodeSymbol AS item"
    renamed = (
        q.replace("(a:", "(x:")
        .replace("(b:", "(y:")
        .replace("b.nodeSymbol AS item", "y.nodeSymbol AS other")
        .replace("*2", "*2..2")
    )
    assert semantic_evidence(q, renamed)["status"] == "equivalent_in_supported_fragment"
    assert semantic_evidence(q, q.replace("*2", "*..2"))["status"] == "semantics_unverified"
    assert semantic_evidence(q, q.replace("->", "-"))["status"] == "semantics_unverified"


def test_all_catalogs_have_single_validation_owner_and_resolve_bindings():
    total = 0
    for path in Path("configs/question_templates").glob("*.yaml"):
        for raw in yaml.safe_load(path.read_text())["templates"]:
            assert not (raw.keys() & INLINE_FIELDS)
        catalog = load_question_catalog(path)
        for t in catalog["templates"]:
            ValidationContract.model_validate(t["validation_contract"])
        total += len(catalog["templates"])
    assert total == 91
    t = load_question_catalog("configs/question_templates/cypher_pddl_3dsg.yaml")["templates"][0]
    c = resolve_contract(t["validation_contract"], {"object": "O42"})
    assert c["result"]["roles"]["object"]["origin"] == {"nodeSymbol": "O42"}


def test_strict_loader_coverage_inline_fields_duplicate_keys_and_versions(tmp_path):
    main = tmp_path / "questions.yaml"
    side = tmp_path / "validation"
    side.mkdir()
    main.write_text("templates:\n- query_type: count\n  cypher: RETURN 0 AS count\n")
    (side / "defaults.yaml").write_text("version: 1\ndefaults: {}\n")
    target = side / main.name
    good = {"version": 1, "templates": {"count": {"result": {"kind": "scalar", "type": "integer"}}}}
    target.write_text(yaml.safe_dump(good))
    assert load_question_catalog(main)["templates"][0]["validation_contract"]
    for version in [True, "1", 2]:
        bad = deepcopy(good)
        bad["version"] = version
        target.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            load_question_catalog(main)
    target.write_text("version: 1\nversion: 1\ntemplates: {}\n")
    with pytest.raises(ValueError, match="Duplicate YAML"):
        load_question_catalog(main)
    target.write_text(yaml.safe_dump({"version": 1, "templates": {}}))
    with pytest.raises(ValueError, match="coverage"):
        load_question_catalog(main)
    target.write_text(yaml.safe_dump(good))
    main.write_text(main.read_text() + "  solution_formats: [int]\n")
    with pytest.raises(ValueError, match="inline fields"):
        load_question_catalog(main)


def test_empty_scalar_remains_grounded_and_map_lineage_follows_declared_path():
    c = {"result": {"kind": "scalar"}, "answer": {"formats": ["str"], "empty": ""}}
    record = {"validation_contract": c, "solution": ""}
    outcome = validate_record_result(record, [], "MATCH (o:Object) RETURN o.class AS x")
    assert outcome["status"] == "pass"
    assert final_grounding_check(record, "", outcome)["match"] is True
    c = roles()
    for role in c["result"]["roles"].values():
        role.update(
            column="payload", path=["chair" if role["origin"]["class"] == "armchair" else "bench"]
        )
    # A shared map column is meaningful only through explicit independent paths.
    c["result"]["accepted_forms"] = ["maps", "scalar_columns"]
    query = (
        'MATCH (a:Object {class:"armchair"}), (b:Object {class:"bench"}) '
        "RETURN {chair:a.nodeSymbol, bench:b.nodeSymbol} AS payload"
    )
    result = decode_result(query, [{"payload": {"chair": "O1", "bench": "O2"}}], c)
    assert result.status == "pass"
    assert result.actual == {"first": ["O1"], "second": ["O2"]}


def test_tie_comparison_keeps_case_and_rejects_duplicate_entries():
    c = {
        "result": {"kind": "set", "type": "node_symbol", "layers": ["Place", "MeshPlace"]},
        "answer": {
            "formats": ["set[str]"],
            "empty": "<>",
            "comparison": {
                "mode": "top_k_with_ties",
                "limit": 1,
                "scores": {"P1": 2, "p1": 1, "P2": 2},
            },
        },
    }
    assert compare_answers("<P2>", "<P1>", c)["match"] is True
    assert compare_answers("<p1>", "<P1>", c)["match"] is False
    assert compare_answers("<P1,P1>", "<P1>", c)["match"] is False
    assert compare_answers("<P1>", "<p1>", c)["status"] == "invalid_expected_solution"


def test_ranked_grounding_matches_execution_without_reusing_oracle_tie_policy():
    contract = {
        "result": {"kind": "set", "type": "string"},
        "answer": {
            "formats": ["set[str]"],
            "empty": "<>",
            "comparison": {
                "mode": "top_k_with_ties",
                "limit": 1,
                "scores": {"chair": 2, "table": 2},
                "score_source": {"layer": "Object", "property": "class"},
            },
        },
    }
    original = deepcopy(contract)
    record = {"validation_contract": contract}
    result = {"solution": "<chair>"}
    assert compare_answers("<table>", "<chair>", contract)["match"] is True
    assert final_grounding_check(record, "<chair>", result)["match"] is True
    assert final_grounding_check(record, "<table>", result)["match"] is False
    assert contract == original


@pytest.mark.parametrize(
    "kind,formats,actual,expected,tolerances",
    [
        ("number", ["float"], "1.05", "1.0", {"numeric_abs_tolerance": 0.1}),
        ("point", ["point"], "POINT(1.05 2 3)", "POINT(1 2 3)", {"point_abs_tolerance": 0.1}),
    ],
)
def test_final_grounding_preserves_configured_tolerances(
    kind, formats, actual, expected, tolerances
):
    contract = {
        "result": {"kind": "scalar", "type": kind},
        "answer": {"formats": formats, "comparison": tolerances},
    }
    assert final_grounding_check(
        {"validation_contract": contract}, actual, {"solution": expected}
    )["match"] is True


def test_equal_role_sets_are_not_ambiguous_because_of_row_order():
    c = roles()
    for role in c["result"]["roles"].values():
        role["origin"] = {}
    decoded = decode_result("unsupported", [{"a": "O1", "b": "O2"}, {"a": "O2", "b": "O1"}], c)
    assert decoded.status == "pass"
    assert decoded.actual == {"first": ["O1", "O2"], "second": ["O1", "O2"]}


def test_point_tolerance_uses_same_matching_for_role_diagnostics():
    c = {
        "result": {
            "kind": "role_sets",
            "roles": {"point": {"type": "point"}},
            "comparison": {"point_abs_tolerance": 0.1},
        }
    }
    r = {"validation_contract": c, "grounding": {"expected": {"point": [[1.0, 2.0, 3.0]]}}}
    outcome = validate_record_result(r, [{"point": [1.05, 2.0, 3.0]}], "")
    assert outcome["status"] == "pass"
    assert outcome["missing"] == {"point": []}
    assert outcome["extra"] == {"point": []}


def test_explicit_catalog_and_defaults_paths_override_discovery(tmp_path):
    main = tmp_path / "questions.yaml"
    main.write_text("templates:\n- query_type: count\n  cypher: RETURN 0 AS total\n")
    contract = tmp_path / "custom-contract.yaml"
    contract.write_text(
        "version: 1\ntemplates:\n  count:\n    result: {kind: scalar, type: integer}\n"
    )
    defaults = tmp_path / "custom-defaults.yaml"
    defaults.write_text('version: 1\ndefaults:\n  answer: {formats: [int], empty: "0"}\n')
    resolved = load_question_catalog(main, contract, defaults)["templates"][0][
        "validation_contract"
    ]
    assert resolved["answer"]["formats"] == ["int"]
    assert resolved["answer"]["empty"] == "0"


def test_semantic_mode_is_runtime_policy_and_fixture_settings_are_rejected(monkeypatch):
    query = "RETURN 1 AS total"
    contract = {
        "result": {"kind": "scalar", "type": "integer"},
        "answer": {"formats": ["int"], "empty": "0"},
    }
    record = {"validation_contract": contract, "gold_cypher": query, "solution": "1"}
    enabled = validate_record_result(record, [{"total": 1}], query)
    assert enabled["semantic_check"]["status"] == "equivalent_in_supported_fragment"
    contract["semantic_checks"] = {"mode": "unverified"}

    def unexpected_analysis(*args):
        raise AssertionError("Disabled semantic analysis should not run")

    monkeypatch.setattr(
        "cypher_3dsg_agent_distillation.cypher.result_validation.semantic_evidence",
        unexpected_analysis,
    )
    disabled = validate_record_result(record, [{"total": 1}], query)
    assert disabled["semantic_check"]["status"] == "semantics_unverified"
    assert enabled["match"] is disabled["match"] is True
