"""Decoder regression tests use only query/row evidence to establish mappings."""

from copy import deepcopy

import pytest

from cypher_3dsg_agent_distillation.cypher.query_provenance import projections
from cypher_3dsg_agent_distillation.cypher.result_validation import (
    decode_result,
    validate_record_result,
)


def class_roles():
    return {
        "result": {
            "kind": "role_sets",
            "roles": {
                name: {
                    "type": "node_symbol",
                    "layers": ["Object"],
                    "property": "nodeSymbol",
                    "origin": {"class": cls},
                }
                for name, cls in [("first", "sofa"), ("second", "painting")]
            },
        }
    }


POOLED = (
    "MATCH (o:Object) WHERE o.class IN ['sofa', 'painting'] "
    "RETURN o.nodeSymbol AS id, o.class AS class"
)
POOLED_ROWS = [{"id": "O1", "class": "sofa"}, {"id": "O2", "class": "painting"}]
SYMBOL_SET = {
    "result": {"kind": "set", "type": "node_symbol", "layers": ["Object"], "property": "nodeSymbol"}
}


@pytest.mark.parametrize(
    "suffix,rows",
    [
        ("point.distance(o.center, p.center) AS arbitrary", [{"id": "O1", "arbitrary": 1.5}]),
        ("o.class AS arbitrary", [{"id": "O1", "arbitrary": "sofa"}]),
        ("labels(o) AS arbitrary", [{"id": "O1", "arbitrary": ["Object"]}]),
        ("count(p) AS arbitrary", [{"id": "O1", "arbitrary": 2}]),
    ],
)
def test_diagnostic_columns_are_typed_without_alias_allowlists(suffix, rows):
    query = f"MATCH (o:Object), (p:Place) RETURN o.nodeSymbol AS id, {suffix}"
    original = deepcopy(rows)
    decoded = decode_result(query, rows, SYMBOL_SET)
    assert decoded.status == "pass"
    assert decoded.actual == ["O1"]
    assert decoded.auxiliary_columns == ["arbitrary"]
    assert rows == original
    strict = deepcopy(SYMBOL_SET)
    strict["result"]["extra_columns"] = "reject"
    assert decode_result(query, rows, strict).status == "invalid_result_shape"


@pytest.mark.parametrize("rows", [[], [{"id": "O1", "distance": 1.5}]])
def test_distance_type_survives_aggregation_and_with_alias(rows):
    query = (
        "MATCH (o:Object), (p:Place) "
        "WITH o, min(point.distance(o.center, p.center)) AS best "
        "RETURN o.nodeSymbol AS id, best AS distance"
    )
    decoded = decode_result(query, rows, SYMBOL_SET)
    assert decoded.status == "pass"
    assert decoded.bindings == {"value": "id"}
    assert decoded.actual == (["O1"] if rows else [])


def test_plausible_answer_column_cannot_be_discarded_to_match_gold():
    query = (
        "MATCH (a:Object), (b:Object) RETURN a.nodeSymbol AS a, b.nodeSymbol AS b, "
        "point.distance(a.center, b.center) AS d"
    )
    for gold in ["<O1>", "<O2>"]:
        result = validate_record_result(
            {"validation_contract": SYMBOL_SET, "solution": gold},
            [{"a": "O1", "b": "O2", "d": 2.0}],
            query,
        )
        assert result["status"] == "mapping_unresolved"
        assert result["match"] is None


@pytest.mark.parametrize("value", ["not a number", float("nan"), None, True])
def test_inferred_diagnostics_do_not_hide_invalid_values(value):
    query = (
        "MATCH (o:Object), (p:Place) RETURN o.nodeSymbol AS id, "
        "point.distance(o.center,p.center) AS d"
    )
    assert decode_result(query, [{"id": "O1", "d": value}], SYMBOL_SET).status != "pass"


def test_unknown_auxiliary_expression_requires_a_declaration():
    query = "MATCH (o:Object) RETURN o.nodeSymbol AS id, o.unspecified AS extra"
    rows = [{"id": "O1", "extra": 3}]
    assert decode_result(query, rows, SYMBOL_SET).status != "pass"
    declared = deepcopy(SYMBOL_SET)
    declared["result"]["auxiliary"] = {"extra": {"type": "integer"}}
    assert decode_result(query, rows, declared).status == "pass"


@pytest.mark.parametrize(
    "rows,expected",
    [
        (POOLED_ROWS, {"first": ["O1"], "second": ["O2"]}),
        ([{"id": "O2", "class": "painting"}], {"first": [], "second": ["O2"]}),
        ([], {"first": [], "second": []}),
    ],
)
def test_pooled_roles_partition_rows_and_preserve_empty_roles(rows, expected):
    decoded = decode_result(POOLED, rows, class_roles())
    assert decoded.status == "pass"
    assert decoded.actual == expected
    assert decoded.bindings == {"first": "id", "second": "id"}
    assert decoded.role_rows == {
        name: [i for i, row in enumerate(rows) if row["class"] == cls]
        for name, cls in [("first", "sofa"), ("second", "painting")]
    }
    assert decoded.auxiliary_columns == ["class"]


def test_role_decoding_ignores_expected_grounding():
    outcomes = [
        validate_record_result(
            {"validation_contract": class_roles(), "grounding": {"expected": expected}},
            POOLED_ROWS,
            POOLED,
        )
        for expected in [
            {"first": ["O1"], "second": ["O2"]},
            {"first": ["O2"], "second": ["O1"]},
        ]
    ]
    for field in ["actual", "bindings", "role_rows", "auxiliary_columns"]:
        assert outcomes[0][field] == outcomes[1][field]
    assert outcomes[0]["match"] is True
    assert outcomes[1]["match"] is False


@pytest.mark.parametrize("rows", [POOLED_ROWS, [{"id": "O2", "class": "painting"}]])
def test_limit_does_not_prove_complete_pooled_roles(rows):
    assert decode_result(POOLED + " LIMIT 1", rows, class_roles()).status != "pass"


@pytest.mark.parametrize(
    "bad_row",
    [
        {"id": "R999", "class": "sofa"},
        {"id": "O3", "class": "other"},
        {"id": None, "class": "sofa"},
        {"id": ["O1", 12], "class": "sofa"},
    ],
)
def test_pooled_partition_never_filters_invalid_members(bad_row):
    assert decode_result(POOLED, [*POOLED_ROWS, bad_row], class_roles()).status != "pass"


def test_class_from_another_node_cannot_establish_a_role():
    query = (
        "MATCH (o:Object), (other:Object) WHERE o.class IN ['sofa','painting'] "
        "RETURN o.nodeSymbol AS id, other.class AS class"
    )
    assert decode_result(query, POOLED_ROWS, class_roles()).status == "mapping_unresolved"


def test_same_node_identity_survives_with_and_property_aliases():
    query = (
        "MATCH (o:Object) WHERE o.class IN ['sofa','painting'] "
        "WITH o AS renamed, o.class AS category "
        "RETURN renamed.nodeSymbol AS id, category AS class"
    )
    assert decode_result(query, POOLED_ROWS, class_roles()).actual == {
        "first": ["O1"],
        "second": ["O2"],
    }
    assert projections(query)[0]["source"] == projections(query)[1]["source"]


def test_explicit_semantic_alias_is_still_authoritative():
    query = POOLED.replace("AS id", "AS first")
    rows = [{"first": row["id"], "class": row["class"]} for row in POOLED_ROWS]
    assert decode_result(query, rows, class_roles()).status != "pass"


def test_unlabelled_nodes_use_paired_labels_and_public_id_constraints():
    contract = {
        "result": {
            "kind": "role_sets",
            "roles": {
                name: {
                    "type": "node_symbol",
                    "layers": [layer],
                    "property": "nodeSymbol",
                    "origin": {"nodeSymbol": symbol},
                }
                for name, symbol, layer in [("first", "P1", "MeshPlace"), ("second", "p1", "Place")]
            },
        }
    }
    query = (
        "MATCH (n) WHERE n.nodeSymbol IN ['P1','p1'] RETURN n.nodeSymbol AS id, labels(n) AS labels"
    )
    rows = [{"id": "p1", "labels": ["Place"]}, {"id": "P1", "labels": ["MeshPlace"]}]
    assert decode_result(query, rows, contract).actual == {"first": ["P1"], "second": ["p1"]}
    rows[0]["labels"] = ["MeshPlace"]
    assert decode_result(query, rows, contract).status != "pass"


@pytest.mark.parametrize("union", ["UNION", "UNION ALL", "union\nall"])
def test_union_branch_provenance_uses_layers_and_constraints(union):
    contract = class_roles()
    contract["result"]["roles"]["second"]["layers"] = ["Room"]
    query = (
        "MATCH (o:Object {class:'sofa'}) RETURN o.nodeSymbol AS id "
        f"{union} MATCH (r:Room {{class:'painting'}}) RETURN r.nodeSymbol AS id"
    )
    result = decode_result(query, [{"id": "O1"}, {"id": "R1"}], contract)
    assert result.status == "pass"
    assert result.actual == {"first": ["O1"], "second": ["R1"]}


def test_union_same_layer_branches_without_a_discriminator_remain_unresolved():
    query = (
        "MATCH (o:Object {class:'sofa'}) RETURN o.nodeSymbol AS id "
        "UNION MATCH (r:Object {class:'painting'}) RETURN r.nodeSymbol AS id"
    )
    assert decode_result(query, [{"id": "O1"}], class_roles()).status == "mapping_unresolved"
    query = query.replace("AS id", "AS id, o.class AS class", 1)
    query = query.rsplit("AS id", 1)[0] + "AS id, r.class AS class"
    assert decode_result(query, POOLED_ROWS, class_roles()).actual == {
        "first": ["O1"],
        "second": ["O2"],
    }


def test_union_keyword_in_literals_and_comments_is_not_a_branch():
    query = "MATCH (o:Object {class:'UNION ALL'}) /* UNION */ RETURN o.nodeSymbol AS id // UNION\n"
    assert len(projections(query)) == 1
    assert decode_result(query, [{"id": "O1"}], SYMBOL_SET).actual == ["O1"]


def test_overlapping_pooled_role_constraints_are_not_assigned_arbitrarily():
    contract = class_roles()
    contract["result"]["roles"]["second"]["origin"] = {"class": "sofa"}
    assert decode_result(POOLED, POOLED_ROWS, contract).status != "pass"


def test_pooled_full_nodes_use_their_own_labels_and_properties():
    query = "MATCH (o:Object) WHERE o.class IN ['sofa','painting'] RETURN o AS node"
    rows = [
        {
            "node": {
                "__neo4j_type__": "node",
                "labels": ["Object"],
                "properties": {"nodeSymbol": row["id"], "class": row["class"]},
            }
        }
        for row in POOLED_ROWS
    ]
    assert decode_result(query, rows, class_roles()).actual == {
        "first": ["O1"],
        "second": ["O2"],
    }
    rows[1]["node"]["labels"] = ["Room"]
    assert decode_result(query, rows, class_roles()).status != "pass"


def test_full_node_properties_cannot_override_query_constraints():
    query = "MATCH (o:Object {class:'sofa'}) RETURN o AS node"
    row = {
        "node": {
            "__neo4j_type__": "node",
            "labels": ["Object"],
            "properties": {"nodeSymbol": "O1", "class": "painting"},
        }
    }
    assert decode_result(query, [row], class_roles()).status == "invalid_result_shape"


def test_union_limit_does_not_establish_an_empty_role():
    query = (
        "MATCH (o:Object {class:'sofa'}) RETURN o.nodeSymbol AS id UNION "
        "MATCH (r:Room {class:'painting'}) RETURN r.nodeSymbol AS id LIMIT 1"
    )
    contract = class_roles()
    contract["result"]["roles"]["second"]["layers"] = ["Room"]
    for rows in [[{"id": "O1"}], [{"id": "R1"}]]:
        assert decode_result(query, rows, contract).status != "pass"


@pytest.mark.parametrize("limit,passes", [(1, True), (0, False)])
def test_empty_pooled_search_is_complete_only_with_a_positive_limit(limit, passes):
    result = decode_result(POOLED + f" LIMIT {limit}", [], class_roles())
    assert (result.status == "pass") is passes


def test_mixed_layer_union_uses_branch_provenance_even_with_a_role_named_alias():
    contract = class_roles()
    contract["result"]["roles"]["second"]["layers"] = ["Room"]
    query = (
        "MATCH (o:Object {class:'sofa'}) RETURN o.nodeSymbol AS first UNION "
        "MATCH (r:Room {class:'painting'}) RETURN r.nodeSymbol AS first"
    )
    rows = [{"first": "O1"}, {"first": "R1"}]
    assert decode_result(query, rows, contract).actual == {"first": ["O1"], "second": ["R1"]}
    explicit = deepcopy(contract)
    explicit["result"]["roles"]["first"]["column"] = "first"
    assert decode_result(query, rows, explicit).status != "pass"


def test_or_equalities_establish_the_same_role_domain_as_in():
    query = POOLED.replace(
        "o.class IN ['sofa', 'painting']", "o.class = 'sofa' OR o.class = 'painting'"
    )
    result = decode_result(query, POOLED_ROWS, class_roles())
    assert result.actual == {"first": ["O1"], "second": ["O2"]}
    # An unrelated OR branch does not prove a constrained class domain.
    query = query.replace("o.class = 'painting'", "o.nodeSymbol = 'O3'")
    assert decode_result(query, POOLED_ROWS, class_roles()).status != "pass"


def test_relationship_variable_and_anonymous_node_preserve_auxiliary_count_type():
    contract = {"result": {"kind": "set", "type": "node_symbol", "layers": ["Room"]}}
    query = (
        "MATCH (r:Room) OPTIONAL MATCH (r)-[e:ROOM_CONNECTED]-(:Room) "
        "WITH r, count(e) AS deg RETURN r.nodeSymbol AS room, deg"
    )
    result = decode_result(query, [{"room": "R1", "deg": 1}], contract)
    assert result.status == "pass"
    assert result.actual == ["R1"]


@pytest.mark.parametrize("index", [0, -1])
def test_indexed_collect_preserves_target_class_provenance(index):
    contract = class_roles()
    contract["result"]["roles"] = {"target": contract["result"]["roles"]["second"]}
    query = (
        "MATCH (a:Object {class:'sofa'}), (b:Object {class:'painting'}) "
        f"WITH a, collect(b.nodeSymbol)[{index}] AS nearest "
        "RETURN a.nodeSymbol AS reference, nearest"
    )
    result = decode_result(query, [{"reference": "O1", "nearest": "O2"}], contract)
    assert result.status == "pass"
    assert result.actual == {"target": ["O2"]}
    assert result.bindings == {"target": "nearest"}


@pytest.mark.parametrize("rows", [[], [{"pair": ["O1", "O2"]}]])
def test_literal_list_projection_retains_independent_tuple_positions(rows):
    contract = {
        "result": {
            "kind": "tuples",
            "roles": {
                name: {"type": "node_symbol", "layers": ["Object"]} for name in ("item_0", "item_1")
            },
        }
    }
    query = "MATCH (a:Object), (b:Object) RETURN [a.nodeSymbol,b.nodeSymbol] AS pair"
    result = decode_result(query, rows, contract)
    assert result.status == "pass"
    assert result.actual == ([["O1", "O2"]] if rows else [])
    assert decode_result(query, [{"pair": ["O1", "R1"]}], contract).status != "pass"


@pytest.mark.parametrize(
    "condition",
    [
        "toLower(o.class) CONTAINS 'paint'",
        "toLower(o.class) = 'painting'",
        "toLower(o.class) CONTAINS 'paint' OR toLower(o.class) CONTAINS 'art'",
    ],
)
def test_empty_string_search_can_establish_a_public_role(condition):
    contract = class_roles()
    contract["result"]["roles"] = {"target": contract["result"]["roles"]["second"]}
    query = f"MATCH (o:Object) WHERE {condition} RETURN o.nodeSymbol AS id LIMIT 1"
    assert decode_result(query, [], contract, ["id"]).actual == {"target": []}
    assert decode_result(query, [{"id": "O2"}], contract).status != "pass"
    assert decode_result(query.replace("paint", "unrelated"), [], contract).status != "pass"


def test_unexplained_column_cannot_displace_the_known_answer_projection():
    contract = {"result": {"kind": "scalar", "type": "integer"}}
    query = "MATCH (o:Object) RETURN count(o) AS count"
    result = decode_result(query, [{"count": 3, "unexplained": 42}], contract)
    assert result.status == "invalid_result_shape"


def test_saved_renamed_rows_are_rejected_when_query_projection_is_known():
    result = decode_result(
        "MATCH (o:Object) RETURN o.nodeSymbol AS object",
        [{"renamed": "O999"}],
        SYMBOL_SET,
    )
    assert result.status == "invalid_result_shape"
    assert "columns disagree" in result.issues[0]


def test_union_branch_identity_is_not_required_for_an_unambiguous_simple_set():
    query = (
        "MATCH (o:Object {class:'sofa'}) RETURN o.nodeSymbol AS id UNION "
        "MATCH (p:Object {class:'painting'}) RETURN p.nodeSymbol AS id"
    )
    decoded = decode_result(query, [{"id": "O1"}, {"id": "O2"}], SYMBOL_SET)
    assert decoded.status == "pass"
    assert decoded.actual == ["O1", "O2"]
