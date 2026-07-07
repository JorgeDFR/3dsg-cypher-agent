from sg_text2cypher.neo4j_validation import schema_policy, static_checks

SCHEMA = {
    "labels": {
        "Object": {"properties": ["nodeSymbol", "class", "center"]},
        "Room": {"properties": ["nodeSymbol", "class", "center"]},
    },
    "relationships": {
        "CONTAINS": {"patterns": ["(Room)-[:CONTAINS]->(Object)"]},
        "ROOM_CONNECTED": {"patterns": ["(Room)-[:ROOM_CONNECTED]-(Room)"]},
    },
    "safety": {
        "forbidden_write_keywords": ["DELETE"],
        "require_bounded_variable_length_paths": True,
    },
}


def test_static_checks_accept_valid_3dsg_query():
    issues = static_checks(
        "MATCH (r:Room)-[:CONTAINS*1..3]->(o:Object) "
        "RETURN o.nodeSymbol, point.distance(r.center, o.center)",
        allow_writes=False,
        policy=schema_policy(SCHEMA),
    )
    assert issues == []


def test_static_checks_reject_unknown_schema_and_unbounded_paths():
    issues = static_checks(
        "MATCH (x:Person)-[:KNOWS*]->(o:Object) RETURN x.name DELETE o",
        allow_writes=False,
        policy=schema_policy(SCHEMA),
    )
    assert "unknown label: Person" in issues
    assert "unknown relationship: KNOWS" in issues
    assert "unknown property: name" in issues
    assert "unbounded variable-length path: [:KNOWS*]" in issues
    assert "write keyword not allowed: DELETE" in issues
