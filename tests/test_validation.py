import argparse
import logging

from cypher_3dsg_agent_distillation.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_URI,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    NEO4J_NOTIFICATION_LOGGER,
    neo4j_connection_settings,
    schema_policy,
    static_checks,
    suppress_neo4j_notification_logs,
)

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


def test_neo4j_connection_settings_default_to_heracles_docker():
    settings = neo4j_connection_settings(
        argparse.Namespace(uri=None, username=None, password=None, database=None)
    )

    assert settings == {
        "uri": DEFAULT_HERACLES_NEO4J_URI,
        "username": DEFAULT_HERACLES_NEO4J_USERNAME,
        "password": DEFAULT_HERACLES_NEO4J_PASSWORD,
        "database": DEFAULT_HERACLES_NEO4J_DATABASE,
    }


def test_suppress_neo4j_notification_logs_sets_logger_to_error():
    logger = logging.getLogger(NEO4J_NOTIFICATION_LOGGER)
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)

        suppress_neo4j_notification_logs()

        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(previous_level)
