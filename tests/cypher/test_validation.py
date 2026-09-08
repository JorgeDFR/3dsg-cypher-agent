import argparse
import logging

import cypher_3dsg_agent_distillation.graph.neo4j_validation as validation
from cypher_3dsg_agent_distillation.core.config import load_yaml
from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
    DEFAULT_HERACLES_NEO4J_DATABASE,
    DEFAULT_HERACLES_NEO4J_PASSWORD,
    DEFAULT_HERACLES_NEO4J_URI,
    DEFAULT_HERACLES_NEO4J_URIS,
    DEFAULT_HERACLES_NEO4J_USERNAME,
    FALLBACK_HERACLES_NEO4J_URI,
    NEO4J_NOTIFICATION_LOGGER,
    neo4j_connection_settings,
    neo4j_uri_candidates,
    schema_policy,
    static_checks,
    static_checks_with_explain,
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


def test_generated_room_object_queries_pass_bounded_path_policy():
    policy = schema_policy(load_yaml("configs/schema_3dsg.yaml"))
    query_types = {
        "objects_in_room_symbol",
        "objects_in_room_class",
        "pddl_visit_room_containing_object",
        "pddl_end_in_room_containing_object",
    }
    templates = []
    for path in (
        "configs/question_templates/cypher_qa_3dsg.yaml",
        "configs/question_templates/cypher_pddl_3dsg.yaml",
    ):
        templates.extend(load_yaml(path)["templates"])

    queries = [
        template["cypher"]
        for template in templates
        if template["query_type"] in query_types
    ]
    assert len(queries) == len(query_types)
    assert all(static_checks(query, allow_writes=False, policy=policy) == [] for query in queries)


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


def test_static_checks_reject_write_keywords_without_space_delimiters():
    cases = {
        "CREATE(o:Object)": "CREATE",
        "MATCH (o:Object)\nSET\no.name = 'x'": "SET",
        "MATCH (o:Object)\nDELETE\no": "DELETE",
        "MATCH (o:Object) DETACH\nDELETE o": "DETACH",
    }

    for cypher, keyword in cases.items():
        issues = static_checks(cypher, allow_writes=False)
        assert f"write keyword not allowed: {keyword}" in issues


def test_static_checks_rejects_unqualified_distance_but_allows_point_distance():
    assert static_checks(
        "MATCH (o:Object) RETURN point.distance(o.center, o.center)",
        allow_writes=False,
    ) == []
    assert "forbidden function: distance" in static_checks(
        "MATCH (o:Object) RETURN distance(o.center, o.center)",
        allow_writes=False,
    )


def test_static_checks_with_explain_rejects_neo4j_parser_error():
    class Validator:
        def explain(self, cypher):
            assert "nodeSymbol <>" in cypher
            return {
                "syntax_error": True,
                "executable": False,
                "error": "Neo.ClientError.Statement.SyntaxError: Invalid input '<>'",
            }

    issues, explain_validation = static_checks_with_explain(
        'MATCH (o:Object {nodeSymbol <> "O1"}) RETURN o',
        allow_writes=False,
        validator=Validator(),
    )

    assert explain_validation["syntax_error"] is True
    assert issues == [
        "Neo4j EXPLAIN rejected query: "
        "Neo.ClientError.Statement.SyntaxError: Invalid input '<>'"
    ]


def test_static_checks_with_explain_stops_before_neo4j_for_policy_errors():
    class Validator:
        def explain(self, _cypher):
            raise AssertionError("policy-invalid Cypher must not reach Neo4j")

    issues, explain_validation = static_checks_with_explain(
        "MATCH (o:Object) DELETE o",
        allow_writes=False,
        validator=Validator(),
    )

    assert "write keyword not allowed: DELETE" in issues
    assert explain_validation is None


def test_neo4j_connection_settings_default_to_heracles_docker_then_host():
    settings = neo4j_connection_settings(
        argparse.Namespace(uri=None, username=None, password=None, database=None)
    )

    assert settings == {
        "uri": DEFAULT_HERACLES_NEO4J_URI,
        "uri_candidates": DEFAULT_HERACLES_NEO4J_URIS,
        "username": DEFAULT_HERACLES_NEO4J_USERNAME,
        "password": DEFAULT_HERACLES_NEO4J_PASSWORD,
        "database": DEFAULT_HERACLES_NEO4J_DATABASE,
    }
    assert DEFAULT_HERACLES_NEO4J_URI == "neo4j://neo4j:7683"
    assert FALLBACK_HERACLES_NEO4J_URI == "neo4j://127.0.0.1:7683"


def test_neo4j_uri_candidates_accept_comma_separated_overrides():
    assert neo4j_uri_candidates(" neo4j://custom:7683, neo4j://127.0.0.1:7683 ") == (
        "neo4j://custom:7683",
        "neo4j://127.0.0.1:7683",
    )


def test_neo4j_validator_falls_back_to_host_uri(monkeypatch):
    calls = []

    class Result:
        def consume(self):
            return None

    class Session:
        def __init__(self, uri):
            self.uri = uri

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def run(self, _query):
            if self.uri == DEFAULT_HERACLES_NEO4J_URI:
                raise RuntimeError("unreachable")
            return Result()

    class Driver:
        def __init__(self, uri):
            self.uri = uri
            self.closed = False

        def session(self, database):
            assert database == DEFAULT_HERACLES_NEO4J_DATABASE
            return Session(self.uri)

        def close(self):
            self.closed = True

    def fake_driver(uri, auth, connection_timeout):
        calls.append((uri, auth, connection_timeout))
        return Driver(uri)

    monkeypatch.setattr(validation.GraphDatabase, "driver", fake_driver)

    validator = validation.Neo4jValidator(
        uri=DEFAULT_HERACLES_NEO4J_URIS,
        username=DEFAULT_HERACLES_NEO4J_USERNAME,
        password=DEFAULT_HERACLES_NEO4J_PASSWORD,
        database=DEFAULT_HERACLES_NEO4J_DATABASE,
        timeout=10.0,
    )

    assert validator.uri == FALLBACK_HERACLES_NEO4J_URI
    assert calls == [
        (
            DEFAULT_HERACLES_NEO4J_URI,
            (DEFAULT_HERACLES_NEO4J_USERNAME, DEFAULT_HERACLES_NEO4J_PASSWORD),
            validation.DEFAULT_NEO4J_CONNECTION_TIMEOUT,
        ),
        (
            FALLBACK_HERACLES_NEO4J_URI,
            (DEFAULT_HERACLES_NEO4J_USERNAME, DEFAULT_HERACLES_NEO4J_PASSWORD),
            validation.DEFAULT_NEO4J_CONNECTION_TIMEOUT,
        ),
    ]


def test_neo4j_validator_streams_only_bounded_rows():
    data_calls = []

    class Record:
        def __init__(self, value):
            self.value = value

        def data(self):
            data_calls.append(self.value)
            return {"value": self.value}

    class Result:
        def __iter__(self):
            return iter(Record(value) for value in range(10))

        def consume(self):
            raise AssertionError("a truncated result must not be drained")

    class Session:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def run(self, _query):
            return Result()

    session = Session()

    class Driver:
        def session(self, database, fetch_size):
            assert database == "neo4j"
            assert fetch_size == 4
            return session

        def execute_query(self, *_args, **_kwargs):
            raise AssertionError("bounded validation must not eagerly execute")

    validator = object.__new__(validation.Neo4jValidator)
    validator.driver = Driver()
    validator.database = "neo4j"
    validator.timeout = 10.0

    result = validator.validate("MATCH (n) RETURN n", execute=True, max_rows=3)

    assert result["rows"] == [{"value": 0}, {"value": 1}, {"value": 2}]
    assert result["rows_truncated"] is True
    assert result["row_limit"] == 3
    assert data_calls == [0, 1, 2]
    assert session.closed is True


def test_suppress_neo4j_notification_logs_sets_logger_to_error():
    logger = logging.getLogger(NEO4J_NOTIFICATION_LOGGER)
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)

        suppress_neo4j_notification_logs()

        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(previous_level)
