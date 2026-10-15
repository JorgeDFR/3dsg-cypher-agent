"""Live integration helpers that never commit changes to an existing database."""

from contextlib import contextmanager

from neo4j import Query

from cypher_3dsg_agent_distillation.cypher.validation_config import fingerprint
from cypher_3dsg_agent_distillation.graph.load_neo4j import (
    extract_graph_records,
    insert_nodes,
    insert_relationships,
)
from cypher_3dsg_agent_distillation.graph.neo4j_validation import Neo4jValidator, typed_record


def graph_fingerprint(driver, database):
    nodes, _, _ = driver.execute_query(
        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties "
        "ORDER BY id",
        database_=database,
    )
    relationships, _, _ = driver.execute_query(
        "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, elementId(a) AS source, "
        "elementId(b) AS target, type(r) AS type, properties(r) AS properties ORDER BY id",
        database_=database,
    )
    return fingerprint(
        {
            "nodes": [typed_record(row) for row in nodes],
            "relationships": [typed_record(row) for row in relationships],
        }
    )


class RollbackGraph:
    def __init__(self, driver, database):
        self.driver = driver
        self.database = database
        self._session = None
        self.tx = None
        self.baseline = graph_fingerprint(driver, database)

    def reset(self):
        self.rollback()
        self._session = self.driver.session(database=self.database)
        self.tx = self._session.begin_transaction(timeout=1200)
        self.tx.run("MATCH (n) DETACH DELETE n").consume()

    def load_scene(self, *, scene_path, **kwargs):
        self.reset()
        nodes, relationships = extract_graph_records(scene_path)
        for label, records in nodes.items():
            insert_nodes(self, self.database, label, records)
        insert_relationships(self, self.database, relationships)

    def execute_query(self, query, database_=None, **parameters):
        result = self.run(query, **parameters)
        columns = list(result.keys())
        rows = list(result)
        return rows, result.consume(), columns

    def run(self, query, **parameters):
        return self.tx.run(query.text if isinstance(query, Query) else query, **parameters)

    @contextmanager
    def session(self, **kwargs):
        yield self

    def rollback(self):
        if self.tx is not None:
            try:
                if not self.tx.closed():
                    self.tx.rollback()
            finally:
                self.tx = None
                self._session.close()
                self._session = None

    def close(self):
        self.rollback()
        assert graph_fingerprint(self.driver, self.database) == self.baseline, (
            "Live verification changed the committed graph"
        )

    def validator_class(self, syntax_queries):
        graph = self

        class TransactionValidator(Neo4jValidator):
            def __init__(self, **kwargs):
                self.driver = graph
                self.database = graph.database
                self.timeout = 30

            def validate(self, cypher, execute, **kwargs):
                if cypher.strip() in syntax_queries:
                    # A parser error aborts an explicit transaction. Check the
                    # controlled repair syntax in a separate EXPLAIN transaction.
                    standalone = Neo4jValidator.__new__(Neo4jValidator)
                    standalone.driver = graph.driver
                    standalone.database = graph.database
                    standalone.timeout = self.timeout
                    return standalone.validate(cypher, execute=False)
                return super().validate(cypher, execute, **kwargs)

            def close(self):
                # The fixture owns the transaction and always rolls it back.
                pass

        return TransactionValidator
