import pytest
from neo4j.exceptions import DatabaseError

from cypher_3dsg_agent_distillation.graph.load_neo4j import initialize_database


def test_index_creation_does_not_hide_database_failures():
    class Driver:
        def execute_query(self, *_args, **_kwargs):
            raise DatabaseError("Index creation denied")

    with pytest.raises(DatabaseError, match="Index creation denied"):
        initialize_database(Driver(), "neo4j", clear=False)
