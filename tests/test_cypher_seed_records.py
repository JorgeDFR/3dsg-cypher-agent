from cypher_3dsg_agent_distillation.cypher_seed_records import (
    CypherSeedRecord,
    prepare_seed_records,
)


def test_prepare_seed_records_adds_schema_context_without_chat_messages():
    records = [
        CypherSeedRecord(
            uid="0A",
            question="What objects exist?",
            gold_cypher="MATCH (o:Object) RETURN DISTINCT o.class AS class",
            query_type="class_listing",
        )
    ]

    prepared = prepare_seed_records(records, "test_schema", "Labels in Database: Object")

    assert prepared[0].schema_name == "test_schema"
    assert prepared[0].schema_text == "Labels in Database: Object"
    assert prepared[0].cypher.startswith("MATCH")
