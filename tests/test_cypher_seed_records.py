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
    assert prepared[0].task_family == "cypher_qa"
    assert prepared[0].output_type == "SLDP"


def test_prepare_seed_records_preserves_native_pddl_fields():
    records = [
        CypherSeedRecord(
            uid="0P",
            question="Grab a wastebin",
            gold_cypher="MATCH (o:Object {class: 'trash'}) RETURN o.nodeSymbol",
            query_type="pddl_direct_class",
            task_family="cypher_pddl",
            output_type="PDDL",
            solution="(or (holding O1) (holding O2))",
        )
    ]

    prepared = prepare_seed_records(records, "test_schema", "Labels in Database: Object")

    assert prepared[0].task_family == "cypher_pddl"
    assert prepared[0].output_type == "PDDL"
    assert prepared[0].solution == "(or (holding O1) (holding O2))"
