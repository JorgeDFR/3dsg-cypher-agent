from cypher_3dsg_agent_distillation.cypher.cypher_seed_records import (
    CypherSeedRecord,
    prepare_seed_records,
)


def test_prepare_seed_records_uses_profile_metadata_without_schema_copy():
    records = [
        CypherSeedRecord(
            uid="0A",
            question="What objects exist?",
            gold_cypher="MATCH (o:Object) RETURN DISTINCT o.class AS class",
            query_type="class_listing",
            solution_formats=["set[str]"],
            answer_validation={
                "mode": "top_k_with_ties",
                "limit": 2,
                "scores": {"chair": 3, "table": 3},
            },
        )
    ]

    prepared = prepare_seed_records(
        records,
        metadata_defaults={"experiment_profile": "cypher_qa_template"},
    )

    assert "schema" not in prepared[0].model_dump()
    assert "schema_name" not in prepared[0].model_dump()
    assert prepared[0].metadata["experiment_profile"] == "cypher_qa_template"
    assert prepared[0].cypher.startswith("MATCH")
    assert prepared[0].task_family == "cypher_qa"
    assert prepared[0].output_type == "SLDP"
    assert prepared[0].solution_formats == ["set[str]"]
    assert prepared[0].answer_validation == {
        "mode": "top_k_with_ties",
        "limit": 2,
        "scores": {"chair": 3, "table": 3},
    }


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

    prepared = prepare_seed_records(records)

    assert prepared[0].task_family == "cypher_pddl"
    assert prepared[0].output_type == "PDDL"
    assert prepared[0].solution == "(or (holding O1) (holding O2))"
