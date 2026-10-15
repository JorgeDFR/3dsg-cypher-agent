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


def test_generated_questions_round_trip_through_seed_preparation_with_shared_contracts(tmp_path):
    from copy import deepcopy

    import yaml

    from cypher_3dsg_agent_distillation.core.io import read_jsonl, write_jsonl
    from cypher_3dsg_agent_distillation.cypher.cypher_seed_records import load_raw_seed_records
    from cypher_3dsg_agent_distillation.cypher.templates import (
        LiteralString,
        SolutionBlock,
        write_questions_yaml,
    )
    from cypher_3dsg_agent_distillation.cypher.validation_config import fingerprint

    qa_contract = {'version': 1, 'result': {'kind': 'set', 'type': 'string'},
                   'answer': {'formats': ['set[str]'], 'empty': '<>'}}
    pddl_contract = {
        'version': 1, 'result': {'kind': 'role_sets', 'roles': {
            'object': {'type': 'node_symbol', 'layers': ['Object']}}},
        'goal': {'fact': 'visited-object', 'arguments': [{'result': 'object'}]},
    }
    questions = []
    for index, template in enumerate([qa_contract, pddl_contract]):
        contract = deepcopy(template)
        record = {
            'uid': str(index), 'question': 'List targets.', 'query_type': 'targets',
            'task_family': 'cypher_pddl' if index else 'cypher_qa',
            'output_type': 'PDDL' if index else 'SLDP',
            'gold_cypher': LiteralString('MATCH (o:Object)\nRETURN o.nodeSymbol AS object\n'),
            'solution': '(visited-object O1)' if index else SolutionBlock('<chair, table>\n'),
            'validation_contract': contract,
            'metadata': {'validation_contract_hash': fingerprint(contract)},
        }
        # Each record contains a repeated mapping. The old per-record dump
        # restarted its anchor counter and made the combined document invalid.
        record['grounding'] = {'validation_contract': contract}
        if index:
            record['grounding']['expected'] = {'object': ['O1']}
        questions.append(record)
    path = tmp_path / 'generated_questions.yaml'
    write_questions_yaml(questions, path)
    assert yaml.safe_load(path.read_text())['questions'] == questions
    assert 'gold_cypher: |' in path.read_text()
    assert 'solution: |' in path.read_text()
    loaded = load_raw_seed_records(path)
    prepared = prepare_seed_records(loaded)
    output = tmp_path / 'cypher_seed_records.jsonl'
    write_jsonl(prepared, output)
    records = read_jsonl(output)
    assert len(records) == 2
    for original, record in zip(questions, records, strict=True):
        assert record['validation_contract'] == original['validation_contract']
        assert record['grounding'] == original['grounding']
        assert record['metadata'] == original['metadata']
        assert record['cypher'] == original['gold_cypher'].strip()
        assert record['solution'] == original['solution']
