"""Current generation and validation handoffs without saved project artifacts or services."""

from copy import deepcopy

import pytest

from cypher_3dsg_agent_distillation.agents.agent_dataset import (
    assert_no_split_leakage,
    grouped_split_records,
    read_agent_jsonl,
)
from cypher_3dsg_agent_distillation.agents.agent_evaluation import recompute_trace_validation
from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord
from cypher_3dsg_agent_distillation.agents.experiment_profiles import load_experiment_profile
from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
    prepare_synthetic_reasoning,
    run_synthetic_teacher_interaction,
)
from cypher_3dsg_agent_distillation.core.io import write_jsonl
from cypher_3dsg_agent_distillation.cypher import templates
from cypher_3dsg_agent_distillation.cypher.cypher_seed_records import (
    load_raw_seed_records,
    prepare_seed_records,
)
from cypher_3dsg_agent_distillation.cypher.task_semantics import prepare_seed_semantics
from cypher_3dsg_agent_distillation.cypher.validation_config import load_question_catalog
from cypher_3dsg_agent_distillation.reporting.workflow_report import write_workflow_report
from tests.helpers import answerable_trace, qa_contract, source_metadata


class Scene:
    def __init__(self, _path):
        pass

    def solution_for(self, query_type, variables):
        return 3 if query_type == "object_total" else variables["object"]


class Validator:
    def validate(self, cypher, execute=True):
        rows = [{"count": 3}] if "count(o)" in cypher else [{"object": "O1"}]
        return {"executable": True, "rows": rows, "columns": list(rows[0]), "error": None}


@pytest.mark.parametrize("reasoning_enabled", [False, True])
def test_fresh_qa_and_pddl_pipeline_preserves_current_contracts(
    tmp_path, monkeypatch, reasoning_enabled
):
    monkeypatch.setattr(templates, "SceneIndex", Scene)
    generated, question_sets, profiles = [], [], {}
    for family, query_type in (
        ("cypher_qa", "object_total"),
        ("cypher_pddl", "pddl_visit_object_symbol"),
    ):
        path = f"configs/question_templates/{family}_3dsg.yaml"
        template = next(
            t for t in load_question_catalog(path)["templates"] if t["query_type"] == query_type
        )
        template["questions"] = template["questions"][:1]
        profile = load_experiment_profile(f"configs/experiment_profiles/{family}_template.yaml")
        profiles[family] = profile
        question_sets.append({"name": family, "question_templates_path": path})
        generated.extend(
            templates.generate_records(
                {
                    "scene_id": "example",
                    "source_3dsg": "mock-scene",
                    "values": {"object_symbols": ["O1"]},
                },
                {"templates": [template]},
                limit_per_template=1,
                task_family=family,
                output_type=profile.output_type,
                experiment_profile=profile.name,
                question_set=family,
                warn_on_target_shortfall=False,
            )
        )
    raw = tmp_path / "questions.yaml"
    templates.write_questions_yaml(generated, raw)
    seeds = [
        r.model_dump(exclude_none=True) for r in prepare_seed_records(load_raw_seed_records(raw))
    ]
    seed_path = tmp_path / "seeds.jsonl"
    write_jsonl(seeds, seed_path)
    prepare_synthetic_reasoning(
        seeds,
        question_sets,
        {"provider": "synthetic", "reasoning": {"enabled": reasoning_enabled}},
    )
    accepted = []
    for seed in seeds:
        assert bool(seed["metadata"].get("reasoning_context")) is reasoning_enabled
        decision = run_synthetic_teacher_interaction(
            seed,
            profile=profiles[seed["task_family"]],
            validator=Validator(),
            policy=None,
            reasoning={"enabled": reasoning_enabled},
        )
        assert decision.accepted, decision.reason
        payload = decision.record.model_dump(exclude_none=True)
        assert "real_solution" not in payload
        assert "answer_validation" not in payload
        assert "validation_contract" not in (payload.get("grounding") or {})
        assert recompute_trace_validation(decision.record)["answer_grounded"] is True
        accepted.append(payload)
    trace_path = tmp_path / "traces.jsonl"
    write_jsonl(accepted, trace_path)
    train, evaluation, test, challenge = grouped_split_records(
        read_agent_jsonl(trace_path), train_size=1, eval_size=0, test_size=0, seed=42
    )
    assert len(train) == 2
    parts = dict(
        zip(
            ("train", "eval", "test", "challenge"),
            (train, evaluation, test, challenge),
            strict=True,
        )
    )
    assert_no_split_leakage(parts)
    split_paths = {}
    for name, records in parts.items():
        split_paths[name] = str(tmp_path / f"{name}.jsonl")
        write_jsonl(records, split_paths[name])
    config = {
        "dataset_id": "current",
        "dataset_dir": str(tmp_path),
        "dataset_version": "example",
        "scenes": [
            {
                "scene_id": "example",
                "scene_path": str(tmp_path / "scene.json"),
                "scene_template_values_path": str(tmp_path / "values.yaml"),
            }
        ],
        "raw_questions_path": str(raw),
        "seed_records_jsonl_path": str(seed_path),
        "teacher_traces_jsonl_path": str(trace_path),
        "accepted_traces_path": str(tmp_path / "accepted.json"),
        "rejected_traces_path": str(tmp_path / "rejected.json"),
        "splits": split_paths,
    }
    output = write_workflow_report(config, None, tmp_path / "workflow.html", dataset_only=True)
    assert "Generated Questions" in output.read_text()


@pytest.mark.parametrize("location", ["missing", "nested"])
def test_trace_boundary_requires_a_top_level_contract(location):
    payload = answerable_trace()
    contract = payload.pop("validation_contract")
    if location == "nested":
        payload["grounding"] = {"validation_contract": contract}
    with pytest.raises(ValueError, match="top-level validation_contract.*regenerate"):
        AgentTraceRecord.model_validate(payload)


def test_task_metadata_cannot_supply_missing_current_trace_fields():
    payload = answerable_trace()
    payload["metadata"] = {
        "task_family": payload.pop("task_family"),
        "output_type": payload.pop("output_type"),
    }
    with pytest.raises(ValueError, match="task_family"):
        AgentTraceRecord.model_validate(payload)


def test_seed_boundary_requires_explicit_current_generation_provenance():
    from cypher_3dsg_agent_distillation.cypher.cypher_seed_records import CypherSeedRecord

    payload = {
        "uid": "seed",
        "question": "Count objects",
        "query_type": "count",
        "cypher": "MATCH (o:Object) RETURN count(o) AS count",
        "validation_contract": qa_contract(["int"]),
    }
    with pytest.raises(ValueError, match="explicit bindings.*regenerate"):
        CypherSeedRecord.model_validate(payload)


@pytest.mark.parametrize("change", ["old_context", "fingerprint", "binding", "contract"])
def test_current_synthetic_preparation_rejects_invalid_provenance(change):
    template = {
        "query_type": "constant",
        "cypher": "RETURN {count} AS total",
        "variables": {"count": "counts"},
        "validation_contract": qa_contract(["int"]),
    }
    seed = {
        "cypher": "RETURN 1 AS total",
        "metadata": source_metadata(template, {"count": 1}),
        "validation_contract": deepcopy(template["validation_contract"]),
    }
    prepare_seed_semantics(seed, template)
    original = deepcopy(seed)
    if change == "old_context":
        seed["metadata"]["reasoning_context"]["version"] = 1
    elif change == "fingerprint":
        seed["metadata"]["reasoning_context"]["fingerprint"] = "tampered"
    elif change == "binding":
        seed["metadata"]["template_bindings"]["count"] = 2
    else:
        seed["validation_contract"]["result"]["type"] = "number"
    with pytest.raises(ValueError):
        prepare_seed_semantics(seed, template)
    assert original["metadata"]["reasoning_context"]["version"] == 2
