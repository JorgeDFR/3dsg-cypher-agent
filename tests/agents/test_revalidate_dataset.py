import json
from copy import deepcopy
from pathlib import Path

import pytest

from cypher_3dsg_agent_distillation.agents.revalidate_dataset import (
    revalidate_attempt,
    revalidate_dataset,
)
from cypher_3dsg_agent_distillation.core.io import write_jsonl
from cypher_3dsg_agent_distillation.core.versioning import atomic_write_yaml
from cypher_3dsg_agent_distillation.cypher.validation_config import load_question_catalog


class Tokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def saved_attempt():
    query = "MATCH (o:Object) RETURN count(o) AS count"
    seed = {
        "uid": "count", "question": "How many objects?", "scene_id": "scene",
        "query_type": "object_class_count", "task_family": "cypher_qa", "output_type": "SLDP",
        "cypher": query, "solution": "3", "metadata": {},
        "validation_contract": {"result": {"kind": "scalar", "type": "integer"},
                                "answer": {"formats": ["int"], "empty": "0"}},
    }
    messages = [
        {"role": "user", "content": seed["question"]},
        {"role": "assistant", "tool_calls": [{"id": "call1", "type": "function", "function": {
            "name": "run_cypher_query", "arguments": {"cypher_string": query},
        }}]},
        {"role": "tool", "tool_call_id": "call1", "content": json.dumps({
            "ok": True, "rows": [{"count": 3}],
        })},
        {"role": "assistant", "content": "<answer>3</answer>", "reasoning": "Use the count."},
    ]
    item = {
        "attempt_id": "count:0", "seed_uid": "count", "accepted": False,
        "reason": "mapping_unresolved", "record": None,
        "debug_record": {"messages": messages, "validation": {
            "executable": True, "rows": [{"count": 3}], "columns": ["count"],
        }},
    }
    return seed, item


def test_saved_attempt_regrades_actual_evidence_and_preserves_teacher_messages():
    seed, item = saved_attempt()
    original = deepcopy(item)
    result = revalidate_attempt(
        item, seed, teacher_model="saved-teacher", max_tool_calls=3,
        tokenizer=Tokenizer(), max_length=5000,
    )
    assert result["accepted"]
    assert result["reason"] == "accepted_revalidated"
    assert result["record"]["metadata"]["seed_gold_cypher"] == seed["cypher"]
    assert result["record"]["messages"] == original["debug_record"]["messages"]
    assert item == original


@pytest.mark.parametrize("reason", ["generation_length", "max_tool_calls_exceeded",
                                   "multiple_tool_calls", "missing_required_tool"])
def test_protocol_rejections_are_not_accepted_by_dropping_failed_messages(reason):
    seed, item = saved_attempt()
    item["reason"] = reason
    result = revalidate_attempt(
        item, seed, teacher_model="saved-teacher", max_tool_calls=3,
        tokenizer=Tokenizer(), max_length=5000,
    )
    assert not result["accepted"]
    assert result["reason"] == reason


def test_dataset_revalidation_rewrites_consistent_outputs_and_preserves_billing(tmp_path):
    seed, item = saved_attempt()
    template_path = "configs/question_templates/cypher_qa_3dsg.yaml"
    template = next(t for t in load_question_catalog(template_path)["templates"]
                    if t["query_type"] == "object_class_count")
    seed["cypher"] = template["cypher"].replace('{class}', 'chair')
    seed["solution_formats"] = ["int"]
    traces = tmp_path / "traces"
    traces.mkdir()
    write_jsonl([seed], tmp_path / "cypher_seed_records.jsonl")
    atomic_write_yaml({"questions": [seed]}, tmp_path / "generated_questions.yaml")
    atomic_write_yaml({"stages": {}}, tmp_path / "metadata.yaml")
    write_jsonl([item], traces / "attempts.jsonl")
    write_jsonl([{"attempt_id": "count:0", "cost_usd": 0.012}], traces / "requests.jsonl")
    billing_bytes = (traces / "requests.jsonl").read_bytes()
    atomic_write_yaml({
        "status": "complete", "outputs": {}, "counts": {"accepted": 0, "rejected": 1},
        "student_config": {"snapshot": {"training": {"max_length": 5000}}},
        "teacher_config": {"snapshot": {"model": "saved-teacher", "max_tool_calls": 3}},
    }, traces / "metadata.yaml")
    atomic_write_yaml({}, traces / "token_length_report.yaml")
    atomic_write_yaml({}, traces / "pilot_report.yaml")
    config = {
        "dataset_dir": str(tmp_path), "metadata_path": str(tmp_path / "metadata.yaml"),
        "teacher_metadata_path": str(traces / "metadata.yaml"),
        "seed_records_jsonl_path": str(tmp_path / "cypher_seed_records.jsonl"),
        "raw_questions_path": str(tmp_path / "generated_questions.yaml"),
        "question_sets": [{"question_templates_path": template_path}],
    }
    preview = revalidate_dataset(config, tokenizer=Tokenizer())
    assert preview["accepted_after"] == 1
    assert not (traces / "teacher_traces.jsonl").exists()
    result = revalidate_dataset(config, tokenizer=Tokenizer(), apply=True)
    assert result["accepted_after"] == 1
    assert result["teacher_requests_added"] == 0
    assert (traces / "requests.jsonl").read_bytes() == billing_bytes
    assert len(json.loads((traces / "accepted.json").read_text())) == 1
    assert json.loads((traces / "rejected.json").read_text()) == []
    backup = Path(result["backup_directory"])
    assert (backup / "traces/attempts.jsonl").exists()
    accepted = json.loads((traces / "teacher_traces.jsonl").read_text())
    assert accepted["validation"]["answer_grounded"]
