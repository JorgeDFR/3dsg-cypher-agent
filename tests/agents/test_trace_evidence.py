import json
from copy import deepcopy

import pytest

from cypher_3dsg_agent_distillation.agents.agent_dataset import (
    optimization_kinds,
    read_student_trace_records_with_counts,
)
from cypher_3dsg_agent_distillation.agents.agent_evaluation import (
    recompute_trace_validation,
    saved_execution,
)
from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord
from cypher_3dsg_agent_distillation.agents.teacher_optimizations import (
    recover_rejected_earlier_retrieval,
)
from cypher_3dsg_agent_distillation.agents.trace_evidence import (
    select_earlier_retrieval as select_evidence,
)
from cypher_3dsg_agent_distillation.agents.trace_evidence import (
    tool_executions,
)
from cypher_3dsg_agent_distillation.cypher.validation_audit import rescore_records, saved_result

TASK = "MATCH (o:Object {class:'chair'}) RETURN o.nodeSymbol AS id"
DIAGNOSTIC = "MATCH (o:Object) RETURN DISTINCT o.class AS class"


def conversation(executions, answer="<O1>"):
    messages = [{"role": "user", "content": "List chair IDs."}]
    for i, (query, rows) in enumerate(executions, 1):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "reasoning": f"Reasoning for call {i}.",
                    "reasoning_details": [{"type": "reasoning.encrypted", "data": f"opaque{i}"}],
                    "tool_calls": [
                        {
                            "id": f"call{i}",
                            "type": "function",
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {"cypher_string": query},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call{i}",
                    "content": json.dumps({"ok": True, "rows": rows}),
                },
            ]
        )
    messages.append(
        {
            "role": "assistant",
            "content": f"<answer>{answer}</answer>",
            "reasoning": "Use the earlier retrieval after checking the classes.",
        }
    )
    return messages


def rejected_trace(executions=None, answer="<O1>"):
    executions = executions or [(TASK, [{"id": "O1"}]), (DIAGNOSTIC, [{"class": "chair"}])]
    return {
        "uid": "teacher",
        "seed_uid": "seed",
        "question": "List chair IDs.",
        "query_type": "chairs",
        "reason": "invalid_result_shape",
        "scene_id": "scene",
        "task_family": "cypher_qa",
        "output_type": "SLDP",
        "real_solution": "<O1>",
        "cypher": executions[-1][0],
        "teacher_model": "teacher",
        "validation": {"static_valid": True, "executable": True, "rows": executions[-1][1]},
        "validation_contract": {
            "result": {
                "kind": "set",
                "type": "node_symbol",
                "layers": ["Object"],
                "property": "nodeSymbol",
            },
            "answer": {"formats": ["set[str]"], "empty": "<>"},
        },
        "metadata": {"seed_gold_cypher": TASK},
        "messages": conversation(executions, answer),
    }


def select_earlier_retrieval(record):
    return select_evidence(
        record["messages"],
        record["validation_contract"],
        record.get("metadata", {}).get("seed_gold_cypher"),
    )


def test_recovery_preserves_all_messages_and_revalidation_uses_earlier_evidence(tmp_path):
    original = rejected_trace()
    before = deepcopy(original)
    recovered = recover_rejected_earlier_retrieval(original)
    assert recovered is not None
    assert original == before
    assert recovered.model_dump(exclude_none=True)["messages"] == original["messages"]
    assert recovered.gold_cypher == TASK
    assert recovered.metadata["grounding_tool_call_index"] == 1
    assert recovered.metadata["successful_tool_call_index"] == 2
    assert recovered.metadata["diagnostic_tool_call_indices"] == [2]
    assert not recovered.metadata.get("final_answer_repaired")
    assert saved_execution(recovered) == (TASK, [{"id": "O1"}], None)
    assert saved_result(recovered.model_dump()) == saved_execution(recovered)
    grade = recompute_trace_validation(recovered)
    assert all(
        grade[k] is True
        for k in [
            "cypher_solution_match",
            "final_answer_match",
            "answer_grounded",
        ]
    )
    rescored = rescore_records([recovered.model_dump()])
    assert rescored["records"][0]["result_validation"]["status"] == "pass"
    path = tmp_path / "traces.jsonl"
    path.write_text(recovered.model_dump_json() + "\n")
    records, counts = read_student_trace_records_with_counts({"teacher_traces_jsonl_path": path})
    assert counts["recovered_earlier_retrieval"] == 1
    assert optimization_kinds(records[0]) == ["recovered_earlier_retrieval"]
    assert recompute_trace_validation(records[0])["answer_grounded"] is True


def test_unmarked_traces_keep_latest_execution_policy():
    original = rejected_trace()
    record = AgentTraceRecord.model_validate(
        {
            **original,
            "gold_cypher": original["cypher"],
            "expected_solution": "<O1>",
        }
    )
    assert saved_execution(record) == (DIAGNOSTIC, [{"class": "chair"}], None)
    assert recompute_trace_validation(record)["result_validation"]["status"] != "pass"


def test_selection_is_independent_of_expected_values_and_final_answer():
    baseline = rejected_trace()
    other = deepcopy(baseline)
    other["real_solution"] = "<O999>"
    other["grounding"] = {"expected": {"object": ["O999"]}}
    other["messages"][-1]["content"] = "<answer><O999></answer>"
    selected = select_earlier_retrieval(baseline)
    changed = select_earlier_retrieval(other)
    assert selected == changed
    assert recover_rejected_earlier_retrieval(other) is None


@pytest.mark.parametrize("answer", ["<O999>", "{O1}", "O1</answer><answer>O1"])
def test_recovery_requires_correct_well_formed_grounded_final_answer(answer):
    assert recover_rejected_earlier_retrieval(rejected_trace(answer=answer)) is None


@pytest.mark.parametrize(
    "query,rows",
    [
        (TASK, [{"id": "O999"}]),
        ("MATCH (o:Object {class:'table'}) RETURN o.nodeSymbol AS id", [{"id": "O999"}]),
        (
            "MATCH (a:Object), (b:Object) RETURN a.nodeSymbol AS a, b.nodeSymbol AS b",
            [{"a": "O1", "b": "O999"}],
        ),
    ],
)
def test_newer_wrong_or_ambiguous_task_retrieval_is_never_bypassed(query, rows):
    trace = rejected_trace(
        [
            (TASK, [{"id": "O1"}]),
            (query, rows),
            (DIAGNOSTIC, [{"class": "chair"}]),
        ]
    )
    assert recover_rejected_earlier_retrieval(trace) is None


def test_latest_task_retrieval_is_selected_even_when_older_one_matches_gold():
    trace = rejected_trace(
        [
            (TASK, [{"id": "O1"}]),
            (TASK, [{"id": "O999"}]),
            (DIAGNOSTIC, [{"class": "chair"}]),
        ]
    )
    selected = select_earlier_retrieval(trace)
    assert selected.execution.call_index == 2
    assert selected.execution.rows == [{"id": "O999"}]
    assert recover_rejected_earlier_retrieval(trace) is None


def test_unknown_later_query_is_not_a_diagnostic_just_because_decoding_fails():
    trace = rejected_trace(
        [
            (TASK, [{"id": "O1"}]),
            ("RETURN unsupported(o) AS x", [{"x": "O999"}]),
        ]
    )
    assert select_earlier_retrieval(trace) is None


def test_unknown_earlier_query_cannot_prove_an_empty_retrieval():
    trace = rejected_trace(
        [
            (
                "MATCH (o:Object) WHERE unsupportedNormalization(o.class) CONTAINS 'chair' "
                "RETURN o.nodeSymbol AS id",
                [],
            ),
            (DIAGNOSTIC, [{"class": "table"}]),
        ],
        answer="<>",
    )
    trace["real_solution"] = "<>"
    assert recover_rejected_earlier_retrieval(trace) is None


@pytest.mark.parametrize(
    "reason", ["max_tool_calls_exceeded", "static_issues", "multiple_tool_calls"]
)
def test_recovery_does_not_salvage_ineligible_failures(reason):
    trace = rejected_trace()
    trace["reason"] = reason
    assert recover_rejected_earlier_retrieval(trace) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t["messages"][-2].update(content=json.dumps({"ok": False, "error": "failed"})),
        lambda t: t["messages"][-2].update(tool_call_id="wrong"),
        lambda t: t["messages"][-2].update(content="not json"),
        lambda t: t["messages"][-2].update(
            content=json.dumps({"ok": True, "rows": [], "rows_truncated": True})
        ),
        lambda t: t["messages"][-2].update(
            content=json.dumps({"ok": True, "rows": [], "rows_omitted": True})
        ),
        lambda t: t["messages"].pop(-2),
        lambda t: t["messages"][-3]["tool_calls"].append(
            deepcopy(t["messages"][-3]["tool_calls"][0])
        ),
    ],
)
def test_incomplete_or_mismatched_tool_evidence_blocks_recovery(mutate):
    trace = rejected_trace()
    mutate(trace)
    assert recover_rejected_earlier_retrieval(trace) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("grounding_tool_call_index", 2),
        ("grounding_tool_call_index", True),
        ("diagnostic_tool_call_indices", []),
        ("retrieval_evidence_policy", "unknown"),
        ("seed_gold_cypher", "MATCH (o:Object) RETURN o.nodeSymbol AS id"),
    ],
)
def test_saved_metadata_cannot_override_recomputed_selection(field, value):
    trace = rejected_trace()
    # Use input inspection, which needs the original task reference to prove intent.
    trace["metadata"]["seed_gold_cypher"] = (
        "MATCH (a:Object {nodeSymbol:'O4'})-[:OBJECT_CONNECTED]-(b:Object) "
        "RETURN b.nodeSymbol AS id"
    )
    query = "MATCH (a:Object {nodeSymbol:'O4'}) RETURN a.nodeSymbol AS id"
    trace["messages"] = conversation([(TASK, [{"id": "O1"}]), (query, [{"id": "O4"}])])
    recovered = recover_rejected_earlier_retrieval(trace)
    assert recovered is not None
    recovered.metadata[field] = value
    assert saved_execution(recovered) is None
    assert saved_result(recovered.model_dump()) is None
    assert recompute_trace_validation(recovered)["result_validation"]["status"] == "not_checked"


def test_deleting_diagnostic_proof_does_not_fall_back_to_cached_validation():
    recovered = recover_rejected_earlier_retrieval(rejected_trace())
    recovered.messages[-3].tool_calls[0].function.arguments["cypher_string"] = TASK
    assert saved_execution(recovered) is None
    assert saved_result(recovered.model_dump()) is None


def test_student_report_uses_its_generated_query_not_source_trace_recovery_metadata():
    recovered = recover_rejected_earlier_retrieval(rejected_trace())
    report = {
        **recovered.model_dump(),
        "tool_call_check": {
            "generated_cypher": TASK,
            "validation": {"executable": True, "rows": [{"id": "O999"}]},
        },
    }
    assert saved_result(report) == (TASK, [{"id": "O999"}], None)


def test_empty_retrieval_can_be_recovered_without_discarding_the_diagnostic_rows():
    trace = rejected_trace([(TASK, []), (DIAGNOSTIC, [{"class": "table"}])], answer="<>")
    trace["real_solution"] = "<>"
    recovered = recover_rejected_earlier_retrieval(trace)
    assert recovered is not None
    assert recovered.cypher_solution == "<>"
    assert recompute_trace_validation(recovered)["answer_grounded"] is True


def test_input_checks_and_bounded_inventories_can_follow_a_ranked_retrieval():
    task = (
        "MATCH (a:Object {nodeSymbol:'O4'}), (b:Object {class:'chair'}) "
        "RETURN b.nodeSymbol AS id ORDER BY point.distance(a.center,b.center) LIMIT 1"
    )
    trace = rejected_trace(
        [
            (task, [{"id": "O1"}]),
            ("MATCH (a:Object {nodeSymbol:'O4'}) RETURN a.nodeSymbol AS id", [{"id": "O4"}]),
            ("MATCH (o:Object) RETURN o.nodeSymbol AS id LIMIT 5", [{"id": "O999"}]),
        ]
    )
    trace["metadata"]["seed_gold_cypher"] = task
    recovered = recover_rejected_earlier_retrieval(trace)
    assert recovered is not None
    assert recovered.metadata["grounding_tool_call_index"] == 1
    assert recovered.metadata["diagnostic_tool_call_indices"] == [2, 3]


def test_inventory_that_is_the_task_cannot_be_skipped_as_a_diagnostic():
    task = "MATCH (o:Object) RETURN o.nodeSymbol AS id LIMIT 5"
    trace = rejected_trace([(task, [{"id": "O1"}]), (task, [{"id": "O999"}])])
    trace["metadata"]["seed_gold_cypher"] = task
    assert select_earlier_retrieval(trace) is None


def test_pairing_uses_call_identity_even_with_repeated_queries_and_empty_rows():
    trace = rejected_trace([(TASK, []), (TASK, []), (DIAGNOSTIC, [])], answer="<>")
    trace["real_solution"] = "<>"
    selected = select_earlier_retrieval(trace)
    assert selected.execution.call_index == 2
    assert [e.call_id for e in tool_executions(trace["messages"])] == ["call1", "call2", "call3"]
    trace["messages"][4]["tool_call_id"] = "call1"
    assert select_earlier_retrieval(trace) is None


def test_a_correct_alternative_answer_still_needs_to_match_the_selected_execution():
    query = "MATCH (o:Object) RETURN o.class AS class LIMIT 1"
    trace = rejected_trace(
        [
            (query, [{"class": "chair"}]),
            ("MATCH (o:Object) RETURN count(o) AS count", [{"count": 2}]),
        ],
        answer="<table>",
    )
    trace["metadata"]["seed_gold_cypher"] = query
    trace["real_solution"] = "<chair>"
    trace["validation_contract"] = {
        "result": {"kind": "set", "type": "string", "property": "class"},
        "answer": {
            "formats": ["set[str]"],
            "empty": "<>",
            "comparison": {
                "mode": "top_k_with_ties",
                "limit": 1,
                "scores": {"chair": 2, "table": 2},
            },
        },
    }
    assert select_earlier_retrieval(trace).execution.call_index == 1
    assert recover_rejected_earlier_retrieval(trace) is None


def test_failure_before_the_selected_retrieval_is_preserved_and_does_not_block_recovery():
    trace = rejected_trace(
        [
            ("malformed Cypher", []),
            (TASK, [{"id": "O1"}]),
            (DIAGNOSTIC, [{"class": "chair"}]),
        ]
    )
    trace["messages"][2]["content"] = json.dumps({"ok": False, "error_type": "static_issues"})
    recovered = recover_rejected_earlier_retrieval(trace)
    assert recovered is not None
    assert recovered.metadata["grounding_tool_call_index"] == 2
    assert recovered.metadata["diagnostic_tool_call_indices"] == [3]
    assert recovered.model_dump(exclude_none=True)["messages"] == trace["messages"]
    assert recompute_trace_validation(recovered)["answer_grounded"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": True, "rows": [], "columns": [{"invalid": "column"}]},
        {"ok": True, "rows": [], "columns": "invalid"},
    ],
)
def test_invalid_column_metadata_cannot_become_recovery_evidence(payload):
    trace = rejected_trace()
    trace["messages"][-2]["content"] = json.dumps(payload)
    assert recover_rejected_earlier_retrieval(trace) is None


def test_wrong_response_tool_name_blocks_recovery():
    trace = rejected_trace()
    trace["messages"][-2]["tool_name"] = "another_tool"
    assert recover_rejected_earlier_retrieval(trace) is None


def test_json_string_arguments_are_paired_with_their_tool_response():
    trace = rejected_trace()
    for message in trace["messages"]:
        for call in message.get("tool_calls") or []:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    assert select_earlier_retrieval(trace).execution.call_index == 1


def test_explicit_answer_adapter_is_used_without_reassigning_diagnostic_aliases():
    trace = rejected_trace()
    trace["validation_contract"]["result"].update(
        mapping="explicit",
        element={
            "type": "node_symbol",
            "layers": ["Object"],
            "property": "nodeSymbol",
            "column": "id",
        },
    )
    recovered = recover_rejected_earlier_retrieval(trace)
    assert recovered is not None
    assert recompute_trace_validation(recovered)["answer_grounded"] is True
    trace["messages"][-3]["tool_calls"][0]["function"]["arguments"]["cypher_string"] = (
        "MATCH (o:Object) RETURN o.class AS id"
    )
    trace["messages"][-2]["content"] = json.dumps({"ok": True, "rows": [{"id": "chair"}]})
    assert recover_rejected_earlier_retrieval(trace) is None
