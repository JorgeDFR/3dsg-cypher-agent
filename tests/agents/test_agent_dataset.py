from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

import cypher_3dsg_agent_distillation.agents.agent_dataset as agent_dataset_module
from cypher_3dsg_agent_distillation.agents.agent_dataset import (
    assert_no_split_leakage,
    cap_by_task_type,
    consolidate_cross_task_cypher,
    grouped_split_records,
    read_student_trace_records,
    read_student_trace_records_with_counts,
    split_command,
    split_report_lines,
    split_report_payload,
)
from cypher_3dsg_agent_distillation.agents.agent_records import AgentMessage, AgentTraceRecord
from cypher_3dsg_agent_distillation.agents.teacher_optimizations import (
    repair_rejected_invalid_answer_trace,
)
from cypher_3dsg_agent_distillation.core.io import write_jsonl
from tests.helpers import answerable_trace


def make_record(uid: str, task_type: str, cypher: str) -> AgentTraceRecord:
    return AgentTraceRecord.model_validate(
        answerable_trace(
            uid=uid,
            task_type=task_type,
            gold_cypher=cypher,
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {"cypher_string": cypher},
                            }
                        }
                    ],
                },
                {"role": "tool", "tool_name": "run_cypher_query", "content": "[]"},
                {"role": "assistant", "content": "<answer>3</answer>"},
            ],
        )
    )


def test_scene_aware_split_option_reports_not_implemented(monkeypatch, capsys):
    monkeypatch.setattr(
        agent_dataset_module,
        "load_data_config",
        lambda *_args, **_kwargs: {"split": {"split_by_scene": True}},
    )

    split_command(
        SimpleNamespace(
            data_config="data.yaml",
            dataset_version=None,
            split_by_scene=False,
        )
    )

    assert capsys.readouterr().out.strip() == ("Scene-aware splitting is not implemented yet.")


def rejected_invalid_answer_trace(*, two_tool_calls: bool = False, **overrides):
    cypher = "MATCH (o:Object) RETURN o.nodeSymbol AS object"
    messages = [{"role": "user", "content": "question"}]
    if two_tool_calls:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "run_cypher_query",
                                "arguments": {"cypher_string": "MATCH broken"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "tool_name": "run_cypher_query",
                    "content": '{"ok": false}',
                },
            ]
        )
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": cypher},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "tool_name": "run_cypher_query",
                "content": '{"ok": true, "rows": [{"object": "O1"}]}',
            },
            {"role": "assistant", "content": "<answer>{O1}</answer>"},
        ]
    )
    payload = {
        "uid": "rejected_teacher",
        "seed_uid": "seed_rejected",
        "scene_id": "scene",
        "query_type": "objects_in_place_symbol",
        "question": "Which object is in place p1?",
        "cypher": cypher,
        "real_solution": "O1",
        "cypher_solution": "O1",
        "final_answer": "{O1}",
        "validation": {
            "executable": True,
            "solution_match": False,
            "cypher_solution_match": True,
            "final_answer_match": None,
            "final_answer_cypher_match": False,
        },
        "messages": messages,
        "reason": "invalid_answer_format",
    }
    payload.update(overrides)
    return payload


def test_assert_no_split_leakage_rejects_repeated_cypher():
    record = make_record("one", "count", "MATCH (n) RETURN count(n) AS count")
    leaked = make_record("two", "count", "MATCH (n) RETURN count(n) AS count")

    with pytest.raises(ValueError, match="Split leakage"):
        assert_no_split_leakage({"train": [record], "test": [leaked]})


def test_cross_family_equivalent_cypher_is_consolidated_into_one_split():
    cypher = "MATCH (o:Object) RETURN o.nodeSymbol AS object"
    qa_record = make_record("qa", "object_lookup", cypher)
    pddl_record = make_record("pddl", "object_lookup", cypher).model_copy(
        update={"task_family": "cypher_pddl", "output_type": "PDDL"}
    )
    splits = {"train": [qa_record], "eval": [], "test": [pddl_record]}

    consolidate_cross_task_cypher(splits)

    occupied = [name for name, records in splits.items() if records]
    assert occupied == ["train"]
    assert {record.uid for record in splits["train"]} == {"qa", "pddl"}


def test_per_task_cap_is_independent_across_task_families():
    qa_record = make_record("qa", "lookup", "RETURN 1 AS value")
    pddl_record = make_record("pddl", "lookup", "RETURN 2 AS value").model_copy(
        update={"task_family": "cypher_pddl", "output_type": "PDDL"}
    )

    assert cap_by_task_type([qa_record, pddl_record], 1) == [qa_record, pddl_record]


def test_grouped_split_keeps_challenge_task_out_of_main_splits():
    records = [
        make_record("one", "count", "MATCH (n) RETURN 1 AS count"),
        make_record("two", "lookup", "MATCH (n) RETURN 2 AS count"),
        make_record("three", "place_hops", "MATCH (n) RETURN 3 AS count"),
    ]

    train, eval_records, test, challenge = grouped_split_records(
        records,
        train_size=0.7,
        eval_size=0.15,
        test_size=0.15,
        seed=1,
        challenge_task_types={"place_hops"},
    )

    assert [record.uid for record in challenge] == ["three"]
    assert "three" not in {record.uid for record in train + eval_records + test}


def test_grouped_split_assigns_every_challenge_task_record_to_challenge():
    records = [
        make_record(
            f"hops-{index}",
            "place_hops",
            f"MATCH (n) RETURN {index} AS count",
        )
        for index in range(20)
    ]

    train, eval_records, test, challenge = grouped_split_records(
        records,
        train_size=0.8,
        eval_size=0.1,
        test_size=0.1,
        seed=7,
        challenge_task_types={"place_hops"},
    )

    assert len(challenge) == 20
    assert not train
    assert not eval_records
    assert not test
    assert {record.uid for record in challenge}.isdisjoint(
        record.uid for record in train + eval_records + test
    )


def test_grouped_split_keeps_identical_cypher_from_multiple_scenes_together():
    shared_cypher = "MATCH (o:Object) RETURN count(o) AS count"
    records = [
        make_record(f"shared-{scene}", "count", shared_cypher).model_copy(
            update={
                "scene_id": scene,
                "split_group": f"{scene}:count:shared",
            }
        )
        for scene in ("scene-1", "scene-2")
    ]
    records.extend(
        make_record(
            f"unique-{index}",
            "count",
            f"MATCH (o:Object) RETURN {index} AS count",
        )
        for index in range(18)
    )

    train, eval_records, test, _challenge = grouped_split_records(
        records,
        train_size=0.8,
        eval_size=0.1,
        test_size=0.1,
        seed=42,
        scene_order=["scene-1", "scene-2"],
    )

    locations = [
        name
        for name, split_records in {
            "train": train,
            "eval": eval_records,
            "test": test,
        }.items()
        if any(record.gold_cypher == shared_cypher for record in split_records)
    ]
    assert len(locations) == 1


def test_grouped_split_rejects_split_sizes_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        grouped_split_records(
            [],
            train_size=0.5,
            eval_size=0.1,
            test_size=0.1,
            seed=1,
        )


def test_grouped_split_is_balanced_by_task_type():
    records = [
        make_record(
            f"{task_type}-{index}",
            task_type,
            f"RETURN '{task_type}' AS task, {index} AS count",
        )
        for task_type in ("count", "lookup")
        for index in range(50)
    ]

    train, eval_records, test, _challenge = grouped_split_records(
        records,
        train_size=0.8,
        eval_size=0.1,
        test_size=0.1,
        seed=1,
    )

    assert Counter(record.task_type for record in train) == {"count": 40, "lookup": 40}
    assert Counter(record.task_type for record in eval_records) == {"count": 5, "lookup": 5}
    assert Counter(record.task_type for record in test) == {"count": 5, "lookup": 5}


def test_grouped_split_balances_pddl_types_that_share_cypher_queries():
    records = [
        make_record(
            f"{task_type}-{index}",
            task_type,
            f"MATCH (o:Object {{nodeSymbol: 'O{index}'}}) RETURN o.nodeSymbol AS object",
        ).model_copy(update={"task_family": "cypher_pddl", "output_type": "PDDL"})
        for task_type in ("pddl_visit_object_symbol", "pddl_end_at_object_symbol")
        for index in range(24)
    ]

    train, eval_records, test, _challenge = grouped_split_records(
        records,
        train_size=0.8,
        eval_size=0.1,
        test_size=0.1,
        seed=42,
    )

    assert Counter(record.task_type for record in train) == {
        "pddl_visit_object_symbol": 19,
        "pddl_end_at_object_symbol": 19,
    }
    assert Counter(record.task_type for record in eval_records) == {
        "pddl_visit_object_symbol": 3,
        "pddl_end_at_object_symbol": 3,
    }
    assert Counter(record.task_type for record in test) == {
        "pddl_visit_object_symbol": 2,
        "pddl_end_at_object_symbol": 2,
    }
    assert_no_split_leakage({"train": train, "eval": eval_records, "test": test})


def test_split_assignments_do_not_depend_on_teacher_query_wording():
    synthetic = [
        make_record(
            f"{task}-{group}-{variant}", task,
            f"MATCH (o:Object {{class: 'class{group}'}}) RETURN o.nodeSymbol AS object",
        )
        for task in ("lookup", "inspect")
        for group in range(10)
        for variant in range(2)
    ]
    real = [
        record.model_copy(update={
            "gold_cypher": f"MATCH (node:Object) WHERE node.class = 'class{index // 2 % 10}' "
                           f"RETURN node.nodeSymbol AS answer{index}",
            "metadata": {"seed_gold_cypher": record.gold_cypher},
        })
        for index, record in enumerate(synthetic)
    ]

    def assignments(records):
        parts = grouped_split_records(
            records, train_size=0.8, eval_size=0.1, test_size=0.1, seed=42,
        )
        assert_no_split_leakage(dict(zip(("train", "eval", "test"), parts[:3], strict=True)))
        return [[record.uid for record in part] for part in parts]

    assert assignments(real) == assignments(synthetic)


def test_split_connects_both_explicit_groups_and_seed_query_identities():
    from cypher_3dsg_agent_distillation.agents.agent_dataset import split_record_groups

    one = make_record("one", "lookup", "RETURN 'O1'")
    two = make_record("two", "lookup", "RETURN 'O2'").model_copy(
        update={"split_group": one.split_group},
    )
    three = make_record("three", "inspect", two.gold_cypher)
    assert [[record.uid for record in group] for group in split_record_groups(
        [one, two, three],
    )] == [["one", "two", "three"]]


def test_grouped_split_represents_low_count_single_group_task_in_eval_and_test():
    records = [
        make_record(f"inventory-{index}", "inventory", "MATCH (n) RETURN n") for index in range(4)
    ]

    with pytest.warns(UserWarning, match="low-count task remains represented"):
        train, eval_records, test, _challenge = grouped_split_records(
            records,
            train_size=0.8,
            eval_size=0.1,
            test_size=0.1,
            seed=1,
        )

    assert [len(train), len(eval_records), len(test)] == [2, 1, 1]


def test_grouped_split_can_suppress_leakage_warnings(recwarn):
    records = [
        make_record(f"inventory-{index}", "inventory", "MATCH (n) RETURN n") for index in range(4)
    ]

    train, eval_records, test, _challenge = grouped_split_records(
        records,
        train_size=0.5,
        eval_size=0.25,
        test_size=0.25,
        seed=1,
        warn_on_leakage=False,
    )

    assert list(recwarn) == []
    assert [len(train), len(eval_records), len(test)] == [2, 1, 1]


def test_grouped_split_breaks_minimum_groups_to_reach_exact_per_type_counts():
    records = [
        make_record(f"bbox-{group}-{variant}", "bbox", f"RETURN {group} AS bbox")
        for group in range(25)
        for variant in range(2)
    ]

    with pytest.warns(UserWarning, match="meet the exact per-type counts"):
        train, eval_records, test, _challenge = grouped_split_records(
            records,
            train_size=0.8,
            eval_size=0.1,
            test_size=0.1,
            seed=1,
        )

    assert [len(train), len(eval_records), len(test)] == [40, 5, 5]


@pytest.mark.parametrize(
    "field,value",
    [
        ("reasoning", "This malformed answer is the right format."),
        ("reasoning_content", "Use this answer."),
        ("thinking", "Use this answer."),
        ("reasoning_details", [{"type": "reasoning.summary", "summary": "Use this answer."}]),
        ("reasoning_details", [{"type": "reasoning.encrypted", "data": "opaque"}]),
        ("content", "<think>Use this answer.</think><answer>{O1}</answer>"),
        ("content", "<think>Unfinished reasoning"),
    ],
)
def test_answer_replacement_preserves_reasoning(field, value):
    rejected = rejected_invalid_answer_trace()
    rejected["messages"][-1][field] = value
    original = deepcopy(rejected)
    repaired = repair_rejected_invalid_answer_trace(rejected)
    assert repaired is not None
    assert repaired.final_answer_text == "<answer>O1</answer>"
    assert repaired.metadata["final_answer_repaired"]
    if field == "content":
        assert repaired.messages[-1].reasoning == (
            "Unfinished reasoning" if "Unfinished" in value else "Use this answer."
        )
    else:
        assert repaired.messages[-1].model_dump()[field] == value
    assert rejected == original
    assert [m.model_dump(exclude_none=True) for m in repaired.messages[:-1]] == [
        AgentMessage.model_validate(m).model_dump(exclude_none=True)
        for m in original["messages"][:-1]
    ]


def test_answer_replacement_preserves_tool_stage_reasoning():
    rejected = rejected_invalid_answer_trace(two_tool_calls=True)
    rejected["messages"][1]["reasoning"] = "Try a query."
    repaired = repair_rejected_invalid_answer_trace(rejected)
    assert repaired is not None
    assert repaired.messages[1].reasoning == "Try a query."
    assert len(repaired.tool_calls) == 2


def contract_format_rejection():
    rejected = rejected_invalid_answer_trace(cypher=None, cypher_solution=None)
    rejected["validation_contract"] = {
        "result": {"kind": "scalar", "type": "node_symbol", "layers": ["Object"]},
        "answer": {"formats": ["str"]},
    }
    rejected["validation"].update({
        "rows": [{"object": "O1"}], "columns": ["object"],
        "final_answer_format_valid": False,
        "answer_check": {"status": "invalid_answer_format", "format_valid": False, "match": False},
    })
    rejected["messages"][-1]["reasoning"] = "Return the retrieved object."
    return rejected


def test_answer_repair_uses_current_contract_evidence_and_refreshes_validation():
    from cypher_3dsg_agent_distillation.agents.agent_evaluation import recompute_trace_validation

    repaired = repair_rejected_invalid_answer_trace(contract_format_rejection())
    assert repaired is not None
    assert repaired.messages[-1].reasoning == "Return the retrieved object."
    assert repaired.final_answer == "O1"
    assert repaired.validation.final_answer_format_valid is True
    assert repaired.validation.answer_check["status"] == "pass"
    assert repaired.validation.grounding_check["match"] is True
    assert recompute_trace_validation(repaired)["answer_grounded"] is True


@pytest.mark.parametrize("broken", ["wrong_rows", "wrong_call_id", "truncated_rows"])
def test_answer_repair_requires_verified_contract_evidence(broken):
    rejected = contract_format_rejection()
    if broken == "wrong_rows":
        rejected["validation"]["rows"] = [{"object": "O2"}]
    elif broken == "wrong_call_id":
        rejected["messages"][-2]["tool_call_id"] = "different_call"
    else:
        rejected["validation"]["rows_truncated"] = True
    assert repair_rejected_invalid_answer_trace(rejected) is None


def test_repair_rejected_invalid_answer_trace_replaces_only_final_answer():
    repaired = repair_rejected_invalid_answer_trace(rejected_invalid_answer_trace())

    assert repaired is not None
    assert repaired.uid == "rejected_teacher_repaired"
    assert repaired.real_solution == "O1"
    assert repaired.cypher_solution == "O1"
    assert repaired.final_answer == "O1"
    assert repaired.final_answer_text == "<answer>O1</answer>"
    assert repaired.messages[-1].content == "<answer>O1</answer>"
    assert repaired.validation.solution_match
    assert repaired.validation.final_answer_match
    assert repaired.metadata["final_answer_repaired"]
    assert repaired.metadata["original_final_answer"] == "{O1}"


@pytest.mark.parametrize("with_reasoning", [False, True])
def test_repair_rejected_invalid_answer_trace_requires_verified_cypher_solution(with_reasoning):
    rejected = rejected_invalid_answer_trace()
    if with_reasoning:
        rejected["messages"][-1]["reasoning"] = "Return the retrieved object."
    assert (
        repair_rejected_invalid_answer_trace(
            {**rejected, "reason": "solution_mismatch"}
        )
        is None
    )
    assert (
        repair_rejected_invalid_answer_trace({**rejected, "cypher_solution": "O2"})
        is None
    )


def test_student_trace_reader_loads_generation_repairs_without_scanning_rejections(tmp_path):
    accepted_path = tmp_path / "teacher_traces.jsonl"
    rejected_path = tmp_path / "rejected.json"
    repaired = repair_rejected_invalid_answer_trace(
        rejected_invalid_answer_trace(two_tool_calls=True)
    )
    assert repaired is not None
    write_jsonl([answerable_trace(uid="accepted"), repaired.model_dump()], accepted_path)
    # Splitting must not inspect or repair rejects, even if this artifact is unreadable.
    rejected_path.write_text("invalid JSON", encoding="utf-8")
    config = {
        "teacher_traces_jsonl_path": str(accepted_path),
        "rejected_traces_path": str(rejected_path),
    }
    records, source_counts = read_student_trace_records_with_counts(config)
    assert [record.uid for record in read_student_trace_records(config)] == [
        "accepted", "rejected_teacher_repaired",
    ]
    assert source_counts["accepted_teacher_traces"] == 2
    assert source_counts["repaired_invalid_answer_format"] == 1
    assert records[-1].metadata["final_answer_repaired"] is True
    assert len(records[-1].tool_calls) == 2


def test_split_report_lines_include_split_sizes_and_optimization_counts():
    train = [make_record("train-one", "count", "MATCH (n) RETURN 1 AS count")]
    eval_records = [make_record("eval-one", "lookup", "MATCH (n) RETURN 2 AS count")]
    test = []
    challenge = [make_record("challenge-one", "place_hops", "MATCH (n) RETURN 3 AS count")]

    lines = split_report_lines(
        source_counts=Counter(
            {
                "accepted_teacher_traces": 2,
                "repaired_invalid_answer_format": 1,
            }
        ),
        cap_dropped_counts=Counter({"count": 1}),
        split_records={
            "train": train,
            "eval": eval_records,
            "test": test,
            "challenge": challenge,
        },
        split_paths={
            "train": "train.jsonl",
            "eval": "eval.jsonl",
            "test": "test.jsonl",
            "challenge": "challenge.jsonl",
        },
    )
    text = "\n".join(lines)

    assert "accepted teacher traces loaded: 2" in text
    assert "generation final-answer repairs included: 1" in text
    assert "rejected traces scanned" not in text
    assert "dropped by max_per_task_type: count=1" in text
    assert "train: 1 traces -> train.jsonl" in text
    assert "test: 0 traces -> test.jsonl" in text
    assert text.count("count=1") == 1
    assert "lookup=1" not in text


def test_split_report_payload_lists_optimized_questions():
    record = AgentTraceRecord.model_validate(
        answerable_trace(
            uid="optimized",
            task_type="object_count",
            metadata={
                "final_answer_repaired": True,
                "tool_call_count": 2,
                "source_trace_uid": "teacher-one",
            },
        )
    )

    report = split_report_payload(
        data_config={
            "scene_id": "scene",
            "teacher_traces_jsonl_path": "teacher.jsonl",
            "rejected_traces_path": "rejected.json",
            "max_per_task_type": None,
            "split": {"seed": 1},
        },
        render_view="ollama_native",
        source_counts=Counter({"accepted_teacher_traces": 1}),
        cap_dropped_counts=Counter(),
        split_records={
            "train": [record],
            "eval": [],
            "test": [],
            "challenge": [],
        },
        split_paths={
            "train": "train.jsonl",
            "eval": "eval.jsonl",
            "test": "test.jsonl",
            "challenge": "challenge.jsonl",
        },
    )

    assert report["metadata"]["optimization_stage"] == "generation"
    assert report["splits"]["train"]["optimized_questions"] == 1
    assert report["splits"]["train"]["scene_counts"] == {"scene": 1}
    assert report["splits"]["train"]["scene_runs"] == 1
    assert report["optimized_questions"] == [
        {
            "uid": "optimized",
            "scene_id": "scene",
            "split": "train",
            "task_type": "object_count",
            "task_family": "cypher_qa",
            "output_type": "SLDP",
            "question": "How many chairs?",
            "split_group": record.split_group,
            "gold_cypher": "MATCH (o:Object) RETURN count(o) AS count",
            "expected_solution": 3,
            "final_answer": None,
            "optimization_kinds": ["repaired_invalid_answer_format"],
            "final_answer_repaired": True,
            "earlier_retrieval_recovered": False,
            "grounding_tool_call_index": None,
            "diagnostic_tool_call_indices": None,
            "tool_call_count": 2,
            "successful_tool_call_index": None,
            "seed_uid": None,
            "source_trace_uid": "teacher-one",
            "source_trace_status": None,
            "source_rejection_reason": None,
            "repair_type": None,
            "original_final_answer": None,
        }
    ]
