import json

from cypher_3dsg_agent_distillation.io import write_jsonl
from cypher_3dsg_agent_distillation.workflow_report import (
    collect_workflow_report_data,
    render_record_detail,
    render_workflow_report,
    scene_template_sample_text,
    validation_example_rows,
    workflow_report_output_path,
)


def minimal_data_config(tmp_path):
    scene_dir = tmp_path / "data" / "3dsg" / "scene"
    reports_dir = scene_dir / "reports"
    splits_dir = scene_dir / "splits"
    reports_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    return {
        "scene_id": "scene",
        "scene_path": str(tmp_path / "scene.json"),
        "scene_template_values_path": str(scene_dir / "scene_template_values.yaml"),
        "raw_questions_path": str(scene_dir / "generated_questions.yaml"),
        "seed_records_jsonl_path": str(scene_dir / "cypher_seed_records.jsonl"),
        "teacher_traces_jsonl_path": str(scene_dir / "traces" / "teacher_traces.jsonl"),
        "accepted_traces_path": str(scene_dir / "traces" / "accepted.json"),
        "rejected_traces_path": str(scene_dir / "traces" / "rejected.json"),
        "splits": {
            "train": str(splits_dir / "train.jsonl"),
            "eval": str(splits_dir / "eval.jsonl"),
            "test": str(splits_dir / "test.jsonl"),
            "challenge": str(splits_dir / "challenge.jsonl"),
        },
        "test_report_path": str(reports_dir / "test.agent.report.json"),
    }


def write_minimal_artifacts(config):
    scene_values_path = config["scene_template_values_path"]
    generated_questions_path = config["raw_questions_path"]
    traces_dir = config["teacher_traces_jsonl_path"].rsplit("/", 1)[0]
    question = {
        "uid": "q1",
        "query_type": "object_class_count",
        "question": "How many chairs?",
        "gold_cypher": "MATCH (o:Object {class: 'chair'}) RETURN count(o) AS count",
        "solution": "<\n  O1,\n  O2\n>",
        "final_answer": "<answer><\n  O1,\n  O2\n></answer>",
        "tags": ["count"],
        "messages": [
            {"role": "user", "content": "Line one\\nLine two"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": "MATCH (o:Object) RETURN o"},
                        }
                    }
                ],
            },
            {"role": "tool", "tool_name": "run_cypher_query", "content": "[]"},
            {"role": "assistant", "content": "<answer><O1, O2></answer>"},
        ],
    }
    optimized_split_record = {
        **question,
        "tags": [],
        "task_type": "object_class_count",
        "metadata": {
            "student_optimized": True,
            "final_answer_repaired": True,
            "raw_tool_call_count": 2,
            "dropped_tool_call_count": 1,
            "seed_uid": "q1",
        },
    }
    accepted_trace = {
        key: value for key, value in question.items() if key != "gold_cypher"
    } | {
        "seed_uid": "q1",
        "cypher": "MATCH (o:Object) RETURN o",
    }
    rejected_trace_without_answer = {
        key: value
        for key, value in question.items()
        if key not in {"gold_cypher", "final_answer"}
    } | {
        "uid": "q1_rejected",
        "seed_uid": "q1",
        "cypher": "MATCH (o:Object) RETURN o",
        "final_answer": None,
        "reason": "invalid_answer_format",
    }

    with open(scene_values_path, "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "scene_summary:",
                    "  nodes:",
                    "    Object: 3",
                    "  edges: 2",
                    "values:",
                    "  object_classes:",
                    "    - chair",
                    "    - table",
                    "  object_radius_cases:",
                    "    - class: chair",
                    "      radius: 2",
                    "      point:",
                    "        x: 1",
                    "        y: 2",
                    "        z: 3",
                    "    - class: table",
                    "      radius: 5",
                    "      point:",
                    "        x: 4",
                    "        y: 5",
                    "        z: 6",
                ]
            )
        )
    with open(generated_questions_path, "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "questions:",
                    "- uid: q1",
                    "  query_type: object_class_count",
                    "  question: How many chairs?",
                    '  gold_cypher: "MATCH (o:Object {class: \'chair\'}) RETURN count(o) AS count"',
                    "  solution: |",
                    "    <",
                    "      O1,",
                    "      O2",
                    "    >",
                    "  tags:",
                    "    - count",
                ]
            )
        )

    write_jsonl([question], config["seed_records_jsonl_path"])
    write_jsonl([question], config["teacher_traces_jsonl_path"])
    for split, path in config["splits"].items():
        write_jsonl([optimized_split_record] if split == "train" else [], path)

    traces_path = config["accepted_traces_path"].rsplit("/", 1)[0]
    assert traces_dir == traces_path
    accepted_path = config["accepted_traces_path"]
    rejected_path = config["rejected_traces_path"]
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump([accepted_trace], handle)
    with open(rejected_path, "w", encoding="utf-8") as handle:
        json.dump([rejected_trace_without_answer], handle)

    reports_dir = config["test_report_path"].rsplit("/", 1)[0]
    with open(f"{reports_dir}/split.report.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "optimized_questions": [
                    {
                        "uid": "q1",
                        "split": "train",
                        "task_type": "object_class_count",
                        "question": "How many chairs?",
                        "optimization_kinds": ["pruned_failed_tool_attempts"],
                    }
                ]
            },
            handle,
        )
    with open(f"{reports_dir}/student.eval-loss.report.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": {
                    "examples": 1,
                    "tokens": 10,
                    "eval_loss": 0.2,
                    "perplexity": 1.22,
                    "eval_entropy": 0.3,
                    "eval_mean_token_accuracy": 0.9,
                    "splits": {
                        "train": {
                            "examples": 1,
                            "tokens": 10,
                            "eval_loss": 0.2,
                            "perplexity": 1.22,
                            "eval_entropy": 0.3,
                            "eval_mean_token_accuracy": 0.9,
                        }
                    },
                }
            },
            handle,
        )
    with open(
        f"{reports_dir}/test.student.raw-generation.report.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "summary": {
                    "examples": 1,
                    "final_answer_from_trace": {
                        "evaluated": 1,
                        "metrics": {"final_answer_match": 1},
                        "rates": {"final_answer_match": 1.0},
                        "statuses": {"pass": 1},
                    },
                    "tool_call_from_prompt": {
                        "evaluated": 1,
                        "metrics": {
                            "tool_call_parse_success": 1,
                            "executable": 1,
                            "cypher_solution_match": 1,
                        },
                        "rates": {
                            "tool_call_parse_success": 1.0,
                            "executable": 1.0,
                            "cypher_solution_match": 1.0,
                        },
                        "statuses": {"pass": 1},
                    },
                },
                "records": [
                    {
                        "uid": "q1",
                        "query_type": "object_class_count",
                        "question": "How many chairs?",
                        "expected": {
                            "cypher": (
                                "MATCH (o:Object {class: 'chair'}) "
                                "RETURN count(o) AS count"
                            ),
                            "cypher_solution": "<\n  O1,\n  O2\n>",
                            "final_answer": "<answer>2</answer>",
                        },
                        "generated": {
                            "cypher": (
                                "MATCH (o:Object {class: 'chair'}) "
                                "RETURN count(o) AS count"
                            ),
                            "cypher_solution": "<\\n  O1,\\n  O2\\n>",
                            "cypher_tool_output": [{"count": 2}],
                            "final_answer": "<answer>2</answer>",
                            "final_answer_value": "2",
                            "end_to_end_final_answer": "<answer>2</answer>",
                        },
                        "generation_debug": {
                            "cypher_generation": {
                                "rendered_prompt": (
                                    "<|im_start|>system\n# Tools\n"
                                    "run_cypher_query<|im_end|>"
                                ),
                                "raw_completion": "<tool_call>...</tool_call>",
                            },
                            "final_answer_generation": {
                                "rendered_prompt": "<|im_start|>user\nHow many chairs?",
                                "raw_completion": "<answer>2</answer><|im_end|>",
                            },
                            "end_to_end_generation": {
                                "turns": [
                                    {
                                        "turn": 1,
                                        "rendered_prompt": (
                                            "<|im_start|>user\nHow many chairs?"
                                        ),
                                        "raw_completion": "<tool_call>...</tool_call>",
                                    },
                                    {
                                        "turn": 2,
                                        "rendered_prompt": (
                                            "<|im_start|>tool\n"
                                            '{"ok": true, "rows": [{"count": 2}]}'
                                        ),
                                        "raw_completion": (
                                            "<answer>2</answer><|im_end|>"
                                        ),
                                    },
                                ]
                            },
                        },
                        "final_answer_from_trace": {
                            "status": "pass",
                            "final_answer_match": "true",
                        },
                        "tool_call_from_prompt": {
                            "status": "pass",
                            "tool_call_parse_success": True,
                            "executable": True,
                            "cypher_solution_match": "true",
                        },
                        "end_to_end_final_answer": {
                            "status": "pass",
                            "final_answer_match": True,
                            "query_attempts": 1,
                            "retries_used": 0,
                            "max_query_retries": 1,
                            "attempts": [
                                {"turn": 1, "status": "tool_success"},
                                {"turn": 2, "status": "pass"},
                            ],
                        },
                    }
                ],
            },
            handle,
        )
    with open(
        f"{reports_dir}/test.student.raw-generation.report.json",
        encoding="utf-8",
    ) as source:
        with open(
            f"{reports_dir}/test.baseline.raw-generation.report.json",
            "w",
            encoding="utf-8",
        ) as target:
            target.write(source.read())


def test_workflow_report_collects_scene_artifacts_and_renders_tabs(tmp_path):
    config = minimal_data_config(tmp_path)
    write_minimal_artifacts(config)

    data = collect_workflow_report_data(config)
    html = render_workflow_report(data)

    assert data["scene_graph"]["summary"]["nodes"]["Object"] == 3
    assert data["generated_questions"]["by_task_type"] == {"object_class_count": 1}
    assert data["splits"]["optimized_questions"][0]["uid"] == "q1"
    assert data["train"]["aggregate_eval_loss"]["summary"]["examples"] == 1
    assert data["baseline"]["raw_generation_reports"]["test"]["summary"]["examples"] == 1
    validation_record = data["validation"]["raw_generation_reports"]["test"][
        "records"
    ][0]
    _validation_rows, validation_details = validation_example_rows(
        data["validation"]["raw_generation_reports"]
    )
    assert "conversations" not in validation_record
    assert set(validation_record["generation_debug"]) == {
        "cypher_generation",
        "end_to_end_generation",
        "final_answer_generation",
    }
    assert set(validation_record["generation_debug"]["cypher_generation"]) == {
        "rendered_prompt",
        "raw_completion",
    }
    assert len(
        validation_record["generation_debug"]["end_to_end_generation"]["turns"]
    ) == 2
    assert list(validation_details[0]["fields"]) == [
        "uid",
        "split",
        "task_family",
        "output_type",
        "task_type",
        "tool_call_status",
        "final_answer_status",
        "tool_call_parse_success",
        "tool_call_executable",
        "cypher_solution_match",
        "grounding_match",
        "potential_hidden_false_positive",
        "final_answer_domain_valid",
        "final_answer_match",
        "end_to_end_match",
    ]
    assert [group["title"] for group in validation_details[0]["groups"]] == [
        "Cypher Validation",
        "Final Answer Validation",
        "End-to-End Validation",
    ]
    assert "cypher_validation_summary" not in validation_details[0]["groups"][0][
        "blocks"
    ]
    assert "final_answer_validation_summary" not in validation_details[0]["groups"][1][
        "blocks"
    ]
    assert "3D Scene Graph" in html
    assert "Generated Questions" in html
    assert "Trace Outcomes" in html
    assert "Validation Examples" in html
    assert "Baseline Examples" in html
    assert "Artifacts" in html
    assert "Object Classes" in html
    assert "Mesh Place Classes" in html
    assert "Room Classes" in html
    assert "Objects</div>" in html
    assert "Rooms</div>" in html
    assert html.index("Object Classes") < html.index("Template Values")
    assert "0 classes" in html
    scene_section = html[html.index('<section id="scene"') : html.index('<section id="questions"')]
    assert "<pre>chair, table</pre>" in scene_section
    assert (
        "{class: chair, radius: 2, point: {x: 1, y: 2, z: 3}}, "
        "{class: table, radius: 5, point: {x: 4, y: 5, z: 6}}"
        in scene_section
    )
    generated_summary = html[
        html.index("<h2>Generated Questions</h2>") : html.index("<h3>Questions</h3>")
    ]
    assert (
        '<div class="metric-label">Generated Questions</div>'
        '<div class="metric-value">1</div>'
        in generated_summary
    )
    assert "Seed Records" not in generated_summary
    assert "Task Types</div>" in generated_summary
    assert "<h3>Task Type</h3>" in generated_summary
    assert "By Task Type" not in generated_summary
    assert "column-filter" in html
    assert "Filter Task Type" in html
    assert "applyColumnFilters" in html
    assert "Optimized" in html
    assert "Optimization Type" in html
    assert "Initial Tool Calls" in html
    assert "Tool Calls Dropped" in html
    assert '<th data-sort-index="3">Task Type</th>' in html
    assert "100.00%" in html
    assert '<span class="status-pill status-good">True</span>' in html
    assert '<span class="status-pill status-good">pass</span>' not in html
    assert "&lt; O1, O2 &gt;" not in html
    assert "<pre>&lt;O1, O2&gt;</pre>" in html
    assert "How many chairs?" in html
    assert "<h3>Trace Details</h3>" in html
    assert "<h3>Split Question Details</h3>" in html
    assert "<h3>Validation Example Details</h3>" in html
    assert "<h4>Expected Cypher</h4>" in html
    assert "<h4>Generated Cypher</h4>" in html
    assert "<h4>Expected Cypher Solution</h4>" in html
    assert "<h4>Generated Cypher Solution</h4>" in html
    assert (
        "<h4>Expected Cypher Solution</h4><pre>&lt;O1, O2&gt;</pre>" in html
    )
    assert (
        "<h4>Generated Cypher Solution</h4><pre>&lt;O1, O2&gt;</pre>" in html
    )
    assert "<h4>Cypher Tool Output</h4>" in html
    assert "<h4>Expected Final Answer</h4>" in html
    assert "<h4>Generated Final Answer</h4>" in html
    assert "Cypher Generation Conversation" not in html
    assert "Final Answer Generation Conversation" not in html
    assert "Tool Schema" not in html
    assert "Cypher Generation Rendered Prompt" in html
    assert "Final Answer Generation Rendered Prompt" in html
    assert "Turn 1 Rendered Prompt" in html
    assert "Turn 2 Raw Completion" in html
    assert "Cypher Validation</summary>" in html
    assert "Final Answer Validation</summary>" in html
    assert "End-to-End Validation</summary>" in html
    assert html.count('class="validation-detail-group"') >= 6
    assert "<h3>How many chairs?</h3>" not in html
    assert "message-card" in html
    assert "Line one\nLine two" in html
    assert "<h4>Question</h4><pre>How many chairs?</pre>" in html
    assert (
        "<h4>Gold Cypher</h4><pre>"
        "MATCH (o:Object {class: &#x27;chair&#x27;}) RETURN count(o) AS count"
        in html
    )
    assert "<h4>Model Cypher</h4><pre>MATCH (o:Object) RETURN o</pre>" in html
    assert (
        "<h4>Answer</h4><pre>"
        "&lt;answer&gt;&lt;O1, O2&gt;&lt;/answer&gt;</pre>"
        in html
    )
    assert "<h4>Answer</h4><pre></pre>" in html
    questions_section = html[
        html.index('id="questions"') : html.index('id="traces"')
    ]
    questions_table = questions_section[
        questions_section.index("<h3>Questions</h3>") :
        questions_section.index("</table>", questions_section.index("<h3>Questions</h3>"))
    ]
    question_headers = [
        "UID",
        "Task Family",
        "Task Type",
        "Tags",
        "Question",
        "Gold Cypher",
        "Solution",
    ]
    questions_header_row = questions_table[
        questions_table.index("<thead><tr>") :
        questions_table.index("</tr>", questions_table.index("<thead><tr>"))
    ]
    assert all(header in questions_header_row for header in question_headers)
    assert [questions_header_row.index(header) for header in question_headers] == sorted(
        questions_header_row.index(header) for header in question_headers
    )
    assert (
        "<pre>MATCH (o:Object {class: &#x27;chair&#x27;}) "
        "RETURN count(o) AS count</pre>"
        in questions_table
    )
    assert "generated-questions-detail" in html
    assert "<h3>Generated Question Details</h3>" in html
    trace_section = html[html.index('id="traces"') : html.index('id="splits"')]
    trace_table = trace_section[
        trace_section.index("<h3>Trace Outcomes</h3>") : trace_section.index("</table>")
    ]
    assert "Teacher JSONL" not in trace_section
    assert "In Teacher JSONL" not in trace_table
    assert "Seed UID" not in trace_table
    assert "Tool Calls" not in trace_table
    assert "Validation" not in trace_table
    assert "Rejected" in trace_table
    split_section = html[html.index('id="splits"') : html.index('id="train"')]
    assert 'data-sort-value="count">count</td>' in split_section
    assert "<h4>Model Cypher</h4><pre>MATCH (o:Object) RETURN o</pre>" in split_section
    validation_section = html[
        html.index('<section id="validation"') : html.index('<section id="baseline"')
    ]
    cypher_group_start = validation_section.index("<summary>Cypher Validation</summary>")
    final_group_start = validation_section.index(
        "<summary>Final Answer Validation</summary>"
    )
    end_to_end_group_start = validation_section.index(
        "<summary>End-to-End Validation</summary>"
    )
    assert cypher_group_start < validation_section.index("<h4>Expected Cypher</h4>")
    assert validation_section.index("<h4>Cypher Tool Output</h4>") < final_group_start
    assert final_group_start < validation_section.index("<h4>Expected Final Answer</h4>")
    assert (
        validation_section.index("<h4>Final Answer Validation Issues</h4>")
        < end_to_end_group_start
    )
    assert end_to_end_group_start < validation_section.index(
        "<summary>Turn 1 Rendered Prompt</summary>"
    )
    assert end_to_end_group_start < validation_section.index(
        "<h4>Turn 2 Raw Completion</h4>"
    )
    assert "<h4>Test</h4>" in validation_section
    assert ">All</td>" in validation_section
    assert "Final Answer From Trace" not in validation_section
    assert "Tool Call From Prompt" not in validation_section
    detail_headings = [
        "Question",
        "Expected Cypher",
        "Generated Cypher",
        "Cypher Tool Output",
        "Expected Cypher Solution",
        "Generated Cypher Solution",
        "Cypher Validation Issues",
        "Expected Final Answer",
        "Generated Final Answer",
        "Final Answer Validation Issues",
    ]
    heading_positions = [
        validation_section.index(f"<h4>{heading}</h4>")
        for heading in detail_headings
    ]
    assert heading_positions == sorted(heading_positions)
    validation_examples_start = validation_section.index("<h3>Validation Examples</h3>")
    validation_examples_table = validation_section[
        validation_examples_start :
        validation_section.index("</table>", validation_examples_start)
    ]
    validation_headers = [
        "UID",
        "Split",
        "Task Family",
        "Task Type",
        "Question",
        "Tool Call Executable",
        "Cypher Solution Match",
        "Grounding Match",
        "Final Answer Match",
    ]
    assert all(header in validation_examples_table for header in validation_headers)
    assert [validation_examples_table.index(header) for header in validation_headers] == sorted(
        validation_examples_table.index(header) for header in validation_headers
    )
    baseline_section = html[
        html.index('<section id="baseline"') : html.index('<section id="artifacts"')
    ]
    assert "<h2>Baseline</h2>" in baseline_section
    assert "<h3>Base Student Raw Generation Summary</h3>" in baseline_section
    assert "<h3>Baseline Examples</h3>" in baseline_section
    assert "<h3>Baseline Example Details</h3>" in baseline_section
    assert "<h4>Test</h4>" in baseline_section
    assert "Final Answer From Trace" not in baseline_section
    assert "Tool Call From Prompt" not in baseline_section
    assert "font: inherit;" in html
    assert "equal-columns" in html
    assert "Agent Trace Reports" not in html
    assert "Teacher Trace JSONL" not in html
    assert "Accepted Traces" not in html
    assert "Rejected Traces" not in html


def test_workflow_report_renders_mixed_qa_and_pddl_records(tmp_path):
    config = minimal_data_config(tmp_path)
    write_minimal_artifacts(config)
    data = collect_workflow_report_data(config)
    goal = """(and
  (visited-object O1)
  (not (at-object O2))
  (or (visited-place p1) (at-place p2))
)"""
    grounding = {
        "expected": {
            "visited_object": ["O1"],
            "excluded_object": ["O2"],
            "place": ["p1", "p2"],
        },
        "fields": {
            "visited_object": {"aliases": ["visited_object"], "symbol_type": "Object"},
            "excluded_object": {
                "aliases": ["excluded_object"],
                "symbol_type": "Object",
            },
            "place": {"aliases": ["place"], "symbol_type": "Place"},
        },
        "match": "exact",
    }
    pddl_record = {
        "uid": "pddl1",
        "query_type": "pddl_mixed_object_place_goal",
        "task_family": "cypher_pddl",
        "output_type": "PDDL",
        "question": "Prepare a goal that visits O1 while avoiding O2.",
        "gold_cypher": "MATCH (o:Object) RETURN o.nodeSymbol AS object",
        "solution": goal,
        "grounding": grounding,
        "tags": ["pddl", "and", "or", "not"],
        "final_answer": f"<answer>{goal}</answer>",
        "messages": [{"role": "assistant", "content": f"<answer>{goal}</answer>"}],
        "validation": {
            "tool_result_match": True,
            "final_answer_domain_valid": True,
            "answer_grounded": True,
        },
    }

    questions = data["generated_questions"]
    questions["records"].append(pddl_record)
    questions["by_task_family"]["cypher_pddl"] = 1
    questions["by_task_type"][pddl_record["query_type"]] = 1
    for tag in pddl_record["tags"]:
        questions["tag_counts"][tag] = 1

    traces = data["traces"]
    traces["accepted_records"].append(pddl_record)
    traces["accepted_by_task_family"]["cypher_pddl"] = 1
    traces["accepted_by_task_type"][pddl_record["query_type"]] = 1
    traces["gold_cypher_lookup"]["pddl1"] = pddl_record["gold_cypher"]

    splits = data["splits"]
    splits["records"]["train"].append(pddl_record)
    splits["by_task_family"]["train"]["cypher_pddl"] = 1
    splits["by_task_type"]["train"][pddl_record["query_type"]] = 1
    splits["gold_cypher_lookup"]["pddl1"] = pddl_record["gold_cypher"]

    pddl_validation_record = {
        "uid": "pddl1",
        "query_type": pddl_record["query_type"],
        "task_family": "cypher_pddl",
        "output_type": "PDDL",
        "question": pddl_record["question"],
        "expected": {
            "cypher": pddl_record["gold_cypher"],
            "cypher_solution": goal,
            "pddl_goal": goal,
            "grounding": grounding,
            "final_answer": f"<answer>{goal}</answer>",
        },
        "generated": {
            "cypher": pddl_record["gold_cypher"],
            "cypher_solution": json.dumps(grounding["expected"], sort_keys=True),
            "cypher_tool_output": [grounding["expected"]],
            "actual_grounding": grounding["expected"],
            "final_answer": f"<answer>{goal}</answer>",
            "final_answer_value": goal,
        },
        "final_answer_from_trace": {
            "status": "pass",
            "final_answer_domain_valid": True,
            "final_answer_match": True,
        },
        "tool_call_from_prompt": {
            "status": "pass",
            "tool_call_parse_success": True,
            "executable": True,
            "cypher_solution_match": True,
            "tool_result_match": True,
            "grounding_match": True,
            "expected_grounding": grounding,
            "actual_grounding": grounding["expected"],
        },
    }
    data["validation"]["raw_generation_reports"]["test"]["records"].append(
        pddl_validation_record
    )

    html = render_workflow_report(data)
    rows, details = validation_example_rows(
        {"test": {"records": [pddl_validation_record]}}
    )
    pddl_detail = render_record_detail(details[0])

    assert "Task Families Per Split" in html
    assert "<h3>Task Family</h3>" in html
    assert "cypher_pddl" in html
    assert "<h5>Cypher QA</h5>" in html
    assert "<h5>Cypher PDDL</h5>" in html
    assert "Grounding Match" in html
    assert rows[0]["task_family"] == "cypher_pddl"
    assert rows[0]["output_type"] == "PDDL"
    assert rows[0]["grounding_match"] is True
    assert rows[0]["cypher_solution_match"] is None
    assert "<h4>Expected PDDL Goal</h4>" in pddl_detail
    assert "(and\n  (visited-object O1)" in pddl_detail
    assert "(not (at-object O2))" in pddl_detail
    assert "(or (visited-place p1) (at-place p2))" in pddl_detail
    assert "<h4>Expected Grounding</h4>" in pddl_detail
    assert "<h4>Actual Grounding</h4>" in pddl_detail
    assert "Expected Cypher Solution" not in pddl_detail
    assert "Generated Cypher Solution" not in pddl_detail
    assert "Tool Result Match" in html
    assert "Final Answer Domain Valid" in html
    assert "Answer Grounded" in html


def test_workflow_report_default_output_lives_in_reports_dir(tmp_path):
    config = minimal_data_config(tmp_path)

    assert workflow_report_output_path(config).name == "workflow_report.html"
    assert workflow_report_output_path(config).parent.name == "reports"


def test_validation_details_surface_hidden_false_positive_diagnostics():
    record = {
        "uid": "risk1",
        "query_type": "objects_in_place_symbol",
        "task_family": "cypher_qa",
        "question": "Which objects are in p1?",
        "expected": {
            "cypher": "MATCH (:Place)-[:CONTAINS]->(:Object) RETURN 1",
            "cypher_solution": "<>",
        },
        "generated": {
            "cypher": "MATCH (:Place)-[:CONTAINS*2..2]->(:Object) RETURN 1",
            "cypher_solution": "<>",
            "cypher_tool_output": [],
        },
        "final_answer_from_trace": {"status": "pass", "final_answer_match": True},
        "tool_call_from_prompt": {
            "status": "pass",
            "executable": True,
            "cypher_solution_match": True,
            "potential_hidden_false_positive": True,
            "hidden_false_positive_reasons": [
                "non_exact_cypher_matched_empty_result",
                "relationship_pattern_mismatch",
            ],
        },
    }

    rows, details = validation_example_rows({"test": {"records": [record]}})

    assert rows[0]["potential_hidden_false_positive"] is True
    assert details[0]["groups"][0]["blocks"]["cypher_validation_issues"][
        "hidden_false_positive_reasons"
    ] == [
        "non_exact_cypher_matched_empty_result",
        "relationship_pattern_mismatch",
    ]


def test_validation_details_backfill_hidden_false_positive_for_old_reports():
    record = {
        "uid": "legacy-risk",
        "query_type": "objects_in_place_symbol",
        "task_family": "cypher_qa",
        "expected": {
            "cypher": "MATCH (:Place)-[:CONTAINS]->(:Object) RETURN 1",
            "cypher_solution": "<>",
        },
        "generated": {
            "cypher": "MATCH (:Place)-[:CONTAINS*2..2]->(:Object) RETURN 1",
            "cypher_solution": "<>",
            "cypher_tool_output": [],
        },
        "final_answer_from_trace": {"status": "pass", "final_answer_match": True},
        "tool_call_from_prompt": {
            "status": "pass",
            "cypher_exact_match": False,
            "executable": True,
            "cypher_solution_match": True,
        },
    }

    rows, details = validation_example_rows({"test": {"records": [record]}})

    assert rows[0]["potential_hidden_false_positive"] is True
    assert "relationship_pattern_mismatch" in details[0]["groups"][0]["blocks"][
        "cypher_validation_issues"
    ]["hidden_false_positive_reasons"]


def test_scene_template_sample_renders_comma_separated_dictionaries():
    cases = [
        {"class": "chair", "point": {"x": 1, "y": 2, "z": 3}},
        {"class": "table", "point": {"x": 4, "y": 5, "z": 6}},
    ]

    rendered = scene_template_sample_text("object_radius_cases", cases)

    assert rendered == (
        "{class: chair, point: {x: 1, y: 2, z: 3}}, "
        "{class: table, point: {x: 4, y: 5, z: 6}}"
    )
