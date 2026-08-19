import json

from cypher_3dsg_agent_distillation.io import write_jsonl
from cypher_3dsg_agent_distillation.workflow_report import (
    collect_workflow_report_data,
    render_workflow_report,
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
                    }
                ],
            },
            handle,
        )


def test_workflow_report_collects_scene_artifacts_and_renders_tabs(tmp_path):
    config = minimal_data_config(tmp_path)
    write_minimal_artifacts(config)

    data = collect_workflow_report_data(config)
    html = render_workflow_report(data)

    assert data["scene_graph"]["summary"]["nodes"]["Object"] == 3
    assert data["generated_questions"]["by_task_type"] == {"object_class_count": 1}
    assert data["splits"]["optimized_questions"][0]["uid"] == "q1"
    assert data["train"]["aggregate_eval_loss"]["summary"]["examples"] == 1
    assert "3D Scene Graph" in html
    assert "Generated Questions" in html
    assert "Trace Outcomes" in html
    assert "Validation Examples" in html
    assert "Artifacts" in html
    assert "Object Classes" in html
    assert "Mesh Place Classes" in html
    assert "Room Classes" in html
    assert "Objects</div>" in html
    assert "Rooms</div>" in html
    assert html.index("Object Classes") < html.index("Template Values")
    assert "0 classes" in html
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
    assert '<th data-sort-index="2">Task Type</th>' in html
    assert "100.00%" in html
    assert '<span class="status-pill status-good">True</span>' in html
    assert '<span class="status-pill status-good">pass</span>' not in html
    assert "&lt; O1, O2 &gt;" not in html
    assert "<pre>&lt;O1, O2&gt;</pre>" in html
    assert "How many chairs?" in html
    assert "<h3>Trace Details</h3>" in html
    assert "<h3>Split Question Details</h3>" in html
    assert "<h3>Validation Example Details</h3>" in html
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
    assert "generated-questions-detail" not in html
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
        html.index('id="validation"') : html.index('id="artifacts"')
    ]
    assert "<h4>Test</h4>" in validation_section
    assert ">All</td>" in validation_section
    validation_examples_start = validation_section.index("<h3>Validation Examples</h3>")
    validation_examples_table = validation_section[
        validation_examples_start :
        validation_section.index("</table>", validation_examples_start)
    ]
    validation_headers = [
        "UID",
        "Split",
        "Task Type",
        "Question",
        "Tool Call Status",
        "Final Answer Status",
        "Tool Call Parse Success",
        "Tool Call Executable",
        "Cypher Solution Match",
        "Final Answer Match",
    ]
    assert all(header in validation_examples_table for header in validation_headers)
    assert [validation_examples_table.index(header) for header in validation_headers] == sorted(
        validation_examples_table.index(header) for header in validation_headers
    )
    assert "font: inherit;" in html
    assert "equal-columns" in html
    assert "Agent Trace Reports" not in html
    assert "Teacher Trace JSONL" not in html
    assert "Accepted Traces" not in html
    assert "Rejected Traces" not in html


def test_workflow_report_default_output_lives_in_reports_dir(tmp_path):
    config = minimal_data_config(tmp_path)

    assert workflow_report_output_path(config).name == "workflow_report.html"
    assert workflow_report_output_path(config).parent.name == "reports"
