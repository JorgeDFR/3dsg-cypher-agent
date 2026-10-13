from copy import deepcopy
from io import StringIO

import pytest
from rich.console import Console

from cypher_3dsg_agent_distillation.agents.teacher_check_reporting import (
    example_report,
    format_cost,
    format_duration,
    print_check_report,
    reasoning_stage_report,
)
from cypher_3dsg_agent_distillation.core import cli_output
from tests.agents.test_teacher_check import report_for, response


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (None, "Unknown"),
        (0, "0.00 s"),
        (6.562171363999369, "6.56 s"),
        (60, "1 min 0 s"),
        (3599.9, "1 h 0 min 0 s"),
        (11211.573754379133, "3 h 6 min 52 s"),
    ],
)
def test_durations_are_readable_and_round_across_unit_boundaries(seconds, expected):
    assert format_duration(seconds) == expected


def test_currency_keeps_small_check_cost_and_rounds_dataset_estimate():
    assert format_cost(None) == "Unknown"
    assert format_cost(0.00042226) == "$0.000422"
    assert format_cost(0.7009516, dataset=True) == "$0.70"
    assert format_cost(12345.6789, dataset=True) == "$12,345.68"


def test_stage_report_distinguishes_text_summary_encrypted_and_partial_coverage():
    tool = response(tool=True)["choices"][0]["message"]
    encrypted = deepcopy(tool)
    encrypted.pop("reasoning")
    encrypted["reasoning_details"] = [{"type": "reasoning.encrypted", "data": "opaque-payload"}]
    final = {
        "role": "assistant",
        "content": "<answer>3</answer>",
        "reasoning_details": [{"type": "reasoning.summary", "summary": "Three objects."}],
    }
    stages = reasoning_stage_report([tool, encrypted, final], [tool, encrypted, final])
    assert stages["tool_call"] == {
        "turns": 2,
        "formats": ["encrypted", "text"],
        "readable_turns": 1,
        "encrypted_turns": 1,
        "all_turns_readable": False,
        "saved_readable_turns": 1,
    }
    assert stages["final_answer"]["formats"] == ["summary"]
    assert stages["final_answer"]["all_turns_readable"]
    assert reasoning_stage_report([tool], [])["tool_call"]["saved_readable_turns"] == 0


def test_example_records_tool_errors_cypher_and_reasoning_without_prompt_scaffolding():
    failed = response(tool=True)["choices"][0]["message"]
    failed["tool_calls"][0]["function"]["arguments"] = {"cypher_string": "MATC broken"}
    fixed = deepcopy(failed)
    fixed["tool_calls"][0]["id"] = "call-2"
    fixed["tool_calls"][0]["function"]["arguments"] = {"cypher_string": "MATCH (o:Object) RETURN o"}
    final = response()["choices"][0]["message"]
    messages = [
        {"role": "system", "content": "PRIVATE PROMPT SCAFFOLDING"},
        {"role": "assistant", "content": "OLD EXAMPLE", "reasoning": "OLD EXAMPLE REASONING"},
        {"role": "user", "content": "Now answer the dataset question."},
        failed,
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"ok":false,"issues":["Syntax error"]}',
        },
        fixed,
        {"role": "tool", "tool_call_id": "call-2", "content": '{"ok":true,"rows":[]}'},
        final,
    ]
    example = example_report(messages, {"question": "How many objects?", "solution": 0})
    assert "PRIVATE PROMPT SCAFFOLDING" not in str(example)
    assert "OLD EXAMPLE" not in str(example)
    assert example["expected_answer"] == 0
    assert example["turns"][0]["tool_calls"][0]["feedback"]["ok"] is False
    assert example["turns"][1]["tool_calls"][0]["feedback"]["ok"] is True
    assert example["turns"][1]["tool_calls"][0]["arguments"]["cypher_string"].startswith("MATCH")
    assert example["turns"][-1]["content"] == "<answer>3</answer>"


def capture_report(report, monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(cli_output, "console", Console(file=stream, width=120, color_system=None))
    print_check_report(report, "check.json")
    return stream.getvalue()


def test_terminal_report_formats_metrics_and_shows_answer_with_validation(monkeypatch):
    report = report_for()
    tool = response(tool=True)["choices"][0]["message"]
    final = response()["choices"][0]["message"]
    report["metrics"]["billed_cost_usd"] = 0.00042226
    report["projection"]["full_dataset_cost_usd"] = 0.7009516
    report["projection"]["sequential_conversation_seconds"] = 11211.573754379133
    report["capability"]["reasoning_stages"] = reasoning_stage_report([tool, final], [tool, final])
    report["outcome"].update(
        {
            "accepted": True,
            "validation": {
                "static_valid": True,
                "executable": True,
                "final_answer_match": True,
            },
        }
    )
    report["example"] = example_report(
        [tool, final], {"question": "How many objects?", "solution": 3}
    )
    report["messages"].insert(0, {"role": "system", "content": "DO NOT DISPLAY FULL PROMPT"})
    output = capture_report(report, monkeypatch)
    assert "Estimated dataset generation time" in output
    assert "3 h 6 min 52 s" in output
    assert "$0.70" in output and "$0.000422" in output
    assert "Reasoning availability" in output and "Readable turns" in output
    assert "How many objects?" in output
    assert "MATCH (o:Object) RETURN count(o) AS count" in output
    assert "Count Object nodes." in output
    assert "Return a number." in output
    assert "<answer>3</answer>" in output
    assert "Answer validation" in output and "Pass" in output
    assert "DO NOT DISPLAY FULL PROMPT" not in output
    assert "Reasoning stages" not in output
    assert "{'tool_reasoning'" not in output


def test_terminal_report_marks_encrypted_reasoning_unusable_and_answer_incorrect(monkeypatch):
    report = report_for(readable=False)
    final = response(reasoning=False)["choices"][0]["message"]
    final["reasoning_details"] = [{"type": "reasoning.encrypted", "data": "opaque-payload"}]
    report["outcome"].update(
        {
            "accepted": False,
            "reason": "final_answer_mismatch",
            "validation": {"final_answer_match": False},
        }
    )
    report["capability"]["reasoning_stages"] = reasoning_stage_report([final], [final])
    report["example"] = example_report([final], {"question": "Count?", "solution": 4})
    output = capture_report(report, monkeypatch)
    assert "Encrypted" in output and "cannot be used as training text" in output
    assert "opaque-payload" not in output
    assert "Rejected (final answer mismatch)" in output
    assert "Final answer matches expected" in output and "Fail" in output
    assert "Expected answer" in output


def test_terminal_report_keeps_unknown_and_incomplete_distinct_from_failures(monkeypatch):
    report = report_for(status="incomplete", cost=None)
    report["outcome"] = {"status": "incomplete", "error": "Reached request limit."}
    report["example"] = example_report([], {"question": "Count?", "solution": 3})
    output = capture_report(report, monkeypatch)
    assert "Incomplete" in output
    assert "Reached request limit." in output
    assert "Unknown" in output
    assert "Not checked" in output
    assert "No final answer generated." in output


@pytest.mark.parametrize("expected", ["<\n  O1,\n  O2\n>\n", r"<\n  O1,\n  O2\n>\n"])
def test_expected_answer_is_compact_only_in_terminal_and_example_is_grouped(monkeypatch, expected):
    report = report_for()
    report["example"] = example_report([], {"question": "Which objects?", "solution": expected})
    output = capture_report(report, monkeypatch)
    assert "Generated text example" in output
    assert "Generated dataset example" not in output
    assert "< O1, O2 >" in output
    assert any("│ ╭" in line and "Question" in line for line in output.splitlines())
    assert report["example"]["expected_answer"] == expected


def test_table_notes_explain_timing_formats_and_checks_inside_their_borders(monkeypatch):
    output = capture_report(report_for(), monkeypatch)
    for explanation in ("Example generation time:", "Returned format:", "Results:"):
        line = next(line for line in output.splitlines() if explanation in line)
        assert line.startswith("│") and line.endswith("│")
    compact = " ".join(output.split())
    assert "Loading the Neo4j scene" in compact
    assert "before this timer starts and are excluded" in compact
    assert "about 3.00 s" in compact and "100 traces" in compact
