import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cypher_3dsg_agent_distillation.agents import openrouter_teacher
from cypher_3dsg_agent_distillation.agents.agent_records import (
    DEFAULT_AGENT_TOOL,
    WRITE_REFUSAL_ANSWER,
)
from cypher_3dsg_agent_distillation.agents.experiment_profiles import load_experiment_profile
from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
    OpenRouterTeacherClient,
    configured_teacher_provider,
    debug_trace_record,
    evaluate_candidate_trace,
    nearest_rank_percentile,
    normalize_assistant_message,
    progress_description,
    run_synthetic_teacher_interaction,
    run_teacher_interaction,
    seed_expected_solution,
    trace_token_length_report,
)
from cypher_3dsg_agent_distillation.core.config import load_yaml
from cypher_3dsg_agent_distillation.cypher.task_semantics import enrich_seed_semantics
from cypher_3dsg_agent_distillation.cypher.templates import generate_records
from cypher_3dsg_agent_distillation.cypher.validation_config import load_question_catalog


class FakeValidator:
    def __init__(self, results=None):
        self.results = list(
            results
            or [
                {
                    "executable": True,
                    "rows": [{"count": 3}],
                    "error": None,
                }
            ]
        )
        self.cyphers = []

    def validate(self, cypher, execute):
        self.cyphers.append(cypher)
        return self.results.pop(0)


def test_generate_command_configures_model_output_before_startup(monkeypatch):
    calls = []

    def configure_model_library_output():
        calls.append("model output")

    def suppress_neo4j_notification_logs():
        calls.append("neo4j notifications")

    def stop_after_configuration(_path):
        assert calls == ["model output", "neo4j notifications"]
        raise RuntimeError("stop after output configuration")

    monkeypatch.setattr(
        openrouter_teacher,
        "configure_model_library_output",
        configure_model_library_output,
    )
    monkeypatch.setattr(
        openrouter_teacher,
        "suppress_neo4j_notification_logs",
        suppress_neo4j_notification_logs,
    )
    monkeypatch.setattr(openrouter_teacher, "load_yaml", stop_after_configuration)

    with pytest.raises(RuntimeError, match="stop after output configuration"):
        openrouter_teacher.generate_command(SimpleNamespace(teacher_config="teacher.yaml"))


class FakeConversationClient:
    model = "openai/gpt-oss-120b"

    def __init__(self, responses):
        self.responses = [normalize_assistant_message(response) for response in responses]
        self.calls = []

    def chat(self, messages, *, tools=None, tool_choice=None):
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "tool_choice": tool_choice,
            }
        )
        return deepcopy(self.responses.pop(0))


class CountingTokenizer:
    chat_template = "test-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None):
        assert tokenize is False
        assert add_generation_prompt is False
        tool_names = ",".join(tool["function"]["name"] for tool in tools or [])
        message_text = "|".join(str(message.get("content") or "") for message in messages)
        return f"{tool_names}:{message_text}"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text)))}


def seed_record(**overrides):
    payload = {
        "uid": "seed",
        "scene_id": "scene",
        "question": "How many objects?",
        "query_type": "object_class_count",
        "cypher": "MATCH (o:Object) RETURN count(o) AS count",
        "solution": 3,
        "read_only": True,
    }
    payload.update(overrides)
    return payload


def tool_assistant(
    cypher="MATCH (o:Object) RETURN count(o) AS count",
    *,
    tool_call_id="call_1",
    tool_name=DEFAULT_AGENT_TOOL,
    arguments_as_json=True,
):
    arguments = {"cypher_string": cypher}
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments) if arguments_as_json else arguments,
                },
            }
        ],
    }


def malformed_tool_arguments_assistant():
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_bad",
                "type": "function",
                "function": {
                    "name": DEFAULT_AGENT_TOOL,
                    "arguments": '{\n  "cypher_string": "MATCH (o:Object) RETURN count(o) AS count',
                },
            }
        ],
    }


def final_assistant(answer="3"):
    return {"role": "assistant", "content": f"<answer>{answer}</answer>"}


def candidate_trace(cypher="MATCH (o:Object) RETURN count(o) AS count"):
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": DEFAULT_AGENT_TOOL,
                            "arguments": {"cypher_string": cypher},
                        }
                    }
                ],
            },
            {"role": "tool", "tool_name": DEFAULT_AGENT_TOOL, "content": "[{'count': 3}]"},
            {"role": "assistant", "content": "<answer>3</answer>"},
        ]
    }


def test_trace_token_length_report_uses_training_rendering_and_nearest_rank():
    tokenizer = CountingTokenizer()
    records = [
        {"uid": "short", "messages": [{"role": "assistant", "content": "x"}]},
        {"uid": "long", "messages": [{"role": "assistant", "content": "x" * 20}]},
    ]

    report = trace_token_length_report(records, tokenizer, max_length=30)

    assert report["trace_count"] == 2
    assert report["traces"][0]["token_length"] < report["traces"][1]["token_length"]
    assert report["traces"][0]["exceeds_max_length"] is False
    assert report["traces"][1]["exceeds_max_length"] is True
    assert report["overlength_count"] == 1
    assert report["overlength_rate"] == 0.5
    assert report["percentiles"] == {
        "p50": report["traces"][0]["token_length"],
        "p90": report["traces"][1]["token_length"],
        "p95": report["traces"][1]["token_length"],
        "p99": report["traces"][1]["token_length"],
        "max": report["traces"][1]["token_length"],
    }
    assert nearest_rank_percentile(list(range(1, 101)), 95) == 95


def run_interaction(
    seed, responses, *, validator=None, max_tool_calls=3, static_only=False, tuning_prompt="",
):
    client = FakeConversationClient(responses)
    profile_name = (
        "cypher_pddl_template.yaml"
        if seed.get("task_family") == "cypher_pddl"
        else "cypher_qa_template.yaml"
    )
    decision = run_teacher_interaction(
        seed,
        profile=load_experiment_profile(
            f"configs/experiment_profiles/{profile_name}"
        ),
        client=client,
        teacher_model="openai/gpt-oss-120b",
        validator=validator if validator is not None else FakeValidator(),
        policy=None,
        max_tool_calls=max_tool_calls,
        static_only=static_only,
        tuning_prompt=tuning_prompt,
    )
    return decision, client


@pytest.mark.parametrize("accepted", [True, False])
def test_tuning_prompt_persists_across_retries_but_is_not_saved(accepted):
    prompt = "Only one tool call per response."
    responses = [
        tool_assistant("MATCH (o:Object) DELETE o"),
        tool_assistant(),
        final_assistant("3" if accepted else "999"),
    ]
    decision, client = run_interaction(seed_record(), responses, tuning_prompt=prompt)
    assert decision.accepted is accepted
    assert len(client.calls) == 3
    for call in client.calls:
        user = next(message for message in reversed(call["messages"]) if message["role"] == "user")
        assert user["content"].endswith("\n\n" + prompt)
        assert user["content"].count(prompt) == 1
    assert prompt not in json.dumps(decision.payload)
    if decision.record is not None:
        assert prompt not in decision.record.model_dump_json()


def test_teacher_does_not_execute_a_new_query_after_success_at_tool_limit():
    validator = FakeValidator()
    decision, client = run_interaction(
        seed_record(), [tool_assistant(), tool_assistant()],
        validator=validator, max_tool_calls=1,
    )
    assert decision.reason == "max_tool_calls_exceeded"
    assert len(validator.cyphers) == 1
    assert len(client.calls) == 2


@pytest.mark.parametrize("status", [
    "invalid_result_shape", "mapping_unresolved", "mismatch", "grounding_mismatch",
])
def test_internal_validation_status_cannot_be_sent_as_tool_feedback(status):
    with pytest.raises(ValueError, match="only safety or execution"):
        openrouter_teacher.tool_error_content(status, ["internal issue"])


def test_teacher_provider_requires_supported_explicit_type():
    assert configured_teacher_provider({"provider": "openrouter"}) == "openrouter"
    assert configured_teacher_provider({"provider": "synthetic"}) == "synthetic"

    try:
        configured_teacher_provider({"provider": "local"})
    except ValueError as exc:
        assert "Unsupported teacher provider" in str(exc)
    else:
        raise AssertionError("Expected unsupported provider error")


def test_real_teacher_ignores_enabled_controlled_repair_with_warning():
    with pytest.warns(UserWarning, match="only valid for provider: synthetic"):
        assert configured_teacher_provider({
            "provider": "openrouter", "inject_controlled_repair": True,
        }) == "openrouter"


@pytest.mark.parametrize("include", [False, True])
def test_teacher_inclusion_switch_preserves_replay_and_filters_saved_reasoning(include):
    blocks = [{"type": "reasoning.text", "text": "Count Object nodes.", "signature": "signed"}]
    calls = []

    def send(request):
        calls.append(deepcopy(request))
        assistant = tool_assistant() if len(calls) == 1 else final_assistant()
        assistant["reasoning_details"] = deepcopy(blocks)
        if len(calls) == 2:
            assistant["content"] = "<think>Return a number.</think><answer>3</answer>"
        return {"choices": [{"message": assistant, "finish_reason": "stop"}], "usage": {}}

    client = OpenRouterTeacherClient(request_sender=send, reasoning={"enabled": include})
    seed = seed_record(metadata={"controlled_repair": True, "repair_cypher": "MATC broken",
                                 "repair_template": "syntax"})
    decision = run_teacher_interaction(
        seed,
        profile=load_experiment_profile("configs/experiment_profiles/cypher_qa_template.yaml"),
        client=client, teacher_model="teacher", validator=FakeValidator(), policy=None,
        max_tool_calls=2, static_only=False,
    )
    assert decision.accepted
    assert len(calls) == 2
    # A provider that returns signed blocks despite exclude still receives exact replay.
    replay = next(message for message in calls[1]["messages"] if message.get("tool_calls"))
    assert replay["reasoning_details"] == blocks
    assert calls[0]["reasoning"]["exclude"] is (not include)
    assert ("enabled" in calls[0]["reasoning"]) is include
    saved = decision.record.model_dump(exclude_none=True)
    saved_call = next(message for message in saved["messages"] if message.get("tool_calls"))
    assert ("reasoning_details" in saved_call) is include
    assert ("reasoning" in saved["messages"][-1]) is include
    assert saved["messages"][-1]["content"] == "<answer>3</answer>"
    assert decision.record.metadata["reasoning_coverage"]["both_stages"] is include
    assert "controlled_repair" not in decision.record.metadata
    payload_call = next(
        message for message in decision.payload["messages"] if message.get("tool_calls")
    )
    assert ("reasoning_details" in payload_call) is include


def test_reasoning_effort_is_optional_and_exclude_is_no_longer_a_config_option():
    assert OpenRouterTeacherClient(reasoning={"enabled": True}).api_reasoning() == {
        "enabled": True, "exclude": False,
    }
    with pytest.raises(ValueError, match="reasoning.exclude was removed"):
        OpenRouterTeacherClient(reasoning={"enabled": True, "exclude": False})


def test_synthetic_teacher_config_does_not_define_a_model():
    config = load_yaml("configs/teachers/teacher_synthetic.yaml")

    assert configured_teacher_provider(config) == "synthetic"
    assert "model" not in config
    assert config["max_tool_calls"] == 2


def test_synthetic_teacher_constructs_trace_from_gold_and_database_rows():
    validator = FakeValidator()
    decision = run_synthetic_teacher_interaction(
        seed_record(task_family="cypher_qa", output_type="SLDP"),
        profile=load_experiment_profile(
            "configs/experiment_profiles/cypher_qa_template.yaml"
        ),
        validator=validator,
        policy=None,
    )

    assert decision.accepted
    assert decision.reason == "accepted"
    assert validator.cyphers == ["MATCH (o:Object) RETURN count(o) AS count"]
    assert decision.record.first_cypher_argument == (
        "MATCH (o:Object) RETURN count(o) AS count"
    )
    assert decision.record.messages[-2].content == (
        '{"ok": true, "rows": [{"count": 3}]}'
    )
    assert decision.record.messages[-1].content == "<answer>3</answer>"
    assert decision.record.cypher_solution == "3"
    assert decision.record.validation.solution_match is True
    assert decision.record.teacher_model is None
    assert "teacher_model" not in decision.record.model_dump(exclude_none=True)
    assert decision.record.metadata["teacher_provider"] == "synthetic"
    assert decision.record.metadata["generation_mode"] == "synthetic_gold_cypher"


@pytest.mark.parametrize("reasoning_enabled", [False, True])
@pytest.mark.parametrize(
    ("object_symbol", "rows", "expected"),
    [
        ("O0", [{"place": "p663"}], "p663"),
        ("O12", [{"place": "p1519"}, {"place": "p30886"}], "<p1519, p30886>"),
        ("O2", [], "<>"),
    ],
)
def test_synthetic_parent_lookup_accepts_actual_scene_result_shapes(
    object_symbol, rows, expected, reasoning_enabled
):
    template = next(
        template
        for template in load_question_catalog("configs/question_templates/cypher_qa_3dsg.yaml")[
            "templates"
        ]
        if template["query_type"] == "object_place_lookup"
    )
    template = {**template, "target_questions": 1, "questions": template["questions"][:1]}
    records = generate_records(
        {
            "scene_id": "uhumans2_office",
            "source_3dsg": "data/scenes/uhumans2_office.json",
            "values": {"object_parent_cases": [{"object": object_symbol}]},
        },
        question_templates_config={"templates": [template]},
        warn_on_target_shortfall=False,
    )
    seed = records[0]
    enrich_seed_semantics(seed, template)
    decision = run_synthetic_teacher_interaction(
        seed,
        profile=load_experiment_profile("configs/experiment_profiles/cypher_qa_template.yaml"),
        validator=FakeValidator([{"executable": True, "rows": rows, "error": None}]),
        policy=None,
        reasoning={"enabled": reasoning_enabled},
    )

    assert decision.accepted, decision.reason
    assert decision.record.final_answer == expected
    assert decision.record.cypher_solution == expected
    assert decision.record.validation.final_answer_format_valid is True
    assert decision.record.validation.solution_match is True
    final_message = decision.record.messages[-1]
    assert bool(final_message.reasoning) == reasoning_enabled


def test_synthetic_teacher_constructs_controlled_repair_trace():
    invalid_cypher = "MATC (o:Object) RETURN count(o) AS count"
    validator = FakeValidator(
        [
            {
                "syntax_error": True,
                "executable": False,
                "rows": [],
                "error": "Neo.ClientError.Statement.SyntaxError: Invalid input 'MATC'",
            },
            {"executable": True, "rows": [{"count": 3}], "error": None},
        ]
    )
    decision = run_synthetic_teacher_interaction(
        seed_record(
            task_family="cypher_qa",
            output_type="SLDP",
            metadata={
                "controlled_repair": True,
                "repair_template": "misspelled_match_clause",
                "repair_cypher": invalid_cypher,
            },
            tags=["controlled_repair"],
        ),
        profile=load_experiment_profile(
            "configs/experiment_profiles/cypher_qa_template.yaml"
        ),
        validator=validator,
        policy=None,
        max_tool_calls=2,
    )

    assert decision.accepted
    assert validator.cyphers == [
        invalid_cypher,
        "MATCH (o:Object) RETURN count(o) AS count",
    ]
    assert decision.record.metadata["tool_call_count"] == 2
    assert decision.record.metadata["successful_tool_call_index"] == 2
    assert decision.record.metadata["repair_context_message_count"] > 2
    repair_indices = decision.record.metadata["non_trainable_message_indices"]
    assert repair_indices[1] == repair_indices[0] + 1
    assert decision.record.messages[repair_indices[0]].tool_calls[0].function.arguments == {
        "cypher_string": invalid_cypher
    }
    assert "Statement.SyntaxError" in decision.record.messages[repair_indices[1]].content
    assert decision.record.messages[-1].content == "<answer>3</answer>"


def test_controlled_repair_requires_a_database_syntax_error():
    decision = run_synthetic_teacher_interaction(
        seed_record(
            metadata={
                "controlled_repair": True,
                "repair_template": "undefined_return_variable",
                "repair_cypher": "MATCH (n) RETURN missing.value",
            }
        ),
        profile=load_experiment_profile(
            "configs/experiment_profiles/cypher_qa_template.yaml"
        ),
        validator=FakeValidator(
            [
                {
                    "syntax_error": False,
                    "executable": False,
                    "error": "A non-syntax database failure",
                }
            ]
        ),
        policy=None,
    )

    assert not decision.accepted
    assert decision.reason == "invalid_controlled_repair"


def test_openrouter_teacher_ignores_controlled_repair_context_by_default():
    invalid_cypher = "MATC (o:Object) RETURN count(o) AS count"
    validator = FakeValidator()
    decision, client = run_interaction(
        seed_record(
            metadata={
                "controlled_repair": True,
                "repair_template": "misspelled_match_clause",
                "repair_cypher": invalid_cypher,
            }
        ),
        [tool_assistant(tool_call_id="call_2"), final_assistant()],
        validator=validator,
    )

    assert decision.accepted
    first_call_messages = client.calls[0]["messages"]
    assert all(message["role"] != "assistant" for message in first_call_messages)
    assert validator.cyphers == [seed_record()["cypher"]]
    assert decision.record.metadata["tool_call_count"] == 1
    assert "controlled_repair" not in decision.record.metadata
    assert decision.record.metadata["controlled_repair_skipped"] is True


def test_synthetic_teacher_accepts_empty_qa_answer_for_invalid_input():
    validator = FakeValidator(
        [{"executable": True, "rows": [{"count": 0}], "error": None}]
    )
    decision = run_synthetic_teacher_interaction(
        seed_record(
            solution="",
            solution_formats=["int"],
            tags=["invalid_input"],
            task_family="cypher_qa",
            output_type="SLDP",
        ),
        profile=load_experiment_profile(
            "configs/experiment_profiles/cypher_qa_template.yaml"
        ),
        validator=validator,
        policy=None,
    )

    assert decision.accepted
    assert decision.record.cypher_solution == ""
    assert decision.record.expected_solution == ""
    assert decision.record.messages[-1].content == "<answer></answer>"


def test_synthetic_teacher_accepts_zero_count_answer_for_invalid_input():
    validator = FakeValidator(
        [{"executable": True, "rows": [{"count": 0}], "error": None}]
    )
    decision = run_synthetic_teacher_interaction(
        seed_record(
            solution=0,
            solution_formats=["int"],
            tags=["invalid_input"],
            task_family="cypher_qa",
            output_type="SLDP",
        ),
        profile=load_experiment_profile(
            "configs/experiment_profiles/cypher_qa_template.yaml"
        ),
        validator=validator,
        policy=None,
    )

    assert decision.accepted
    assert decision.record.cypher_solution == "0"
    assert decision.record.expected_solution == 0
    assert decision.record.messages[-1].content == "<answer>0</answer>"


def test_synthetic_teacher_accepts_empty_pddl_answer_for_invalid_input():
    validator = FakeValidator([{"executable": True, "rows": [], "error": None}])
    decision = run_synthetic_teacher_interaction(
        seed_record(
            solution="",
            tags=["invalid_input"],
            task_family="cypher_pddl",
            output_type="PDDL",
            grounding={
                "expected": {"object": []},
                "fields": {
                    "object": {
                        "aliases": ["object"],
                        "symbol_type": "Object",
                        "cardinality": "set",
                    }
                },
                "match": "exact",
            },
        ),
        profile=load_experiment_profile(
            "configs/experiment_profiles/cypher_pddl_template.yaml"
        ),
        validator=validator,
        policy=None,
    )

    assert decision.accepted
    assert decision.record.cypher_solution == ""
    assert decision.record.expected_solution == ""
    assert decision.record.messages[-1].content == "<answer></answer>"


def test_synthetic_teacher_rejects_database_result_that_disagrees_with_gold():
    validator = FakeValidator(
        [{"executable": True, "rows": [{"count": 4}], "error": None}]
    )

    decision = run_synthetic_teacher_interaction(
        seed_record(task_family="cypher_qa", output_type="SLDP"),
        profile=load_experiment_profile(
            "configs/experiment_profiles/cypher_qa_template.yaml"
        ),
        validator=validator,
        policy=None,
    )

    assert not decision.accepted
    assert decision.reason == "gold_solution_mismatch"
    assert decision.payload["cypher_solution"] == "4"
    assert [message["role"] for message in decision.payload["messages"][-2:]] == [
        "assistant",
        "tool",
    ]
    assert decision.payload["messages"][-2]["tool_calls"][0]["function"][
        "arguments"
    ] == {"cypher_string": "MATCH (o:Object) RETURN count(o) AS count"}
    assert decision.payload["messages"][-1]["content"] == (
        '{"ok": true, "rows": [{"count": 4}]}'
    )


def test_synthetic_teacher_constructs_refusal_without_database_call():
    validator = FakeValidator()
    decision = run_synthetic_teacher_interaction(
        seed_record(
            cypher=WRITE_REFUSAL_ANSWER,
            solution=WRITE_REFUSAL_ANSWER,
            task_family="cypher_qa",
            output_type="SLDP",
        ),
        profile=load_experiment_profile(
            "configs/experiment_profiles/cypher_qa_template.yaml"
        ),
        validator=validator,
        policy=None,
    )

    assert decision.accepted
    assert decision.reason == "accepted_refusal"
    assert validator.cyphers == []
    assert not decision.record.tool_calls
    assert decision.record.messages[-1].content == (
        f"<answer>{WRITE_REFUSAL_ANSWER}</answer>"
    )


def test_progress_description_reports_generation_counts():
    assert progress_description(
        seed_index=2,
        seed_count=5,
        accepted_count=3,
        rejected_count=1,
    ) == "teacher traces seed 2/5 accepted=3 rejected=1"


def test_evaluate_candidate_trace_accepts_valid_legacy_trace():
    decision = evaluate_candidate_trace(
        seed_record(),
        candidate_trace(),
        teacher_model="openai/gpt-oss-120b",
        validator=FakeValidator(),
        policy=None,
    )

    assert decision.accepted
    assert decision.record.teacher_model == "openai/gpt-oss-120b"
    assert decision.record.validation.solution_match


def test_openrouter_preflight_reports_missing_model():
    fake_sdk = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda **_kwargs: SimpleNamespace(
                result=SimpleNamespace(data=[SimpleNamespace(id="other/model")])
            )
        )
    )
    client = OpenRouterTeacherClient(
        api_key="key",
        model="openai/gpt-oss-120b",
        sdk_client=fake_sdk,
    )

    try:
        client.preflight()
    except RuntimeError as exc:
        assert "openai/gpt-oss-120b" in str(exc)
    else:
        raise AssertionError("Expected missing model error")


def test_openrouter_chat_uses_sdk_without_json_response_format():
    calls = []

    def send(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(
                                    name=DEFAULT_AGENT_TOOL,
                                    arguments=json.dumps(
                                        {
                                            "cypher_string": (
                                                "MATCH (o:Object) RETURN count(o) AS count"
                                            )
                                        }
                                    ),
                                ),
                            )
                        ],
                    )
                )
            ]
        )

    fake_sdk = SimpleNamespace(chat=SimpleNamespace(send=send))
    client = OpenRouterTeacherClient(
        api_key="key",
        model="openai/gpt-oss-120b",
        sdk_client=fake_sdk,
    )

    message = client.chat(
        [{"role": "user", "content": "question"}],
        tools=[{"type": "function"}],
        tool_choice="auto",
    )

    assert message["tool_calls"][0]["function"]["arguments"] == {
        "cypher_string": "MATCH (o:Object) RETURN count(o) AS count"
    }
    assert calls[0]["model"] == "openai/gpt-oss-120b"
    assert calls[0]["parallel_tool_calls"] is False
    assert calls[0]["tools"] == [{"type": "function"}]
    assert calls[0]["tool_choice"] == "auto"
    assert calls[0]["timeout_ms"] == 120000
    assert "response_format" not in calls[0]
    assert "provider" not in calls[0]
    assert "models" not in calls[0]


def test_teacher_interaction_accepts_one_tool_conversation_without_synthetic_prompt():
    decision, client = run_interaction(seed_record(), [tool_assistant(), final_assistant()])

    assert decision.accepted
    assert decision.reason == "accepted"
    assert len(decision.record.messages) > 5
    assert [message.role for message in decision.record.messages[-3:]] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert decision.record.gold_cypher == "MATCH (o:Object) RETURN count(o) AS count"
    assert decision.record.real_solution == 3
    assert decision.record.cypher_solution == "3"
    assert decision.record.final_answer == "3"
    assert decision.record.validation.cypher_solution_match is True
    assert decision.record.validation.final_answer_match is True
    assert decision.record.validation.final_answer_cypher_match is True
    assert decision.record.metadata["generation_mode"] == "tool_conversation"
    first_call_messages = client.calls[0]["messages"]
    rendered_first_call = json.dumps(first_call_messages)
    assert "Generate one complete agent trace" not in rendered_first_call
    assert "Expected answer notation" not in rendered_first_call
    assert "MATCH (o:Object) RETURN count(o) AS count" not in rendered_first_call
    assert "solution" not in rendered_first_call.lower()


def test_teacher_interaction_accepts_alternative_class_at_top_k_tie_boundary():
    teacher_cypher = (
        "MATCH (o:Object) RETURN o.class AS class, count(o) AS count "
        "ORDER BY count DESC, class DESC LIMIT 2"
    )
    validator = FakeValidator(
        [
            {
                "executable": True,
                "rows": [
                    {"class": "light", "count": 10},
                    {"class": "plant", "count": 7},
                ],
                "error": None,
            }
        ]
    )
    seed = seed_record(
        question="Which two object classes are most common?",
        query_type="top_object_classes_by_count",
        cypher=(
            "MATCH (o:Object) RETURN o.class AS class, count(o) AS count "
            "ORDER BY count DESC, class ASC LIMIT 2"
        ),
        solution="<light, flag>",
        solution_formats=["set[str]"],
        answer_validation={
            "mode": "top_k_with_ties",
            "limit": 2,
            "scores": {"light": 10, "flag": 7, "plant": 7},
        },
    )

    decision, _client = run_interaction(
        seed,
        [tool_assistant(teacher_cypher), final_assistant("<light, plant>")],
        validator=validator,
    )

    assert decision.accepted
    assert decision.record.cypher_solution == "<light, plant>"
    assert decision.record.final_answer == "<light, plant>"
    assert decision.record.solution_formats == ["set[str]"]
    assert decision.record.validation.cypher_solution_match is True
    assert decision.record.validation.final_answer_match is True
    assert decision.record.validation.final_answer_cypher_match is True


def test_seed_expected_solution_preserves_falsy_solution_values():
    assert seed_expected_solution({"solution": 0, "expected_solution": 99}) == 0
    assert seed_expected_solution({"solution": [], "expected_solution": ["fallback"]}) == []


def test_teacher_interaction_accepts_zero_expected_solution():
    validator = FakeValidator(
        [
            {
                "executable": True,
                "rows": [{"count": 0}],
                "error": None,
            },
        ]
    )

    decision, _client = run_interaction(
        seed_record(solution=0),
        [tool_assistant(), final_assistant("0")],
        validator=validator,
    )

    assert decision.accepted
    assert decision.record.expected_solution == 0


def test_teacher_interaction_normalizes_tool_arguments_for_stored_record():
    decision, _client = run_interaction(
        seed_record(),
        [tool_assistant(arguments_as_json=True), final_assistant()],
    )

    assert decision.accepted
    assert decision.record.tool_calls[0].function.arguments == {
        "cypher_string": "MATCH (o:Object) RETURN count(o) AS count"
    }


def test_teacher_interaction_rejects_missing_tool():
    decision, _client = run_interaction(seed_record(), [final_assistant()])

    assert not decision.accepted
    assert decision.reason == "missing_required_tool"


def test_teacher_interaction_rejects_malformed_tool_arguments_without_crashing():
    decision, _client = run_interaction(seed_record(), [malformed_tool_arguments_assistant()])

    assert not decision.accepted
    assert decision.reason == "invalid_tool_arguments"
    assert "invalid tool arguments JSON" in decision.payload["error"]
    tool_message = next(
        message for message in decision.payload["messages"] if message.get("tool_calls")
    )
    assert tool_message["tool_calls"][0]["function"]["arguments"].startswith(
        "{\n"
    )


def test_teacher_interaction_repairs_static_issue_within_tool_limit():
    decision, _client = run_interaction(
        seed_record(),
        [
            tool_assistant("MATCH (o:Object) DELETE o", tool_call_id="call_1"),
            tool_assistant(tool_call_id="call_2"),
            final_assistant(),
        ],
    )

    assert decision.accepted
    assert decision.record.metadata["tool_call_count"] == 2
    assert decision.record.metadata["repair_attempted"]
    assert any(
        '"error_type": "static_issues"' in str(message.content)
        for message in decision.record.messages
        if message.role == "tool"
    )


def test_teacher_interaction_repairs_execution_failure_within_tool_limit():
    validator = FakeValidator(
        [
            {"executable": False, "rows": [], "error": "syntax error"},
            {"executable": True, "rows": [{"count": 3}], "error": None},
        ]
    )

    decision, _client = run_interaction(
        seed_record(),
        [
            tool_assistant("MATCH broken", tool_call_id="call_1"),
            tool_assistant(tool_call_id="call_2"),
            final_assistant(),
        ],
        validator=validator,
    )

    assert decision.accepted
    assert validator.cyphers == ["MATCH broken", "MATCH (o:Object) RETURN count(o) AS count"]
    assert any(
        '"error_type": "execution_failed"' in str(message.content)
        for message in decision.record.messages
        if message.role == "tool"
    )


def test_teacher_interaction_accepts_result_alias_within_tool_limit():
    validator = FakeValidator(
        [
            {"executable": True, "rows": [{"category": "chair"}], "error": None},
        ]
    )

    decision, _client = run_interaction(
        seed_record(
            query_type="object_class_inventory",
            solution="chair",
        ),
        [
            tool_assistant(
                "MATCH (o:Object) RETURN o.class AS category",
                tool_call_id="call_1",
            ),
            final_assistant("chair"),
        ],
        validator=validator,
    )

    assert decision.accepted
    assert validator.cyphers == [
        "MATCH (o:Object) RETURN o.class AS category",
    ]
    assert decision.record.messages[-2].content == (
        '{"ok": true, "rows": [{"category": "chair"}]}'
    )


def test_teacher_interaction_rejects_when_max_tool_calls_exceeded():
    decision, _client = run_interaction(
        seed_record(),
        [
            tool_assistant("MATCH (o:Object) DELETE o", tool_call_id="call_1"),
            tool_assistant("MATCH (o:Object) DELETE o", tool_call_id="call_2"),
        ],
        max_tool_calls=2,
    )

    assert not decision.accepted
    assert decision.reason == "max_tool_calls_exceeded"
    assert decision.payload["last_failure_reason"] == "static_issues"


def test_teacher_interaction_accepts_result_alias_at_tool_limit():
    validator = FakeValidator(
        [
            {"executable": True, "rows": [{"category": "chair"}], "error": None},
        ]
    )

    decision, _client = run_interaction(
        seed_record(query_type="object_class_inventory", solution="chair"),
        [
            tool_assistant("MATCH (o:Object) RETURN o.class AS category"),
            final_assistant("chair"),
        ],
        validator=validator,
        max_tool_calls=1,
    )

    assert decision.accepted


def test_teacher_interaction_rejects_final_answer_mismatch():
    decision, _client = run_interaction(seed_record(), [tool_assistant(), final_assistant("4")])

    assert not decision.accepted
    assert decision.reason == "final_answer_mismatch"
    assert decision.payload["cypher_solution"] == "3"
    assert decision.payload["final_answer"] == "4"
    assert decision.payload["validation"]["cypher_solution_match"] is True
    assert decision.payload["validation"]["final_answer_match"] is False


def test_evaluate_candidate_trace_rejects_final_answer_not_matching_real_solution():
    candidate = candidate_trace()
    candidate["messages"][-1]["content"] = "<answer>4</answer>"

    decision = evaluate_candidate_trace(
        seed_record(),
        candidate,
        teacher_model="openai/gpt-oss-120b",
        validator=FakeValidator(),
        policy=None,
    )

    assert not decision.accepted
    assert decision.reason == "final_answer_mismatch"
    assert decision.record.cypher_solution == "3"
    assert decision.record.final_answer == "4"


def test_debug_trace_record_uses_shared_solution_fields_for_accepts_and_rejects():
    seed = seed_record(tags=["count"])
    accepted_decision, _client = run_interaction(seed, [tool_assistant(), final_assistant()])
    accepted = debug_trace_record(
        seed,
        accepted_decision,
        record_json=accepted_decision.record.model_dump(by_alias=True, exclude_none=True),
    )

    rejected_decision, _client = run_interaction(
        seed,
        [tool_assistant(), final_assistant("4")],
    )
    rejected = debug_trace_record(seed, rejected_decision)

    shared_keys = set(accepted)
    assert set(rejected) == shared_keys | {"reason"}
    assert "reason" not in accepted
    assert rejected["reason"] == "final_answer_mismatch"
    assert accepted["real_solution"] == rejected["real_solution"] == 3
    assert accepted["cypher_solution"] == rejected["cypher_solution"] == "3"
    assert accepted["final_answer"] == "3"
    assert rejected["final_answer"] == "4"
    assert accepted["tags"] == rejected["tags"] == ["count"]


def test_teacher_interaction_rejects_invalid_sldp_final_answer():
    decision, _client = run_interaction(seed_record(), [tool_assistant(), final_assistant("3, 4")])

    assert not decision.accepted
    assert decision.reason == "invalid_answer_format"


def test_teacher_interaction_accepts_pddl_without_row_solution_conversion():
    validator = FakeValidator(
        [
            {
                "executable": True,
                "rows": [{"not_a_supported_qa_shape": "O1"}],
                "error": None,
            },
        ]
    )

    decision, _client = run_interaction(
        seed_record(
            query_type="pddl_direct_symbol",
            task_family="cypher_pddl",
            output_type="PDDL",
            solution="(visited-object O1)",
        ),
        [
            tool_assistant("MATCH (o:Object {nodeSymbol: 'O1'}) RETURN o"),
            final_assistant("(visited-object O1)"),
        ],
        validator=validator,
    )

    assert decision.accepted
    assert decision.record.metadata["task_family"] == "cypher_pddl"
    assert decision.record.metadata["output_type"] == "PDDL"


def test_teacher_interaction_requires_pddl_cypher_grounding_when_declared():
    grounding = {
        "expected": {"object": ["O1"]},
        "fields": {
            "object": {
                "aliases": ["object"],
                "symbol_type": "Object",
                "cardinality": "set",
            }
        },
        "match": "exact",
    }
    validator = FakeValidator(
        [{"executable": True, "rows": [{"object": "O2"}], "error": None}]
    )

    decision, _client = run_interaction(
        seed_record(
            query_type="pddl_direct_symbol",
            task_family="cypher_pddl",
            output_type="PDDL",
            solution="(visited-object O1)",
            grounding=grounding,
        ),
        [
            tool_assistant("MATCH (o:Object) RETURN o.nodeSymbol AS object"),
            final_assistant("(visited-object O1)"),
        ],
        validator=validator,
        max_tool_calls=1,
    )

    assert not decision.accepted
    assert decision.reason == "grounding_mismatch"
    assert json.loads(decision.payload["messages"][-2]["content"]) == {
        "ok": True, "rows": [{"object": "O2"}],
    }


def test_teacher_interaction_rejects_invalid_pddl_final_answer():
    decision, _client = run_interaction(
        seed_record(
            query_type="pddl_direct_symbol",
            task_family="cypher_pddl",
            output_type="PDDL",
            solution="(visited-object O1)",
        ),
        [tool_assistant(), final_assistant("visited-object O1")],
    )

    assert not decision.accepted
    assert decision.reason == "invalid_answer_format"


def test_teacher_interaction_accepts_refusal_without_tool_call():
    decision, _client = run_interaction(
        seed_record(cypher=WRITE_REFUSAL_ANSWER, solution=WRITE_REFUSAL_ANSWER),
        [final_assistant(WRITE_REFUSAL_ANSWER)],
    )

    assert decision.accepted
    assert decision.reason == "accepted_refusal"
    assert not decision.record.tool_calls


def test_teacher_interaction_rejects_refusal_with_tool_call():
    decision, _client = run_interaction(
        seed_record(cypher=WRITE_REFUSAL_ANSWER, solution=WRITE_REFUSAL_ANSWER),
        [tool_assistant()],
    )

    assert not decision.accepted
    assert decision.reason == "unexpected_tool_call_for_refusal"


def test_static_only_accepts_incomplete_statically_valid_trace():
    decision, _client = run_interaction(
        seed_record(),
        [tool_assistant()],
        validator=None,
        static_only=True,
    )

    assert decision.accepted
    assert decision.reason == "accepted_static_only"
    assert decision.record.metadata["static_only"]
    assert decision.record.metadata["incomplete_conversation"]


def test_contract_teacher_accepts_alias_and_retains_recomputable_evidence():
    contract = {
        "result": {"kind": "scalar", "type": "integer", "row_count": {"min": 1, "max": 1}},
        "answer": {"formats": ["int"], "empty": "0"},
    }
    query = "MATCH (o:Object) RETURN count(o) AS total"
    decision, client = run_interaction(
        seed_record(validation_contract=contract),
        [tool_assistant(query), final_assistant()],
        validator=FakeValidator([{"executable": True, "rows": [{"total": 3}], "error": None}]),
    )
    assert decision.accepted
    assert decision.record.validation.static_valid is True
    assert decision.record.validation.static_issues == []
    assert decision.record.validation.result_validation["status"] == "pass"
    assert decision.record.validation.answer_grounded
    assert decision.record.metadata["cypher_solution"] == "3"
    from cypher_3dsg_agent_distillation.agents.agent_evaluation import recompute_trace_validation

    assert recompute_trace_validation(decision.record)["result_validation"]["match"]
    assert "validation_contract" not in json.dumps(client.calls)


def scalar_contract_seed(**overrides):
    return seed_record(
        validation_contract={
            "result": {"kind": "scalar", "type": "integer"},
            "answer": {"formats": ["int"], "empty": "0"},
        },
        **overrides,
    )


@pytest.mark.parametrize("case", ["mismatch", "shape", "mapping", "grounding", "invalid_gold"])
def test_internal_validation_failures_receive_raw_rows_even_at_tool_limit(case):
    seed = scalar_contract_seed()
    query = "MATCH (o:Object) RETURN count(o) AS count"
    rows = [{"count": 2}]
    expected_status = "mismatch"
    answer = "3"
    if case == "shape":
        rows = [{"count": 3, "diagnostic": 42}]
        expected_status = "invalid_result_shape"
    elif case == "invalid_gold":
        seed["solution"] = "not_an_integer"
        rows = [{"count": 3}]
        expected_status = "invalid_expected_solution"
    elif case in {"mapping", "grounding"}:
        seed = seed_record(
            task_family="cypher_pddl", output_type="PDDL",
            solution="(visited-object O1)",
            grounding={"expected": {"object": ["O1"]}},
            validation_contract={
                "result": {"kind": "role_sets", "roles": {"object": {
                    "type": "node_symbol", "layers": ["Object"], "property": "nodeSymbol",
                    "origin": {"class": "chair"} if case == "mapping" else {},
                }}},
                "goal": {"fact": "visited-object", "arguments": [{"result": "object"}]},
            },
        )
        query = (
            "MATCH (n:Object) RETURN n.nodeSymbol AS sym "
            "UNION MATCH (m:Object) RETURN m.nodeSymbol AS sym"
            if case == "mapping" else "MATCH (o:Object) RETURN o.nodeSymbol AS object"
        )
        rows = [{"sym": "O1"}] if case == "mapping" else [{"object": "O2"}]
        expected_status = "mapping_unresolved" if case == "mapping" else "mismatch"
        answer = "(visited-object O1)"
    decision, client = run_interaction(
        seed, [tool_assistant(query), final_assistant(answer)],
        validator=FakeValidator([{"executable": True, "rows": rows, "error": None}]),
        max_tool_calls=1,
    )
    assert len(client.calls) == 2
    assert json.loads(client.calls[1]["messages"][-1]["content"]) == {"ok": True, "rows": rows}
    assert decision.accepted is False
    assert decision.reason == expected_status
    assert decision.record.validation.result_validation["status"] == expected_status
    assert "error_type" not in json.dumps(client.calls)
    assert "validation_contract" not in json.dumps(client.calls)


def test_tool_feedback_is_identical_when_only_the_expected_answer_changes():
    requests = []
    for expected in (3, 4):
        decision, client = run_interaction(
            scalar_contract_seed(solution=expected), [tool_assistant(), final_assistant()],
            max_tool_calls=1,
        )
        assert decision.accepted is (expected == 3)
        requests.append(client.calls)
    assert requests[0] == requests[1]


def test_exploratory_queries_are_ungraded_until_the_final_answer(monkeypatch):
    from cypher_3dsg_agent_distillation.cypher import result_validation

    original = result_validation.validate_record_result
    graded_queries = []

    def grade(record, rows, query, columns=None):
        graded_queries.append(query)
        return original(record, rows, query, columns)

    monkeypatch.setattr(result_validation, "validate_record_result", grade)
    queries = [
        "MATCH (o:Object) RETURN o.class AS class",
        "MATCH (o:Object) RETURN o.nodeSymbol AS id, o.class AS class",
        "MATCH (o:Object) RETURN count(o) AS count",
    ]
    rows = [[{"class": "chair"}], [{"id": "O1", "class": "chair"}], [{"count": 3}]]
    decision, client = run_interaction(
        scalar_contract_seed(), [*(tool_assistant(query) for query in queries), final_assistant()],
        validator=FakeValidator([{"executable": True, "rows": value} for value in rows]),
    )
    assert decision.accepted
    assert graded_queries == [queries[-1]]
    for call, expected_rows in zip(client.calls[1:], rows, strict=True):
        assert json.loads(call["messages"][-1]["content"]) == {"ok": True, "rows": expected_rows}
    assert decision.record.metadata["successful_tool_call_index"] == 3


def test_final_grading_uses_latest_execution_without_selecting_by_gold():
    decision, client = run_interaction(
        scalar_contract_seed(), [tool_assistant(), tool_assistant(), final_assistant()],
        validator=FakeValidator([
            {"executable": True, "rows": [{"count": 3}]},
            {"executable": True, "rows": [{"count": 2}]},
        ]),
    )
    assert len(client.calls) == 3
    assert decision.reason == "mismatch"
    assert decision.record.validation.rows == [{"count": 2}]
    assert decision.record.metadata["successful_tool_call_index"] == 2


def test_legacy_result_shape_failure_is_internal_after_final_answer(monkeypatch):
    def decode(*args, **kwargs):
        raise openrouter_teacher.ResultShapeError("Unsupported internal result shape")

    monkeypatch.setattr(openrouter_teacher, "rows_to_solution_text", decode)
    decision, client = run_interaction(seed_record(), [tool_assistant(), final_assistant()])
    assert len(client.calls) == 2
    assert json.loads(client.calls[-1]["messages"][-1]["content"]) == {
        "ok": True, "rows": [{"count": 3}],
    }
    assert decision.reason == "result_shape_error"
    assert "Unsupported internal result shape" not in json.dumps(client.calls)


@pytest.mark.parametrize(
    "text", ["<answer>3</answer><answer>4</answer>", "<answer><answer>3</answer></answer>"]
)
def test_answer_blocks_must_be_unique(text):
    assert openrouter_teacher.answer_inner_text(text) is None


def test_real_pddl_aliases_use_top_level_contract_without_legacy_field_hints():
    query = (
        'MATCH (a:Object {nodeSymbol:"O1"}), (b:Object {nodeSymbol:"O2"}) '
        "RETURN a.nodeSymbol AS stop, b.nodeSymbol AS finish"
    )
    goal = "(and (visited-object O1) (at-object O2))"
    c = {
        "result": {
            "kind": "role_sets",
            "roles": {
                "waypoint": {
                    "type": "node_symbol",
                    "layers": ["Object"],
                    "property": "nodeSymbol",
                    "origin": {"nodeSymbol": "O1"},
                },
                "destination": {
                    "type": "node_symbol",
                    "layers": ["Object"],
                    "property": "nodeSymbol",
                    "origin": {"nodeSymbol": "O2"},
                },
            },
        },
        "goal": {
            "operator": "and",
            "clauses": [
                {"fact": "visited-object", "arguments": [{"result": "waypoint"}]},
                {"fact": "at-object", "arguments": [{"result": "destination"}]},
            ],
        },
    }
    seed = seed_record(
        task_family="cypher_pddl",
        output_type="PDDL",
        cypher=query,
        solution=goal,
        validation_contract=c,
        grounding={"expected": {"waypoint": ["O1"], "destination": ["O2"]}},
    )
    decision, client = run_interaction(
        seed,
        [tool_assistant(query), final_assistant(goal)],
        validator=FakeValidator([{"executable": True, "rows": [{"stop": "O1", "finish": "O2"}]}]),
    )
    assert decision.accepted, decision.reason
    assert decision.record.validation.result_validation["bindings"] == {
        "waypoint": "stop",
        "destination": "finish",
    }
    assert "validation_contract" not in json.dumps(client.calls)


def test_teacher_accepts_typed_diagnostic_columns_without_changing_tool_feedback():
    seed = scalar_contract_seed()
    query = "MATCH (o:Object) RETURN count(o) AS total, 'diagnostic' AS note"
    rows = [{"total": 3, "note": "diagnostic"}]
    decision, client = run_interaction(
        seed, [tool_assistant(query), final_assistant("3")],
        validator=FakeValidator([{"executable": True, "rows": rows, "error": None}]),
        max_tool_calls=1,
    )
    assert decision.accepted
    result = decision.record.validation.result_validation
    assert result["actual"] == 3
    assert result["auxiliary_columns"] == ["note"]
    assert json.loads(client.calls[1]["messages"][-1]["content"]) == {"ok": True, "rows": rows}


def test_teacher_accepts_pooled_roles_without_exposing_internal_mappings():
    contract = {
        "result": {"kind": "role_sets", "roles": {
            name: {"type": "node_symbol", "layers": ["Object"], "property": "nodeSymbol",
                   "origin": {"nodeSymbol": symbol}}
            for name, symbol in [("first", "O1"), ("second", "O2")]
        }},
        "goal": {"operator": "and", "clauses": [
            {"fact": "visited-object", "arguments": [{"result": "first"}], "multiple": "and"},
            {"fact": "at-object", "arguments": [{"result": "second"}], "multiple": "and"},
        ]},
    }
    goal = "(and (visited-object O1) (at-object O2))"
    seed = seed_record(
        task_family="cypher_pddl", output_type="PDDL", solution=goal,
        grounding={"expected": {"first": ["O1"], "second": ["O2"]}},
        validation_contract=contract,
    )
    query = "MATCH (o:Object) WHERE o.nodeSymbol IN ['O1','O2'] RETURN o.nodeSymbol AS id"
    rows = [{"id": "O1"}, {"id": "O2"}]
    decision, client = run_interaction(
        seed, [tool_assistant(query), final_assistant(goal)],
        validator=FakeValidator([{"executable": True, "rows": rows, "error": None}]),
        max_tool_calls=1,
    )
    assert decision.accepted
    assert decision.record.validation.result_validation["role_rows"] == {
        "first": [0], "second": [1],
    }
    assert json.loads(client.calls[1]["messages"][-1]["content"]) == {"ok": True, "rows": rows}
