import json
from copy import deepcopy
from types import SimpleNamespace

from cypher_3dsg_agent_distillation.agent_records import (
    DEFAULT_AGENT_TOOL,
    WRITE_REFUSAL_ANSWER,
)
from cypher_3dsg_agent_distillation.config import load_yaml
from cypher_3dsg_agent_distillation.experiment_profiles import load_experiment_profile
from cypher_3dsg_agent_distillation.openrouter_teacher import (
    OpenRouterTeacherClient,
    configured_teacher_provider,
    debug_trace_record,
    evaluate_candidate_trace,
    normalize_assistant_message,
    progress_description,
    run_synthetic_teacher_interaction,
    run_teacher_interaction,
    seed_expected_solution,
)


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


def run_interaction(seed, responses, *, validator=None, max_tool_calls=3, static_only=False):
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
    )
    return decision, client


def test_teacher_provider_requires_supported_explicit_type():
    assert configured_teacher_provider({"provider": "openrouter"}) == "openrouter"
    assert configured_teacher_provider({"provider": "synthetic"}) == "synthetic"

    try:
        configured_teacher_provider({"provider": "local"})
    except ValueError as exc:
        assert "Unsupported teacher provider" in str(exc)
    else:
        raise AssertionError("Expected unsupported provider error")


def test_synthetic_teacher_config_does_not_define_a_model():
    config = load_yaml("configs/teacher_synthetic.yaml")

    assert configured_teacher_provider(config) == "synthetic"
    assert "model" not in config


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
        [tool_assistant("MATCH (o:Object) RETURN o.nodeSymbol AS object")],
        validator=validator,
        max_tool_calls=1,
    )

    assert not decision.accepted
    assert decision.reason == "max_tool_calls_exceeded"
    assert decision.payload["last_failure_reason"] == "grounding_mismatch"


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
