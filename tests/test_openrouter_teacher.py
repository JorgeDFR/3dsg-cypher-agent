from cypher_3dsg_agent_distillation.openrouter_teacher import (
    OpenRouterTeacherClient,
    evaluate_candidate_trace,
)


class FakeValidator:
    def validate(self, cypher, execute):
        return {
            "executable": True,
            "rows": [{"count": 3}],
            "error": None,
        }


def seed_record():
    return {
        "uid": "seed",
        "scene_id": "scene",
        "question": "How many objects?",
        "query_type": "object_class_count",
        "cypher": "MATCH (o:Object) RETURN count(o) AS count",
        "solution": 3,
        "read_only": True,
    }


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
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": cypher},
                        }
                    }
                ],
            },
            {"role": "tool", "tool_name": "run_cypher_query", "content": "[{'count': 3}]"},
            {"role": "assistant", "content": "<answer>3</answer>"},
        ]
    }


def test_evaluate_candidate_trace_accepts_valid_teacher_trace():
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


def test_evaluate_candidate_trace_rejects_static_issue():
    decision = evaluate_candidate_trace(
        seed_record(),
        candidate_trace("MATCH (o:Object) DELETE o"),
        teacher_model="openai/gpt-oss-120b",
        validator=FakeValidator(),
        policy=None,
    )

    assert not decision.accepted
    assert decision.reason == "static_issues"


def test_openrouter_preflight_reports_missing_model(monkeypatch):
    client = OpenRouterTeacherClient(api_key="key", model="openai/gpt-oss-120b")
    monkeypatch.setattr(client, "get_json", lambda _url: {"data": [{"id": "other/model"}]})

    try:
        client.preflight()
    except RuntimeError as exc:
        assert "openai/gpt-oss-120b" in str(exc)
    else:
        raise AssertionError("Expected missing model error")
