from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from cypher_3dsg_agent_distillation.agents.raw_generation_runtimes import (
    DEFAULT_OLLAMA_CONTEXT_LENGTH,
    OllamaRawGenerationRuntime,
    ollama_messages,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RecordingOpener:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, request, *, timeout):
        payload = json.loads(request.data) if request.data is not None else None
        self.requests.append((request.full_url, payload, timeout))
        return FakeResponse(next(self.responses))


def test_default_ollama_context_length_covers_long_challenge_conversations():
    assert DEFAULT_OLLAMA_CONTEXT_LENGTH == 16_384


def test_ollama_messages_preserve_native_tool_history():
    messages = ollama_messages(
        [
            {"role": "user", "content": "query"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "not sent",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": "MATCH (n) RETURN n"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "run_cypher_query",
                "content": '{"ok": true}',
            },
        ]
    )

    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "cypher_string": "MATCH (n) RETURN n"
    }
    assert "reasoning" not in messages[1]
    assert messages[2]["tool_name"] == "run_cypher_query"


def test_ollama_runtime_validates_server_and_registered_model():
    opener = RecordingOpener(
        {"version": "0.33.3"},
        {"details": {"format": "gguf", "quantization_level": "Q4_K_M"}},
    )
    runtime = OllamaRawGenerationRuntime(
        model="student-q4",
        host="http://ollama:11434/",
        opener=opener,
    )

    runtime.validate()

    assert [request[0] for request in opener.requests] == [
        "http://ollama:11434/api/version",
        "http://ollama:11434/api/show",
    ]
    assert opener.requests[1][1] == {"model": "student-q4"}
    assert runtime.metadata()["server_version"] == "0.33.3"
    assert runtime.metadata()["model_details"]["quantization_level"] == "Q4_K_M"


def test_ollama_runtime_returns_native_tool_calls_and_runtime_metrics():
    opener = RecordingOpener(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": "MATCH (n) RETURN n"},
                        }
                    }
                ],
            },
            "prompt_eval_count": 100,
            "eval_count": 17,
            "done_reason": "stop",
            "total_duration": 123,
        }
    )
    runtime = OllamaRawGenerationRuntime(
        model="student-q4",
        host="http://ollama:11434",
        temperature=0.0,
        seed=123,
        context_length=4096,
        keep_alive="10m",
        opener=opener,
    )
    progress = SimpleNamespace(updates=[])
    progress.update = lambda task_id, **kwargs: progress.updates.append((task_id, kwargs))

    result = runtime.generate(
        [{"role": "user", "content": "query"}],
        max_new_tokens=256,
        enable_thinking=False,
        tools=[{"type": "function", "function": {"name": "run_cypher_query"}}],
        progress=progress,
        progress_task_id=9,
    )

    request = opener.requests[0][1]
    assert request["stream"] is False
    assert request["think"] is False
    assert request["options"] == {
        "temperature": 0.0,
        "seed": 123,
        "num_predict": 256,
        "num_ctx": 4096,
    }
    assert request["keep_alive"] == "10m"
    assert result.prompt_tokens == 100
    assert result.generated_tokens == 17
    assert result.tool_calls[0]["function"]["name"] == "run_cypher_query"
    assert result.provider_metadata == {"done_reason": "stop", "total_duration": 123}
    assert progress.updates[-1] == (9, {"completed": 17})


def test_ollama_runtime_reports_connection_failures_with_host():
    def unavailable(_request, *, timeout):
        raise URLError("connection refused")

    runtime = OllamaRawGenerationRuntime(
        model="student-q4",
        host="http://ollama:11434",
        opener=unavailable,
    )

    with pytest.raises(RuntimeError, match="Could not reach Ollama at http://ollama:11434"):
        runtime.validate()


def test_ollama_cache_identity_uses_registered_digest():
    opener = RecordingOpener(
        {"models": [{"name": "student:latest", "digest": "sha256:new"}]},
        {"version": "test-server"},
    )
    runtime = OllamaRawGenerationRuntime(model="student", opener=opener)
    assert runtime.cache_identity() == {
        "model": "student:latest", "digest": "sha256:new", "server_version": "test-server",
    }


def test_ollama_cache_identity_rejects_missing_registration():
    runtime = OllamaRawGenerationRuntime(model="missing", opener=RecordingOpener({"models": []}))
    with pytest.raises(RuntimeError, match="No registered Ollama model digest"):
        runtime.cache_identity()
