from copy import deepcopy

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord
from cypher_3dsg_agent_distillation.agents.agent_rendering import render_custom_xml
from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
    normalize_assistant_message,
    sdk_outbound_messages,
)
from cypher_3dsg_agent_distillation.agents.reasoning import (
    reasoning_text,
    split_generated_reasoning,
    student_messages,
    without_reasoning,
)
from cypher_3dsg_agent_distillation.training.train import build_training_example
from tests.helpers import answerable_trace


class TemplateTokenizer:
    """Character tokens isolate template/loss behavior without loading model weights."""

    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, tools=None, chat_template=None
    ):
        assert not tokenize
        return _compile_jinja_template(chat_template or self.chat_template).render(
            messages=messages,
            tools=tools or [],
            add_generation_prompt=add_generation_prompt,
        )

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_reasoning_normalization_preserves_provider_blocks_and_avoids_replay_duplicates():
    blocks = [
        {"type": "reasoning.text", "text": "Use Object nodes.", "signature": "signed", "index": 0},
        {"type": "reasoning.encrypted", "data": "opaque", "index": 1},
    ]
    message = normalize_assistant_message(
        {"role": "assistant", "content": None, "reasoning_details": deepcopy(blocks)}
    )
    assert message["reasoning"] == "Use Object nodes."
    assert message["reasoning_details"] == blocks
    outbound = sdk_outbound_messages([message])[0]
    assert outbound["reasoning_details"] == blocks
    assert "reasoning" not in outbound
    assert "reasoning_kind" not in outbound


def test_encrypted_reasoning_never_becomes_student_text():
    message = {
        "role": "assistant",
        "content": "<answer>3</answer>",
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
    }
    assert reasoning_text(message) == ("", None)
    assert "opaque" not in str(student_messages([message], object(), include_reasoning=True))


def test_excluding_reasoning_removes_unfinished_blocks_and_provider_fields():
    message = {
        "role": "assistant", "content": "<think>unfinished thought",
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
        "reasoning_content": "another thought",
    }
    assert without_reasoning(message) == {"role": "assistant", "content": ""}


def test_summary_is_distinguished_from_raw_reasoning():
    assert reasoning_text(
        {
            "reasoning_details": [
                {"type": "reasoning.summary", "summary": "Checked the schema."},
            ]
        }
    ) == ("Checked the schema.", "summary")


@pytest.mark.parametrize(
    "text,started,status,action",
    [
        ("<think>Use Object.</think><answer>3</answer>", False, "complete", "<answer>3</answer>"),
        ("Use Object.</think><answer>3</answer>", True, "complete", "<answer>3</answer>"),
        ("<think>MATCH (o:Object) RETURN o", False, "incomplete", ""),
        ("MATCH (o:Object) RETURN o", True, "incomplete", ""),
        ("<answer>3</answer>", False, "absent", "<answer>3</answer>"),
        (
            "<think>first</think><think>second</think><answer>3</answer>",
            False,
            "complete",
            "<answer>3</answer>",
        ),
        ("<think>first</think><think>unfinished", False, "incomplete", ""),
    ],
)
def test_unfinished_thinking_cannot_be_executed(text, started, status, action):
    parsed, _reasoning, found = split_generated_reasoning(text, thinking_started=started)
    assert (parsed, found) == (action, status)


@pytest.mark.parametrize("view", ["native", "custom"])
@pytest.mark.parametrize("native_reasoning", [True, False])
def test_both_reasoning_stages_are_supervised_and_off_mode_removes_them(view, native_reasoning):
    # This captures the Qwen historical-thought branch that XML tool feedback changes.
    template = (
        (
            "{%- set ns = namespace(last_query_index=0) %}"
            "{%- for message in messages %}"
            "{%- if message.role == 'user' %}{%- set ns.last_query_index=loop.index0 %}"
            "{%- endif %}{%- endfor %}"
            "{%- for message in messages %}"
            "{{ message.role }}:"
            "{%- if loop.index0 > ns.last_query_index %}"
            "{{ message.reasoning_content|default('') }}"
            "{%- endif %}"
            "{{ message.content|default('') }}{{ message.tool_calls|default('') }}\n"
            "{%- endfor %}"
        )
        if native_reasoning
        else (
            "{% for m in messages %}{{ m.role }}:{{ m.content }}"
            "{{ m.tool_calls|default('') }}\n{% endfor %}"
        )
    )
    tokenizer = TemplateTokenizer(template)
    payload = answerable_trace()
    payload["messages"][2]["reasoning"] = "QUERY_REASONING"
    payload["messages"][-1]["reasoning"] = "ANSWER_REASONING"
    if view == "custom":
        payload = render_custom_xml(AgentTraceRecord.model_validate(payload))
    for include in (False, True):
        example = build_training_example(
            payload,
            tokenizer,
            loss_mode="assistant-only",
            max_length=None,
            include_reasoning=include,
        )
        active = "".join(chr(label) for label in example["labels"] if label != -100)
        assert ("QUERY_REASONING" in active) is include
        assert ("ANSWER_REASONING" in active) is include
        assert "<answer>3</answer>" in active
        assert "[{'count': 3}]" not in active


def test_injected_faulty_turn_stays_masked_with_reasoning_enabled():
    payload = answerable_trace()
    payload["messages"][2]["reasoning"] = "FAULTY_REASONING"
    payload["messages"][-1]["reasoning"] = "FINAL_REASONING"
    payload["metadata"] = {"non_trainable_message_indices": [2, 3]}
    tokenizer = TemplateTokenizer(
        "{% for m in messages %}{{ m.role }}:{{ m.content }}"
        "{{ m.tool_calls|default('') }}\n{% endfor %}"
    )
    example = build_training_example(
        payload, tokenizer, loss_mode="assistant-only", max_length=None, include_reasoning=True
    )
    active = "".join(chr(label) for label in example["labels"] if label != -100)
    assert "FAULTY_REASONING" not in active
    assert "FINAL_REASONING" in active
