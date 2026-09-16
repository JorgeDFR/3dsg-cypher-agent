from __future__ import annotations

from copy import deepcopy

import pytest

from cypher_3dsg_agent_distillation.core.io import write_jsonl
from cypher_3dsg_agent_distillation.training.train import (
    ASSISTANT_ONLY_LOSS_MODE,
    FULL_LOSS_MODE,
    OverlengthTrainingTraceError,
    OverlengthTrainingTraceWarning,
    align_model_token_ids,
    build_dataset,
    build_training_dataset,
    build_training_example,
    enable_gradient_checkpointing_for_kbit_training,
    gemma4_effective_target_modules,
    is_gemma4_effective_model,
    lora_target_modules,
    pad_causal_lm_features,
    prepare_quantized_model_for_training,
    sft_config_kwargs,
    sft_trainer_kwargs,
    should_upcast_gemma4_effective_param,
    training_arguments_kwargs,
    training_tool_schemas,
)


class DummyTokenizer:
    chat_template = None


class CharTokenizer(DummyTokenizer):
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, input_ids, skip_special_tokens=False):
        return "".join(chr(input_id) for input_id in input_ids)


class NativeToolTokenizer(CharTokenizer):
    chat_template = "native-tool-template"

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        tools=None,
    ):
        assert tokenize is False
        assert add_generation_prompt is False
        self.template_calls.append({"messages": deepcopy(messages), "tools": tools})
        tool_names = [tool["function"]["name"] for tool in tools or []]
        prefix = f"TOOLS:{','.join(tool_names)}\n" if tool_names else ""
        body = "\n".join(
            f"{message['role']}:{message.get('content') or message.get('tool_calls') or ''}"
            for message in messages
        )
        return prefix + body


def active_label_text(example, tokenizer):
    return tokenizer.decode(
        [
            input_id
            for input_id, label in zip(example["input_ids"], example["labels"], strict=True)
            if label != -100
        ],
        skip_special_tokens=False,
    )


def test_build_dataset_only_keeps_rendered_text_for_mixed_metadata(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(
        [
            {
                "uid": "one",
                "solution": "O1",
                "messages": [
                    {"role": "user", "content": "Question one"},
                    {"role": "assistant", "content": "MATCH (n) RETURN n"},
                ],
            },
            {
                "uid": "two",
                "solution": 4,
                "messages": [
                    {"role": "user", "content": "Question two"},
                    {"role": "assistant", "content": "MATCH (n) RETURN count(n)"},
                ],
            },
        ],
        path,
    )

    dataset = build_dataset(str(path), DummyTokenizer())

    assert dataset.column_names == ["text"]
    assert len(dataset) == 2
    assert dataset[0]["text"] == "user: Question one\n\nassistant: MATCH (n) RETURN n"


def test_build_training_example_masks_context_and_supervises_assistant_turns():
    tokenizer = CharTokenizer()
    record = {
        "uid": "one",
        "messages": [
            {"role": "user", "content": "Question one"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": "MATCH (n) RETURN n"},
                        }
                    }
                ],
            },
            {"role": "tool", "tool_name": "run_cypher_query", "content": "[{'n': 1}]"},
            {"role": "assistant", "content": "<answer>O1</answer>"},
        ],
    }

    example = build_training_example(
        record,
        tokenizer,
        loss_mode=ASSISTANT_ONLY_LOSS_MODE,
        max_length=4096,
    )
    supervised_text = active_label_text(example, tokenizer)

    assert "run_cypher_query" in supervised_text
    assert "MATCH (n) RETURN n" in supervised_text
    assert "<answer>O1</answer>" in supervised_text
    assert "Question one" not in supervised_text
    assert "[{'n': 1}]" not in supervised_text


@pytest.mark.parametrize("loss_mode", [ASSISTANT_ONLY_LOSS_MODE, FULL_LOSS_MODE])
def test_build_training_example_masks_controlled_repair_context(loss_mode):
    tokenizer = CharTokenizer()
    record = {
        "uid": "repair",
        "metadata": {"non_trainable_message_indices": [2, 3]},
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": "MATC (n) RETURN n"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Syntax error near MATC"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": "MATCH (n) RETURN n"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "[]"},
            {"role": "assistant", "content": "<answer><></answer>"},
        ],
    }

    example = build_training_example(
        record,
        tokenizer,
        loss_mode=loss_mode,
        max_length=4096,
    )
    supervised_text = active_label_text(example, tokenizer)

    assert "MATC (n) RETURN n" not in supervised_text
    assert "Syntax error near MATC" not in supervised_text
    assert "MATCH (n) RETURN n" in supervised_text
    assert "<answer><></answer>" in supervised_text


def test_training_renders_native_tool_schema_once_without_mutating_messages():
    tokenizer = NativeToolTokenizer()
    record = {
        "uid": "one",
        "tools": ["run_cypher_query", "run_cypher_query"],
        "messages": [
            {"role": "user", "content": "Question one"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": "MATCH (n) RETURN n"},
                        }
                    }
                ],
            },
            {"role": "tool", "tool_name": "run_cypher_query", "content": "[]"},
            {"role": "assistant", "content": "<answer>0</answer>"},
        ],
    }
    original_messages = deepcopy(record["messages"])

    example = build_training_example(
        record,
        tokenizer,
        loss_mode=ASSISTANT_ONLY_LOSS_MODE,
        max_length=4096,
    )
    rendered = tokenizer.decode(example["input_ids"])

    assert rendered.count("TOOLS:run_cypher_query") == 1
    assert record["messages"] == original_messages
    assert tokenizer.template_calls
    for call in tokenizer.template_calls:
        assert len(call["tools"]) == 1
        assert call["tools"][0]["function"]["name"] == "run_cypher_query"
    schemas = training_tool_schemas(record)
    assert len(schemas) == 1
    assert schemas[0]["function"]["parameters"]["required"] == ["cypher_string"]


def test_training_arguments_kwargs_uses_current_eval_strategy_name():
    kwargs = training_arguments_kwargs(
        {
            "output_dir": "out",
            "warmup_steps": 0.05,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        },
        has_eval_dataset=True,
    )

    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["disable_tqdm"] is True
    assert kwargs["warmup_steps"] == 0.05
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False
    assert "evaluation_strategy" not in kwargs
    assert "warmup_ratio" not in kwargs


def test_training_arguments_kwargs_defaults_to_five_percent_warmup():
    kwargs = training_arguments_kwargs({"output_dir": "out"}, has_eval_dataset=False)

    assert kwargs["warmup_steps"] == 0.05


@pytest.mark.parametrize("warmup_steps", [-0.01, 1.01, True])
def test_training_arguments_kwargs_rejects_invalid_warmup_steps(warmup_steps):
    with pytest.raises(ValueError, match="warmup_steps must be"):
        training_arguments_kwargs(
            {"output_dir": "out", "warmup_steps": warmup_steps},
            has_eval_dataset=False,
        )


def test_training_arguments_kwargs_accepts_exact_warmup_step_count():
    kwargs = training_arguments_kwargs(
        {"output_dir": "out", "warmup_steps": 125},
        has_eval_dataset=False,
    )

    assert kwargs["warmup_steps"] == 125


def test_training_arguments_kwargs_maps_legacy_warmup_ratio():
    kwargs = training_arguments_kwargs(
        {"output_dir": "out", "warmup_ratio": 0.1},
        has_eval_dataset=False,
    )

    assert kwargs["warmup_steps"] == 0.1
    assert "warmup_ratio" not in kwargs


@pytest.mark.parametrize("use_gradient_checkpointing", [False, True])
def test_quantized_model_preparation_respects_gradient_checkpointing_setting(
    use_gradient_checkpointing,
):
    model = object()
    calls = []

    def prepare_model_for_kbit_training(prepared_model, **kwargs):
        calls.append((prepared_model, kwargs))
        return "prepared"

    result = prepare_quantized_model_for_training(
        model,
        {"base_model": "Qwen/Qwen3.5-4B"},
        prepare_model_for_kbit_training,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    assert result == "prepared"
    assert calls == [
        (
            model,
            {
                "use_gradient_checkpointing": use_gradient_checkpointing,
                "gradient_checkpointing_kwargs": (
                    {"use_reentrant": False} if use_gradient_checkpointing else None
                ),
            },
        )
    ]


def test_gradient_checkpointing_kwargs_are_forwarded_to_transformers_api():
    class Model:
        received_kwargs = None

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.received_kwargs = gradient_checkpointing_kwargs

    model = Model()
    enable_gradient_checkpointing_for_kbit_training(
        model,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    assert model.received_kwargs == {"use_reentrant": False}


def test_sft_config_kwargs_uses_current_sft_config_fields():
    kwargs = sft_config_kwargs(
        {"output_dir": "out", "max_length": 2048, "loss_mode": FULL_LOSS_MODE},
        has_eval_dataset=True,
    )

    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["max_length"] == 2048
    assert kwargs["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert kwargs["remove_unused_columns"] is False
    assert "dataset_text_field" not in kwargs
    assert "max_seq_length" not in kwargs


def test_sft_config_kwargs_skips_trl_preparation_for_pretokenized_labels():
    kwargs = sft_config_kwargs(
        {"output_dir": "out", "max_length": 2048, "loss_mode": ASSISTANT_ONLY_LOSS_MODE},
        has_eval_dataset=True,
    )

    assert kwargs["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert kwargs["remove_unused_columns"] is False
    assert "dataset_text_field" not in kwargs


def test_pad_causal_lm_features_restores_missing_attention_mask():
    tokenizer = CharTokenizer()

    batch = pad_causal_lm_features(
        [
            {"input_ids": [2, 3], "labels": [2, 3]},
            {"input_ids": [4], "labels": [4]},
        ],
        tokenizer,
    )

    assert batch == {
        "input_ids": [[2, 3], [4, 0]],
        "attention_mask": [[1, 1], [1, 0]],
        "labels": [[2, 3], [4, -100]],
    }


def test_sft_trainer_kwargs_uses_current_processing_class_parameter():
    kwargs = sft_trainer_kwargs(
        model="model",
        tokenizer="tokenizer",
        train_dataset="train",
        eval_dataset=None,
        training_args="args",
        peft_config="peft",
    )

    assert kwargs["processing_class"] == "tokenizer"
    assert "tokenizer" not in kwargs
    assert "dataset_text_field" not in kwargs
    assert "max_seq_length" not in kwargs


def test_sft_trainer_kwargs_passes_configured_callbacks():
    callback = object()

    kwargs = sft_trainer_kwargs(
        model="model",
        tokenizer="tokenizer",
        train_dataset="train",
        eval_dataset="eval",
        training_args="args",
        peft_config="peft",
        callbacks=[callback],
    )

    assert kwargs["callbacks"] == [callback]


def test_build_training_example_rejects_overlength_trace_without_truncating():
    with pytest.raises(OverlengthTrainingTraceError) as exc_info:
        build_training_example(
            {
                "uid": "too-long",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "A" * 100},
                ],
            },
            CharTokenizer(),
            loss_mode=ASSISTANT_ONLY_LOSS_MODE,
            max_length=40,
        )

    assert exc_info.value.uid == "too-long"
    assert exc_info.value.token_length > exc_info.value.max_length


def test_build_training_dataset_rejects_overlength_traces_with_total_warning(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(
        [
            {
                "uid": "retained",
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ],
            },
            {
                "uid": "rejected",
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A" * 100},
                ],
            },
        ],
        path,
    )

    with pytest.warns(OverlengthTrainingTraceWarning, match="Rejected 1 of 2 traces"):
        dataset = build_training_dataset(
            str(path),
            CharTokenizer(),
            loss_mode=ASSISTANT_ONLY_LOSS_MODE,
            max_length=40,
        )

    assert len(dataset) == 1
    assert dataset[0]["uid"] == "retained"
    assert len(dataset[0]["input_ids"]) < 40


def test_build_full_loss_dataset_also_rejects_overlength_traces(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(
        [
            {"uid": "retained", "text": "short"},
            {"uid": "rejected", "text": "x" * 41},
        ],
        path,
    )

    with pytest.warns(OverlengthTrainingTraceWarning, match="Rejected 1 of 2 traces"):
        dataset = build_training_dataset(
            str(path),
            CharTokenizer(),
            loss_mode=FULL_LOSS_MODE,
            max_length=40,
        )

    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == dataset[0]["labels"]
    assert CharTokenizer().decode(dataset[0]["input_ids"]) == "short"


def test_align_model_token_ids_updates_model_and_generation_config():
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2

    class Config:
        pad_token_id = None
        eos_token_id = None
        bos_token_id = None

    class Model:
        config = Config()
        generation_config = Config()

    model = Model()

    align_model_token_ids(model, Tokenizer())

    assert model.config.pad_token_id == 0
    assert model.config.eos_token_id == 1
    assert model.config.bos_token_id == 2
    assert model.generation_config.pad_token_id == 0
    assert model.generation_config.eos_token_id == 1
    assert model.generation_config.bos_token_id == 2


def test_is_gemma4_effective_model_matches_e2b_and_e4b_ids():
    assert is_gemma4_effective_model(object(), {"base_model": "google/gemma-4-E2B-it"})
    assert is_gemma4_effective_model(object(), {"base_model": "google/gemma-4-E4B-it"})
    assert not is_gemma4_effective_model(object(), {"base_model": "google/gemma-4-12B-it"})


def test_is_gemma4_effective_model_matches_ple_config_for_local_paths():
    class TextConfig:
        vocab_size_per_layer_input = 262144
        hidden_size_per_layer_input = 256
        num_hidden_layers = 35

    class Config:
        model_type = "gemma4"
        text_config = TextConfig()

    class Model:
        config = Config()

    assert is_gemma4_effective_model(Model(), {"base_model": "/models/local-gemma"})


def test_gemma4_effective_precision_policy_only_upcasts_norm_params():
    class Param:
        pass

    assert should_upcast_gemma4_effective_param("model.layers.0.input_layernorm.weight", Param())
    assert not should_upcast_gemma4_effective_param(
        "model.embed_tokens_per_layer.weight",
        Param(),
    )


def test_gemma4_effective_target_modules_resolves_inner_text_linears():
    class Linear4bit:
        pass

    class Gemma4ClippableLinear:
        pass

    class Model:
        def named_modules(self):
            return iter(
                [
                    ("model.language_model.layers.0.self_attn.q_proj", Gemma4ClippableLinear()),
                    ("model.language_model.layers.0.self_attn.q_proj.linear", Linear4bit()),
                    ("model.language_model.layers.0.mlp.down_proj.linear", Linear4bit()),
                    ("model.vision_tower.layers.0.self_attn.q_proj.linear", Linear4bit()),
                    ("model.audio_tower.layers.0.self_attn.q_proj.linear", Linear4bit()),
                ]
            )

    assert gemma4_effective_target_modules(Model(), "auto") == [
        "model.language_model.layers.0.self_attn.q_proj.linear",
        "model.language_model.layers.0.mlp.down_proj.linear",
    ]


def test_lora_target_modules_uses_default_suffixes_for_non_gemma4_effective_models():
    assert lora_target_modules(object(), {"base_model": "google/gemma-4-12B-it"}, "auto") == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
