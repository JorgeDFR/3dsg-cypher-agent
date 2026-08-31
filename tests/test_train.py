from __future__ import annotations

from copy import deepcopy

from cypher_3dsg_agent_distillation.io import write_jsonl
from cypher_3dsg_agent_distillation.train import (
    ASSISTANT_ONLY_LOSS_MODE,
    FULL_LOSS_MODE,
    align_model_token_ids,
    build_dataset,
    build_training_example,
    gemma4_effective_target_modules,
    is_gemma4_effective_model,
    lora_target_modules,
    pad_causal_lm_features,
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
            "warmup_steps": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        },
        has_eval_dataset=True,
    )

    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["disable_tqdm"] is True
    assert kwargs["warmup_steps"] == 2
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False
    assert "evaluation_strategy" not in kwargs
    assert "warmup_ratio" not in kwargs


def test_sft_config_kwargs_uses_current_sft_config_fields():
    kwargs = sft_config_kwargs(
        {"output_dir": "out", "max_length": 2048, "loss_mode": FULL_LOSS_MODE},
        has_eval_dataset=True,
    )

    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["dataset_text_field"] == "text"
    assert kwargs["max_length"] == 2048
    assert "max_seq_length" not in kwargs


def test_sft_config_kwargs_skips_trl_preparation_for_assistant_only_labels():
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
