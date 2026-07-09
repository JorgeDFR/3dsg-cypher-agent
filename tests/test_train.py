from __future__ import annotations

from text2cypher_3dsg.dataset import write_jsonl
from text2cypher_3dsg.train import (
    align_model_token_ids,
    build_dataset,
    gemma4_effective_target_modules,
    is_gemma4_effective_model,
    lora_target_modules,
    sft_config_kwargs,
    sft_trainer_kwargs,
    should_upcast_gemma4_effective_param,
    training_arguments_kwargs,
)


class DummyTokenizer:
    chat_template = None


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


def test_training_arguments_kwargs_uses_current_eval_strategy_name():
    kwargs = training_arguments_kwargs(
        {"output_dir": "out", "warmup_steps": 0.03},
        has_eval_dataset=True,
    )

    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["warmup_steps"] == 0.03
    assert "evaluation_strategy" not in kwargs
    assert "warmup_ratio" not in kwargs


def test_sft_config_kwargs_uses_current_sft_config_fields():
    kwargs = sft_config_kwargs(
        {"output_dir": "out", "max_length": 2048},
        has_eval_dataset=True,
    )

    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["dataset_text_field"] == "text"
    assert kwargs["max_length"] == 2048
    assert "max_seq_length" not in kwargs


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
