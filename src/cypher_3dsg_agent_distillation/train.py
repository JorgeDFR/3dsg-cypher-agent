from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from cypher_3dsg_agent_distillation.config import load_yaml
from cypher_3dsg_agent_distillation.io import read_jsonl

DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def require_training_dependencies():
    try:
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        msg = "Install training dependencies with `pip install -e '.[train]'` before fine-tuning."
        raise RuntimeError(msg) from exc
    return LoraConfig, prepare_model_for_kbit_training, SFTConfig, SFTTrainer


def dtype_from_name(name: str | None):
    import torch

    if name in {None, "auto"}:
        return "auto"
    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype: {name}")
    return getattr(torch, name)


def quantization_config(model_config: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not model_config.get("load_in_4bit", False):
        return None
    compute_dtype = dtype_from_name(model_config.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(model_config.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(config: dict[str, Any]):
    model_config = config["model"]
    model_id = model_config["base_model"]
    tokenizer_id = model_config.get("tokenizer") or model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype_from_name(model_config.get("dtype", "auto")),
        quantization_config=quantization_config(model_config),
        device_map="auto",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    align_model_token_ids(model, tokenizer)
    return model, tokenizer


def config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def is_gemma4_effective_model(model: Any, model_config: dict[str, Any]) -> bool:
    model_id = str(model_config.get("base_model", "")).lower()
    if "gemma-4-e2b" in model_id or "gemma-4-e4b" in model_id:
        return True

    config = getattr(model, "config", None)
    text_config = config_value(config, "text_config")
    if config_value(config, "model_type") != "gemma4" or text_config is None:
        return False

    ple_fields = (
        "vocab_size_per_layer_input",
        "hidden_size_per_layer_input",
        "num_hidden_layers",
    )
    return all(config_value(text_config, field) is not None for field in ple_fields)


def should_upcast_gemma4_effective_param(name: str, param: Any) -> bool:
    return "norm" in name.lower() and param.__class__.__name__ != "Params4bit"


def enable_gradient_checkpointing_for_kbit_training(
    model: Any,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
) -> None:
    gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
    use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):
            input_embeddings = model.get_input_embeddings()

            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)

            input_embeddings.register_forward_hook(make_inputs_require_grad)

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
        except TypeError:
            model.gradient_checkpointing_enable()


def prepare_gemma4_effective_model_for_kbit_training(model: Any):
    import torch

    for name, param in model.named_parameters():
        param.requires_grad = False
        if (
            should_upcast_gemma4_effective_param(name, param)
            and param.dtype in {torch.float16, torch.bfloat16}
        ):
            param.data = param.data.to(torch.float32)

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        enable_gradient_checkpointing_for_kbit_training(model)
    return model


def prepare_quantized_model_for_training(
    model: Any,
    model_config: dict[str, Any],
    prepare_model_for_kbit_training,
):
    if is_gemma4_effective_model(model, model_config):
        return prepare_gemma4_effective_model_for_kbit_training(model)
    return prepare_model_for_kbit_training(model)


def align_model_token_ids(model, tokenizer) -> None:
    token_id_names = ("pad_token_id", "eos_token_id", "bos_token_id")
    for name in token_id_names:
        value = getattr(tokenizer, name, None)
        if value is None:
            continue
        setattr(model.config, name, value)
        if model.generation_config is not None:
            setattr(model.generation_config, name, value)


def target_modules(value: Any) -> list[str] | str:
    if value is None or value == "auto":
        return DEFAULT_LORA_TARGETS
    if value == "all-linear":
        return "all-linear"
    if isinstance(value, list):
        return value
    raise TypeError("lora.target_modules must be a list, `auto`, or `all-linear`.")


def is_non_text_gemma4_module(name: str) -> bool:
    name_parts = name.lower().split(".")
    return any("audio" in part or "vision" in part for part in name_parts)


def is_supported_lora_linear_module(module: Any) -> bool:
    class_name = module.__class__.__name__
    return "Linear" in class_name and class_name != "Gemma4ClippableLinear"


def gemma4_effective_target_modules(model: Any, value: Any) -> list[str] | str:
    configured_targets = target_modules(value)
    if configured_targets == "all-linear":
        return configured_targets

    target_names = set(configured_targets)
    resolved_targets = []
    for name, module in model.named_modules():
        if not name or is_non_text_gemma4_module(name):
            continue

        name_parts = name.split(".")
        if len(name_parts) >= 2 and name_parts[-1] == "linear" and name_parts[-2] in target_names:
            resolved_targets.append(name)
            continue

        if name_parts[-1] in target_names and is_supported_lora_linear_module(module):
            resolved_targets.append(name)

    return resolved_targets or configured_targets


def lora_target_modules(model: Any, model_config: dict[str, Any], value: Any) -> list[str] | str:
    if is_gemma4_effective_model(model, model_config):
        return gemma4_effective_target_modules(model, value)
    return target_modules(value)


def normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        payload = dict(message)
        if payload.get("content") is None:
            payload["content"] = ""
        normalized.append(payload)
    return normalized


def message_to_plain_text(message: dict[str, Any]) -> str:
    role = message.get("role", "unknown")
    if message.get("tool_calls"):
        return f"{role}: {json.dumps(message['tool_calls'], ensure_ascii=False, sort_keys=True)}"
    return f"{role}: {message.get('content', '')}"


def to_chat_text(tokenizer, messages: list[dict[str, Any]]) -> str:
    messages = normalize_chat_messages(messages)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n\n".join(message_to_plain_text(message) for message in messages)


def build_dataset(path: str, tokenizer) -> Dataset:
    records = read_jsonl(path)
    return Dataset.from_list(
        [
            {"text": record.get("text") or to_chat_text(tokenizer, record["messages"])}
            for record in records
        ]
    )


def training_arguments_kwargs(
    training_config: dict[str, Any],
    has_eval_dataset: bool,
) -> dict[str, Any]:
    return {
        "output_dir": training_config["output_dir"],
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(training_config.get("learning_rate", 2e-4)),
        "num_train_epochs": float(training_config.get("num_train_epochs", 3)),
        "warmup_steps": training_config.get("warmup_steps", 0),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "eval_steps": int(training_config.get("eval_steps", 100)),
        "save_steps": int(training_config.get("save_steps", 100)),
        "save_total_limit": int(training_config.get("save_total_limit", 3)),
        "bf16": bool(training_config.get("bf16", True)),
        "fp16": bool(training_config.get("fp16", False)),
        "gradient_checkpointing": bool(training_config.get("gradient_checkpointing", True)),
        "report_to": training_config.get("report_to", "none"),
        "eval_strategy": "steps" if has_eval_dataset else "no",
    }


def sft_config_kwargs(
    training_config: dict[str, Any],
    has_eval_dataset: bool,
) -> dict[str, Any]:
    kwargs = training_arguments_kwargs(training_config, has_eval_dataset)
    kwargs["dataset_text_field"] = "text"
    kwargs["max_length"] = int(training_config.get("max_length", 4096))
    return kwargs


def sft_trainer_kwargs(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    training_args,
    peft_config,
) -> dict[str, Any]:
    return {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
        "peft_config": peft_config,
        "processing_class": tokenizer,
    }


def command(args: argparse.Namespace) -> None:
    LoraConfig, prepare_model_for_kbit_training, SFTConfig, SFTTrainer = (
        require_training_dependencies()
    )
    config = load_yaml(args.model_config)
    model, tokenizer = load_model_and_tokenizer(config)

    if config["model"].get("load_in_4bit", False):
        model = prepare_quantized_model_for_training(
            model,
            config["model"],
            prepare_model_for_kbit_training,
        )

    lora_config = config["lora"]
    peft_config = LoraConfig(
        r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(
            model,
            config["model"],
            lora_config.get("target_modules", "auto"),
        ),
    )

    train_dataset = build_dataset(args.train, tokenizer)
    eval_dataset = build_dataset(args.eval, tokenizer) if args.eval else None
    training_config = config["training"]
    training_args = SFTConfig(
        **sft_config_kwargs(
            training_config,
            has_eval_dataset=eval_dataset is not None,
        )
    )

    trainer = SFTTrainer(
        **sft_trainer_kwargs(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            peft_config=peft_config,
        )
    )
    trainer.train()
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a 3DSG Cypher tool-using agent model."
    )
    parser.add_argument(
        "--model-config",
        default="configs/student_qwen3_4b.yaml",
    )
    parser.add_argument("--train", default="data/3dsg/agent/splits/train.ollama_native.jsonl")
    parser.add_argument("--eval", default="data/3dsg/agent/splits/eval.ollama_native.jsonl")
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
