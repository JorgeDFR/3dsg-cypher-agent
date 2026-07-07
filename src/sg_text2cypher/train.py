from __future__ import annotations

import argparse
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from sg_text2cypher.config import load_yaml
from sg_text2cypher.dataset import read_jsonl

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
        from trl import SFTTrainer
    except ImportError as exc:
        msg = "Install training dependencies with `pip install -e '.[train]'` before fine-tuning."
        raise RuntimeError(msg) from exc
    return LoraConfig, prepare_model_for_kbit_training, SFTTrainer


def dtype_from_name(name: str | None):
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
        torch_dtype=dtype_from_name(model_config.get("torch_dtype", "auto")),
        quantization_config=quantization_config(model_config),
        device_map="auto",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    return model, tokenizer


def target_modules(value: Any) -> list[str] | str:
    if value is None or value == "auto":
        return DEFAULT_LORA_TARGETS
    if value == "all-linear":
        return "all-linear"
    if isinstance(value, list):
        return value
    raise TypeError("lora.target_modules must be a list, `auto`, or `all-linear`.")


def to_chat_text(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)


def build_dataset(path: str, tokenizer) -> Dataset:
    records = read_jsonl(path)
    return Dataset.from_list(
        [{"text": to_chat_text(tokenizer, record["messages"]), **record} for record in records]
    )


def command(args: argparse.Namespace) -> None:
    LoraConfig, prepare_model_for_kbit_training, SFTTrainer = require_training_dependencies()
    config = load_yaml(args.model_config)
    model, tokenizer = load_model_and_tokenizer(config)

    if config["model"].get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = config["lora"]
    peft_config = LoraConfig(
        r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules(lora_config.get("target_modules", "auto")),
    )

    train_dataset = build_dataset(args.train, tokenizer)
    eval_dataset = build_dataset(args.eval, tokenizer) if args.eval else None
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training_config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        num_train_epochs=float(training_config.get("num_train_epochs", 3)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.03)),
        logging_steps=int(training_config.get("logging_steps", 10)),
        eval_steps=int(training_config.get("eval_steps", 100)),
        save_steps=int(training_config.get("save_steps", 100)),
        save_total_limit=int(training_config.get("save_total_limit", 3)),
        bf16=bool(training_config.get("bf16", True)),
        fp16=bool(training_config.get("fp16", False)),
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        report_to=training_config.get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=int(training_config.get("max_seq_length", 4096)),
        args=training_args,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a text-to-Cypher model.")
    parser.add_argument("--model-config", default="configs/model_gemma4.yaml")
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--eval", default="data/splits/eval.jsonl")
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
