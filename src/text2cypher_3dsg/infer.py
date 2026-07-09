from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from text2cypher_3dsg.config import ensure_parent, load_yaml
from text2cypher_3dsg.dataset import read_jsonl

MODEL_SOURCE_BASE = "base"
MODEL_SOURCE_FINETUNED = "finetuned"


def require_torch():
    import torch

    return torch


def require_peft():
    try:
        from peft import PeftModel
    except ImportError as exc:
        msg = "Install training dependencies with `pip install -e '.[train]'` to load adapters."
        raise RuntimeError(msg) from exc
    return PeftModel


def dtype_from_name(name: str | None):
    torch = require_torch()
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


def training_output_dir(config: dict[str, Any]) -> str:
    try:
        return str(config["training"]["output_dir"])
    except KeyError as exc:
        msg = "training.output_dir is required when --model-source finetuned is used."
        raise KeyError(msg) from exc


def tokenizer_id_for_source(config: dict[str, Any], model_source: str) -> str:
    model_config = config["model"]
    if model_config.get("tokenizer"):
        return model_config["tokenizer"]
    if model_source == MODEL_SOURCE_FINETUNED and Path(training_output_dir(config)).exists():
        return training_output_dir(config)
    return model_config["base_model"]


def load_tokenizer(config: dict[str, Any], model_source: str):
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id_for_source(config, model_source),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(config: dict[str, Any]):
    model_config = config["model"]
    model_id = model_config["base_model"]
    dtype = dtype_from_name(model_config.get("dtype", "auto"))

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        quantization_config=quantization_config(model_config),
        device_map="auto",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )


def load_model_and_tokenizer(config: dict[str, Any], model_source: str = MODEL_SOURCE_BASE):
    tokenizer = load_tokenizer(config, model_source)
    model = load_base_model(config)
    if model_source == MODEL_SOURCE_FINETUNED:
        PeftModel = require_peft()
        model = PeftModel.from_pretrained(model, training_output_dir(config))
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    prompt_messages = [message for message in messages if message["role"] != "assistant"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in prompt_messages)


def generate_cypher(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    torch = require_torch()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return clean_generated_cypher(tokenizer.decode(generated, skip_special_tokens=True))


def clean_generated_cypher(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.lower().startswith("cypher"):
        stripped = stripped[len("cypher") :].strip()
    return stripped


def command(args: argparse.Namespace) -> None:
    config = load_yaml(args.model_config)
    records = read_jsonl(args.input)
    model, tokenizer = load_model_and_tokenizer(config, args.model_source)

    output_path = ensure_parent(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in track(records, description="Generating predictions"):
            prompt = render_prompt(tokenizer, record["messages"])
            prediction = generate_cypher(model, tokenizer, prompt, args.max_new_tokens)
            payload = {
                **record,
                "predicted_cypher": prediction,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Cypher predictions for a prepared dataset."
    )
    parser.add_argument("--model-config", default="configs/model_gemma4.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-source",
        choices=[MODEL_SOURCE_BASE, MODEL_SOURCE_FINETUNED],
        default=MODEL_SOURCE_BASE,
        help="Use the configured base model or the LoRA adapter saved in training.output_dir.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
