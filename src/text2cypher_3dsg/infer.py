from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from text2cypher_3dsg.config import ensure_parent, load_yaml
from text2cypher_3dsg.dataset import read_jsonl


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

    dtype_name = model_config.get("torch_dtype", "auto")
    dtype = getattr(torch, dtype_name) if dtype_name not in {None, "auto"} else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
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
    model, tokenizer = load_model_and_tokenizer(config)

    output_path = ensure_parent(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt = render_prompt(tokenizer, record["messages"])
            prediction = generate_cypher(model, tokenizer, prompt, args.max_new_tokens)
            payload = {
                **record,
                "predicted_cypher": prediction,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Cypher predictions for a prepared dataset."
    )
    parser.add_argument("--model-config", default="configs/model_gemma4.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
