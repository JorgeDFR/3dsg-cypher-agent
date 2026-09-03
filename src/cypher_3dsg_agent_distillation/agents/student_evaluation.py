from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.core.cli_output import (
    phase_progress,
    print_breakdown,
    print_summary,
)
from cypher_3dsg_agent_distillation.core.config import ensure_parent, load_data_config, load_yaml
from cypher_3dsg_agent_distillation.core.model_runtime_output import (
    configure_model_library_output,
)

EVAL_LOSS_MODE = "eval-loss"
RAW_GENERATION_MODE = "raw-generation"
SUPPORTED_SPLITS = ("train", "eval", "test", "challenge")
ALL_SPLITS = "all"
SPLIT_CHOICES = (*SUPPORTED_SPLITS, ALL_SPLITS)
DEFAULT_ENTROPY_CHUNK_SIZE = 32
FINAL_ANSWER_CHECK = "final_answer_from_trace"
TOOL_CALL_CHECK = "tool_call_from_prompt"
END_TO_END_CHECK = "end_to_end_final_answer"
DEFAULT_MAX_QUERY_RETRIES = 1
DEFAULT_E2E_MAX_TOOL_ROWS = 128
DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS = 16384
DEFAULT_E2E_MAX_QUERY_CHARS = 4096
E2E_REPAIR_MESSAGE = (
    "The query was not executed successfully. Return one corrected "
    "run_cypher_query tool call."
)
GENERATION_CONFIG_REPORT_KEYS = (
    "max_length",
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "eos_token_id",
    "pad_token_id",
)


@dataclass(frozen=True)
class ToolCallParseResult:
    tool_call_generated: bool
    tool_calls: list[dict[str, Any]]
    error: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    raw_text: str
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    max_new_tokens: int | None
    token_budget: int | None


def require_eval_loss_dependencies():
    try:
        import torch
        from peft import PeftModel
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
        )
    except ImportError as exc:
        msg = (
            "Install training dependencies with `pip install -e '.[train]'` "
            "before running student model evaluation."
        )
        raise RuntimeError(msg) from exc
    return torch, PeftModel, DataLoader, AutoModelForCausalLM, AutoTokenizer, (
        DataCollatorForLanguageModeling
    )


def default_adapter_dir(student_config: dict[str, Any]) -> str:
    return str(student_config["training"]["output_dir"])


def tokenizer_source(
    *,
    adapter_dir: str,
    explicit_tokenizer: str | None,
    model_config: dict[str, Any],
    use_adapter: bool = True,
) -> str:
    if explicit_tokenizer:
        return explicit_tokenizer
    if use_adapter and Path(adapter_dir).exists():
        return adapter_dir
    return str(model_config.get("tokenizer") or model_config["base_model"])


def report_output_path(
    *,
    data_config: dict[str, Any],
    split: str,
    mode: str,
    output: str | None,
    baseline: bool = False,
) -> Path:
    if output:
        return Path(output)
    model_name = "baseline" if baseline else "student"
    if split == ALL_SPLITS:
        return Path(data_config["test_report_path"]).with_name(
            f"{model_name}.{mode}.report.json"
        )
    return Path(data_config["test_report_path"]).with_name(
        f"{split}.{model_name}.{mode}.report.json"
    )


def model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def disable_cache_for_scoring(model) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False


def shifted_token_metrics(
    outputs,
    labels,
    *,
    entropy_chunk_size: int = DEFAULT_ENTROPY_CHUNK_SIZE,
) -> dict[str, float | int]:
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    active_mask = shift_labels != -100
    token_count = int(active_mask.sum().item())
    if token_count == 0:
        return {
            "tokens": 0,
            "nll_sum": 0.0,
            "entropy_sum": 0.0,
            "correct_tokens": 0,
        }

    if entropy_chunk_size <= 0:
        entropy_chunk_size = DEFAULT_ENTROPY_CHUNK_SIZE

    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    active_indices = active_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    nll_sum = 0.0
    entropy_sum = 0.0
    correct_tokens = 0
    for start in range(0, token_count, entropy_chunk_size):
        indices = active_indices[start : start + entropy_chunk_size]
        logits = flat_logits.index_select(0, indices)
        targets = flat_labels.index_select(0, indices)
        correct_tokens += int((logits.argmax(dim=-1) == targets).sum().item())

        float_logits = logits.float()
        nll_sum += cross_entropy_sum(float_logits, targets)
        entropy_sum += float(torch_entropy_from_logits(float_logits).sum().item())
        del float_logits, logits, targets, indices

    return {
        "tokens": token_count,
        "nll_sum": nll_sum,
        "entropy_sum": entropy_sum,
        "correct_tokens": correct_tokens,
    }


def cross_entropy_sum(logits, targets) -> float:
    import torch.nn.functional as torch_functional

    return torch_functional.cross_entropy(logits, targets, reduction="sum").item()


def torch_entropy_from_logits(logits):
    probabilities = logits.softmax(dim=-1)
    log_partition = logits.logsumexp(dim=-1)
    expected_logits = (probabilities * logits).sum(dim=-1)
    return log_partition - expected_logits


def finite_perplexity(eval_loss: float) -> float | None:
    if eval_loss >= 700:
        return None
    value = math.exp(eval_loss)
    return value if math.isfinite(value) else None


def load_student_model_and_tokenizer(
    *,
    student_config: dict[str, Any],
    adapter_dir: str,
    explicit_tokenizer: str | None,
    disable_cache: bool = True,
    show_progress: bool = True,
    use_adapter: bool = True,
):
    configure_model_library_output()
    (
        _torch,
        PeftModel,
        _DataLoader,
        AutoModelForCausalLM,
        AutoTokenizer,
        _DataCollatorForLanguageModeling,
    ) = require_eval_loss_dependencies()
    from cypher_3dsg_agent_distillation.training.train import (
        align_model_token_ids,
        dtype_from_name,
        quantization_config,
    )

    model_config = student_config["model"]
    phase_count = 3 if use_adapter else 2
    initial_description = (
        "Loading student tokenizer" if use_adapter else "Loading baseline tokenizer"
    )
    with phase_progress(initial_description, phase_count, enabled=show_progress) as (
        progress,
        task_id,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source(
                adapter_dir=adapter_dir,
                explicit_tokenizer=explicit_tokenizer,
                model_config=model_config,
                use_adapter=use_adapter,
            ),
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if progress is not None:
            progress.update(task_id, advance=1, description="Loading base model weights")

        model = AutoModelForCausalLM.from_pretrained(
            model_config["base_model"],
            dtype=dtype_from_name(model_config.get("dtype", "auto")),
            quantization_config=quantization_config(model_config),
            device_map="auto",
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
        align_model_token_ids(model, tokenizer)
        if disable_cache:
            disable_cache_for_scoring(model)
        if use_adapter:
            if progress is not None:
                progress.update(task_id, advance=1, description="Loading LoRA adapter")
            model = PeftModel.from_pretrained(model, adapter_dir)
            if disable_cache:
                disable_cache_for_scoring(model)
        model.eval()
        if progress is not None:
            progress.update(
                task_id,
                advance=1,
                description=(
                    "Student model loaded" if use_adapter else "Baseline model loaded"
                ),
            )
    return model, tokenizer


def evaluate_eval_loss(
    *,
    data_config: dict[str, Any],
    student_config: dict[str, Any],
    split: str,
    adapter_dir: str,
    explicit_tokenizer: str | None,
    batch_size: int,
    max_length: int,
    loss_mode: str,
    progress=None,
    progress_task_id=None,
    show_progress: bool = False,
) -> dict[str, Any]:
    model, tokenizer = load_student_model_and_tokenizer(
        student_config=student_config,
        adapter_dir=adapter_dir,
        explicit_tokenizer=explicit_tokenizer,
        show_progress=show_progress,
    )
    return evaluate_eval_loss_with_loaded_model(
        data_config=data_config,
        split=split,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        loss_mode=loss_mode,
        progress=progress,
        progress_task_id=progress_task_id,
        show_progress=show_progress,
    )


def evaluate_eval_loss_with_loaded_model(
    *,
    data_config: dict[str, Any],
    split: str,
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
    loss_mode: str,
    progress=None,
    progress_task_id=None,
    show_progress: bool = False,
) -> dict[str, Any]:
    (
        torch,
        _PeftModel,
        DataLoader,
        _AutoModelForCausalLM,
        _AutoTokenizer,
        _DataCollatorForLanguageModeling,
    ) = require_eval_loss_dependencies()
    from cypher_3dsg_agent_distillation.training.train import (
        CausalLmDataCollator,
        build_training_dataset,
    )

    dataset = build_training_dataset(
        data_config["splits"][split],
        tokenizer,
        loss_mode=loss_mode,
        max_length=max_length,
    )
    collator = CausalLmDataCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    total_nll = 0.0
    total_entropy = 0.0
    total_correct_tokens = 0
    total_tokens = 0
    input_device = model_input_device(model)
    progress_context = nullcontext(progress)
    if progress is None and show_progress:
        progress_context = eval_loss_progress_context(True)

    with progress_context as active_progress:
        active_task_id = progress_task_id
        if active_progress is not None and active_task_id is None:
            active_task_id = active_progress.add_task(
                "eval-loss",
                total=1,
                visible=False,
            )
        if active_progress is not None and active_task_id is not None:
            active_progress.update(
                active_task_id,
                total=len(loader),
                completed=0,
                visible=True,
                description=f"Evaluating loss · {split_display_name(split)}",
            )

        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(input_device) for key, value in batch.items()}
                # Transformers casts the complete logits tensor to float32 when labels
                # are supplied. Score supervised tokens in bounded chunks instead.
                labels = batch.pop("labels")
                outputs = model(**batch, use_cache=False)
                metrics = shifted_token_metrics(outputs, labels)
                token_count = metrics["tokens"]
                total_nll += metrics["nll_sum"]
                total_entropy += metrics["entropy_sum"]
                total_correct_tokens += metrics["correct_tokens"]
                total_tokens += token_count
                del outputs
                del labels
                del batch
                if active_progress is not None and active_task_id is not None:
                    active_progress.update(active_task_id, advance=1)

    if total_tokens == 0:
        raise RuntimeError(f"No loss-bearing tokens found for split `{split}`.")

    eval_loss = total_nll / total_tokens
    return {
        "examples": len(dataset),
        "tokens": total_tokens,
        "eval_loss": eval_loss,
        "perplexity": finite_perplexity(eval_loss),
        "eval_entropy": total_entropy / total_tokens,
        "eval_mean_token_accuracy": total_correct_tokens / total_tokens,
    }


def evaluate_eval_loss_splits(
    *,
    data_config: dict[str, Any],
    student_config: dict[str, Any],
    splits: tuple[str, ...],
    adapter_dir: str,
    explicit_tokenizer: str | None,
    batch_size: int,
    max_length: int,
    loss_mode: str,
    progress=None,
    progress_task_id=None,
    show_progress: bool = False,
) -> dict[str, dict[str, Any]]:
    model, tokenizer = load_student_model_and_tokenizer(
        student_config=student_config,
        adapter_dir=adapter_dir,
        explicit_tokenizer=explicit_tokenizer,
        show_progress=show_progress,
    )
    return {
        split: evaluate_eval_loss_with_loaded_model(
            data_config=data_config,
            split=split,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            loss_mode=loss_mode,
            progress=progress,
            progress_task_id=progress_task_id,
            show_progress=show_progress,
        )
        for split in splits
    }


def combine_eval_loss_summaries(
    split_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total_examples = sum(int(summary["examples"]) for summary in split_summaries.values())
    total_tokens = sum(int(summary["tokens"]) for summary in split_summaries.values())
    if total_tokens == 0:
        raise RuntimeError("No loss-bearing tokens found across evaluated splits.")

    eval_loss = (
        sum(
            float(summary["eval_loss"]) * int(summary["tokens"])
            for summary in split_summaries.values()
        )
        / total_tokens
    )
    eval_entropy = (
        sum(
            float(summary["eval_entropy"]) * int(summary["tokens"])
            for summary in split_summaries.values()
        )
        / total_tokens
    )
    eval_mean_token_accuracy = (
        sum(
            float(summary["eval_mean_token_accuracy"]) * int(summary["tokens"])
            for summary in split_summaries.values()
        )
        / total_tokens
    )
    return {
        "examples": total_examples,
        "tokens": total_tokens,
        "eval_loss": eval_loss,
        "perplexity": finite_perplexity(eval_loss),
        "eval_entropy": eval_entropy,
        "eval_mean_token_accuracy": eval_mean_token_accuracy,
        "splits": split_summaries,
    }


QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\s]+)>\s*"
    r"(?P<body>.*?)</function>\s*</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)
QWEN_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
    flags=re.IGNORECASE | re.DOTALL,
)
CUSTOM_TOOL_CALL_RE = re.compile(
    r"<tool>\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<arguments>.*?)\)\s*</tool>",
    flags=re.IGNORECASE | re.DOTALL,
)
QUOTED_ARGUMENT_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<literal>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")",
    flags=re.DOTALL,
)
CODE_BLOCK_RE = re.compile(
    r"```(?:json|python)?\s*(?P<body>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
THINK_END_RE = re.compile(r"</think>", flags=re.IGNORECASE)
LEADING_ASSISTANT_TURN_RE = re.compile(
    r"^\s*<\|im_start\|>assistant\s*",
    flags=re.IGNORECASE,
)
CHAT_TERMINATOR_RE = re.compile(
    r"<\|im_end\|>|<\|endoftext\|>",
    flags=re.IGNORECASE,
)
GENERATION_ARTIFACT_RE = re.compile(
    r"</?think>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>",
    flags=re.IGNORECASE,
)


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def clean_generated_text(text: str) -> str:
    cleaned = LEADING_ASSISTANT_TURN_RE.sub("", text).strip()
    think_matches = list(THINK_END_RE.finditer(cleaned))
    if think_matches:
        cleaned = cleaned[think_matches[-1].end() :].lstrip()
    terminator = CHAT_TERMINATOR_RE.search(cleaned)
    if terminator:
        cleaned = cleaned[: terminator.start()]
    return cleaned.strip()


def generation_artifacts_present(text: str) -> bool:
    return bool(GENERATION_ARTIFACT_RE.search(text))


def record_expected_solution(record: dict[str, Any]) -> Any:
    for key in ("expected_solution", "real_solution", "solution"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def record_gold_cypher(record: dict[str, Any]) -> str:
    return str(record.get("gold_cypher") or record.get("cypher") or "").strip()


def record_output_type(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(record.get("output_type") or metadata.get("output_type") or "SLDP")


def record_task_family(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(record.get("task_family") or metadata.get("task_family") or "cypher_qa")


def record_query_type(record: dict[str, Any]) -> str:
    return str(record.get("query_type") or record.get("task_type") or "")


def record_for_solution_conversion(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("query_type"):
        return record
    return {**record, "query_type": record_query_type(record)}


def record_is_refusal(record: dict[str, Any]) -> bool:
    from cypher_3dsg_agent_distillation.agents.agent_records import (
        WRITE_REFUSAL_ANSWER,
        normalize_cypher,
    )

    gold_cypher = normalize_cypher(record_gold_cypher(record))
    expected_solution = record_expected_solution(record)
    return gold_cypher in {
        WRITE_REFUSAL_ANSWER,
        f"// {WRITE_REFUSAL_ANSWER}",
    } or str(expected_solution or "").strip() == WRITE_REFUSAL_ANSWER


def initial_prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    from cypher_3dsg_agent_distillation.training.train import normalize_chat_messages

    messages = normalize_chat_messages(record.get("messages", []))
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            return messages[:index]
    return messages


def final_answer_prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    from cypher_3dsg_agent_distillation.training.train import normalize_chat_messages

    messages = normalize_chat_messages(record.get("messages", []))
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            message.get("role") == "assistant"
            and message.get("content")
            and not message.get("tool_calls")
        ):
            return messages[:index]
    return messages


def render_messages_for_generation(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool | None = False,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    from cypher_3dsg_agent_distillation.training.train import (
        message_to_plain_text,
        normalize_chat_messages,
    )

    messages = normalize_chat_messages(messages)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            if "enable_thinking" not in kwargs:
                raise
            kwargs.pop("enable_thinking")
            return tokenizer.apply_chat_template(messages, **kwargs)

    prompt = "\n\n".join(message_to_plain_text(message) for message in messages)
    return f"{prompt}\n\nassistant:" if prompt else "assistant:"


def generation_length_kwargs(max_new_tokens: int | None) -> dict[str, Any]:
    if max_new_tokens is None:
        return {}
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero when provided.")
    return {"max_new_tokens": max_new_tokens}


def generation_config_report(model, *, max_new_tokens: int | None) -> dict[str, Any]:
    config = getattr(model, "generation_config", None)
    report = {}
    if config is not None:
        for key in GENERATION_CONFIG_REPORT_KEYS:
            if hasattr(config, key):
                report[key] = jsonable(getattr(config, key))
    overrides = generation_length_kwargs(max_new_tokens)
    if overrides:
        report["generation_call_kwargs"] = overrides
    return report


def configured_generation_token_budget(model, max_new_tokens: int | None) -> int | None:
    if max_new_tokens is not None:
        return max_new_tokens
    config = getattr(model, "generation_config", None)
    configured_max_new_tokens = getattr(config, "max_new_tokens", None)
    return int(configured_max_new_tokens) if configured_max_new_tokens is not None else None


def generation_progress_stopping_criteria(
    *,
    input_length: int,
    progress,
    task_id,
):
    if progress is None or task_id is None:
        return None

    from transformers import StoppingCriteria, StoppingCriteriaList

    class ProgressStoppingCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.completed = 0

        def __call__(self, input_ids, _scores, **_kwargs) -> bool:
            completed = max(0, int(input_ids.shape[-1]) - input_length)
            if completed != self.completed:
                progress.update(task_id, completed=completed)
                self.completed = completed
            return False

    return StoppingCriteriaList([ProgressStoppingCriteria()])


def generation_report_fields(result: GenerationResult) -> dict[str, int | None]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "max_new_tokens": result.max_new_tokens,
        "token_budget": result.token_budget,
    }


def add_raw_generation_debug_fields(
    score: dict[str, Any],
    generation: GenerationResult,
) -> None:
    score["raw_generated_text"] = generation.raw_text
    score["raw_generated_text_had_artifacts"] = generation_artifacts_present(
        generation.raw_text
    )


def generation_debug_report(
    generation: GenerationResult | None,
) -> dict[str, Any]:
    return {
        "rendered_prompt": generation.prompt if generation is not None else None,
        "raw_completion": generation.raw_text if generation is not None else None,
    }


def expected_final_answer(record: dict[str, Any]) -> str | None:
    solution = record_expected_solution(record)
    return f"<answer>{solution}</answer>" if solution is not None else None


def generate_assistant_response(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int | None,
    enable_thinking: bool | None = False,
    tools: list[dict[str, Any]] | None = None,
    progress=None,
    progress_task_id=None,
    progress_description: str | None = None,
) -> GenerationResult:
    (torch, *_unused) = require_eval_loss_dependencies()
    prompt = render_messages_for_generation(
        tokenizer,
        messages,
        enable_thinking=enable_thinking,
        tools=tools,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_device = model_input_device(model)
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    token_budget = configured_generation_token_budget(model, max_new_tokens)
    if progress is not None and progress_task_id is not None:
        progress_kwargs = {
            "completed": 0,
            "visible": True,
            "description": progress_description or "generation",
        }
        if token_budget is not None:
            progress_kwargs["total"] = token_budget
        progress.update(progress_task_id, **progress_kwargs)

    generate_kwargs = generation_length_kwargs(max_new_tokens)
    stopping_criteria = generation_progress_stopping_criteria(
        input_length=input_length,
        progress=progress,
        task_id=progress_task_id,
    )
    if stopping_criteria is not None:
        generate_kwargs["stopping_criteria"] = stopping_criteria

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            **generate_kwargs,
        )

    generated_ids = generated[0][input_length:]
    generated_tokens = int(generated_ids.shape[-1])
    if progress is not None and progress_task_id is not None:
        progress.update(progress_task_id, completed=generated_tokens)
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    result = GenerationResult(
        text=clean_generated_text(raw_text),
        raw_text=raw_text,
        prompt=prompt,
        prompt_tokens=int(input_length),
        generated_tokens=generated_tokens,
        max_new_tokens=max_new_tokens,
        token_budget=token_budget,
    )
    # Do not keep GPU input/output tensors alive while the caller scores or stores
    # string-only generation results. This matters most for multi-turn retries.
    del generated_ids, generated, encoded
    return result


def generate_assistant_text(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int | None,
    enable_thinking: bool | None = False,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    return generate_assistant_response(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        tools=tools,
    ).text


def normalized_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def qwen_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in QWEN_TOOL_CALL_RE.finditer(text):
        arguments = {
            parameter.group("name").strip(): parameter.group("value").strip()
            for parameter in QWEN_PARAMETER_RE.finditer(match.group("body"))
        }
        calls.append(normalized_tool_call(match.group("name").strip(), arguments))
    return calls


def normalize_structured_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
        normalize_tool_call_arguments,
    )

    if "function" not in tool_call and "name" in tool_call:
        tool_call = {
            **{key: value for key, value in tool_call.items() if key not in {"name", "arguments"}},
            "type": "function",
            "function": {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments", {}),
            },
        }
    return normalize_tool_call_arguments(tool_call)


def structured_tool_calls_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "tool_calls" in payload:
            raw_calls = payload.get("tool_calls") or []
        elif "function" in payload or "name" in payload:
            raw_calls = [payload]
        else:
            return []
    elif isinstance(payload, list):
        raw_calls = payload
    else:
        return []

    return [
        normalize_structured_tool_call(tool_call)
        for tool_call in raw_calls
        if isinstance(tool_call, dict)
    ]


def parse_json_like_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def json_candidate_texts(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates.extend(match.group("body").strip() for match in CODE_BLOCK_RE.finditer(text))
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        candidates.append(text[first_bracket : last_bracket + 1])
    return [candidate for candidate in candidates if candidate]


def json_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    errors = []
    for candidate in json_candidate_texts(text):
        try:
            tool_calls = structured_tool_calls_from_payload(parse_json_like_payload(candidate))
        except (SyntaxError, ValueError, TypeError) as exc:
            errors.append(str(exc))
            continue
        if tool_calls:
            return tool_calls
    if "tool_calls" in text:
        raise ValueError("; ".join(errors) if errors else "malformed JSON tool call")
    return []


def custom_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in CUSTOM_TOOL_CALL_RE.finditer(text):
        arguments = {}
        for argument in QUOTED_ARGUMENT_RE.finditer(match.group("arguments")):
            try:
                value = ast.literal_eval(argument.group("literal"))
            except (SyntaxError, ValueError):
                value = argument.group("literal").strip("\"'")
            arguments[argument.group("name")] = value
        calls.append(normalized_tool_call(match.group("name").strip(), arguments))
    return calls


def generated_tool_marker_present(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in ("<tool_call", "<tool>", "tool_calls", "run_cypher_query")
    )


def parse_generated_tool_calls(text: str) -> ToolCallParseResult:
    try:
        tool_calls = qwen_tool_calls_from_text(text)
        if tool_calls:
            return ToolCallParseResult(True, tool_calls)

        tool_calls = json_tool_calls_from_text(text)
        if tool_calls:
            return ToolCallParseResult(True, tool_calls)

        tool_calls = custom_tool_calls_from_text(text)
        if tool_calls:
            return ToolCallParseResult(True, tool_calls)
    except ValueError as exc:
        return ToolCallParseResult(True, [], str(exc))

    if generated_tool_marker_present(text):
        return ToolCallParseResult(True, [], "tool-call markers were present but unparseable")
    return ToolCallParseResult(False, [])


def score_final_answer_generation(
    record: dict[str, Any],
    generated_text: str,
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import answer_inner_text
    from cypher_3dsg_agent_distillation.graph.heracles_validation import (
        answer_matches_with_heracles,
    )

    expected_solution = record_expected_solution(record)
    answer = answer_inner_text(generated_text)
    format_valid = False
    answer_match = False
    if answer is not None and expected_solution is not None:
        format_valid, answer_match = answer_matches_with_heracles(
            answer,
            str(expected_solution),
            record_output_type(record),
            answer_validation=record.get("answer_validation"),
        )

    status = "pass"
    if expected_solution is None:
        status = "missing_expected_solution"
    elif not generated_text.strip():
        status = "missing_generation"
    elif answer is None:
        status = "missing_answer_tags"
    elif not format_valid:
        status = "invalid_answer_format"
    elif not answer_match:
        status = "final_answer_mismatch"

    return {
        "status": status,
        "generated_text": generated_text,
        "generated_answer": answer,
        "expected_solution": expected_solution,
        "final_answer_generated": bool(generated_text.strip()),
        "final_answer_has_tags": answer is not None,
        "final_answer_format_valid": format_valid,
        "final_answer_domain_valid": format_valid,
        "final_answer_match": answer_match,
    }


def final_answer_generation_error(record: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "status": "generation_error",
        "generation_error": f"{type(exc).__name__}: {exc}",
        "generated_text": "",
        "generated_answer": None,
        "expected_solution": record_expected_solution(record),
        "final_answer_generated": False,
        "final_answer_has_tags": False,
        "final_answer_format_valid": False,
        "final_answer_match": False,
    }


def release_generation_memory() -> None:
    """Release per-record objects and unused CUDA allocator blocks."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def bounded_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    marker = "...[truncated]"
    return f"{text[: max(0, max_chars - len(marker))]}{marker}"


def end_to_end_error_content(
    error_type: str,
    issues: list[Any],
    *,
    max_chars: int,
) -> str:
    """Build small retry feedback without copying arbitrary database errors."""
    payload = {
        "ok": False,
        "error_type": error_type,
        "issues": [bounded_text(issue, 512) for issue in issues[:3]],
        "message": E2E_REPAIR_MESSAGE,
    }
    content = json.dumps(payload, ensure_ascii=False, default=str)
    if len(content) <= max_chars:
        return content
    # max_chars is validated to be >= 256, so this compact payload remains valid
    # JSON instead of slicing JSON at an arbitrary byte/character boundary.
    return json.dumps(
        {
            "ok": False,
            "error_type": bounded_text(error_type, 64),
            "message": "Return one corrected run_cypher_query tool call.",
        },
        ensure_ascii=False,
    )


def bounded_tool_success_content(
    rows: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str | None:
    """Serialize tool rows incrementally and stop before an oversized copy is built."""
    encoder = json.JSONEncoder(ensure_ascii=False, default=str)
    chunks = []
    length = 0
    for chunk in encoder.iterencode({"ok": True, "rows": rows}):
        length += len(chunk)
        if length > max_chars:
            return None
        chunks.append(chunk)
    return "".join(chunks)


def end_to_end_skipped_result(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": reason,
        "final_answer_match": False,
        "generated_text": None,
        "generated_answer": None,
        "query_attempts": 0,
        "retries_used": 0,
        "attempts": [],
    }


def end_to_end_failure_result(
    record: dict[str, Any],
    *,
    status: str,
    query_attempts: int,
    max_query_retries: int,
    attempts: list[dict[str, Any]],
    last_failure_reason: str | None = None,
    generation_error: str | None = None,
) -> dict[str, Any]:
    result = {
        "status": status,
        "final_answer_match": False,
        "generated_text": None,
        "generated_answer": None,
        "expected_solution": record_expected_solution(record),
        "query_attempts": query_attempts,
        "retries_used": max(0, query_attempts - 1),
        "max_query_retries": max_query_retries,
        "attempts": attempts,
    }
    if last_failure_reason is not None:
        result["last_failure_reason"] = last_failure_reason
    if generation_error is not None:
        result["generation_error"] = generation_error
    return result


def append_end_to_end_retry_feedback(
    messages: list[dict[str, Any]],
    *,
    generated_text: str,
    tool_call: dict[str, Any] | None,
    feedback: str,
    max_query_chars: int,
) -> None:
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
        build_tool_result_message,
    )

    if tool_call is not None:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call],
            }
        )
        tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        messages.append(build_tool_result_message(tool_call_id, feedback))
        return

    # An unparseable or wrongly named call cannot be followed by a valid tool-role
    # message. Keep only a bounded preview and provide corrective user feedback.
    messages.append(
        {
            "role": "assistant",
            "content": bounded_text(generated_text, max_query_chars),
        }
    )
    messages.append({"role": "user", "content": feedback})


def evaluate_end_to_end_record(
    record: dict[str, Any],
    *,
    model,
    tokenizer,
    validator,
    policy: dict[str, Any] | None,
    tool_schemas: list[dict[str, Any]],
    max_new_tokens: int | None,
    enable_thinking: bool,
    max_query_retries: int = DEFAULT_MAX_QUERY_RETRIES,
    max_tool_rows: int = DEFAULT_E2E_MAX_TOOL_ROWS,
    max_tool_response_chars: int = DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS,
    max_query_chars: int = DEFAULT_E2E_MAX_QUERY_CHARS,
    progress=None,
    progress_task_id=None,
    progress_description: str | None = None,
    generation_debug_turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one bounded query/tool/final-answer interaction.

    Query failures receive compact feedback and may be retried. The only quality
    metric produced by this interaction is ``final_answer_match``; query fields are
    operational diagnostics for understanding retries and resource guards.
    """
    from cypher_3dsg_agent_distillation.agents.agent_records import DEFAULT_AGENT_TOOL
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import static_checks

    if max_query_retries < 0:
        raise ValueError("max_query_retries must be zero or greater.")
    if max_tool_rows <= 0:
        raise ValueError("max_tool_rows must be greater than zero.")
    if max_tool_response_chars < 256:
        raise ValueError("max_tool_response_chars must be at least 256.")
    if max_query_chars < 256:
        raise ValueError("max_query_chars must be at least 256.")

    base_messages = initial_prompt_messages(record)
    messages = list(base_messages)
    attempts: list[dict[str, Any]] = []
    query_attempts = 0
    max_query_attempts = max_query_retries + 1

    while True:
        try:
            generation = generate_assistant_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                tools=tool_schemas,
                progress=progress,
                progress_task_id=progress_task_id,
                progress_description=progress_description,
            )
        except Exception as exc:
            return end_to_end_failure_result(
                record,
                status="generation_error",
                query_attempts=query_attempts,
                max_query_retries=max_query_retries,
                attempts=attempts,
                generation_error=bounded_text(f"{type(exc).__name__}: {exc}", 1024),
            )

        generated_text = generation.text
        parsed = parse_generated_tool_calls(generated_text)
        if generation_debug_turns is not None:
            generation_debug_turns.append(
                {
                    "turn": len(generation_debug_turns) + 1,
                    **generation_debug_report(generation),
                }
            )
        attempt = {
            "turn": len(attempts) + 1,
            "completion_preview": bounded_text(generated_text, 1024),
            **generation_report_fields(generation),
        }
        attempts.append(attempt)

        if not parsed.tool_calls and not parsed.tool_call_generated:
            score = score_final_answer_generation(record, generated_text)
            attempt["status"] = score["status"]
            return {
                **score,
                "query_attempts": query_attempts,
                "retries_used": max(0, query_attempts - 1),
                "max_query_retries": max_query_retries,
                "attempts": attempts,
            }

        if query_attempts >= max_query_attempts:
            attempt["status"] = "query_retry_limit_reached"
            return end_to_end_failure_result(
                record,
                status="query_retries_exhausted",
                query_attempts=query_attempts,
                max_query_retries=max_query_retries,
                attempts=attempts,
                last_failure_reason="query_retry_limit_reached",
            )

        query_attempts += 1
        retry_tool_call = None
        failure_reason = None
        failure_issues: list[Any] = []
        validation = None
        tool_content = None

        if not parsed.tool_calls:
            failure_reason = "tool_call_parse_failed"
            failure_issues = [parsed.error or "The tool call could not be parsed."]
        elif len(parsed.tool_calls) != 1:
            failure_reason = "multiple_tool_calls"
            failure_issues = ["Return exactly one run_cypher_query tool call."]
        else:
            tool_call = parsed.tool_calls[0]
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or {}
            tool_name = function.get("name")
            cypher = str(arguments.get("cypher_string") or "").strip()
            attempt["tool_name"] = tool_name
            attempt["generated_cypher"] = bounded_text(cypher, max_query_chars)

            if tool_name != DEFAULT_AGENT_TOOL:
                failure_reason = "wrong_tool_name"
                failure_issues = [f"Expected tool {DEFAULT_AGENT_TOOL}."]
            else:
                retry_tool_call = tool_call
                if not cypher:
                    failure_reason = "missing_cypher_argument"
                    failure_issues = ["cypher_string must be a non-empty query."]
                elif len(cypher) > max_query_chars:
                    retry_tool_call = None
                    failure_reason = "query_too_large"
                    failure_issues = [
                        f"Query exceeded the {max_query_chars}-character safety limit."
                    ]
                else:
                    static_issues = static_checks(
                        cypher,
                        allow_writes=not bool(record.get("read_only", True)),
                        policy=policy,
                    )
                    if static_issues:
                        failure_reason = "static_error"
                        failure_issues = static_issues
                    else:
                        try:
                            validation = validator.validate(
                                cypher,
                                execute=True,
                                max_rows=max_tool_rows,
                                compute_result_hash=False,
                            )
                        except Exception as exc:
                            failure_reason = "execution_error"
                            failure_issues = [f"{type(exc).__name__}: {exc}"]
                        else:
                            attempt["executable"] = bool(validation.get("executable"))
                            attempt["elapsed_ms"] = validation.get("elapsed_ms")
                            rows = validation.get("rows") or []
                            attempt["row_count"] = len(rows)
                            attempt["rows_truncated"] = bool(
                                validation.get("rows_truncated")
                            )
                            if not validation.get("executable"):
                                failure_reason = "execution_error"
                                failure_issues = [
                                    validation.get("error") or "Query execution failed."
                                ]
                            elif validation.get("rows_truncated"):
                                failure_reason = "result_too_large"
                                failure_issues = [
                                    f"Result exceeded the {max_tool_rows}-row safety limit."
                                ]
                            else:
                                candidate_content = bounded_tool_success_content(
                                    rows,
                                    max_chars=max_tool_response_chars,
                                )
                                if candidate_content is None:
                                    failure_reason = "result_too_large"
                                    failure_issues = [
                                        "Serialized result exceeded the "
                                        f"{max_tool_response_chars}-character safety limit."
                                    ]
                                else:
                                    attempt["tool_response_chars"] = len(
                                        candidate_content
                                    )
                                    tool_content = candidate_content

        if tool_content is not None and retry_tool_call is not None:
            attempt["status"] = "tool_success"
            # Keep the initial prompt plus only the most recent tool exchange. Old
            # failed/incorrect outputs must not grow the next GPU input each retry.
            messages = list(base_messages)
            append_end_to_end_retry_feedback(
                messages,
                generated_text=generated_text,
                tool_call=retry_tool_call,
                feedback=tool_content,
                max_query_chars=max_query_chars,
            )
            del generation, validation, tool_content
            continue

        failure_reason = failure_reason or "query_failed"
        attempt["status"] = failure_reason
        feedback = end_to_end_error_content(
            failure_reason,
            failure_issues,
            max_chars=max_tool_response_chars,
        )
        attempt["tool_response_chars"] = len(feedback)
        if query_attempts >= max_query_attempts:
            return end_to_end_failure_result(
                record,
                status="query_retries_exhausted",
                query_attempts=query_attempts,
                max_query_retries=max_query_retries,
                attempts=attempts,
                last_failure_reason=failure_reason,
            )
        messages = list(base_messages)
        append_end_to_end_retry_feedback(
            messages,
            generated_text=generated_text,
            tool_call=retry_tool_call,
            feedback=feedback,
            max_query_chars=max_query_chars,
        )
        del generation, validation, feedback


def skipped_tool_call_result(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": reason,
        "generated_text": None,
        "tool_call_generated": False,
        "tool_call_parse_success": False,
        "correct_tool_name": False,
        "cypher_argument_present": False,
        "cypher_exact_match": False,
        "static_valid": False,
        "executable": False,
        "cypher_solution_match": False,
        "tool_result_match": False,
        "result_match_without_exact_cypher_match": False,
        "potential_hidden_false_positive": False,
        "hidden_false_positive_reasons": [],
    }


def tool_call_generation_error(record: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "status": "generation_error",
        "generation_error": f"{type(exc).__name__}: {exc}",
        "generated_text": "",
        "tool_call_generated": False,
        "tool_call_parse_success": False,
        "correct_tool_name": False,
        "cypher_argument_present": False,
        "cypher_exact_match": False,
        "static_valid": False,
        "executable": False,
        "gold_cypher": record_gold_cypher(record),
        "expected_solution": record_expected_solution(record),
        "cypher_solution_match": False,
        "tool_result_match": False,
        "result_match_without_exact_cypher_match": False,
        "potential_hidden_false_positive": False,
        "hidden_false_positive_reasons": [],
    }


def score_refusal_tool_generation(
    record: dict[str, Any],
    generated_text: str,
    parsed: ToolCallParseResult,
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.agent_records import WRITE_REFUSAL_ANSWER
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import answer_inner_text

    answer = answer_inner_text(generated_text)
    refusal_match = str(answer or "").strip() == WRITE_REFUSAL_ANSWER
    status = "pass"
    if parsed.tool_call_generated:
        status = "unexpected_tool_call_for_refusal"
    elif not refusal_match:
        status = "missing_refusal"

    tool_calls = parsed.tool_calls
    function = (tool_calls[0].get("function") or {}) if tool_calls else {}
    arguments = function.get("arguments") or {}
    generated_cypher = str(arguments.get("cypher_string") or "").strip()
    return {
        "status": status,
        "generated_text": generated_text,
        "generated_answer": answer,
        "generated_cypher": generated_cypher or None,
        "tool_calls": tool_calls,
        "expected_solution": record_expected_solution(record),
        "tool_call_generated": parsed.tool_call_generated,
        "tool_call_parse_success": bool(parsed.tool_calls),
        "correct_tool_name": not parsed.tool_call_generated,
        "cypher_argument_present": False,
        "cypher_exact_match": False,
        "static_valid": refusal_match and not parsed.tool_call_generated,
        "executable": refusal_match and not parsed.tool_call_generated,
        "cypher_solution": WRITE_REFUSAL_ANSWER if refusal_match else None,
        "cypher_solution_match": refusal_match,
        "tool_result_match": refusal_match,
        "result_match_without_exact_cypher_match": False,
        "potential_hidden_false_positive": False,
        "hidden_false_positive_reasons": [],
    }


def score_tool_call_generation(
    record: dict[str, Any],
    generated_text: str,
    *,
    validator,
    policy: dict[str, Any] | None,
    max_rows: int | None = None,
    max_result_chars: int | None = None,
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.agent_records import (
        DEFAULT_AGENT_TOOL,
        normalize_cypher,
    )
    from cypher_3dsg_agent_distillation.cypher.cypher_match_diagnostics import (
        result_match_diagnostics,
    )
    from cypher_3dsg_agent_distillation.cypher.evaluation import (
        ResultShapeError,
        rows_to_solution_text,
    )
    from cypher_3dsg_agent_distillation.cypher.pddl_goals import validate_grounding_rows
    from cypher_3dsg_agent_distillation.graph.heracles_validation import (
        answer_matches_with_heracles,
    )
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import result_hash, static_checks

    parsed = parse_generated_tool_calls(generated_text)
    base = {
        "generated_text": generated_text,
        "tool_call_generated": parsed.tool_call_generated,
        "tool_call_parse_success": bool(parsed.tool_calls),
        "parse_error": parsed.error,
        "correct_tool_name": False,
        "cypher_argument_present": False,
        "cypher_exact_match": False,
        "static_valid": False,
        "static_issues": [],
        "executable": False,
        "syntax_error": False,
        "timeout": False,
        "elapsed_ms": None,
        "row_count": 0,
        "result_hash": None,
        "execution_error": None,
        "result_shape_error": None,
        "generated_cypher": None,
        "gold_cypher": record_gold_cypher(record),
        "cypher_solution": None,
        "expected_solution": record_expected_solution(record),
        "cypher_solution_format_valid": False,
        "cypher_solution_match": False,
        "tool_result_match": False,
        "result_match_without_exact_cypher_match": False,
        "potential_hidden_false_positive": False,
        "hidden_false_positive_reasons": [],
    }

    if record_is_refusal(record):
        return score_refusal_tool_generation(record, generated_text, parsed)
    if not parsed.tool_calls:
        status = "tool_call_parse_failed" if parsed.tool_call_generated else "missing_required_tool"
        return {**base, "status": status}
    if len(parsed.tool_calls) > 1:
        return {**base, "status": "multiple_tool_calls", "tool_calls": parsed.tool_calls}

    tool_call = parsed.tool_calls[0]
    function = tool_call.get("function") or {}
    arguments = function.get("arguments") or {}
    tool_name = function.get("name")
    cypher = str(arguments.get("cypher_string") or "").strip()
    correct_tool_name = tool_name == DEFAULT_AGENT_TOOL
    cypher_argument_present = bool(cypher)
    base.update(
        {
            "tool_calls": [tool_call],
            "tool_name": tool_name,
            "correct_tool_name": correct_tool_name,
            "cypher_argument_present": cypher_argument_present,
            "generated_cypher": cypher or None,
            "cypher_exact_match": normalize_cypher(cypher) == normalize_cypher(
                record_gold_cypher(record)
            ),
        }
    )
    if not correct_tool_name:
        return {**base, "status": "wrong_tool_name"}
    if not cypher_argument_present:
        return {**base, "status": "missing_cypher_argument"}

    static_issues = static_checks(
        cypher,
        allow_writes=not bool(record.get("read_only", True)),
        policy=policy,
    )
    base.update({"static_valid": not static_issues, "static_issues": static_issues})
    if static_issues:
        return {**base, "status": "static_error"}

    try:
        validation = (
            validator.validate(
                cypher,
                execute=True,
                max_rows=max_rows,
                compute_result_hash=False,
            )
            if max_rows is not None
            else validator.validate(cypher, execute=True)
        )
    except Exception as exc:
        return {
            **base,
            "status": "execution_error",
            "execution_error": f"{type(exc).__name__}: {exc}",
        }
    rows = validation.get("rows", [])
    result_too_large = bool(validation.get("rows_truncated"))
    if (
        not result_too_large
        and max_result_chars is not None
        and bounded_tool_success_content(rows, max_chars=max_result_chars) is None
    ):
        result_too_large = True
    report_validation = (
        {**validation, "rows": [], "rows_omitted": True}
        if result_too_large
        else validation
    )
    base.update(
        {
            "validation": jsonable(report_validation),
            "executable": bool(validation.get("executable")),
            "syntax_error": bool(validation.get("syntax_error")),
            "timeout": bool(validation.get("timeout")),
            "elapsed_ms": validation.get("elapsed_ms"),
            "row_count": len(rows),
            "result_hash": result_hash(rows) if validation.get("executable") else None,
            "execution_error": validation.get("error"),
        }
    )
    if not validation.get("executable"):
        return {**base, "status": "execution_error"}
    if result_too_large:
        return {**base, "status": "result_too_large"}

    expected_solution = record_expected_solution(record)
    is_pddl = record_task_family(record) == "cypher_pddl"
    if is_pddl:
        cypher_solution_match, cypher_solution, actual_grounding = validate_grounding_rows(
            rows, record.get("grounding")
        )
        format_valid = True
        base["expected_grounding"] = record.get("grounding")
        base["actual_grounding"] = actual_grounding
        base["grounding_match"] = cypher_solution_match
    else:
        try:
            cypher_solution = rows_to_solution_text(record_for_solution_conversion(record), rows)
        except ResultShapeError as exc:
            return {**base, "status": "result_shape_error", "result_shape_error": str(exc)}
        format_valid = False
        cypher_solution_match = False
        if expected_solution is not None:
            format_valid, cypher_solution_match = answer_matches_with_heracles(
                str(cypher_solution),
                str(expected_solution),
                record_output_type(record),
                answer_validation=record.get("answer_validation"),
            )
    status = "pass"
    if expected_solution is None:
        status = "missing_expected_solution"
    elif not format_valid:
        status = "invalid_cypher_solution_format"
    elif not cypher_solution_match:
        status = (
            "grounding_mismatch"
            if record_task_family(record) == "cypher_pddl"
            else "cypher_solution_mismatch"
        )

    match_diagnostics = result_match_diagnostics(
        generated_cypher=cypher,
        gold_cypher=record_gold_cypher(record),
        cypher_exact_match=bool(base["cypher_exact_match"]),
        solution_match=bool(cypher_solution_match),
        rows=rows,
        expected_solution=expected_solution,
    )

    return {
        **base,
        **match_diagnostics,
        "status": status,
        "cypher_solution": cypher_solution,
        "cypher_solution_format_valid": format_valid,
        "cypher_solution_match": cypher_solution_match,
        "tool_result_match": cypher_solution_match,
    }


def summarize_raw_section(
    records: list[dict[str, Any]],
    section: str,
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    total = len(records)
    section_records = [
        record.get(section) or {"status": "skipped"} for record in records
    ]
    status_counts = Counter(str(record.get("status") or "unknown") for record in section_records)
    skipped = status_counts.get("skipped", 0)
    evaluated = total - skipped
    metrics = {
        key: sum(1 for record in section_records if bool(record.get(key)))
        for key in metric_keys
    }
    denominator = evaluated if evaluated else total
    return {
        "total": total,
        "evaluated": evaluated,
        "skipped": skipped,
        "metrics": metrics,
        "rates": {
            key: (value / denominator if denominator else 0.0)
            for key, value in metrics.items()
        },
        "statuses": dict(status_counts),
    }


def summarize_raw_generation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "examples": len(records),
        FINAL_ANSWER_CHECK: summarize_raw_section(
            records,
            FINAL_ANSWER_CHECK,
            (
                "final_answer_generated",
                "final_answer_has_tags",
                "final_answer_format_valid",
                "final_answer_match",
            ),
        ),
        TOOL_CALL_CHECK: summarize_raw_section(
            records,
            TOOL_CALL_CHECK,
            (
                "tool_call_generated",
                "tool_call_parse_success",
                "correct_tool_name",
                "cypher_argument_present",
                "cypher_exact_match",
                "static_valid",
                "executable",
                "cypher_solution_match",
                "tool_result_match",
                "result_match_without_exact_cypher_match",
                "potential_hidden_false_positive",
            ),
        ),
        END_TO_END_CHECK: summarize_raw_section(
            records,
            END_TO_END_CHECK,
            ("final_answer_match",),
        ),
    }
    by_family = {}
    for family in sorted({record.get("task_family", "cypher_qa") for record in records}):
        family_records = [
            record for record in records if record.get("task_family", "cypher_qa") == family
        ]
        family_tool_metrics = [
            "static_valid",
            "executable",
            "tool_result_match",
            "potential_hidden_false_positive",
        ]
        if family == "cypher_pddl":
            family_tool_metrics.append("grounding_match")
        else:
            family_tool_metrics.append("cypher_solution_match")
        by_family[family] = {
            "examples": len(family_records),
            FINAL_ANSWER_CHECK: summarize_raw_section(
                family_records,
                FINAL_ANSWER_CHECK,
                (
                    "final_answer_format_valid",
                    "final_answer_domain_valid",
                    "final_answer_match",
                ),
            ),
            TOOL_CALL_CHECK: summarize_raw_section(
                family_records,
                TOOL_CALL_CHECK,
                tuple(family_tool_metrics),
            ),
            END_TO_END_CHECK: summarize_raw_section(
                family_records,
                END_TO_END_CHECK,
                ("final_answer_match",),
            ),
        }
    summary["by_task_family"] = by_family
    return summary


def raw_generation_progress_context(show_progress: bool):
    if not show_progress:
        return nullcontext(None)

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def eval_loss_progress_context(show_progress: bool):
    return raw_generation_progress_context(show_progress)


def split_display_name(split: str) -> str:
    return split.replace("_", " ").replace("-", " ").title()


def raw_generation_progress_description(
    *,
    split: str,
    phase: str,
) -> str:
    phase_labels = {
        "final-answer": "Validating final answer",
        "final-answer tokens": "Generating final answer",
        "tool-call": "Validating tool call",
        "tool-call tokens": "Generating tool call",
        "end-to-end": "Validating end to end",
        "end-to-end tokens": "Generating end-to-end response",
        "done": "Validating records",
    }
    return f"{phase_labels[phase]} · {split_display_name(split)}"


def raw_generation_report_record(
    record: dict[str, Any],
    *,
    final_answer_result: dict[str, Any],
    tool_call_result: dict[str, Any],
    final_generation_debug: dict[str, Any],
    tool_generation_debug: dict[str, Any],
    end_to_end_result: dict[str, Any] | None = None,
    end_to_end_generation_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one raw-generation report record with task-aware result fields."""
    family = record_task_family(record)
    expected_solution = record_expected_solution(record)
    expected = {
        "cypher": record_gold_cypher(record),
        "cypher_solution": expected_solution,
        "final_answer": expected_final_answer(record),
    }
    generated = {
        "cypher": tool_call_result.get("generated_cypher"),
        "cypher_solution": tool_call_result.get("cypher_solution"),
        "final_answer": final_answer_result.get("generated_text"),
        "final_answer_value": final_answer_result.get("generated_answer"),
        "cypher_tool_output": (tool_call_result.get("validation") or {}).get("rows"),
        "end_to_end_final_answer": (end_to_end_result or {}).get("generated_text"),
    }
    if family == "cypher_pddl":
        empty_answer = expected_solution is not None and not str(expected_solution).strip()
        expected.update(
            {
                "pddl_goal": None if empty_answer else expected_solution,
                "empty_answer": empty_answer,
                "grounding": record.get("grounding"),
            }
        )
        generated["actual_grounding"] = tool_call_result.get("actual_grounding")

    return {
        "uid": record.get("uid"),
        "query_type": record_query_type(record),
        "task_family": family,
        "output_type": record_output_type(record),
        "question": record.get("question"),
        "expected": expected,
        "generated": generated,
        "generation_debug": {
            "cypher_generation": tool_generation_debug,
            "final_answer_generation": final_generation_debug,
            "end_to_end_generation": end_to_end_generation_debug or {"turns": []},
        },
        FINAL_ANSWER_CHECK: final_answer_result,
        TOOL_CALL_CHECK: tool_call_result,
        END_TO_END_CHECK: end_to_end_result or end_to_end_skipped_result(
            "end-to-end validation was not run"
        ),
    }


def evaluate_raw_generation(
    *,
    data_config: dict[str, Any],
    student_config: dict[str, Any],
    split: str,
    adapter_dir: str,
    explicit_tokenizer: str | None,
    max_new_tokens: int | None,
    limit: int | None,
    validator,
    tool_call_skip_reason: str | None,
    policy: dict[str, Any] | None,
    show_progress: bool,
    enable_thinking: bool,
    max_query_retries: int = DEFAULT_MAX_QUERY_RETRIES,
    max_tool_rows: int = DEFAULT_E2E_MAX_TOOL_ROWS,
    max_tool_response_chars: int = DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS,
    max_query_chars: int = DEFAULT_E2E_MAX_QUERY_CHARS,
    baseline: bool = False,
) -> dict[str, Any]:
    from cypher_3dsg_agent_distillation.agents.agent_prompts import run_cypher_tool_schema
    from cypher_3dsg_agent_distillation.core.io import read_jsonl

    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero when provided.")
    if max_query_retries < 0:
        raise ValueError("max_query_retries must be zero or greater.")
    if max_tool_rows <= 0:
        raise ValueError("max_tool_rows must be greater than zero.")
    if max_tool_response_chars < 256:
        raise ValueError("max_tool_response_chars must be at least 256.")
    if max_query_chars < 256:
        raise ValueError("max_query_chars must be at least 256.")
    if tool_call_skip_reason is None and validator is None:
        raise ValueError("validator is required when tool-call evaluation is not skipped.")

    model, tokenizer = load_student_model_and_tokenizer(
        student_config=student_config,
        adapter_dir=adapter_dir,
        explicit_tokenizer=explicit_tokenizer,
        disable_cache=False,
        show_progress=show_progress,
        use_adapter=not baseline,
    )
    records = read_jsonl(data_config["splits"][split])
    if limit is not None:
        records = records[:limit]
    tool_schemas = [run_cypher_tool_schema()]

    evaluated_records = []
    with raw_generation_progress_context(show_progress) as progress:
        record_task_id = None
        token_task_id = None
        if progress is not None:
            record_task_id = progress.add_task(
                f"Validating records · {split_display_name(split)}",
                total=len(records),
            )
            token_task_id = progress.add_task(
                "Generating response",
                total=1,
                visible=False,
            )

        for record in records:
            final_messages = final_answer_prompt_messages(record)
            tool_messages = initial_prompt_messages(record)
            if progress is not None:
                progress.update(
                    record_task_id,
                    description=raw_generation_progress_description(
                        split=split,
                        phase="final-answer",
                    ),
                )
            try:
                final_generation = generate_assistant_response(
                    model,
                    tokenizer,
                    final_messages,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    tools=tool_schemas,
                    progress=progress,
                    progress_task_id=token_task_id,
                    progress_description=raw_generation_progress_description(
                        split=split,
                        phase="final-answer tokens",
                    ),
                )
                final_answer_result = score_final_answer_generation(
                    record,
                    final_generation.text,
                )
                add_raw_generation_debug_fields(final_answer_result, final_generation)
                final_answer_result["generation"] = generation_report_fields(
                    final_generation
                )
                final_generation_debug = generation_debug_report(
                    final_generation,
                )
            except Exception as exc:
                final_answer_result = final_answer_generation_error(record, exc)
                final_generation_debug = generation_debug_report(
                    None,
                )

            if tool_call_skip_reason is not None:
                tool_call_result = skipped_tool_call_result(tool_call_skip_reason)
                tool_generation_debug = generation_debug_report(
                    None,
                )
            else:
                if progress is not None:
                    progress.update(
                        record_task_id,
                        description=raw_generation_progress_description(
                            split=split,
                            phase="tool-call",
                        ),
                    )
                try:
                    tool_generation = generate_assistant_response(
                        model,
                        tokenizer,
                        tool_messages,
                        max_new_tokens=max_new_tokens,
                        enable_thinking=enable_thinking,
                        tools=tool_schemas,
                        progress=progress,
                        progress_task_id=token_task_id,
                        progress_description=raw_generation_progress_description(
                            split=split,
                            phase="tool-call tokens",
                        ),
                    )
                    tool_call_result = score_tool_call_generation(
                        record,
                        tool_generation.text,
                        validator=validator,
                        policy=policy,
                        max_rows=max_tool_rows,
                        max_result_chars=max_tool_response_chars,
                    )
                    add_raw_generation_debug_fields(tool_call_result, tool_generation)
                    tool_call_result["generation"] = generation_report_fields(
                        tool_generation
                    )
                    tool_generation_debug = generation_debug_report(
                        tool_generation,
                    )
                except Exception as exc:
                    tool_call_result = tool_call_generation_error(record, exc)
                    tool_generation_debug = generation_debug_report(
                        None,
                    )

            end_to_end_generation_debug: dict[str, Any] = {"turns": []}
            if tool_call_skip_reason is not None:
                end_to_end_result = end_to_end_skipped_result(tool_call_skip_reason)
            else:
                if progress is not None:
                    progress.update(
                        record_task_id,
                        description=raw_generation_progress_description(
                            split=split,
                            phase="end-to-end",
                        ),
                    )
                try:
                    end_to_end_result = evaluate_end_to_end_record(
                        record,
                        model=model,
                        tokenizer=tokenizer,
                        validator=validator,
                        policy=policy,
                        tool_schemas=tool_schemas,
                        max_new_tokens=max_new_tokens,
                        enable_thinking=enable_thinking,
                        max_query_retries=max_query_retries,
                        max_tool_rows=max_tool_rows,
                        max_tool_response_chars=max_tool_response_chars,
                        max_query_chars=max_query_chars,
                        progress=progress,
                        progress_task_id=token_task_id,
                        progress_description=raw_generation_progress_description(
                            split=split,
                            phase="end-to-end tokens",
                        ),
                        generation_debug_turns=end_to_end_generation_debug["turns"],
                    )
                except Exception as exc:
                    end_to_end_result = end_to_end_failure_result(
                        record,
                        status="evaluation_error",
                        query_attempts=0,
                        max_query_retries=max_query_retries,
                        attempts=[],
                        generation_error=bounded_text(
                            f"{type(exc).__name__}: {exc}", 1024
                        ),
                    )

            evaluated_records.append(
                raw_generation_report_record(
                    record,
                    final_answer_result=final_answer_result,
                    tool_call_result=tool_call_result,
                    final_generation_debug=final_generation_debug,
                    tool_generation_debug=tool_generation_debug,
                    end_to_end_result=end_to_end_result,
                    end_to_end_generation_debug=end_to_end_generation_debug,
                )
            )
            release_generation_memory()
            if progress is not None:
                progress.update(
                    record_task_id,
                    advance=1,
                    description=raw_generation_progress_description(
                        split=split,
                        phase="done",
                    ),
                )
        if progress is not None:
            progress.update(
                record_task_id,
                description=f"{split_display_name(split)} validation complete",
            )
            progress.update(token_task_id, visible=False)

    return {
        "summary": summarize_raw_generation_records(evaluated_records),
        "records": evaluated_records,
        "generation_config": generation_config_report(
            model,
            max_new_tokens=max_new_tokens,
        ),
    }


def build_report(
    *,
    summary: dict[str, Any],
    data_config_path: str,
    student_config_path: str,
    data_config: dict[str, Any],
    student_config: dict[str, Any],
    split: str,
    mode: str,
    adapter_dir: str,
    explicit_tokenizer: str | None,
    baseline: bool = False,
    batch_size: int | None = None,
    max_length: int | None = None,
    loss_mode: str | None = None,
    records: list[dict[str, Any]] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "split": split,
        "data_config": data_config_path,
        "student_config": student_config_path,
        "split_path": (
            data_config["splits"][split]
            if split in data_config["splits"]
            else {name: data_config["splits"][name] for name in SUPPORTED_SPLITS}
        ),
        "base_model": student_config["model"]["base_model"],
        "model_variant": "baseline" if baseline else "fine-tuned",
        "adapter_dir": None if baseline else adapter_dir,
        "tokenizer": tokenizer_source(
            adapter_dir=adapter_dir,
            explicit_tokenizer=explicit_tokenizer,
            model_config=student_config["model"],
            use_adapter=not baseline,
        ),
    }
    if batch_size is not None:
        metadata["batch_size"] = batch_size
    if max_length is not None:
        metadata["max_length"] = max_length
    if loss_mode is not None:
        metadata["loss_mode"] = loss_mode
    if extra_metadata:
        metadata.update(extra_metadata)

    report = {
        "metadata": metadata,
        "summary": summary,
    }
    if records is not None:
        report["records"] = records
    return report


def write_report(report: dict[str, Any], output: str | Path) -> Path:
    output_path = ensure_parent(output)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def connect_raw_generation_validator(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
        Neo4jValidator,
        neo4j_connection_settings,
    )

    connection = neo4j_connection_settings(args)
    metadata = {
        "uri_candidates": connection["uri_candidates"],
        "username": connection["username"],
        "database": connection["database"],
        "connected": False,
        "connected_uri": None,
        "skip_reason": None,
    }
    try:
        validator = Neo4jValidator(
            uri=connection["uri_candidates"],
            username=connection["username"],
            password=connection["password"],
            database=connection["database"],
            timeout=args.neo4j_timeout,
            connection_timeout=args.neo4j_connection_timeout,
        )
    except Exception as exc:
        metadata["skip_reason"] = f"Neo4j unavailable: {type(exc).__name__}: {exc}"
        return None, metadata

    metadata["connected"] = True
    metadata["connected_uri"] = validator.uri
    return validator, metadata


def configure_neo4j_notification_output(*, show_notifications: bool) -> None:
    """Keep Neo4j notifications out of the terminal unless explicitly requested."""
    if show_notifications:
        return
    from cypher_3dsg_agent_distillation.graph.neo4j_validation import (
        suppress_neo4j_notification_logs,
    )

    suppress_neo4j_notification_logs()


def print_eval_loss_summary(
    *,
    split: str,
    mode: str,
    loss_mode: str,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    perplexity = summary["perplexity"]
    perplexity_text = f"{perplexity:.6f}" if perplexity is not None else "overflow"
    print_summary(
        "Student eval-loss complete",
        [
            ("Split", split),
            ("Mode", mode),
            ("Loss mode", loss_mode),
            ("Examples", summary["examples"]),
            ("Tokens", summary["tokens"]),
            ("Evaluation loss", f"{summary['eval_loss']:.6f}"),
            ("Perplexity", perplexity_text),
            ("Evaluation entropy", f"{summary['eval_entropy']:.6f}"),
            ("Mean token accuracy", f"{summary['eval_mean_token_accuracy']:.6f}"),
            ("Report", output_path),
        ],
    )


def print_eval_loss_all_splits_summary(
    *,
    mode: str,
    loss_mode: str,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    perplexity = summary["perplexity"]
    perplexity_text = f"{perplexity:.6f}" if perplexity is not None else "overflow"
    print_summary(
        "Student eval-loss complete",
        [
            ("Split", ALL_SPLITS),
            ("Mode", mode),
            ("Loss mode", loss_mode),
            ("Examples", summary["examples"]),
            ("Tokens", summary["tokens"]),
            ("Evaluation loss", f"{summary['eval_loss']:.6f}"),
            ("Perplexity", perplexity_text),
            ("Evaluation entropy", f"{summary['eval_entropy']:.6f}"),
            ("Mean token accuracy", f"{summary['eval_mean_token_accuracy']:.6f}"),
            ("Report", output_path),
        ],
    )
    print_breakdown(
        "Eval loss by split",
        ("Split", "Examples", "Tokens", "Evaluation loss"),
        (
            (
                split,
                split_summary["examples"],
                split_summary["tokens"],
                f"{split_summary['eval_loss']:.6f}",
            )
            for split, split_summary in summary["splits"].items()
        ),
        right_align=(1, 2, 3),
    )


def format_count_rate(section: dict[str, Any], metric: str) -> str:
    count = section["metrics"][metric]
    evaluated = section["evaluated"]
    rate = section["rates"][metric]
    return f"{count}/{evaluated} ({rate:.6f})"


def print_raw_generation_summary(
    *,
    split: str,
    mode: str,
    summary: dict[str, Any],
    output_path: Path,
    baseline: bool = False,
) -> None:
    final_answer = summary[FINAL_ANSWER_CHECK]
    tool_call = summary[TOOL_CALL_CHECK]
    end_to_end = summary[END_TO_END_CHECK]
    rows = [
        ("Split", split),
        ("Mode", mode),
        ("Examples", summary["examples"]),
        ("Final-answer match", format_count_rate(final_answer, "final_answer_match")),
        ("Final-answer tags", format_count_rate(final_answer, "final_answer_has_tags")),
    ]
    if end_to_end["skipped"] == end_to_end["total"]:
        rows.append(
            (
                "End-to-end validation",
                f"skipped {end_to_end['skipped']}/{end_to_end['total']}",
            )
        )
    else:
        rows.append(
            (
                "End-to-end final-answer match",
                format_count_rate(end_to_end, "final_answer_match"),
            )
        )
    if tool_call["skipped"]:
        rows.append(
            ("Tool-call validation", f"skipped {tool_call['skipped']}/{tool_call['total']}")
        )
    else:
        rows.extend(
            [
                (
                    "Tool-call parse success",
                    format_count_rate(tool_call, "tool_call_parse_success"),
                ),
                ("Static-valid Cypher", format_count_rate(tool_call, "static_valid")),
                ("Executable Cypher", format_count_rate(tool_call, "executable")),
                ("Cypher solution match", format_count_rate(tool_call, "cypher_solution_match")),
                (
                    "Potential hidden false positives",
                    format_count_rate(tool_call, "potential_hidden_false_positive"),
                ),
            ]
        )
    rows.append(("Report", output_path))
    title = (
        "Baseline raw-generation validation complete"
        if baseline
        else "Student raw-generation validation complete"
    )
    print_summary(title, rows)


def command(args: argparse.Namespace) -> None:
    configure_model_library_output()
    data_config = load_data_config(args.data_config)
    student_config = load_yaml(args.student_config)
    adapter_dir = args.adapter_dir or default_adapter_dir(student_config)
    if args.baseline and args.mode != RAW_GENERATION_MODE:
        raise ValueError("--baseline is supported only with --mode raw-generation.")

    records = None
    extra_metadata = None
    batch_size = None
    max_length = None
    loss_mode = None
    if args.mode == EVAL_LOSS_MODE:
        from cypher_3dsg_agent_distillation.training.train import (
            loss_mode_from_training_config,
            normalize_loss_mode,
        )

        batch_size = args.batch_size
        max_length = args.max_length or int(student_config["training"].get("max_length", 4096))
        loss_mode = (
            normalize_loss_mode(args.loss_mode)
            if args.loss_mode is not None
            else loss_mode_from_training_config(student_config["training"])
        )
        if args.split == ALL_SPLITS:
            summary = combine_eval_loss_summaries(
                evaluate_eval_loss_splits(
                    data_config=data_config,
                    student_config=student_config,
                    splits=SUPPORTED_SPLITS,
                    adapter_dir=adapter_dir,
                    explicit_tokenizer=args.tokenizer,
                    batch_size=batch_size,
                    max_length=max_length,
                    loss_mode=loss_mode,
                    show_progress=not args.no_progress,
                )
            )
        else:
            summary = evaluate_eval_loss(
                data_config=data_config,
                student_config=student_config,
                split=args.split,
                adapter_dir=adapter_dir,
                explicit_tokenizer=args.tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                loss_mode=loss_mode,
                show_progress=not args.no_progress,
            )
    else:
        if args.split == ALL_SPLITS:
            raise ValueError("raw-generation mode does not support `--split all`.")
        from cypher_3dsg_agent_distillation.graph.neo4j_validation import schema_policy

        show_db_notifications = bool(getattr(args, "show_db_notifications", False))
        configure_neo4j_notification_output(
            show_notifications=show_db_notifications,
        )

        validator, neo4j_metadata = connect_raw_generation_validator(args)
        try:
            raw_generation = evaluate_raw_generation(
                data_config=data_config,
                student_config=student_config,
                split=args.split,
                adapter_dir=adapter_dir,
                explicit_tokenizer=args.tokenizer,
                max_new_tokens=args.max_new_tokens,
                limit=args.limit,
                validator=validator,
                tool_call_skip_reason=neo4j_metadata["skip_reason"],
                policy=schema_policy(load_yaml(data_config["schema_path"])),
                show_progress=not args.no_progress,
                enable_thinking=args.enable_thinking,
                max_query_retries=args.max_query_retries,
                max_tool_rows=args.e2e_max_tool_rows,
                max_tool_response_chars=args.e2e_max_tool_response_chars,
                max_query_chars=args.e2e_max_query_chars,
                baseline=args.baseline,
            )
        finally:
            if validator is not None:
                validator.close()
        summary = raw_generation["summary"]
        records = raw_generation["records"]
        extra_metadata = {
            "raw_generation": {
                "checks": [FINAL_ANSWER_CHECK, TOOL_CALL_CHECK, END_TO_END_CHECK],
                "limit": args.limit,
                "max_new_tokens": args.max_new_tokens,
                "max_query_retries": args.max_query_retries,
                "max_query_attempts": args.max_query_retries + 1,
                "e2e_max_tool_rows": args.e2e_max_tool_rows,
                "e2e_max_tool_response_chars": args.e2e_max_tool_response_chars,
                "e2e_max_query_chars": args.e2e_max_query_chars,
                "enable_thinking": args.enable_thinking,
                "progress": not args.no_progress,
                "model_variant": "baseline" if args.baseline else "fine-tuned",
                "generation_config": raw_generation["generation_config"],
                "neo4j": neo4j_metadata,
                "show_db_notifications": show_db_notifications,
            }
        }

    report = build_report(
        summary=summary,
        data_config_path=args.data_config,
        student_config_path=args.student_config,
        data_config=data_config,
        student_config=student_config,
        split=args.split,
        mode=args.mode,
        adapter_dir=adapter_dir,
        explicit_tokenizer=args.tokenizer,
        baseline=args.baseline,
        batch_size=batch_size,
        max_length=max_length,
        loss_mode=loss_mode,
        records=records,
        extra_metadata=extra_metadata,
    )
    output_path = write_report(
        report,
        report_output_path(
            data_config=data_config,
            split=args.split,
            mode=args.mode,
            output=args.output,
            baseline=args.baseline,
        ),
    )

    if args.mode == EVAL_LOSS_MODE:
        if args.split == ALL_SPLITS:
            print_eval_loss_all_splits_summary(
                mode=args.mode,
                loss_mode=loss_mode,
                summary=summary,
                output_path=output_path,
            )
        else:
            print_eval_loss_summary(
                split=args.split,
                mode=args.mode,
                loss_mode=loss_mode,
                summary=summary,
                output_path=output_path,
            )
    else:
        print_raw_generation_summary(
            split=args.split,
            mode=args.mode,
            summary=summary,
            output_path=output_path,
            baseline=args.baseline,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned or baseline student model on configured splits."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--student-config",
        default="configs/student_qwen3.5_4b.yaml",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="test",
        help="Configured split to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=(EVAL_LOSS_MODE, RAW_GENERATION_MODE),
        default=EVAL_LOSS_MODE,
        help=(
            "`eval-loss` computes held-out loss/perplexity. "
            "`raw-generation` evaluates the existing isolated checks plus a bounded "
            "end-to-end query/tool/final-answer interaction."
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="LoRA adapter directory. Defaults to training.output_dir in the student config.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Evaluate the configured base student model without loading its LoRA "
            "adapter. Supported only in raw-generation mode."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer path/name. Fine-tuned evaluation defaults to the adapter "
            "tokenizer when available; baseline evaluation uses the configured model "
            "tokenizer."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--loss-mode",
        choices=("full", "assistant-only"),
        default=None,
        help=(
            "Token labels to score for eval-loss. Defaults to training.loss_mode in "
            "the student config, or assistant-only when unset."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Maximum tokenized example length for eval-loss mode. Defaults to "
            "training.max_length in the student config."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to evaluate in raw-generation mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Optional generated-token limit for raw-generation mode. When unset, "
            "the loaded model generation_config is used."
        ),
    )
    parser.add_argument(
        "--max-query-retries",
        type=int,
        default=DEFAULT_MAX_QUERY_RETRIES,
        help=(
            "Maximum corrected query retries in end-to-end validation after the "
            "initial query attempt (default: 1)."
        ),
    )
    parser.add_argument(
        "--e2e-max-tool-rows",
        type=int,
        default=DEFAULT_E2E_MAX_TOOL_ROWS,
        help=(
            "Maximum rows streamed from an end-to-end query before it is retried "
            f"(default: {DEFAULT_E2E_MAX_TOOL_ROWS})."
        ),
    )
    parser.add_argument(
        "--e2e-max-tool-response-chars",
        type=int,
        default=DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS,
        help=(
            "Maximum serialized tool-response characters fed back to the model "
            f"(default: {DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS})."
        ),
    )
    parser.add_argument(
        "--e2e-max-query-chars",
        type=int,
        default=DEFAULT_E2E_MAX_QUERY_CHARS,
        help="Maximum generated Cypher characters accepted for execution.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Allow tokenizer chat-template thinking mode for raw-generation. By "
            "default, thinking is disabled when the tokenizer supports it."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Neo4j URI or comma-separated URI fallback list for raw-generation mode.",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Neo4j username for raw-generation mode.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Neo4j password for raw-generation mode.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Neo4j database name for raw-generation mode.",
    )
    parser.add_argument(
        "--neo4j-timeout",
        type=float,
        default=10.0,
        help="Per-query Neo4j timeout in seconds for raw-generation mode.",
    )
    parser.add_argument(
        "--neo4j-connection-timeout",
        type=float,
        default=3.0,
        help="Neo4j connection timeout in seconds for raw-generation mode.",
    )
    parser.add_argument(
        "--show-db-notifications",
        action="store_true",
        help=(
            "Show Neo4j DBMS notification warnings during raw-generation queries. "
            "They are hidden by default but remain available in validation results."
        ),
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
