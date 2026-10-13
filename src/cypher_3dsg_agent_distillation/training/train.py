from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.agents.agent_prompts import run_cypher_tool_schema
from cypher_3dsg_agent_distillation.core.cli_output import phase_progress, print_summary
from cypher_3dsg_agent_distillation.core.config import (
    PROJECT_ROOT,
    load_data_config,
    load_student_config,
    resolve_project_path,
)
from cypher_3dsg_agent_distillation.core.io import read_jsonl
from cypher_3dsg_agent_distillation.core.model_runtime_output import (
    configure_model_library_output,
)
from cypher_3dsg_agent_distillation.core.versioning import (
    atomic_write_yaml,
    config_record,
    file_fingerprint,
    git_provenance,
    iso_now,
    read_metadata,
    serializable_arguments,
    split_fingerprints,
)
from cypher_3dsg_agent_distillation.training.chunked_loss import chunked_loss_scaling

DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
FULL_LOSS_MODE = "full"
ASSISTANT_ONLY_LOSS_MODE = "assistant-only"
SUPPORTED_LOSS_MODES = (FULL_LOSS_MODE, ASSISTANT_ONLY_LOSS_MODE)
DEFAULT_LOSS_MODE = ASSISTANT_ONLY_LOSS_MODE
IGNORE_INDEX = -100


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


def quantization_config(model_config: dict[str, Any]):
    if not model_config.get("load_in_4bit", False):
        return None
    from transformers import BitsAndBytesConfig

    compute_dtype = dtype_from_name(model_config.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(model_config.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(
    config: dict[str, Any],
    *,
    progress=None,
    progress_task_id=None,
):
    from transformers import AutoModelForCausalLM

    model_config = config["model"]
    model_id = model_config["base_model"]
    if progress is not None:
        progress.update(progress_task_id, description="Loading tokenizer")
    tokenizer = load_tokenizer(config)
    if progress is not None:
        progress.update(progress_task_id, advance=1)

    if progress is not None:
        progress.update(progress_task_id, description="Loading base model weights")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype_from_name(model_config.get("dtype", "auto")),
        quantization_config=quantization_config(model_config),
        device_map="auto",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    align_model_token_ids(model, tokenizer)
    if progress is not None:
        progress.update(progress_task_id, advance=1)
    return model, tokenizer


def load_tokenizer(config: dict[str, Any], tokenizer_id: str | None = None):
    from transformers import AutoTokenizer

    model_config = config["model"]
    tokenizer_id = tokenizer_id or model_config.get("tokenizer") or model_config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


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
        if not gradient_checkpointing_kwargs:
            model.gradient_checkpointing_enable()
            return
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            try:
                model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
            except TypeError:
                model.gradient_checkpointing_enable()


def prepare_gemma4_effective_model_for_kbit_training(
    model: Any,
    *,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
):
    import torch

    for name, param in model.named_parameters():
        param.requires_grad = False
        if (
            should_upcast_gemma4_effective_param(name, param)
            and param.dtype in {torch.float16, torch.bfloat16}
        ):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing and (
        getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    ):
        enable_gradient_checkpointing_for_kbit_training(
            model,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )
    return model


def prepare_quantized_model_for_training(
    model: Any,
    model_config: dict[str, Any],
    prepare_model_for_kbit_training,
    *,
    use_gradient_checkpointing: bool = True,
):
    gradient_checkpointing_kwargs = {"use_reentrant": False} if use_gradient_checkpointing else None
    if is_gemma4_effective_model(model, model_config):
        return prepare_gemma4_effective_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )
    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )


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
        content = message.get("content") or ""
        calls = json.dumps(message['tool_calls'], ensure_ascii=False, sort_keys=True)
        return f"{role}: {content + chr(10) if content else ''}{calls}"
    return f"{role}: {message.get('content', '')}"


def training_tool_schemas(record: dict[str, Any]) -> list[dict[str, Any]]:
    native_schema = run_cypher_tool_schema()
    native_name = native_schema["function"]["name"]
    declared_tools = record.get("tools") or [native_name]
    schemas: list[dict[str, Any]] = []
    names = set()
    for declared_tool in declared_tools:
        schema = (
            declared_tool
            if isinstance(declared_tool, dict)
            else native_schema if declared_tool == native_name else None
        )
        if not isinstance(schema, dict):
            continue
        function = schema.get("function") or {}
        name = str(function.get("name") or "")
        if name and name not in names:
            schemas.append(schema)
            names.add(name)
    return schemas or [native_schema]


def to_chat_text(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    include_reasoning: bool = False,
) -> str:
    from cypher_3dsg_agent_distillation.agents.reasoning import (
        student_messages,
        training_template,
    )

    messages = student_messages(
        normalize_chat_messages(messages), tokenizer, include_reasoning=include_reasoning,
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
        if tools:
            kwargs["tools"] = tools
        template = training_template(tokenizer, include_reasoning=include_reasoning)
        if template is not None:
            kwargs["chat_template"] = template
        return tokenizer.apply_chat_template(messages, **kwargs)
    return "\n\n".join(message_to_plain_text(message) for message in messages)


def tokenize_rendered_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = min(len(left), len(right))
    for index in range(length):
        if left[index] != right[index]:
            return index
    return length


def message_token_span(
    tokenizer,
    messages: list[dict[str, Any]],
    input_ids: list[int],
    message_index: int,
    *,
    tools: list[dict[str, Any]] | None = None,
    include_reasoning: bool = False,
) -> tuple[int, int]:
    before_ids = tokenize_rendered_text(
        tokenizer,
        to_chat_text(tokenizer, messages[:message_index], tools=tools,
                     include_reasoning=include_reasoning),
    )
    after_ids = tokenize_rendered_text(
        tokenizer,
        to_chat_text(tokenizer, messages[: message_index + 1], tools=tools,
                     include_reasoning=include_reasoning),
    )
    start = common_prefix_length(input_ids, before_ids)
    end = common_prefix_length(input_ids, after_ids)
    if start != len(before_ids) or end != len(after_ids) or end <= start:
        raise ValueError(
            f"Could not locate rendered message span at index {message_index}. "
            "Check that the tokenizer chat template renders partial transcripts "
            "as prefixes of the full transcript."
        )
    return start, end


def assistant_token_spans(
    tokenizer,
    messages: list[dict[str, Any]],
    input_ids: list[int],
    *,
    tools: list[dict[str, Any]] | None = None,
    excluded_message_indices: set[int] | None = None,
    include_reasoning: bool = False,
) -> list[tuple[int, int]]:
    excluded_message_indices = excluded_message_indices or set()
    return [
        message_token_span(
            tokenizer,
            messages,
            input_ids,
            index,
            tools=tools,
            include_reasoning=include_reasoning,
        )
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and index not in excluded_message_indices
    ]


def labels_for_loss_mode(
    *,
    tokenizer,
    messages: list[dict[str, Any]],
    input_ids: list[int],
    loss_mode: str,
    tools: list[dict[str, Any]] | None = None,
    non_trainable_message_indices: Sequence[int] = (),
    include_reasoning: bool = False,
) -> list[int]:
    loss_mode = normalize_loss_mode(loss_mode)
    excluded_indices = {int(index) for index in non_trainable_message_indices}
    if loss_mode == FULL_LOSS_MODE:
        labels = list(input_ids)
        for index in excluded_indices:
            start, end = message_token_span(
                tokenizer,
                messages,
                input_ids,
                index,
                tools=tools,
                include_reasoning=include_reasoning,
            )
            labels[start:end] = [IGNORE_INDEX] * (end - start)
        return labels

    labels = [IGNORE_INDEX] * len(input_ids)
    for start, end in assistant_token_spans(
        tokenizer,
        messages,
        input_ids,
        tools=tools,
        excluded_message_indices=excluded_indices,
        include_reasoning=include_reasoning,
    ):
        labels[start:end] = input_ids[start:end]
    return labels


class OverlengthTrainingTraceError(ValueError):
    def __init__(self, uid: str, token_length: int, max_length: int):
        self.uid = uid
        self.token_length = token_length
        self.max_length = max_length
        super().__init__(
            f"Trace {uid or '<unknown>'} has {token_length} tokens, which exceeds "
            f"max_length={max_length}."
        )


class OverlengthTrainingTraceWarning(UserWarning):
    pass


def validate_training_length(
    *,
    uid: str,
    token_length: int,
    max_length: int | None,
) -> None:
    if max_length is None:
        return
    if max_length <= 0:
        raise ValueError("training.max_length must be greater than zero.")
    if token_length > max_length:
        raise OverlengthTrainingTraceError(uid, token_length, max_length)


def warn_about_rejected_traces(
    *,
    path: str,
    rejected_count: int,
    total_count: int,
    max_length: int | None,
) -> None:
    if rejected_count == 0:
        return
    warnings.warn(
        f"Rejected {rejected_count} of {total_count} traces from {path} because their "
        f"rendered token length exceeds max_length={max_length}. Overlength traces are "
        "excluded instead of truncated.",
        OverlengthTrainingTraceWarning,
        stacklevel=2,
    )


def require_retained_traces(
    *,
    path: str,
    retained_count: int,
    total_count: int,
    max_length: int | None,
) -> None:
    if retained_count > 0 or total_count == 0:
        return
    raise ValueError(
        f"All {total_count} traces from {path} exceed max_length={max_length}. "
        "Increase training.max_length before training."
    )


def active_label_count(labels: Sequence[int]) -> int:
    return sum(1 for label in labels if label != IGNORE_INDEX)


def build_training_example(
    record: dict[str, Any],
    tokenizer,
    *,
    loss_mode: str,
    max_length: int | None,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    loss_mode = normalize_loss_mode(loss_mode)
    messages = normalize_chat_messages(record.get("messages", []))
    tools = training_tool_schemas(record)
    if loss_mode == FULL_LOSS_MODE and record.get("text"):
        text = (
            to_chat_text(tokenizer, messages, tools=tools, include_reasoning=include_reasoning)
            if messages
            else str(record["text"])
        )
    else:
        if not messages:
            raise ValueError(
                "Assistant-only training requires records with structured `messages`."
            )
        text = to_chat_text(tokenizer, messages, tools=tools, include_reasoning=include_reasoning)
    input_ids = tokenize_rendered_text(tokenizer, text)
    validate_training_length(
        uid=str(record.get("uid", "")),
        token_length=len(input_ids),
        max_length=max_length,
    )
    example = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels_for_loss_mode(
            tokenizer=tokenizer,
            messages=messages,
            input_ids=input_ids,
            loss_mode=loss_mode,
            include_reasoning=include_reasoning,
            tools=tools,
            non_trainable_message_indices=(
                (record.get("metadata") or {}).get("non_trainable_message_indices") or []
            ),
        ),
        "uid": str(record.get("uid", "")),
    }
    if active_label_count(example["labels"]) == 0:
        raise ValueError(
            f"Record {record.get('uid', '<unknown>')} has no supervised tokens after "
            f"applying loss mode `{loss_mode}`."
        )
    return example


def rendered_record_text(
    record: dict[str, Any], tokenizer, *, include_reasoning: bool = False,
) -> str:
    return (
        to_chat_text(
            tokenizer,
            record["messages"],
            tools=training_tool_schemas(record),
            include_reasoning=include_reasoning,
        )
        if record.get("messages")
        else str(record.get("text") or "")
    )


def build_dataset(
    path: str, tokenizer, *, max_length: int | None = None, include_reasoning: bool = False,
):
    from datasets import Dataset

    records = read_jsonl(path)
    examples = []
    rejected_count = 0
    for record in records:
        text = rendered_record_text(record, tokenizer, include_reasoning=include_reasoning)
        try:
            if max_length is not None:
                validate_training_length(
                    uid=str(record.get("uid", "")),
                    token_length=len(tokenize_rendered_text(tokenizer, text)),
                    max_length=max_length,
                )
        except OverlengthTrainingTraceError:
            rejected_count += 1
            continue
        examples.append({"text": text})
    warn_about_rejected_traces(
        path=path,
        rejected_count=rejected_count,
        total_count=len(records),
        max_length=max_length,
    )
    require_retained_traces(
        path=path,
        retained_count=len(examples),
        total_count=len(records),
        max_length=max_length,
    )
    return Dataset.from_list(examples)


def build_training_dataset(
    path: str,
    tokenizer,
    *,
    loss_mode: str,
    max_length: int | None,
    include_reasoning: bool = False,
    retained_uids: set[str] | None = None,
):
    from datasets import Dataset

    records = read_jsonl(path)
    examples = []
    rejected_count = 0
    if retained_uids is not None:
        records = [record for record in records if str(record.get("uid")) in retained_uids]
    for record in records:
        try:
            example = build_training_example(
                record,
                tokenizer,
                loss_mode=loss_mode,
                max_length=max_length,
                include_reasoning=include_reasoning,
            )
        except OverlengthTrainingTraceError:
            rejected_count += 1
            continue
        examples.append(example)
    warn_about_rejected_traces(
        path=path,
        rejected_count=rejected_count,
        total_count=len(records),
        max_length=max_length,
    )
    require_retained_traces(
        path=path,
        retained_count=len(examples),
        total_count=len(records),
        max_length=max_length,
    )
    return Dataset.from_list(examples)


def pad_causal_lm_features(
    features: list[dict[str, Any]],
    tokenizer,
) -> dict[str, list[list[int]]]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    padding_side = getattr(tokenizer, "padding_side", "right")
    max_length = max(len(feature["input_ids"]) for feature in features)

    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        if "labels" not in feature:
            raise KeyError(
                "labels are required for assistant-only training. "
                "Set `remove_unused_columns: false` or use the configured SFTConfig path."
            )
        input_ids = list(feature["input_ids"])
        raw_attention_mask = feature.get("attention_mask")
        attention_mask = (
            [1] * len(input_ids) if raw_attention_mask is None else list(raw_attention_mask)
        )
        labels = list(feature["labels"])
        if not len(input_ids) == len(attention_mask) == len(labels):
            raise ValueError(
                "input_ids, attention_mask, and labels must have the same length "
                f"before padding; got {len(input_ids)}, {len(attention_mask)}, "
                f"and {len(labels)}."
            )

        pad_length = max_length - len(input_ids)
        if padding_side == "left":
            batch["input_ids"].append([pad_token_id] * pad_length + input_ids)
            batch["attention_mask"].append([0] * pad_length + attention_mask)
            batch["labels"].append([IGNORE_INDEX] * pad_length + labels)
        else:
            batch["input_ids"].append(input_ids + [pad_token_id] * pad_length)
            batch["attention_mask"].append(attention_mask + [0] * pad_length)
            batch["labels"].append(labels + [IGNORE_INDEX] * pad_length)

    return batch


class CausalLmDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = pad_causal_lm_features(features, self.tokenizer)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def normalize_loss_mode(value: str | None) -> str:
    normalized = str(value or DEFAULT_LOSS_MODE).strip().lower().replace("_", "-")
    if normalized not in SUPPORTED_LOSS_MODES:
        raise ValueError(
            f"Unsupported training loss mode `{value}`. "
            f"Expected one of: {', '.join(SUPPORTED_LOSS_MODES)}."
        )
    return normalized


def loss_mode_from_training_config(training_config: dict[str, Any]) -> str:
    return normalize_loss_mode(training_config.get("loss_mode", DEFAULT_LOSS_MODE))


def retained_uids_from_training_config(training_config: dict[str, Any]) -> set[str] | None:
    path = training_config.get("retained_uids_path")
    if not path:
        return None
    values = json.loads(resolve_project_path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("training.retained_uids_path must contain a JSON array of trace UIDs.")
    return set(values)


def training_arguments_kwargs(
    training_config: dict[str, Any],
    has_eval_dataset: bool,
) -> dict[str, Any]:
    configured_warmup = training_config.get(
        "warmup_steps",
        training_config.get("warmup_ratio", 0.05),
    )
    if isinstance(configured_warmup, bool):
        raise ValueError(
            "training.warmup_steps must be a non-negative integer step count "
            "or a float in [0, 1)."
        )
    warmup_steps = float(configured_warmup)
    if not math.isfinite(warmup_steps) or warmup_steps < 0 or (
        warmup_steps >= 1 and not warmup_steps.is_integer()
    ):
        raise ValueError(
            "training.warmup_steps must be a non-negative integer step count "
            "or a float in [0, 1)."
        )
    normalized_warmup_steps: int | float = (
        warmup_steps if warmup_steps < 1 else int(warmup_steps)
    )
    kwargs = {
        "output_dir": training_config["output_dir"],
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(training_config.get("learning_rate", 2e-4)),
        "num_train_epochs": float(training_config.get("num_train_epochs", 3)),
        "warmup_steps": normalized_warmup_steps,
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "eval_steps": int(training_config.get("eval_steps", 100)),
        "save_steps": int(training_config.get("save_steps", 100)),
        "save_total_limit": int(training_config.get("save_total_limit", 3)),
        "bf16": bool(training_config.get("bf16", True)),
        "fp16": bool(training_config.get("fp16", False)),
        "gradient_checkpointing": bool(training_config.get("gradient_checkpointing", True)),
        "report_to": training_config.get("report_to", "none"),
        "eval_strategy": "steps" if has_eval_dataset else "no",
        "save_strategy": training_config.get("save_strategy", "steps"),
        "disable_tqdm": True,
    }
    if has_eval_dataset:
        kwargs.update(
            {
                "load_best_model_at_end": bool(
                    training_config.get("load_best_model_at_end", False)
                ),
                "metric_for_best_model": training_config.get(
                    "metric_for_best_model", "eval_loss"
                ),
                "greater_is_better": bool(
                    training_config.get("greater_is_better", False)
                ),
            }
        )
    return kwargs


def sft_config_kwargs(
    training_config: dict[str, Any],
    has_eval_dataset: bool,
) -> dict[str, Any]:
    kwargs = training_arguments_kwargs(training_config, has_eval_dataset)
    # Keep the memory-efficient path explicit rather than inheriting TRL defaults.
    loss_type = training_config.get("loss_type")
    if loss_type is None:
        loss_type = "chunked_nll"
    if loss_type not in ("nll", "chunked_nll", "dft"):
        raise ValueError(
            "training.loss_type must be one of: nll, chunked_nll, dft."
        )
    kwargs["loss_type"] = loss_type
    kwargs["max_length"] = int(training_config.get("max_length", 4096))
    dataset_kwargs = dict(training_config.get("dataset_kwargs") or {})
    dataset_kwargs["skip_prepare_dataset"] = True
    kwargs["dataset_kwargs"] = dataset_kwargs
    kwargs["remove_unused_columns"] = False
    return kwargs


def sft_trainer_kwargs(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args,
    peft_config,
    data_collator=None,
    callbacks=None,
) -> dict[str, Any]:
    kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
        "peft_config": peft_config,
        "processing_class": tokenizer,
    }
    if data_collator is not None:
        kwargs["data_collator"] = data_collator
    if callbacks:
        kwargs["callbacks"] = callbacks
    return kwargs


def early_stopping_callback(training_config: dict[str, Any], *, has_eval_dataset: bool):
    patience = int(training_config.get("early_stopping_patience", 0))
    if patience <= 0:
        return None
    if not has_eval_dataset:
        raise ValueError("Early stopping requires an evaluation dataset.")
    if not training_config.get("load_best_model_at_end", False):
        raise ValueError("Early stopping requires training.load_best_model_at_end=true.")

    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=float(
            training_config.get("early_stopping_threshold", 0.0)
        ),
    )


def rich_trainer_progress_callback(progress, task_id):
    from transformers import TrainerCallback

    class RichTrainerProgressCallback(TrainerCallback):
        def on_train_begin(self, _args, state, control, **_kwargs):
            progress.update(
                task_id,
                total=max(1, state.max_steps),
                completed=state.global_step,
                description="Training student LoRA adapter",
            )
            return control

        def on_step_end(self, _args, state, control, **_kwargs):
            progress.update(task_id, completed=state.global_step)
            return control

        def on_log(self, _args, state, control, logs=None, **_kwargs):
            logs = logs or {}
            details = []
            for label, key in (("loss", "loss"), ("eval", "eval_loss"), ("lr", "learning_rate")):
                value = logs.get(key)
                if isinstance(value, int | float):
                    details.append(f"{label}={value:.4g}")
            description = "Training student LoRA adapter"
            if details:
                description = f"{description} ({', '.join(details)})"
            progress.update(task_id, completed=state.global_step, description=description)
            return control

        def on_train_end(self, _args, state, control, **_kwargs):
            progress.update(
                task_id,
                completed=max(state.global_step, state.max_steps),
                description="Student LoRA training complete",
            )
            return control

    return RichTrainerProgressCallback()


def command(args: argparse.Namespace) -> None:
    configure_model_library_output()
    LoraConfig, prepare_model_for_kbit_training, SFTConfig, SFTTrainer = (
        require_training_dependencies()
    )
    data_config = load_data_config(
        args.data_config, dataset_version=args.dataset_version
    )
    config = load_student_config(
        args.student_config,
        model_version=args.model_version,
        create_version=True,
    )
    if getattr(args, "include_reasoning", None) is not None:
        config["training"]["include_reasoning"] = args.include_reasoning
    split_files, split_fingerprint = split_fingerprints(
        data_config["splits"], project_root=PROJECT_ROOT
    )
    created_at = iso_now()
    atomic_write_yaml(
        {
            "schema_version": 1,
            "artifact_type": "lora_model",
            "model_family": config["model_family"],
            "version": config["model_version"],
            "created_at": created_at,
            "updated_at": created_at,
            "status": "running",
            "dataset": {
                "dataset_id": data_config["dataset_id"],
                "scene_ids": [scene["scene_id"] for scene in data_config["scenes"]],
                "version": data_config["dataset_version"],
                "teacher_name": data_config.get("teacher_name"),
                "split_fingerprint": split_fingerprint,
                "split_files": split_files,
            },
            "student_config": config_record(
                config["student_config_path"], project_root=PROJECT_ROOT
            ),
            "effective_config": {
                "model": config["model"],
                "lora": config["lora"],
                "training": config["training"],
            },
            "arguments": serializable_arguments(args),
            "code": git_provenance(PROJECT_ROOT),
        },
        config["model_metadata_path"],
    )
    with phase_progress("Loading student model", 2) as (progress, task_id):
        model, tokenizer = load_model_and_tokenizer(
            config,
            progress=progress,
            progress_task_id=task_id,
        )

    training_config = config["training"]
    if config["model"].get("load_in_4bit", False):
        model = prepare_quantized_model_for_training(
            model,
            config["model"],
            prepare_model_for_kbit_training,
            use_gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
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

    max_length = int(training_config.get("max_length", 4096))
    include_reasoning = bool(training_config.get("include_reasoning", False))
    retained_uids = retained_uids_from_training_config(training_config)
    loss_mode = loss_mode_from_training_config(training_config)
    with phase_progress("Preparing training datasets", 2) as (progress, task_id):
        train_dataset = build_training_dataset(
            data_config["splits"]["train"],
            tokenizer,
            loss_mode=loss_mode,
            max_length=max_length,
            include_reasoning=include_reasoning,
            retained_uids=retained_uids,
        )
        progress.update(task_id, advance=1, description="Preparing evaluation dataset")
        eval_dataset = build_training_dataset(
            data_config["splits"]["eval"],
            tokenizer,
            loss_mode=loss_mode,
            max_length=max_length,
            include_reasoning=include_reasoning,
            retained_uids=retained_uids,
        )
        data_collator = CausalLmDataCollator(tokenizer)
        progress.update(task_id, advance=1)
    training_args = SFTConfig(
        **sft_config_kwargs(
            training_config,
            has_eval_dataset=eval_dataset is not None,
        )
    )
    metadata = read_metadata(config["model_metadata_path"])
    metadata["effective_training_arguments"] = json.loads(
        json.dumps(training_args.to_dict(), default=str)
    )
    atomic_write_yaml(metadata, config["model_metadata_path"])
    callbacks = []
    stopping_callback = early_stopping_callback(
        training_config,
        has_eval_dataset=eval_dataset is not None,
    )
    if stopping_callback is not None:
        callbacks.append(stopping_callback)

    with chunked_loss_scaling(model, training_args.loss_type):
        trainer = SFTTrainer(
            **sft_trainer_kwargs(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_args=training_args,
                peft_config=peft_config,
                data_collator=data_collator,
                callbacks=callbacks,
            )
        )
    try:
        from transformers import PrinterCallback

        trainer.remove_callback(PrinterCallback)
    except (ImportError, ValueError):
        pass
    with phase_progress("Training student LoRA adapter", 1) as (progress, task_id):
        trainer.add_callback(rich_trainer_progress_callback(progress, task_id))
        train_result = trainer.train()
    with phase_progress("Saving student adapter", 3) as (progress, task_id):
        trainer.save_model(training_config["output_dir"])
        progress.update(task_id, advance=1, description="Saving tokenizer")
        tokenizer.save_pretrained(training_config["output_dir"])
        progress.update(task_id, advance=1, description="Saving trainer state")
        trainer.save_state()
        progress.update(task_id, advance=1)
    metrics = getattr(train_result, "metrics", {}) or {}
    summary_rows = [
        ("Base model", config["model"]["base_model"]),
        ("Training examples", len(train_dataset)),
        ("Evaluation examples", len(eval_dataset)),
        ("Loss mode", loss_mode),
        ("Epochs", training_config.get("num_train_epochs", 3)),
        ("LoRA rank", lora_config.get("r", 16)),
        ("LoRA alpha", lora_config.get("alpha", 32)),
    ]
    for label, metric in (
        ("Training loss", "train_loss"),
        ("Training runtime", "train_runtime"),
        ("Samples/second", "train_samples_per_second"),
    ):
        if metric in metrics:
            value = metrics[metric]
            summary_rows.append((label, f"{value:.6f}" if isinstance(value, float) else value))
    if trainer.state.best_model_checkpoint is not None:
        summary_rows.append(("Best checkpoint", trainer.state.best_model_checkpoint))
    if trainer.state.best_metric is not None:
        summary_rows.append(("Best eval metric", f"{trainer.state.best_metric:.6f}"))
    metadata = read_metadata(config["model_metadata_path"])
    metadata.update(
        {
            "updated_at": iso_now(),
            "status": "complete",
            "resolved_base_model_revision": getattr(model.config, "_commit_hash", None),
            "training_result": {
                "metrics": metrics,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
            },
            "artifacts": {
                name: file_fingerprint(
                    str(Path(config["model_dir"]) / name), relative_to=PROJECT_ROOT
                )
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                    "tokenizer_config.json",
                    "trainer_state.json",
                )
                if (Path(config["model_dir"]) / name).is_file()
            },
        }
    )
    atomic_write_yaml(metadata, config["model_metadata_path"])
    summary_rows.append(("Dataset version", data_config["dataset_version"]))
    summary_rows.append(("Model version", config["model_version"]))
    summary_rows.append(("Adapter output", training_config["output_dir"]))
    print_summary("Student LoRA training complete", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a 3DSG Cypher tool-using agent model."
    )
    parser.add_argument(
        "--student-config",
        default="configs/students/student_qwen3.5_4b.yaml",
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--include-reasoning", action=argparse.BooleanOptionalAction, default=None,
                        help="Override training.include_reasoning for a matched ablation.")
    parser.add_argument(
        "--model-version",
        default=None,
        help="Optional new model version in YYYY-MM-DD-T-HH-MM-SS-Z format.",
    )
    args = parser.parse_args()
    command(args)


if __name__ == "__main__":
    main()
