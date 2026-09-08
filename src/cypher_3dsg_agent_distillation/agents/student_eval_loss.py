from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

from cypher_3dsg_agent_distillation.agents.student_model import (
    load_student_model_and_tokenizer,
    model_input_device,
    require_transformers_dependencies,
)

DEFAULT_ENTROPY_CHUNK_SIZE = 32


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


def eval_loss_progress_context(show_progress: bool):
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


def split_display_name(split: str) -> str:
    return split.replace("_", " ").replace("-", " ").title()


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
    ) = require_transformers_dependencies()
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
            active_task_id = active_progress.add_task("eval-loss", total=1, visible=False)
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
                # Avoid Transformers' full-logit float32 cast when labels are supplied.
                labels = batch.pop("labels")
                outputs = model(**batch, use_cache=False)
                metrics = shifted_token_metrics(outputs, labels)
                total_nll += metrics["nll_sum"]
                total_entropy += metrics["entropy_sum"]
                total_correct_tokens += metrics["correct_tokens"]
                total_tokens += metrics["tokens"]
                del outputs, labels, batch
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

    def weighted(metric: str) -> float:
        return sum(
            float(summary[metric]) * int(summary["tokens"])
            for summary in split_summaries.values()
        ) / total_tokens

    eval_loss = weighted("eval_loss")
    return {
        "examples": total_examples,
        "tokens": total_tokens,
        "eval_loss": eval_loss,
        "perplexity": finite_perplexity(eval_loss),
        "eval_entropy": weighted("eval_entropy"),
        "eval_mean_token_accuracy": weighted("eval_mean_token_accuracy"),
        "splits": split_summaries,
    }
