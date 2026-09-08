from __future__ import annotations

from pathlib import Path
from typing import Any

from cypher_3dsg_agent_distillation.core.cli_output import phase_progress
from cypher_3dsg_agent_distillation.core.model_runtime_output import (
    configure_model_library_output,
)


def require_transformers_dependencies():
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
            "before running Transformers student evaluation."
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
    ) = require_transformers_dependencies()
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
