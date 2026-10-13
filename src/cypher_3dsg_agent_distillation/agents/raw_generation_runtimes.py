from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from cypher_3dsg_agent_distillation.agents.reasoning import (
    reasoning_text,
    split_generated_reasoning,
    student_messages,
    training_template,
)
from cypher_3dsg_agent_distillation.agents.student_model import (
    load_student_model_and_tokenizer,
    model_input_device,
    require_transformers_dependencies,
)

TRANSFORMERS_RUNTIME = "transformers"
OLLAMA_RUNTIME = "ollama"
RAW_GENERATION_RUNTIMES = (TRANSFORMERS_RUNTIME, OLLAMA_RUNTIME)
DEFAULT_OLLAMA_HOST = "http://ollama:11434"
DEFAULT_OLLAMA_TIMEOUT = 300.0
DEFAULT_OLLAMA_TEMPERATURE = 0.0
DEFAULT_OLLAMA_SEED = 42
DEFAULT_OLLAMA_CONTEXT_LENGTH = 16384

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
    cleaned, _reasoning, _status = split_generated_reasoning(cleaned)
    terminator = CHAT_TERMINATOR_RE.search(cleaned)
    if terminator:
        cleaned = cleaned[: terminator.start()]
    return cleaned.strip()


def generation_artifacts_present(text: str) -> bool:
    return bool(GENERATION_ARTIFACT_RE.search(text))


@dataclass(frozen=True)
class GenerationResult:
    text: str
    raw_text: str
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    max_new_tokens: int | None
    token_budget: int | None
    tool_calls: tuple[dict[str, Any], ...] = ()
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    reasoning_status: str = "absent"


class RawGenerationRuntime(Protocol):
    name: str

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool,
        tools: list[dict[str, Any]] | None,
        progress=None,
        progress_task_id=None,
        progress_description: str | None = None,
    ) -> GenerationResult: ...

    def generation_config(self, *, max_new_tokens: int | None) -> dict[str, Any]: ...

    def metadata(self) -> dict[str, Any]: ...

    def release_memory(self) -> None: ...


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

    messages = student_messages(
        normalize_chat_messages(messages), tokenizer,
        include_reasoning=bool(enable_thinking),
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools is not None:
            kwargs["tools"] = tools
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        template = training_template(tokenizer, include_reasoning=bool(enable_thinking))
        if template is not None:
            kwargs["chat_template"] = template
        try:
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            if "enable_thinking" not in kwargs:
                raise
            kwargs.pop("enable_thinking")
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
        if enable_thinking and "reasoning_content" not in str(tokenizer.chat_template):
            prompt += "<think>\n"
        return prompt

    prompt = "\n\n".join(message_to_plain_text(message) for message in messages)
    rendered = f"{prompt}\n\nassistant:" if prompt else "assistant:"
    return rendered + "\n<think>\n" if enable_thinking else rendered


def generation_length_kwargs(max_new_tokens: int | None) -> dict[str, Any]:
    if max_new_tokens is None:
        return {}
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero when provided.")
    return {"max_new_tokens": max_new_tokens}


def configured_generation_token_budget(model, max_new_tokens: int | None) -> int | None:
    if max_new_tokens is not None:
        return max_new_tokens
    config = getattr(model, "generation_config", None)
    configured = getattr(config, "max_new_tokens", None)
    return int(configured) if configured is not None else None


def generation_progress_stopping_criteria(*, input_length: int, progress, task_id):
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


class TransformersRawGenerationRuntime:
    name = TRANSFORMERS_RUNTIME

    def __init__(self, model, tokenizer, *, variant: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.variant = variant

    @classmethod
    def load(
        cls,
        *,
        student_config: dict[str, Any],
        adapter_dir: str,
        explicit_tokenizer: str | None,
        show_progress: bool,
        baseline: bool,
    ) -> TransformersRawGenerationRuntime:
        model, tokenizer = load_student_model_and_tokenizer(
            student_config=student_config,
            adapter_dir=adapter_dir,
            explicit_tokenizer=explicit_tokenizer,
            disable_cache=False,
            show_progress=show_progress,
            use_adapter=not baseline,
        )
        return cls(model, tokenizer, variant="baseline" if baseline else "fine-tuned")

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool,
        tools: list[dict[str, Any]] | None,
        progress=None,
        progress_task_id=None,
        progress_description: str | None = None,
    ) -> GenerationResult:
        (torch, *_unused) = require_transformers_dependencies()
        prompt = render_messages_for_generation(
            self.tokenizer,
            messages,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {
            key: value.to(model_input_device(self.model)) for key, value in encoded.items()
        }
        input_length = encoded["input_ids"].shape[-1]
        token_budget = configured_generation_token_budget(self.model, max_new_tokens)
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
            generated = self.model.generate(**encoded, **generate_kwargs)

        generated_ids = generated[0][input_length:]
        generated_tokens = int(generated_ids.shape[-1])
        if progress is not None and progress_task_id is not None:
            progress.update(progress_task_id, completed=generated_tokens)
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        action, reasoning, reasoning_status = split_generated_reasoning(
            raw_text, thinking_started=prompt.rstrip().endswith("<think>"),
        )
        result = GenerationResult(
            text=clean_generated_text(action),
            raw_text=raw_text,
            prompt=prompt,
            prompt_tokens=int(input_length),
            generated_tokens=generated_tokens,
            max_new_tokens=max_new_tokens,
            token_budget=token_budget,
            reasoning=reasoning,
            reasoning_status=reasoning_status,
        )
        del generated_ids, generated, encoded
        return result

    def generation_config(self, *, max_new_tokens: int | None) -> dict[str, Any]:
        config = getattr(self.model, "generation_config", None)
        report = {}
        if config is not None:
            for key in GENERATION_CONFIG_REPORT_KEYS:
                if hasattr(config, key):
                    report[key] = jsonable(getattr(config, key))
        overrides = generation_length_kwargs(max_new_tokens)
        if overrides:
            report["generation_call_kwargs"] = overrides
        return report

    def metadata(self) -> dict[str, Any]:
        return {"provider": self.name, "model_variant": self.variant}

    def release_memory(self) -> None:
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()


def ollama_host(value: str | None = None) -> str:
    return (value or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")


def ollama_messages(
    messages: list[dict[str, Any]], *, include_reasoning: bool = False,
) -> list[dict[str, Any]]:
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
        normalize_tool_call_arguments,
    )
    from cypher_3dsg_agent_distillation.training.train import normalize_chat_messages

    outbound = []
    for message in normalize_chat_messages(messages):
        payload = {
            "role": str(message.get("role") or "user"),
            "content": str(message.get("content") or ""),
        }
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            payload["tool_calls"] = [
                normalize_tool_call_arguments(tool_call) for tool_call in tool_calls
            ]
        if message.get("tool_name"):
            payload["tool_name"] = str(message["tool_name"])
        if message.get("tool_call_id"):
            payload["tool_call_id"] = str(message["tool_call_id"])
        if include_reasoning and message.get("role") == "assistant":
            thinking, _kind = reasoning_text(message)
            if thinking:
                payload["thinking"] = thinking
        outbound.append(payload)
    return outbound


class OllamaRawGenerationRuntime:
    name = OLLAMA_RUNTIME

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
        seed: int = DEFAULT_OLLAMA_SEED,
        context_length: int | None = None,
        keep_alive: str = "5m",
        opener=None,
    ) -> None:
        if not model.strip():
            raise ValueError("An Ollama model name is required.")
        if timeout <= 0:
            raise ValueError("Ollama timeout must be greater than zero.")
        if context_length is not None and context_length <= 0:
            raise ValueError("Ollama context length must be greater than zero.")
        self.model = model.strip()
        self.host = ollama_host(host)
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.context_length = context_length
        self.keep_alive = keep_alive
        self._opener = opener or urlopen
        self._server_version: str | None = None
        self._model_details: dict[str, Any] = {}

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            urljoin(f"{self.host}/", path.lstrip("/")),
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request {path} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}: {exc.reason}"
            ) from exc
        try:
            result = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON for {path}.") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Ollama returned a non-object response for {path}.")
        if result.get("error"):
            raise RuntimeError(f"Ollama request {path} failed: {result['error']}")
        return result

    def cache_identity(self) -> dict[str, Any]:
        """Identify the registered model even when its mutable tag is reused."""
        name = self.model if ":" in self.model else f"{self.model}:latest"
        for model in self._request("/api/tags").get("models", []):
            if name in {model.get("name"), model.get("model")} and model.get("digest"):
                return {
                    "model": name,
                    "digest": model["digest"],
                    "server_version": self._request("/api/version").get("version"),
                }
        raise RuntimeError(f"No registered Ollama model digest found for {self.model}.")

    def validate(self) -> None:
        version = self._request("/api/version")
        self._server_version = str(version.get("version") or "unknown")
        shown = self._request("/api/show", {"model": self.model})
        details = shown.get("details")
        self._model_details = jsonable(details) if isinstance(details, dict) else {}

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int | None,
        enable_thinking: bool,
        tools: list[dict[str, Any]] | None,
        progress=None,
        progress_task_id=None,
        progress_description: str | None = None,
    ) -> GenerationResult:
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero when provided.")
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if max_new_tokens is not None:
            options["num_predict"] = max_new_tokens
        if self.context_length is not None:
            options["num_ctx"] = self.context_length
        outbound_messages = ollama_messages(messages, include_reasoning=enable_thinking)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": outbound_messages,
            "stream": False,
            "think": bool(enable_thinking),
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if tools is not None:
            payload["tools"] = tools
        if progress is not None and progress_task_id is not None:
            progress_kwargs = {
                "completed": 0,
                "visible": True,
                "description": progress_description or "generation",
            }
            if max_new_tokens is not None:
                progress_kwargs["total"] = max_new_tokens
            progress.update(progress_task_id, **progress_kwargs)

        response = self._request("/api/chat", payload)
        message = response.get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError("Ollama chat response did not contain a message object.")
        content = str(message.get("content") or "")
        action, inline_reasoning, reasoning_status = split_generated_reasoning(content)
        reasoning = str(message.get("thinking") or inline_reasoning)
        if message.get("thinking"):
            reasoning_status = "complete" if action or message.get("tool_calls") else "incomplete"
        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise RuntimeError("Ollama chat response contained invalid tool_calls.")
        tool_calls = tuple(
            jsonable(tool_call) for tool_call in raw_tool_calls if isinstance(tool_call, dict)
        )
        generated_tokens = int(response.get("eval_count") or 0)
        if progress is not None and progress_task_id is not None:
            progress.update(progress_task_id, completed=generated_tokens)
        raw_text = content.strip()
        if message.get("thinking"):
            raw_text = f"<think>\n{reasoning}\n</think>\n\n{raw_text}"
        if tool_calls:
            structured_text = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
            raw_text = f"{raw_text}\n{structured_text}".strip()
        return GenerationResult(
            text=clean_generated_text(action),
            raw_text=raw_text,
            prompt=json.dumps(
                {"messages": outbound_messages, "tools": tools or []},
                ensure_ascii=False,
            ),
            prompt_tokens=int(response.get("prompt_eval_count") or 0),
            generated_tokens=generated_tokens,
            max_new_tokens=max_new_tokens,
            token_budget=max_new_tokens,
            tool_calls=tool_calls,
            reasoning=reasoning,
            reasoning_status=reasoning_status,
            provider_metadata={
                key: response.get(key)
                for key in (
                    "done_reason",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                )
                if response.get(key) is not None
            },
        )

    def generation_config(self, *, max_new_tokens: int | None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if max_new_tokens is not None:
            options["num_predict"] = max_new_tokens
        if self.context_length is not None:
            options["num_ctx"] = self.context_length
        return {
            "provider": self.name,
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "host": self.host,
            "server_version": self._server_version,
            "model_details": self._model_details,
        }

    def release_memory(self) -> None:
        return None
