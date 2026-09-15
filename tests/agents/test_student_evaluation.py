import json
from contextlib import nullcontext

import pytest

import cypher_3dsg_agent_distillation.agents.student_eval_loss as eval_loss_module
import cypher_3dsg_agent_distillation.agents.student_evaluation as student_evaluation
import cypher_3dsg_agent_distillation.agents.student_model as student_model
import cypher_3dsg_agent_distillation.training.train as train_module
from cypher_3dsg_agent_distillation.agents.agent_prompts import run_cypher_tool_schema
from cypher_3dsg_agent_distillation.agents.raw_generation_runtimes import (
    GenerationResult,
    clean_generated_text,
    generation_length_kwargs,
    render_messages_for_generation,
)
from cypher_3dsg_agent_distillation.agents.student_eval_loss import (
    combine_eval_loss_summaries,
    evaluate_eval_loss_splits,
    evaluate_eval_loss_with_loaded_model,
    finite_perplexity,
)
from cypher_3dsg_agent_distillation.agents.student_evaluation import (
    ALL_SPLITS,
    DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS,
    DEFAULT_E2E_MAX_TOOL_ROWS,
    END_TO_END_CHECK,
    EVAL_LOSS_MODE,
    FINAL_ANSWER_CHECK,
    TOOL_CALL_CHECK,
    configure_neo4j_notification_output,
    default_adapter_dir,
    evaluate_end_to_end_record,
    evaluate_raw_generation,
    final_answer_prompt_messages,
    generation_debug_report,
    initial_prompt_messages,
    parse_generated_tool_calls,
    raw_generation_progress_description,
    report_output_path,
    score_final_answer_generation,
    score_tool_call_generation,
    summarize_raw_generation_records,
    tokenizer_source,
)
from cypher_3dsg_agent_distillation.agents.student_model import (
    load_student_model_and_tokenizer,
)
from cypher_3dsg_agent_distillation.core.io import write_jsonl


def raw_generation_record():
    cypher = (
        "MATCH (p:Place {nodeSymbol: 'p1'})-[:CONTAINS]->(o:Object) "
        "RETURN o.nodeSymbol AS object"
    )
    return {
        "uid": "trace-one",
        "question": "What objects are directly contained in place p1?",
        "task_type": "objects_in_place_symbol",
        "gold_cypher": cypher,
        "expected_solution": "<O1>",
        "solution_formats": ["set[str]"],
        "read_only": True,
        "metadata": {"output_type": "SLDP"},
        "messages": [
            {"role": "user", "content": "Use Cypher to answer."},
            {"role": "user", "content": "What objects are directly contained in place p1?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_cypher_query",
                            "arguments": {"cypher_string": cypher},
                        },
                    }
                ],
                "reasoning": "Need direct containment.",
            },
            {
                "role": "tool",
                "tool_name": "run_cypher_query",
                "content": '{"ok": true, "rows": [{"object": "O1"}]}',
            },
            {"role": "assistant", "content": "<answer><O1></answer>"},
        ],
    }


def pddl_generation_record():
    record = raw_generation_record()
    record.update(
        {
            "task_family": "cypher_pddl",
            "output_type": "PDDL",
            "expected_solution": "(visited-object O1)",
            "grounding": {
                "expected": {"object": ["O1"]},
                "fields": {
                    "object": {
                        "aliases": ["object"],
                        "symbol_type": "Object",
                        "cardinality": "set",
                    }
                },
                "match": "exact",
            },
            "metadata": {"task_family": "cypher_pddl", "output_type": "PDDL"},
        }
    )
    return record


class FakeValidator:
    def validate(self, cypher, execute):
        return {
            "syntax_error": False,
            "timeout": False,
            "executable": True,
            "elapsed_ms": 1.0,
            "rows": [{"object": "O1"}],
            "error": None,
        }


def generated_result(text):
    return GenerationResult(
        text=text,
        raw_text=text,
        prompt="bounded prompt",
        prompt_tokens=10,
        generated_tokens=5,
        max_new_tokens=32,
        token_budget=32,
    )


def generated_tool_call(cypher):
    return (
        "<tool_call>\n"
        "<function=run_cypher_query>\n"
        "<parameter=cypher_string>\n"
        f"{cypher}\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def generated_native_tool_call(cypher):
    return GenerationResult(
        text="",
        raw_text='{"tool_calls": [...] }',
        prompt="native Ollama request",
        prompt_tokens=10,
        generated_tokens=5,
        max_new_tokens=32,
        token_budget=32,
        tool_calls=(
            {
                "function": {
                    "name": "run_cypher_query",
                    "arguments": {"cypher_string": cypher},
                }
            },
        ),
    )


class FakeTensor:
    def to(self, _device):
        return self


class FakeLoss:
    def item(self):
        return 0.25


class FakeOutputs:
    loss = FakeLoss()
    logits = None


class FakeModel:
    def __init__(self):
        self.calls = []

    def parameters(self):
        return iter(())

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOutputs()


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeDataLoader:
    def __init__(self, dataset, batch_size, collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _record in self.dataset:
            yield {"labels": FakeTensor()}


class FakeProgress:
    def __init__(self):
        self.updates = []

    def add_task(self, *_args, **_kwargs):
        return 7

    def update(self, task_id, **kwargs):
        self.updates.append((task_id, kwargs))


class RecordingProgressContext:
    def __init__(self, events):
        self.events = events
        self.progress = FakeProgress()

    def __enter__(self):
        self.events.append("progress_enter")
        return self.progress

    def __exit__(self, *_args):
        self.events.append("progress_exit")


class FakeGenerationRuntime:
    def __init__(self, generate):
        self._generate = generate

    def generate(self, messages, **kwargs):
        return self._generate(None, None, messages, **kwargs)


class MinimalRawGenerationRuntime:
    def generate(self, _messages, **_kwargs):
        return generated_result("<answer><O1></answer>")

    def release_memory(self):
        pass

    def generation_config(self, **_kwargs):
        return {}

    def metadata(self):
        return {}


def test_raw_generation_groups_records_and_reloads_each_scene_once(tmp_path):
    split_path = tmp_path / "test.jsonl"
    records = []
    for index, scene_id in enumerate(("scene-1", "scene-2", "scene-1", "scene-2")):
        record = raw_generation_record()
        record.update({"uid": f"trace-{index}", "scene_id": scene_id})
        records.append(record)
    write_jsonl(records, split_path)
    loaded_scenes = []

    result = evaluate_raw_generation(
        data_config={
            "splits": {"test": str(split_path)},
            "scenes": [
                {"scene_id": "scene-1", "scene_path": "one.json"},
                {"scene_id": "scene-2", "scene_path": "two.json"},
            ],
        },
        split="test",
        runtime=MinimalRawGenerationRuntime(),
        max_new_tokens=32,
        limit=None,
        validator=object(),
        tool_call_skip_reason="tool validation skipped for test",
        policy=None,
        show_progress=False,
        enable_thinking=False,
        scene_loader=lambda scene: loaded_scenes.append(scene["scene_id"]),
    )

    assert loaded_scenes == ["scene-1", "scene-2"]
    assert result["scene_database_reloads"] == 2


def test_default_adapter_dir_uses_training_output_dir():
    assert default_adapter_dir({"training": {"output_dir": "outputs/adapter"}}) == (
        "outputs/adapter"
    )


def test_tokenizer_source_prefers_explicit_tokenizer():
    assert (
        tokenizer_source(
            adapter_dir="missing-adapter",
            explicit_tokenizer="custom-tokenizer",
            model_config={"base_model": "base-model"},
        )
        == "custom-tokenizer"
    )


def test_tokenizer_source_baseline_ignores_existing_adapter(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    assert tokenizer_source(
        adapter_dir=str(adapter_dir),
        explicit_tokenizer=None,
        model_config={"base_model": "base-model", "tokenizer": "base-tokenizer"},
        use_adapter=False,
    ) == "base-tokenizer"


def test_report_output_path_uses_split_and_mode_name():
    output = report_output_path(
        data_config={"test_report_path": "data/datasets/scene/reports/test.agent.report.json"},
        split="challenge",
        mode=EVAL_LOSS_MODE,
        output=None,
    )

    assert output.name == "challenge.student.eval-loss.report.json"


def test_report_output_path_keeps_baseline_separate_from_fine_tuned_results():
    output = report_output_path(
        data_config={"test_report_path": "data/datasets/scene/reports/test.agent.report.json"},
        split="test",
        mode="raw-generation",
        output=None,
        baseline=True,
    )

    assert output.name == "test.baseline.raw-generation.report.json"


def test_report_output_path_keeps_ollama_models_separate():
    output = report_output_path(
        data_config={"test_report_path": "data/scene/reports/test.agent.report.json"},
        split="challenge",
        mode="raw-generation",
        output=None,
        runtime="ollama",
        runtime_model="qwen3.5-4b-finetune:q4_k_m",
    )

    assert output.name == (
        "challenge.student.ollama.qwen3.5-4b-finetune-q4_k_m."
        "raw-generation.report.json"
    )


def test_report_output_path_uses_aggregate_name_for_all_splits():
    output = report_output_path(
        data_config={"test_report_path": "data/datasets/scene/reports/test.agent.report.json"},
        split=ALL_SPLITS,
        mode=EVAL_LOSS_MODE,
        output=None,
    )

    assert output.name == "student.eval-loss.report.json"


def test_report_output_path_uses_selected_model_version_directory(tmp_path):
    reports = (
        tmp_path
        / "reports"
        / "models"
        / "qwen3.5-4b-finetune"
        / "2026-09-08-T-15-05-21-Z"
    )

    output = report_output_path(
        data_config={"test_report_path": str(tmp_path / "reports" / "dataset" / "test.json")},
        split="test",
        mode="raw-generation",
        output=None,
        reports_directory=reports,
    )

    assert output.parent == reports
    assert output.name == "test.student.raw-generation.report.json"


def test_load_baseline_model_does_not_load_lora_adapter(monkeypatch):
    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2

    class FakeAutoTokenizer:
        sources = []

        @classmethod
        def from_pretrained(cls, source, **_kwargs):
            cls.sources.append(source)
            return FakeTokenizer()

    class FakeBaseModel:
        config = type("Config", (), {})()
        generation_config = type("GenerationConfig", (), {})()

        def eval(self):
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _source, **_kwargs):
            return FakeBaseModel()

    class FakePeftModel:
        calls = []

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            cls.calls.append((_args, _kwargs))
            raise AssertionError("The baseline must not load a LoRA adapter.")

    monkeypatch.setattr(student_model, "configure_model_library_output", lambda: None)
    monkeypatch.setattr(
        student_model,
        "require_transformers_dependencies",
        lambda: (
            None,
            FakePeftModel,
            None,
            FakeAutoModel,
            FakeAutoTokenizer,
            None,
        ),
    )
    monkeypatch.setattr(train_module, "dtype_from_name", lambda _name: "auto")

    model, _tokenizer = load_student_model_and_tokenizer(
        student_config={
            "model": {
                "base_model": "base-model",
                "tokenizer": "base-tokenizer",
            }
        },
        adapter_dir="outputs/adapter",
        explicit_tokenizer=None,
        show_progress=False,
        use_adapter=False,
    )

    assert isinstance(model, FakeBaseModel)
    assert FakeAutoTokenizer.sources == ["base-tokenizer"]
    assert FakePeftModel.calls == []


def test_combine_eval_loss_summaries_weights_by_token_count():
    summary = combine_eval_loss_summaries(
        {
            "train": {
                "examples": 2,
                "tokens": 10,
                "eval_loss": 0.2,
                "perplexity": 1.2,
                "eval_entropy": 0.4,
                "eval_mean_token_accuracy": 0.9,
            },
            "eval": {
                "examples": 1,
                "tokens": 30,
                "eval_loss": 0.6,
                "perplexity": 1.8,
                "eval_entropy": 0.8,
                "eval_mean_token_accuracy": 0.7,
            },
        }
    )

    assert summary["examples"] == 3
    assert summary["tokens"] == 40
    assert summary["eval_loss"] == pytest.approx(0.5)
    assert summary["eval_entropy"] == pytest.approx(0.7)
    assert summary["eval_mean_token_accuracy"] == pytest.approx(0.75)
    assert set(summary["splits"]) == {"train", "eval"}


def test_evaluate_eval_loss_updates_progress_per_batch(monkeypatch):
    monkeypatch.setattr(
        eval_loss_module,
        "require_transformers_dependencies",
        lambda: (FakeTorch, None, FakeDataLoader, None, None, None),
    )
    monkeypatch.setattr(train_module, "CausalLmDataCollator", lambda _tokenizer: None)
    monkeypatch.setattr(
        train_module,
        "build_training_dataset",
        lambda *_args, **_kwargs: [{"uid": "one"}, {"uid": "two"}, {"uid": "three"}],
    )
    monkeypatch.setattr(
        eval_loss_module,
        "shifted_token_metrics",
        lambda _outputs, _labels: {
            "tokens": 5,
            "nll_sum": 1.25,
            "entropy_sum": 1.0,
            "correct_tokens": 4,
        },
    )
    progress = FakeProgress()
    model = FakeModel()

    summary = evaluate_eval_loss_with_loaded_model(
        data_config={"splits": {"train": "train.jsonl"}},
        split="train",
        model=model,
        tokenizer=object(),
        batch_size=1,
        max_length=128,
        loss_mode="assistant-only",
        progress=progress,
        progress_task_id=7,
    )

    assert progress.updates[0] == (
        7,
        {
            "total": 3,
            "completed": 0,
            "visible": True,
            "description": "Evaluating loss · Train",
        },
    )
    assert [update for _task, update in progress.updates[1:]] == [
        {"advance": 1},
        {"advance": 1},
        {"advance": 1},
    ]
    assert summary["examples"] == 3
    assert summary["tokens"] == 15
    assert summary["eval_loss"] == pytest.approx(0.25)
    assert summary["eval_mean_token_accuracy"] == pytest.approx(0.8)
    assert all("labels" not in call for call in model.calls)
    assert all(call["use_cache"] is False for call in model.calls)


def test_eval_loss_progress_starts_after_model_and_dataset_setup(monkeypatch):
    events = []

    monkeypatch.setattr(
        eval_loss_module,
        "load_student_model_and_tokenizer",
        lambda **_kwargs: (events.append("load_model") or FakeModel(), object()),
    )
    monkeypatch.setattr(
        eval_loss_module,
        "eval_loss_progress_context",
        lambda _show_progress: RecordingProgressContext(events),
    )
    monkeypatch.setattr(
        eval_loss_module,
        "require_transformers_dependencies",
        lambda: (FakeTorch, None, FakeDataLoader, None, None, None),
    )
    monkeypatch.setattr(train_module, "CausalLmDataCollator", lambda _tokenizer: None)

    def fake_build_training_dataset(*_args, **_kwargs):
        events.append("build_dataset")
        return [{"uid": "one"}]

    monkeypatch.setattr(train_module, "build_training_dataset", fake_build_training_dataset)
    monkeypatch.setattr(
        eval_loss_module,
        "shifted_token_metrics",
        lambda _outputs, _labels: {
            "tokens": 5,
            "nll_sum": 1.25,
            "entropy_sum": 1.0,
            "correct_tokens": 4,
        },
    )

    evaluate_eval_loss_splits(
        data_config={"splits": {"train": "train.jsonl"}},
        student_config={"model": {}, "training": {}},
        splits=("train",),
        adapter_dir="adapter",
        explicit_tokenizer=None,
        batch_size=1,
        max_length=128,
        loss_mode="assistant-only",
        show_progress=True,
    )

    assert events[:3] == ["load_model", "build_dataset", "progress_enter"]


def test_generation_length_kwargs_uses_max_new_tokens():
    assert generation_length_kwargs(256) == {"max_new_tokens": 256}


def test_raw_generation_progress_description_is_stable_and_concise():
    assert raw_generation_progress_description(
        split="challenge",
        phase="tool-call tokens",
    ) == "Generating tool call · Challenge"


def test_clean_generated_text_removes_qwen_thinking_and_turn_artifacts():
    assert clean_generated_text(
        "Need to answer with tags.\n</think>\n\n<answer>O29</answer><|im_end|>"
    ) == "<answer>O29</answer>"


def test_render_messages_for_generation_disables_thinking_by_default():
    class ChatTemplateTokenizer:
        chat_template = "template"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "rendered"

    tokenizer = ChatTemplateTokenizer()

    assert render_messages_for_generation(
        tokenizer,
        [{"role": "user", "content": "question"}],
    ) == "rendered"
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert tokenizer.kwargs["enable_thinking"] is False


def test_render_messages_for_generation_passes_native_tool_schema():
    class ChatTemplateTokenizer:
        chat_template = "template"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "rendered"

    tokenizer = ChatTemplateTokenizer()
    tools = [run_cypher_tool_schema()]

    assert render_messages_for_generation(
        tokenizer,
        [{"role": "user", "content": "question"}],
        tools=tools,
    ) == "rendered"
    assert tokenizer.kwargs["tools"] == tools
    assert tokenizer.kwargs["tools"][0]["function"]["name"] == "run_cypher_query"
    parameters = tokenizer.kwargs["tools"][0]["function"]["parameters"]
    assert parameters["required"] == ["cypher_string"]


def test_generation_debug_report_captures_exact_prompt_and_completion_only():
    generation = GenerationResult(
        text=(
            '<tool_call>\n{"name":"run_cypher_query",'
            '"arguments":{"cypher_string":"MATCH (o:Object) RETURN o"}}\n'
            "</tool_call>"
        ),
        raw_text=(
            '<tool_call>\n{"name":"run_cypher_query",'
            '"arguments":{"cypher_string":"MATCH (o:Object) RETURN o"}}\n'
            "</tool_call><|im_end|>"
        ),
        prompt="<|im_start|>system\n# Tools\nrun_cypher_query<|im_end|>",
        prompt_tokens=20,
        generated_tokens=10,
        max_new_tokens=64,
        token_budget=64,
    )

    report = generation_debug_report(generation)

    assert report["rendered_prompt"] == generation.prompt
    assert report["raw_completion"] == generation.raw_text
    assert set(report) == {
        "rendered_prompt",
        "raw_completion",
        "structured_tool_calls",
        "provider_metadata",
    }


def test_render_messages_for_generation_falls_back_without_thinking_kwarg():
    class StrictChatTemplateTokenizer:
        chat_template = "template"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            return "rendered"

    assert render_messages_for_generation(
        StrictChatTemplateTokenizer(),
        [{"role": "user", "content": "question"}],
    ) == "rendered"


def test_raw_generation_prompt_slices_do_not_leak_gold_targets():
    record = raw_generation_record()

    initial_messages = initial_prompt_messages(record)
    final_answer_messages = final_answer_prompt_messages(record)

    assert [message["role"] for message in initial_messages] == ["user", "user"]
    assert all("tool_calls" not in message for message in initial_messages)
    assert all("<answer>" not in str(message.get("content")) for message in final_answer_messages)
    assert final_answer_messages[-1]["role"] == "tool"
    assert final_answer_messages[-2]["tool_calls"]


def test_raw_generation_repair_prompts_include_failed_tool_context():
    record = raw_generation_record()
    invalid_call = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "run_cypher_query",
                    "arguments": {"cypher_string": "MATC (n) RETURN n"},
                }
            }
        ],
    }
    error_result = {"role": "tool", "content": "Syntax error near MATC"}
    record["messages"] = [
        *record["messages"][:2],
        invalid_call,
        error_result,
        *record["messages"][2:],
    ]
    record["metadata"] = {
        "controlled_repair": True,
        "repair_context_message_count": 4,
    }

    initial_messages = initial_prompt_messages(record)
    final_messages = final_answer_prompt_messages(record)

    assert len(initial_messages) == 4
    assert initial_messages[-2]["tool_calls"] == invalid_call["tool_calls"]
    assert initial_messages[-1]["content"] == error_result["content"]
    assert len(final_messages) == len(record["messages"]) - 1
    assert final_messages[-1]["role"] == "tool"


def test_parse_generated_tool_calls_accepts_qwen_tool_call_tags():
    parsed = parse_generated_tool_calls(
        "<think>\n</think>\n"
        "<tool_call>\n"
        "<function=run_cypher_query>\n"
        "<parameter=cypher_string>\n"
        "MATCH (n) RETURN n\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    assert parsed.tool_call_generated is True
    assert parsed.error is None
    assert parsed.tool_calls[0]["function"]["name"] == "run_cypher_query"
    assert parsed.tool_calls[0]["function"]["arguments"] == {
        "cypher_string": "MATCH (n) RETURN n"
    }


def test_parse_generated_tool_calls_reports_malformed_marker():
    parsed = parse_generated_tool_calls("<tool_call><function=run_cypher_query>")

    assert parsed.tool_call_generated is True
    assert parsed.tool_calls == []
    assert parsed.error


def test_score_tool_call_generation_accepts_ollama_native_tool_call():
    record = raw_generation_record()
    generation = generated_native_tool_call(record["gold_cypher"])

    result = score_tool_call_generation(
        record,
        generation.text,
        validator=FakeValidator(),
        policy=None,
        structured_tool_calls=generation.tool_calls,
    )

    assert result["status"] == "pass"
    assert result["tool_call_parse_success"] is True
    assert result["tool_result_match"] is True


def test_score_final_answer_generation_matches_expected_solution():
    result = score_final_answer_generation(
        raw_generation_record(),
        "<answer><O1></answer>",
    )

    assert result["status"] == "pass"
    assert result["final_answer_has_tags"] is True
    assert result["final_answer_match"] is True


def test_score_tool_call_generation_executes_and_matches_expected_solution():
    record = raw_generation_record()
    result = score_tool_call_generation(
        record,
        (
            "<tool_call>\n"
            "<function=run_cypher_query>\n"
            "<parameter=cypher_string>\n"
            f"{record['gold_cypher']}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
        validator=FakeValidator(),
        policy=None,
    )

    assert result["status"] == "pass"
    assert result["tool_call_parse_success"] is True
    assert result["cypher_exact_match"] is True
    assert result["static_valid"] is True
    assert result["executable"] is True
    assert result["cypher_solution_match"] is True


def test_score_tool_call_generation_marks_explain_rejection_static_invalid():
    record = raw_generation_record()

    class ExplainRejectingValidator:
        def explain(self, _cypher):
            return {
                "syntax_error": True,
                "executable": False,
                "error": "Neo.ClientError.Statement.SyntaxError: Invalid input '<>'",
            }

        def validate(self, *_args, **_kwargs):
            raise AssertionError("EXPLAIN-invalid Cypher must not be executed")

    result = score_tool_call_generation(
        record,
        generated_tool_call(record["gold_cypher"]),
        validator=ExplainRejectingValidator(),
        policy=None,
    )

    assert result["status"] == "static_error"
    assert result["static_valid"] is False
    assert result["syntax_error"] is True
    assert result["explain_validation"]["executable"] is False
    assert "Neo4j EXPLAIN rejected query" in result["static_issues"][0]


def test_score_tool_call_generation_flags_empty_result_hidden_false_positive_risk():
    record = raw_generation_record()
    record["expected_solution"] = "<>"
    generated_cypher = (
        "MATCH (p:Place {nodeSymbol: 'p1'})-[:CONTAINS*2..2]->(o:Object) "
        "RETURN o.nodeSymbol AS object"
    )

    class EmptyValidator:
        def validate(self, cypher, execute):
            return {
                "syntax_error": False,
                "timeout": False,
                "executable": True,
                "elapsed_ms": 1.0,
                "rows": [],
                "error": None,
            }

    result = score_tool_call_generation(
        record,
        (
            "<tool_call>\n"
            "<function=run_cypher_query>\n"
            "<parameter=cypher_string>\n"
            f"{generated_cypher}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
        validator=EmptyValidator(),
        policy=None,
    )

    assert result["status"] == "pass"
    assert result["cypher_solution_match"] is True
    assert result["result_match_without_exact_cypher_match"] is True
    assert result["potential_hidden_false_positive"] is True
    assert result["hidden_false_positive_reasons"] == [
        "non_exact_cypher_matched_empty_result",
        "relationship_pattern_mismatch",
    ]


def test_non_exact_result_match_without_structural_change_is_reviewable_not_high_risk():
    record = raw_generation_record()
    generated_cypher = f"{record['gold_cypher']} ORDER BY object"

    result = score_tool_call_generation(
        record,
        (
            "<tool_call>\n"
            "<function=run_cypher_query>\n"
            "<parameter=cypher_string>\n"
            f"{generated_cypher}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
        validator=FakeValidator(),
        policy=None,
    )

    assert result["status"] == "pass"
    assert result["result_match_without_exact_cypher_match"] is True
    assert result["potential_hidden_false_positive"] is False
    assert result["hidden_false_positive_reasons"] == []


def test_score_pddl_tool_call_generation_compares_grounding_not_goal_text():
    record = pddl_generation_record()
    result = score_tool_call_generation(
        record,
        (
            "<tool_call>\n"
            "<function=run_cypher_query>\n"
            "<parameter=cypher_string>\n"
            f"{record['gold_cypher']}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
        validator=FakeValidator(),
        policy=None,
    )

    assert result["status"] == "pass"
    assert result["cypher_solution_match"] is True
    assert result["tool_result_match"] is True
    assert result["grounding_match"] is True
    assert result["expected_grounding"] == record["grounding"]
    assert result["actual_grounding"] == {"object": ["O1"]}


def test_pddl_invalid_input_empty_answer_passes_tool_and_final_answer_scoring():
    record = pddl_generation_record()
    record.update(
        {
            "expected_solution": "",
            "tags": ["invalid_input"],
            "grounding": {
                "expected": {"object": []},
                "fields": {
                    "object": {
                        "aliases": ["object"],
                        "symbol_type": "Object",
                        "cardinality": "set",
                    }
                },
                "match": "exact",
            },
        }
    )

    class EmptyValidator:
        def validate(self, cypher, execute):
            return {
                "syntax_error": False,
                "timeout": False,
                "executable": True,
                "elapsed_ms": 1.0,
                "rows": [],
                "error": None,
            }

    tool_result = score_tool_call_generation(
        record,
        (
            "<tool_call>\n"
            "<function=run_cypher_query>\n"
            "<parameter=cypher_string>\n"
            f"{record['gold_cypher']}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
        validator=EmptyValidator(),
        policy=None,
    )
    final_result = score_final_answer_generation(
        record,
        "<answer></answer>",
    )
    report_record = student_evaluation.raw_generation_report_record(
        record,
        final_answer_result=final_result,
        tool_call_result=tool_result,
        final_generation_debug={},
        tool_generation_debug={},
        end_to_end_generation_debug={
            "turns": [
                {
                    "turn": 1,
                    "rendered_prompt": "end-to-end prompt",
                    "raw_completion": "<answer>(visited-object O1)</answer>",
                }
            ]
        },
    )

    assert tool_result["status"] == "pass"
    assert tool_result["cypher_solution"] == ""
    assert tool_result["grounding_match"] is True
    assert final_result["status"] == "pass"
    assert final_result["final_answer_domain_valid"] is True
    assert report_record["expected"]["empty_answer"] is True
    assert report_record["expected"]["pddl_goal"] is None
    assert report_record["generation_debug"]["end_to_end_generation"]["turns"][0][
        "rendered_prompt"
    ] == "end-to-end prompt"


def test_raw_generation_report_record_exposes_pddl_goal_and_grounding():
    record = pddl_generation_record()
    report_record = student_evaluation.raw_generation_report_record(
        record,
        final_answer_result={
            "generated_text": "<answer>(visited-object O1)</answer>",
            "generated_answer": "(visited-object O1)",
            "final_answer_domain_valid": True,
            "final_answer_match": True,
        },
        tool_call_result={
            "generated_cypher": record["gold_cypher"],
            "cypher_solution": '{"object": ["O1"]}',
            "actual_grounding": {"object": ["O1"]},
            "grounding_match": True,
            "tool_result_match": True,
            "validation": {"rows": [{"object": "O1"}]},
        },
        final_generation_debug={},
        tool_generation_debug={},
    )

    assert report_record["task_family"] == "cypher_pddl"
    assert report_record["output_type"] == "PDDL"
    assert report_record["expected"]["pddl_goal"] == "(visited-object O1)"
    assert report_record["expected"]["grounding"] == record["grounding"]
    assert report_record["generated"]["actual_grounding"] == {"object": ["O1"]}
    # Retained for consumers of the previous raw-report schema.
    assert report_record["expected"]["cypher_solution"] == "(visited-object O1)"


def test_score_pddl_final_answer_uses_pddl_comparator():
    result = score_final_answer_generation(
        pddl_generation_record(),
        "<answer>(visited-object O1)</answer>",
    )

    assert result["status"] == "pass"
    assert result["final_answer_domain_valid"] is True


def test_score_tool_call_generation_preserves_expected_set_syntax_without_format_field():
    record = raw_generation_record()
    record.update(
        {
            "task_type": "top_object_classes_by_count",
            "gold_cypher": (
                "MATCH (o:Object) RETURN o.class AS class, count(o) AS count "
                "ORDER BY count DESC, class ASC LIMIT 2"
            ),
            "expected_solution": "<light, cabinet>",
        }
    )
    record.pop("solution_formats")

    class TopClassesValidator:
        def validate(self, cypher, execute):
            return {
                "syntax_error": False,
                "timeout": False,
                "executable": True,
                "elapsed_ms": 1.0,
                "rows": [
                    {"class": "light", "count": 57},
                    {"class": "cabinet", "count": 28},
                ],
                "error": None,
            }

    result = score_tool_call_generation(
        record,
        (
            "<tool_call>\n"
            "<function=run_cypher_query>\n"
            "<parameter=cypher_string>\n"
            f"{record['gold_cypher']}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
        validator=TopClassesValidator(),
        policy=None,
    )

    assert result["status"] == "pass"
    assert result["cypher_solution"] == "<light, cabinet>"
    assert result["cypher_solution_format_valid"] is True
    assert result["cypher_solution_match"] is True


def test_end_to_end_retries_failed_query_then_scores_final_answer(monkeypatch):
    record = raw_generation_record()
    responses = iter(
        [
            generated_result(generated_tool_call("INVALID CYPHER")),
            generated_result(generated_tool_call(record["gold_cypher"])),
            generated_result("<answer><O1></answer>"),
        ]
    )
    conversations = []
    generation_debug_turns = []

    def fake_generate(_model, _tokenizer, messages, **_kwargs):
        conversations.append(json.loads(json.dumps(messages)))
        return next(responses)

    class RetryValidator:
        def __init__(self):
            self.calls = []

        def validate(self, cypher, execute, max_rows=None, compute_result_hash=True):
            assert compute_result_hash is False
            self.calls.append((cypher, execute, max_rows))
            if cypher == "INVALID CYPHER":
                return {
                    "executable": False,
                    "error": "Neo.ClientError.Statement.SyntaxError: invalid",
                    "rows": [],
                    "rows_truncated": False,
                }
            return {
                "executable": True,
                "error": None,
                "rows": [{"object": "O1"}],
                "rows_truncated": False,
            }

    validator = RetryValidator()
    result = evaluate_end_to_end_record(
        record,
        runtime=FakeGenerationRuntime(fake_generate),
        validator=validator,
        policy=None,
        tool_schemas=[run_cypher_tool_schema()],
        max_new_tokens=32,
        enable_thinking=False,
        generation_debug_turns=generation_debug_turns,
    )

    assert result["status"] == "pass"
    assert result["final_answer_match"] is True
    assert result["query_attempts"] == 2
    assert result["retries_used"] == 1
    assert validator.calls[0][2] == DEFAULT_E2E_MAX_TOOL_ROWS
    assert "execution_error" in json.dumps(conversations[1])
    assert any('"ok": true' in message.get("content", "") for message in conversations[2])
    assert "INVALID CYPHER" not in json.dumps(conversations[2])
    assert [turn["turn"] for turn in generation_debug_turns] == [1, 2, 3]
    assert [turn["rendered_prompt"] for turn in generation_debug_turns] == [
        "bounded prompt",
        "bounded prompt",
        "bounded prompt",
    ]
    assert generation_debug_turns[-1]["raw_completion"] == "<answer><O1></answer>"


def test_end_to_end_uses_explain_failure_as_static_retry_feedback():
    record = raw_generation_record()

    def fake_generate(_model, _tokenizer, _messages, **_kwargs):
        return generated_result(generated_tool_call("INVALID CYPHER"))

    class ExplainRejectingValidator:
        def explain(self, cypher):
            assert cypher == "INVALID CYPHER"
            return {
                "syntax_error": True,
                "executable": False,
                "error": "Neo.ClientError.Statement.SyntaxError: invalid",
            }

        def validate(self, *_args, **_kwargs):
            raise AssertionError("EXPLAIN-invalid Cypher must not be executed")

    result = evaluate_end_to_end_record(
        record,
        runtime=FakeGenerationRuntime(fake_generate),
        validator=ExplainRejectingValidator(),
        policy=None,
        tool_schemas=[run_cypher_tool_schema()],
        max_new_tokens=32,
        enable_thinking=False,
        max_query_retries=0,
    )

    assert result["status"] == "query_retries_exhausted"
    assert result["last_failure_reason"] == "static_error"
    assert result["attempts"][0]["status"] == "static_error"
    assert result["attempts"][0]["explain_validation"]["syntax_error"] is True


def test_end_to_end_accepts_ollama_native_tool_calls():
    record = raw_generation_record()
    responses = iter(
        [
            generated_native_tool_call(record["gold_cypher"]),
            generated_result("<answer><O1></answer>"),
        ]
    )
    conversations = []

    def fake_generate(_model, _tokenizer, messages, **_kwargs):
        conversations.append(json.loads(json.dumps(messages)))
        return next(responses)

    class NativeValidator:
        def validate(self, _cypher, execute, **_kwargs):
            assert execute is True
            return {
                "executable": True,
                "rows": [{"object": "O1"}],
                "rows_truncated": False,
                "error": None,
            }

    result = evaluate_end_to_end_record(
        record,
        runtime=FakeGenerationRuntime(fake_generate),
        validator=NativeValidator(),
        policy=None,
        tool_schemas=[run_cypher_tool_schema()],
        max_new_tokens=32,
        enable_thinking=False,
    )

    assert result["status"] == "pass"
    assert result["final_answer_match"] is True
    assert conversations[1][-2]["tool_calls"][0]["function"]["name"] == (
        "run_cypher_query"
    )
    assert conversations[1][-1]["role"] == "tool"


def test_default_end_to_end_limits_accept_bounded_room_result(monkeypatch):
    record = raw_generation_record()
    rows = [
        {
            "object": f"O{index}",
            "class": "office chair",
            "center": [1.234567, 2.345678, 3.456789],
        }
        for index in range(112)
    ]
    serialized = json.dumps({"ok": True, "rows": rows})
    assert len(serialized) > 8192
    assert len(serialized) < DEFAULT_E2E_MAX_TOOL_RESPONSE_CHARS

    responses = iter(
        [
            generated_result(generated_tool_call(record["gold_cypher"])),
            generated_result("<answer><O1></answer>"),
        ]
    )
    conversations = []

    def fake_generate(_model, _tokenizer, messages, **_kwargs):
        conversations.append(json.loads(json.dumps(messages)))
        return next(responses)

    class RoomResultValidator:
        def validate(self, _cypher, execute, max_rows=None, compute_result_hash=True):
            assert execute is True
            assert compute_result_hash is False
            assert max_rows == DEFAULT_E2E_MAX_TOOL_ROWS
            return {
                "executable": True,
                "error": None,
                "rows": rows,
                "rows_truncated": False,
            }

    result = evaluate_end_to_end_record(
        record,
        runtime=FakeGenerationRuntime(fake_generate),
        validator=RoomResultValidator(),
        policy=None,
        tool_schemas=[run_cypher_tool_schema()],
        max_new_tokens=32,
        enable_thinking=False,
    )

    assert result["status"] == "pass"
    assert result["query_attempts"] == 1
    assert any('"ok": true' in message.get("content", "") for message in conversations[1])


def test_neo4j_notifications_are_suppressed_unless_requested(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "cypher_3dsg_agent_distillation.graph.neo4j_validation."
        "suppress_neo4j_notification_logs",
        lambda: calls.append(True),
    )

    configure_neo4j_notification_output(show_notifications=False)
    configure_neo4j_notification_output(show_notifications=True)

    assert calls == [True]


def test_isolated_tool_check_omits_oversized_rows_from_report():
    record = raw_generation_record()

    class LargeResultValidator:
        def validate(
            self,
            cypher,
            execute,
            max_rows=None,
            compute_result_hash=True,
        ):
            assert max_rows == 2
            assert compute_result_hash is False
            return {
                "executable": True,
                "rows": [{"object": "DO-NOT-STORE" * 100}],
                "rows_truncated": False,
                "result_hash": None,
                "error": None,
            }

    result = score_tool_call_generation(
        record,
        generated_tool_call(record["gold_cypher"]),
        validator=LargeResultValidator(),
        policy=None,
        max_rows=2,
        max_result_chars=256,
    )

    assert result["status"] == "result_too_large"
    assert result["validation"]["rows"] == []
    assert result["validation"]["rows_omitted"] is True
    assert "DO-NOT-STORE" not in json.dumps(result)


def test_end_to_end_retries_oversized_results_without_feeding_rows(monkeypatch):
    record = raw_generation_record()
    responses = iter(
        [
            generated_result(generated_tool_call("MATCH (o:Object) RETURN o")),
            generated_result(generated_tool_call(record["gold_cypher"])),
            generated_result("<answer><O1></answer>"),
        ]
    )
    conversations = []

    def fake_generate(_model, _tokenizer, messages, **_kwargs):
        conversations.append(json.loads(json.dumps(messages)))
        return next(responses)

    class BoundedValidator:
        def validate(self, cypher, execute, max_rows=None, compute_result_hash=True):
            assert compute_result_hash is False
            assert max_rows == 2
            if cypher.startswith("MATCH (o:Object)"):
                return {
                    "executable": True,
                    "rows": [{"object": "SECRET-LARGE-ROW" * 100}],
                    "rows_truncated": False,
                }
            return {
                "executable": True,
                "rows": [{"object": "O1"}],
                "rows_truncated": False,
            }

    result = evaluate_end_to_end_record(
        record,
        runtime=FakeGenerationRuntime(fake_generate),
        validator=BoundedValidator(),
        policy=None,
        tool_schemas=[run_cypher_tool_schema()],
        max_new_tokens=32,
        enable_thinking=False,
        max_tool_rows=2,
        max_tool_response_chars=512,
    )

    retry_prompt = json.dumps(conversations[1])
    report_text = json.dumps(result)
    assert result["final_answer_match"] is True
    assert result["attempts"][0]["status"] == "result_too_large"
    assert "SECRET-LARGE-ROW" not in retry_prompt
    assert "SECRET-LARGE-ROW" not in report_text
    assert result["attempts"][0]["tool_response_chars"] <= 512


def test_end_to_end_stops_after_default_one_query_retry(monkeypatch):
    responses = iter(
        [
            generated_result("<tool_call><function=run_cypher_query>"),
            generated_result("<tool_call><function=run_cypher_query>"),
        ]
    )
    result = evaluate_end_to_end_record(
        raw_generation_record(),
        runtime=FakeGenerationRuntime(
            lambda *_args, **_kwargs: next(responses)
        ),
        validator=FakeValidator(),
        policy=None,
        tool_schemas=[run_cypher_tool_schema()],
        max_new_tokens=32,
        enable_thinking=False,
    )

    assert result["status"] == "query_retries_exhausted"
    assert result["final_answer_match"] is False
    assert result["query_attempts"] == 2
    assert result["retries_used"] == 1
    assert len(result["attempts"]) == 2


def test_end_to_end_does_not_replay_an_overlong_query(monkeypatch):
    oversized_query = "MATCH (o:Object) RETURN o // " + ("X" * 1000)
    responses = iter(
        [
            generated_result(generated_tool_call(oversized_query)),
            generated_result("<answer><O1></answer>"),
        ]
    )
    conversations = []

    def fake_generate(_model, _tokenizer, messages, **_kwargs):
        conversations.append(json.loads(json.dumps(messages)))
        return next(responses)

    result = evaluate_end_to_end_record(
        raw_generation_record(),
        runtime=FakeGenerationRuntime(fake_generate),
        validator=FakeValidator(),
        policy=None,
        tool_schemas=[run_cypher_tool_schema()],
        max_new_tokens=32,
        enable_thinking=False,
        max_query_chars=256,
    )

    retry_assistant = conversations[1][-2]
    assert result["final_answer_match"] is True
    assert result["attempts"][0]["status"] == "query_too_large"
    assert len(retry_assistant["content"]) <= 256
    assert oversized_query not in json.dumps(conversations[1])


def test_summarize_raw_generation_records_counts_both_checks():
    record = {
        FINAL_ANSWER_CHECK: {
            "status": "pass",
            "final_answer_generated": True,
            "final_answer_has_tags": True,
            "final_answer_format_valid": True,
            "final_answer_match": True,
        },
        TOOL_CALL_CHECK: {
            "status": "missing_required_tool",
            "tool_call_generated": False,
            "tool_call_parse_success": False,
            "correct_tool_name": False,
            "cypher_argument_present": False,
            "cypher_exact_match": False,
            "static_valid": False,
            "executable": False,
            "cypher_solution_match": False,
        },
    }

    summary = summarize_raw_generation_records([record])

    assert summary["examples"] == 1
    assert summary[FINAL_ANSWER_CHECK]["rates"]["final_answer_match"] == 1.0
    assert summary[TOOL_CALL_CHECK]["statuses"] == {"missing_required_tool": 1}
    assert summary[END_TO_END_CHECK]["metrics"] == {"final_answer_match": 0}
    assert summary[END_TO_END_CHECK]["skipped"] == 1


def test_end_to_end_summary_exposes_only_final_answer_match_metric():
    record = {
        FINAL_ANSWER_CHECK: {"status": "pass", "final_answer_match": True},
        TOOL_CALL_CHECK: {"status": "pass", "executable": True},
        END_TO_END_CHECK: {
            "status": "pass",
            "final_answer_match": True,
            "query_attempts": 2,
            "retries_used": 1,
        },
    }

    section = summarize_raw_generation_records([record])[END_TO_END_CHECK]

    assert section["metrics"] == {"final_answer_match": 1}
    assert section["rates"] == {"final_answer_match": 1.0}


def test_summarize_raw_generation_records_reports_pddl_grounding_metrics():
    record = {
        "task_family": "cypher_pddl",
        FINAL_ANSWER_CHECK: {
            "status": "pass",
            "final_answer_format_valid": True,
            "final_answer_domain_valid": True,
            "final_answer_match": True,
        },
        TOOL_CALL_CHECK: {
            "status": "pass",
            "static_valid": True,
            "executable": True,
            "cypher_solution_match": True,
            "tool_result_match": True,
            "grounding_match": True,
        },
    }

    family_summary = summarize_raw_generation_records([record])["by_task_family"][
        "cypher_pddl"
    ]

    assert family_summary["examples"] == 1
    assert family_summary[FINAL_ANSWER_CHECK]["rates"][
        "final_answer_domain_valid"
    ] == 1.0
    assert family_summary[TOOL_CALL_CHECK]["rates"]["grounding_match"] == 1.0


def test_summarize_raw_generation_records_counts_hidden_false_positive_risks():
    record = {
        "task_family": "cypher_qa",
        FINAL_ANSWER_CHECK: {
            "status": "pass",
            "final_answer_format_valid": True,
            "final_answer_domain_valid": True,
            "final_answer_match": True,
        },
        TOOL_CALL_CHECK: {
            "status": "pass",
            "static_valid": True,
            "executable": True,
            "cypher_solution_match": True,
            "tool_result_match": True,
            "result_match_without_exact_cypher_match": True,
            "potential_hidden_false_positive": True,
        },
    }

    summary = summarize_raw_generation_records([record])

    assert summary[TOOL_CALL_CHECK]["metrics"]["potential_hidden_false_positive"] == 1
    assert summary["by_task_family"]["cypher_qa"][TOOL_CALL_CHECK]["metrics"][
        "potential_hidden_false_positive"
    ] == 1


def test_finite_perplexity_reports_none_before_json_overflow():
    assert finite_perplexity(700.0) is None
