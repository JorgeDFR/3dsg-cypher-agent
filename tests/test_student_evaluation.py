from contextlib import nullcontext

import pytest

import cypher_3dsg_agent_distillation.student_evaluation as student_evaluation
import cypher_3dsg_agent_distillation.train as train_module
from cypher_3dsg_agent_distillation.student_evaluation import (
    ALL_SPLITS,
    EVAL_LOSS_MODE,
    FINAL_ANSWER_CHECK,
    TOOL_CALL_CHECK,
    clean_generated_text,
    combine_eval_loss_summaries,
    default_adapter_dir,
    evaluate_eval_loss_splits,
    evaluate_eval_loss_with_loaded_model,
    final_answer_prompt_messages,
    finite_perplexity,
    generation_length_kwargs,
    initial_prompt_messages,
    parse_generated_tool_calls,
    render_messages_for_generation,
    report_output_path,
    score_final_answer_generation,
    score_tool_call_generation,
    summarize_raw_generation_records,
    tokenizer_source,
)


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
    def parameters(self):
        return iter(())

    def __call__(self, **_kwargs):
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


def test_report_output_path_uses_split_and_mode_name():
    output = report_output_path(
        data_config={"test_report_path": "data/3dsg/scene/reports/test.agent.report.json"},
        split="challenge",
        mode=EVAL_LOSS_MODE,
        output=None,
    )

    assert output.name == "challenge.student.eval-loss.report.json"


def test_report_output_path_uses_aggregate_name_for_all_splits():
    output = report_output_path(
        data_config={"test_report_path": "data/3dsg/scene/reports/test.agent.report.json"},
        split=ALL_SPLITS,
        mode=EVAL_LOSS_MODE,
        output=None,
    )

    assert output.name == "student.eval-loss.report.json"


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
        student_evaluation,
        "require_eval_loss_dependencies",
        lambda: (FakeTorch, None, FakeDataLoader, None, None, None),
    )
    monkeypatch.setattr(train_module, "CausalLmDataCollator", lambda _tokenizer: None)
    monkeypatch.setattr(
        train_module,
        "build_training_dataset",
        lambda *_args, **_kwargs: [{"uid": "one"}, {"uid": "two"}, {"uid": "three"}],
    )
    monkeypatch.setattr(
        student_evaluation,
        "shifted_token_metrics",
        lambda _outputs, _labels: {
            "tokens": 5,
            "entropy_sum": 1.0,
            "correct_tokens": 4,
        },
    )
    progress = FakeProgress()

    summary = evaluate_eval_loss_with_loaded_model(
        data_config={"splits": {"train": "train.jsonl"}},
        split="train",
        model=FakeModel(),
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
            "description": "eval-loss train",
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


def test_eval_loss_progress_starts_after_model_and_dataset_setup(monkeypatch):
    events = []

    monkeypatch.setattr(
        student_evaluation,
        "load_student_model_and_tokenizer",
        lambda **_kwargs: (events.append("load_model") or FakeModel(), object()),
    )
    monkeypatch.setattr(
        student_evaluation,
        "eval_loss_progress_context",
        lambda _show_progress: RecordingProgressContext(events),
    )
    monkeypatch.setattr(
        student_evaluation,
        "require_eval_loss_dependencies",
        lambda: (FakeTorch, None, FakeDataLoader, None, None, None),
    )
    monkeypatch.setattr(train_module, "CausalLmDataCollator", lambda _tokenizer: None)

    def fake_build_training_dataset(*_args, **_kwargs):
        events.append("build_dataset")
        return [{"uid": "one"}]

    monkeypatch.setattr(train_module, "build_training_dataset", fake_build_training_dataset)
    monkeypatch.setattr(
        student_evaluation,
        "shifted_token_metrics",
        lambda _outputs, _labels: {
            "tokens": 5,
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


def test_finite_perplexity_reports_none_before_json_overflow():
    assert finite_perplexity(700.0) is None
