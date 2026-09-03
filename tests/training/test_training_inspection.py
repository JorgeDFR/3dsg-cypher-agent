from cypher_3dsg_agent_distillation.training.train import ASSISTANT_ONLY_LOSS_MODE
from cypher_3dsg_agent_distillation.training.training_inspection import inspect_record


class CharTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, input_ids, skip_special_tokens=False):
        return "".join(chr(input_id) for input_id in input_ids)


def test_inspect_record_reports_supervised_assistant_spans():
    report = inspect_record(
        {
            "uid": "one",
            "task_type": "object_class_count",
            "render_view": "ollama_native",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "<answer>3</answer>"},
            ],
        },
        CharTokenizer(),
        loss_mode=ASSISTANT_ONLY_LOSS_MODE,
        max_length=128,
        max_chars=1024,
    )

    assert report["supervised_tokens"] > 0
    assert len(report["supervised_spans"]) == 1
    assert "<answer>3</answer>" in report["supervised_spans"][0]["text"]
