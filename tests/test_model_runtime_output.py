from __future__ import annotations

import logging

from cypher_3dsg_agent_distillation.model_runtime_output import (
    suppress_hugging_face_auth_log,
)


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_hugging_face_auth_filter_suppresses_only_missing_token_warning():
    logger = logging.Logger("test.huggingface_hub")
    handler = RecordingHandler()
    logger.addHandler(handler)
    suppress_hugging_face_auth_log(logger)

    logger.warning(
        "Warning: You are sending unauthenticated requests to the HF Hub. "
        "Please set a HF_TOKEN to enable higher rate limits."
    )
    logger.warning("A different model download warning")

    assert handler.messages == ["A different model download warning"]
