from __future__ import annotations

import os
import warnings

HF_TOKEN_WARNING_PATTERN = r".*unauthenticated requests to the HF Hub.*"


def configure_model_library_output() -> None:
    """Disable third-party progress bars while retaining repository-owned Rich output."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    warnings.filterwarnings("ignore", message=HF_TOKEN_WARNING_PATTERN)

    try:
        from huggingface_hub.utils import disable_progress_bars
        from huggingface_hub.utils import logging as huggingface_logging

        disable_progress_bars()
        suppress_hugging_face_auth_log(huggingface_logging.get_logger())
    except ImportError:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
    except ImportError:
        pass

    try:
        from datasets.utils import logging as datasets_logging

        datasets_logging.disable_progress_bar()
    except ImportError:
        pass


def suppress_hugging_face_auth_log(logger) -> None:
    """Drop only Hugging Face's irrelevant missing-token download warning."""

    class HuggingFaceAuthWarningFilter:
        _cypher_3dsg_hf_auth_filter = True

        def filter(self, record) -> bool:
            message = record.getMessage().lower()
            return not (
                "unauthenticated requests to the hf hub" in message
                and "hf_token" in message
            )

    targets = [logger, *logger.handlers]
    for target in targets:
        if any(
            getattr(filter_, "_cypher_3dsg_hf_auth_filter", False)
            for filter_ in target.filters
        ):
            continue
        target.addFilter(HuggingFaceAuthWarningFilter())
