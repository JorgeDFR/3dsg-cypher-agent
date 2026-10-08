from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace


def _trl_needs_scaling_bridge() -> bool:
    """Check actual chunked-loss behavior, independently of TRL source spelling.

    Use tiny deterministic CPU tensors, without loading a model or consuming RNG
    state. Recognize both legacy multiplier-only and upstream divisor-aware loss.
    Anything else fails before the real model's config is changed.
    """
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from trl.trainer import sft_trainer

    class ProbeConfig(SimpleNamespace):
        def get_text_config(self, **kwargs):
            return self

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            hidden = torch.tensor([[[1.0, -0.5], [0.25, 2.0], [-1.0, 0.5]]])
            return BaseModelOutputWithPast(last_hidden_state=hidden)

    class ProbeModel(torch.nn.Module):
        def __init__(self, multiplier, divisor):
            super().__init__()
            self.config = ProbeConfig(logit_scale=multiplier, logits_scaling=divisor)
            self.model = Backbone()
            self.lm_head = torch.nn.Module()
            self.lm_head.weight = torch.nn.Parameter(
                torch.tensor([[2.0, -1.0], [-0.5, 1.0], [0.25, 0.5]])
            )
            self.lm_head.bias = None

        @property
        def base_model(self):
            return self.model

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, **kwargs):
            raise AssertionError("The scaling probe must use the patched loss path.")

    modes = []
    try:
        for multiplier, divisor in ((0.7, 3.0), (1.3, 7.0)):
            model = ProbeModel(multiplier, divisor)
            input_ids = torch.tensor([[0, 1, 2]])
            labels = torch.tensor([[-100, 1, 2]])
            with torch.no_grad():
                hidden = model.model(input_ids).last_hidden_state[:, :-1]
                logits = hidden @ model.lm_head.weight.t() * multiplier
                legacy = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 3), labels[:, 1:].reshape(-1)
                )
                native = torch.nn.functional.cross_entropy(
                    (logits / divisor).reshape(-1, 3), labels[:, 1:].reshape(-1)
                )
                sft_trainer._patch_chunked_ce_lm_head(model, chunk_size=2)
                actual = model(input_ids=input_ids, labels=labels).loss
            if torch.allclose(actual, legacy, atol=1e-6, rtol=1e-5):
                modes.append(True)
            elif torch.allclose(actual, native, atol=1e-6, rtol=1e-5):
                modes.append(False)
            else:
                raise ValueError("Chunked loss matches neither supported scaling behavior.")
        if modes[0] != modes[1]:
            raise ValueError("Chunked loss scaling is inconsistent across probe configurations.")
    except Exception as exc:
        raise RuntimeError(
            "Cannot verify TRL chunked loss scaling numerically. "
            "Use training.loss_type: nll or a validated TRL version."
        ) from exc
    return modes[0]


@contextmanager
def chunked_loss_scaling(model, loss_type: str):
    """Expose native divisive scaling while SFTTrainer captures its loss config.

    This must surround trainer construction, before Accelerate wraps forward.
    TRL retains the multiplier in its chunked forward closure. Restoring config
    afterward keeps native inference, serialization, and adapter reload intact.
    """
    if loss_type != "chunked_nll":
        yield
        return

    config = model.config
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        configs.append(text_config)

    changes = []
    for config in configs:
        divisor = getattr(config, "logits_scaling", None)
        if divisor is None:
            continue
        divisor = float(divisor)
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("Model logits_scaling must be finite and positive.")
        if divisor == 1.0:
            continue
        multiplier = getattr(config, "logit_scale", None)
        if multiplier is None:
            multiplier = getattr(config, "output_multiplier", None)
        multiplier = 1.0 if multiplier is None else float(multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Model logit_scale must be finite and positive.")
        changes.append(
            (
                config,
                multiplier / divisor,
                "logit_scale" in vars(config),
                vars(config).get("logit_scale"),
            )
        )

    if changes and not _trl_needs_scaling_bridge():
        changes = []
    try:
        for config, scale, _, _ in changes:
            config.logit_scale = scale
        yield
    finally:
        for config, _, had_value, old_value in reversed(changes):
            if had_value:
                config.logit_scale = old_value
            else:
                delattr(config, "logit_scale")
