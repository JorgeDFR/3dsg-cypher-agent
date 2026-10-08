from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from cypher_3dsg_agent_distillation.training.chunked_loss import chunked_loss_scaling


@pytest.mark.parametrize("loss_type", ["nll", "dft"])
def test_native_loss_does_not_change_scaling(loss_type):
    config = SimpleNamespace(logits_scaling=10.0)
    with chunked_loss_scaling(SimpleNamespace(config=config), loss_type):
        assert vars(config) == {"logits_scaling": 10.0}


@pytest.mark.parametrize("divisor", [0, -1, float("inf"), float("nan")])
def test_invalid_divisor_is_rejected_without_mutating_config(divisor):
    config = SimpleNamespace(logits_scaling=divisor)
    with pytest.raises(ValueError, match="logits_scaling"):
        with chunked_loss_scaling(SimpleNamespace(config=config), "chunked_nll"):
            pytest.fail("Invalid scaling was accepted")
    assert "logit_scale" not in vars(config)


def require_training_stack():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("trl")
    return torch


def test_nested_scaling_composes_and_restores_on_failure():
    require_training_stack()
    text_config = SimpleNamespace(logits_scaling=10.0, logit_scale=2.0)
    config = SimpleNamespace(text_config=text_config)
    with pytest.raises(RuntimeError, match="constructor failed"):
        with chunked_loss_scaling(SimpleNamespace(config=config), "chunked_nll"):
            assert text_config.logit_scale == 0.2
            assert "logit_scale" not in vars(config)
            raise RuntimeError("constructor failed")
    assert text_config.logit_scale == 2.0


@pytest.mark.parametrize("output_multiplier, expected", [(None, 0.1), (2.0, 0.2)])
def test_nullable_multiplier_uses_output_multiplier_fallback(output_multiplier, expected):
    require_training_stack()
    config = SimpleNamespace(
        logits_scaling=10.0, logit_scale=None, output_multiplier=output_multiplier
    )
    with chunked_loss_scaling(SimpleNamespace(config=config), "chunked_nll"):
        assert config.logit_scale == expected
    assert config.logit_scale is None
    assert config.output_multiplier == output_multiplier


def test_unknown_trl_scaling_contract_fails_before_mutation(monkeypatch):
    require_training_stack()
    from trl.trainer import sft_trainer

    def unsupported_patch(model, chunk_size):
        raise ValueError("Unsupported loss implementation")

    monkeypatch.setattr(sft_trainer, "_patch_chunked_ce_lm_head", unsupported_patch)
    config = SimpleNamespace(logits_scaling=10.0)
    with pytest.raises(RuntimeError, match="Cannot verify TRL"):
        with chunked_loss_scaling(SimpleNamespace(config=config), "chunked_nll"):
            pytest.fail("Unknown TRL implementation was accepted")
    assert "logit_scale" not in vars(config)


def test_upstream_divisor_support_is_not_applied_twice(monkeypatch):
    require_training_stack()
    from trl.trainer import sft_trainer

    original_patch = sft_trainer._patch_chunked_ce_lm_head

    def divisor_aware_patch(model, chunk_size):
        config = model.config
        original_scale = config.logit_scale
        config.logit_scale = original_scale / config.logits_scaling
        try:
            original_patch(model, chunk_size=chunk_size)
        finally:
            config.logit_scale = original_scale

    monkeypatch.setattr(sft_trainer, "_patch_chunked_ce_lm_head", divisor_aware_patch)
    config = SimpleNamespace(logits_scaling=10.0, logit_scale=2.0)
    with chunked_loss_scaling(SimpleNamespace(config=config), "chunked_nll"):
        assert config.logit_scale == 2.0
    assert config.logit_scale == 2.0


def test_scaling_probe_does_not_consume_rng_state():
    torch = require_training_stack()
    from cypher_3dsg_agent_distillation.training.chunked_loss import _trl_needs_scaling_bridge

    before = torch.get_rng_state().clone()
    assert _trl_needs_scaling_bridge() is True
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("family", ["granite", "llama", "gemma2"])
def test_chunked_loss_matches_native_loss_gradients_and_inference(family):
    torch = require_training_stack()
    from transformers import (
        Gemma2Config,
        Gemma2ForCausalLM,
        GraniteConfig,
        GraniteForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
    )
    from trl.trainer.sft_trainer import _patch_chunked_ce_lm_head

    config_cls, model_cls, extra = {
        "granite": (GraniteConfig, GraniteForCausalLM, {"logits_scaling": 10.0}),
        "llama": (LlamaConfig, LlamaForCausalLM, {}),
        "gemma2": (Gemma2Config, Gemma2ForCausalLM, {"final_logit_softcapping": 2.0}),
    }[family]
    torch.manual_seed(42)
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        **extra,
    )
    config._attn_implementation = "eager"
    native = model_cls(config).float().eval()
    chunked = deepcopy(native)
    original_config = deepcopy(chunked.config.to_dict())
    input_ids = torch.tensor([[1, 4, 7, 9, 2], [1, 3, 5, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels[1, -1] = -100

    with chunked_loss_scaling(chunked, "chunked_nll"):
        _patch_chunked_ce_lm_head(chunked, chunk_size=2)
    assert chunked.config.to_dict() == original_config
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
    native_output = native(**kwargs, labels=labels)
    chunked_output = chunked(**kwargs, labels=labels)
    assert chunked_output.logits is None  # Keep the memory-saving path.
    torch.testing.assert_close(chunked_output.loss, native_output.loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(chunked(**kwargs).logits, native_output.logits)
    native_output.loss.backward()
    chunked_output.loss.backward()
    for (name, reference), (_, actual) in zip(
        native.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert (reference.grad is None) == (actual.grad is None), name
        if reference.grad is not None:
            torch.testing.assert_close(actual.grad, reference.grad, atol=1e-6, rtol=1e-4)


def test_sft_trainer_captures_scaling_with_lora_and_saves_native_config(tmp_path):
    torch = require_training_stack()
    pytest.importorskip("peft")
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GraniteConfig, GraniteForCausalLM, PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer

    from cypher_3dsg_agent_distillation.training.train import CausalLmDataCollator

    config = GraniteConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        logits_scaling=10.0,
        max_position_embeddings=32,
    )
    model = GraniteForCausalLM(config).float()
    reload_base = deepcopy(model)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 4, 7, 9, 2],
                "attention_mask": [1] * 5,
                "labels": [-100, -100, 7, 9, 2],
            },
        ]
    )
    with chunked_loss_scaling(model, "chunked_nll"):
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=CausalLmDataCollator(tokenizer),
            peft_config=LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
            args=SFTConfig(
                output_dir=str(tmp_path),
                use_cpu=True,
                bf16=False,
                report_to="none",
                loss_type="chunked_nll",
                gradient_checkpointing=False,
                dataset_kwargs={"skip_prepare_dataset": True},
            ),
        )
    assert "logit_scale" not in model.config.to_dict()
    trainer.model.eval()
    batch = CausalLmDataCollator(tokenizer)([dataset[0]])
    loss = trainer.compute_loss(trainer.model, dict(batch))
    logits = trainer.model(input_ids=batch["input_ids"], use_cache=False).logits
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 32), batch["labels"][:, 1:].reshape(-1)
    )
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
    loss.backward()
    assert any(p.grad is not None for p in trainer.model.parameters() if p.requires_grad)
    torch.optim.SGD([p for p in trainer.model.parameters() if p.requires_grad], lr=0.1).step()
    with torch.no_grad():
        expected_logits = trainer.model(input_ids=batch["input_ids"], use_cache=False).logits
    trainer.save_model(str(tmp_path / "adapter"))
    assert "logit_scale" not in model.config.to_dict()
    reloaded = PeftModel.from_pretrained(reload_base, str(tmp_path / "adapter")).eval()
    with torch.no_grad():
        actual_logits = reloaded(input_ids=batch["input_ids"], use_cache=False).logits
    torch.testing.assert_close(actual_logits, expected_logits)
    assert "logit_scale" not in reloaded.config.to_dict()
