# Student Training

Fine-tune a Hugging Face model with LoRA or QLoRA using validated agent traces.
Commands use the [Docker stack](../docker/README.md).

## Prerequisites

Complete [dataset generation](dataset_generation.md), including teacher traces
and train/evaluation splits. The configured training workflows require an
NVIDIA GPU.

| Student configuration | Base model | Method |
| --- | --- | --- |
| `configs/students/student_qwen3.5_4b.yaml` | `Qwen/Qwen3.5-4B` | LoRA |
| `configs/students/student_granite4.1_3b.yaml` | `ibm-granite/granite-4.1-3b` | QLoRA |

`model.load_in_4bit: true` enables QLoRA. Set it to `false` to load the base
model at the configured dtype for LoRA.

## Configure training

| Student YAML section | Controls |
| --- | --- |
| `model` | Base model, tokenizer, dtype, and 4-bit loading |
| `lora` | Rank, alpha, dropout, and target modules |
| `training` | Output path, loss, sequence length, batches, schedule, and checkpoints |
| `inference` | Thinking mode for evaluation |
| `ollama_export` | Export settings, described in [GGUF export](ollama_export.md#student-export-configuration) |

### Supervision and sequence length

| `training.loss_mode` | Supervised tokens |
| --- | --- |
| `assistant-only` (default) | Assistant tool calls and final answers |
| `full` | All non-padding conversation tokens |

Controlled-repair examples keep the intentionally faulty call and its error as
context. These spans are masked in both loss modes. The corrected call and
final answer remain supervised.

`training.max_length` sets the token limit. Overlength traces are excluded
without truncation, with a warning for each affected split. Training shuffles
retained examples each epoch.

Set `training.include_reasoning` or use the CLI reasoning flags to include
reasoning in supervision. See [Reasoning experiments](reasoning.md) for matched
comparisons and retained-example selection.

### Loss and warmup

| `training.loss_type` | Use |
| --- | --- |
| `chunked_nll` (default) | Cross-entropy with lower memory use |
| `nll` | Native loss when an architecture is unsupported by the chunked path |
| `dft` | Alternative training objective, not directly comparable to report NLL |

Native loss can require substantially more GPU memory.

`training.warmup_steps` accepts an integer step count or a fraction in `[0, 1)`.
For example, `0.05` warms up over 5% of optimizer steps.

## Inspect the rendered dataset

Inspect text, tokenization, and supervised spans before a long run:

```bash
docker/stack.sh main python scripts/training/inspect_training_dataset.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split train --index 0
```

Repeat this after changing prompts, tokenizer settings, loss mode, or sequence
length.

## Train a student

```bash
docker/stack.sh main python scripts/training/train_agent_lora.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml
```

Training creates a new model version under `training.output_dir`. Add
`--dataset-version DATASET_VERSION` to select exact input data and
`--model-version MODEL_VERSION` to choose the output identifier.

The model's `metadata.yaml` records its inputs, metrics, and selected checkpoint.
See [Artifact versioning](artifact_versioning.md) for selection and lineage rules.

## Automated training workflow

Run `docker/stack.sh demo-train` to train on the current dataset. See
[Docker automated workflows](../docker/README.md#automated-workflows) for the
complete sequence and input overrides.

## Next steps

- Compare the student and base model with [Validation](validation.md).
- Package the selected model with [GGUF export](ollama_export.md).
- Browse training results in the [HTML workflow report](workflow_report.md).
