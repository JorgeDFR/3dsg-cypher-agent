# Student Training

The training pipeline fine-tunes a Hugging Face causal language model on
validated agent traces. It supports both LoRA and QLoRA through the same student
configuration.

## Prerequisites

Before training:

1. Build the recommended [Docker environment](../docker/README.md).
2. Complete [dataset generation](dataset_generation.md).
3. Confirm that the configured train and evaluation splits exist.

The default training path requires an NVIDIA GPU. Available student profiles
include:

| Profile | Base model | Fine-tuning method |
| --- | --- | --- |
| `configs/students/student_qwen3.5_4b.yaml` | `Qwen/Qwen3.5-4B` | LoRA |
| `configs/students/student_granite4.1_3b.yaml` | `ibm-granite/granite-4.1-3b` | QLoRA |

`model.load_in_4bit: true` enables 4-bit loading for QLoRA. `false` keeps the
base model at the configured `dtype` for LoRA.

## Configuration map

Each `configs/students/student_*.yaml` file has five main sections:

| Section | Controls |
| --- | --- |
| `model` | Base model, tokenizer, dtype, remote code, and 4-bit loading |
| `lora` | Rank, alpha, dropout, and target modules |
| `training` | Output root, loss mode, sequence length, batches, optimizer schedule, checkpoints, and precision |
| `inference` | Default thinking mode for student generation |
| `ollama_export` | GGUF output type, quantizations, and Modelfile defaults |

### Loss modes

| Mode | Supervised tokens |
| --- | --- |
| `assistant-only` | Assistant turns, including generated tool calls and final answers |
| `full` | All non-padding tokens in the rendered conversation |

`assistant-only` is the default. The tokenizer receives the split's native
`run_cypher_query` JSON schema once through its chat template. The schema is not
duplicated in each stored trace message.

Set `training.include_reasoning: true` or pass `--include-reasoning` to supervise
reasoning before tool calls and final answers. Use `--no-include-reasoning` for
the paired baseline on the same split files. See [reasoning experiments](reasoning_traces.md)
for sentence selection, matched retained UIDs, teacher pilots, and inference controls.

Controlled-repair traces identify the initial mutated Cypher call and syntax
error as context-only messages. Their token spans are masked in both loss modes.
The corrected Cypher call and final answer remain supervised, teaching the model
to use tool feedback without training it to reproduce the intentional error.

The training split does not apply scene-specific ordering. The standard SFT
sampler shuffles the prepared traces in every training epoch.

### Loss implementation

`training.loss_type` defaults to `chunked_nll` (including when explicitly set to
`null`). TRL projects supervised tokens and computes cross-entropy in chunks,
avoiding the full sequence-by-vocabulary logits tensor during training and
trainer evaluation. This is independent of `loss_mode`, which selects the
supervised tokens. The resolved loss type is recorded in `metadata.yaml` under
`effective_training_arguments`.

The training compatibility bridge reads `logits_scaling` from the model config
(including nested `text_config`) and combines its reciprocal with TRL's existing
`logit_scale` multiplier. This preserves divisive scaling such as Granite's
division by 10 without model-name checks. TRL's softcapping remains in place.
The temporary config adjustment is restored after trainer construction; native
inference, standalone evaluation, and saved configs retain their original scaling.
The bridge runs a tiny deterministic CPU calculation to check TRL's actual
scaling behavior, without relying on source-code formatting. It skips the
correction if TRL already applies the divisor, and fails with an actionable
error for unsupported behavior rather than risking applying the divisor twice.

For each new architecture or dependency upgrade, verify loss and gradient
agreement with native loss on identical examples and masks. Config-based scaling
does not cover every possible custom output transformation. Use explicit `nll`
for architectures whose native behavior is not supported by the chunked path;
it can require substantially more GPU memory. `dft` remains an explicit alternative
objective whose training loss is not directly comparable to report NLL.

Earlier Granite runs using TRL's `chunked_nll` omitted Granite's division of
logits by `logits_scaling`, while report evaluation applied it. Their training
losses are therefore not comparable to the native evaluation losses. This
scaling correction fixes future runs; train a new adapter to use the corrected objective.
Existing adapters and reports are not modified.

`training.warmup_steps` accepts an integer number of optimizer steps or a float
in `[0, 1)`. A value of `0.05` warms up for 5% of the total optimizer steps, so
changes to the split size, batch size, gradient accumulation, or epoch count
automatically change the number of warmup steps. Before training,
every rendered trace is tokenized and checked against `training.max_length`.
Overlength traces are excluded without truncation, and a warning reports the
number rejected from each split.

## Inspect the rendered dataset

Before a long run, inspect the exact text, tokenization, and supervised spans
for a split record:

```bash
docker/stack.sh main python scripts/training/inspect_training_dataset.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split train \
  --index 0
```

Use this check after changing prompts, render views, tokenizer settings, loss
mode, or maximum sequence length.

## Train a student

Start a new timestamped model version:

```bash
docker/stack.sh main python scripts/training/train_agent_lora.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml
```

To bind training to an exact dataset or choose a model-version identifier:

```bash
docker/stack.sh main python scripts/training/train_agent_lora.py \
  --data-config configs/data.yaml \
  --dataset-version 2026-09-08-T-12-05-47-Z \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-version 2026-09-08-T-13-00-00-Z
```

The output directory comes from `training.output_dir`. Each successful run
creates a version below that directory and records the dataset version, split
hashes, model configuration, metrics, and key output fingerprints in
`metadata.yaml`.

## Automated training workflow

For a reproducible end-to-end demonstration, run:

```bash
docker/stack.sh demo-dataset
docker/stack.sh demo-train
docker/stack.sh demo-validate
```

`demo-dataset` creates questions, teacher traces, and splits. `demo-train` only
trains a new student model using the current dataset, so rerun it to try another
student or training configuration without regenerating data. `demo-validate`
runs fine-tuned loss evaluation, test/challenge generation for the student model and
baseline, and HTML reporting. Matching completed validations are skipped.
Database-backed stages load every configured scene when needed. Override inputs with:

| Variable | Default |
| --- | --- |
| `DEMO_DATA_CONFIG` | `configs/data.yaml` |
| `DEMO_TEACHER_CONFIG` | `configs/teachers/teacher_synthetic.yaml` |
| `DEMO_STUDENT_CONFIG` | `configs/students/student_qwen3.5_4b.yaml` |

## Next steps

- Compare the fine-tuned model with its base model using [Validation](validation.md).
- Package the selected model using [Ollama GGUF export](ollama_export.md).
- Review lineage and latest-version behavior in
  [Artifact versioning](artifact_versioning.md).
