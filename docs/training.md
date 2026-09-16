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

| Profile | Base model | Default adapter mode |
| --- | --- | --- |
| `configs/student_qwen3.5_4b.yaml` | `Qwen/Qwen3.5-4B` | LoRA |
| `configs/student_granite4.1_3b.yaml` | `ibm-granite/granite-4.1-3b` | QLoRA |

`model.load_in_4bit: true` enables 4-bit loading for QLoRA; `false` keeps the
base model at the configured `dtype` for LoRA.

## Configuration map

Each `configs/student_*.yaml` file has four main sections:

| Section | Controls |
| --- | --- |
| `model` | Base model, tokenizer, dtype, remote code, and 4-bit loading |
| `lora` | Rank, alpha, dropout, and target modules |
| `training` | Output root, loss mode, sequence length, batches, optimizer schedule, checkpoints, and precision |
| `ollama_export` | GGUF output type, quantizations, and Modelfile defaults |

### Loss modes

| Mode | Supervised tokens |
| --- | --- |
| `assistant-only` | Assistant turns, including generated tool calls and final answers |
| `full` | All non-padding tokens in the rendered conversation |

`assistant-only` is the default. The tokenizer receives the split's native
`run_cypher_query` JSON schema once through its chat template; the schema is not
duplicated in each stored trace message.

Controlled-repair traces identify the initial mutated Cypher call and syntax
error as context-only messages. Their token spans are masked in both loss modes.
The corrected Cypher call and final answer remain supervised, teaching the model
to use tool feedback without training it to reproduce the intentional error.

The training split does not apply scene-specific ordering. The standard SFT
sampler shuffles the prepared traces in every training epoch.

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
docker/stack.sh main python scripts/inspect_training_dataset.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --split train \
  --index 0
```

Use this check after changing prompts, render views, tokenizer settings, loss
mode, or maximum sequence length.

## Train a student

Start a new timestamped model version:

```bash
docker/stack.sh main python scripts/train_agent_lora.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml
```

To bind training to an exact dataset or choose a model-version identifier:

```bash
docker/stack.sh main python scripts/train_agent_lora.py \
  --data-config configs/data.yaml \
  --dataset-version 2026-09-08-T-12-05-47-Z \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-version 2026-09-08-T-13-00-00-Z
```

The output directory comes from `training.output_dir`. Each successful run
creates a version below that directory and records the dataset version, split
hashes, model configuration, metrics, and key output fingerprints in
`metadata.yaml`.

## Automated training workflow

For a reproducible end-to-end demonstration, run:

```bash
docker/stack.sh demo-train
```

The command performs question and trace generation, splitting, base-model
raw-generation validation, training, fine-tuned eval-loss and raw generation,
and HTML report generation. Database-backed stages load every configured scene
when needed. Override its inputs with:

| Variable | Default |
| --- | --- |
| `DEMO_DATA_CONFIG` | `configs/data.yaml` |
| `DEMO_TEACHER_CONFIG` | `configs/teacher_synthetic.yaml` |
| `DEMO_STUDENT_CONFIG` | `configs/student_qwen3.5_4b.yaml` |

## Next steps

- Compare the trained adapter with its base model using [Validation](validation.md).
- Package the selected model using [Ollama GGUF export](ollama_export.md).
- Review lineage and latest-version behavior in
  [Artifact versioning](artifact_versioning.md).
