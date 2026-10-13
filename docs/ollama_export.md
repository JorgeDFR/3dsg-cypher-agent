# Ollama GGUF Export

The exporter packages either a fine-tuned model or its configured base model as
GGUF artifacts for Ollama.

| Variant | Input | Pipeline |
| --- | --- | --- |
| Fine-tuned | Base model plus fine-tuned LoRA/QLoRA weights | Merge, convert, quantize, write Modelfile |
| Baseline | Hugging Face base checkpoint | Convert, quantize, write Modelfile |

The generated Modelfile is neutral: it selects a GGUF and stop parameters but
does not embed a task prompt. Project validation and Heracles Benchmark supply
their own Cypher or PDDL prompts at inference time.

## Prerequisites

The recommended [Docker stack](../docker/README.md) includes llama.cpp and the
export dependencies. Start it before running individual export commands:

```bash
docker/stack.sh up
```

A fine-tuned export also requires a completed student model from
[Student training](training.md).

For host execution, complete the centralized [Native setup](setup.md), including
its PyTorch, llama.cpp, and optional Ollama sections.

## Quick export

### Fine-tuned model

Export the latest completed Qwen student model through every stage:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages all
```

Use `--model-version VERSION` to select a fine-tuned model other than the latest
completed version.

### Baseline model

Export the exact base checkpoint configured by the same student profile:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-baseline \
  --stages all \
  --base-only
```

Baseline export skips merging fine-tuned weights and converts the local Hugging Face
path or resolved Hub snapshot directly.

## Export stages

`--stages` accepts one or more stages, or `all`.

| Stage | Output or action |
| --- | --- |
| `merge` | Merge the fine-tuned weights with the base model into a temporary Hugging Face checkpoint |
| `convert` | Convert the merged or base checkpoint to the configured GGUF type |
| `quantize` | Produce each configured llama.cpp quantization |
| `modelfile` | Write an Ollama Modelfile for one selected GGUF |
| `all` | Run the complete applicable sequence |

Run selected stages when earlier outputs are already available:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages quantize modelfile \
  --quantizations Q4_K_M Q8_0
```

`--dry-run` prints external operations without merging, converting,
quantizing, or writing files.

`--quiet` replaces third-party Hugging Face and llama.cpp output with the
project's Rich progress and completion summary. If an external command fails,
the exporter restores its last 80 output lines before raising the error. The
automated `demo-deploy` workflow enables this mode.

The llama.cpp Python environment follows llama.cpp's own dependency pins. It
can therefore use an older Transformers release than the project environment.
Before converting a merged checkpoint, the exporter rewrites the generic
`TokenizersBackend` class name to its backward-compatible alias,
`PreTrainedTokenizerFast`, in `tokenizer_config.json`. This avoids the
`Tokenizer class TokenizersBackend does not exist` error in Transformers 4
without changing `tokenizer.json` or the original training checkpoint. The
same normalization runs when retrying only the `convert` stage.
For newer architectures such as Qwen3.5, `convert_hf_to_gguf.py` may be unable
to instantiate `AutoConfig`. It then reads `config.json` directly. That
fallback is expected when GGUF conversion continues successfully.

## Output layout

With BF16 conversion plus Q8_0 and Q4_K_M quantization, the output resembles:

```text
outputs/ollama/
|-- qwen3.5-4b-baseline/
|   |-- qwen3.5-4b-baseline.bf16.gguf
|   |-- qwen3.5-4b-baseline.Q8_0.gguf
|   |-- qwen3.5-4b-baseline.Q4_K_M.gguf
|   |-- Modelfile
|   |-- Modelfile.json
|   `-- metadata.yaml
`-- qwen3.5-4b-finetune/
    |-- qwen3.5-4b-finetune.bf16.gguf
    |-- qwen3.5-4b-finetune.Q8_0.gguf
    |-- qwen3.5-4b-finetune.Q4_K_M.gguf
    |-- Modelfile
    |-- Modelfile.json
    `-- metadata.yaml
```

The directory identifies the model variant. The filename and metadata identify
the quantization. Fine-tuned export uses `merged-hf` only as a temporary
conversion input.

Exports are intentionally unversioned because the files are large. Review the
replacement and lineage rules in
[GGUF export policy](artifact_versioning.md#gguf-export-policy).

## Student export configuration

The `ollama_export` section of each `configs/students/student_*.yaml` profile controls:

| Key | Purpose |
| --- | --- |
| `model_name` | Default export directory and Ollama model name |
| `export_root` | Parent output directory |
| `gguf_outtype` | Intermediate conversion type, such as `bf16` |
| `quantizations` | Quantized artifacts to generate |
| `modelfile_quantization` | GGUF selected by the default Modelfile |
| `convert_args` | Extra `convert_hf_to_gguf.py` arguments |
| `quantize_args` | Extra `llama-quantize` arguments |

CLI arguments can override the fine-tuned model, merged checkpoint, GGUF, and Modelfile
paths.

## Select a Modelfile artifact

`--modelfile-gguf` selects an existing artifact without repeating conversion.
It accepts:

- unquantized types such as `bf16`, `f16`, and `f32`.
- short aliases such as `q4` and `q8`.
- full llama.cpp names such as `Q4_K_M` and `Q8_0`.

For example, point the default Modelfile at BF16:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile \
  --modelfile-gguf bf16
```

The command fails if the selected GGUF is absent. The selector and resolved
path are saved in `Modelfile.json`.

Generate conventionally named Modelfiles for several existing GGUFs in one
operation:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile \
  --modelfile-variants bf16 Q8_0 Q4_K_M \
  --quiet
```

This writes `Modelfile.bf16`, `Modelfile.q8_0`, and
`Modelfile.q4_k_m` with one Rich progress display and summary.

## Register models in the project stack

The automated deployment workflow exports the configured base GGUF and quantizations, generates
their Modelfiles as one phase, starts Ollama and registers all variants as one
phase, validates the challenge split, and updates the HTML report:

`demo-deploy` uses `ollama_export.gguf_outtype` and `ollama_export.quantizations`
from the selected student config for Modelfile generation, registration, and
validation. For example, a BF16 export with only Q4_K_M configured registers and
validates exactly those two variants. The step count adjusts to the variant list.

```bash
docker/stack.sh demo-deploy
```

Override its inputs with `DEMO_DATA_CONFIG` and `DEMO_STUDENT_CONFIG`. The
exported/Ollama model prefix comes from `ollama_export.model_name` in the selected
student config. Validation loads each scene represented by the selected
split when needed.

For manual comparisons, create a separate Modelfile for each GGUF:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile \
  --modelfile-gguf bf16 \
  --modelfile-path outputs/ollama/qwen3.5-4b-finetune/Modelfile.bf16

docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile \
  --modelfile-gguf Q4_K_M \
  --modelfile-path outputs/ollama/qwen3.5-4b-finetune/Modelfile.q4_k_m
```

Start the optional Ollama service and register the files:

```bash
docker/stack.sh ollama-up
docker/stack.sh ollama create qwen3.5-4b-finetune-bf16 \
  -f /models/qwen3.5-4b-finetune/Modelfile.bf16
docker/stack.sh ollama create qwen3.5-4b-finetune-q4_k_m \
  -f /models/qwen3.5-4b-finetune/Modelfile.q4_k_m
docker/stack.sh ollama list
```

Continue with [Ollama raw-generation validation](validation.md#raw-generation-with-ollama).

## Heracles Benchmark handoff

The dedicated Heracles stack validates enabled export directories and runs
`ollama create` in its isolated Ollama container:

```bash
examples/herables_benchmark/run_benchmark.sh models
examples/herables_benchmark/run_benchmark.sh
```

See the [benchmark integration guide](../examples/herables_benchmark/README.md)
for model lists, question configurations, metrics, and reports.

## Qwen3.5 MTP metadata

Some merged Qwen3.5 checkpoints retain `mtp_num_hidden_layers` without the MTP
tensors. Such a GGUF can fail with a missing final block. The Qwen profile
avoids that invalid metadata during conversion:

```yaml
ollama_export:
  convert_args:
    - --no-mtp
```

## Native export

The [Native setup](setup.md#8-install-llamacpp-for-gguf-export) centralizes the
host dependencies, llama.cpp checkout and build, required environment variable,
and optional Ollama installation. After completing it, invoke the exporter
directly with the same arguments documented in this guide, omitting the Docker
command wrapper.
