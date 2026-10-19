# GGUF Export for Ollama

Package a fine-tuned model or its configured base checkpoint as GGUF artifacts.
Commands use the [Docker stack](../docker/README.md), which includes llama.cpp
and export dependencies. For host execution, follow
[Native setup](native_setup.md) and omit the Docker command wrapper.

| Variant | Input | Stages |
| --- | --- | --- |
| Fine-tuned | Base model and trained LoRA/QLoRA weights | Merge, convert, quantize, write Modelfile |
| Baseline | Configured Hugging Face checkpoint | Convert, quantize, write Modelfile |

Modelfiles select a GGUF and stop parameters. Evaluation supplies the task prompt.
A fine-tuned export requires completed [student training](training.md).

## Quick export

Start the stack with `docker/stack.sh up`, then export the latest trained model:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune --stages all
```

Add `--model-version MODEL_VERSION` to select a specific trained model.
For a baseline, add `--base-only` and use a separate name such as
`qwen3.5-4b-baseline`.

## Student export configuration

Set export defaults under `ollama_export` in the student YAML:

| Key | Purpose |
| --- | --- |
| `model_name` | Export directory and Ollama model-name prefix |
| `export_root` | Parent output directory |
| `gguf_outtype` | Conversion type, such as `bf16` |
| `quantizations` | Quantized artifacts to generate |
| `modelfile_quantization` | GGUF selected by the default Modelfile |
| `convert_args` | Extra llama.cpp conversion arguments |
| `quantize_args` | Extra quantizer arguments |

CLI arguments can override these settings and output paths.

## Run selected stages

| Stage | Action |
| --- | --- |
| `merge` | Merge fine-tuned weights into a temporary checkpoint |
| `convert` | Convert the checkpoint to GGUF |
| `quantize` | Produce configured quantizations |
| `modelfile` | Select an existing GGUF for Ollama |
| `all` | Run all applicable stages |

For example, reuse an existing conversion:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages quantize modelfile --quantizations Q4_K_M Q8_0
```

`--dry-run` previews operations without writing artifacts. `--quiet` shows
project progress and summaries while retaining error details on failure.

## Output layout

```text
outputs/ollama/qwen3.5-4b-finetune/
|-- qwen3.5-4b-finetune.bf16.gguf
|-- qwen3.5-4b-finetune.Q8_0.gguf
|-- qwen3.5-4b-finetune.Q4_K_M.gguf
|-- Modelfile
|-- Modelfile.json
`-- metadata.yaml
```

The exact files depend on the configured conversion type and quantizations.
See [GGUF export policy](artifact_versioning.md#gguf-export-policy) for
replacement behavior and lineage.

## Select Modelfile artifacts

Generate a Modelfile for one existing GGUF:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile --modelfile-gguf bf16
```

The selector accepts types such as `bf16`, short aliases such as `q4`, and
full quantization names such as `Q4_K_M`. The selected GGUF must already exist.
Use `--modelfile-path PATH` for a custom destination.

Generate named Modelfiles for several variants at once:

```bash
docker/stack.sh main python scripts/training/export_agent_ollama.py \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile --modelfile-variants bf16 Q8_0 Q4_K_M
```

This writes `Modelfile.bf16`, `Modelfile.q8_0`, and `Modelfile.q4_k_m`.

## Register models in the project stack

For automatic export, registration, challenge validation, and reporting:

```bash
docker/stack.sh demo-deploy
```

The workflow uses `ollama_export.gguf_outtype`, `quantizations`, and `model_name`
from the selected student YAML. See
[Docker automated workflows](../docker/README.md#automated-workflows) for input
overrides.

For manual registration, generate the named Modelfiles above, then run:

```bash
docker/stack.sh ollama-up
docker/stack.sh ollama create qwen3.5-4b-finetune-bf16 \
  -f /models/qwen3.5-4b-finetune/Modelfile.bf16
docker/stack.sh ollama create qwen3.5-4b-finetune-q4_k_m \
  -f /models/qwen3.5-4b-finetune/Modelfile.q4_k_m
docker/stack.sh ollama list
```

Continue with [Ollama validation](validation.md#raw-generation-with-ollama).
For an isolated benchmark sweep, follow the
[Heracles integration](../examples/herables_benchmark/README.md).

## Qwen3.5 conversion

If a checkpoint contains MTP metadata without its tensors, conversion must
exclude that metadata. The supplied Qwen profile sets:

```yaml
ollama_export:
  convert_args:
    - --no-mtp
```
