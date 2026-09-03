# Ollama GGUF Export

The exporter supports two variants:

- fine-tuned: merge a trained LoRA/QLoRA adapter into its base model, convert
  the merged checkpoint, and quantize it;
- base: convert the configured Hugging Face base checkpoint directly, with no
  adapter merge and no `merged-hf` output.

Both variants produce one neutral `Modelfile`. It contains the selected GGUF
and stop parameters but no embedded system prompt, so Heracles supplies the
task-specific Cypher or PDDL prompt during each experiment.

## Export with Docker

Start the development stack:

```bash
docker/stack.sh up
```

Export a trained Qwen adapter:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages all
```

The fine-tuned stages are:

```text
base + adapter -> temporary merged-hf -> bf16 GGUF -> quantized GGUF -> Modelfile
```

Export the exact base model configured in the same student YAML:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-baseline \
  --stages all \
  --base-only
```

For a base export the merge stage is deliberately skipped. The converter reads
the local Hugging Face path or resolved Hub snapshot directly.

## Output structure

```text
outputs/ollama/
|-- qwen3.5-4b-baseline/
|   |-- qwen3.5-4b-baseline.bf16.gguf
|   |-- qwen3.5-4b-baseline.Q4_K_M.gguf
|   |-- Modelfile
|   `-- Modelfile.json
`-- qwen3.5-4b-finetune/
    |-- qwen3.5-4b-finetune.bf16.gguf
    |-- qwen3.5-4b-finetune.Q4_K_M.gguf
    |-- Modelfile
    `-- Modelfile.json
```

Directory names identify the model variant and do not encode the current GGUF
quantization. Quantization remains in the GGUF filename and metadata. Fine-tuned
export uses `merged-hf` only as a temporary conversion input.

## Selected stages and quantizations

`--stages` accepts `merge`, `convert`, `quantize`, `modelfile`, or `all`.
Multiple stages can be run when an earlier artifact is already valid:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages merge convert quantize modelfile
```

Generate more than one quantization with:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages quantize \
  --quantizations Q4_K_M Q5_K_M
```

The first configured quantization is selected for `Modelfile` unless
`--selected-quantization` is passed.

Use `--modelfile-gguf` to select a quantization independently of the conversion
and quantization settings. It accepts unquantized types (`bf16`, `f16`, `f32`),
short aliases (`q2`, `q3`, `q4`, `q5`, `q6`, `q8`), and full llama.cpp names
such as `Q4_K_S` or `Q5_K_M`. For example, switch an existing finetune export
from Q4_K_M to its unquantized BF16 GGUF without repeating conversion:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --modelfile-gguf bf16 \
  --stages modelfile
```

The exporter derives the artifact path and reports an error if that file does
not exist. The selector and resolved path are stored as
`modelfile_quantization` and `modelfile_gguf_path` in `Modelfile.json`. The
quantized output paths remain unchanged. `--dry-run` prints external operations
without performing a merge, conversion, quantization, or write.

## Configuration

Settings under `ollama_export` in `configs/data.yaml` include:

- `model_name`: default export directory and Ollama model name;
- `export_root`: parent output directory;
- `gguf_outtype`: intermediate converter type such as `bf16`;
- `quantizations`: quantizations to generate;
- `selected_quantization`: optional Modelfile selection;
- `modelfile_quantization`: optional GGUF quantization selector for the Modelfile;
- `convert_args`: extra `convert_hf_to_gguf.py` arguments;
- `quantize_args`: extra `llama-quantize` arguments.

Explicit CLI overrides remain available for adapter, merged checkpoint, GGUF,
and Modelfile paths. The student config supplies `model.base_model`, optional
tokenizer and remote-code settings, and the default adapter `training.output_dir`.

## Qwen3.5 MTP metadata

Some merged Qwen3.5 checkpoints retain `mtp_num_hidden_layers` without MTP
tensors. A GGUF containing that metadata can fail with a missing final block.
This repository configures:

```yaml
ollama_export:
  convert_args:
    - --no-mtp
```

## Registration and benchmarking

The development stack no longer includes Ollama. The dedicated Heracles stack
validates the enabled exported artifacts and performs `ollama create` inside its
own GPU Ollama container:

```bash
examples/herables_benchmark/run_benchmark.sh models
examples/herables_benchmark/run_benchmark.sh ollama list
examples/herables_benchmark/run_benchmark.sh --benchmarks cypher
```

See the [benchmark guide](../examples/herables_benchmark/README.md) for
the manifest and report layout.

## Host-only export

Install export dependencies and provide a llama.cpp checkout:

```bash
pip install -e '.[export]'
export LLAMA_CPP_DIR=external/llama.cpp
.venv/bin/python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages all
```

When `<llama-cpp-dir>/.venv/bin/python` exists, it is selected for conversion;
`--python-bin` overrides it. Registration for the benchmark should still use
the project-local benchmark launcher so local metrics observe the intended
Ollama container.
