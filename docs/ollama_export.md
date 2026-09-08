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
  --quantizations Q4_K_M Q8_0
```

`modelfile_quantization` selects the artifact referenced by `Modelfile`. When
it is omitted, the first configured quantization is used. The
`--modelfile-gguf` argument overrides the configured value.

Use `--modelfile-gguf` to select a quantization independently of the conversion
and quantization settings. It accepts unquantized types (`bf16`, `f16`, `f32`),
short aliases (`q2`, `q3`, `q4`, `q5`, `q6`, `q8`), and full llama.cpp names
such as `Q8_0`. For example, switch an existing finetune export
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

Each `student_*.yaml` model profile contains its own `ollama_export` settings:

- `model_name`: default export directory and Ollama model name;
- `export_root`: parent output directory;
- `gguf_outtype`: intermediate converter type such as `bf16`;
- `quantizations`: quantizations to generate;
- `modelfile_quantization`: optional GGUF quantization selector for the Modelfile;
- `convert_args`: extra `convert_hf_to_gguf.py` arguments;
- `quantize_args`: extra `llama-quantize` arguments.

Explicit CLI overrides remain available for adapter, merged checkpoint, GGUF,
and Modelfile paths. The same student config also supplies `model.base_model`,
optional tokenizer and remote-code settings, and the default adapter
`training.output_dir`.

## Qwen3.5 MTP metadata

Some merged Qwen3.5 checkpoints retain `mtp_num_hidden_layers` without MTP
tensors. A GGUF containing that metadata can fail with a missing final block.
The Qwen student profile configures:

```yaml
ollama_export:
  convert_args:
    - --no-mtp
```

## Registration and benchmarking

The dedicated Heracles stack
validates the enabled exported artifacts and performs `ollama create` inside its Ollama container:

```bash
examples/herables_benchmark/run_benchmark.sh models
examples/herables_benchmark/run_benchmark.sh ollama list
examples/herables_benchmark/run_benchmark.sh
```

See the [benchmark guide](../examples/herables_benchmark/README.md) for
the manifest and report layout.

## Validate BF16 versus quantized GGUF on project splits

Challenge/test split validation belongs to the main project stack rather than
the Heracles benchmark stack. The main stack owns the generated JSONL splits
and matching scene database. Its Ollama service is optional, uses a separate
model cache and container name, and mounts `outputs/ollama` read-only.

After exporting BF16, Q8_0, and Q4_K_M in one `--stages all` run, create one Modelfile
per artifact. Keeping distinct files and registered model names prevents one
variant from replacing the other during a paired comparison:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile \
  --modelfile-gguf bf16 \
  --modelfile-path outputs/ollama/qwen3.5-4b-finetune/Modelfile.bf16

docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile \
  --modelfile-gguf Q8_0 \
  --modelfile-path outputs/ollama/qwen3.5-4b-finetune/Modelfile.q8_0

docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages modelfile \
  --modelfile-gguf Q4_K_M \
  --modelfile-path outputs/ollama/qwen3.5-4b-finetune/Modelfile.q4_k_m
```

Start the optional service and register both names inside it:

```bash
docker/stack.sh ollama-up

docker/stack.sh ollama create qwen3.5-4b-finetune-bf16 \
  -f /models/qwen3.5-4b-finetune/Modelfile.bf16

docker/stack.sh ollama create qwen3.5-4b-finetune-q8_0 \
  -f /models/qwen3.5-4b-finetune/Modelfile.q8_0

docker/stack.sh ollama create qwen3.5-4b-finetune-q4_k_m \
  -f /models/qwen3.5-4b-finetune/Modelfile.q4_k_m

docker/stack.sh ollama list
```

Load the configured scene into the main Neo4j database, then run both variants
with identical generation and query limits:

```bash
docker/stack.sh load-neo4j data/scenes/uhumans2_office.json

docker/stack.sh main python scripts/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --split challenge \
  --mode raw-generation \
  --runtime ollama \
  --ollama-model qwen3.5-4b-finetune-bf16 \
  --max-new-tokens 1024

docker/stack.sh main python scripts/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --split challenge \
  --mode raw-generation \
  --runtime ollama \
  --ollama-model qwen3.5-4b-finetune-q8_0 \
  --max-new-tokens 1024

docker/stack.sh main python scripts/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --split challenge \
  --mode raw-generation \
  --runtime ollama \
  --ollama-model qwen3.5-4b-finetune-q4_k_m \
  --max-new-tokens 1024
```

Ollama runs use deterministic defaults (`temperature=0`, `seed=42`) and send
non-streaming `/api/chat` requests with native tools. Override these with
`--ollama-temperature`, `--ollama-seed`, `--ollama-timeout`, or
`--ollama-context-length`. The context length defaults to 16384 so the longest
current challenge tool-response conversations can complete, and
`--ollama-keep-alive` controls model residency. The default report name includes the runtime and model,
so BF16 and Q4_K_M results remain separate. Eval-loss remains a Transformers
operation; Ollama is supported only for raw-generation validation.

## Host-only export

Install export dependencies and provide a llama.cpp checkout:

```bash
pip install -e '.[export]'
export LLAMA_CPP_DIR=external/llama.cpp
.venv/bin/python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages all
```

When `<llama-cpp-dir>/.venv/bin/python` exists, it is selected for conversion;
`--python-bin` overrides it. Registration for the benchmark should still use
the project-local benchmark launcher so local metrics observe the intended
Ollama container.
