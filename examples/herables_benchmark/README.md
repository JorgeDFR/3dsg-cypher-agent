# Local Heracles Ollama Model Sweep

This directory contains a self-contained Docker setup for benchmarking this
repository's exported base and fine-tuned models with the vendored Heracles
and Heracles Agents code. It does not mount, modify, or depend on the separate
`heracles_devel` checkout.

The stack runs only the Ollama portion of the benchmark. An NVIDIA GPU is
mandatory so the benchmark can collect GPU, container, Ollama runtime, latency,
tool-use, answer-quality, token, and energy-related metrics supported by the
current Heracles result schema. The runner passes `--debug`, preserving raw
local-metric samples beside each result YAML, and generates one static
`report.html` without changing Heracles result reporting.

## 1. Export the models

The enabled manifest initially expects these artifact directories:

```text
outputs/ollama/
|-- qwen3.5-4b-baseline/
|   |-- Modelfile
|   |-- Modelfile.json
|   |-- qwen3.5-4b-baseline.bf16.gguf
|   `-- qwen3.5-4b-baseline.Q4_K_M.gguf
`-- qwen3.5-4b-finetune/
    |-- Modelfile
    |-- Modelfile.json
    |-- qwen3.5-4b-finetune.bf16.gguf
    `-- qwen3.5-4b-finetune.Q4_K_M.gguf
```

Export the baseline model:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-baseline \
  --stages all \
  --base-only
```

Export the finetuned model:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages all
```

The Modelfile uses the selected quantized GGUF by default. To benchmark the
unquantized BF16 output, regenerate only the Modelfile and select it explicitly:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --modelfile-gguf bf16 \
  --stages modelfile
```

## 2. Select models

Edit `config/model_lists/students.yaml` to enable or disable variants. Each
enabled entry supplies:

- `alias`: stable name used in result filenames;
- `model`: name created in the benchmark Ollama instance;
- `artifact_dir`: directory below `outputs/ollama`;
- `enabled`: whether to include the model in the sweep.

The initial manifest enables the `qwen3.5-4b-baseline` and
`qwen3.5-4b-finetune` variants. Granite entries are present but disabled as an
additional example.

## 3. Run

Cypher QA is the non-interactive default:

```bash
examples/herables_benchmark/run_benchmark.sh \
  --benchmarks cypher
```

PDDL and combined runs remain available for future models:

```bash
examples/herables_benchmark/run_benchmark.sh --benchmarks pddl
examples/herables_benchmark/run_benchmark.sh --benchmarks both
```

With no arguments in an interactive terminal, the launcher presents the same
three choices. It validates Docker GPU access, builds the project-local
benchmark image, starts isolated Neo4j/Ollama/metrics-proxy containers, validates
all enabled exports, and runs `ollama create` inside the new benchmark Ollama
container before launching Heracles.

Results are written to:

```text
outputs/heracles_benchmark/model_sweep/
|-- ollama/
|   |-- cypher_student_model_sweep/
|   `-- pddl_student_model_sweep/
`-- report.html
```

The per-model directories contain result YAML and raw local-metric sample YAML.
The report is created from every result YAML produced by the selected sweep.

The benchmark image is built only when its stable local tag does not exist.
Later runs reuse it even if source files changed. Force a rebuild after changing
the Dockerfile, benchmark runner, experiment configuration, or vendored
Heracles sources with:

```bash
examples/herables_benchmark/run_benchmark.sh build
```

## Stack utilities

```bash
# Build/start the stack and create enabled models, without running questions
examples/herables_benchmark/run_benchmark.sh models

# Inspect the benchmark Ollama instance; this does not build the benchmark image
examples/herables_benchmark/run_benchmark.sh ollama list

# Keep services after a run for inspection
examples/herables_benchmark/run_benchmark.sh run \
  --benchmarks cypher --keep-stack

# Inspect or stop them later
examples/herables_benchmark/run_benchmark.sh status
examples/herables_benchmark/run_benchmark.sh down
```

The named Ollama cache survives `down`. Neither Neo4j nor Ollama exposes a host
port, which avoids collisions with this project's normal development stack.
