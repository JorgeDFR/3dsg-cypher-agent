# Heracles Benchmark model overlay

This integration runs this repository's exported baseline and fine-tuned
Ollama models against questions owned by the current
[`heracles-benchmark`](https://github.com/JorgeDFR/heracles-benchmark) project.
The pinned `external/heracles_benchmark` submodule remains the source of truth
for the scene graph, question files and metadata, prompts, manifest schema,
runner, metrics, report generation, and compatible Heracles package revisions.
This directory only supplies the local model list and the Docker integration
needed to register local GGUF exports.

The main project also uses the nested Heracles and Heracles Agents checkouts
for scene loading and SLDP/PDDL answer validation outside this benchmark
integration. Keeping those dependencies under `external/heracles_benchmark`
means a recursive clone fetches each repository only once. The benchmark image
installs the same nested revisions; those package sources are baked into the
image and are not mounted at runtime.

## 1. Prepare the submodules

Initialize all top-level and nested submodules and install this project so the
small host-side manifest adapter can import PyYAML:

```bash
git submodule update --init --recursive
source .venv/bin/activate
pip install -e '.[dev]'
```

The benchmark repository currently declares its nested submodules with GitHub
SSH URLs. On a host without GitHub SSH access, initialize them through HTTPS
without changing the recorded URLs:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --recursive
```

Detached HEADs inside submodules are expected. Git records each submodule as an
exact commit in the parent repository, and `git submodule update` checks out
that commit directly for reproducible builds. A branch checkout is only needed
when actively developing and committing inside a submodule.

The selected Heracles Benchmark config identifies a matched scene graph, QA
questions, PDDL questions, and question metadata. Its own validator verifies
the paths, checksums, scene ID, and expected question counts after the local
models have been overlaid.

## 2. Export and select models

Enabled entries in `config/model_lists/students.yaml` are created from
directories below `outputs/ollama`. The default Qwen baseline/fine-tune pair
expects:

```text
outputs/ollama/
|-- qwen3.5-4b-baseline/
|   |-- Modelfile
|   |-- Modelfile.json
|   `-- <selected model>.gguf
`-- qwen3.5-4b-finetune/
    |-- Modelfile
    |-- Modelfile.json
    `-- <selected model>.gguf
```

Export the pair with:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-baseline \
  --stages all \
  --base-only

docker/stack.sh main python scripts/export_agent_ollama.py \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-name qwen3.5-4b-finetune \
  --stages all
```

Each model-list entry has the stable result `alias`, Ollama `model` name,
local `artifact_dir`, `enabled` switch, and the parameter `capabilities`
required by the current benchmark schema. Optional `parameters` use the
upstream per-model override format. Keep capabilities accurate when adding a
model; the upstream manifest validator rejects incompatible reasoning,
temperature, or seed settings.

Use another model list inside this repository with `--models`:

```bash
examples/herables_benchmark/run_benchmark.sh \
  --models path/to/models.yaml
```

## 3. Select questions and run

The default uses `configs/benchmark.yaml` in the pinned Heracles Benchmark
submodule and runs both QA and PDDL experiments:

```bash
examples/herables_benchmark/run_benchmark.sh
```

Use any other benchmark-owned config, including its small smoke set:

```bash
examples/herables_benchmark/run_benchmark.sh \
  --config configs/benchmark-smoke.yaml
```

The path must be a YAML file below that submodule's `configs/` directory. This
ensures all question paths stay relative to the project that generated and
versioned them. Compose uses `external/heracles_benchmark` as its build context
and invokes that repository's own `docker/benchmark/Dockerfile.benchmark`,
which copies its `configs/` and `data/` trees into
`/home/benchmark/workspace/benchmark`. The adapter reads the selected source
config from the same submodule, writes a generated manifest under
`outputs/heracles_benchmark/.runtime/`, mounts it at
`configs/3dsg-cypher-agent.yaml`, and lets the upstream
`scripts/benchmark_manifest.py` resolve the QA, PDDL, metadata, and scene paths
against the copied benchmark tree before performing final validation and
experiment materialization.

An NVIDIA GPU and the NVIDIA Container Toolkit are required. The launcher
checks both host GPU availability and GPU access from Docker before starting
the benchmark; there is intentionally no CPU fallback.

Results retain the `benchmark.output_dir` layout from the selected config,
rooted on the host at `outputs/heracles_benchmark/`. For the default config the
report is therefore:

```text
outputs/heracles_benchmark/example_dsg/model_sweep/report.html
```

## Stack utilities

```bash
# Validate/create enabled local models without running questions.
examples/herables_benchmark/run_benchmark.sh models

# Inspect the isolated Ollama instance.
examples/herables_benchmark/run_benchmark.sh ollama list

# Keep services after a run for inspection.
examples/herables_benchmark/run_benchmark.sh run --keep-stack

# Print configuration, inspect status, rebuild, or stop the stack.
examples/herables_benchmark/run_benchmark.sh config
examples/herables_benchmark/run_benchmark.sh status
examples/herables_benchmark/run_benchmark.sh build
examples/herables_benchmark/run_benchmark.sh down
```

The Ollama cache survives `down`. Neo4j and Ollama do not publish host ports,
so this isolated stack does not collide with the normal development stack.
