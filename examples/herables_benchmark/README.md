# Heracles Benchmark Model Overlay

This integration evaluates locally exported baseline and fine-tuned Ollama
models against questions owned by the pinned
[`heracles-benchmark`](https://github.com/JorgeDFR/heracles-benchmark)
submodule.

## Responsibility split

| Component | Source of truth |
| --- | --- |
| Local model list and GGUF directories | This repository |
| Docker integration and generated overlay manifest | `examples/herables_benchmark/` |
| Scene graph, QA/PDDL questions, metadata, and prompts | Pinned Heracles Benchmark submodule |
| Manifest schema, runner, metrics, and report generation | Pinned Heracles Benchmark submodule |
| Compatible Heracles and Heracles Agents revisions | Nested pinned submodules |

The nested Heracles packages are also used by the main project for scene loading
and SLDP/PDDL answer validation. A recursive clone fetches each dependency once.

## Prerequisites

- NVIDIA GPU and NVIDIA Container Toolkit
- Enabled GGUF exports under `outputs/ollama/`
- All top-level and nested submodules initialized

Prepare the submodules, virtual environment, and development dependencies using
the centralized [Native setup](../../docs/setup.md). The benchmark launcher uses
that environment for its small host-side manifest adapter.

## Prepare models

Enabled entries in `config/model_lists/students.yaml` resolve model artifacts
below `outputs/ollama/`. The default list expects:

```text
outputs/ollama/
|-- qwen3.5-4b-baseline/
|   |-- Modelfile
|   |-- Modelfile.json
|   `-- <selected-model>.gguf
`-- qwen3.5-4b-finetune/
    |-- Modelfile
    |-- Modelfile.json
    `-- <selected-model>.gguf
```

Follow [Ollama GGUF export](../../docs/ollama_export.md#quick-export) to create
the baseline and fine-tuned directories.

Each model-list entry declares:

| Field | Purpose |
| --- | --- |
| `alias` | Stable name used in benchmark results |
| `model` | Ollama model name |
| `artifact_dir` | Local export directory |
| `enabled` | Whether the launcher includes the model |
| `capabilities` | Parameters supported by the model and required by the manifest schema |
| `parameters` | Optional upstream per-model inference overrides |

Keep `capabilities` accurate. The upstream validator rejects incompatible
reasoning, temperature, or seed settings.

Select another repository-owned model list with:

```bash
examples/herables_benchmark/run_benchmark.sh \
  --models path/to/models.yaml
```

## Select questions

The default uses `configs/benchmark.yaml` from the submodule and includes both
QA and PDDL experiments:

```bash
examples/herables_benchmark/run_benchmark.sh
```

Select another benchmark-owned configuration, such as the smoke set:

```bash
examples/herables_benchmark/run_benchmark.sh \
  --config configs/benchmark-smoke.yaml
```

The configuration must be a YAML file below the submodule's `configs/`
directory. Its validator checks scene ID, paths, checksums, and expected
question counts after the local models are overlaid.

## Runtime flow

1. The adapter reads the selected upstream configuration.
2. It overlays the enabled local model entries.
3. It writes a generated manifest under `outputs/heracles_benchmark/.runtime/`.
4. Compose mounts the manifest at `configs/3dsg-cypher-agent.yaml`.
5. The upstream validator materializes the experiments against its copied
   benchmark tree.
6. The isolated Neo4j and Ollama services run the selected sweeps.

The image uses `external/heracles_benchmark` as its build context. Package
sources are baked into the image instead of being mounted at runtime.
separate mount replaces the installed Heracles or Heracles Agents packages.

There is no CPU fallback. The launcher verifies host GPU availability and
Docker GPU access before starting.

## Results

Results retain `benchmark.output_dir` from the selected upstream configuration
and are rooted on the host at `outputs/heracles_benchmark/`.

For the default configuration, open:

```text
outputs/heracles_benchmark/example_dsg/model_sweep/report.html
```

## Stack utilities

| Command | Action |
| --- | --- |
| `run_benchmark.sh models` | Validate and create enabled local models without running questions |
| `run_benchmark.sh ollama list` | Inspect the isolated Ollama instance |
| `run_benchmark.sh run --keep-stack` | Keep services running after a sweep |
| `run_benchmark.sh config` | Print the resolved configuration |
| `run_benchmark.sh status` | Inspect service status |
| `run_benchmark.sh build` | Rebuild the benchmark image |
| `run_benchmark.sh down` | Stop the isolated stack |

Run these from the repository root, prefixed with
`examples/herables_benchmark/` as shown in earlier examples.

The Ollama cache survives `down`. Neo4j and Ollama publish no host ports, so the
benchmark stack does not collide with the normal development stack.
