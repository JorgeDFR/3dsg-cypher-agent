# Documentation

The root [README](../README.md) introduces the project and provides the shortest
path to a complete run. Use this index for detailed workflows and reference
material.

## Core workflow

| Stage | Guide | What it covers |
| --- | --- | --- |
| 1 | [Dataset generation](dataset_generation.md) | Scene extraction, question templates, seed records, teacher traces, and splits |
| 2 | [Student training](training.md) | LoRA/QLoRA configuration, training, and token inspection |
| 3 | [Validation](validation.md) | Trace checks, eval-loss, raw generation, and Ollama evaluation |
| 4 | [Ollama GGUF export](ollama_export.md) | Fine-tuned weight merging, GGUF conversion, quantization, and registration |

## Setup and operations

| Topic | Guide |
| --- | --- |
| Recommended GPU environment and service commands | [Docker development stack](../docker/README.md) |
| Optional host installation and dependencies | [Native setup](setup.md) |
| Dataset, teacher, model, and report lineage | [Artifact versioning](artifact_versioning.md) |
| Consolidated static results and artifact inspection | [HTML workflow report](workflow_report.md) |
| External performance sweeps | [Heracles Benchmark integration](../examples/herables_benchmark/README.md) |
| Controlled repair mutations and coverage | [Cypher repair supervision](cypher_repair_supervision.md) |
| Reasoning generation, sentence selection, teacher capability checks, pilots, and matched experiments | [Reasoning traces](reasoning_traces.md) |
| Room labels and 3DSG editing | [Scene utilities](scene_tools.md) |

## Suggested reading paths

### First complete run

1. Follow the root [Quick start](../README.md#quick-start).
2. Read [Dataset generation](dataset_generation.md) to customize the scene or
   questions.
3. Tune the student with [Student training](training.md).
4. Compare models with [Validation](validation.md).
5. Inspect the run with the [HTML workflow report](workflow_report.md).
6. Package the selected model with [Ollama GGUF export](ollama_export.md).

### Reproduce an existing run

1. Select exact inputs using [Artifact versioning](artifact_versioning.md).
2. Run the relevant [validation mode](validation.md#student-validation).
3. Use the [Heracles integration](../examples/herables_benchmark/README.md) for
   benchmark-owned question sets and metrics.
