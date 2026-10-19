# Documentation

Start with the root [README](../README.md) for the project overview and quick
start. Commands in the workflow guides use the recommended Docker environment.

## Setup

| Guide | Use it to |
| --- | --- |
| [Docker stack](../docker/README.md) | Set up services and run automated workflows |
| [Native host setup](native_setup.md) | Install and run directly on a Linux host |
| [Scene utilities](scene_tools.md) | Edit room labels and repair scene connections |

## Data and teacher workflows

| Guide | Use it to |
| --- | --- |
| [Dataset generation](dataset_generation.md) | Configure scenes, prepare questions, and create splits |
| [Teacher generation](teacher_generation.md) | Generate and inspect validated teacher traces |
| [Teacher checks and pilots](teacher_checks.md) | Compare remote teacher capabilities, cost, and coverage |
| [Reasoning experiments](reasoning.md) | Configure reasoning and compare matched training runs |

## Models and results

| Guide | Use it to |
| --- | --- |
| [Student training](training.md) | Configure LoRA/QLoRA and inspect supervision |
| [Validation](validation.md) | Evaluate traces, student models, and deployed GGUFs |
| [GGUF export for Ollama](ollama_export.md) | Convert, quantize, and register models |
| [HTML workflow report](workflow_report.md) | Browse artifacts and compare results |
| [Heracles Benchmark](../examples/herables_benchmark/README.md) | Run external benchmark sweeps |

## Configuration and maintenance

| Guide | Use it to |
| --- | --- |
| [Artifact versioning](artifact_versioning.md) | Select versions and understand lineage and output paths |
| [Validation contracts](validation_contracts.md) | Customize result types, answer formats, and PDDL goals |
| [Dataset audits](dataset_audits.md) | Audit catalogs and regrade saved evidence |

## Suggested reading paths

- **First run:** [Quick start](../README.md#quick-start), then
  [HTML workflow report](workflow_report.md).
- **Custom data:** [Dataset generation](dataset_generation.md),
  [Teacher generation](teacher_generation.md), then [Student training](training.md).
- **Teacher comparison:** [Teacher checks and pilots](teacher_checks.md), then
  [Reasoning experiments](reasoning.md).
- **Reproduction:** [Artifact versioning](artifact_versioning.md), then
  [Validation](validation.md).
