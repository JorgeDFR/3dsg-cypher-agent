# 3DSG Cypher Agent Distillation

This project fine-tunes small tool-using agents to query Neo4j-backed 3D Scene
Graphs (3DSGs). The agents call `run_cypher_query(cypher_string: str)` and
produce either a direct SLDP answer or a grounded PDDL goal for robot planning.

Teacher traces can come from another model through OpenRouter or from a
deterministic synthetic teacher. Student models are trained with LoRA or QLoRA
through Hugging Face and can be exported to GGUF for Ollama deployment.

## Project overview

The repository covers the complete distillation workflow:

- **[3DSG data extraction and question generation](docs/dataset_generation.md#prepare-a-dataset-version)** —
  extract 3D scene graph data, render question templates, and prepare validated
  seed records.
- **[Teacher trace generation](docs/dataset_generation.md#generate-teacher-traces)** —
  collect tool-using traces, execute Cypher against Neo4j, and retain
  interactions that pass safety and task checks.
- **[Dataset splitting](docs/dataset_generation.md#create-student-splits)** —
  build grouped, leakage-checked train, evaluation, test, and challenge splits.
- **[LoRA and QLoRA training](docs/training.md)** — fine-tune a configured
  Hugging Face student and inspect its supervised token spans.
- **[Model validation](docs/validation.md)** — evaluate trace quality, held-out
  loss, and raw generation through Transformers or Ollama.
- **[GGUF model export](docs/ollama_export.md)** — merge adapters, convert
  models, quantize GGUF artifacts, and register them in Ollama.

## Repository structure

| Path | Description |
| --- | --- |
| `configs/` | Data, teacher, student, prompt, and schema configuration |
| `data/` | Input scenes and versioned generated datasets |
| `docker/` | Main development Compose stack and command wrapper |
| `docs/` | Detailed project documentation |
| `examples/herables_benchmark/` | Local-model overlay for Heracles Benchmark |
| `external/` | Pinned external dependencies and benchmark submodules |
| `outputs/` | Versioned LoRA outputs, reports, GGUF exports, and benchmark results |
| `scripts/` | Command-line entry points for each workflow stage |
| `src/` | Python package implementation |
| `tests/` | Unit and integration tests |

## Setup

The project supports two execution environments:

- **[Docker (recommended)](docker/README.md)** provides the project Python environment, Neo4j,
  CUDA-enabled PyTorch, llama.cpp, and an optional Ollama service. It is the
  supported path for the complete GPU training and deployment workflow.
- **[Native setup](docs/setup.md)** installs the Python package and dependencies directly on the
  host. Use it when Docker is unavailable or for host-side development; CUDA,
  PyTorch, Neo4j, and llama.cpp may require separate setup.

Both approaches use the same YAML configuration and versioned artifacts. The
commands below use Docker.

## Configuration and versioning

The workflow is driven by a small set of YAML files. Data configuration selects
the scene, question families, and splits; teacher configuration controls trace
generation; and student configuration defines the base model, LoRA/QLoRA
training, and GGUF export. Experiment profiles and prompt fragments define the
task-specific behavior shared across generation, training, and validation.

| Path | Purpose |
| --- | --- |
| `configs/data.yaml` | Scene, question sets, split proportions, and challenge task types |
| `configs/teacher_*.yaml` | Teacher provider, model, sampling, and tool-call limits |
| `configs/student_*.yaml` | Base model, LoRA/QLoRA, training, and export settings |
| `configs/experiment_profiles/` | Task-specific prompts and output contracts |
| `configs/prompts/` | Shared agent instructions and domain descriptions |
| `configs/question_templates/` | QA and PDDL question templates |
| `configs/schema_3dsg.yaml` | Scene graph schema used for generation and validation |

Dataset extraction, teacher generation, and model training create timestamped
artifact versions. Downstream commands select the latest compatible version by
default and accept explicit dataset, teacher-run, or model selectors for
reproducible experiments. See [Artifact versioning](docs/artifact_versioning.md)
for the directory layout, selection rules, lineage checks, and legacy migration.

The data configuration gives the combined dataset a `dataset_id` and lists one
or more explicit `scene_id`/`scene_path` pairs. Trace generation loads the
correct scene into Neo4j as it moves between scene-specific seeds. A scene can
also be loaded manually for inspection:

```bash
docker/stack.sh main python scripts/load_3dsg_neo4j.py \
  --data-config configs/data.yaml \
  --scene-id uhumans2_office
```

## Quick start

The Docker workflow requires an NVIDIA GPU, a compatible driver, Docker Compose
v2, and NVIDIA Container Toolkit.

Clone the repository and its submodules:

```bash
git clone --recursive https://github.com/JorgeDFR/3dsg-cypher-agent.git
cd 3dsg-cypher-agent
```

Create the local environment file, then validate and start the stack:

```bash
cp docker/.env.example docker/.env
docker/stack.sh setup
```

Set `OPENROUTER_API_KEY` in `docker/.env` only when using an OpenRouter teacher.
The default demonstration uses the synthetic teacher and needs no API key:

```bash
docker/stack.sh demo-train
```

This loads the configured scene, prepares the dataset, trains the student,
runs baseline and fine-tuned validation, and generates an HTML workflow report.

After an adapter has been trained, export and validate BF16, Q8_0, and Q4_K_M
GGUF variants:

```bash
docker/stack.sh demo-deploy
```

Run `docker/stack.sh help` for available commands and environment overrides.

## Heracles Benchmark

The integration under `examples/herables_benchmark/` evaluates exported local
models on the pinned Heracles Benchmark scenes, QA/PDDL questions, manifests,
and metrics. It uses an isolated Neo4j and Ollama stack so benchmark sweeps do
not interfere with the main development services.

Run the configured model and question sweep with:

```bash
examples/herables_benchmark/run_benchmark.sh
```

See the [Heracles Benchmark integration guide](examples/herables_benchmark/README.md)
for model preparation, alternate question sets, output locations, and stack
commands.

## Additional tools and reports

- **[HTML workflow report](docs/workflow_report.md)** — browse the 3DSG,
  generated questions, trace outcomes, dataset splits, training state,
  validation results, baseline comparisons, deployment comparisons, and
  artifact inventory in one static report.
- **[Scene utilities](docs/scene_tools.md)** — inspect and update room labels in
  a Spark DSG scene before generating a dataset.
