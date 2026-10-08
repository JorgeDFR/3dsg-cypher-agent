# Docker Development Stack

Docker is the recommended interface for dataset generation, training,
validation, and GGUF export.

## Services

| Service | Role | Started by `setup` / `up` |
| --- | --- | --- |
| `main` | Project scripts, tests, LoRA/QLoRA training, evaluation, and export | Yes |
| `neo4j` | 3D Scene Graph database used for tool execution and validation | Yes |
| `ollama` | Optional inference service for project-split GGUF validation | No |

The Ollama service is behind a Compose profile. Heracles Benchmark uses a
separate stack, scene, questions, and metrics pipeline under
`examples/herables_benchmark/`.

## Requirements

- NVIDIA GPU and compatible driver
- Docker Engine
- Docker Compose v2
- NVIDIA Container Toolkit

Check the host and container GPU setup at any time:

```bash
docker/stack.sh check
```

## First run

Create the local environment file. Add `OPENROUTER_API_KEY` only if the remote
teacher will be used:

```bash
cp docker/.env.example docker/.env
```

Validate prerequisites, build the main image, and start `main` plus Neo4j:

```bash
docker/stack.sh setup
```

Load the default scene and inspect service status:

```bash
docker/stack.sh load-neo4j
docker/stack.sh status
```

## Automated workflows

### Create data, train, and validate

```bash
docker/stack.sh demo-dataset
docker/stack.sh demo-train
docker/stack.sh demo-validate
```

These commands independently prepare a new dataset, train on existing splits,
and validate the selected student model plus baseline. Validation reuses matching
completed reports. Database-backed stages load each configured scene when needed.
For their stages and overrides, see
[Student training](../docs/training.md#automated-training-workflow).

### Export and deploy

After training has produced a fine-tuned model:

```bash
docker/stack.sh demo-deploy
```

This workflow exports and registers the base GGUF and quantization variants
specified in the student YAML (`ollama_export.gguf_outtype` and
`ollama_export.quantizations`), validates
them on the challenge split, and regenerates the report. See
[Ollama GGUF export](../docs/ollama_export.md#register-models-in-the-project-stack).

## Daily use

```bash
docker/stack.sh up
docker/stack.sh main python scripts/agents/evaluate_traces.py \
  --data-config configs/data.yaml --split train
docker/stack.sh shell
docker/stack.sh logs main
docker/stack.sh logs neo4j
docker/stack.sh down
```

The project virtual environment is activated automatically for `main` and
`shell`, so use `python` directly inside the container.

Start Ollama only when validating exported GGUFs on project splits:

```bash
docker/stack.sh ollama-up
docker/stack.sh ollama list
```

## Command reference

### Setup and lifecycle

| Command | Action |
| --- | --- |
| `setup` | Check prerequisites, build `main`, and start `main` and `neo4j` |
| `check` | Check Docker, Compose, NVIDIA GPU, and container GPU access |
| `build` | Rebuild the main image |
| `up` | Start `main` and `neo4j` |
| `ollama-up` | Start the optional project Ollama service |
| `down` | Stop containers without deleting persistent volumes |
| `restart [SERVICE...]` | Restart all or selected services |

### Project execution

| Command | Action |
| --- | --- |
| `main COMMAND [ARGS...]` | Run a command in the `main` container |
| `shell` | Open an interactive Bash shell in `main` |
| `exec COMMAND [ARGS...]` | Compatibility alias for `main` |
| `ollama COMMAND [ARGS...]` | Run the CLI in the optional Ollama container |
| `load-neo4j [SCENE]` | Load a scene into the project Neo4j service |
| `demo-dataset` | Generate a new dataset, teacher traces, and splits |
| `demo-train` | Train a new student model using the current dataset |
| `demo-validate` | Validate the student model and baseline and generate the HTML report |
| `demo-deploy` | Export, register, and validate deployable GGUF variants |

### Inspection

| Command | Action |
| --- | --- |
| `status` / `ps` | Show service status |
| `logs [SERVICE...]` | Follow all or selected service logs |
| `config` | Validate and print the resolved Compose configuration |

Run `docker/stack.sh help` for the terminal version of this reference and the
supported `DEMO_*` environment variables.

## Network addresses

| Service | From `main` | From the host |
| --- | --- | --- |
| Neo4j Bolt | `neo4j://neo4j:7683` | `neo4j://127.0.0.1:7683` |
| Neo4j HTTP | — | `http://127.0.0.1:7469` |
| Ollama | `http://ollama:11434` | Not published |

The default Neo4j username is `neo4j` and the password is `neo4j_pw`.

## Image and persistent data

The main image provides CUDA-enabled PyTorch, installs project dependencies in
`/home/user/workspace/.venv`, and builds llama.cpp under
`/home/user/workspace/llama.cpp`.

The repository is bind-mounted into `main`, so generated outputs remain on the
host. Compose preserves the named volume `3dsg-cypher-agent-huggingface-cache` across `down`.

## Direct Compose inspection

The wrapper is the supported interface. Inspect the resolved Compose file with:

```bash
docker compose -f docker/docker-compose.yaml config --quiet
```

## Related guides

- [Dataset generation](../docs/dataset_generation.md)
- [Student training](../docs/training.md)
- [Validation](../docs/validation.md)
- [Ollama GGUF export](../docs/ollama_export.md)
- [HTML workflow report](../docs/workflow_report.md)
- [Heracles Benchmark integration](../examples/herables_benchmark/README.md)
